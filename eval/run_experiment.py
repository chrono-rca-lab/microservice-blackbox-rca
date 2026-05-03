"""End-to-end run: fault injection → Prometheus scrape → RCA.

Fixed-duration path: waits the full `--duration`; the background SLO thread only logs
latency for later analysis (it never short-circuits this runner).

injectors:
  • chaos_inject.py (Chaos Mesh) — cpu/mem/net faults; works on distroless images.
  • inject.py (kubectl exec + shell) — disk_hog only; IOChaos needs FUSE kind doesn't expose.

Rough order: loadgen → baseline → inject + SLO poller → sleep(duration) → pull metrics →
pinpoint roots → reap injector → cool down.

Examples:
    python eval/run_experiment.py --fault cpu_hog   --service frontend       --duration 120
    python eval/run_experiment.py --fault net_delay --service cartservice     --duration 120
    python eval/run_experiment.py --fault disk_hog  --service currencyservice --duration 120
"""

import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # infra/, rca_engine/, fault_injection/ live next to eval/

from infra.loadgen import WorkloadGenerator
from rca_engine.metrics_client import PrometheusMetricsClient
import fault_injection.ground_truth as gt
from results import ResultsStore

EXPERIMENTS_DIR    = ROOT / "experiments"
CHAOS_INJECT_SCRIPT = ROOT / "fault_injection" / "chaos_inject.py"
EXEC_INJECT_SCRIPT  = ROOT / "fault_injection" / "inject.py"

EXEC_ONLY_FAULTS = {"disk_hog"}  # shell inside pod; Chaos IO path isn't usable on kind

P95_THRESHOLD_MS     = 500   # min dynamic SLO line for logging-only poller
BASELINE_DURATION    = 70    # soak before injection
BASELINE_END_BUFFER  = 10    # shave end of baseline so the "normal" fit skips pre-fault creep
RECOVERY_WAIT        = 30    # watch cluster after RCA before killing loadgen
SLO_POLL_INTERVAL    = 5
FRONTEND_URL         = "http://localhost:8080"
PROMETHEUS_URL       = "http://localhost:9090"
NAMESPACE            = "boutique"


def _ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _format_rca_output(
    logs: list,
    ranked: list,
    layer_timings: dict[str, float] | None = None,
) -> list[str]:
    """Pretty-print RCA stages and per-service onsets into plain-text lines."""

    # Map engine log stages to headings (first timestamp wins).
    LAYER_LABELS = {
        "LAYER1_CUSUM": "Layer 1 (CUSUM Detection)",
        "LAYER3_FFT_FILTER": "Layer 3 (FFT Predictability Filter)",
        "LAYER4_ROLLBACK": "Layer 4 (Onset Rollback)",
        "FINAL_RANKING": "Final Ranking",
        "START_PINPOINT": "RCA Pipeline Start",
    }

    layer_events = {}
    for entry in logs:
        stage = entry.get("stage", "")
        if stage in LAYER_LABELS:
            label = LAYER_LABELS[stage]
            ts = entry.get("timestamp", 0)
            if label not in layer_events:
                layer_events[label] = ts
    
    txt_lines = []
    if layer_timings:
        txt_lines.append("Layer-by-Layer Timing (seconds):")
        for i in range(9):
            key = f"layer{i}"
            val = layer_timings.get(key)
            txt_lines.append(
                f"  Layer {i}: {'n/a' if val is None else f'{float(val):.6f}'}"
            )
        txt_lines.append("")

    txt_lines.append("Service-wise RCA Timeline:\n")
    
    for svc in ranked:
        name = svc.get("service")
        onset = svc.get("onset_time")
        confidence = svc.get("confidence", 0.0)
        is_root = svc.get("is_root_cause", False)
        abnormal_metrics = svc.get("abnormal_metrics") or []

        if isinstance(onset, (int, float)):
            onset_str = datetime.fromtimestamp(onset, timezone.utc).isoformat()
        else:
            onset_str = str(onset)

        txt_lines.append(f"Service: {name}")
        txt_lines.append(f"  Onset: {onset_str}")
        txt_lines.append(f"  Confidence: {confidence:.3f}")
        txt_lines.append(f"  Root Cause: {is_root}")

        if abnormal_metrics:
            txt_lines.append(f"\n  Metrics:")
            txt_lines.append(f"    {', '.join(abnormal_metrics)}")

        # Stage timings vs this service’s onset (if we have timestamps).
        if layer_events and isinstance(onset, (int, float)):
            txt_lines.append(f"\n  RCA Pipeline Stages:")
            for layer_label in [
                "RCA Pipeline Start",
                "Layer 1 (CUSUM Detection)",
                "Layer 3 (FFT Predictability Filter)",
                "Layer 4 (Onset Rollback)",
                "Final Ranking",
            ]:
                if layer_label in layer_events:
                    layer_ts = layer_events[layer_label]
                    delta = layer_ts - onset
                    txt_lines.append(f"    - {layer_label:<40} → +{delta:.3f}s")

        txt_lines.append("")

    return txt_lines


class SLOMonitor:
    """Poll frontend p95 in the background for logging only (doesn't unblock the runner)."""

    def __init__(self, gen: WorkloadGenerator, threshold_ms: float) -> None:
        self._gen = gen
        self._threshold_ms = threshold_ms
        self._violation_time: float | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="slo-monitor"
        )
        self._thread.start()

    def stop(self) -> float | None:
        """Signal stop, wait for thread to exit, return violation timestamp (or None)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=SLO_POLL_INTERVAL + 2)
        with self._lock:
            return self._violation_time

    def _run(self) -> None:
        # First iteration happens after injection; we only emit on threshold breach.
        while not self._stop.wait(SLO_POLL_INTERVAL):
            p95 = self._gen.current_p95(window_seconds=10)
            if p95 is None:
                continue
            ms = p95 * 1000
            with self._lock:
                if self._violation_time is None and ms > self._threshold_ms:
                    self._violation_time = time.time()
                    click.echo(
                        f"  [slo] VIOLATION detected — p95={ms:.0f}ms"
                        f" (threshold={self._threshold_ms:.0f}ms)"
                    )


def run_rca(
    metric_matrix: dict,
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    run_dir: Path,
    propagation_map_path: str | None = None,
) -> dict:
    """Run pinpoint and stash timing / text artifacts under run_dir."""
    try:
        from rca_engine import fault_chain

        start_time = time.time()
        logs: list[dict] = []
        layer_timings: dict[str, float] = {}
        
        ranked = fault_chain.pinpoint(
            metric_matrix=metric_matrix,
            baseline_window=baseline_window,
            fault_window=fault_window,
            propagation_map_path=propagation_map_path,
            start_time=start_time,
            logs=logs,
            layer_timings=layer_timings,
        )

        total_time = time.time() - start_time
        result = {
            "ranked_services": ranked,
            "total_time_seconds": total_time,
            "layer_timings_seconds": layer_timings,
            "timing_logs": logs,
        }

        output_file = run_dir / "rca_timing.json"
        output_file.write_text(json.dumps(result, indent=2))
        click.echo(f"  [rca] timing data saved to {output_file}")

        txt_lines = _format_rca_output(logs, ranked, layer_timings=layer_timings)
        txt_lines.insert(0, f"Total RCA time: {total_time:.3f} seconds")
        txt_lines.insert(0, "")
        txt_lines.insert(0, f"Saved: {datetime.now(timezone.utc).isoformat()}")
        txt_lines.insert(0, "RCA Service Timeline (Onset-based)")

        output_txt = run_dir / "output.txt"
        output_txt.write_text("\n".join(txt_lines))
        click.echo(f"  [rca] text summary saved to {output_txt}")

        return {"ranked_services": ranked}
    except NotImplementedError:
        click.echo("  [rca] pinpoint not wired up yet — stub result only")
        return {"status": "rca_not_implemented"}
    except Exception as exc:
        click.echo(f"  [rca] error: {exc}", err=True)
        return {"status": "error", "error": str(exc)}


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


@click.command()
@click.option("--fault",      required=True,
              type=click.Choice(["cpu_hog", "mem_leak", "net_delay", "disk_hog", "packet_loss"]))
@click.option("--service",    required=True, help="Primary target service name.")
@click.option("--duration",   default=120, show_default=True,
              help="Fault duration in seconds.")
@click.option("--run-id",     default=None,
              help="Run ID (timestamp-based ID generated if omitted).")
@click.option("--rps",        default=5.0, show_default=True,
              help="Load generator base RPS.")
@click.option("--concurrent", default=None,
              help="Comma-separated additional services to fault simultaneously.")
@click.option("--propagation-map", default=None,
              help="Path to calibration/propagation_delays.json for edge-aware RCA.")
@click.option("--isolate", is_flag=True, default=False,
              help="Move the target service to the experiment-target node before injecting "
                   "to eliminate hardware contention from neighbouring pods.")
def run(
    fault: str,
    service: str,
    duration: int,
    run_id: str | None,
    rps: float,
    concurrent: str | None,
    propagation_map: str | None,
    isolate: bool,
) -> None:
    """Run a full end-to-end fault injection experiment."""
    if run_id is None:
        run_id = gt.make_run_id()

    original_selector = None
    if isolate:
        from eval.isolate_service import move_to_experiment_node, restore_service
        original_selector = move_to_experiment_node(service)

    try:
        _run_body(
            fault=fault, service=service, duration=duration, run_id=run_id,
            rps=rps, concurrent=concurrent, propagation_map=propagation_map,
        )
    finally:
        if isolate and original_selector is not None:
            from eval.isolate_service import restore_service
            restore_service(service, original_selector)


def _run_body(
    fault: str,
    service: str,
    duration: int,
    run_id: str,
    rps: float,
    concurrent: str | None,
    propagation_map: str | None,
) -> None:
    """Heavy lifting; outer `run()` may wrap pod isolation around this."""

    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"  run_id  : {run_id}")
    click.echo(f"  fault   : {fault}  into  {service}")
    click.echo(f"  duration: {duration}s")
    click.echo(f"{'='*60}\n")

    timeline: dict = {
        "run_id":           run_id,
        "fault":            fault,
        "service":          service,
        "duration_seconds": duration,
        "events":           {},
        "windows":          {},
        "slo":              {},
    }
    results = ResultsStore(
        run_id=run_id,
        fault=fault,
        service=service,
        duration_seconds=duration,
    )
    client = PrometheusMetricsClient(prometheus_url=PROMETHEUS_URL)

    # 1. Load generator
    click.echo("[1/7] Starting load generator …")
    gen = WorkloadGenerator(frontend_url=FRONTEND_URL, quiet=True)
    total_gen_duration = BASELINE_DURATION + duration + RECOVERY_WAIT + 60
    gen.run(duration_seconds=total_gen_duration, base_rps=rps)
    experiment_start = _ts()
    timeline["events"]["experiment_start"] = experiment_start

    # 2. Baseline
    click.echo(f"[2/7] Baseline period ({BASELINE_DURATION}s) …")
    baseline_start = _ts()
    time.sleep(BASELINE_DURATION)
    baseline_end = _ts()
    timeline["events"]["baseline_end"] = baseline_end

    # Threshold = max(1.5 * baseline p95, P95_THRESHOLD_MS) — only used for logging here.
    baseline_p95 = gen.current_p95(window_seconds=BASELINE_DURATION)
    if baseline_p95 is not None:
        dynamic_threshold_ms = max(baseline_p95 * 1000 * 1.5, float(P95_THRESHOLD_MS))
    else:
        dynamic_threshold_ms = float(P95_THRESHOLD_MS)
    click.echo(
        f"  baseline p95={baseline_p95*1000:.0f}ms, SLO threshold={dynamic_threshold_ms:.0f}ms"
        if baseline_p95 else
        f"  baseline p95 unavailable, SLO threshold={dynamic_threshold_ms:.0f}ms"
    )
    click.echo(
        f"  baseline summary: p95={baseline_p95*1000:.0f}ms "
        f"threshold={dynamic_threshold_ms:.0f}ms"
    )
    timeline["baseline_p95_ms"]  = round(baseline_p95 * 1000, 1) if baseline_p95 else None
    timeline["slo_threshold_ms"] = dynamic_threshold_ms
    results.add_step(
        step="baseline",
        summary="Baseline period completed",
        details={
            "baseline_p95_ms": round(baseline_p95 * 1000, 1) if baseline_p95 else None,
            "slo_threshold_ms": dynamic_threshold_ms,
            "baseline_duration_s": BASELINE_DURATION,
        },
        timestamp=baseline_end,
    )

    # 3. Inject + spin up SLO poller (metadata only on this runner)
    # disk_hog → kubectl exec injector; everything else Chaos Mesh (distroless-safe).
    use_exec = fault in EXEC_ONLY_FAULTS
    inject_script = EXEC_INJECT_SCRIPT if use_exec else CHAOS_INJECT_SCRIPT
    injector_label = "exec" if use_exec else "chaos"

    click.echo(f"[3/7] Injecting fault '{fault}' into '{service}' (injector={injector_label}) …")
    inject_cmd = [
        sys.executable, str(inject_script),
        "--fault",     fault,
        "--service",   service,
        "--duration",  str(duration),
        "--run-id",    run_id,
        "--namespace", NAMESPACE,
    ]
    if concurrent:
        inject_cmd += ["--concurrent", concurrent]

    inject_proc = subprocess.Popen(
        inject_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    injection_time = _ts()
    timeline["events"]["injection_start"] = injection_time
    timeline["injector"] = injector_label
    results.add_step(
        step="injection",
        summary="Fault injection started",
        details={
            "injector": injector_label,
            "fault": fault,
            "service": service,
            "duration_seconds": duration,
        },
        timestamp=injection_time,
    )
    click.echo(f"  inject PID={inject_proc.pid}")

    slo_monitor = SLOMonitor(gen, threshold_ms=dynamic_threshold_ms)
    slo_monitor.start()

    # 4. Hold fault for the configured duration
    click.echo(f"[4/7] Fault active — waiting {duration}s …")
    time.sleep(duration)
    fault_end_time = _ts()
    timeline["events"]["fault_end"] = fault_end_time

    # First SLO crossing if any — often unset because this runner is duration-gated.
    violation_time = slo_monitor.stop()
    timeline["slo"]["violation_time"] = violation_time
    if violation_time is not None:
        diag_latency = round(violation_time - injection_time, 2)
        timeline["slo"]["diagnosis_latency_seconds"] = diag_latency
    else:
        timeline["slo"]["diagnosis_latency_seconds"] = None

    # 5. Pull Prom series
    # Baseline ends BASELINE_END_BUFFER before inject so onset isn’t in the fitted “normal”.
    # Fault slice is injection → fault_off.

    baseline_window_start = injection_time - BASELINE_DURATION
    baseline_window_end   = injection_time - BASELINE_END_BUFFER
    fault_window_start    = injection_time
    fault_window_end      = fault_end_time

    timeline["windows"] = {
        "baseline_start": baseline_window_start,
        "baseline_end":   baseline_window_end,
        "fault_start":    fault_window_start,
        "fault_end":      fault_window_end,
    }

    click.echo(
        f"[5/7] Collecting metrics …\n"
        f"  baseline [{_iso(baseline_window_start)} to {_iso(baseline_window_end)}]\n"
        f"  fault    [{_iso(fault_window_start)} to {_iso(fault_window_end)}]"
    )
    try:
        df     = client.fetch_metrics(baseline_window_start, fault_window_end)
        matrix = client.fetch_metric_matrix(baseline_window_start, fault_window_end)
        if not df.empty:
            df.to_parquet(run_dir / "metrics.parquet", index=False)
            click.echo(
                f"  saved metrics.parquet  ({len(df):,} rows, {df['service'].nunique()} services)"
            )
            click.echo(
                f"  collected summary: {df['service'].nunique()} services, "
                f"{df['metric'].nunique()} metrics"
            )
            results.add_step(
                step="metrics",
                summary="Metrics collection completed",
                details={
                    "rows": len(df),
                    "services": df["service"].nunique(),
                    "metrics": df["metric"].nunique(),
                },
                timestamp=_ts(),
            )
        else:
            click.echo("  WARNING: no metrics returned — metrics.parquet not saved")
            matrix = {}
            results.add_step(
                step="metrics",
                summary="Metrics collection returned no data",
                details={
                    "services": 0,
                    "metrics": 0,
                    "rows": 0,
                },
                timestamp=_ts(),
            )
    except Exception as exc:
        click.echo(f"  WARNING: metrics fetch failed: {exc}", err=True)
        matrix = {}
        results.add_step(
            step="metrics",
            summary="Metrics collection failed",
            details={"error": str(exc)},
            timestamp=_ts(),
        )

    # 6. RCA / pinpoint
    click.echo("[6/7] Running RCA …")
    rca_start = _ts()
    rca_results = run_rca(
        metric_matrix=matrix,
        baseline_window=(baseline_window_start, baseline_window_end),
        fault_window=(fault_window_start, fault_window_end),
        run_dir=run_dir,
        propagation_map_path=propagation_map,
    )
    rca_end = _ts()
    timeline["events"]["rca_start"] = rca_start
    timeline["events"]["rca_end"]   = rca_end
    _save_json(run_dir / "rca_results.json", rca_results)
    click.echo(f"  rca done in {rca_end - rca_start:.1f}s")
    ranked_services = rca_results.get("ranked_services") if isinstance(rca_results, dict) else []
    if ranked_services:
        top = ranked_services[0]
        click.echo(
            f"  rca summary: top={top.get('service', 'unknown')} "
            f"confidence={top.get('confidence', 0.0):.3f} "
            f"ranked={len(ranked_services)}"
        )
    else:
        click.echo("  rca summary: no ranked services returned")
    results.add_step(
        step="rca",
        summary="RCA completed",
        details={
            "ranked_services": len(ranked_services),
            "top_service": top.get("service", "unknown") if ranked_services else None,
            "top_confidence": top.get("confidence", 0.0) if ranked_services else None,
        },
        timestamp=rca_end,
    )

    # 7. Drain injector + short recovery soak
    click.echo(f"[7/7] Collecting inject output + recovery ({RECOVERY_WAIT}s) …")
    try:
        # injector should exit around now; tolerate a slower cleanup
        stdout, stderr = inject_proc.communicate(timeout=30)
        if stdout:
            click.echo(stdout.strip())
        if inject_proc.returncode != 0 and stderr:
            click.echo(f"  inject stderr: {stderr.strip()}", err=True)
    except subprocess.TimeoutExpired:
        click.echo("  inject script still running after fault window — killing", err=True)
        inject_proc.kill()
        inject_proc.communicate()
    timeline["events"]["fault_stopped"] = _ts()

    time.sleep(RECOVERY_WAIT)
    gen.stop()
    experiment_end = _ts()
    timeline["events"]["recovery_end"]   = experiment_end
    timeline["events"]["experiment_end"] = experiment_end

    results.add_step(
        step="recovery",
        summary="Recovery completed",
        details={
            "slo_violation_time": violation_time,
            "diagnosis_latency_seconds": timeline["slo"]["diagnosis_latency_seconds"],
            "recovery_wait_seconds": RECOVERY_WAIT,
        },
        timestamp=experiment_end,
    )

    results.set_summary({
        "total_duration_seconds": experiment_end - experiment_start,
        "top_rca_service": ranked_services[0].get("service") if ranked_services else None,
        "top_rca_confidence": ranked_services[0].get("confidence") if ranked_services else None,
    })
    results.save(run_dir / "results.json")

    _save_json(run_dir / "timeline.json", timeline)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Experiment complete  —  artifacts in {run_dir}")
    click.echo(f"  Total duration : {experiment_end - experiment_start:.0f}s")
    diag = timeline["slo"].get("diagnosis_latency_seconds")
    click.echo(
        f"  SLO violation  : t+{diag}s after injection"
        if diag is not None else
        f"  SLO violation  : not detected (p95 stayed below {dynamic_threshold_ms:.0f}ms)"
    )
    click.echo("  Files:")
    for f in sorted(run_dir.iterdir()):
        click.echo(f"    {f.name}")
    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    run()
