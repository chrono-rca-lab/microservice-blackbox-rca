"""Experiment orchestrator: inject → collect → RCA → score.

Fixed-schedule pipeline — the SLO monitor runs in the background and records
diagnosis latency as metadata, but never gates the experiment.

Injector selection:
  - cpu_hog / mem_leak / net_delay / packet_loss → chaos_inject.py (Chaos Mesh)
      Works on all services including distroless Go containers.
  - disk_hog → inject.py (kubectl exec)
      Fallback: IOChaos requires FUSE kernel support not available on kind.
      disk_hog only works on services with /bin/sh (adservice, recommendationservice,
      emailservice, paymentservice, currencyservice).

Schedule:
    1. Start load generator
    2. Baseline period (BASELINE_DURATION s)
    3. Inject fault + start SLO monitor (background)
    4. Wait fault duration
    5. Collect metrics (windows anchored to injection time)
    6. Run RCA
    7. Reap inject subprocess + recovery

Usage:
    python eval/run_experiment.py --fault cpu_hog   --service frontend            --duration 120
    python eval/run_experiment.py --fault net_delay --service cartservice          --duration 120
    python eval/run_experiment.py --fault disk_hog  --service currencyservice      --duration 120
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

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.loadgen import WorkloadGenerator
from rca_engine.metrics_client import PrometheusMetricsClient
import fault_injection.ground_truth as gt
from results import ResultsStore

EXPERIMENTS_DIR    = ROOT / "experiments"
CHAOS_INJECT_SCRIPT = ROOT / "fault_injection" / "chaos_inject.py"
EXEC_INJECT_SCRIPT  = ROOT / "fault_injection" / "inject.py"

# Faults handled by the old kubectl-exec injector (shell required in container).
# IOChaos needs FUSE which is unavailable on kind — fall back to shell script.
EXEC_ONLY_FAULTS = {"disk_hog"}

P95_THRESHOLD_MS     = 500   # floor for the dynamic SLO threshold
BASELINE_DURATION    = 310   # seconds of steady-state before injection
BASELINE_END_BUFFER  = 10    # seconds trimmed from the tail of the baseline window
                              # (avoids capturing the transition moment in the normal distribution)
RECOVERY_WAIT        = 30    # seconds of post-fault observation before stopping loadgen
SLO_POLL_INTERVAL    = 5     # seconds between SLO log lines
FRONTEND_URL         = "http://localhost:8080"
PROMETHEUS_URL       = "http://localhost:9090"
NAMESPACE            = "boutique"


# ---------------------------------------------------------------------------
# SLO monitor — background thread, non-blocking, metadata only
# ---------------------------------------------------------------------------

class SLOMonitor:
    """Logs frontend p95 latency every SLO_POLL_INTERVAL seconds.

    Records the first violation timestamp.  Runs entirely in the background;
    does NOT gate the experiment — the caller decides when to start/stop it.
    """

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
        # Wait first so the initial window is post-injection, not pre-injection.
        # Runs silently — only logs when a violation fires.
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


# ---------------------------------------------------------------------------
# RCA — calls fault_chain.pinpoint once implemented
# ---------------------------------------------------------------------------

def run_rca(
    metric_matrix: dict,
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    run_dir: Path,
    propagation_map_path: str | None = None,
) -> dict:
    """Call the RCA engine and return a results dict."""
    try:
        from rca_engine import fault_chain
        
        # Initialize timing collection
        start_time = time.time()
        logs: list[dict] = []
        
        ranked = fault_chain.pinpoint(
            metric_matrix=metric_matrix,
            baseline_window=baseline_window,
            fault_window=fault_window,
            propagation_map_path=propagation_map_path,
            start_time=start_time,
            logs=logs,
        )
        
        # Prepare output with timing information
        total_time = time.time() - start_time
        result = {
            "ranked_services": ranked,
            "total_time_seconds": total_time,
            "timing_logs": logs,
        }
        
        # Save timing JSON to file
        output_file = run_dir / "rca_timing.json"
        output_file.write_text(json.dumps(result, indent=2))
        click.echo(f"  [rca] timing data saved to {output_file}")
        
        # Also write a human-readable text summary for quick inspection
        txt_lines: list[str] = []
        txt_lines.append("RCA Timing Results")
        txt_lines.append(f"Saved: {datetime.now(timezone.utc).isoformat()}")
        txt_lines.append("")
        txt_lines.append(f"Total RCA time: {total_time:.3f} seconds")
        txt_lines.append("")
        txt_lines.append("Stages:")
        for entry in logs:
            try:
                ts = datetime.fromtimestamp(entry.get("timestamp", 0), timezone.utc).isoformat()
            except Exception:
                ts = str(entry.get("timestamp", ""))
            txt_lines.append(
                f"- {entry.get('stage',''):<20} | {entry.get('file',''):<20} | "
                f"duration={entry.get('duration_seconds', 0):.3f}s | "
                f"since_start={entry.get('since_start_seconds', 0):.3f}s | {ts}"
            )

        txt_lines.append("")
        txt_lines.append("Ranked Services:")
        if not ranked:
            txt_lines.append("(none)")
        else:
            for svc in ranked:
                onset = svc.get("onset_time")
                onset_str = (
                    datetime.fromtimestamp(onset, timezone.utc).isoformat()
                    if isinstance(onset, (int, float))
                    else str(onset)
                )
                txt_lines.append(
                    f"{svc.get('rank', ''):>2}. {svc.get('service',''):<30} onset={onset_str} "
                    f"conf={svc.get('confidence', 0.0):.3f} root={svc.get('is_root_cause', False)}"
                )
                am = svc.get('abnormal_metrics') or []
                if am:
                    txt_lines.append(f"    metrics: {', '.join(am)}")

        output_txt = run_dir / "output.txt"
        output_txt.write_text("\n".join(txt_lines))
        click.echo(f"  [rca] text summary saved to {output_txt}")

        return {"ranked_services": ranked}
    except NotImplementedError:
        click.echo("  [rca] fault_chain.pinpoint not yet implemented — saving placeholder")
        return {"status": "rca_not_implemented"}
    except Exception as exc:
        click.echo(f"  [rca] error: {exc}", err=True)
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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
def run(
    fault: str,
    service: str,
    duration: int,
    run_id: str | None,
    rps: float,
    concurrent: str | None,
    propagation_map: str | None,
) -> None:
    """Run a full end-to-end fault injection experiment."""
    if run_id is None:
        run_id = gt.make_run_id()

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

    # ------------------------------------------------------------------
    # 1. Start load generator
    # ------------------------------------------------------------------
    click.echo("[1/7] Starting load generator …")
    gen = WorkloadGenerator(frontend_url=FRONTEND_URL, quiet=True)
    total_gen_duration = BASELINE_DURATION + duration + RECOVERY_WAIT + 60
    gen.run(duration_seconds=total_gen_duration, base_rps=rps)
    experiment_start = _ts()
    timeline["events"]["experiment_start"] = experiment_start

    # ------------------------------------------------------------------
    # 2. Baseline period
    # ------------------------------------------------------------------
    click.echo(f"[2/7] Baseline period ({BASELINE_DURATION}s) …")
    baseline_start = _ts()
    time.sleep(BASELINE_DURATION)
    baseline_end = _ts()
    timeline["events"]["baseline_end"] = baseline_end

    # Dynamic SLO threshold: 1.5× measured baseline p95, floored at P95_THRESHOLD_MS
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

    # ------------------------------------------------------------------
    # 3. Inject fault + start SLO monitor in background
    # ------------------------------------------------------------------
    # Routing: disk_hog falls back to kubectl-exec (shell-based) because
    # IOChaos requires FUSE kernel support unavailable on kind clusters.
    # All other faults go through Chaos Mesh (works on distroless containers).
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

    # ------------------------------------------------------------------
    # 4. Wait for fault to run its full duration
    # ------------------------------------------------------------------
    click.echo(f"[4/7] Fault active — waiting {duration}s …")
    time.sleep(duration)
    fault_end_time = _ts()
    timeline["events"]["fault_end"] = fault_end_time

    # Harvest SLO result — violation_time is None if threshold was never crossed
    violation_time = slo_monitor.stop()
    timeline["slo"]["violation_time"] = violation_time
    if violation_time is not None:
        diag_latency = round(violation_time - injection_time, 2)
        timeline["slo"]["diagnosis_latency_seconds"] = diag_latency
    else:
        timeline["slo"]["diagnosis_latency_seconds"] = None

    # ------------------------------------------------------------------
    # 5. Collect metrics — windows anchored to injection time
    # ------------------------------------------------------------------
    # baseline_window: [injection_time - BASELINE_DURATION,
    #                   injection_time - BASELINE_END_BUFFER]
    #   Trims the final BASELINE_END_BUFFER seconds before injection to avoid
    #   capturing any transition effects in the normal-behaviour distribution.
    #
    # fault_window: [injection_time, fault_end_time]
    #   The exact period the fault was active.

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

    # ------------------------------------------------------------------
    # 6. Run RCA
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 7. Reap inject subprocess + recovery period
    # ------------------------------------------------------------------
    click.echo(f"[7/7] Collecting inject output + recovery ({RECOVERY_WAIT}s) …")
    try:
        # Fault duration has already elapsed — subprocess should be done or nearly so
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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
