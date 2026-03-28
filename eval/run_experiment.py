"""Experiment orchestrator: inject → collect → RCA → score.

Automates a single fault injection experiment end-to-end.

Usage:
    python eval/run_experiment.py --fault cpu_hog --service checkoutservice --duration 120
    python eval/run_experiment.py --fault cpu_hog --service checkoutservice --duration 120 --run-id run_001
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.loadgen import WorkloadGenerator
from rca_engine.metrics_client import PrometheusMetricsClient
import fault_injection.ground_truth as gt

EXPERIMENTS_DIR = ROOT / "experiments"
INJECT_SCRIPT   = ROOT / "fault_injection" / "inject.py"

P95_THRESHOLD_MS  = 500
BASELINE_DURATION = 60    # seconds of steady-state before injection
PROPAGATION_WAIT  = 30    # seconds after SLO violation before calling RCA
RECOVERY_WAIT     = 30    # seconds after fault stops for recovery metrics
SLO_POLL_INTERVAL = 5     # seconds between SLO checks
SLO_TIMEOUT       = 300   # give up waiting for violation after this many seconds
FRONTEND_URL      = "http://localhost:8080"
PROMETHEUS_URL    = "http://localhost:9090"
NAMESPACE         = "boutique"


# ---------------------------------------------------------------------------
# SLO monitoring
# ---------------------------------------------------------------------------

def poll_for_slo_violation(
    gen: WorkloadGenerator,
    timeout: float,
) -> float | None:
    """Poll for SLO violation every SLO_POLL_INTERVAL seconds using loadgen p95.

    Returns the POSIX timestamp of the first violation, or None if timeout is reached.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        p95 = gen.current_p95(window_seconds=10)
        if p95 is not None:
            ms = p95 * 1000
            click.echo(f"  [slo] p95={ms:.0f}ms")
            if ms > P95_THRESHOLD_MS:
                return time.time()
        time.sleep(SLO_POLL_INTERVAL)

    return None


# ---------------------------------------------------------------------------
# RCA stub — replaced once fault_chain is implemented
# ---------------------------------------------------------------------------

def run_rca(
    metric_matrix: dict,
    window_start: float,
    window_end: float,
    baseline_start: float,
    baseline_end: float,
    run_dir: Path,
) -> dict:
    """Call the RCA engine and return a results dict."""
    try:
        from rca_engine import fault_chain
        ranked = fault_chain.pinpoint(
            anomaly_scores={},          # filled in once normal_model is built
            dependency_graph={},
            change_points={},
        )
        return {"ranked_services": ranked}
    except NotImplementedError:
        click.echo("  [rca] fault_chain.pinpoint not yet implemented — saving placeholder")
        return {"status": "rca_not_implemented"}
    except Exception as exc:
        click.echo(f"  [rca] error: {exc}", err=True)
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Artifact helpers
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
@click.option("--fault",    required=True,
              type=click.Choice(["cpu_hog", "mem_leak", "net_delay", "disk_hog"]))
@click.option("--service",  required=True, help="Primary target service name.")
@click.option("--duration", default=120, show_default=True,
              help="Fault duration in seconds.")
@click.option("--run-id",   default=None,
              help="Run ID (timestamp-based ID generated if omitted).")
@click.option("--rps",      default=5.0, show_default=True,
              help="Load generator base RPS.")
@click.option("--concurrent", default=None,
              help="Comma-separated additional services to fault simultaneously.")
def run(
    fault: str,
    service: str,
    duration: int,
    run_id: str | None,
    rps: float,
    concurrent: str | None,
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

    timeline: dict = {"run_id": run_id, "fault": fault, "service": service,
                      "duration_seconds": duration, "events": {}}
    client = PrometheusMetricsClient(prometheus_url=PROMETHEUS_URL)

    # ------------------------------------------------------------------
    # 1. Start load generator
    # ------------------------------------------------------------------
    click.echo("[1/9] Starting load generator …")
    gen = WorkloadGenerator(frontend_url=FRONTEND_URL, quiet=True)
    # Run for baseline + duration + propagation + recovery + buffer
    total_gen_duration = BASELINE_DURATION + duration + PROPAGATION_WAIT + RECOVERY_WAIT + 60
    gen.run(duration_seconds=total_gen_duration, base_rps=rps, pattern="constant")
    timeline["events"]["experiment_start"] = _ts()

    # ------------------------------------------------------------------
    # 2. Baseline period
    # ------------------------------------------------------------------
    click.echo(f"[2/9] Baseline period ({BASELINE_DURATION}s) …")
    baseline_start = _ts()
    time.sleep(BASELINE_DURATION)
    baseline_end = _ts()
    timeline["events"]["baseline_end"] = baseline_end

    # ------------------------------------------------------------------
    # 3. Inject fault (background subprocess)
    # ------------------------------------------------------------------
    click.echo(f"[3/9] Injecting fault '{fault}' into '{service}' …")
    inject_cmd = [
        sys.executable, str(INJECT_SCRIPT),
        "--fault", fault,
        "--service", service,
        "--duration", str(duration),
        "--run-id", run_id,
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
    click.echo(f"  inject PID={inject_proc.pid}")

    # ------------------------------------------------------------------
    # 4. Poll for SLO violation
    # ------------------------------------------------------------------
    click.echo(f"[4/9] Polling for SLO violation (p95 > {P95_THRESHOLD_MS}ms, timeout={SLO_TIMEOUT}s) …")
    violation_time = poll_for_slo_violation(gen, timeout=SLO_TIMEOUT)
    if violation_time:
        click.echo(f"  SLO violated at t+{violation_time - injection_time:.0f}s after injection")
        timeline["events"]["slo_violation"] = violation_time
    else:
        click.echo("  No SLO violation detected — continuing with injection end time as reference")
        violation_time = injection_time + duration
        timeline["events"]["slo_violation"] = None

    # ------------------------------------------------------------------
    # 5. Propagation window
    # ------------------------------------------------------------------
    click.echo(f"[5/9] Waiting {PROPAGATION_WAIT}s for fault propagation …")
    time.sleep(PROPAGATION_WAIT)

    # ------------------------------------------------------------------
    # 6. Collect metrics window for RCA
    # ------------------------------------------------------------------
    rca_window_end   = _ts()
    rca_window_start = max(baseline_start, violation_time - 100)
    click.echo(
        f"[6/9] Collecting metrics window "
        f"[{_iso(rca_window_start)} to {_iso(rca_window_end)}] …"
    )
    try:
        df = client.fetch_metrics(rca_window_start, rca_window_end)
        matrix = client.fetch_metric_matrix(rca_window_start, rca_window_end)
        if not df.empty:
            df.to_parquet(run_dir / "metrics.parquet", index=False)
            click.echo(f"  saved metrics.parquet  ({len(df):,} rows, {df['service'].nunique()} services)")
        else:
            click.echo("  WARNING: no metrics returned — metrics.parquet not saved")
            matrix = {}
    except Exception as exc:
        click.echo(f"  WARNING: metrics fetch failed: {exc}", err=True)
        matrix = {}

    # ------------------------------------------------------------------
    # 7. Run RCA
    # ------------------------------------------------------------------
    click.echo("[7/9] Running RCA …")
    rca_start = _ts()
    rca_results = run_rca(matrix, rca_window_start, rca_window_end,
                          baseline_start, baseline_end, run_dir)
    rca_end = _ts()
    timeline["events"]["rca_start"] = rca_start
    timeline["events"]["rca_end"]   = rca_end
    _save_json(run_dir / "rca_results.json", rca_results)
    click.echo(f"  rca done in {rca_end - rca_start:.1f}s")

    # ------------------------------------------------------------------
    # 8. Wait for fault subprocess to finish
    # ------------------------------------------------------------------
    click.echo("[8/9] Waiting for fault script to finish …")
    try:
        stdout, stderr = inject_proc.communicate(timeout=duration + 30)
        if stdout:
            click.echo(stdout.strip())
        if inject_proc.returncode != 0 and stderr:
            click.echo(f"  inject stderr: {stderr.strip()}", err=True)
    except subprocess.TimeoutExpired:
        click.echo("  inject timed out — terminating", err=True)
        inject_proc.kill()
    timeline["events"]["fault_stopped"] = _ts()

    # ------------------------------------------------------------------
    # 9. Recovery + wrap-up
    # ------------------------------------------------------------------
    click.echo(f"[9/9] Recovery period ({RECOVERY_WAIT}s) …")
    time.sleep(RECOVERY_WAIT)
    gen.stop()
    experiment_end = _ts()
    timeline["events"]["recovery_end"]    = experiment_end
    timeline["events"]["experiment_end"]  = experiment_end

    # Diagnosis latency — time from injection to SLO detection
    if timeline["events"]["slo_violation"]:
        timeline["diagnosis_latency_seconds"] = round(
            timeline["events"]["slo_violation"] - injection_time, 2
        )

    _save_json(run_dir / "timeline.json", timeline)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    click.echo(f"\n{'='*60}")
    click.echo(f"  Experiment complete  —  artifacts in {run_dir}")
    click.echo(f"  Duration : {experiment_end - timeline['events']['experiment_start']:.0f}s total")
    if timeline.get("diagnosis_latency_seconds") is not None:
        click.echo(f"  Diagnosis latency: {timeline['diagnosis_latency_seconds']}s")
    click.echo(f"  Files:")
    for f in sorted(run_dir.iterdir()):
        click.echo(f"    {f.name}")
    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    run()
