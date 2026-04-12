"""Chaos Mesh fault injection CLI.

Applies pre-templated Chaos Mesh CRs, waits for injection confirmation,
then cleans up on exit.  Drop-in replacement for inject.py that works on
distroless containers (no shell required).

Usage:
    python fault_injection/chaos_inject.py --fault cpu_hog   --service frontend    --duration 60
    python fault_injection/chaos_inject.py --fault net_delay --service cartservice  --duration 120
    python fault_injection/chaos_inject.py --fault cpu_hog   --service emailservice --concurrent currencyservice --duration 90
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import fault_injection.ground_truth as gt

MANIFESTS_DIR = Path(__file__).parent / "chaos_manifests"
EXPERIMENTS_DIR = ROOT / "experiments"
NAMESPACE = "boutique"

# Maps fault name → (template filename, Chaos Mesh kind)
FAULT_MAP: dict[str, tuple[str, str]] = {
    "cpu_hog":     ("cpu_hog.yaml",     "StressChaos"),
    "mem_leak":    ("mem_leak.yaml",    "StressChaos"),
    "net_delay":   ("net_delay.yaml",   "NetworkChaos"),
    "disk_hog":    ("disk_hog.yaml",    "IOChaos"),
    "packet_loss": ("packet_loss.yaml", "NetworkChaos"),
}


def _kubectl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, check=check)


def _render_manifest(fault: str, service: str, duration_s: int) -> dict:
    """Load template YAML and substitute TARGET_SERVICE + DURATION_VALUE."""
    template_file, _ = FAULT_MAP[fault]
    raw = (MANIFESTS_DIR / template_file).read_text()
    # Replace both the name suffix and the selector label
    raw = raw.replace("TARGET_SERVICE", service)
    raw = raw.replace("TARGET", service)
    raw = raw.replace("DURATION_VALUE", f"{duration_s}s")
    return yaml.safe_load(raw)


def _apply_manifest(manifest: dict) -> tuple[str, str]:
    """kubectl apply a manifest dict. Returns (kind, name)."""
    kind = manifest["metadata"].get("kind") or manifest.get("kind")
    # kind is at top level in the yaml
    kind = manifest["kind"]
    name = manifest["metadata"]["name"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(manifest, f)
        tmpfile = f.name
    try:
        result = _kubectl("apply", "-f", tmpfile)
        if result.returncode != 0:
            click.echo(f"  ERROR applying {kind}/{name}: {result.stderr}", err=True)
            sys.exit(1)
    finally:
        Path(tmpfile).unlink(missing_ok=True)
    return kind, name


def _wait_for_injection(kind: str, name: str, timeout: int = 60) -> bool:
    """Poll until the chaos resource reports Injected=True or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _kubectl(
            "get", kind.lower(), name,
            "-n", NAMESPACE,
            "-o", "jsonpath={.status.conditions}",
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            try:
                conditions = json.loads(result.stdout)
                for cond in conditions:
                    if cond.get("type") == "AllInjected" and cond.get("status") == "True":
                        return True
                    if cond.get("type") == "Failed":
                        click.echo(
                            f"  ERROR: {kind}/{name} failed: {cond.get('reason', 'unknown')}",
                            err=True,
                        )
                        return False
            except (json.JSONDecodeError, TypeError):
                pass
        time.sleep(3)
    click.echo(f"  WARNING: {kind}/{name} did not confirm injection within {timeout}s", err=True)
    return False


def _delete_resource(kind: str, name: str) -> None:
    result = _kubectl("delete", kind.lower(), name, "-n", NAMESPACE, "--ignore-not-found", check=False)
    if result.returncode != 0:
        click.echo(f"  WARNING: could not delete {kind}/{name}: {result.stderr}", err=True)


def inject_one(fault: str, service: str, duration_s: int) -> tuple[str, str]:
    """Render, apply, and wait for one chaos resource. Returns (kind, name)."""
    manifest = _render_manifest(fault, service, duration_s)
    kind, name = _apply_manifest(manifest)
    click.echo(f"  applied {kind}/{name}")
    injected = _wait_for_injection(kind, name)
    if injected:
        click.echo(f"  confirmed Injected: {kind}/{name}")
    return kind, name


@click.command()
@click.option("--fault",      required=True, type=click.Choice(list(FAULT_MAP.keys())))
@click.option("--service",    required=True, help="Primary target service label.")
@click.option("--duration",   default=120,   show_default=True, help="Fault duration in seconds.")
@click.option("--run-id",     default=None,  help="Run ID for ground truth recording.")
@click.option("--namespace",  default=NAMESPACE, show_default=True)
@click.option("--concurrent", default=None,  help="Comma-separated additional services.")
def run(
    fault: str,
    service: str,
    duration: int,
    run_id: str | None,
    namespace: str,
    concurrent: str | None,
) -> None:
    """Apply Chaos Mesh fault CRs and wait for duration, then clean up."""
    if run_id is None:
        run_id = gt.make_run_id()

    services = [service]
    if concurrent:
        services += [s.strip() for s in concurrent.split(",")]

    click.echo(
        f"Injecting '{fault}' into {services} for {duration}s  (run_id={run_id})"
    )

    # Save ground truth
    run_dir = EXPERIMENTS_DIR / run_id
    gt_path = gt.write(
        run_id=run_id,
        fault_type=fault,
        target_services=services,
        duration_seconds=duration,
        output_dir=run_dir,
    )
    click.echo(f"Ground truth in {gt_path}")

    # Apply all chaos resources
    resources: list[tuple[str, str]] = []
    for svc in services:
        kind, name = inject_one(fault, svc, duration)
        resources.append((kind, name))

    # Wait for fault to run its duration
    click.echo(f"  waiting {duration}s …")
    time.sleep(duration)

    # Cleanup
    click.echo("  cleaning up chaos resources …")
    for kind, name in resources:
        _delete_resource(kind, name)
    click.echo("Injection complete.")


if __name__ == "__main__":
    run()
