"""Walk YAML matrix rows and invoke run_experiment.py one after another.

    python eval/run_batch.py --matrix experiments/experiment_matrix.yaml
    python eval/run_batch.py --matrix experiments/experiment_matrix.yaml --cooldown 120
"""

import subprocess
import sys
import time
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).parent.parent
RUN_SCRIPT = Path(__file__).parent / "run_experiment.py"


@click.command()
@click.option(
    "--matrix",
    required=True,
    type=click.Path(exists=True),
    help="Path to experiment_matrix.yaml.",
)
@click.option(
    "--cooldown",
    default=None,
    type=int,
    help="Seconds to wait between runs (overrides value in YAML).",
)
@click.option("--dry-run", is_flag=True, help="Print commands without running.")
def run_batch(matrix: str, cooldown: int | None, dry_run: bool) -> None:
    """Fire every stanza from the YAML in order."""
    config = yaml.safe_load(Path(matrix).read_text())
    experiments = config.get("experiments", [])
    cooldown_s   = cooldown if cooldown is not None else config.get("cooldown_seconds", 120)

    click.echo(f"Loaded {len(experiments)} experiment(s) from {matrix}")
    click.echo(f"Cooldown between runs: {cooldown_s}s\n")

    failed = []

    for i, exp in enumerate(experiments, start=1):
        fault    = exp["fault"]
        service  = exp["service"]
        duration = exp.get("duration", 120)
        run_id   = exp.get("run_id", None)
        rps      = exp.get("rps", 5.0)
        concurrent = exp.get("concurrent", None)

        click.echo(f"[{i}/{len(experiments)}] fault={fault}  service={service}  duration={duration}s")

        cmd = [
            sys.executable, str(RUN_SCRIPT),
            "--fault", fault,
            "--service", service,
            "--duration", str(duration),
            "--rps", str(rps),
        ]
        if run_id:
            cmd += ["--run-id", run_id]
        if concurrent:
            cmd += ["--concurrent", concurrent]

        if dry_run:
            click.echo(f"  [dry-run] {' '.join(cmd)}\n")
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            click.echo(f"  FAILED (exit {result.returncode})", err=True)
            failed.append(f"{fault}/{service}")

        if i < len(experiments):
            click.echo(f"\n{cooldown_s}s before next run …\n")
            time.sleep(cooldown_s)

    click.echo(f"\n{'='*50}")
    click.echo(f"Batch complete.  {len(experiments) - len(failed)}/{len(experiments)} succeeded.")
    if failed:
        click.echo(f"Failed: {failed}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    run_batch()
