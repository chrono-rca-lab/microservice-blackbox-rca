"""Train and save Markov chain checkpoints for all services and metrics.

Usage
-----
    python scripts/train_markov_checkpoints.py [OPTIONS]

Options
-------
    --config            Path to checkpoint_config.json
                        (default: checkpoint_config.json)
    --prometheus-url    Override Prometheus URL from config
    --step              Override scrape step in seconds from config
    --checkpoint-dir    Override checkpoint root directory from config
    --services          Comma-separated list of services to train
                        (default: all Online Boutique services)
    --frontend-url      Frontend URL for workload generation
    --rps               Base RPS for load generator (sine pattern)
    --warmup-seconds    Warmup before baseline capture starts
    --no-loadgen        Do not start load generator (use existing traffic)
    --dry-run           Print what would be trained without saving

Cumulative window model
-----------------------
Each checkpoint is trained on data from t=0 to t=window_seconds,
measured backward from the query end time.  This means:

    5m  checkpoint: data[0 : 300]      (first 5 minutes)
    30m checkpoint: data[0 : 1800]     (first 30 minutes, includes 5m)
    1h  checkpoint: data[0 : 3600]     (first 1 hour, includes 30m)
    ... and so on.

When loadgen is enabled (default), baseline capture mirrors run_experiment:
traffic is generated first, then baseline is collected from Prometheus over
the largest configured window. All smaller windows are contiguous prefixes
of that baseline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from infra.loadgen import WorkloadGenerator
import rca_engine.metrics_client as metrics_client
from rca_engine.metrics_client import PrometheusMetricsClient
from rca_engine.markov_checkpoint import (
    MarkovCheckpoint,
    checkpoint_path,
    get_window_seconds,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train_checkpoint,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_FRONTEND_URL = "http://localhost:8080"
_LOADGEN_TAIL_BUFFER_SECONDS = 10
_PROGRESS_LOG_INTERVAL_SECONDS = 30


def _sleep_with_progress(
    label: str,
    total_seconds: int,
    interval_seconds: int = _PROGRESS_LOG_INTERVAL_SECONDS,
) -> None:
    """Sleep while logging elapsed progress in seconds."""
    total = max(0, int(total_seconds))
    if total == 0:
        logger.info("%s progress: 0/0 seconds done.", label)
        return

    logger.info("%s progress: 0/%d seconds done.", label, total)
    started_at = time.time()
    next_log_at = started_at + max(1, int(interval_seconds))
    end_at = started_at + total

    while True:
        now = time.time()
        if now >= end_at:
            break

        sleep_for = min(max(0.0, next_log_at - now), end_at - now)
        if sleep_for > 0:
            time.sleep(sleep_for)

        now = time.time()
        elapsed = min(total, int(now - started_at))
        if now >= next_log_at or elapsed == total:
            logger.info("%s progress: %d/%d seconds done.", label, elapsed, total)
            next_log_at += max(1, int(interval_seconds))

    logger.info("%s progress: %d/%d seconds done.", label, total, total)

# ---------------------------------------------------------------------------
# Online Boutique services and Prometheus metric queries
# ---------------------------------------------------------------------------

SERVICES: tuple[str, ...] = (
    "frontend",
    "cartservice",
    "productcatalogservice",
    "currencyservice",
    "paymentservice",
    "shippingservice",
    "emailservice",
    "checkoutservice",
    "recommendationservice",
    "adservice",
    "redis-cart",
)

METRIC_QUERIES: dict[str, str] = metrics_client.QUERIES.copy()

# Headroom added to metric_max so fault-period excursions are inside
# the bin range and trigger the unseen-state path correctly.
_RANGE_HEADROOM = 0.20


# ---------------------------------------------------------------------------
# Prometheus helper
# ---------------------------------------------------------------------------

def query_service_metric(
    client: PrometheusMetricsClient,
    metric_name: str,
    service: str,
    start: float,
    end: float,
    step: int,
) -> np.ndarray | None:
    """Return one service/metric 1-D array averaged across pods.

    Uses the same PromQL definitions and query path as metrics_client.py.
    """
    query = METRIC_QUERIES[metric_name]
    try:
        result = client._query_range(query, start, end, f"{step}s")
    except (ConnectionError, RuntimeError) as exc:
        logger.warning("Prometheus query failed: %s", exc)
        return None

    if not result:
        return None

    series_list: list[np.ndarray] = []
    for series in result:
        pod = series.get("metric", {}).get("pod", "")
        if not pod:
            continue
        if metrics_client._pod_to_service(pod) != service:
            continue
        values = np.array([float(v[1]) for v in series["values"]], dtype=float)
        if len(values) > 0:
            series_list.append(values)

    if not series_list:
        return None

    min_len = min(len(s) for s in series_list)
    return np.stack([s[:min_len] for s in series_list]).mean(axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all_checkpoints(
    config: dict,
    prometheus_url: str,
    step: int,
    checkpoint_dir: Path,
    services: tuple[str, ...],
    dry_run: bool,
    config_path: Path,
    baseline_start: float,
    baseline_end: float,
) -> None:
    """Train and save cumulative checkpoints for all (service, metric, window)."""

    windows = get_window_seconds(config_path)
    largest_window = max(windows)
    num_bins = int(config.get("num_bins", 100))
    client = PrometheusMetricsClient(prometheus_url=prometheus_url)
    total = len(services) * len(METRIC_QUERIES) * len(windows)
    done  = 0

    for service in services:
        for metric_name in METRIC_QUERIES:

            # Fetch the full largest window from Prometheus once.
            # All smaller windows will be prefixes of this data.
            logger.info(
                "Fetching %ds of baseline data for %s/%s ...",
                largest_window, service, metric_name,
            )
            full_data = query_service_metric(
                client=client,
                metric_name=metric_name,
                service=service,
                start = baseline_start,
                end   = baseline_end,
                step  = step,
            )

            if full_data is None or len(full_data) < 2:
                logger.warning(
                    "No data returned for %s/%s — skipping.",
                    service, metric_name,
                )
                done += len(windows)
                continue

            # Metric range from the full data plus headroom.
            data_min   = float(np.min(full_data))
            data_max   = float(np.max(full_data))
            data_range = data_max - data_min
            if data_range == 0:
                data_range = max(abs(data_min) * 0.1, 1e-6)

            metric_min = data_min - data_range * _RANGE_HEADROOM
            metric_max = data_max + data_range * _RANGE_HEADROOM

            trained_checkpoints: list[MarkovCheckpoint] = []

            for window in sorted(windows):
                done += 1

                # Cumulative prefix: first N samples from the start of the
                # fetch window, not the most recent N.
                n_samples   = min(len(full_data), window // step)
                window_data = full_data[:n_samples]   # cumulative from t=0

                if len(window_data) < 2:
                    logger.warning(
                        "  [%d/%d] %s/%s window=%ds — too few samples (%d), skip.",
                        done, total, service, metric_name,
                        window, len(window_data),
                    )
                    continue

                dest = checkpoint_path(
                    service, metric_name, window, checkpoint_dir
                )

                if dry_run:
                    logger.info(
                        "  [%d/%d] DRY RUN %s/%s window=%ds "
                        "(%d samples, cumulative) -> %s",
                        done, total, service, metric_name,
                        window, len(window_data), dest,
                    )
                    continue

                logger.info(
                    "  [%d/%d] Training %s/%s window=%ds (%d samples, cumulative) ...",
                    done, total, service, metric_name,
                    window, len(window_data),
                )

                cp = train_checkpoint(
                    baseline_data  = window_data,
                    metric_min     = metric_min,
                    metric_max     = metric_max,
                    num_bins       = num_bins,
                    window_seconds = window,
                    service        = service,
                    metric_name    = metric_name,
                )
                save_checkpoint(cp, dest)
                trained_checkpoints.append(cp)
                logger.info("    Saved -> %s", dest)

            if trained_checkpoints and not dry_run:
                write_manifest(
                    service, metric_name, trained_checkpoints,
                    checkpoint_dir,
                )

    logger.info("Done. %d combinations processed.", total)


def train_window_checkpoints(
    config: dict,
    prometheus_url: str,
    step: int,
    checkpoint_dir: Path,
    services: tuple[str, ...],
    dry_run: bool,
    baseline_start: float,
    baseline_end: float,
    window_seconds: int,
    windows: tuple[int, ...],
) -> None:
    """Train and save one checkpoint window for all (service, metric)."""
    num_bins = int(config.get("num_bins", 100))
    client = PrometheusMetricsClient(prometheus_url=prometheus_url)
    total = len(services) * len(METRIC_QUERIES)
    done = 0
    elapsed = int(baseline_end - baseline_start)

    logger.info(
        "Training window=%ds using baseline [%.0f, %.0f] (elapsed=%ds).",
        window_seconds, baseline_start, baseline_end, elapsed,
    )

    for service in services:
        for metric_name in METRIC_QUERIES:
            done += 1
            full_data = query_service_metric(
                client=client,
                metric_name=metric_name,
                service=service,
                start=baseline_start,
                end=baseline_end,
                step=step,
            )

            if full_data is None or len(full_data) < 2:
                logger.warning(
                    "  [%d/%d] %s/%s window=%ds — no data, skip.",
                    done, total, service, metric_name, window_seconds,
                )
                continue

            data_min = float(np.min(full_data))
            data_max = float(np.max(full_data))
            data_range = data_max - data_min
            if data_range == 0:
                data_range = max(abs(data_min) * 0.1, 1e-6)

            metric_min = data_min - data_range * _RANGE_HEADROOM
            metric_max = data_max + data_range * _RANGE_HEADROOM

            dest = checkpoint_path(service, metric_name, window_seconds, checkpoint_dir)
            if dry_run:
                logger.info(
                    "  [%d/%d] DRY RUN %s/%s window=%ds (%d samples) -> %s",
                    done, total, service, metric_name, window_seconds, len(full_data), dest,
                )
                continue

            cp = train_checkpoint(
                baseline_data=full_data,
                metric_min=metric_min,
                metric_max=metric_max,
                num_bins=num_bins,
                window_seconds=window_seconds,
                service=service,
                metric_name=metric_name,
            )
            save_checkpoint(cp, dest)
            logger.info(
                "  [%d/%d] Saved %s/%s window=%ds -> %s",
                done, total, service, metric_name, window_seconds, dest,
            )
            window_checkpoints: list[MarkovCheckpoint] = []
            for w in sorted(windows):
                path = checkpoint_path(service, metric_name, w, checkpoint_dir)
                if path.exists():
                    window_checkpoints.append(load_checkpoint(path))
            if window_checkpoints:
                write_manifest(service, metric_name, window_checkpoints, checkpoint_dir)


def train_checkpoints_as_ready(
    config: dict,
    prometheus_url: str,
    step: int,
    checkpoint_dir: Path,
    services: tuple[str, ...],
    dry_run: bool,
    windows: tuple[int, ...],
    frontend_url: str,
    rps: float,
    warmup_seconds: int,
) -> None:
    """Start loadgen and train each window immediately when its data is ready."""
    max_window = max(windows)
    logger.info(
        "Starting progressive training under loadgen (max_window=%ds).",
        max_window,
    )
    gen = WorkloadGenerator(frontend_url=frontend_url, quiet=True)
    run_seconds = warmup_seconds + max_window + _LOADGEN_TAIL_BUFFER_SECONDS
    gen.run(duration_seconds=run_seconds, base_rps=rps)

    try:
        if warmup_seconds > 0:
            _sleep_with_progress("Warmup", warmup_seconds)
        baseline_start = time.time()
        logger.info("Baseline start set at %.0f.", baseline_start)

        for window in sorted(windows):
            target_end = baseline_start + window
            remaining = max(0, int(round(target_end - time.time())))
            if remaining > 0:
                _sleep_with_progress(
                    f"Baseline capture until window={window}s", remaining
                )
            baseline_end = time.time()
            train_window_checkpoints(
                config=config,
                prometheus_url=prometheus_url,
                step=step,
                checkpoint_dir=checkpoint_dir,
                services=services,
                dry_run=dry_run,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                window_seconds=window,
                windows=windows,
            )
    finally:
        gen.stop()
        logger.info("Load generator stopped.")


def _capture_baseline_window(
    largest_window: int,
    frontend_url: str,
    rps: float,
    warmup_seconds: int,
    use_loadgen: bool,
) -> tuple[float, float]:
    """Return (baseline_start, baseline_end) for model training."""
    if not use_loadgen:
        baseline_end = time.time()
        baseline_start = baseline_end - largest_window
        logger.info(
            "Loadgen disabled. Using existing traffic window [%.0f, %.0f].",
            baseline_start, baseline_end,
        )
        return baseline_start, baseline_end

    logger.info(
        "Starting load generator for baseline capture "
        "(frontend=%s, rps=%.2f, warmup=%ds, capture=%ds).",
        frontend_url, rps, warmup_seconds, largest_window,
    )
    gen = WorkloadGenerator(frontend_url=frontend_url, quiet=True)
    run_seconds = warmup_seconds + largest_window + _LOADGEN_TAIL_BUFFER_SECONDS
    gen.run(duration_seconds=run_seconds, base_rps=rps)

    try:
        if warmup_seconds > 0:
            _sleep_with_progress("Warmup", warmup_seconds)
        baseline_start = time.time()
        logger.info("Baseline capture started at %.0f.", baseline_start)
        _sleep_with_progress("Baseline capture", largest_window)
        baseline_end = time.time()
        logger.info("Baseline capture ended at %.0f.", baseline_end)
    finally:
        gen.stop()
        logger.info("Load generator stopped.")

    return baseline_start, baseline_end


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train cumulative Markov checkpoints from Prometheus data."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoint_config.json"),
        help="Path to checkpoint_config.json (default: checkpoint_config.json)",
    )
    p.add_argument(
        "--prometheus-url",
        default=None,
        help="Override Prometheus URL from config",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Override scrape step in seconds from config",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint root directory from config",
    )
    p.add_argument(
        "--services",
        default=",".join(SERVICES),
        help="Comma-separated list of services (default: all)",
    )
    p.add_argument(
        "--frontend-url",
        default=_DEFAULT_FRONTEND_URL,
        help=f"Frontend URL for loadgen (default: {_DEFAULT_FRONTEND_URL})",
    )
    p.add_argument(
        "--rps",
        type=float,
        default=5.0,
        help="Base RPS for sine workload generation (default: 5.0)",
    )
    p.add_argument(
        "--warmup-seconds",
        type=int,
        default=30,
        help="Warmup duration before baseline capture starts (default: 30)",
    )
    p.add_argument(
        "--no-loadgen",
        action="store_true",
        help="Skip loadgen and train from current ambient traffic window",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be trained without saving",
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    config = load_config(args.config)

    prometheus_url = args.prometheus_url or config.get(
        "prometheus_url", "http://localhost:9090"
    )
    step           = args.step or int(config.get("step_seconds", 1))
    checkpoint_dir = args.checkpoint_dir or Path(
        config.get("checkpoint_dir", "checkpoints/markov")
    )
    services       = tuple(
        s.strip() for s in args.services.split(",") if s.strip()
    )
    windows        = get_window_seconds(args.config)
    largest_window = max(windows)

    logger.info("Config file    : %s", args.config)
    logger.info("Prometheus URL : %s", prometheus_url)
    logger.info("Step           : %ds", step)
    logger.info("Checkpoint dir : %s", checkpoint_dir)
    logger.info("Num bins       : %d", config.get("num_bins", 100))
    logger.info("Services       : %s", services)
    logger.info("Windows (s)    : %s", windows)
    logger.info("Cumulative     : yes (each window is a prefix from t=0)")
    logger.info("Loadgen        : %s", not args.no_loadgen)
    logger.info("Frontend URL   : %s", args.frontend_url)
    logger.info("Base RPS       : %.2f", args.rps)
    logger.info("Warmup (s)     : %d", args.warmup_seconds)
    logger.info("Dry run        : %s", args.dry_run)
    logger.info("")

    if args.no_loadgen:
        baseline_start, baseline_end = _capture_baseline_window(
            largest_window=largest_window,
            frontend_url=args.frontend_url,
            rps=args.rps,
            warmup_seconds=max(0, args.warmup_seconds),
            use_loadgen=False,
        )
        logger.info(
            "Training baseline window: [%.0f, %.0f] (%ds)",
            baseline_start, baseline_end, int(baseline_end - baseline_start),
        )
        logger.info("")
        train_all_checkpoints(
            config         = config,
            prometheus_url = prometheus_url,
            step           = step,
            checkpoint_dir = checkpoint_dir,
            services       = services,
            dry_run        = args.dry_run,
            config_path    = args.config,
            baseline_start = baseline_start,
            baseline_end   = baseline_end,
        )
        return

    train_checkpoints_as_ready(
        config=config,
        prometheus_url=prometheus_url,
        step=step,
        checkpoint_dir=checkpoint_dir,
        services=services,
        dry_run=args.dry_run,
        windows=windows,
        frontend_url=args.frontend_url,
        rps=args.rps,
        warmup_seconds=max(0, args.warmup_seconds),
    )


if __name__ == "__main__":
    main()