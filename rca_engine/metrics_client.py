"""Prometheus metrics client.

Queries Prometheus for per-pod system metrics and returns them as
pandas DataFrames or nested dicts ready for downstream analysis.
"""

import re
import time
from typing import Any, cast

import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# PromQL expressions keyed by a short metric name.
#
# cAdvisor housekeeping interval on this kind cluster is 10s (set via
# kubeletExtraArgs housekeeping-interval in infra/kind-cluster.yaml).
# Counters update approximately once every 10 seconds regardless of how often
# Prometheus scrapes.  Rate/deriv windows use [30s]/[45s] to guarantee each
# evaluation window spans at least 2-3 real counter updates, giving stable
# rate estimates with enough samples for CUSUM to distinguish sustained changes
# from per-service noise.
QUERIES: dict[str, str] = {
    "cpu_rate": (
        'rate(container_cpu_usage_seconds_total{namespace="boutique",container!=""}[30s])'
    ),
    # Fraction of CFS scheduling periods where the pod was CPU-throttled.
    # Rises sharply when a cpu_hog fault hits a resource-limited container,
    # even when cpu_rate stays flat at its limit.
    # Note: cAdvisor emits this without a container label — it is pod-scoped.
    "cpu_throttle_ratio": (
        'sum by (pod, namespace) (rate(container_cpu_cfs_throttled_periods_total{namespace="boutique"}[30s]))'
        ' / '
        'sum by (pod, namespace) (rate(container_cpu_cfs_periods_total{namespace="boutique"}[30s]))'
    ),
    # Rate of memory growth (bytes/sec) over a 45s window.
    # Using deriv() instead of the raw gauge makes this metric stationary:
    # normal fluctuation stays near zero while a mem_leak shows a sustained
    # positive slope.  The raw gauge drifts upward over time under any load,
    # causing CUSUM to fire false change points for every non-memory fault.
    # Window is 45s (> 2× the ~20s housekeeping interval) to guarantee at
    # least 2 samples for the linear regression deriv uses internally.
    "mem_wss": (
        'deriv(container_memory_working_set_bytes{namespace="boutique",container!=""}[45s])'
    ),
    # interface="eth0" selects the pod's primary NIC only.  Without this
    # filter, cAdvisor returns one series per virtual interface (lo, eth0,
    # erspan0, gre0, tunl0, …) and the aggregation across all interfaces
    # produces meaningless totals.
    "net_rx_rate": (
        'rate(container_network_receive_bytes_total{namespace="boutique",interface="eth0"}[30s])'
    ),
    "net_tx_rate": (
        'rate(container_network_transmit_bytes_total{namespace="boutique",interface="eth0"}[30s])'
    ),
    "fs_read_rate": (
        'rate(container_fs_reads_bytes_total{namespace="boutique",container!=""}[30s])'
    ),
    "fs_write_rate": (
        'rate(container_fs_writes_bytes_total{namespace="boutique",container!=""}[30s])'
    ),
}

# Regex to strip the two random suffixes appended to pod names, e.g.
# "cartservice-7d9b4f6c8-xkz9p"  to  "cartservice"
_POD_SUFFIX_RE = re.compile(r"-[a-f0-9]+-[a-z0-9]+$")


def _pod_to_service(pod_name: str) -> str:
    """Strip the ReplicaSet hash and pod hash from pod_name to get the service name."""
    return _POD_SUFFIX_RE.sub("", pod_name)


class PrometheusMetricsClient:
    """HTTP client for pulling range metrics from a Prometheus instance."""

    def __init__(self, prometheus_url: str = "http://localhost:9090") -> None:
        self.prometheus_url = prometheus_url.rstrip("/")
        self._range_endpoint = f"{self.prometheus_url}/api/v1/query_range"

    # Internal helpers

    def _query_range(
        self,
        query: str,
        start: float,
        end: float,
        step: str,
    ) -> list[dict[str, Any]]:
        """Execute a PromQL range query and return the raw result list.

        Raises ``ConnectionError`` if Prometheus is unreachable.
        Raises ``RuntimeError`` if the API returns a non-success status.
        """
        params = {"query": query, "start": start, "end": end, "step": step}
        try:
            resp = requests.get(self._range_endpoint, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach Prometheus at {self.prometheus_url}: {exc}"
            ) from exc

        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {body.get('error', body)}")

        return body["data"]["result"]  # list of {metric: {...}, values: [[ts, val], ...]}

    # Public API

    def fetch_metrics(
        self,
        start_time: float,
        end_time: float,
        step: str = "1s",
    ) -> pd.DataFrame:
        """Fetch all metrics for the default namespace over [start_time, end_time].

        Args:
            start_time: POSIX timestamp (seconds) for the start of the window.
            end_time:   POSIX timestamp (seconds) for the end of the window.
            step:       Prometheus step string, e.g. ``"5s"``, ``"15s"``.

        Returns:
            DataFrame with columns: [timestamp, pod, service, metric, value].
            Rows where ``pod`` is empty (node-level series) are dropped.
        """
        rows: list[dict[str, Any]] = []

        logger.info(
            "Fetching Prometheus metrics from %s for window [%s, %s] step=%s",
            self.prometheus_url,
            start_time,
            end_time,
            step,
        )

        for metric_name, query in QUERIES.items():
            try:
                results = self._query_range(query, start_time, end_time, step)
            except (ConnectionError, RuntimeError) as exc:
                logger.warning("Skipping metric '%s': %s", metric_name, exc)
                continue

            for series in results:
                pod = series["metric"].get("pod", "")
                if not pod:
                    continue  # skip node-level / non-pod series
                service = _pod_to_service(pod)
                for ts_str, val_str in series["values"]:
                    rows.append(
                        {
                            "timestamp": float(ts_str),
                            "pod": pod,
                            "service": service,
                            "metric": metric_name,
                            "value": float(val_str),
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "pod", "service", "metric", "value"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df.sort_values(["service", "metric", "timestamp"]).reset_index(drop=True)

    def fetch_metric_matrix(
        self,
        start_time: float,
        end_time: float,
        step: str = "1s",
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return metrics as a nested dict for algorithmic processing.

        Returns:
            ``{service_name: {metric_name: np.ndarray of float values}}``

            Values are averaged across pods belonging to the same service so
            that each entry is a single 1-D array aligned to a common time axis.
        """
        df = self.fetch_metrics(start_time, end_time, step)
        if df.empty:
            return {}

        matrix: dict[str, dict[str, np.ndarray]] = {}
        for key, group in df.groupby(["service", "metric"]):
            # Pandas typing exposes group keys as Hashable, so avoid direct tuple unpacking.
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            service, metric = str(key[0]), str(key[1])
            # Average over pods — keeps the time axis consistent
            mean_by_timestamp = cast(
                pd.Series, group.groupby("timestamp")["value"].mean()
            )
            averaged = mean_by_timestamp.sort_index()
            matrix.setdefault(service, {})[metric] = averaged.to_numpy()

        self._log_matrix_summary(matrix)
        return matrix

    def _log_matrix_summary(self, matrix: dict[str, dict[str, np.ndarray]]) -> None:
        total_series = sum(len(metrics) for metrics in matrix.values())
        logger.info(
            "Built metric matrix with %d services and %d service-metric streams",
            len(matrix),
            total_series,
        )


# Demo
if __name__ == "__main__":
    END = time.time()
    START = END - 300  # last 5 minutes

    client = PrometheusMetricsClient()
    logger.info("Fetching metrics from %s …", client.prometheus_url)

    try:
        df = client.fetch_metrics(START, END)
    except ConnectionError as exc:
        logger.error("ERROR: %s", exc)
        raise SystemExit(1)

    if df.empty:
        logger.warning("No data returned — is Prometheus running and scraping the cluster?")
    else:
        logger.info(
            "Rows: %d  |  Services: %d  |  Metrics: %d",
            len(df),
            df['service'].nunique(),
            df['metric'].nunique(),
        )
        logger.info("Per-service, per-metric summary (mean value):")
        summary_mean = cast(pd.Series, df.groupby(["service", "metric"])["value"].mean())
        logger.info("\n%s", summary_mean.unstack(fill_value=0).round(4).to_string())
