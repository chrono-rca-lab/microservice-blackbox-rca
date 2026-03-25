"""Prometheus metrics client.

Queries Prometheus for per-pod system metrics and returns them as
pandas DataFrames or nested dicts ready for downstream analysis.
"""

import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


# PromQL expressions keyed by a short metric name.
# Rate metrics use a 30s window, which safely spans two 5s-interval scrapes.
QUERIES: dict[str, str] = {
    "cpu_rate": (
        'rate(container_cpu_usage_seconds_total{namespace="default",container!=""}[30s])'
    ),
    "mem_wss": (
        'container_memory_working_set_bytes{namespace="default",container!=""}'
    ),
    "net_rx_rate": (
        'rate(container_network_receive_bytes_total{namespace="default"}[30s])'
    ),
    "net_tx_rate": (
        'rate(container_network_transmit_bytes_total{namespace="default"}[30s])'
    ),
    "fs_read_rate": (
        'rate(container_fs_reads_bytes_total{namespace="default",container!=""}[30s])'
    ),
    "fs_write_rate": (
        'rate(container_fs_writes_bytes_total{namespace="default",container!=""}[30s])'
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
        step: str = "5s",
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

        for metric_name, query in QUERIES.items():
            try:
                results = self._query_range(query, start_time, end_time, step)
            except (ConnectionError, RuntimeError) as exc:
                print(f"[metrics_client] WARNING: skipping '{metric_name}': {exc}")
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
        step: str = "5s",
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
        for (service, metric), group in df.groupby(["service", "metric"]):
            # Average over pods — keeps the time axis consistent
            averaged = (
                group.groupby("timestamp")["value"].mean().sort_index()
            )
            matrix.setdefault(service, {})[metric] = averaged.to_numpy()

        return matrix


# Demo
if __name__ == "__main__":
    END = time.time()
    START = END - 300  # last 5 minutes

    client = PrometheusMetricsClient()
    print(f"Fetching metrics from {client.prometheus_url} …")

    try:
        df = client.fetch_metrics(START, END)
    except ConnectionError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    if df.empty:
        print("No data returned — is Prometheus running and scraping the cluster?")
    else:
        print(f"\nRows: {len(df):,}  |  Services: {df['service'].nunique()}  |  Metrics: {df['metric'].nunique()}")
        print("\nPer-service, per-metric summary (mean value):")
        print(
            df.groupby(["service", "metric"])["value"]
            .mean()
            .unstack(fill_value=0)
            .round(4)
            .to_string()
        )
