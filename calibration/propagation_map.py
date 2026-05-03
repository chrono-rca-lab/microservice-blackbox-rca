"""Measured fault propagation delays per dependency edge.

Used when deciding whether two services flipped abnormal around the same time
because one fault spread to the other, or they're independent (`fault_chain.pinpoint()`
and `--propagation-map` on the eval scripts).

JSON schema (calibration/propagation_delays.json):
{
  "version": 1,
  "created_utc": "2026-04-17T...",
  "step_seconds": 1.0,
  "default_threshold_s": 2.0,
  "calibration_fault": "cpu_hog",
  "trials": 3,
  "edges": {
    "frontend->checkoutservice": {
      "observed_delays_s": [1.0, 2.0, 1.0],
      "median_delay_s": 1.0,
      "threshold_s": 1.5
    },
    ...
  }
}

Threshold = median_delay + max(1.0, median_delay * 0.5). The +1 bucket is there
because scrapes are 1s apart, so fast propagation sometimes shows up as 0–1s.

Usage:
    from calibration.propagation_map import PropagationMap

    pm = PropagationMap.load("calibration/propagation_delays.json")
    thr = pm.get_path_threshold("paymentservice", "frontend", dep_graph)
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Threshold formula
# ---------------------------------------------------------------------------

def _compute_threshold(median_delay_s: float) -> float:
    """Bump the median delay into a usable per-edge concurrency threshold.

    Same formula as saved in JSON: median + max(1.0, 0.5 * median).
    Prometheus only gives second-granularity timings, so we never add less than 1s
    on top of the median—that way real propagation isn't misread as coincidence.
    """
    return median_delay_s + max(1.0, median_delay_s * 0.5)


# ---------------------------------------------------------------------------
# PropagationMap
# ---------------------------------------------------------------------------

class PropagationMap:
    """Loaded JSON map; returns thresholds per edge or along a dependency path."""


    def __init__(
        self,
        edges: dict[str, dict[str, Any]],
        default_threshold_s: float = 2.0,
        step_seconds: float = 1.0,
        calibration_fault: str = "cpu_hog",
        trials: int = 0,
        created_utc: str = "",
    ) -> None:
        # edges keyed as "caller->callee"
        self._edges = edges
        self.default_threshold_s = default_threshold_s
        self.step_seconds = step_seconds
        self.calibration_fault = calibration_fault
        self.trials = trials
        self.created_utc = created_utc

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "PropagationMap":
        """Load a propagation map from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            edges=data.get("edges", {}),
            default_threshold_s=data.get("default_threshold_s", 2.0),
            step_seconds=data.get("step_seconds", 1.0),
            calibration_fault=data.get("calibration_fault", "cpu_hog"),
            trials=data.get("trials", 0),
            created_utc=data.get("created_utc", ""),
        )

    def save(self, path: str | Path) -> None:
        """Serialise the map to *path* as JSON."""
        data: dict[str, Any] = {
            "version": 1,
            "created_utc": self.created_utc or datetime.now(timezone.utc).isoformat(),
            "step_seconds": self.step_seconds,
            "default_threshold_s": self.default_threshold_s,
            "calibration_fault": self.calibration_fault,
            "trials": self.trials,
            "edges": self._edges,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_edge_threshold(self, caller: str, callee: str) -> float:
        """Return the threshold for a direct *caller* → *callee* edge.

        Falls back to *default_threshold_s* if the edge was not calibrated or
        if no propagation was observed (median_delay_s is None).
        """
        key = f"{caller}->{callee}"
        entry = self._edges.get(key)
        if entry is None:
            return self.default_threshold_s
        thr = entry.get("threshold_s")
        if thr is None:
            return self.default_threshold_s
        return float(thr)

    def get_path_threshold(
        self,
        src: str,
        dst: str,
        graph: dict[str, list[str]],
    ) -> float:
        """Sum thresholds along the short path between two services.

        Lazy-imports ``find_path`` to avoid circular imports. Any edge without data
        uses ``default_threshold_s``. Tries upstream and downstream directions
        (handy when effects bounce back up the stack) and keeps the smaller total
        so we don't over-credit coincidence as propagation.
        """
        from rca_engine.dependency import find_path

        # Try forward path (src depends on dst, or src calls dst)
        fwd_path = find_path(graph, src, dst)
        rev_path = find_path(graph, dst, src)

        candidates: list[float] = []
        for path in [p for p in [fwd_path, rev_path] if p is not None]:
            if len(path) < 2:
                # src == dst — not a propagation scenario
                continue
            total = 0.0
            for i in range(len(path) - 1):
                total += self.get_edge_threshold(path[i], path[i + 1])
            candidates.append(total)

        if not candidates:
            return self.default_threshold_s

        # Use the minimum (most conservative: harder to classify as propagation)
        return min(candidates)

    # ------------------------------------------------------------------
    # Building / updating
    # ------------------------------------------------------------------

    def record_observation(
        self,
        caller: str,
        callee: str,
        delay_s: float,
    ) -> None:
        """Append one observed delay to the edge and recompute the threshold."""
        key = f"{caller}->{callee}"
        entry = self._edges.setdefault(key, {"observed_delays_s": []})
        entry["observed_delays_s"].append(delay_s)
        self._recompute_edge(key)

    def _recompute_edge(self, key: str) -> None:
        """Recompute median and threshold from observed_delays_s."""
        entry = self._edges[key]
        delays = [d for d in entry["observed_delays_s"] if d is not None]
        if not delays:
            entry["median_delay_s"] = None
            entry["threshold_s"] = None
            return
        med = statistics.median(delays)
        entry["median_delay_s"] = round(med, 3)
        entry["threshold_s"] = round(_compute_threshold(med), 3)

    def finalize(self) -> None:
        """Recompute all edge thresholds (call after all observations loaded)."""
        for key in self._edges:
            self._recompute_edge(key)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def edge_keys(self) -> list[str]:
        """Return all edge keys present in the map."""
        return list(self._edges.keys())

    def __repr__(self) -> str:
        return (
            f"PropagationMap(edges={len(self._edges)}, "
            f"default={self.default_threshold_s}s, "
            f"trials={self.trials})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def empty_map(
    default_threshold_s: float = 2.0,
    step_seconds: float = 1.0,
    calibration_fault: str = "cpu_hog",
) -> PropagationMap:
    """Return a new, empty PropagationMap with sensible defaults."""
    return PropagationMap(
        edges={},
        default_threshold_s=default_threshold_s,
        step_seconds=step_seconds,
        calibration_fault=calibration_fault,
        trials=0,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
