"""Per-edge propagation delay map for the FChain RCA algorithm.

Stores empirically calibrated propagation delays between connected services and
exposes them for use in the edge-aware Layer 7 concurrency check in fault_chain.py.

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

Threshold formula:  threshold = median_delay + max(1.0, median_delay * 0.5)
The +1.0 floor compensates for 1-second metric resolution quantization — a
sub-second propagation can appear as a 0-1s gap depending on scrape alignment.

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
    """Return the per-edge concurrency threshold from a measured median delay.

    threshold = median + max(1.0, median * 0.5)

    The +1.0 floor accounts for 1-second Prometheus scrape resolution: a fault
    that propagates in 200 ms can show up as a 0-1 s detected onset gap
    depending on when the scrape fires.  Using max(1.0, ...) ensures the
    threshold is always at least 1 s above the median, which avoids classifying
    genuine propagation (onset_diff >= threshold) as concurrent.
    """
    return median_delay_s + max(1.0, median_delay_s * 0.5)


# ---------------------------------------------------------------------------
# PropagationMap
# ---------------------------------------------------------------------------

class PropagationMap:
    """Holds per-edge propagation delays and answers threshold queries.

    All public query methods fall back to *default_threshold_s* when an edge
    has no calibration data (no propagation observed or edge not calibrated).
    """

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
        """Return the summed threshold along the shortest dependency path src→dst.

        Imports ``find_path`` from ``rca_engine.dependency`` lazily to avoid
        circular imports.  If no path exists, or any edge on the path has no
        calibration data, the method returns *default_threshold_s* for that
        missing edge and sums normally.

        For back-pressure paths (dst→src in the dependency graph), the method
        tries both directions and returns the smaller threshold — the more
        conservative choice.
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
