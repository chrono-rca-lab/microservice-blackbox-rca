"""Tangent-based rollback for anomaly onset time refinement.

Design follows the FChain Section II-B specification exactly:

Algorithm
---------
Starting from a CUSUM-detected abnormal change point, walk backward
through the Layer-1 change point list (ALL detected change points,
including those filtered out by Layers 2 and 3).

At each step compare the tangent at the current change point against
the tangent at the immediately preceding change point.

  |tangent(current) - tangent(preceding)| < 0.1  ->  same slope
      -> roll back: onset = preceding_cp
      -> continue from preceding_cp

  |tangent(current) - tangent(preceding)| >= 0.1 ->  slope changed
      -> stop: current_cp is the true onset

Tangent is computed via central difference:
  tangent(t) = (series[t+1] - series[t-1]) / 2
with forward / backward difference at the boundaries.

Multi-metric aggregation
------------------------
Each component exposes 7 metrics.  Rollback is run independently per
metric.  The component's onset time is the minimum across all metrics
that show at least one abnormal change point.

Public API
----------
  compute_tangent(series, t)                     -> float
  rollback_onset(series, abnormal_cp,
                 all_change_points,
                 tangent_threshold)              -> int
  compute_component_onset(series_per_metric,
                           abnormal_cps_per_metric,
                           all_cps_per_metric,
                           tangent_threshold)    -> int | None
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TANGENT_THRESHOLD: float = 0.1   # FChain paper Section II-B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_tangent(series: np.ndarray, t: int) -> float:
    """Estimate the instantaneous slope of *series* at index *t*.

    Uses a central difference for interior points and one-sided differences
    at the boundaries.

    Formula
    -------
    Interior (1 <= t <= n-2):
        tangent = (series[t+1] - series[t-1]) / 2

    Left boundary (t == 0):
        tangent = series[1] - series[0]

    Right boundary (t == n-1):
        tangent = series[-1] - series[-2]

    Parameters
    ----------
    series:
        1-D array of raw metric values (look-back window).
    t:
        Index at which to compute the tangent.
        Clamped to [0, n-1] if out of range.

    Returns
    -------
    float
        Local slope in units of metric-value per sample.
        Operates on raw values — no normalisation applied.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)

    if n == 0:
        return 0.0
    if n == 1:
        return 0.0

    # Clamp to valid range
    t = max(0, min(t, n - 1))

    if t == 0:
        # Forward difference
        return float(series[1] - series[0])
    if t == n - 1:
        # Backward difference
        return float(series[-1] - series[-2])

    # Central difference (standard for interior points)
    return float((series[t + 1] - series[t - 1]) / 2.0)


def rollback_onset(
    series: np.ndarray,
    abnormal_cp: int,
    all_change_points: list[int],
    tangent_threshold: float = _DEFAULT_TANGENT_THRESHOLD,
) -> int:
    """Walk backward from *abnormal_cp* to find the true anomaly onset.

    Rollback steps through the Layer-1 change point list (ALL detected
    change points, not just the abnormal ones), moving to the immediately
    preceding change point at each iteration.

    At each step, compare the tangent at the current change point against
    the tangent at the preceding change point.  If they are similar
    (difference < *tangent_threshold*), the fault was already manifesting
    at the preceding change point, so roll back.  Stop when tangents
    diverge or no more preceding change points exist.

    Parameters
    ----------
    series:
        1-D array of raw metric values for the look-back window.
    abnormal_cp:
        Index of the abnormal change point (output of Layer 3).
        This is the starting point for rollback.
    all_change_points:
        Complete list of ALL change points detected by CUSUM + Bootstrap
        (Layer 1 output), including those filtered out by Layers 2 and 3.
        Rollback walks through this list, not through raw sample indices.
    tangent_threshold:
        Maximum tangent difference to consider two change points as being
        on the same slope.  FChain paper default: 0.1.
        Comparison is strict less-than: similar if diff < threshold.

    Returns
    -------
    int
        Index of the true anomaly onset.  Always a member of
        *all_change_points* (or *abnormal_cp* itself if no rollback
        is possible).  Always <= *abnormal_cp*.
    """
    series = np.asarray(series, dtype=float).ravel()

    # Guard: if abnormal_cp is not in all_change_points, add it so
    # the rollback has a valid starting point.
    all_cps_set = set(all_change_points)
    if abnormal_cp not in all_cps_set:
        all_cps_set.add(abnormal_cp)

    # Safety: max iterations = total number of change points
    max_iterations = len(all_cps_set)

    current_cp = abnormal_cp
    onset      = abnormal_cp

    for _ in range(max_iterations):

        # Find the change point immediately preceding current_cp
        preceding_candidates = [cp for cp in all_cps_set if cp < current_cp]

        if not preceding_candidates:
            # current_cp is already the earliest — stop here
            break

        preceding_cp = max(preceding_candidates)   # closest earlier cp

        # Compute tangents at both points
        tangent_current   = compute_tangent(series, current_cp)
        tangent_preceding = compute_tangent(series, preceding_cp)

        # Decision: are the two points on the same slope?
        if abs(tangent_current - tangent_preceding) < tangent_threshold:
            # Same slope — fault was already manifesting at preceding_cp
            onset      = preceding_cp
            current_cp = preceding_cp
            # Continue rolling back
        else:
            # Slope changed — current_cp is the true onset
            break

    return onset


def compute_component_onset(
    series_per_metric: dict[str, np.ndarray],
    abnormal_cps_per_metric: dict[str, list[int]],
    all_cps_per_metric: dict[str, list[int]],
    tangent_threshold: float = _DEFAULT_TANGENT_THRESHOLD,
) -> int | None:
    """Compute a component's true anomaly onset across all its metrics.

    Runs rollback independently for every metric that has at least one
    abnormal change point.  Returns the minimum onset time found across
    all metrics — the earliest detectable sign of abnormal behavior on
    this component.

    Parameters
    ----------
    series_per_metric:
        Dict mapping metric name to its raw look-back window time series.
        Expected keys: 'cpu', 'memory', 'net_in', 'net_out',
                       'disk_read', 'disk_write'  (or any subset).
    abnormal_cps_per_metric:
        Dict mapping metric name to the list of abnormal change point
        indices for that metric (output of Layer 3).
        Metrics with no abnormal change points should map to [].
    all_cps_per_metric:
        Dict mapping metric name to ALL change points detected by CUSUM
        for that metric (Layer 1 output), including those filtered out
        by Layers 2 and 3.
    tangent_threshold:
        Passed through to rollback_onset for every metric.

    Returns
    -------
    int | None
        The minimum onset time across all metrics, or None if no metric
        on this component shows any abnormal change point.
    """
    per_metric_onsets: list[int] = []

    for metric_name, series in series_per_metric.items():

        abnormal_cps = abnormal_cps_per_metric.get(metric_name, [])
        all_cps      = all_cps_per_metric.get(metric_name, [])

        if not abnormal_cps:
            # This metric shows no abnormal behavior — skip it
            continue

        # Run rollback independently for each abnormal change point
        # on this metric, then take the minimum onset for this metric.
        metric_onsets: list[int] = []
        for acp in abnormal_cps:
            onset = rollback_onset(
                series,
                abnormal_cp=acp,
                all_change_points=all_cps,
                tangent_threshold=tangent_threshold,
            )
            metric_onsets.append(onset)

        # Earliest onset across all abnormal cps on this metric
        per_metric_onsets.append(min(metric_onsets))

    if not per_metric_onsets:
        # No metric on this component showed abnormal behavior
        return None

    # Final component onset: minimum across all metrics
    return min(per_metric_onsets)