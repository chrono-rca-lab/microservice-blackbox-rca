"""Tangent-based rollback for onset time refinement."""

import numpy as np


def rollback_onset(
    series: np.ndarray,
    change_point_idx: int,
    slope_threshold: float = 0.1,
    relative: bool = True,
    max_rollback: int | None = None,
) -> int:
    """Roll back from a change point to find the true anomaly onset.

    Starting from *change_point_idx*, compute the local slope via central
    differences.  Walk backward while consecutive slopes are similar
    (difference below *slope_threshold*).  Stop when slopes diverge.

    When *relative* is True the slopes are normalised by the series range so
    that the threshold is scale-invariant across metrics with very different
    magnitudes.

    *max_rollback* limits how far back we look (default: half of
    *change_point_idx*).  This prevents rollback from collapsing to index 0
    on noisy series where slopes are uniformly small.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    idx = change_point_idx

    if n < 3 or idx <= 1 or idx >= n - 1:
        return max(0, idx)

    scale = float(np.ptp(series)) if relative else 1.0
    if scale == 0:
        return idx

    if max_rollback is None:
        max_rollback = max(1, idx // 2)
    earliest = max(1, idx - max_rollback)

    def _slope(i: int) -> float:
        if i <= 0:
            return (series[1] - series[0]) / scale
        if i >= n - 1:
            return (series[-1] - series[-2]) / scale
        return (series[i + 1] - series[i - 1]) / (2.0 * scale)

    prev_slope = _slope(idx)
    i = idx - 1
    while i >= earliest:
        cur_slope = _slope(i)
        if abs(prev_slope - cur_slope) >= slope_threshold:
            break
        prev_slope = cur_slope
        i -= 1

    return max(earliest, i)
