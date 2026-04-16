"""Online normal-behaviour model for per-service metrics (Markov chain prediction).

Design follows the FChain / PRESS specification exactly:
  - M = 40 fixed bins (PRESS paper default)
  - Continuous online updates via update() during normal operation
  - Model is frozen during fault localisation (freeze() / unfreeze())
  - Unseen states return maximum possible prediction error
  - prediction_error_at(t) returns a single float for one change-point index
  - prediction_errors_for(change_points) returns a dict {index: error}

Typical lifecycle
-----------------
1.  fit(baseline)          — warm-up on initial clean baseline data
2.  update(value)          — called every second during normal operation
3.  freeze()               — called when SLO violation is detected
4.  prediction_errors_for(change_points, series)
                           — called once per fault localisation run
5.  unfreeze()             — called after fault is resolved
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants (from PRESS paper and FChain spec)
# ---------------------------------------------------------------------------

_DEFAULT_BINS: int = 40          # PRESS paper: M = 40 equal-width bins
_UNIFORM_FILL: float = 1.0       # used to initialise unseen rows before normalisation


class NormalModel:
    """Discrete-time Markov chain model for a single metric stream.

    Parameters
    ----------
    num_bins:
        Number of equal-width discretisation bins.  PRESS paper uses 40.
    metric_min:
        Lower bound of the expected metric range.  Used to compute bin edges
        and to derive the maximum possible prediction error for unseen states.
    metric_max:
        Upper bound of the expected metric range.

    Notes
    -----
    *Bin edges* are fixed at construction time from ``metric_min`` and
    ``metric_max``.  Values outside this range are clamped to the nearest
    valid bin so the model never raises an index error at runtime.

    The *transition count matrix* ``_counts[i, j]`` records how many times
    the metric moved from bin *i* to bin *j* during the normal-operation
    window.  The *probability matrix* ``_transition[i, j]`` is derived from
    ``_counts`` by row-normalisation on every update.  Rows that have never
    been visited are assigned a uniform distribution (maximum uncertainty)
    so that ``argmax`` always returns a valid prediction — but the resulting
    prediction error equals the full metric range, which is treated as the
    maximum possible error by Layer 3.
    """

    def __init__(
        self,
        num_bins: int = _DEFAULT_BINS,
        metric_min: float = 0.0,
        metric_max: float = 100.0,
    ) -> None:
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if metric_min >= metric_max:
            raise ValueError(
                f"metric_min ({metric_min}) must be strictly less than "
                f"metric_max ({metric_max})"
            )

        self._num_bins: int = num_bins
        self._metric_min: float = float(metric_min)
        self._metric_max: float = float(metric_max)
        self._metric_range: float = self._metric_max - self._metric_min

        # Fixed bin edges computed once — never change after construction
        self._bin_edges: np.ndarray = np.linspace(
            self._metric_min, self._metric_max, num_bins + 1
        )

        # Raw transition counts — updated on every new observation pair
        self._counts: np.ndarray = np.zeros((num_bins, num_bins), dtype=np.float64)

        # Derived probability matrix — recomputed after each update
        # Initialised to uniform (no knowledge yet)
        self._transition: np.ndarray = (
            np.ones((num_bins, num_bins), dtype=np.float64) / num_bins
        )

        # Whether model updates are currently frozen (fault localisation mode)
        self._frozen: bool = False

        # Last observed bin — needed for online single-step updates
        self._last_bin: int | None = None

        # Track whether the model has been fit on any data at all
        self._is_fit: bool = False

    # ------------------------------------------------------------------
    # Public API — model lifecycle
    # ------------------------------------------------------------------

    def fit(self, baseline_data: np.ndarray) -> "NormalModel":
        """Warm-up the model on a clean baseline window (batch mode).

        This replaces whatever was learned before.  Call once on startup
        with a representative segment of normal metric data.

        Parameters
        ----------
        baseline_data:
            1-D array of metric values observed during normal operation.
            Must contain at least 2 samples.
        """
        data = np.asarray(baseline_data, dtype=float).ravel()
        if len(data) < 2:
            # Not enough data to build any transitions — leave model as-is
            return self

        # Reset counts before re-fitting
        self._counts[:] = 0.0
        self._last_bin = None

        bins = self._discretize_array(data)

        for a, b in zip(bins[:-1], bins[1:]):
            self._counts[a, b] += 1.0

        self._recompute_transition()
        self._last_bin = int(bins[-1])
        self._is_fit = True
        return self

    def update(self, new_value: float) -> None:
        """Incorporate one new normal observation into the model (online mode).

        Call this every sampling interval (e.g. every second) while the
        system is operating normally.  Has no effect when the model is frozen.

        Parameters
        ----------
        new_value:
            The latest observed metric value.
        """
        if self._frozen:
            return

        current_bin = self._discretize(new_value)

        if self._last_bin is not None:
            # Record the transition from last_bin → current_bin
            self._counts[self._last_bin, current_bin] += 1.0
            # Incrementally recompute only the affected row for efficiency
            self._recompute_row(self._last_bin)

        self._last_bin = current_bin
        self._is_fit = True

    def freeze(self) -> None:
        """Freeze the model — stop accepting online updates.

        Call when an SLO violation is detected so that fault-induced metric
        behaviour does not corrupt the normal-behaviour model.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Resume online updates after fault resolution."""
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """True while the model is in frozen (fault localisation) mode."""
        return self._frozen

    @property
    def is_fit(self) -> bool:
        """True once the model has ingested at least one transition."""
        return self._is_fit

    # ------------------------------------------------------------------
    # Public API — prediction errors (Layer 2 output)
    # ------------------------------------------------------------------

    def prediction_error_at(self, t: int, series: np.ndarray) -> float:
        """Prediction error for a single change-point index *t*.

        This is the primary interface consumed by Layer 3.

        Parameters
        ----------
        t:
            Index within *series* of the candidate change point.
            Must satisfy ``1 <= t < len(series)`` because a predecessor
            value ``series[t-1]`` is required to make the prediction.
        series:
            The full metric time series for the look-back window.  A 1-D
            array of float values.

        Returns
        -------
        float
            ``|predicted_value - actual_value|`` at index *t*.

            If *t* == 0 (no predecessor), returns 0.0.
            If the predecessor state has never been seen, returns
            ``metric_max - metric_min`` (maximum possible error).
        """
        series = np.asarray(series, dtype=float).ravel()

        # Guard: first element has no predecessor
        if t <= 0 or t >= len(series):
            return 0.0

        if not self._is_fit:
            # Model has no knowledge — treat every change point as maximally
            # surprising so it passes through to Layer 3
            return self._metric_range

        predecessor_bin = self._discretize(series[t - 1])
        actual_value = series[t]

        # Unseen state: return maximum possible prediction error directly
        if self._is_unseen_state(predecessor_bin):
            return self._metric_range

        predicted_value = self._predict_from_bin(predecessor_bin)
        return abs(actual_value - predicted_value)

    def prediction_errors_for(
        self,
        change_points: list[int],
        series: np.ndarray,
    ) -> dict[int, float]:
        """Compute prediction errors for a list of candidate change points.

        Parameters
        ----------
        change_points:
            List of integer indices (within *series*) output by Layer 1
            (CUSUM + Bootstrap).
        series:
            The metric time series for the look-back window.

        Returns
        -------
        dict mapping each change-point index to its prediction error (float).
        This dict is passed directly into Layer 3 (burst-based threshold).
        """
        series = np.asarray(series, dtype=float).ravel()
        return {
            cp: self.prediction_error_at(cp, series)
            for cp in change_points
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discretize(self, value: float) -> int:
        """Map a single float metric value to its bin index (0 … M-1).

        Values below metric_min are clamped to bin 0.
        Values above metric_max are clamped to bin M-1.
        """
        # np.searchsorted returns the insertion point; subtract 1 for bin index
        idx = int(np.searchsorted(self._bin_edges, value, side="right")) - 1
        return max(0, min(idx, self._num_bins - 1))

    def _discretize_array(self, arr: np.ndarray) -> np.ndarray:
        """Vectorised version of _discretize for a full array."""
        idxs = np.searchsorted(self._bin_edges, arr, side="right").astype(int) - 1
        return np.clip(idxs, 0, self._num_bins - 1)

    def _bin_center(self, bin_idx: int) -> float:
        """Return the centre value of bin *bin_idx*."""
        return float(
            (self._bin_edges[bin_idx] + self._bin_edges[bin_idx + 1]) / 2.0
        )

    def _predict_from_bin(self, current_bin: int) -> float:
        """Predict the next metric value given the current bin.

        If *current_bin* has never been observed as a source state (all
        counts in that row are zero), we cannot make a meaningful prediction.
        We signal this to Layer 3 by returning a sentinel value of
        ``metric_min - metric_range``, which guarantees that the prediction
        error ``|actual - predicted|`` is at least ``metric_range`` for any
        actual value in [metric_min, metric_max].

        Returns
        -------
        float — predicted metric value (bin centre of most probable next bin),
                or the unseen-state sentinel.
        """
        row_sum = self._counts[current_bin].sum()
        if row_sum == 0:
            # Unseen state — sentinel guarantees max prediction error
            # For any actual in [metric_min, metric_max]:
            #   |actual - sentinel| >= metric_range
            return self._metric_min - self._metric_range

        predicted_bin = int(np.argmax(self._transition[current_bin]))
        return self._bin_center(predicted_bin)

    def _is_unseen_state(self, bin_idx: int) -> bool:
        """Return True if this bin has never appeared as a source transition."""
        return self._counts[bin_idx].sum() == 0

    def _recompute_transition(self) -> None:
        """Recompute the full transition probability matrix from counts."""
        row_sums = self._counts.sum(axis=1, keepdims=True)  # shape (M, 1)

        # Rows with zero counts get uniform distribution
        unseen_mask = row_sums.ravel() == 0
        T = self._counts.copy()
        T[unseen_mask] = _UNIFORM_FILL
        row_sums[unseen_mask] = self._num_bins

        self._transition = T / row_sums

    def _recompute_row(self, row: int) -> None:
        """Incrementally recompute a single row of the transition matrix.

        More efficient than recomputing the full matrix on every online
        update — only the row that just received a new count is touched.
        """
        row_sum = self._counts[row].sum()
        if row_sum == 0:
            self._transition[row] = _UNIFORM_FILL / self._num_bins
        else:
            self._transition[row] = self._counts[row] / row_sum