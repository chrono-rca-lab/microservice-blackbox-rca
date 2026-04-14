"""Online normal-behaviour model for per-service metrics (Markov chain prediction)."""

import numpy as np


class NormalModel:
    """Discrete Markov chain model for metric time series.

    Discretises metric values into bins and builds a transition matrix
    ``T[i][j] = P(bin_j | bin_i)``.  Prediction at time *t+1* is the centre
    of ``argmax_j T[bin(x_t)][j]``.
    """

    def __init__(self, num_bins: int = 100, window_size: int = 300) -> None:
        self._max_bins = num_bins
        self._window_size = window_size  # retained for API compat
        self._bin_edges: np.ndarray | None = None
        self._transition: np.ndarray | None = None
        self._n_bins: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, baseline_data: np.ndarray) -> "NormalModel":
        """Fit the model on baseline data (batch mode)."""
        data = np.asarray(baseline_data, dtype=float).ravel()
        if len(data) < 2:
            self._n_bins = 0
            return self

        # Adaptive bin count — avoids degenerate matrices with sparse data
        self._n_bins = min(self._max_bins, max(3, len(data) // 2))

        lo, hi = float(data.min()), float(data.max())
        span = hi - lo
        if span == 0:
            span = max(abs(lo) * 1e-6, 1e-10)
        eps = span * 0.01
        self._bin_edges = np.linspace(lo - eps, hi + eps, self._n_bins + 1)

        # Build transition matrix
        bins = self._discretize_array(data)
        T = np.zeros((self._n_bins, self._n_bins), dtype=float)
        for a, b in zip(bins[:-1], bins[1:]):
            T[a, b] += 1.0

        # Row-normalise; unseen rows get uniform distribution
        row_sums = T.sum(axis=1, keepdims=True)
        mask = row_sums.ravel() == 0
        T[mask] = 1.0
        row_sums[mask] = self._n_bins
        self._transition = T / row_sums
        return self

    def prediction_errors(self, series: np.ndarray) -> np.ndarray:
        """Prediction errors for *series*.

        Returns an array of the same length as *series*.  The first element is
        0 (no predecessor to predict from).
        """
        series = np.asarray(series, dtype=float).ravel()
        if self._transition is None or self._n_bins == 0 or len(series) < 2:
            return np.zeros(len(series))

        bins = self._discretize_array(series)
        errors = np.zeros(len(series))
        for t in range(len(series) - 1):
            predicted_bin = int(np.argmax(self._transition[bins[t]]))
            predicted_val = self._bin_center(predicted_bin)
            errors[t + 1] = abs(series[t + 1] - predicted_val)
        return errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discretize(self, value: float) -> int:
        if self._bin_edges is None:
            return 0
        idx = int(np.searchsorted(self._bin_edges, value, side="right")) - 1
        return max(0, min(idx, self._n_bins - 1))

    def _discretize_array(self, arr: np.ndarray) -> np.ndarray:
        if self._bin_edges is None:
            return np.zeros(len(arr), dtype=int)
        idxs = np.searchsorted(self._bin_edges, arr, side="right").astype(int) - 1
        return np.clip(idxs, 0, self._n_bins - 1)

    def _bin_center(self, bin_idx: int) -> float:
        if self._bin_edges is None:
            return 0.0
        return float((self._bin_edges[bin_idx] + self._bin_edges[bin_idx + 1]) / 2)
