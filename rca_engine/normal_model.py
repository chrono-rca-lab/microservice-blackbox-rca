"""Normal fluctuation model for per-service metrics.

Implements the PRESS-style discrete-time Markov chain prediction model
from the FChain paper (reference [12]).  For each metric time series the
model learns — from baseline data — which bin-to-bin transitions are normal.
At inference time it predicts the expected next value and returns the
absolute deviation as an anomaly signal.

Key design choices
------------------
* Equal-width binning over the observed [min, max] range.
* Laplace smoothing (epsilon) so every transition has a non-zero probability,
  preventing division-by-zero on unseen transitions during inference.
* Out-of-range values are clipped to the nearest boundary bin.
* The model is stateless after fit() — predict() and prediction_error() are
  pure functions of the stored transition matrix and bin metadata.
"""

import numpy as np


class NormalFluctuationModel:
    """Discrete-time Markov chain model of normal metric fluctuation.

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins to discretize the value space.
    history_window : int
        Maximum number of consecutive (value[t], value[t+1]) pairs used
        when building the transition matrix.  Only the most recent
        *history_window* samples from the baseline series are considered,
        so that a long baseline does not let very old behaviour dominate.
    """

    def __init__(self, n_bins: int = 100, history_window: int = 1000) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self.history_window = history_window

        # Set by fit()
        self._fitted: bool = False
        self._bin_edges: np.ndarray | None = None   # shape (n_bins + 1,)
        self._bin_centers: np.ndarray | None = None  # shape (n_bins,)
        self._T: np.ndarray | None = None            # shape (n_bins, n_bins)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, values: np.ndarray) -> "NormalFluctuationModel":
        """Learn the transition matrix from baseline observations.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Ordered metric samples from the baseline period.
            Must contain at least 2 values.

        Returns
        -------
        self  (for method chaining)
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("values must be a 1-D array")
        if len(values) < 2:
            raise ValueError("Need at least 2 baseline samples to build transition matrix")

        # Trim to history_window most recent samples
        if len(values) > self.history_window:
            values = values[-self.history_window :]

        v_min, v_max = float(values.min()), float(values.max())

        # If all values are identical the range is zero — widen by a small
        # epsilon so binning is well-defined.
        if v_max == v_min:
            v_min -= 1e-6
            v_max += 1e-6

        self._bin_edges = np.linspace(v_min, v_max, self.n_bins + 1)
        self._bin_centers = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])

        # Discretize the entire series
        bins = self._digitize(values)

        # Count transitions: T_counts[i, j] = number of times bin i → bin j
        T_counts = np.zeros((self.n_bins, self.n_bins), dtype=float)
        for t in range(len(bins) - 1):
            T_counts[bins[t], bins[t + 1]] += 1.0

        # Laplace smoothing: add epsilon to every cell so unseen transitions
        # still produce a valid (non-zero) probability.
        epsilon = 1e-6
        T_counts += epsilon

        # Row-normalise to get probabilities
        row_sums = T_counts.sum(axis=1, keepdims=True)
        self._T = T_counts / row_sums

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, current_value: float) -> float:
        """Predict the expected next value given the current observation.

        Uses the learned transition probabilities:
            E[next] = sum_j( bin_center[j] * T[current_bin, j] )

        Parameters
        ----------
        current_value : float
            The most recently observed metric value.

        Returns
        -------
        float
            Expected next value according to the Markov model.
        """
        self._check_fitted()
        b = self._digitize_scalar(current_value)
        return float(self._T[b] @ self._bin_centers)

    def prediction_error(self, current_value: float, next_value: float) -> float:
        """Absolute error between the model's prediction and the actual next value.

        Parameters
        ----------
        current_value : float
            Observation at time t.
        next_value : float
            Observation at time t+1.

        Returns
        -------
        float
            |next_value - predict(current_value)|
        """
        return abs(next_value - self.predict(current_value))

    def batch_prediction_errors(self, values: np.ndarray) -> np.ndarray:
        """Compute prediction errors for every consecutive pair in *values*.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Time-ordered metric observations (typically from the fault window).

        Returns
        -------
        np.ndarray, shape (N-1,)
            Prediction error at each time step t = 0 … N-2.
            errors[t] = |values[t+1] - predict(values[t])|
        """
        self._check_fitted()
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            return np.array([], dtype=float)

        bins = self._digitize(values)
        # Vectorised: predicted[t] = T[bins[t]] @ bin_centers
        predicted = self._T[bins[:-1]] @ self._bin_centers   # shape (N-1,)
        return np.abs(values[1:] - predicted)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _digitize(self, values: np.ndarray) -> np.ndarray:
        """Map an array of floats to bin indices in [0, n_bins-1]."""
        # np.digitize returns 1-based indices; subtract 1 for 0-based.
        # Clip to handle values exactly at v_max (digitize puts them in
        # the overflow bucket n_bins).
        idx = np.digitize(values, self._bin_edges[1:])  # edges without left boundary
        return np.clip(idx, 0, self.n_bins - 1)

    def _digitize_scalar(self, value: float) -> int:
        """Map a single float to its bin index in [0, n_bins-1]."""
        return int(np.clip(np.digitize([value], self._bin_edges[1:])[0], 0, self.n_bins - 1))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    PASS = "PASS"
    FAIL = "FAIL"

    def _result(condition: bool) -> str:
        return PASS if condition else FAIL

    print("=" * 55)
    print("  NormalFluctuationModel — unit tests")
    print("=" * 55)

    # ── Test 1: sine wave has low mean prediction error ──────────────
    t = np.linspace(0, 4 * math.pi, 500)
    sine = np.sin(t)

    model = NormalFluctuationModel(n_bins=50)
    model.fit(sine[:250])                        # first half = baseline
    errors = model.batch_prediction_errors(sine[250:])   # second half = test
    mean_err = errors.mean()
    passed = mean_err < 0.15
    print(f"\n[1] Sine wave — low mean prediction error")
    print(f"    mean error = {mean_err:.4f}  (want < 0.15)  →  {_result(passed)}")

    # ── Test 2: sine + step jump → high error at jump, low elsewhere ─
    sine_with_jump = sine.copy()
    jump_idx = 375                               # midpoint of test window
    sine_with_jump[jump_idx:] += 5.0             # sudden +5 step

    model2 = NormalFluctuationModel(n_bins=50)
    model2.fit(sine_with_jump[:250])
    errors2 = model2.batch_prediction_errors(sine_with_jump[250:])

    # The error at the jump step should be substantially larger than
    # the median error of the non-jump region.
    local_jump = jump_idx - 250                  # index within the test window
    error_at_jump = errors2[local_jump]
    median_no_jump = float(np.median(np.concatenate([errors2[:local_jump],
                                                      errors2[local_jump + 5:]])))
    passed2 = error_at_jump > 5 * median_no_jump
    print(f"\n[2] Sine + step jump — high error at jump, low elsewhere")
    print(f"    error at jump   = {error_at_jump:.4f}")
    print(f"    median elsewhere= {median_no_jump:.4f}")
    print(f"    ratio           = {error_at_jump / (median_no_jump + 1e-9):.1f}×  (want > 5×)  →  {_result(passed2)}")

    # ── Test 3: constant series → near-zero prediction error ─────────
    constant = np.ones(300) * 42.0

    model3 = NormalFluctuationModel(n_bins=50)
    model3.fit(constant[:150])
    errors3 = model3.batch_prediction_errors(constant[150:])
    mean_err3 = errors3.mean()
    passed3 = mean_err3 < 1e-3
    print(f"\n[3] Constant series — near-zero prediction error")
    print(f"    mean error = {mean_err3:.6f}  (want < 0.001)  →  {_result(passed3)}")

    # ── Test 4: out-of-range values don't crash (edge case) ──────────
    model4 = NormalFluctuationModel(n_bins=20)
    model4.fit(np.linspace(0, 1, 100))
    try:
        err = model4.prediction_error(-999.0, 999.0)
        passed4 = True
    except Exception as e:
        passed4 = False
        print(f"    exception: {e}")
    print(f"\n[4] Out-of-range values — no crash")
    print(f"    →  {_result(passed4)}")

    # ── Summary ──────────────────────────────────────────────────────
    all_passed = passed and passed2 and passed3 and passed4
    print(f"\n{'='*55}")
    print(f"  {'All tests passed' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*55}\n")
