"""FFT-based burst threshold filter for abnormal change point selection.

Design follows the FChain Section II-B specification exactly:

Algorithm per change point
--------------------------
1.  Extract a local window of 2*Q+1 samples centred on the change point
    (truncated at series boundaries).
2.  Apply real-valued FFT (np.fft.rfft) to the local window.
3.  Zero out the bottom 10% of frequency coefficients by index
    (low-frequency / slow-trend components).  Keep the top 90% by index
    (high-frequency / burst components).  The DC component (index 0) is
    always zeroed because it falls in the bottom 10%.
4.  Apply inverse FFT (np.fft.irfft) to reconstruct the burst-only signal.
5.  Threshold = 90th percentile of |burst_signal|.
    Floor at 1e-10 to avoid a trivially zero threshold on flat signals.
6.  Decision: prediction_error[t] > threshold  ->  ABNORMAL  (keep)
              prediction_error[t] <= threshold ->  NORMAL    (discard)

Fallback (window < 4 samples)
------------------------------
FFT is not meaningful with fewer than 4 samples.  In that case the
threshold is set to the global 90th percentile of all prediction errors
supplied for this call.  If no errors are available the change point is
conservatively treated as abnormal.

Parameters
----------
Q : int = 20
    Half-window size in samples (seconds at 1 Hz).  FChain paper default.
high_freq_fraction : float = 0.90
    Fraction of FFT coefficients (by index, high end) to keep as the
    burst signal.  FChain paper: top 90%.
burst_percentile : float = 90.0
    Percentile of |burst_signal| used as the threshold.
    FChain paper: 90th percentile.
threshold_floor : float = 1e-10
    Minimum threshold to avoid false alarms on perfectly flat windows.
"""

from __future__ import annotations

import math

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_abnormal_change_points(
    series: np.ndarray,
    change_point_errors: dict[int, float],
    *,
    Q: int = 20,
    high_freq_fraction: float = 0.90,
    burst_percentile: float = 90.0,
    threshold_floor: float = 1e-10,
) -> list[int]:
    """Return the subset of change points whose prediction error exceeds the
    local burst-based threshold.

    Parameters
    ----------
    series:
        1-D array of raw metric values for the look-back window.
    change_point_errors:
        Dict mapping each candidate change-point index to its prediction
        error (output of Layer 2 / NormalModel.prediction_errors_for).
    Q:
        Half-window size in samples.  The local window around change point t
        spans [t-Q, t+Q] (inclusive), truncated at series boundaries.
    high_freq_fraction:
        Fraction of FFT coefficients, counted from the high-frequency end,
        to retain when reconstructing the burst signal.  The complementary
        low-frequency fraction (1 - high_freq_fraction) is zeroed out.
    burst_percentile:
        Percentile of |burst_signal| used as the adaptive threshold.
    threshold_floor:
        Absolute minimum threshold value; prevents a zero threshold on
        constant-valued windows from flagging every change point.

    Returns
    -------
    list[int]
        Indices of change points classified as ABNORMAL, in the same order
        they appear in change_point_errors.  Passed to the rollback step.
    """
    series = np.asarray(series, dtype=float).ravel()
    n_series = len(series)

    if n_series == 0 or not change_point_errors:
        return []

    # Pre-compute the global fallback threshold once (used when the local
    # window is too short for a meaningful FFT).
    all_errors = list(change_point_errors.values())
    global_fallback_threshold = max(
        float(np.percentile(all_errors, burst_percentile)),
        threshold_floor,
    )

    abnormal: list[int] = []

    for t, prediction_error in change_point_errors.items():

        # Validate index
        if t < 0 or t >= n_series:
            continue

        # ------------------------------------------------------------------
        # Step 1: Extract local window centred on t
        # ------------------------------------------------------------------
        win_start    = max(0, t - Q)
        win_end      = min(n_series, t + Q + 1)    # exclusive upper bound
        local_window = series[win_start:win_end]

        # ------------------------------------------------------------------
        # Fallback: window too short for meaningful FFT
        # ------------------------------------------------------------------
        if len(local_window) < 4:
            threshold = global_fallback_threshold
            if prediction_error > threshold:
                abnormal.append(t)
            continue

        # ------------------------------------------------------------------
        # Steps 2-5: FFT -> zero low-freq -> IFFT -> percentile threshold
        # ------------------------------------------------------------------
        threshold = _burst_threshold(
            local_window,
            high_freq_fraction=high_freq_fraction,
            burst_percentile=burst_percentile,
            threshold_floor=threshold_floor,
        )

        # ------------------------------------------------------------------
        # Step 6: Decision  (strict greater-than per spec)
        # ------------------------------------------------------------------
        if prediction_error > threshold:
            abnormal.append(t)

    _log_filter_summary(len(change_point_errors), len(abnormal))
    return abnormal


def _log_filter_summary(n_candidates: int, n_abnormal: int) -> None:
    logger.debug(
        "Layer 3 filter: %d abnormal change points from %d candidates",
        n_abnormal,
        n_candidates,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _burst_threshold(
    window: np.ndarray,
    *,
    high_freq_fraction: float,
    burst_percentile: float,
    threshold_floor: float,
) -> float:
    """Compute the burst-based threshold for one local window.

    Steps
    -----
    1.  rfft(window)  ->  complex coefficients, length N//2 + 1.
    2.  Zero out the bottom (1 - high_freq_fraction) of coefficients
        by frequency index (i.e. the slow / DC components).
    3.  irfft(filtered_coeffs, n=len(window))  ->  burst signal.
    4.  Return max(percentile(|burst|, burst_percentile), threshold_floor).

    Parameters
    ----------
    window:
        Local metric values centred on the change point.
        Must have at least 4 elements (caller's responsibility).
    high_freq_fraction:
        Fraction of coefficients (from the high-frequency end) to keep.
        FChain paper: 0.90  ->  keep top 90%, zero bottom 10%.
    burst_percentile:
        Percentile of |burst_signal| to use as the threshold.
    threshold_floor:
        Absolute minimum returned value.
    """
    N = len(window)

    # Step 2: Real-valued FFT
    # rfft on a length-N real signal produces N//2 + 1 complex coefficients.
    # Index 0 = DC (mean), index N//2 = Nyquist (highest frequency).
    fft_coeffs = np.fft.rfft(window)
    num_coeffs  = len(fft_coeffs)           # = N // 2 + 1

    # Step 3: Identify and zero out the low-frequency components
    #
    # "Top 90% by frequency index" means:
    #   - Sort coefficient indices 0 ... num_coeffs-1 from lowest to highest
    #   - The bottom 10% by index are slow / DC components  ->  zero them
    #   - The top 90% by index are fast / burst components  ->  keep them
    #
    # cutoff_index is the first index to KEEP.
    # Everything strictly below cutoff_index is zeroed.
    #
    # math.ceil ensures we always zero at least 1 coefficient (the DC term).
    low_freq_count = max(1, math.ceil((1.0 - high_freq_fraction) * num_coeffs))
    cutoff_index   = low_freq_count             # first kept index

    high_freq_coeffs = fft_coeffs.copy()        # never mutate the original
    high_freq_coeffs[:cutoff_index] = 0.0       # zero DC + low-frequency band

    # Step 4: Inverse FFT -> burst-only time-domain signal
    # Always pass n=len(window) so irfft reconstructs the correct length
    # regardless of whether N is odd or even.
    burst_signal = np.fft.irfft(high_freq_coeffs, n=N)

    # Step 5: Threshold = burst_percentile-th percentile of |burst_signal|
    threshold = float(np.percentile(np.abs(burst_signal), burst_percentile))

    return max(threshold, threshold_floor)


# Alias used by tests written against an earlier internal name.
_compute_burst_threshold = _burst_threshold