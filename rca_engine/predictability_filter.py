"""FFT-based predictability filtering for abnormal change point selection."""

import numpy as np


def filter_abnormal_change_points(
    series: np.ndarray,
    change_points: list[int],
    prediction_errors: np.ndarray,
    Q: int = 20,
    top_k_freq_pct: float = 0.9,
    burst_percentile: float = 90.0,
) -> list[int]:
    """Return the subset of *change_points* that are truly abnormal.

    For each candidate change point at index *t*:

    1. Extract a window of ``2*Q_eff+1`` samples centred on *t*.
    2. FFT the window; keep only the top *top_k_freq_pct* frequencies by
       magnitude.
    3. IFFT to reconstruct the "burst" signal *B*.
    4. Threshold = *burst_percentile*-th percentile of ``|B|``.
    5. If ``prediction_errors[t] > threshold`` the change point is ABNORMAL.

    When the series is too short for a meaningful FFT window the change point
    is conservatively marked abnormal.
    """
    series = np.asarray(series, dtype=float).ravel()
    prediction_errors = np.asarray(prediction_errors, dtype=float).ravel()
    if len(series) == 0 or len(change_points) == 0:
        return []

    abnormal: list[int] = []
    for cp in change_points:
        if cp < 0 or cp >= len(series):
            continue

        # Adaptive half-window
        Q_local = min(Q, cp, len(series) - 1 - cp)

        if Q_local < 3:
            # Too few samples for FFT.  Change points at the very start
            # (index 0–1) are likely baseline→fault transition noise — skip.
            # For index >= 2, fall back to a simple error-magnitude check:
            # accept the change point only if the prediction error is nonzero.
            if cp < 2:
                continue
            err = prediction_errors[cp] if cp < len(prediction_errors) else 0.0
            if err > 0:
                abnormal.append(cp)
            continue

        window = series[cp - Q_local : cp + Q_local + 1]
        threshold = _compute_burst_threshold(window, top_k_freq_pct, burst_percentile)

        err = prediction_errors[cp] if cp < len(prediction_errors) else 0.0
        if err > threshold:
            abnormal.append(cp)

    return abnormal


def _compute_burst_threshold(
    window: np.ndarray,
    top_k_freq_pct: float,
    burst_percentile: float,
) -> float:
    """Compute the burst-based abnormality threshold for a windowed signal.

    1. FFT the window.
    2. Sort frequency magnitudes descending; keep the top *top_k_freq_pct*
       fraction of frequency components.
    3. IFFT the kept components to reconstruct the dominant-frequency signal.
    4. Return the *burst_percentile*-th percentile of ``|reconstruction|``.
    """
    F = np.fft.fft(window)
    magnitudes = np.abs(F)
    n = len(F)

    n_keep = max(1, int(np.ceil(n * top_k_freq_pct)))
    keep_indices = np.argsort(magnitudes)[::-1][:n_keep]

    F_filtered = np.zeros_like(F)
    F_filtered[keep_indices] = F[keep_indices]

    B = np.fft.ifft(F_filtered).real
    threshold = float(np.percentile(np.abs(B), burst_percentile))

    # Avoid trivially zero threshold
    if threshold == 0.0:
        threshold = 1e-10
    return threshold
