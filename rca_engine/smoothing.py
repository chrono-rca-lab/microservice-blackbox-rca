"""
Time-series smoothing utilities for RCA preprocessing.

Light smoothing before CUSUM in ``fault_chain``. Mostly EMA; rolling mean
is there if you want something dumber. Don't smooth so hard you kill real
steps; keep NaNs from turning into fake plateaus; output length must match
input for the rest of the pipeline.
"""

import numpy as np
from typing import Dict


def smooth_series(
    series: np.ndarray,
    method: str = "ema",
    window: int = 5,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Smooth a time series using specified method.

    Used in RCA preprocessing to reduce noise before change-point detection.
    EMA is preferred for real-time anomaly detection as it adapts quickly to
    recent changes while maintaining some historical context.

    Args:
        series: 1D numpy array of time series values
        method: Smoothing method ("ema" or "rolling")
        window: Window size for rolling mean (ignored for EMA)
        alpha: Smoothing factor for EMA (0 < alpha <= 1)
                 Higher alpha = more weight on recent values

    Returns:
        Smoothed array with same length as input

    Raises:
        ValueError: If method is not supported or parameters are invalid

    Examples:
        >>> import numpy as np
        >>> data = np.array([1.0, 2.0, 3.0, 2.5, 4.0, np.nan, 3.5])
        >>> smoothed = smooth_series(data, method="ema", alpha=0.3)
        >>> # EMA adapts quickly to changes, good for anomaly detection
    """
    if series.ndim != 1:
        raise ValueError("Input series must be 1D numpy array")

    if len(series) == 0:
        return np.array([])

    if method not in ["ema", "rolling"]:
        raise ValueError(f"Method '{method}' not supported. Use 'ema' or 'rolling'")

    if method == "ema":
        return _exponential_moving_average(series, alpha)
    else:  # rolling
        return _rolling_mean(series, window)


def _exponential_moving_average(series: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute exponential moving average.

    Formula: s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    Handles NaN values by propagating them (maintains gaps in data).
    For RCA, this preserves the timing of missing data points.
    """
    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")

    smoothed = np.full_like(series, np.nan, dtype=float)

    # Find first non-NaN value to initialize
    valid_mask = ~np.isnan(series)
    if not np.any(valid_mask):
        return smoothed  # All NaN, return all NaN

    first_valid_idx = np.argmax(valid_mask)
    smoothed[first_valid_idx] = series[first_valid_idx]

    # Apply EMA recursively
    for i in range(first_valid_idx + 1, len(series)):
        if np.isnan(series[i]):
            smoothed[i] = np.nan  # Propagate NaN
        else:
            smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def _rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean with trailing window.

    Uses trailing window (past values) rather than centered window,
    making it suitable for real-time processing in RCA pipelines.

    Handles NaN values by using available data within each window.
    """
    if window < 1:
        raise ValueError("Window size must be >= 1")

    if len(series) < window:
        # For short series, compute cumulative mean
        smoothed = np.full_like(series, np.nan, dtype=float)
        for i in range(len(series)):
            valid_values = series[:i+1][~np.isnan(series[:i+1])]
            if len(valid_values) > 0:
                smoothed[i] = np.mean(valid_values)
            else:
                smoothed[i] = np.nan
        return smoothed

    smoothed = np.full_like(series, np.nan, dtype=float)

    # Manual implementation to avoid pandas dependency
    for i in range(len(series)):
        if i < window - 1:
            # Use mean of available values at start
            valid_values = series[:i+1][~np.isnan(series[:i+1])]
            if len(valid_values) > 0:
                smoothed[i] = np.mean(valid_values)
            else:
                smoothed[i] = np.nan
        else:
            # Use trailing window
            window_slice = series[i-window+1:i+1]
            valid_values = window_slice[~np.isnan(window_slice)]
            if len(valid_values) > 0:
                smoothed[i] = np.mean(valid_values)
            else:
                smoothed[i] = np.nan

    return smoothed


def smooth_multivariate(
    metrics: Dict[str, np.ndarray],
    method: str = "ema",
    window: int = 5,
    alpha: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Apply smoothing to multiple metrics simultaneously.

    Used in RCA preprocessing to smooth all system metrics (CPU, memory,
    network, disk) with consistent parameters before change-point detection.

    Args:
        metrics: Dictionary mapping metric names to 1D numpy arrays
        method: Smoothing method ("ema" or "rolling")
        window: Window size for rolling mean (ignored for EMA)
        alpha: Smoothing factor for EMA (0 < alpha <= 1)

    Returns:
        Dictionary with same keys, smoothed arrays as values

    Raises:
        ValueError: If metrics dict is empty or contains invalid data

    Examples:
        >>> metrics = {
        ...     "cpu_usage": np.array([50.0, 55.0, 60.0, 58.0, 65.0]),
        ...     "memory_mb": np.array([1024.0, 1050.0, 1100.0, 1080.0, 1150.0])
        ... }
        >>> smoothed = smooth_multivariate(metrics, method="ema", alpha=0.3)
        >>> # All metrics smoothed consistently for RCA pipeline
    """
    if not metrics:
        raise ValueError("Metrics dictionary cannot be empty")

    smoothed_metrics = {}
    for name, series in metrics.items():
        try:
            smoothed_metrics[name] = smooth_series(series, method, window, alpha)
        except Exception as e:
            raise ValueError(f"Failed to smooth metric '{name}': {e}")

    return smoothed_metrics