"""Change-point detection via CUSUM."""

import numpy as np


def detect_change_points(
    series: np.ndarray,
    mu_0: float,
    sigma_0: float,
    k_factor: float = 0.5,
    h_factor: float = 5.0,
) -> list[int]:
    """Upper CUSUM -- detects *increases* in the series mean.

    ``S_t = max(0, S_{t-1} + (x_t - mu_0) - k)``

    A change point is flagged when ``S_t > h``.  After each detection the
    cumulative sum is reset to zero.
    """
    series = np.asarray(series, dtype=float).ravel()
    if len(series) == 0:
        return []

    sigma = max(sigma_0, 1e-10)
    k = k_factor * sigma
    h = h_factor * sigma

    S = 0.0
    cps: list[int] = []
    for t in range(len(series)):
        S = max(0.0, S + (series[t] - mu_0) - k)
        if S > h:
            cps.append(t)
            S = 0.0
    return cps


def detect_change_points_bilateral(
    series: np.ndarray,
    mu_0: float,
    sigma_0: float,
    k_factor: float = 0.5,
    h_factor: float = 5.0,
) -> tuple[list[int], list[str]]:
    """Bilateral CUSUM -- detects both increases and decreases.

    Returns ``(indices, directions)`` where each direction is ``"up"`` or
    ``"down"``.
    """
    series = np.asarray(series, dtype=float).ravel()
    if len(series) == 0:
        return [], []

    sigma = max(sigma_0, 1e-10)
    k = k_factor * sigma
    h = h_factor * sigma

    S_up = 0.0
    S_dn = 0.0
    cps: list[tuple[int, str]] = []
    for t in range(len(series)):
        S_up = max(0.0, S_up + (series[t] - mu_0) - k)
        S_dn = max(0.0, S_dn + (mu_0 - series[t]) - k)

        triggered_up = S_up > h
        triggered_dn = S_dn > h

        if triggered_up and triggered_dn:
            direction = "up" if S_up >= S_dn else "down"
            cps.append((t, direction))
            S_up = S_dn = 0.0
        elif triggered_up:
            cps.append((t, "up"))
            S_up = 0.0
        elif triggered_dn:
            cps.append((t, "down"))
            S_dn = 0.0

    indices = [c[0] for c in cps]
    directions = [c[1] for c in cps]
    return indices, directions
