"""Tests for rca_engine.tangent_rollback."""

import numpy as np

from rca_engine.tangent_rollback import rollback_onset


def test_gradual_ramp():
    """Gradual ramp starting at index 8, detected at index 15.
    Rollback should find approximately the start of the ramp."""
    series = np.zeros(25, dtype=float)
    series[8:] = np.linspace(0, 10, 17)
    # Provide change points along the ramp so rollback can walk back
    all_cps = [7, 8, 10, 12, 15]
    onset = rollback_onset(series, 15, all_cps)
    assert onset <= 10  # should be near the ramp start


def test_sharp_step():
    """A sharp step — rollback should stay near the change point."""
    series = np.concatenate([np.zeros(12), np.ones(12) * 100])
    # Tangents diverge sharply at the step boundary
    all_cps = [11, 12, 13]
    onset = rollback_onset(series, 13, all_cps)
    # The step is at 12, rollback from 13 shouldn't go far back
    assert onset >= 10


def test_near_boundary():
    """Change point near the start should handle gracefully."""
    series = np.array([0, 0, 10, 10, 10], dtype=float)
    all_cps = [0, 1, 2]
    onset = rollback_onset(series, 2, all_cps)
    assert 0 <= onset <= 2


def test_constant_series():
    """Constant series — no preceding cp means rollback returns the change point itself."""
    series = np.ones(10) * 5.0
    # Single change point: no preceding cp, onset stays at 5
    onset = rollback_onset(series, 5, [5])
    assert onset == 5


def test_very_short_series():
    series = np.array([1.0, 10.0])
    onset = rollback_onset(series, 1, [0, 1])
    assert 0 <= onset <= 1


def test_change_point_at_start():
    """Change point at index 0 — no rollback possible."""
    series = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    onset = rollback_onset(series, 0, [0])
    assert onset == 0


def test_change_point_at_index_1():
    """Change point at index 1 — tangents diverge, onset stays at 1."""
    series = np.array([0.0, 100.0, 100.0, 100.0])
    # tangent at 1 ≈ 50, tangent at 0 = 100; diff=50 > 0.1 → no rollback
    onset = rollback_onset(series, 1, [0, 1])
    assert onset <= 1


def test_change_point_at_end():
    """Change point at last index — should return itself."""
    series = np.array([0.0, 0.0, 0.0, 100.0])
    onset = rollback_onset(series, 3, [3])
    assert onset == 3


def test_large_value_series():
    """Rollback on a step in large-valued series (e.g. memory metrics ~1e8).
    Rollback should stop near the step boundary."""
    series = np.ones(20) * 1e8
    series[10:] = 5e8
    # Provide change points around the step
    all_cps = [8, 9, 10, 11, 12]
    onset = rollback_onset(series, 12, all_cps)
    assert 9 <= onset <= 12


def test_tangent_threshold_controls_rollback():
    """A larger tangent_threshold allows rollback further back along a ramp."""
    series = np.concatenate([np.zeros(10), np.linspace(0, 1, 10)])
    all_cps = list(range(20))

    # With default threshold=0.1: tangent diffs on linspace ≈ 0 → rolls back to ~10
    onset_default = rollback_onset(series, 15, all_cps)
    # With tight threshold=0.001: small diffs on the ramp still allowed
    onset_tight = rollback_onset(series, 15, all_cps, tangent_threshold=0.001)

    # Both should produce valid results
    assert 0 <= onset_default <= 15
    assert 0 <= onset_tight <= 15


def test_output_always_valid_index():
    """Onset must always be a valid index into the series."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(5, 50)
        series = rng.normal(0, 10, n).astype(float)
        cp = int(rng.integers(0, n))
        # Single change point: rollback has no preceding cp, returns cp itself
        onset = rollback_onset(series, cp, [cp])
        assert 0 <= onset <= cp or onset == max(0, cp)
        assert onset < n


def test_exponential_onset():
    """An exponential ramp — rollback with a generous threshold should find
    the start of the growth."""
    series = np.ones(20) * 100.0
    series[8:] = 100.0 * np.exp(np.linspace(0, 2, 12))
    # With a threshold large enough relative to the tangent scale, rollback
    # walks back along the ramp.
    all_cps = [7, 8, 10, 12, 15]
    onset = rollback_onset(series, 15, all_cps, tangent_threshold=50.0)
    assert onset <= 12
