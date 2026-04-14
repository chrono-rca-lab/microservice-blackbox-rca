"""Tests for rca_engine.tangent_rollback."""

import numpy as np

from rca_engine.tangent_rollback import rollback_onset


def test_gradual_ramp():
    """Gradual ramp starting at index 8, detected at index 15.
    Rollback should find approximately the start of the ramp."""
    series = np.zeros(25, dtype=float)
    series[8:] = np.linspace(0, 10, 17)

    onset = rollback_onset(series, 15)
    assert onset <= 10  # should be near the ramp start


def test_sharp_step():
    """A sharp step — rollback should stay near the change point."""
    series = np.concatenate([np.zeros(12), np.ones(12) * 100])
    onset = rollback_onset(series, 13)
    # The step is at 12, rollback from 13 shouldn't go far back
    assert onset >= 10


def test_near_boundary():
    """Change point near the start should handle gracefully."""
    series = np.array([0, 0, 10, 10, 10], dtype=float)
    onset = rollback_onset(series, 2)
    assert 0 <= onset <= 2


def test_constant_series():
    """Constant series — rollback returns the change point itself."""
    series = np.ones(10) * 5.0
    onset = rollback_onset(series, 5)
    assert onset == 5


def test_very_short_series():
    series = np.array([1.0, 10.0])
    onset = rollback_onset(series, 1)
    assert 0 <= onset <= 1


def test_change_point_at_start():
    """Change point at index 0 — no rollback possible."""
    series = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    onset = rollback_onset(series, 0)
    assert onset == 0


def test_change_point_at_index_1():
    """Change point at index 1 — cannot go below 1 (needs central diff)."""
    series = np.array([0.0, 100.0, 100.0, 100.0])
    onset = rollback_onset(series, 1)
    assert onset <= 1


def test_change_point_at_end():
    """Change point at last index — should return itself."""
    series = np.array([0.0, 0.0, 0.0, 100.0])
    onset = rollback_onset(series, 3)
    assert onset == 3  # idx >= n-1 triggers early return


def test_relative_mode_large_values():
    """Relative mode should handle large absolute values correctly.
    Memory metrics might be ~1e8."""
    series = np.ones(20) * 1e8
    series[10:] = 5e8
    onset = rollback_onset(series, 12, relative=True)
    assert 9 <= onset <= 12


def test_absolute_mode():
    """Absolute mode uses raw slope values."""
    series = np.concatenate([np.zeros(10), np.linspace(0, 1, 10)])
    onset_rel = rollback_onset(series, 15, relative=True)
    onset_abs = rollback_onset(series, 15, relative=False, slope_threshold=0.01)
    # Both should produce valid results
    assert 0 <= onset_rel <= 15
    assert 0 <= onset_abs <= 15


def test_output_always_valid_index():
    """Onset must always be a valid index into the series."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(5, 50)
        series = rng.normal(0, 10, n).astype(float)
        cp = rng.integers(0, n)
        onset = rollback_onset(series, cp)
        assert 0 <= onset <= cp or onset == max(0, cp)
        assert onset < n


def test_exponential_onset():
    """An exponential ramp (e.g., memory leak) — rollback should find
    the start of the growth."""
    series = np.ones(20) * 100.0
    series[8:] = 100.0 * np.exp(np.linspace(0, 2, 12))
    onset = rollback_onset(series, 15)
    assert onset <= 10
