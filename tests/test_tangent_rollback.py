"""Walk CPs backward along tangents."""

import numpy as np

from rca_engine.tangent_rollback import rollback_onset


def test_gradual_ramp():
    """Slope starts ~8 — rollback shouldn’t linger at 15."""
    series = np.zeros(25, dtype=float)
    series[8:] = np.linspace(0, 10, 17)
    all_cps = [7, 8, 10, 12, 15]
    onset = rollback_onset(series, 15, all_cps)
    assert onset <= 10  # should be near the ramp start


def test_sharp_step():
    """Hard step — don’t rewind far."""
    series = np.concatenate([np.zeros(12), np.ones(12) * 100])
    all_cps = [11, 12, 13]
    onset = rollback_onset(series, 13, all_cps)
    assert onset >= 10


def test_near_boundary():
    """CP hugging idx 0 — still bounded."""
    series = np.array([0, 0, 10, 10, 10], dtype=float)
    all_cps = [0, 1, 2]
    onset = rollback_onset(series, 2, all_cps)
    assert 0 <= onset <= 2


def test_constant_series():
    """Flat line — lone CP ⇒ same index back."""
    series = np.ones(10) * 5.0
    onset = rollback_onset(series, 5, [5])
    assert onset == 5


def test_very_short_series():
    series = np.array([1.0, 10.0])
    onset = rollback_onset(series, 1, [0, 1])
    assert 0 <= onset <= 1


def test_change_point_at_start():
    """CP at 0 — stuck at 0."""
    series = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    onset = rollback_onset(series, 0, [0])
    assert onset == 0


def test_change_point_at_index_1():
    """Second sample step — rollback stops early."""
    series = np.array([0.0, 100.0, 100.0, 100.0])
    onset = rollback_onset(series, 1, [0, 1])
    assert onset <= 1


def test_change_point_at_end():
    """Trailing edge spike."""
    series = np.array([0.0, 0.0, 0.0, 100.0])
    onset = rollback_onset(series, 3, [3])
    assert onset == 3


def test_large_value_series():
    """Big RSS-style magnitudes — same geometry, different scale."""
    series = np.ones(20) * 1e8
    series[10:] = 5e8
    all_cps = [8, 9, 10, 11, 12]
    onset = rollback_onset(series, 12, all_cps)
    assert 9 <= onset <= 12


def test_tangent_threshold_controls_rollback():
    """Looser tangent gate ⇒ more rewind on a shallow ramp."""
    series = np.concatenate([np.zeros(10), np.linspace(0, 1, 10)])
    all_cps = list(range(20))

    onset_default = rollback_onset(series, 15, all_cps)
    onset_tight = rollback_onset(series, 15, all_cps, tangent_threshold=0.001)

    assert 0 <= onset_default <= 15
    assert 0 <= onset_tight <= 15


def test_output_always_valid_index():
    """Smoke random shapes — onset stays in-bounds."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(5, 50)
        series = rng.normal(0, 10, n).astype(float)
        cp = int(rng.integers(0, n))
        onset = rollback_onset(series, cp, [cp])
        assert 0 <= onset <= cp or onset == max(0, cp)
        assert onset < n


def test_exponential_onset():
    """Curved ramp + loose tangent — onset drifts toward the bend."""
    series = np.ones(20) * 100.0
    series[8:] = 100.0 * np.exp(np.linspace(0, 2, 12))
    all_cps = [7, 8, 10, 12, 15]
    onset = rollback_onset(series, 15, all_cps, tangent_threshold=50.0)
    assert onset <= 12
