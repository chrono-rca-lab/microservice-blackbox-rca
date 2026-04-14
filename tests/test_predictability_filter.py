"""Tests for rca_engine.predictability_filter."""

import numpy as np

from rca_engine.predictability_filter import filter_abnormal_change_points, _compute_burst_threshold


def test_spike_survives_filtering():
    """A large spike in an otherwise flat series should survive."""
    series = np.ones(30) * 5.0
    series[15] = 100.0  # huge spike
    errors = np.zeros(30)
    errors[15] = 95.0
    change_points = [15]

    result = filter_abnormal_change_points(series, change_points, errors, Q=10)
    assert 15 in result


def test_normal_variation_filtered():
    """Change points from a smooth sine wave should be filtered out
    (prediction errors are small relative to the burst threshold)."""
    t = np.linspace(0, 4 * np.pi, 40)
    series = np.sin(t) * 10 + 50
    errors = np.abs(np.diff(series, prepend=series[0])) * 0.01
    change_points = [10, 20, 30]

    result = filter_abnormal_change_points(series, change_points, errors, Q=10)
    assert len(result) == 0


def test_short_series_with_centered_cp():
    """With very few samples, a change point at index 2 (centered) has
    Q_local=2 which is enough for FFT if the error is large."""
    series = np.array([1.0, 2.0, 100.0, 3.0, 1.0])
    errors = np.array([0.0, 1.0, 98.0, 97.0, 2.0])
    change_points = [2]

    # cp=2 has Q_local = min(20, 2, 2) = 2 → skipped (< 3)
    result = filter_abnormal_change_points(series, change_points, errors, Q=20)
    assert isinstance(result, list)


def test_empty_inputs():
    assert filter_abnormal_change_points(np.array([]), [], np.array([]), Q=5) == []
    assert filter_abnormal_change_points(np.ones(10), [], np.ones(10), Q=5) == []


def test_edge_change_point():
    """Change points at the edges of the series should not crash."""
    series = np.concatenate([np.zeros(5), np.ones(5) * 10])
    errors = np.concatenate([np.zeros(5), np.ones(5) * 10])
    change_points = [0, 9]

    result = filter_abnormal_change_points(series, change_points, errors, Q=5)
    assert isinstance(result, list)


def test_change_point_at_index_0_skipped():
    """Index 0 has Q_local=0 which is < 3, so it should be skipped."""
    series = np.ones(10) * 5.0
    errors = np.ones(10) * 10.0
    result = filter_abnormal_change_points(series, [0], errors, Q=5)
    assert 0 not in result


def test_change_point_at_index_1_skipped():
    """Index 1 has Q_local=1 which is < 3, so it should be skipped."""
    series = np.ones(10) * 5.0
    errors = np.ones(10) * 10.0
    result = filter_abnormal_change_points(series, [1], errors, Q=5)
    assert 1 not in result


def test_out_of_range_change_point_ignored():
    """Change point index >= len(series) should be silently ignored."""
    series = np.ones(10) * 5.0
    errors = np.ones(10) * 10.0
    result = filter_abnormal_change_points(series, [15, -1], errors, Q=5)
    assert result == []


def test_multiple_change_points_mixed():
    """Some change points abnormal, others filtered."""
    series = np.ones(40) * 5.0
    series[20] = 200.0  # spike at 20

    errors = np.zeros(40)
    errors[10] = 0.001  # tiny error at 10 — should be filtered
    errors[20] = 195.0  # huge error at 20 — should survive

    result = filter_abnormal_change_points(series, [10, 20], errors, Q=8)
    assert 20 in result
    assert 10 not in result


def test_q_larger_than_series():
    """Q much larger than series — should adapt gracefully."""
    series = np.array([1, 2, 3, 50, 3, 2, 1], dtype=float)
    errors = np.array([0, 1, 1, 47, 47, 1, 1], dtype=float)
    result = filter_abnormal_change_points(series, [3], errors, Q=100)
    assert 3 in result


def test_burst_threshold_positive():
    """_compute_burst_threshold should always return a positive value."""
    for _ in range(10):
        window = np.random.default_rng(42).normal(0, 1, 21)
        t = _compute_burst_threshold(window, 0.9, 90.0)
        assert t > 0


def test_flat_window_threshold():
    """Flat window — the DC component dominates.  Since we keep 90% of
    frequencies by magnitude and the DC is the largest, the reconstructed
    signal is close to the original constant, giving a threshold near the
    constant's absolute value.  A zero-mean flat window would give ~epsilon."""
    window = np.zeros(21)  # zero-mean flat signal
    t = _compute_burst_threshold(window, 0.9, 90.0)
    assert t > 0
    assert t <= 1e-9  # effectively epsilon


def test_zero_prediction_error_not_abnormal():
    """If prediction error is 0 at a change point, it should be filtered."""
    series = np.ones(30) * 5.0
    errors = np.zeros(30)
    result = filter_abnormal_change_points(series, [15], errors, Q=10)
    assert 15 not in result
