"""FFT burst gate on CP prediction errors."""

import numpy as np

from rca_engine.predictability_filter import filter_abnormal_change_points, _compute_burst_threshold


def test_spike_survives_filtering():
    """A large spike in an otherwise flat series should survive."""
    series = np.ones(30) * 5.0
    series[15] = 100.0  # huge spike
    change_point_errors = {15: 95.0}

    result = filter_abnormal_change_points(series, change_point_errors, Q=10)
    assert 15 in result


def test_normal_variation_filtered():
    """Sinewave — errors stay tiny next to spectral threshold."""
    t = np.linspace(0, 4 * np.pi, 40)
    series = np.sin(t) * 10 + 50
    errors_arr = np.abs(np.diff(series, prepend=series[0])) * 0.01
    change_point_errors = {10: float(errors_arr[10]),
                           20: float(errors_arr[20]),
                           30: float(errors_arr[30])}

    result = filter_abnormal_change_points(series, change_point_errors, Q=10)
    assert len(result) == 0


def test_short_series_with_centered_cp():
    """Short window — falls back paths still return a list type."""
    series = np.array([1.0, 2.0, 100.0, 3.0, 1.0])
    change_point_errors = {2: 98.0}

    result = filter_abnormal_change_points(series, change_point_errors, Q=20)
    assert isinstance(result, list)


def test_empty_inputs():
    assert filter_abnormal_change_points(np.array([]), {}, Q=5) == []
    assert filter_abnormal_change_points(np.ones(10), {}, Q=5) == []


def test_edge_change_point():
    """Change points at the edges of the series should not crash."""
    series = np.concatenate([np.zeros(5), np.ones(5) * 10])
    change_point_errors = {0: 10.0, 9: 10.0}

    result = filter_abnormal_change_points(series, change_point_errors, Q=5)
    assert isinstance(result, list)


def test_change_point_at_index_0_skipped():
    """Left edge CP — narrow local spectrum, uses global floor."""
    series = np.ones(10) * 5.0
    change_point_errors = {0: 10.0}
    result = filter_abnormal_change_points(series, change_point_errors, Q=5)
    assert isinstance(result, list)


def test_change_point_at_index_1_skipped():
    """Same idea one step in."""
    series = np.ones(10) * 5.0
    change_point_errors = {1: 10.0}
    result = filter_abnormal_change_points(series, change_point_errors, Q=5)
    assert isinstance(result, list)


def test_out_of_range_change_point_ignored():
    """OOB index skipped — empty result."""
    series = np.ones(10) * 5.0
    change_point_errors = {15: 10.0}
    result = filter_abnormal_change_points(series, change_point_errors, Q=5)
    assert result == []


def test_multiple_change_points_mixed():
    """Mix of tame vs huge errors — keep the spike."""
    series = np.ones(40) * 5.0
    series[20] = 200.0  # spike at 20
    change_point_errors = {10: 0.0, 20: 195.0}

    result = filter_abnormal_change_points(series, change_point_errors, Q=8)
    assert 20 in result
    assert 10 not in result


def test_q_larger_than_series():
    """FFT window clamps — still spikes through."""
    series = np.array([1, 2, 3, 50, 3, 2, 1], dtype=float)
    change_point_errors = {3: 47.0}
    result = filter_abnormal_change_points(series, change_point_errors, Q=100)
    assert 3 in result


def test_burst_threshold_positive():
    """_compute_burst_threshold should always return a positive value."""
    for _ in range(10):
        window = np.random.default_rng(42).normal(0, 1, 21)
        t = _compute_burst_threshold(
            window,
            high_freq_fraction=0.9,
            burst_percentile=90.0,
            threshold_floor=1e-10,
        )
        assert t > 0


def test_flat_window_threshold():
    """Zero-mean flat window — reconstructed burst is near zero."""
    window = np.zeros(21)
    t = _compute_burst_threshold(
        window,
        high_freq_fraction=0.9,
        burst_percentile=90.0,
        threshold_floor=1e-10,
    )
    assert t > 0
    assert t <= 1e-9  # effectively epsilon (floor)


def test_zero_prediction_error_not_abnormal():
    """Zero error ⇒ drop."""
    series = np.ones(30) * 5.0
    change_point_errors = {15: 0.0}
    result = filter_abnormal_change_points(series, change_point_errors, Q=10)
    assert 15 not in result
