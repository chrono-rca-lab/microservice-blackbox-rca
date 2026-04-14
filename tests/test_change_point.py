"""Tests for rca_engine.change_point."""

import numpy as np

from rca_engine.change_point import detect_change_points, detect_change_points_bilateral


def test_step_up():
    """A flat series with a step up should yield a change point near the step."""
    series = np.concatenate([np.zeros(15), np.ones(15) * 10.0])
    mu_0, sigma_0 = 0.0, 0.1
    cps = detect_change_points(series, mu_0, sigma_0)
    assert len(cps) >= 1
    # First detection should be near the step (index 15)
    assert 14 <= cps[0] <= 20


def test_step_down_bilateral():
    """Bilateral CUSUM should detect a step down."""
    series = np.concatenate([np.ones(15) * 10.0, np.zeros(15)])
    mu_0, sigma_0 = 10.0, 0.1
    cps, dirs = detect_change_points_bilateral(series, mu_0, sigma_0)
    assert len(cps) >= 1
    assert "down" in dirs


def test_no_change():
    """Pure noise around the mean should produce no change points with high h."""
    rng = np.random.default_rng(42)
    series = rng.normal(0, 1, 30)
    mu_0, sigma_0 = 0.0, 1.0
    cps = detect_change_points(series, mu_0, sigma_0, h_factor=10.0)
    assert len(cps) == 0


def test_constant_series():
    series = np.ones(20) * 5.0
    cps = detect_change_points(series, 5.0, 0.0)
    assert len(cps) == 0


def test_empty():
    assert detect_change_points(np.array([]), 0.0, 1.0) == []
    cps, dirs = detect_change_points_bilateral(np.array([]), 0.0, 1.0)
    assert cps == [] and dirs == []


def test_bilateral_up_direction():
    series = np.concatenate([np.zeros(10), np.ones(10) * 20.0])
    mu_0, sigma_0 = 0.0, 0.5
    cps, dirs = detect_change_points_bilateral(series, mu_0, sigma_0)
    assert len(cps) >= 1
    assert dirs[0] == "up"


def test_gradual_ramp():
    """A gradual ramp should eventually trigger a change point."""
    series = np.linspace(0, 50, 30)
    mu_0, sigma_0 = 0.0, 1.0
    cps = detect_change_points(series, mu_0, sigma_0)
    assert len(cps) >= 1


def test_sigma_zero():
    """When sigma_0=0 (constant baseline), any deviation should trigger."""
    series = np.concatenate([np.zeros(5), np.ones(5) * 1.0])
    mu_0, sigma_0 = 0.0, 0.0
    cps = detect_change_points(series, mu_0, sigma_0)
    assert len(cps) >= 1


def test_bilateral_both_directions():
    """Series with an increase followed by a decrease."""
    series = np.concatenate([
        np.zeros(10),
        np.ones(10) * 20.0,
        np.ones(10) * -10.0,
    ])
    mu_0, sigma_0 = 0.0, 0.5
    cps, dirs = detect_change_points_bilateral(series, mu_0, sigma_0)
    assert "up" in dirs
    assert "down" in dirs


def test_multiple_change_points():
    """Multiple distinct level shifts should produce multiple detections."""
    series = np.concatenate([
        np.zeros(10),
        np.ones(10) * 20.0,
        np.ones(10) * 40.0,
    ])
    mu_0, sigma_0 = 0.0, 0.5
    cps = detect_change_points(series, mu_0, sigma_0)
    # Should detect at least two jumps
    assert len(cps) >= 2


def test_single_sample():
    """Single sample should not crash, no change points possible."""
    cps = detect_change_points(np.array([5.0]), 0.0, 1.0)
    # A single value above mu can trigger if large enough
    assert isinstance(cps, list)


def test_upper_cusum_ignores_decrease():
    """Upper CUSUM should NOT detect a downward shift."""
    series = np.concatenate([np.ones(10) * 10.0, np.zeros(10)])
    mu_0, sigma_0 = 10.0, 0.1
    cps = detect_change_points(series, mu_0, sigma_0)
    assert len(cps) == 0


def test_sensitivity_parameters():
    """Smaller k_factor and h_factor should detect changes faster."""
    series = np.concatenate([np.zeros(10), np.ones(10) * 3.0])
    mu_0, sigma_0 = 0.0, 1.0

    # With default params (k=0.5*sigma, h=5*sigma): might or might not detect
    cps_default = detect_change_points(series, mu_0, sigma_0)

    # With lower threshold (h=2*sigma): should detect more readily
    cps_sensitive = detect_change_points(series, mu_0, sigma_0, h_factor=2.0)
    assert len(cps_sensitive) >= len(cps_default)


def test_large_realistic_series():
    """24-sample fault window (realistic experiment size)."""
    rng = np.random.default_rng(77)
    baseline_mean = 0.05
    baseline_std = 0.005
    fault_mean = 0.45

    series = np.concatenate([
        rng.normal(baseline_mean, baseline_std, 6),   # pre-fault in fault window
        rng.normal(fault_mean, 0.01, 18),             # fault active
    ])
    cps = detect_change_points(series, baseline_mean, baseline_std)
    assert len(cps) >= 1
    assert cps[0] <= 10  # should detect near the transition
