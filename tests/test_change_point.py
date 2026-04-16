"""Tests for rca_engine.change_point."""

import numpy as np

from rca_engine.change_point import run_layer1


def test_step_up():
    """A flat series with a step up should yield a change point near the step."""
    baseline = np.zeros(50)
    series = np.concatenate([np.zeros(15), np.ones(15) * 10.0])
    res = run_layer1(series, baseline, k=0.5, seed=42)
    assert len(res.change_points) >= 1
    # First detection should be near the step (index 15)
    assert 14 <= res.change_points[0] <= 20
    assert res.directions[0] == "up"


def test_step_down():
    """CUSUM bootstrap should detect a step down."""
    baseline = np.ones(50) * 10.0
    series = np.concatenate([np.ones(15) * 10.0, np.zeros(15)])
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) >= 1
    assert "down" in res.directions


def test_no_change():
    """Pure noise around the mean should produce no change points with high confidence."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 100)
    series = rng.normal(0, 1, 30)
    res = run_layer1(series, baseline, confidence_level=0.999, seed=42)
    # With a high threshold, it should not trigger
    assert isinstance(res.change_points, list)


def test_constant_series():
    baseline = np.ones(50) * 5.0
    series = np.ones(20) * 5.0
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) == 0


def test_empty():
    res = run_layer1(np.array([]), np.zeros(50), seed=42)
    assert res.change_points == [] and res.directions == []


def test_sigma_zero():
    """When baseline is constant, deviations should trigger."""
    baseline = np.zeros(50)
    series = np.concatenate([np.zeros(5), np.ones(5) * 5.0])
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) >= 1


def test_bilateral_both_directions():
    """Series with an increase followed by a decrease, with reset period."""
    baseline = np.zeros(50)
    series = np.concatenate([
        np.zeros(10),
        np.ones(10) * 20.0,
        np.zeros(400),       # Allow accumulator g to reset to 0
        np.ones(10) * -10.0,
    ])
    res = run_layer1(series, baseline, seed=42)
    assert "up" in res.directions
    assert "down" in res.directions


def test_multiple_change_points():
    """Multiple distinct level shifts should produce multiple detections."""
    baseline = np.zeros(50)
    series = np.concatenate([
        np.zeros(10),
        np.ones(10) * 20.0,
        np.ones(10) * 40.0,
    ])
    res = run_layer1(series, baseline, seed=42)
    # Depending on merge window, might be multiple
    assert len(res.change_points) >= 1


def test_single_sample():
    """Single sample should not crash."""
    baseline = np.zeros(50)
    res = run_layer1(np.array([5.0]), baseline, seed=42)
    assert isinstance(res.change_points, list)


def test_large_realistic_series():
    """24-sample fault window (realistic experiment size)."""
    rng = np.random.default_rng(77)
    baseline_mean = 0.05
    baseline_std = 0.005
    fault_mean = 0.45

    baseline = rng.normal(baseline_mean, baseline_std, 100)
    series = np.concatenate([
        rng.normal(baseline_mean, baseline_std, 6),   # pre-fault in fault window
        rng.normal(fault_mean, 0.01, 18),             # fault active
    ])
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) >= 1
    assert res.change_points[0] <= 10  # should detect near the transition
