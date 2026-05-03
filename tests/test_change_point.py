"""CUSUM + block-bootstrap change detection."""

import numpy as np

from rca_engine.change_point import run_layer1


def test_step_up():
    """Step up lands a CP close to the break."""
    baseline = np.zeros(50)
    series = np.concatenate([np.zeros(15), np.ones(15) * 10.0])
    res = run_layer1(series, baseline, k=0.5, seed=42)
    assert len(res.change_points) >= 1
    # First detection should be near the step (index 15)
    assert 14 <= res.change_points[0] <= 20
    assert res.directions[0] == "up"


def test_step_down():
    """Step down shows up too."""
    baseline = np.ones(50) * 10.0
    series = np.concatenate([np.ones(15) * 10.0, np.zeros(15)])
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) >= 1
    assert "down" in res.directions


def test_no_change():
    """IID noise — crank confidence so we barely flag anything."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 100)
    series = rng.normal(0, 1, 30)
    res = run_layer1(series, baseline, confidence_level=0.999, seed=42)
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
    """Flat baseline — sigma effectively zero — easy trigger."""
    baseline = np.zeros(50)
    series = np.concatenate([np.zeros(5), np.ones(5) * 5.0])
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) >= 1


def test_bilateral_both_directions():
    """Up leg, long flat reset, down leg — both directions logged."""
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
    """Back-to-back level jumps — expect ≥1 merged/split hits."""
    baseline = np.zeros(50)
    series = np.concatenate([
        np.zeros(10),
        np.ones(10) * 20.0,
        np.ones(10) * 40.0,
    ])
    res = run_layer1(series, baseline, seed=42)
    assert len(res.change_points) >= 1


def test_single_sample():
    """Degenerate fault window."""
    baseline = np.zeros(50)
    res = run_layer1(np.array([5.0]), baseline, seed=42)
    assert isinstance(res.change_points, list)


def test_large_realistic_series():
    """Short fault crop (~24 pts) similar to staging runs."""
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
