"""Tests for rca_engine.normal_model."""

import numpy as np
import pytest

from rca_engine.normal_model import NormalModel


def test_sine_wave_low_error():
    """After fitting on a sine wave, prediction errors on the same wave
    should be small relative to the amplitude."""
    t = np.linspace(0, 2 * np.pi, 50)
    baseline = np.sin(t[:20])
    fault = np.sin(t[20:])

    model = NormalModel(num_bins=100).fit(baseline)
    errors = model.prediction_errors(fault)

    assert len(errors) == len(fault)
    # Mean error should be under the amplitude (1.0); with sparse data
    # and adaptive bins the Markov model is coarse
    assert np.mean(errors) < 0.8


def test_step_change_high_error():
    """Constant baseline, then a step change -> errors after the step
    should be large."""
    baseline = np.ones(15) * 10.0
    fault = np.concatenate([np.ones(5) * 10.0, np.ones(15) * 50.0])

    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)

    # Errors after the step (indices 6+) should be large
    assert np.mean(errors[6:]) > 10.0
    # Errors before the step should be small
    assert np.mean(errors[1:5]) < 5.0


def test_constant_series():
    baseline = np.ones(10) * 5.0
    fault = np.ones(20) * 5.0

    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert np.allclose(errors, 0, atol=1e-6)


def test_degenerate_short_baseline():
    """Model should not crash with very few samples."""
    baseline = np.array([1.0])
    fault = np.array([1.0, 2.0, 3.0])

    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert len(errors) == 3


def test_empty_input():
    model = NormalModel().fit(np.array([]))
    errors = model.prediction_errors(np.array([1.0, 2.0]))
    assert len(errors) == 2


def test_adaptive_bin_count():
    """With 10 samples, bins should be capped well below 100."""
    baseline = np.arange(10, dtype=float)
    model = NormalModel(num_bins=100).fit(baseline)
    assert model._n_bins <= 10
    assert model._n_bins >= 3


def test_errors_length_equals_series():
    """prediction_errors must return the same length as the input series."""
    for n_baseline, n_fault in [(5, 10), (10, 24), (20, 50), (2, 3)]:
        baseline = np.random.default_rng(42).normal(0, 1, n_baseline)
        fault = np.random.default_rng(42).normal(0, 1, n_fault)
        model = NormalModel().fit(baseline)
        errors = model.prediction_errors(fault)
        assert len(errors) == n_fault, f"n_baseline={n_baseline}, n_fault={n_fault}"


def test_first_error_is_zero():
    """The first element should always be 0 (no predecessor to predict from)."""
    baseline = np.arange(10, dtype=float)
    fault = np.arange(10, 30, dtype=float)
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert errors[0] == 0.0


def test_errors_are_nonnegative():
    """All prediction errors must be >= 0."""
    rng = np.random.default_rng(99)
    baseline = rng.normal(50, 10, 15)
    fault = rng.normal(80, 10, 24)
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert np.all(errors >= 0)


def test_out_of_range_fault_values():
    """Fault values far outside the baseline range should produce large errors."""
    baseline = np.ones(10) * 10.0
    fault = np.ones(10) * 1000.0
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    # Errors after index 0 should be very large
    assert np.mean(errors[1:]) > 100


def test_negative_values():
    """Model handles negative metric values correctly."""
    baseline = np.linspace(-10, -5, 10)
    fault = np.linspace(-5, 5, 20)
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert len(errors) == 20
    assert np.all(np.isfinite(errors))


def test_single_fault_sample():
    """Single fault sample — should return zeros (can't predict)."""
    baseline = np.arange(10, dtype=float)
    fault = np.array([5.0])
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert len(errors) == 1
    assert errors[0] == 0.0


def test_identical_values_baseline():
    """Baseline with all identical values — zero range edge case."""
    baseline = np.ones(10) * 42.0
    fault = np.ones(5) * 42.0
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert np.allclose(errors, 0, atol=1e-4)


def test_two_sample_baseline():
    """Minimum viable baseline (2 samples) — should fit without crash."""
    baseline = np.array([1.0, 2.0])
    fault = np.array([1.0, 2.0, 3.0, 10.0])
    model = NormalModel().fit(baseline)
    errors = model.prediction_errors(fault)
    assert len(errors) == 4
    # Error for the jump to 10.0 should be large
    assert errors[3] > errors[1]
