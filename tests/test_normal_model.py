import numpy as np
from rca_engine.normal_model import NormalModel

def test_fit_and_predict():
    baseline = np.linspace(0, 100, 100)
    model = NormalModel(num_bins=100, metric_min=0, metric_max=100)
    model.fit(baseline)
    
    assert model.is_fit
    assert not model.is_frozen

    # Update with some values
    model.update(50.0)
    model.update(51.0)
    
    # Freeze
    model.freeze()
    assert model.is_frozen
    model.update(99.0) # shouldn't update the model
    
    series = np.array([50.0, 50.0, 99.0])
    err = model.prediction_error_at(2, series)
    # The previous value is 50.0. The next expected is ~51.0 based on linear or 50.0 based on latest update.
    # The actual is 99.0. The error should be large.
    assert err > 10.0

def test_unseen_state_generates_max_error():
    model = NormalModel(num_bins=100, metric_min=0, metric_max=100)
    model.fit(np.array([10.0, 10.0, 10.0]))  # Only knows about the bin containing 10.0
    
    series = np.array([80.0, 50.0]) # Predecessor is 80.0, but we never learned what follows 80.0
    err = model.prediction_error_at(1, series)
    
    # Sentinel error defaults to the metric range (100)
    assert err == 100.0

def test_prediction_errors_for():
    model = NormalModel(num_bins=100, metric_min=0, metric_max=100)
    model.fit(np.ones(10) * 50.0)
    
    series = np.array([50.0, 50.0, 99.0, 99.0])
    cps = [2, 3]
    errors = model.prediction_errors_for(cps, series)
    
    assert len(errors) == 2
    assert errors[2] > 0
    # For cp=3, predecessor is 99.0 (unseen), so it should return max error
    assert errors[3] == 100.0
