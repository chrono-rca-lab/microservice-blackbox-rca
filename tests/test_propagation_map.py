"""Tests for calibration.propagation_map."""

import json
import tempfile
from pathlib import Path

import pytest

from calibration.propagation_map import PropagationMap, empty_map, _compute_threshold
from rca_engine.dependency import get_dependency_graph


# ---------------------------------------------------------------------------
# _compute_threshold
# ---------------------------------------------------------------------------

def test_threshold_floor():
    """Threshold always adds at least 1.0 s (metric resolution floor)."""
    assert _compute_threshold(0.0) == 1.0
    assert _compute_threshold(0.5) == 1.5   # 0.5 + max(1.0, 0.25) = 0.5 + 1.0
    assert _compute_threshold(1.0) == 2.0   # 1.0 + max(1.0, 0.5)  = 1.0 + 1.0


def test_threshold_scales_with_large_median():
    """For large medians the 50% factor dominates over the 1.0 floor."""
    # median=4.0 -> 4.0 + max(1.0, 2.0) = 6.0
    assert _compute_threshold(4.0) == 6.0
    # median=10.0 -> 10.0 + max(1.0, 5.0) = 15.0
    assert _compute_threshold(10.0) == 15.0


# ---------------------------------------------------------------------------
# empty_map / construction
# ---------------------------------------------------------------------------

def test_empty_map_defaults():
    pm = empty_map()
    assert pm.default_threshold_s == 2.0
    assert pm.step_seconds == 1.0
    assert pm.edge_keys() == []


def test_empty_map_custom_defaults():
    pm = empty_map(default_threshold_s=3.0, step_seconds=5.0)
    assert pm.default_threshold_s == 3.0
    assert pm.step_seconds == 5.0


# ---------------------------------------------------------------------------
# record_observation / get_edge_threshold
# ---------------------------------------------------------------------------

def test_record_single_observation():
    pm = empty_map()
    pm.record_observation("frontend", "checkoutservice", 1.0)
    # median=1.0, threshold=1.0+max(1.0, 0.5)=2.0
    thr = pm.get_edge_threshold("frontend", "checkoutservice")
    assert thr == pytest.approx(2.0)


def test_record_multiple_observations_uses_median():
    pm = empty_map()
    for d in [1.0, 3.0, 2.0]:
        pm.record_observation("frontend", "checkoutservice", d)
    # median of [1,3,2] = 2.0 -> threshold = 2.0 + max(1.0, 1.0) = 3.0
    thr = pm.get_edge_threshold("frontend", "checkoutservice")
    assert thr == pytest.approx(3.0)


def test_get_edge_threshold_unknown_edge_returns_default():
    pm = empty_map(default_threshold_s=2.0)
    assert pm.get_edge_threshold("a", "b") == 2.0


# ---------------------------------------------------------------------------
# get_path_threshold
# ---------------------------------------------------------------------------

def test_path_threshold_direct_edge():
    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("checkoutservice", "paymentservice", 1.0)
    graph = get_dependency_graph()
    # Direct edge: checkoutservice -> paymentservice
    thr = pm.get_path_threshold("checkoutservice", "paymentservice", graph)
    # threshold for that edge = 1.0 + max(1.0, 0.5) = 2.0
    assert thr == pytest.approx(2.0)


def test_path_threshold_two_hop_sums_edges():
    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("frontend", "checkoutservice", 1.0)       # threshold=2.0
    pm.record_observation("checkoutservice", "paymentservice", 0.5)  # threshold=1.5
    graph = get_dependency_graph()
    # Path: frontend -> checkoutservice -> paymentservice  (sum = 3.5)
    thr = pm.get_path_threshold("frontend", "paymentservice", graph)
    assert thr == pytest.approx(3.5)


def test_path_threshold_no_path_returns_default():
    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    # No path from adservice to paymentservice
    thr = pm.get_path_threshold("adservice", "paymentservice", graph)
    assert thr == 2.0


def test_path_threshold_uncalibrated_edge_uses_default_per_hop():
    """Each missing edge falls back to default, and they sum."""
    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    # Path: frontend -> checkoutservice -> paymentservice
    # Neither edge calibrated — each gets default 2.0 -> sum = 4.0
    thr = pm.get_path_threshold("frontend", "paymentservice", graph)
    assert thr == pytest.approx(4.0)


def test_path_threshold_self_returns_default():
    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    # src == dst is not a propagation scenario
    thr = pm.get_path_threshold("frontend", "frontend", graph)
    assert thr == 2.0


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    pm = empty_map(default_threshold_s=1.5)
    pm.record_observation("frontend", "checkoutservice", 1.0)
    pm.record_observation("frontend", "checkoutservice", 2.0)
    pm.trials = 2

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        pm.save(path)
        pm2 = PropagationMap.load(path)

        assert pm2.default_threshold_s == 1.5
        assert pm2.trials == 2
        assert "frontend->checkoutservice" in pm2.edge_keys()
        thr_orig = pm.get_edge_threshold("frontend", "checkoutservice")
        thr_load = pm2.get_edge_threshold("frontend", "checkoutservice")
        assert thr_orig == pytest.approx(thr_load)
    finally:
        path.unlink(missing_ok=True)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        PropagationMap.load("/nonexistent/path/to/map.json")


def test_saved_json_is_valid():
    pm = empty_map()
    pm.record_observation("cartservice", "redis-cart", 0.5)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = Path(f.name)

    try:
        pm.save(path)
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "edges" in data
        assert "cartservice->redis-cart" in data["edges"]
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Edge-aware Layer 7 integration tests (via pinpoint_faults)
# ---------------------------------------------------------------------------

def test_propagation_map_victim_classified_correctly():
    """onset_diff <= edge_threshold → within propagation window → victim (not pinpointed)."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    # record_observation(caller=checkoutservice, callee=paymentservice, delay=1.0)
    # _compute_threshold(1.0) = 1.0 + max(1.0, 0.5) = 2.0s
    pm.record_observation("checkoutservice", "paymentservice", 1.0)

    graph = get_dependency_graph()
    # paymentservice onset=100, checkoutservice onset=101.5 (diff=1.5 <= threshold=2.0)
    onsets = {"paymentservice": 100.0, "checkoutservice": 101.5}
    trends = {"paymentservice": "up", "checkoutservice": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "paymentservice" in result
    assert "checkoutservice" not in result   # propagation victim


def test_propagation_map_concurrent_classified_correctly():
    """onset_diff > edge_threshold → outside propagation window → concurrent (pinpointed)."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    # Same calibrated edge, threshold = 2.0
    pm.record_observation("checkoutservice", "paymentservice", 1.0)

    graph = get_dependency_graph()
    # onset gap = 3.0 > threshold 2.0 → outside window → concurrent independent fault
    onsets = {"paymentservice": 100.0, "checkoutservice": 103.0}
    trends = {"paymentservice": "up", "checkoutservice": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "paymentservice" in result
    assert "checkoutservice" in result   # concurrent independent fault


def test_propagation_map_unconnected_service_pinpointed():
    """Service with no dependency path to root → always pinpointed."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    # adservice and paymentservice have no dependency path between them
    onsets = {"adservice": 100.0, "paymentservice": 110.0}
    trends = {"adservice": "up", "paymentservice": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "adservice" in result
    assert "paymentservice" in result   # independent — no path from adservice


def test_propagation_map_none_preserves_original_behavior():
    """When propagation_map=None, behavior matches the original algorithm exactly."""
    from rca_engine.fault_chain import pinpoint_faults

    graph = get_dependency_graph()
    # Classic propagation scenario: paymentservice faults, checkoutservice is victim
    onsets = {"paymentservice": 100.0, "checkoutservice": 105.0}
    trends = {"paymentservice": "up", "checkoutservice": "up"}

    result_no_map   = pinpoint_faults(onsets, trends, graph, propagation_map=None)
    result_with_map = pinpoint_faults(onsets, trends, graph, propagation_map=None,
                                      concurrency_threshold_s=2.0)
    assert result_no_map == result_with_map


def test_propagation_map_multi_hop_path_threshold():
    """Multi-hop path sums edge thresholds; service within sum is a victim."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    # frontend -> checkoutservice (threshold=2.0), checkoutservice -> paymentservice (threshold=2.0)
    pm.record_observation("frontend", "checkoutservice", 1.0)
    pm.record_observation("checkoutservice", "paymentservice", 1.0)

    graph = get_dependency_graph()
    # paymentservice onset=100, frontend onset=103.5
    # Path: paymentservice->checkoutservice->frontend, summed threshold = 4.0
    # onset_diff = 3.5 <= 4.0 → within propagation window → victim
    onsets = {"paymentservice": 100.0, "frontend": 103.5}
    trends = {"paymentservice": "up", "frontend": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "paymentservice" in result
    assert "frontend" not in result   # 3.5s <= 4.0s threshold → propagation victim
