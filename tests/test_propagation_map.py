"""PropagationMap: calibrated edge delays and JSON round-trip."""

import json
import tempfile
from pathlib import Path

import pytest

from calibration.propagation_map import PropagationMap, empty_map, _compute_threshold
from rca_engine.dependency import get_dependency_graph


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


def test_empty_map_defaults():
    pm = empty_map()
    assert pm.default_threshold_s == 2.0
    assert pm.step_seconds == 1.0
    assert pm.edge_keys() == []


def test_empty_map_custom_defaults():
    pm = empty_map(default_threshold_s=3.0, step_seconds=5.0)
    assert pm.default_threshold_s == 3.0
    assert pm.step_seconds == 5.0


def test_record_single_observation():
    pm = empty_map()
    pm.record_observation("frontend", "checkoutservice", 1.0)
    thr = pm.get_edge_threshold("frontend", "checkoutservice")
    assert thr == pytest.approx(2.0)


def test_record_multiple_observations_uses_median():
    pm = empty_map()
    for d in [1.0, 3.0, 2.0]:
        pm.record_observation("frontend", "checkoutservice", d)
    thr = pm.get_edge_threshold("frontend", "checkoutservice")
    assert thr == pytest.approx(3.0)


def test_get_edge_threshold_unknown_edge_returns_default():
    pm = empty_map(default_threshold_s=2.0)
    assert pm.get_edge_threshold("a", "b") == 2.0


def test_path_threshold_direct_edge():
    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("checkoutservice", "paymentservice", 1.0)
    graph = get_dependency_graph()
    thr = pm.get_path_threshold("checkoutservice", "paymentservice", graph)
    assert thr == pytest.approx(2.0)


def test_path_threshold_two_hop_sums_edges():
    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("frontend", "checkoutservice", 1.0)
    pm.record_observation("checkoutservice", "paymentservice", 0.5)
    graph = get_dependency_graph()
    thr = pm.get_path_threshold("frontend", "paymentservice", graph)
    assert thr == pytest.approx(3.5)


def test_path_threshold_no_path_returns_default():
    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    thr = pm.get_path_threshold("adservice", "paymentservice", graph)
    assert thr == 2.0


def test_path_threshold_uncalibrated_edge_uses_default_per_hop():
    """Uncalibrated hops stack the default slack."""
    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    thr = pm.get_path_threshold("frontend", "paymentservice", graph)
    assert thr == pytest.approx(4.0)


def test_path_threshold_self_returns_default():
    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    thr = pm.get_path_threshold("frontend", "frontend", graph)
    assert thr == 2.0


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


def test_propagation_map_victim_classified_correctly():
    """Small onset gap vs calibrated edge → drop the downstream service."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("checkoutservice", "paymentservice", 1.0)

    graph = get_dependency_graph()
    onsets = {"paymentservice": 100.0, "checkoutservice": 101.5}
    trends = {"paymentservice": "up", "checkoutservice": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "paymentservice" in result
    assert "checkoutservice" not in result


def test_propagation_map_concurrent_classified_correctly():
    """Gap wider than calibrated edge slack → two independent faults."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("checkoutservice", "paymentservice", 1.0)

    graph = get_dependency_graph()
    onsets = {"paymentservice": 100.0, "checkoutservice": 103.0}
    trends = {"paymentservice": "up", "checkoutservice": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "paymentservice" in result
    assert "checkoutservice" in result


def test_propagation_map_unconnected_service_pinpointed():
    """No graph path tying the services — neither is demoted."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    graph = get_dependency_graph()
    onsets = {"adservice": 100.0, "paymentservice": 110.0}
    trends = {"adservice": "up", "paymentservice": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "adservice" in result
    assert "paymentservice" in result


def test_propagation_map_none_preserves_original_behavior():
    """Explicit None map should behave the same as omitting calibrated delays."""
    from rca_engine.fault_chain import pinpoint_faults

    graph = get_dependency_graph()
    onsets = {"paymentservice": 100.0, "checkoutservice": 105.0}
    trends = {"paymentservice": "up", "checkoutservice": "up"}

    result_no_map   = pinpoint_faults(onsets, trends, graph, propagation_map=None)
    result_with_map = pinpoint_faults(onsets, trends, graph, propagation_map=None,
                                      concurrency_threshold_s=2.0)
    assert result_no_map == result_with_map


def test_propagation_map_multi_hop_path_threshold():
    """Stack delays along the shortest path — FE within sum is ruled a victim."""
    from rca_engine.fault_chain import pinpoint_faults

    pm = empty_map(default_threshold_s=2.0)
    pm.record_observation("frontend", "checkoutservice", 1.0)
    pm.record_observation("checkoutservice", "paymentservice", 1.0)

    graph = get_dependency_graph()
    onsets = {"paymentservice": 100.0, "frontend": 103.5}
    trends = {"paymentservice": "up", "frontend": "up"}

    result = pinpoint_faults(onsets, trends, graph, propagation_map=pm)
    assert "paymentservice" in result
    assert "frontend" not in result
