"""Black-box RCA: pinpoint() plus the helpers it uses."""

import numpy as np

from rca_engine.fault_chain import (
    pinpoint,
    pinpoint_faults,
    _split_series,
    _determine_trend,
    _analyze_metric,
    STEP_SECONDS,
)
from rca_engine.dependency import get_dependency_graph


# Synthetic metric fixtures (7 metrics per service).

METRICS = [
    "cpu_rate", "cpu_throttle_ratio", "mem_wss",
    "net_rx_rate", "net_tx_rate", "fs_read_rate", "fs_write_rate",
]


def _flat_metrics(n_samples: int, value: float = 1.0) -> dict[str, np.ndarray]:
    """Quiet baseline — all metrics flat."""
    return {
        "cpu_rate": np.ones(n_samples) * value,
        "cpu_throttle_ratio": np.zeros(n_samples),
        "mem_wss": np.ones(n_samples) * 1e8,
        "net_rx_rate": np.ones(n_samples) * 1000,
        "net_tx_rate": np.ones(n_samples) * 1000,
        "fs_read_rate": np.ones(n_samples) * 500,
        "fs_write_rate": np.ones(n_samples) * 500,
    }


def _faulty_metrics(
    n_baseline: int,
    n_fault: int,
    fault_start_offset: int = 2,
) -> dict[str, np.ndarray]:
    """CPU steps up mid-fault window; everything else stays flat."""
    total = n_baseline + n_fault
    cpu = np.ones(total) * 0.1
    cpu[n_baseline + fault_start_offset :] = 0.9

    metrics = _flat_metrics(total, value=0.1)
    metrics["cpu_rate"] = cpu
    return metrics


def _make_windows(n_bl: int, n_ft: int, gap: float = 10.0):
    """Baseline/fault timestamps using the same step spacing as the eval runners."""
    bl_start = 1000.0
    bl_end = bl_start + n_bl * STEP_SECONDS
    ft_start = bl_end + gap
    ft_end = ft_start + n_ft * STEP_SECONDS
    return bl_start, bl_end, ft_start, ft_end


class TestPinpointPipeline:

    def test_single_faulty_service(self):
        """Single CPU step-up should float that service to the top."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {}
        matrix["currencyservice"] = _faulty_metrics(n_bl, n_ft)
        for svc in ["frontend", "cartservice", "adservice", "paymentservice"]:
            matrix[svc] = _flat_metrics(total, value=0.1)

        result = pinpoint(
            metric_matrix=matrix,
            baseline_window=(bl_start, bl_end),
            fault_window=(ft_start, ft_end),
        )

        assert len(result) >= 1
        assert result[0]["service"] == "currencyservice"
        assert "cpu_rate" in result[0]["abnormal_metrics"]

    def test_empty_matrix(self):
        assert pinpoint({}, (0, 50), (60, 180)) == []

    def test_no_fault(self):
        """Quiet data — expect an empty culprit list."""
        matrix = {
            "frontend": _flat_metrics(34),
            "adservice": _flat_metrics(34),
        }
        result = pinpoint(matrix, (0, 50), (60, 170))
        assert result == []

    def test_output_format_fields(self):
        """Each ranked entry should expose the documented keys and types."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {
            "currencyservice": _faulty_metrics(n_bl, n_ft),
            "adservice": _flat_metrics(total, value=0.1),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))

        for entry in result:
            assert "rank" in entry and isinstance(entry["rank"], int)
            assert "service" in entry and isinstance(entry["service"], str)
            assert "onset_time" in entry and isinstance(entry["onset_time"], float)
            assert "confidence" in entry and 0.0 <= entry["confidence"] <= 1.0
            assert "abnormal_metrics" in entry and isinstance(entry["abnormal_metrics"], list)

    def test_ranks_are_sequential(self):
        """Ranks should be dense starting at 1."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {"currencyservice": _faulty_metrics(n_bl, n_ft)}
        for svc in ["adservice", "emailservice"]:
            matrix[svc] = _flat_metrics(total, value=0.1)

        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        ranks = [e["rank"] for e in result]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_root_causes_ranked_before_propagation(self):
        """Upstream failure should sort before knock-on effects downstream."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        # paymentservice spikes first; checkoutservice lags (calls payment)
        matrix = {
            "paymentservice": _faulty_metrics(n_bl, n_ft, fault_start_offset=2),
            "adservice": _flat_metrics(total, value=0.1),
        }
        # checkout CPU rises later than payment
        checkout_metrics = _flat_metrics(total, value=0.1)
        cpu = np.ones(total) * 0.1
        cpu[n_bl + 6:] = 0.8  # later onset
        checkout_metrics["cpu_rate"] = cpu
        matrix["checkoutservice"] = checkout_metrics

        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        names = [e["service"] for e in result]
        if "paymentservice" in names and "checkoutservice" in names:
            assert names.index("paymentservice") < names.index("checkoutservice")

    def test_multi_metric_fault(self):
        """Joint CPU/mem shift — both signals should land in abnormal_metrics."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        metrics = _flat_metrics(total, value=0.1)
        # CPU spike
        cpu = np.ones(total) * 0.1
        cpu[n_bl + 2:] = 0.9
        metrics["cpu_rate"] = cpu
        # Memory spike
        mem = np.ones(total) * 1e8
        mem[n_bl + 2:] = 5e8
        metrics["mem_wss"] = mem

        matrix = {
            "currencyservice": metrics,
            "adservice": _flat_metrics(total, value=0.1),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))

        assert len(result) >= 1
        assert result[0]["service"] == "currencyservice"
        abnormal = result[0]["abnormal_metrics"]
        assert "cpu_rate" in abnormal
        assert "mem_wss" in abnormal
        assert result[0]["confidence"] > 1 / 7  # not single-metric noise

    def test_realistic_magnitudes(self):
        """Cluster-scale CPU and RSS magnitudes — both shifts should register."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        rng = np.random.default_rng(42)
        metrics = {}
        for m in METRICS:
            metrics[m] = np.ones(total) * 0.05

        # Small CPU change (realistic: 0.05 → 0.45)
        metrics["cpu_rate"] = np.concatenate([
            rng.normal(0.05, 0.005, n_bl),
            rng.normal(0.45, 0.01, n_ft),
        ])
        # Large memory change (realistic: 100MB → 400MB)
        metrics["mem_wss"] = np.concatenate([
            rng.normal(1e8, 1e6, n_bl),
            rng.normal(4e8, 1e6, n_ft),
        ])

        matrix = {
            "emailservice": metrics,
            "adservice": _flat_metrics(total, value=0.05),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert len(result) >= 1
        assert result[0]["service"] == "emailservice"

    def test_service_not_in_dependency_graph(self):
        """Unknown service name: still run through the scorer, no graph edge assumptions."""
        n_bl, n_ft = 10, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {
            "unknownservice": _faulty_metrics(n_bl, n_ft),
            "adservice": _flat_metrics(total, value=0.1),
        }
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert len(result) >= 1
        services = [e["service"] for e in result]
        assert "unknownservice" in services

    def test_very_short_fault_window(self):
        """Tiny fault slice — shouldn't throw."""
        n_bl, n_ft = 10, 3
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {"adservice": _faulty_metrics(n_bl, n_ft)}
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert isinstance(result, list)

    def test_very_short_baseline(self):
        """Minimal baseline length (2 pts) — still returns a list."""
        n_bl, n_ft = 2, 24
        total = n_bl + n_ft
        bl_start, bl_end, ft_start, ft_end = _make_windows(n_bl, n_ft)

        matrix = {"adservice": _faulty_metrics(n_bl, n_ft)}
        result = pinpoint(matrix, (bl_start, bl_end), (ft_start, ft_end))
        assert isinstance(result, list)


class TestPinpointFaults:

    def test_single_root_cause(self):
        graph = get_dependency_graph()
        onsets = {"currencyservice": 100.0, "frontend": 105.0}
        trends = {"currencyservice": "up", "frontend": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert result[0] == "currencyservice"

    def test_concurrent_faults(self):
        graph = get_dependency_graph()
        onsets = {"emailservice": 100.0, "paymentservice": 101.5}
        trends = {"emailservice": "up", "paymentservice": "up"}
        result = pinpoint_faults(onsets, trends, graph, concurrency_threshold_s=2.0)
        assert "emailservice" in result
        assert "paymentservice" in result

    def test_propagation_filtered(self):
        """Payment breaks first; checkout is a victim and should drop out."""
        graph = get_dependency_graph()
        onsets = {"paymentservice": 100.0, "checkoutservice": 105.0}
        trends = {"paymentservice": "up", "checkoutservice": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert "paymentservice" in result
        assert "checkoutservice" not in result

    def test_independent_faults(self):
        """Two leaves with no path between them — keep both culprits."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "paymentservice": 108.0}
        trends = {"adservice": "up", "paymentservice": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert "adservice" in result
        assert "paymentservice" in result

    def test_external_cause(self):
        """Global coordinated uptrend everywhere — bail out as non-local."""
        graph = get_dependency_graph()
        all_svcs = list(graph.keys())
        onsets = {s: 100.0 + i * 0.5 for i, s in enumerate(all_svcs)}
        trends = {s: "up" for s in all_svcs}
        # pass fleet size so coordinated fleet-wide outage logic can fire
        result = pinpoint_faults(onsets, trends, graph, n_monitored_services=len(all_svcs))
        assert result == []

    def test_empty_input(self):
        assert pinpoint_faults({}, {}, {}) == []

    def test_external_cause_not_triggered_with_mixed_trends(self):
        """Mixed up/down trends — should not classify as ambient outage."""
        graph = get_dependency_graph()
        all_svcs = list(graph.keys())
        onsets = {s: 100.0 + i for i, s in enumerate(all_svcs)}
        trends = {s: ("up" if i % 2 == 0 else "down") for i, s in enumerate(all_svcs)}
        result = pinpoint_faults(onsets, trends, graph)
        assert len(result) > 0

    def test_external_cause_not_triggered_with_subset(self):
        """Subset abnormal — still local enough to emit suspects."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "paymentservice": 101.0, "emailservice": 102.0}
        trends = {"adservice": "up", "paymentservice": "up", "emailservice": "up"}
        result = pinpoint_faults(onsets, trends, graph)
        assert len(result) > 0

    def test_propagation_three_hops(self):
        """catalog fault ripples reco then FE — only catalog stays as culprit."""
        graph = get_dependency_graph()
        onsets = {
            "productcatalogservice": 100.0,
            "recommendationservice": 103.0,
            "frontend": 106.0,
        }
        trends = {s: "up" for s in onsets}
        result = pinpoint_faults(onsets, trends, graph)
        assert "productcatalogservice" in result
        assert "recommendationservice" not in result
        assert "frontend" not in result

    def test_concurrent_threshold_exact_boundary(self):
        """Onset gap exactly at concurrency window — treat as concurrent."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "emailservice": 102.0}
        trends = {"adservice": "up", "emailservice": "up"}
        result = pinpoint_faults(onsets, trends, graph, concurrency_threshold_s=2.0)
        assert "adservice" in result
        assert "emailservice" in result

    def test_concurrent_threshold_just_outside(self):
        """Barely past concurrency window — still independent leaves, both culprits."""
        graph = get_dependency_graph()
        onsets = {"adservice": 100.0, "emailservice": 102.1}
        trends = {"adservice": "up", "emailservice": "up"}
        result = pinpoint_faults(onsets, trends, graph, concurrency_threshold_s=2.0)
        assert "adservice" in result
        assert "emailservice" in result


class TestSplitSeries:

    def test_standard_split(self):
        """1s spacing: 50 baseline + gap + 120 fault → expected slice lengths."""
        n = 180
        series = np.arange(n, dtype=float)
        bl_start = 1000.0
        bl_end = 1050.0
        ft_start = 1060.0
        ft_end = 1180.0

        bl, ft = _split_series(series, bl_start, (bl_start, bl_end), (ft_start, ft_end))
        assert len(bl) == 50
        assert len(ft) == 120
        np.testing.assert_array_equal(bl, np.arange(50, dtype=float))
        np.testing.assert_array_equal(ft, np.arange(60, 180, dtype=float))

    def test_no_gap(self):
        """Back-to-back windows."""
        series = np.arange(100, dtype=float)
        bl, ft = _split_series(series, 0.0, (0.0, 50.0), (50.0, 100.0))
        assert len(bl) == 50
        assert len(ft) == 50

    def test_series_shorter_than_window(self):
        """Truncated series — fault tail clips to what's available."""
        series = np.arange(30, dtype=float)
        bl, ft = _split_series(series, 0.0, (0.0, 25.0), (25.0, 100.0))
        assert len(bl) == 25
        assert len(ft) <= 5


class TestDetermineTrend:

    def test_all_up(self):
        assert _determine_trend(["up", "up", "up"]) == "up"

    def test_all_down(self):
        assert _determine_trend(["down", "down"]) == "down"

    def test_mixed(self):
        assert _determine_trend(["up", "down"]) == "mixed"

    def test_empty(self):
        assert _determine_trend([]) == "mixed"


class TestAnalyzeMetric:

    def test_clear_step_detects_change(self):
        """Obvious level shift → at least one onset index."""
        baseline = np.ones(10) * 5.0
        fault = np.concatenate([np.ones(5) * 5.0, np.ones(19) * 50.0])
        result = _analyze_metric(baseline, fault)
        assert result is not None
        # tuple: onsets, directions, confidences
        onsets, dirs, confs = result
        assert len(onsets) >= 1
        assert all(isinstance(o, (int, np.integer)) for o in onsets)

    def test_flat_no_change(self):
        """Flat fault segment → None."""
        baseline = np.ones(100) * 5.0
        fault = np.ones(24) * 5.0
        result = _analyze_metric(baseline, fault)
        assert result is None

    def test_constant_baseline_detects_any_shift(self):
        """Zero-variance baseline — any drift should pop."""
        baseline = np.ones(10) * 100.0
        fault = np.concatenate([np.ones(5) * 100.0, np.ones(19) * 200.0])
        result = _analyze_metric(baseline, fault)
        assert result is not None
        onsets, _, _ = result
        assert len(onsets) >= 1

    def test_noisy_baseline_small_fault(self):
        """Fault buried in baseline noise — at most tiny CP list."""
        rng = np.random.default_rng(123)
        baseline = rng.normal(10.0, 2.0, 100)
        fault = rng.normal(10.5, 2.0, 24)
        result = _analyze_metric(baseline, fault)
        if result is not None:
            onsets, _, _ = result
            assert len(onsets) <= 2
