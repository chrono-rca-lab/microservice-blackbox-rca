"""Fault-chain integrated pinpointing algorithm.

Entry point called by ``eval/run_experiment.py``::

    from rca_engine import fault_chain
    ranked = fault_chain.pinpoint(metric_matrix, baseline_window, fault_window)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rca_engine.change_point import detect_change_points_bootstrap
from rca_engine.dependency import get_dependency_graph, has_path
from rca_engine.normal_model import NormalModel
from rca_engine.predictability_filter import filter_abnormal_change_points
from rca_engine.tangent_rollback import rollback_onset

logger = logging.getLogger(__name__)

STEP_SECONDS = 1.0  # Prometheus scrape step
TOTAL_METRICS = 7    # number of metric types we track


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def pinpoint(
    metric_matrix: dict[str, dict[str, np.ndarray]],
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
) -> list[dict[str, Any]]:
    """Run the full FChain RCA pipeline.

    Parameters
    ----------
    metric_matrix :
        ``{service: {metric: np.ndarray}}`` spanning the full time range
        ``[baseline_start, fault_end]`` at ``STEP_SECONDS`` intervals.
    baseline_window :
        ``(start_posix, end_posix)`` of the normal baseline period.
    fault_window :
        ``(start_posix, end_posix)`` of the fault period.

    Returns
    -------
    list[dict]
        Ranked suspects ordered per FChain Section II-C: root causes first
        (by onset time), then propagation victims.
    """
    if not metric_matrix:
        return []

    bl_start, bl_end = baseline_window
    ft_start, ft_end = fault_window
    full_start = bl_start  # arrays start at baseline_start

    # ---- Per-service, per-metric analysis --------------------------------
    # Two timestamps per service:
    #   detection_time — earliest CUSUM trigger (for concurrency grouping)
    #   onset_time     — earliest rollback-refined onset (for ranking)
    service_detection: dict[str, float] = {}
    service_onsets: dict[str, float] = {}
    service_trends: dict[str, str] = {}
    service_abnormal_metrics: dict[str, list[str]] = {}
    service_metric_confidences: dict[str, dict[str, float]] = {}

    for service, metrics in metric_matrix.items():
        earliest_detection: float | None = None
        earliest_onset: float | None = None
        all_directions: list[str] = []
        abnormal_metric_names: list[str] = []
        metric_confs: dict[str, float] = {}

        for metric_name, full_series in metrics.items():
            baseline_data, fault_data = _split_series(
                full_series, full_start, baseline_window, fault_window,
            )
            if len(fault_data) < 3 or len(baseline_data) < 2:
                continue

            result = _analyze_metric(baseline_data, fault_data)
            if result is None:
                continue

            cusum_indices, onset_indices, directions, confidences = result
            abnormal_metric_names.append(metric_name)
            all_directions.extend(directions)
            if confidences:
                metric_confs[metric_name] = max(confidences)

            # CUSUM detection times (for concurrency grouping)
            for idx in cusum_indices:
                ts = ft_start + idx * STEP_SECONDS
                if earliest_detection is None or ts < earliest_detection:
                    earliest_detection = ts

            # Refined onset times (for ranking)
            for idx in onset_indices:
                ts = ft_start + idx * STEP_SECONDS
                if earliest_onset is None or ts < earliest_onset:
                    earliest_onset = ts

        if earliest_detection is not None:
            service_detection[service] = earliest_detection
            service_onsets[service] = earliest_onset or earliest_detection
            service_trends[service] = _determine_trend(all_directions)
            service_abnormal_metrics[service] = abnormal_metric_names
            service_metric_confidences[service] = metric_confs

    if not service_onsets:
        return []

    # ---- Integrated pinpointing (FChain Section II-C) --------------------
    dep_graph = get_dependency_graph()
    pinpointed = pinpoint_faults(
        service_detection, service_trends, dep_graph,
    )

    pinpointed_set = set(pinpointed)

    # ---- Format output (FChain ranking: onset-time order) ----------------
    # Root causes first by onset, then propagation victims by onset.
    def _sort_key(svc: str) -> tuple:
        is_rc = svc in pinpointed_set
        return (
            0 if is_rc else 1,
            service_onsets[svc],
        )

    sorted_services = sorted(service_onsets, key=_sort_key)
    result: list[dict[str, Any]] = []

    for i, svc in enumerate(sorted_services, 1):
        entry = _make_entry(
            svc,
            service_onsets[svc],
            service_abnormal_metrics[svc],
            is_root_cause=(svc in pinpointed_set),
            metric_confidences=service_metric_confidences.get(svc, {}),
        )
        entry["rank"] = i
        result.append(entry)

    return result


# -----------------------------------------------------------------------
# Pinpointing logic (FChain Section II-C)
# -----------------------------------------------------------------------

def pinpoint_faults(
    service_onsets: dict[str, float],
    service_trends: dict[str, str],
    dependency_graph: dict[str, list[str]],
    concurrency_threshold_s: float = 2.0,
) -> list[str]:
    """Integrated faulty component pinpointing.

    1. External cause check — if ALL services are abnormal with the same
       trend, flag as external and return ``[]``.
    2. Sort services by onset; pinpoint the earliest.
    3. Concurrent faults — services within *concurrency_threshold_s* of the
       earliest are also pinpointed.
    4. Dependency filter — remaining abnormal services reachable from a
       pinpointed service are propagation (not root causes).
    """
    if not service_onsets:
        return []

    sorted_svcs = sorted(service_onsets, key=lambda s: service_onsets[s])

    # -- External cause check --
    unique_trends = set(service_trends.values()) - {"mixed"}
    if len(unique_trends) == 1 and len(sorted_svcs) >= len(dependency_graph):
        logger.info("External cause detected — all services abnormal with uniform trend")
        return []

    # -- Pinpoint earliest + concurrent --
    pinpointed: list[str] = [sorted_svcs[0]]
    first_onset = service_onsets[sorted_svcs[0]]
    for svc in sorted_svcs[1:]:
        if service_onsets[svc] - first_onset <= concurrency_threshold_s:
            pinpointed.append(svc)
        else:
            break

    # -- Dependency filter for remaining services --
    for svc in sorted_svcs:
        if svc in pinpointed:
            continue
        # svc is propagation if it *calls* (depends on) a pinpointed service
        is_propagation = any(
            has_path(dependency_graph, svc, p) for p in pinpointed
        )
        if not is_propagation:
            pinpointed.append(svc)

    return pinpointed


# -----------------------------------------------------------------------
# Per-metric analysis pipeline
# -----------------------------------------------------------------------

def _analyze_metric(
    baseline_data: np.ndarray,
    fault_data: np.ndarray,
    num_bins: int = 100,
    cusum_k_factor: float = 0.5,
    fft_Q: int = 20,
) -> tuple[list[int], list[int], list[str], list[float]] | None:
    """NormalModel -> CUSUM -> FFT filter -> tangent rollback.

    Returns ``(cusum_indices, refined_onset_indices, directions, confidences)`` where
    indices are into *fault_data*.  Returns ``None`` if no abnormal change
    points are found.
    """
    # 1. Fit normal model on baseline, compute prediction errors on fault data
    model = NormalModel(num_bins=num_bins).fit(baseline_data)
    pred_errors = model.prediction_errors(fault_data)

    # 2. Baseline statistics for CUSUM
    mu_0 = float(np.mean(baseline_data))
    sigma_0 = float(np.std(baseline_data, ddof=1)) if len(baseline_data) > 1 else 0.0

    # Floor sigma to avoid trivial triggers on constant/near-constant baselines.
    # Use 1% of |mean| as a minimum; for near-zero baselines, require the
    # change to exceed 1% of the combined data range before triggering.
    combined_range = float(np.ptp(np.concatenate([baseline_data, fault_data])))
    sigma_floor = max(abs(mu_0) * 0.01, combined_range * 0.01, 1e-6)
    # 3. CUSUM change-point detection
    result = detect_change_points_bootstrap(
        time_series=fault_data,
        baseline_data=baseline_data,
        k=cusum_k_factor,
    )
    if not result.change_points:
        return None

    # 4. FFT predictability filter on CUSUM candidate alarms
    g = result.cusum_scores
    h = result.bootstrap_threshold
    crossings = np.where(g >= h)[0].tolist()

    abnormal_crossings = filter_abnormal_change_points(
        fault_data, crossings, pred_errors, Q=fft_Q,
    )
    if not abnormal_crossings:
        return None

    # 5. Validate the bootstrap onsets using the abnormal alarms
    surviving_onsets = set()
    for cp in abnormal_crossings:
        # Walk back to the last reset of the CUSUM trajectory to find the onset
        j = cp - 1
        while j >= 0 and g[j] > 0:
            j -= 1
        onset = max(0, j + 1)
        surviving_onsets.add(onset)

    merged_onsets = []
    merged_dirs = []
    merged_confs = []
    for cp, d in zip(result.change_points, result.directions):
        # A change point survives if its onset was verified by an abnormal alarm
        if any(abs(cp - s_onset) <= 2 for s_onset in surviving_onsets):
            merged_onsets.append(cp)
            merged_dirs.append(d)
            merged_confs.append(result.confidence_scores.get(cp, 0.0))

    if not merged_onsets:
        return None

    return abnormal_crossings, merged_onsets, merged_dirs, merged_confs


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _split_series(
    series: np.ndarray,
    full_start: float,
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    step: float = STEP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a full-range series into baseline and fault portions."""
    bl_start_idx = max(0, round((baseline_window[0] - full_start) / step))
    bl_end_idx = round((baseline_window[1] - full_start) / step)
    ft_start_idx = round((fault_window[0] - full_start) / step)
    ft_end_idx = min(len(series), round((fault_window[1] - full_start) / step))

    baseline = series[bl_start_idx:bl_end_idx]
    fault = series[ft_start_idx:ft_end_idx]
    return baseline, fault


def _determine_trend(directions: list[str]) -> str:
    """Overall trend from a list of per-change-point directions."""
    if not directions:
        return "mixed"
    ups = directions.count("up")
    downs = directions.count("down")
    if ups > 0 and downs == 0:
        return "up"
    if downs > 0 and ups == 0:
        return "down"
    return "mixed"


def _make_entry(
    service: str,
    onset_time: float,
    abnormal_metrics: list[str],
    is_root_cause: bool,
    metric_confidences: dict[str, float] | None = None,
) -> dict[str, Any]:
    if metric_confidences and abnormal_metrics:
        confidence = sum(metric_confidences.get(m, 0.0) for m in abnormal_metrics) / len(abnormal_metrics)
    else:
        confidence = len(abnormal_metrics) / TOTAL_METRICS
    return {
        "service": service,
        "onset_time": onset_time,
        "confidence": round(confidence, 3),
        "abnormal_metrics": sorted(abnormal_metrics),
        "is_root_cause": is_root_cause,
    }
