"""Fault-chain integrated pinpointing algorithm (FChain Sections II-B/C).

Implements the full RCA pipeline described in the FChain paper:

  * Section II-B — per-service, per-metric abnormal change point selection:
      Layer 1  CUSUM + Bootstrap        change-point candidates
      Layer 2  Markov prediction model  prediction-error filtering
      Layer 3  FFT burst threshold      burst-aware abnormality filter
      Layer 4  Tangent rollback         onset-time refinement
      Layer 5  Multi-metric aggregation earliest onset across metrics

  * Section II-C — integrated fault diagnosis across services:
      Layer 6  Propagation chain        sort services by onset time
      Layer 7  Root cause candidates    earliest onset + concurrency check
      Layer 8  Dependency filter        remove spurious propagation paths

Typical usage from ``eval/run_experiment.py``::

    from rca_engine import fault_chain

    ranked = fault_chain.pinpoint(
        metric_matrix,
        baseline_window=(bl_start, bl_end),
        fault_window=(ft_start, ft_end),
        step_seconds=5.0,
    )

``ranked`` is a list of dicts ordered by FChain Section II-C ranking:
root causes first (earliest onset), then downstream propagation victims.
Each entry contains ``service``, ``onset_time``, ``confidence``,
``abnormal_metrics``, ``is_root_cause``, and ``rank``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rca_engine.change_point import run_layer1
from rca_engine.dependency import get_dependency_graph, has_path
from rca_engine.normal_model import NormalModel
from rca_engine.predictability_filter import filter_abnormal_change_points
from rca_engine.tangent_rollback import rollback_onset
from rca_engine.aggregation import MONITORED_METRICS

logger = logging.getLogger(__name__)

# Number of metric types tracked per service.
# Used as the denominator when computing fallback per-service confidence.
TOTAL_METRICS: int = len(MONITORED_METRICS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def pinpoint(
    metric_matrix: dict[str, dict[str, np.ndarray]],
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    step_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Run the full FChain RCA pipeline and return a ranked suspect list.

    For each service in *metric_matrix*, every metric is analysed through
    Layers 1-5 (Section II-B) to produce a rollback-refined onset timestamp.
    The per-service onset times are then passed to Layers 6-8 (Section II-C)
    which build the propagation chain, identify root causes, and filter out
    services whose abnormality is explained by downstream propagation.

    Parameters
    ----------
    metric_matrix :
        ``{service: {metric: np.ndarray}}`` covering the full time range
        ``[baseline_start, fault_end]`` sampled at *step_seconds* intervals.
        Arrays must be aligned: index 0 corresponds to *baseline_window[0]*.
    baseline_window :
        ``(start_posix, end_posix)`` of the clean baseline period used to
        learn normal behaviour for each metric.
    fault_window :
        ``(start_posix, end_posix)`` of the fault period to analyse.
        The look-back window for change-point detection is drawn from here.
    step_seconds :
        Prometheus scrape step in seconds.  Must match the resolution at
        which *metric_matrix* was built.  Default 1.0.

    Returns
    -------
    list[dict]
        One entry per abnormal service, ordered by FChain Section II-C
        ranking: root causes first (by onset time), then propagation victims
        (by onset time).  Each dict contains:

        ``service``          service name
        ``onset_time``       POSIX timestamp of the earliest detected onset
        ``confidence``       fraction of monitored metrics that were abnormal
        ``abnormal_metrics`` sorted list of metrics that showed abnormality
        ``is_root_cause``    True if Section II-C identifies this as a fault
        ``rank``             1-based position in the output list
    """
    if not metric_matrix:
        return []

    bl_start, bl_end = baseline_window
    ft_start, ft_end = fault_window
    full_start = bl_start  # metric arrays are aligned to baseline start

    # Total services being monitored — used by the external cause check in
    # Layer 6 to decide whether all services became abnormal simultaneously.
    n_monitored_services = len(metric_matrix)

    # ------------------------------------------------------------------
    # Layers 1-5: per-service, per-metric change-point analysis
    # ------------------------------------------------------------------
    # For each service we collect:
    #   service_onsets            — earliest rollback-refined onset (POSIX)
    #   service_trends            — overall metric trend ('up'/'down'/'mixed')
    #   service_abnormal_metrics  — metrics that showed abnormal behaviour
    #   service_metric_confidences— per-metric maximum confidence score
    service_onsets: dict[str, float] = {}
    service_trends: dict[str, str] = {}
    service_abnormal_metrics: dict[str, list[str]] = {}
    service_metric_confidences: dict[str, dict[str, float]] = {}

    for service, metrics in metric_matrix.items():
        earliest_onset: float | None = None
        all_directions: list[str] = []
        abnormal_metric_names: list[str] = []
        metric_confs: dict[str, float] = {}

        for metric_name, full_series in metrics.items():
            baseline_data, fault_data = _split_series(
                full_series, full_start, baseline_window, fault_window,
                step=step_seconds,
            )
            if len(fault_data) < 3 or len(baseline_data) < 2:
                continue

            metric_analysis = _analyze_metric(baseline_data, fault_data)
            if metric_analysis is None:
                continue

            onset_indices, directions, confidences = metric_analysis
            abnormal_metric_names.append(metric_name)
            all_directions.extend(directions)
            if confidences:
                metric_confs[metric_name] = max(confidences)

            # Convert the earliest onset index into a POSIX timestamp and
            # track the minimum across all metrics for this service (Layer 5).
            for idx in onset_indices:
                ts = ft_start + idx * step_seconds
                if earliest_onset is None or ts < earliest_onset:
                    earliest_onset = ts

        if earliest_onset is not None:
            service_onsets[service] = earliest_onset
            service_trends[service] = _determine_trend(all_directions)
            service_abnormal_metrics[service] = abnormal_metric_names
            service_metric_confidences[service] = metric_confs

    if not service_onsets:
        return []

    # ------------------------------------------------------------------
    # Layers 6-8: integrated fault diagnosis (FChain Section II-C)
    # ------------------------------------------------------------------
    dep_graph = get_dependency_graph()
    pinpointed = pinpoint_faults(
        service_onsets,
        service_trends,
        dep_graph,
        n_monitored_services=n_monitored_services,
        concurrency_threshold_s=2.0,
    )

    pinpointed_set = set(pinpointed)

    # ------------------------------------------------------------------
    # Rank and format output
    # ------------------------------------------------------------------
    # Root causes are ordered before propagation victims; within each
    # group services are sorted by onset time (earliest first).
    def _sort_key(svc: str) -> tuple:
        return (
            0 if svc in pinpointed_set else 1,
            service_onsets[svc],
        )

    sorted_services = sorted(service_onsets, key=_sort_key)
    ranked_results: list[dict[str, Any]] = []

    for i, svc in enumerate(sorted_services, 1):
        entry = _make_entry(
            svc,
            service_onsets[svc],
            service_abnormal_metrics[svc],
            is_root_cause=(svc in pinpointed_set),
            metric_confidences=service_metric_confidences.get(svc, {}),
        )
        entry["rank"] = i
        ranked_results.append(entry)

    return ranked_results


# ---------------------------------------------------------------------------
# Pinpointing logic (FChain Section II-C)
# ---------------------------------------------------------------------------

def pinpoint_faults(
    service_onsets: dict[str, float],
    service_trends: dict[str, str],
    dependency_graph: dict[str, list[str]],
    n_monitored_services: int = 0,
    concurrency_threshold_s: float = 2.0,
) -> list[str]:
    """Identify faulty services using the FChain Section II-C algorithm.

    Runs three sequential layers:

    Layer 6 — Build propagation chain
        Sort all abnormal services by their rollback-refined onset time.
        If every monitored service is abnormal and all share the same
        metric trend direction (all rising or all falling), the anomaly
        is attributed to an external cause (e.g. workload spike or shared
        infrastructure failure) and an empty list is returned — no
        individual service is pinpointed.

    Layer 7 — Identify root cause candidates
        The service with the earliest onset is the primary root cause.
        Any subsequent service whose onset falls within
        *concurrency_threshold_s* of the primary is treated as a
        concurrent independent fault and is also pinpointed.  The first
        service that started too late to be concurrent breaks the scan —
        all later services are assumed to be downstream victims.

    Layer 8 — Filter spurious propagation paths
        For every remaining abnormal service ``C``, check whether its
        abnormality can be explained by propagation from an already-
        pinpointed root cause ``R``.  Two propagation directions are
        considered:

        * Forward (``R → C``): the fault propagated downstream from R to C
          along a normal dependency edge.
        * Back-pressure (``C → R``): a faulty C caused its upstream
          caller R to stall, making R appear abnormal even though it is
          not the root cause.

        If neither path exists, C cannot be a propagation victim and must
        have an independent fault; it is added to the pinpointed list.

    Parameters
    ----------
    service_onsets :
        Rollback-refined onset timestamps (POSIX seconds) keyed by service.
        Only services that showed abnormal behaviour should be present.
    service_trends :
        Per-service overall metric trend direction: ``'up'``, ``'down'``,
        or ``'mixed'``.  Used for the external cause check in Layer 6.
    dependency_graph :
        Directed adjacency list ``{service: [services it calls]}``.
    n_monitored_services :
        Total number of services under observation in this experiment run.
        Pass ``0`` to skip the external cause check.
    concurrency_threshold_s :
        Maximum onset-time gap in seconds for two services to be
        classified as concurrent independent faults.  FChain default: 2.0.

    Returns
    -------
    list[str]
        Services identified as root causes, ordered by onset time.
        Returns ``[]`` if an external cause is detected or no services
        are abnormal.
    """
    if not service_onsets:
        return []

    # Layer 6 — build propagation chain by sorting on onset time
    sorted_svcs = sorted(service_onsets, key=lambda s: service_onsets[s])

    # External cause check: every monitored service is abnormal AND all
    # metric changes follow the same directional trend.  A uniform upward
    # trend suggests a workload spike; uniform downward suggests a shared
    # resource collapse (e.g. NFS).  In either case no single service is
    # at fault, so we return early.
    if n_monitored_services > 0:
        unique_trends = set(service_trends.values()) - {"mixed"}
        all_services_abnormal = len(sorted_svcs) >= n_monitored_services
        if len(unique_trends) == 1 and all_services_abnormal:
            logger.info(
                "External cause detected — all %d services abnormal "
                "with uniform trend '%s'",
                len(sorted_svcs),
                next(iter(unique_trends)),
            )
            return []

    # Layer 7 — primary root cause is the earliest-onset service.
    # Any service that started within the concurrency window is also
    # pinpointed as an independent concurrent fault rather than a victim.
    pinpointed: list[str] = [sorted_svcs[0]]
    first_onset = service_onsets[sorted_svcs[0]]

    for svc in sorted_svcs[1:]:
        onset_diff = service_onsets[svc] - first_onset
        if onset_diff <= concurrency_threshold_s:
            # Started close enough that propagation cannot explain the gap.
            pinpointed.append(svc)
        else:
            # This service started late enough to be a downstream victim.
            # All subsequent services started even later, so stop scanning.
            break

    # Layer 8 — for every remaining abnormal service, determine whether
    # its abnormality is reachable from any already-pinpointed root cause
    # via forward propagation or back-pressure.  If not, it is an
    # independent fault and is added to the pinpointed set.
    for svc in sorted_svcs:
        if svc in pinpointed:
            continue

        is_propagation_victim = any(
            has_path(dependency_graph, root, svc)   # forward propagation
            or has_path(dependency_graph, svc, root) # back-pressure
            for root in pinpointed
        )

        if not is_propagation_victim:
            pinpointed.append(svc)

    return pinpointed


# ---------------------------------------------------------------------------
# Per-metric analysis pipeline
# ---------------------------------------------------------------------------

def _analyze_metric(
    baseline_data: np.ndarray,
    fault_data: np.ndarray,
    num_bins: int = 40,
    cusum_k_factor: float = 0.5,
    fft_Q: int = 20,
) -> tuple[list[int], list[str], list[float]] | None:
    """Run Layers 1-4 for a single metric of a single service.

    Applies the Section II-B pipeline in sequence:

    1. Fit a Markov-chain NormalModel on *baseline_data* so that the
       model captures the metric's normal transition patterns before any
       fault was introduced (Layer 2 setup).

    2. Run CUSUM + Bootstrap on *fault_data* to find all candidate
       change points (Layer 1).  If CUSUM detects nothing, return None.

    3. Compute Markov prediction errors at every index where the CUSUM
       score crosses the bootstrap threshold.  A high prediction error
       means the model was genuinely surprised — the change was not
       seen during normal operation (Layer 2).

    4. Apply the FFT burst-threshold filter to the prediction errors.
       For each crossing, compare the error against the 90th-percentile
       magnitude of the local high-frequency burst signal.  Only crossings
       whose error exceeds that threshold survive (Layer 3).  If none
       survive, return None.

    5. Walk each surviving crossing back through the full Layer-1 change-
       point list using tangent-based rollback to find the true start of
       the abnormal slope rather than the point where CUSUM first alarmed
       (Layer 4).

    Parameters
    ----------
    baseline_data :
        Clean metric samples from the baseline window.  Used to fit the
        NormalModel and compute CUSUM baseline statistics.
    fault_data :
        Metric samples from the fault window.  Change-point detection
        and filtering operate on this array; returned indices index it.
    num_bins :
        Number of equal-width discretisation bins for the Markov model.
        40 bins per the PRESS paper specification.
    cusum_k_factor :
        CUSUM reference value *k*.  A value of 0.5 targets shifts of
        approximately 1 standard deviation (k = Δ/2).
    fft_Q :
        Half-window size (samples) for the FFT burst-threshold filter.
        The local window around each crossing spans [t-Q, t+Q].

    Returns
    -------
    tuple[list[int], list[str], list[float]] | None
        ``(onset_indices, directions, confidences)`` where every index
        is into *fault_data*.  ``directions`` is ``'up'`` or ``'down'``
        per change point.  ``confidences`` are bootstrap confidence
        scores in [0, 1].  Returns ``None`` if no abnormal change points
        survive the full pipeline.
    """
    # Layer 2 setup — fit the Markov model on clean baseline behaviour.
    # Bin edges are fixed to [lo, hi] derived from the baseline so that
    # fault-period values outside this range land in the boundary bins
    # and are still handled without index errors.
    lo = float(np.min(baseline_data))
    hi = float(np.max(baseline_data))
    if hi == lo:
        hi = lo + 1e-6   # prevent degenerate single-bin model

    model = NormalModel(
        num_bins=num_bins,
        metric_min=lo,
        metric_max=hi,
    ).fit(baseline_data)

    # Layer 1 — CUSUM + Bootstrap change-point detection on fault window.
    result = run_layer1(
        time_series=fault_data,
        baseline_data=baseline_data,
        k=cusum_k_factor,
    )
    if not result.change_points:
        return None

    # Layer 2 — prediction errors at CUSUM threshold-crossing indices.
    # The CUSUM score g crosses the bootstrap threshold h at the indices
    # where the detector first accumulated enough evidence of a change.
    # We compute the Markov prediction error at each such index: a high
    # error confirms the change was genuinely novel, not a known pattern.
    g = result.cusum_combined
    h = result.bootstrap_threshold
    cusum_crossings = np.where(g >= h)[0].tolist()

    pred_errors_dict = model.prediction_errors_for(cusum_crossings, fault_data)

    # Layer 3 — FFT burst-threshold filter.
    # For each crossing, estimate the local burstiness of the metric using
    # FFT (top 90% frequencies by index).  A crossing is abnormal only if
    # its prediction error exceeds the 90th-percentile burst magnitude,
    # ensuring that naturally bursty metrics are not over-flagged.
    abnormal_crossings = filter_abnormal_change_points(
        fault_data, pred_errors_dict, Q=fft_Q,
    )
    if not abnormal_crossings:
        return None

    # Layer 4 — tangent-based rollback.
    # Each surviving crossing was where CUSUM alarmed, but the true onset
    # of the abnormal behaviour may be earlier on the same slope.  Walk
    # backward through the full Layer-1 change-point list comparing
    # adjacent tangents; stop when the slope changes direction.
    all_layer1_cps = result.change_points

    onset_indices: list[int] = []
    directions: list[str] = []
    confidences: list[float] = []

    for cp in abnormal_crossings:
        refined_onset = rollback_onset(
            series=fault_data,
            abnormal_cp=cp,
            all_change_points=all_layer1_cps,
        )
        onset_indices.append(refined_onset)

        # Attach the direction and confidence of the nearest Layer-1
        # change point to the refined onset for use in trend aggregation
        # and output confidence scoring.
        closest_l1_cp = min(
            all_layer1_cps,
            key=lambda c: abs(c - cp),
            default=cp,
        )
        directions.append(result.directions[
            all_layer1_cps.index(closest_l1_cp)
        ] if closest_l1_cp in all_layer1_cps else "up")
        confidences.append(
            result.confidence_scores.get(closest_l1_cp, 0.0)
        )

    if not onset_indices:
        return None

    return onset_indices, directions, confidences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_series(
    series: np.ndarray,
    full_start: float,
    baseline_window: tuple[float, float],
    fault_window: tuple[float, float],
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a full-range time series into baseline and fault sub-arrays.

    Parameters
    ----------
    series :
        Full metric array aligned to *full_start* at *step*-second intervals.
    full_start :
        POSIX timestamp corresponding to ``series[0]``.
    baseline_window :
        ``(start, end)`` POSIX timestamps for the baseline slice.
    fault_window :
        ``(start, end)`` POSIX timestamps for the fault slice.
    step :
        Sample interval in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(baseline_data, fault_data)`` as contiguous views of *series*.
    """
    bl_start_idx = max(0, round((baseline_window[0] - full_start) / step))
    bl_end_idx   = round((baseline_window[1] - full_start) / step)
    ft_start_idx = round((fault_window[0] - full_start) / step)
    ft_end_idx   = min(len(series), round((fault_window[1] - full_start) / step))

    baseline = series[bl_start_idx:bl_end_idx]
    fault    = series[ft_start_idx:ft_end_idx]
    return baseline, fault


def _determine_trend(directions: list[str]) -> str:
    """Collapse a list of per-change-point directions into a single trend.

    Returns ``'up'`` if all directions are upward, ``'down'`` if all are
    downward, and ``'mixed'`` otherwise (including the empty list).
    """
    if not directions:
        return "mixed"
    ups   = directions.count("up")
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
    """Construct a ranked result dict for one service.

    Confidence is the mean of per-metric confidence scores when available,
    or the fraction of monitored metrics that were abnormal as a fallback.
    """
    if metric_confidences and abnormal_metrics:
        confidence = (
            sum(metric_confidences.get(m, 0.0) for m in abnormal_metrics)
            / len(abnormal_metrics)
        )
    else:
        # Fallback: treat each abnormal metric as equally weighted evidence.
        confidence = len(abnormal_metrics) / TOTAL_METRICS

    return {
        "service":          service,
        "onset_time":       onset_time,
        "confidence":       round(confidence, 3),
        "abnormal_metrics": sorted(abnormal_metrics),
        "is_root_cause":    is_root_cause,
    }