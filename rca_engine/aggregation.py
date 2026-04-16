"""Multi-metric onset aggregation — Layer 5 of the FChain pipeline.

This is the final per-component step.  It collapses the per-metric
onset times produced by Layer 4 (tangent-based rollback) into a single
component-level onset timestamp.

Algorithm
---------
  component_onset = min(t for t in onset_times.values() if t is not None)

If no metric on the component shows any abnormal behavior the result is
None, and the FChain master excludes this component from the propagation
chain.

FChain monitors exactly 6 system-level metrics per VM:
  cpu, memory, net_in, net_out, disk_read, disk_write

Not all 6 need to be present.  Aggregation works on whatever subset is
available.

Public API
----------
  aggregate_component_onset(onset_per_metric)  -> int | None
  aggregate_all_components(onsets_per_component) -> dict[str, int | None]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants — single source of truth for metric names used across the pipeline
# ---------------------------------------------------------------------------

#: The 6 system-level metrics FChain monitors per VM (FChain paper Section III-A).
MONITORED_METRICS: tuple[str, ...] = (
    "cpu_rate", "cpu_throttle_ratio", "mem_wss",
    "net_rx_rate", "net_tx_rate", "fs_read_rate", "fs_write_rate"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_component_onset(
    onset_per_metric: dict[str, int | None],
) -> int | None:
    """Collapse per-metric onset times into a single component onset.

    Takes the minimum across all metrics that returned a valid (non-None)
    onset from Layer 4.  Returns None if no metric showed abnormal behavior.

    Parameters
    ----------
    onset_per_metric:
        Dict mapping metric name to its rollback-corrected onset index
        (Layer 4 output).  Metrics with no abnormal change point map to
        None.  Unknown metric names are accepted — the function operates
        on whatever keys are present.

    Returns
    -------
    int | None
        Minimum onset index across all valid metrics, or None if every
        metric returned None.

    Examples
    --------
    >>> aggregate_component_onset({'cpu': 45, 'memory': 30, 'net_in': None})
    30
    >>> aggregate_component_onset({'cpu': None, 'memory': None})
    None
    >>> aggregate_component_onset({})
    None
    """
    valid_onsets = [t for t in onset_per_metric.values() if t is not None]

    if not valid_onsets:
        return None

    return min(valid_onsets)


def aggregate_all_components(
    onsets_per_component: dict[str, dict[str, int | None]],
) -> dict[str, int | None]:
    """Run onset aggregation for every component in the application.

    Applies ``aggregate_component_onset`` to each component independently
    and returns a component-keyed dict of onset times.  This is the dict
    consumed by the FChain master (Section II-C) to build the propagation
    chain.

    Parameters
    ----------
    onsets_per_component:
        Nested dict:
          { component_id -> { metric_name -> onset_index | None } }

    Returns
    -------
    dict[str, int | None]
        { component_id -> component_onset | None }

        Components whose onset is None showed no abnormal behavior and
        will be excluded from the propagation chain by the master.

    Examples
    --------
    >>> aggregate_all_components({
    ...     'app_server_1': {'cpu': 200, 'memory': 210},
    ...     'db_server':    {'cpu': None, 'memory': None},
    ...     'web_server':   {'cpu': 215, 'memory': None},
    ... })
    {'app_server_1': 200, 'db_server': None, 'web_server': 215}
    """
    return {
        component_id: aggregate_component_onset(onset_per_metric)
        for component_id, onset_per_metric in onsets_per_component.items()
    }