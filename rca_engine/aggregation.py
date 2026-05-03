"""Layer 5 — squash per-metric onsets down to one per service/pod.

Roll-back (layer 4) gives you an onset per metric when something looks wrong.
Here we take the earliest of those; if none fire, there's nothing for the
later cross-service ranking to chew on.

We track seven infra-style metrics everywhere else in this repo:

"cpu_rate", "cpu_throttle_ratio", "mem_wss",
"net_rx_rate", "net_tx_rate", "fs_read_rate", "fs_write_rate"

You don't need all seven in the dict; min over whatever showed up works.

Exports
-------
  aggregate_component_onset(...)
  aggregate_all_components(...)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants — single source of truth for metric names used across the pipeline
# ---------------------------------------------------------------------------

# Canonical metric names for this stack (seven tracked in Boutique runs).
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
    and returns a component-keyed dict of onset times for the cross-service
    stage in ``fault_chain``.

    Parameters
    ----------
    onsets_per_component:
        Nested dict:
          { component_id -> { metric_name -> onset_index | None } }

    Returns
    -------
    dict[str, int | None]
        { component_id -> component_onset | None }

        Components whose onset is None never looked abnormal here, so
        ``fault_chain`` has nothing to rank for them.

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