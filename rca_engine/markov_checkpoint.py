"""Markov chain checkpoint management for Layer 2.

A checkpoint captures a fully trained Markov transition model for one
(service, metric, time-window) combination.  Checkpoints are trained
once from baseline Prometheus data and reused across all experiments.

Checkpoint windows
------------------
Six windows are supported, each representing how much baseline data
was used to train the model:

    5 minutes  =   300 seconds
    30 minutes =  1800 seconds
    1 hour     =  3600 seconds
    2 hours    =  7200 seconds
    3 hours    = 10800 seconds
    4 hours    = 14400 seconds

At prediction time the window whose duration best fits the available
baseline data is selected automatically.

File layout
-----------
checkpoints/markov/{service}/{metric}/
    window_300s.npz
    window_1800s.npz
    window_3600s.npz
    window_7200s.npz
    window_10800s.npz
    window_14400s.npz
    manifest.json

Each .npz stores three arrays (counts, transition, bin_edges) plus
scalar metadata as zero-dimensional arrays.  No pickle is used.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/markov")
_DEFAULT_CONFIG_PATH     = Path("checkpoint_config.json")

# Fallback windows used when no config file is found.
_FALLBACK_WINDOWS: tuple[int, ...] = (
    300, 1800, 3600, 5400, 7200, 10800, 14400
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path | str = _DEFAULT_CONFIG_PATH) -> dict:
    """Load checkpoint_config.json.

    Returns the parsed config dict, or a default config if the file is
    not found.  The config controls window durations, num_bins, and
    directory paths.
    """
    config_path = Path(config_path)
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    logger.warning(
        "Config file not found at %s — using default windows.", config_path
    )
    return {
        "windows": [
            {"window_seconds": w, "label": _window_label(w)}
            for w in _FALLBACK_WINDOWS
        ],
        "num_bins": 100,
        "checkpoint_dir":  str(_DEFAULT_CHECKPOINT_ROOT),
        "prometheus_url":  "http://localhost:9090",
        "step_seconds":    1,
    }


def get_window_seconds(
    config_path: Path | str = _DEFAULT_CONFIG_PATH,
) -> tuple[int, ...]:
    """Return checkpoint window durations in seconds, sorted ascending."""
    config = load_config(config_path)
    return tuple(
        sorted(int(w["window_seconds"]) for w in config["windows"])
    )


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarkovCheckpoint:
    """A fully trained Markov transition model for one (service, metric, window).

    Attributes
    ----------
    counts : np.ndarray, shape (num_bins, num_bins)
        Raw transition count matrix accumulated during training.
    transition : np.ndarray, shape (num_bins, num_bins)
        Row-normalised probability matrix derived from counts.
    bin_edges : np.ndarray, shape (num_bins + 1,)
        Fixed bin boundaries used to discretise metric values.
    num_bins : int
        Number of equal-width bins.
    metric_min : float
        Lower bound of the metric range covered by bin_edges.
    metric_max : float
        Upper bound of the metric range covered by bin_edges.
    metric_range : float
        metric_max - metric_min.  Used as the unseen-state error signal.
    window_seconds : int
        Duration of the baseline window used for training.
    service : str
        Service name this checkpoint belongs to.
    metric_name : str
        Metric name this checkpoint belongs to.
    trained_at : str
        ISO-8601 UTC timestamp of when training was performed.
    n_samples : int
        Number of baseline samples consumed during training.
    """

    counts:         np.ndarray
    transition:     np.ndarray
    bin_edges:      np.ndarray
    num_bins:       int
    metric_min:     float
    metric_max:     float
    metric_range:   float
    window_seconds: int
    service:        str        = ""
    metric_name:    str        = ""
    trained_at:     str        = ""
    n_samples:      int        = 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_checkpoint(
    baseline_data: np.ndarray,
    metric_min: float,
    metric_max: float,
    num_bins: int = 100,
    window_seconds: int = 0,
    service: str = "",
    metric_name: str = "",
) -> MarkovCheckpoint:
    """Train a Markov checkpoint from a baseline time series.

    Parameters
    ----------
    baseline_data :
        1-D array of metric values representing normal system behavior.
        Must contain at least 2 samples.
    metric_min, metric_max :
        Range used to define bin edges.  Should cover the full observed
        range including fault-period values so the unseen-state signal
        is correctly sized.
    num_bins :
        Number of equal-width discretisation bins.  Default 100 (PRESS paper).
    window_seconds :
        Duration label in seconds.  One of the windows defined in checkpoint_config.json.
        Used only for metadata — does not affect training.
    service, metric_name :
        Labels stored in the checkpoint metadata.

    Returns
    -------
    MarkovCheckpoint
    """
    data = np.asarray(baseline_data, dtype=float).ravel()
    if len(data) < 2:
        raise ValueError(
            f"baseline_data must have at least 2 samples, got {len(data)}"
        )

    if metric_max <= metric_min:
        raise ValueError(
            f"metric_max ({metric_max}) must be > metric_min ({metric_min})"
        )

    # Fixed bin edges
    bin_edges = np.linspace(metric_min, metric_max, num_bins + 1)

    # Discretise baseline
    bins = _discretize_array(data, bin_edges, num_bins)

    # Build count matrix
    counts = np.zeros((num_bins, num_bins), dtype=np.float64)
    for a, b in zip(bins[:-1], bins[1:]):
        counts[a, b] += 1.0

    # Row-normalise; unseen rows get uniform distribution
    transition = _normalise(counts, num_bins)

    return MarkovCheckpoint(
        counts         = counts,
        transition     = transition,
        bin_edges      = bin_edges,
        num_bins       = num_bins,
        metric_min     = float(metric_min),
        metric_max     = float(metric_max),
        metric_range   = float(metric_max - metric_min),
        window_seconds = window_seconds,
        service        = service,
        metric_name    = metric_name,
        trained_at     = datetime.now(timezone.utc).isoformat(),
        n_samples      = len(data),
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint: MarkovCheckpoint, path: Path | str) -> None:
    """Save a checkpoint to disk as a .npz file.

    Arrays are stored natively in the .npz.  Scalar metadata is stored
    as zero-dimensional numpy arrays so the file needs no pickle.

    Parameters
    ----------
    checkpoint :
        The trained checkpoint to save.
    path :
        Destination file path.  The .npz extension is added if absent.
        Parent directories are created automatically.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        counts         = checkpoint.counts,
        transition     = checkpoint.transition,
        bin_edges      = checkpoint.bin_edges,
        num_bins       = np.array(checkpoint.num_bins),
        metric_min     = np.array(checkpoint.metric_min),
        metric_max     = np.array(checkpoint.metric_max),
        metric_range   = np.array(checkpoint.metric_range),
        window_seconds = np.array(checkpoint.window_seconds),
        n_samples      = np.array(checkpoint.n_samples),
        service        = np.array(checkpoint.service),
        metric_name    = np.array(checkpoint.metric_name),
        trained_at     = np.array(checkpoint.trained_at),
    )
    logger.debug("Saved checkpoint: %s", path)


def load_checkpoint(path: Path | str) -> MarkovCheckpoint:
    """Load a checkpoint from a .npz file.

    Parameters
    ----------
    path :
        Path to the .npz checkpoint file.

    Returns
    -------
    MarkovCheckpoint

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = np.load(path, allow_pickle=False)

    return MarkovCheckpoint(
        counts         = data["counts"],
        transition     = data["transition"],
        bin_edges      = data["bin_edges"],
        num_bins       = int(data["num_bins"]),
        metric_min     = float(data["metric_min"]),
        metric_max     = float(data["metric_max"]),
        metric_range   = float(data["metric_range"]),
        window_seconds = int(data["window_seconds"]),
        n_samples      = int(data["n_samples"]),
        service        = str(data["service"]),
        metric_name    = str(data["metric_name"]),
        trained_at     = str(data["trained_at"]),
    )


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def checkpoint_path(
    service: str,
    metric_name: str,
    window_seconds: int,
    root: Path | str = _DEFAULT_CHECKPOINT_ROOT,
) -> Path:
    """Return the expected file path for a given (service, metric, window)."""
    return (
        Path(root) / service / metric_name / f"window_{window_seconds}s.npz"
    )


def select_checkpoint(
    service: str,
    metric_name: str,
    available_seconds: float,
    root: Path | str = _DEFAULT_CHECKPOINT_ROOT,
    config_path: Path | str = _DEFAULT_CONFIG_PATH,
    force_window: int | None = None,
) -> MarkovCheckpoint | None:
    """Load the appropriate checkpoint for the given (service, metric).

    Selection priority
    ------------------
    1. ``force_window`` parameter — if provided, load exactly that window.
    2. ``active_window_seconds`` in checkpoint_config.json — if set, load
       that window for every call.
    3. Automatic — largest window whose duration does not exceed
       ``available_seconds``.

    You can pin the window two ways:

    a) Per-call override::

        select_checkpoint("cartservice", "cpu_usage", 3600,
                          force_window=300)   # always use 5m checkpoint

    b) Global config pin (edit checkpoint_config.json)::

        { "active_window_seconds": 3600, ... }

    Set ``active_window_seconds`` to ``null`` or omit it to restore
    automatic selection.

    Parameters
    ----------
    service, metric_name :
        Identifiers used to locate the checkpoint directory.
    available_seconds :
        Duration of the baseline data available at call time, in seconds.
        Ignored when a window is pinned via force_window or config.
    root :
        Root directory for checkpoint storage.
    config_path :
        Path to checkpoint_config.json.
    force_window :
        If provided, load exactly this window (seconds) and skip all
        other selection logic.  Raises FileNotFoundError if the checkpoint
        does not exist on disk.

    Returns
    -------
    MarkovCheckpoint | None
        The selected checkpoint, or None if no checkpoint files exist
        for this (service, metric_name) combination.
    """
    root = Path(root)

    # --- Priority 1: explicit per-call override -------------------------
    if force_window is not None:
        path = checkpoint_path(service, metric_name, force_window, root)
        if not path.exists():
            raise FileNotFoundError(
                f"Forced checkpoint window={force_window}s not found: {path}"
            )
        logger.debug(
            "Using forced window=%ds for %s/%s",
            force_window, service, metric_name,
        )
        return load_checkpoint(path)

    # --- Priority 2: global pin from config ----------------------------
    config = load_config(config_path)
    active = config.get("active_window_seconds")
    if active is not None:
        active = int(active)
        path   = checkpoint_path(service, metric_name, active, root)
        if not path.exists():
            logger.warning(
                "active_window_seconds=%ds set in config but checkpoint "
                "not found for %s/%s — falling back to automatic selection.",
                active, service, metric_name,
            )
        else:
            logger.debug(
                "Using config-pinned window=%ds for %s/%s",
                active, service, metric_name,
            )
            return load_checkpoint(path)

    # --- Priority 3: automatic selection --------------------------------
    windows = get_window_seconds(config_path)

    for window in sorted(windows, reverse=True):
        if available_seconds >= window:
            path = checkpoint_path(service, metric_name, window, root)
            if path.exists():
                logger.debug(
                    "Auto-selected window=%ds for %s/%s (available=%.0fs)",
                    window, service, metric_name, available_seconds,
                )
                return load_checkpoint(path)

    # Do not fall back to the smallest checkpoint when available baseline
    # is too short for all configured windows. This keeps auto-selection
    # strict; smallest-window usage requires explicit pinning via
    # force_window or active_window_seconds in checkpoint_config.json.
    smallest_window = min(windows) if windows else None
    if smallest_window is not None and available_seconds < smallest_window:
        logger.info(
            "Available baseline (%.0fs) is shorter than the smallest "
            "configured window (%ds) for %s/%s. Skipping checkpoint "
            "auto-selection.",
            available_seconds, smallest_window, service, metric_name,
        )

    logger.debug(
        "No suitable checkpoint selected for %s/%s under %s",
        service,
        metric_name,
        root,
    )
    return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    service: str,
    metric_name: str,
    checkpoints: list[MarkovCheckpoint],
    root: Path | str = _DEFAULT_CHECKPOINT_ROOT,
) -> None:
    """Write a manifest.json summarising all windows for one (service, metric).

    Parameters
    ----------
    checkpoints :
        List of trained checkpoints (all windows) for this service/metric.
    """
    root = Path(root)
    manifest_path = root / service / metric_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for cp in sorted(checkpoints, key=lambda c: c.window_seconds):
        entries.append({
            "window_seconds": cp.window_seconds,
            "window_label":   _window_label(cp.window_seconds),
            "n_samples":      cp.n_samples,
            "num_bins":       cp.num_bins,
            "metric_min":     cp.metric_min,
            "metric_max":     cp.metric_max,
            "trained_at":     cp.trained_at,
            "file":           f"window_{cp.window_seconds}s.npz",
        })

    manifest = {
        "service":     service,
        "metric_name": metric_name,
        "windows":     entries,
        "updated_at":  datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.debug("Wrote manifest: %s", manifest_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _discretize_array(
    arr: np.ndarray,
    bin_edges: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    idxs = np.searchsorted(bin_edges, arr, side="right").astype(int) - 1
    return np.clip(idxs, 0, num_bins - 1)


def _normalise(counts: np.ndarray, num_bins: int) -> np.ndarray:
    row_sums = counts.sum(axis=1, keepdims=True)
    T = counts.copy()
    unseen = row_sums.ravel() == 0
    T[unseen] = 1.0
    row_sums[unseen] = num_bins
    return T / row_sums


def _window_label(seconds: int) -> str:
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"