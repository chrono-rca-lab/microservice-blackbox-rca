"""Structured logging and timing for RCA pipeline stages."""

import time
from typing import Any


def log_stage(
    stage: str,
    file_name: str,
    start_time: float,
    stage_start: float,
    logs: list[dict[str, Any]],
) -> None:
    """Record a stage completion with timing information.

    Parameters
    ----------
    stage : str
        Stage name (e.g., "LAYER1_CUSUM", "LAYER2_PREDICTION_ERROR")
    file_name : str
        Source file name (e.g., "change_point.py", "fault_chain.py")
    start_time : float
        POSIX timestamp of pipeline start (fault_chain.pinpoint entry)
    stage_start : float
        POSIX timestamp when this stage began
    logs : list[dict]
        Shared log list to append to (mutated in place)
    """
    now = time.time()
    logs.append(
        {
            "stage": stage,
            "file": file_name,
            "timestamp": now,
            "since_start_seconds": round(now - start_time, 3),
            "duration_seconds": round(now - stage_start, 3),
        }
    )
