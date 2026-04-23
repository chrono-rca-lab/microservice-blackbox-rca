"""Structured logging and timing for RCA pipeline stages."""

import time
from typing import Any


def log_stage(stage, file, start_time, current_time, logs):
    if any(entry["stage"] == stage for entry in logs):
        return

    duration = current_time - start_time

    logs.append({
        "stage": stage,
        "timestamp": current_time,
        "since_start_seconds": duration,
    })

    print(f"[RCA] {stage:<25} (+{duration:.3f}s)")