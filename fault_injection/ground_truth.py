"""Ground-truth JSON writer and validator for fault injection experiments.

Schema:
    {
        "run_id":            str,   # e.g. "20260326_143022"
        "fault_type":        str,   # cpu_hog | mem_leak | net_delay | disk_hog
        "target_services":   list[str],
        "inject_time_utc":   str,   # ISO-8601
        "duration_seconds":  int
    }
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_FIELDS = {"run_id", "fault_type", "target_services", "inject_time_utc", "duration_seconds"}
VALID_FAULTS = {"cpu_hog", "mem_leak", "net_delay", "disk_hog"}


def make_run_id() -> str:
    """Return a timestamp-based run ID, e.g. '20260326_143022'."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write(
    run_id: str,
    fault_type: str,
    target_services: list[str],
    duration_seconds: int,
    output_dir: Path,
) -> Path:
    """Write a ground_truth.json into *output_dir* and return its path.

    Creates *output_dir* if it doesn't exist.
    Raises ValueError if the inputs don't pass validation.
    """
    record = {
        "run_id": run_id,
        "fault_type": fault_type,
        "target_services": target_services,
        "inject_time_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
    }
    validate(record)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ground_truth.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def load(path: Path) -> dict:
    """Load and validate a ground_truth.json file."""
    record = json.loads(Path(path).read_text())
    validate(record)
    return record


def validate(record: dict) -> None:
    """Raise ValueError if *record* is missing fields or has invalid values."""
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"ground_truth missing fields: {missing}")

    if record["fault_type"] not in VALID_FAULTS:
        raise ValueError(f"unknown fault_type '{record['fault_type']}' — must be one of {VALID_FAULTS}")

    if not isinstance(record["target_services"], list) or not record["target_services"]:
        raise ValueError("target_services must be a non-empty list")

    if not isinstance(record["duration_seconds"], int) or record["duration_seconds"] <= 0:
        raise ValueError("duration_seconds must be a positive integer")
