"""Compact experiment result tracking for end-to-end pipeline steps.

This module stores only the important post-major-step results, separate from
full timeline or raw metric artifacts. It is intentionally small and JSON-
friendly so it can be used to showcase the experiment process.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    step: str
    summary: str
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultsStore:
    run_id: str
    fault: str
    service: str
    duration_seconds: int
    created_at: float = field(default_factory=time.time)
    steps: list[StepResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        step: str,
        summary: str,
        details: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.steps.append(
            StepResult(
                step=step,
                summary=summary,
                timestamp=timestamp if timestamp is not None else time.time(),
                details=details or {},
            )
        )

    def set_summary(self, summary: dict[str, Any]) -> None:
        self.summary = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fault": self.fault,
            "service": self.service,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
            "steps": [
                {
                    "step": step.step,
                    "summary": step.summary,
                    "timestamp": step.timestamp,
                    "details": step.details,
                }
                for step in self.steps
            ],
            "summary": self.summary,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
