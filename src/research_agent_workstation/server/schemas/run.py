from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "unavailable"]


@dataclass(slots=True)
class RunResult:
    run_id: str
    command: list[str]
    cwd: Path
    return_code: int
    stdout_path: Path
    stderr_path: Path
    output_dir: Path | None = None


@dataclass(slots=True)
class ResourceEstimate:
    provider: str
    available: bool
    expected_seconds: int | None = None
    requires_human_gate: bool = False
    notes: str = ""


@dataclass(slots=True)
class RemoteJob:
    job_id: str
    provider: str
    status: JobStatus
    metadata: dict[str, Any] = field(default_factory=dict)

