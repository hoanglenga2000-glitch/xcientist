from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskProfile:
    task_id: str
    name: str
    task_type: str
    target: str
    metric: str
    task_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    overview_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

