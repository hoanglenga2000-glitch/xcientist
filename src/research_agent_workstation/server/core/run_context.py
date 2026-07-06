from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunContext:
    task_id: str
    run_id: str | None
    workspace_root: Path
    output_dir: Path | None = None
    task_profile: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

