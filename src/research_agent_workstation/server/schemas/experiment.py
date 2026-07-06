from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ExperimentSourceType = Literal["local_template", "codex", "claude_code", "manual"]


@dataclass(slots=True)
class ExperimentRecord:
    run_id: str
    task_id: str
    source_type: ExperimentSourceType
    code_agent_provider: str
    code_patch_id: str | None
    runner_provider: str
    gpu_provider: str
    llm_provider: str
    dataset_version: str
    code_commit: str | None
    seed: int
    metric: dict[str, Any]
    artifacts: list[Path] = field(default_factory=list)
    output_dir: Path | None = None

