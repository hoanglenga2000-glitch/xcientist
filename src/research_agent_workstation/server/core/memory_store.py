from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .json_utils import append_jsonl, write_json

SuccessLabel = Literal["success", "fail", "neutral"]


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    task_type: str
    dataset_type: str
    hypothesis: str
    method_summary: str
    code_summary: str
    metric_before: float | None
    metric_after: float | None
    success_label: SuccessLabel
    failure_reason: str | None
    useful_for: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class MemoryStore:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.records: list[MemoryRecord] = []

    def add(self, **kwargs) -> MemoryRecord:
        record = MemoryRecord(record_id=f"memory_{uuid4().hex[:10]}", **kwargs)
        self.records.append(record)
        append_jsonl(self.workspace_root / "workspace" / "memory_records.jsonl", record)
        return record

    def find_related(self, task_type: str, dataset_type: str) -> list[MemoryRecord]:
        return [record for record in self.records if record.task_type == task_type or record.dataset_type == dataset_type]

    def write_run_memory(self, output_dir: Path) -> Path:
        return write_json(output_dir / "memory_records.json", {"records": self.records})
