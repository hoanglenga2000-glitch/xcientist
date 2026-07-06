"""JSON-backed retrospective memory for reusable MLE lessons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemoryRecord:
    memory_id: str
    task_type: str
    dataset_profile: dict[str, Any]
    method: str
    what_worked: str
    what_failed: str
    metric_delta: float | None
    reusable_strategy: str
    failure_pattern: str
    linked_exp_ids: list[str]


class RetrospectiveMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("records", [])
        return [MemoryRecord(**record) for record in records]

    def _save(self, records: list[MemoryRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_memory(self, record: MemoryRecord) -> None:
        records = [item for item in self._load() if item.memory_id != record.memory_id]
        records.append(record)
        self._save(records)

    def retrieve_by_task_type(self, task_type: str) -> list[MemoryRecord]:
        return [record for record in self._load() if record.task_type == task_type]

    def retrieve_failures(self, task_type: str | None = None) -> list[MemoryRecord]:
        records = self._load()
        if task_type is not None:
            records = [record for record in records if record.task_type == task_type]
        return [record for record in records if record.what_failed or record.failure_pattern]

    def retrieve_successes(self, task_type: str | None = None) -> list[MemoryRecord]:
        records = self._load()
        if task_type is not None:
            records = [record for record in records if record.task_type == task_type]
        return [record for record in records if record.what_worked or record.reusable_strategy]
