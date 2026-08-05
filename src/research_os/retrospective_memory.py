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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        """Load the current schema plus the one-record pre-schema format.

        Early EvoMind builds wrote ``{memory_id, task, what_worked,
        reusable_strategy}``.  That record remains valid historical evidence,
        but ``MemoryRecord(**payload)`` made the entire shared store unreadable
        after the typed schema was introduced.  Normalize only that exact
        legacy shape; malformed or unknown fields still fail closed.
        """

        if not isinstance(payload, dict):
            raise ValueError("retrospective memory record must be an object")
        normalized = dict(payload)
        legacy_task = normalized.pop("task", None)
        current_fields = {
            "memory_id", "task_type", "dataset_profile", "method",
            "what_worked", "what_failed", "metric_delta",
            "reusable_strategy", "failure_pattern", "linked_exp_ids",
        }
        unknown = sorted(set(normalized) - current_fields)
        if unknown:
            raise ValueError(f"retrospective memory record has unknown fields: {', '.join(unknown)}")

        if "task_type" not in normalized:
            legacy_keys = {"memory_id", "what_worked", "reusable_strategy"}
            if not legacy_keys.issubset(normalized) or not isinstance(legacy_task, str) or not legacy_task.strip():
                raise ValueError("retrospective memory record does not match a supported schema")
            normalized.update({
                "task_type": "unknown",
                "dataset_profile": {
                    "task_name": legacy_task.strip(),
                    "source": "legacy_retrospective_memory_v0",
                    "evidence_level": "provisional",
                },
                "method": "legacy_memory",
                "what_failed": "",
                "metric_delta": None,
                "failure_pattern": "",
                "linked_exp_ids": [],
            })
        elif legacy_task is not None:
            if not isinstance(legacy_task, str) or not legacy_task.strip():
                raise ValueError("retrospective memory legacy task alias is invalid")
            profile = normalized.get("dataset_profile")
            if not isinstance(profile, dict):
                raise ValueError("retrospective memory dataset_profile must be an object")
            normalized["dataset_profile"] = {"legacy_task": legacy_task.strip(), **profile}

        try:
            return cls(**normalized)
        except TypeError as error:
            raise ValueError("retrospective memory record does not match the current schema") from error


class RetrospectiveMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError("retrospective memory store must be a list or a records object")
        return [MemoryRecord.from_dict(record) for record in records]

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
