from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .json_utils import append_jsonl, write_json


@dataclass(slots=True)
class RuntimeEvent:
    timestamp: str
    task_id: str
    event_type: str
    stage: str
    message: str
    run_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.events: list[RuntimeEvent] = []

    def emit(self, event_type: str, stage: str, message: str, run_id: str | None = None, **payload: Any) -> RuntimeEvent:
        event = RuntimeEvent(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            task_id=self.task_id,
            run_id=run_id,
            event_type=event_type,
            stage=stage,
            message=message,
            payload=payload,
        )
        self.events.append(event)
        return event

    def flush(self, output_dir: Path) -> None:
        event_log = output_dir / "event_log.jsonl"
        if event_log.exists():
            event_log.unlink()
        for event in self.events:
            append_jsonl(event_log, event)
        write_json(output_dir / "event_log.json", {"events": self.events})
