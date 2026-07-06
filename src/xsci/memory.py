"""Read-only retrospective-memory inspector for xsci."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from research_os.retrospective_memory import MemoryRecord, RetrospectiveMemoryStore

from .results import evolution_dir


def default_memory_path(project_root: Optional[Path] = None) -> Path:
    return evolution_dir(project_root) / "retrospective_memory.json"


def _load_records(path: Path) -> list[MemoryRecord]:
    if not path.exists():
        return []
    try:
        return RetrospectiveMemoryStore(path)._load()
    except Exception:
        return []


def _score(v: Any) -> str:
    return f"{v:+.6f}" if isinstance(v, (int, float)) else "-"


def _print_table(records: list[MemoryRecord]) -> None:
    if not records:
        print("no retrospective memory records found.")
        return
    print(f"{'memory_id':<42} {'type':<16} {'method':<18} {'delta':>10} lesson")
    for r in records:
        lesson = r.reusable_strategy or r.what_worked or r.what_failed or r.failure_pattern or "-"
        if len(lesson) > 90:
            lesson = lesson[:87] + "..."
        print(f"{r.memory_id:<42} {r.task_type:<16} {r.method:<18} {_score(r.metric_delta):>10} {lesson}")


def run_memory(
    command: str = "list",
    *,
    task_type: str = "",
    limit: int = 20,
    json_output: bool = False,
    path: str = "",
) -> int:
    """List/export reusable lessons from the shared evolution memory file."""
    memory_path = Path(path) if path else default_memory_path()
    records = _load_records(memory_path)
    if task_type:
        records = [r for r in records if r.task_type == task_type]

    if command == "successes":
        records = [r for r in records if r.what_worked or r.reusable_strategy]
    elif command == "failures":
        records = [r for r in records if r.what_failed or r.failure_pattern]
    elif command != "list":
        print("usage: xsci memory {list|successes|failures}")
        return 2

    records = records[-max(1, limit):]
    if json_output:
        print(json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2))
    else:
        print(f"memory: {memory_path}")
        _print_table(records)
    return 0
