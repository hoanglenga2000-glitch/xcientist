from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.reconcile_action_log_mirror import reconcile


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE action_logs (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                task_id TEXT,
                run_id TEXT,
                message TEXT NOT NULL,
                artifact_path TEXT,
                metadata_json TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO action_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("action_1", "create", "task_1", None, "created", None, '{"n":1}', 1_700_000_000_000),
                ("action_2", "finish", "task_1", "run_1", "finished", "artifact.json", None, 1_700_000_001_000),
            ],
        )


def test_reconcile_rebuilds_exact_mirror_and_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "workstation.db"
    runtime = tmp_path / "workspace" / "runtime"
    runtime.mkdir(parents=True)
    _database(database)
    (runtime / "action_log.jsonl").write_text('{"action_id":"orphan"}\n', encoding="utf-8")

    first = reconcile(database, runtime)
    records = [json.loads(line) for line in (runtime / "action_log.jsonl").read_text(encoding="utf-8").splitlines()]

    assert first["ok"] is True
    assert first["rewritten"] is True
    assert first["archive_path"]
    assert [record["action_id"] for record in records] == ["action_1", "action_2"]
    checkpoint = json.loads((runtime / "action_log.checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["sqlite_is_canonical"] is True
    assert checkpoint["action_count"] == 2
    assert checkpoint["last_action_id"] == "action_2"

    archive_count = len(list(runtime.glob("action_log.pre-reconcile.*.jsonl")))
    second = reconcile(database, runtime)
    assert second["rewritten"] is False
    assert second["archive_path"] is None
    assert len(list(runtime.glob("action_log.pre-reconcile.*.jsonl"))) == archive_count
