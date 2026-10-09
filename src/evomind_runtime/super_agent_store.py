from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIGRATION_ID = "super_agent_v1_001"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capability_descriptors (
  capability_id TEXT PRIMARY KEY,
  provider_id TEXT NOT NULL,
  version TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS connector_instances (
  connector_id TEXT PRIMARY KEY,
  connector_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS directory_capabilities (
  directory_id TEXT PRIMARY KEY,
  connector_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_graphs_v2 (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS execution_attempts_v2 (
  attempt_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,
  UNIQUE(run_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS failure_observations_v2 (
  failure_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  signature TEXT NOT NULL,
  category TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failure_v2_signature
  ON failure_observations_v2(run_id, node_id, signature);
CREATE TABLE IF NOT EXISTS repair_actions_v2 (
  repair_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operation_receipts_v2 (
  receipt_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  operation TEXT NOT NULL,
  ok INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_packages_v2 (
  package_id TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  status TEXT NOT NULL,
  package_sha256 TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_memory_v2 (
  record_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  memory_scope TEXT NOT NULL,
  evidence_status TEXT NOT NULL,
  failure_signature TEXT NOT NULL,
  evidence_sha256 TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_memory_v2_lookup
  ON evidence_memory_v2(tenant_id, project_id, failure_signature, evidence_status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SuperAgentStore:
    """Additive V2 state store that can safely share the existing runtime SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(SCHEMA)
            self._connection.execute(
                "INSERT OR IGNORE INTO schema_migrations VALUES (?,?)",
                (MIGRATION_ID, _now()),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def migration_applied(self) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version=?",
                (MIGRATION_ID,),
            ).fetchone()
        return row is not None

    def put_capability(self, payload: dict[str, Any]) -> None:
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO capability_descriptors VALUES (?,?,?,?,?)",
                (payload["capability_id"], payload["provider_id"], payload["version"], _json(payload), now),
            )

    def put_connector(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO connector_instances VALUES (?,?,?,?)",
                (payload["connector_id"], payload["connector_type"], _json(payload), _now()),
            )

    def put_directory(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO directory_capabilities VALUES (?,?,?,?,?,?)",
                (
                    payload["directory_id"],
                    payload["connector_id"],
                    payload["tenant_id"],
                    payload["project_id"],
                    _json(payload),
                    _now(),
                ),
            )

    def list_directories(self, *, tenant_id: str, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM directory_capabilities WHERE tenant_id=? AND project_id=? ORDER BY directory_id",
                (tenant_id, project_id),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put_task_graph(self, run_id: str, status: str, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO task_graphs_v2 VALUES (?,?,?,?)",
                (run_id, status, _json(payload), _now()),
            )

    def get_task_graph(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT status,payload_json,updated_at_utc FROM task_graphs_v2 WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {"status": row[0], "payload": json.loads(row[1]), "updated_at_utc": row[2]}

    def put_attempt(self, payload: dict[str, Any]) -> None:
        now = _now()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at_utc FROM execution_attempts_v2 WHERE attempt_id=?",
                (payload["attempt_id"],),
            ).fetchone()
            self._connection.execute(
                "INSERT OR REPLACE INTO execution_attempts_v2 VALUES (?,?,?,?,?,?,?,?)",
                (
                    payload["attempt_id"], payload["run_id"], payload["node_id"],
                    payload["idempotency_key"], payload["status"], _json(payload),
                    existing[0] if existing else now, now,
                ),
            )

    def append_failure(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO failure_observations_v2 VALUES (?,?,?,?,?,?,?)",
                (
                    payload["failure_id"], payload["run_id"], payload["node_id"],
                    payload["signature"], payload["category"], _json(payload), _now(),
                ),
            )

    def failure_count(self, *, run_id: str, node_id: str, signature: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM failure_observations_v2 WHERE run_id=? AND node_id=? AND signature=?",
                (run_id, node_id, signature),
            ).fetchone()
        return int(row[0])

    def put_repair(self, payload: dict[str, Any]) -> None:
        now = _now()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at_utc FROM repair_actions_v2 WHERE repair_id=?",
                (payload["repair_id"],),
            ).fetchone()
            self._connection.execute(
                "INSERT OR REPLACE INTO repair_actions_v2 VALUES (?,?,?,?,?,?,?)",
                (
                    payload["repair_id"], payload["run_id"], payload["node_id"], payload["status"],
                    _json(payload), existing[0] if existing else now, now,
                ),
            )

    def append_receipt(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO operation_receipts_v2 VALUES (?,?,?,?,?,?,?)",
                (
                    payload["receipt_id"], payload["run_id"], payload["node_id"],
                    payload["operation"], int(bool(payload["ok"])), _json(payload), _now(),
                ),
            )

    def put_tool_package(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO tool_packages_v2 VALUES (?,?,?,?,?,?)",
                (
                    payload["package_id"], payload["version"], payload["status"],
                    payload["package_sha256"], _json(payload), _now(),
                ),
            )

    def put_memory(self, payload: dict[str, Any]) -> None:
        now = _now()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at_utc FROM evidence_memory_v2 WHERE record_id=?",
                (payload["record_id"],),
            ).fetchone()
            self._connection.execute(
                "INSERT OR REPLACE INTO evidence_memory_v2 VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    payload["record_id"], payload["tenant_id"], payload["project_id"],
                    payload["memory_scope"], payload["evidence_status"],
                    payload.get("failure_signature", ""), payload["evidence_sha256"],
                    _json(payload), existing[0] if existing else now, now,
                ),
            )

    def search_memory(
        self,
        *,
        tenant_id: str,
        project_id: str,
        failure_signature: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses = ["((tenant_id=? AND project_id=?) OR memory_scope='anonymous_global')"]
        params: list[Any] = [tenant_id, project_id]
        if failure_signature:
            clauses.append("failure_signature=?")
            params.append(failure_signature)
        params.append(max(1, min(int(limit), 100)))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT payload_json FROM evidence_memory_v2 WHERE {' AND '.join(clauses)} ORDER BY updated_at_utc DESC LIMIT ?",
                params,
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


__all__ = ["MIGRATION_ID", "SuperAgentStore"]
