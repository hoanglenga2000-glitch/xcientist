from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import ApprovalRequest, RunEvent, Session, ToolCall, ToolResult, new_id, utc_now

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, workspace_root TEXT NOT NULL, permission_level TEXT NOT NULL,
  status TEXT NOT NULL, title TEXT NOT NULL, objective TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, parent_session_id TEXT NOT NULL,
  selected_model_policy TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content_json TEXT NOT NULL,
  created_at TEXT NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tool_name TEXT NOT NULL,
  arguments_json TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT NOT NULL,
  created_at TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT NOT NULL,
  approval_id TEXT NOT NULL, idempotency_key TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, tool_call_id TEXT NOT NULL,
  tool_name TEXT NOT NULL, argument_fingerprint TEXT NOT NULL,
  normalized_arguments_json TEXT NOT NULL, impact_scope_json TEXT NOT NULL,
  risk_level TEXT NOT NULL, reversible INTEGER NOT NULL, status TEXT NOT NULL,
  created_at TEXT NOT NULL, expires_at TEXT NOT NULL, decided_at TEXT NOT NULL,
  decision_note TEXT NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER NOT NULL,
  event_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(session_id, seq), FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS checkpoints (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER NOT NULL,
  state_json TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sha256 TEXT NOT NULL, path TEXT NOT NULL,
  media_type TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS benchmark_runs (
  id TEXT PRIMARY KEY, suite_id TEXT NOT NULL, tool_name TEXT NOT NULL, status TEXT NOT NULL,
  metrics_json TEXT NOT NULL, artifact_path TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id, created_at);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class RuntimeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_session(self, session: Session) -> Session:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (session.id, session.workspace_root, session.permission_level, session.status,
                 session.title, session.objective, session.created_at, session.updated_at,
                 session.parent_session_id, session.selected_model_policy, _json(session.metadata)),
            )
        self.append_event(session.id, "session.created", session.to_dict())
        return session

    def update_session(self, session_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"permission_level", "status", "title", "objective", "selected_model_policy", "metadata_json"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()
        if "metadata_json" in updates and not isinstance(updates["metadata_json"], str):
            updates["metadata_json"] = _json(updates["metadata_json"])
        assignments = ",".join(f"{key}=?" for key in updates)
        with self._lock, self._connection:
            self._connection.execute(f"UPDATE sessions SET {assignments} WHERE id=?", (*updates.values(), session_id))
        result = self.get_session(session_id)
        if result is None:
            raise KeyError(session_id)
        return result

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = _row(self._connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone())
        if row:
            row["metadata"] = json.loads(row.pop("metadata_json"))
        return row

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for item in map(dict, rows):
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def add_turn(self, session_id: str, role: str, content: Any) -> str:
        turn_id = new_id("turn")
        with self._lock, self._connection:
            self._connection.execute("INSERT INTO turns VALUES (?,?,?,?,?)", (turn_id, session_id, role, _json(content), utc_now()))
        self.append_event(session_id, "turn.recorded", {"turn_id": turn_id, "role": role})
        return turn_id

    def list_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM turns WHERE session_id=? ORDER BY created_at", (session_id,)).fetchall()
        result = []
        for item in map(dict, rows):
            item["content"] = json.loads(item.pop("content_json"))
            result.append(item)
        return result

    def put_tool_call(self, call: ToolCall, result: ToolResult | None = None) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (call.id, call.session_id, call.tool_name, _json(call.arguments), call.status,
                 _json(result.to_dict() if result else {}), call.created_at, call.started_at,
                 call.completed_at, call.approval_id, call.idempotency_key),
            )

    def get_tool_call(self, call_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = _row(self._connection.execute("SELECT * FROM tool_calls WHERE id=?", (call_id,)).fetchone())
        if item:
            item["arguments"] = json.loads(item.pop("arguments_json"))
            item["result"] = json.loads(item.pop("result_json"))
        return item

    def pending_tool_call(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM tool_calls WHERE session_id=? AND status='waiting_approval' ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        item = _row(row)
        if item:
            item["arguments"] = json.loads(item.pop("arguments_json"))
            item["result"] = json.loads(item.pop("result_json"))
        return item

    def put_approval(self, approval: ApprovalRequest) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (approval.id, approval.session_id, approval.tool_call_id, approval.tool_name,
                 approval.argument_fingerprint, _json(approval.normalized_arguments),
                 _json(approval.impact_scope), approval.risk_level, int(approval.reversible),
                 approval.status, approval.created_at, approval.expires_at, approval.decided_at,
                 approval.decision_note),
            )

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = _row(self._connection.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone())
        if item:
            item["normalized_arguments"] = json.loads(item.pop("normalized_arguments_json"))
            item["impact_scope"] = json.loads(item.pop("impact_scope_json"))
            item["reversible"] = bool(item["reversible"])
        return item

    def list_approvals(self, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM approvals"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        result = []
        for item in map(dict, rows):
            item["normalized_arguments"] = json.loads(item.pop("normalized_arguments_json"))
            item["impact_scope"] = json.loads(item.pop("impact_scope_json"))
            item["reversible"] = bool(item["reversible"])
            result.append(item)
        return result

    def decide_approval(self, approval_id: str, approved: bool, note: str = "") -> dict[str, Any]:
        status = "approved" if approved else "rejected"
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE approvals SET status=?,decided_at=?,decision_note=? WHERE id=? AND status='pending'",
                (status, utc_now(), note[:2000], approval_id),
            )
        approval = self.get_approval(approval_id)
        if approval is None:
            raise KeyError(approval_id)
        self.append_event(approval["session_id"], f"approval.{status}", {"approval_id": approval_id, "note": note[:500]})
        return approval

    def next_seq(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COALESCE(MAX(seq),0)+1 AS value FROM events WHERE session_id=?", (session_id,)).fetchone()
        return int(row["value"])

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> RunEvent:
        with self._lock, self._connection:
            event = RunEvent(new_id("evt"), session_id, self.next_seq(session_id), event_type, payload)
            self._connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (event.id, event.session_id, event.seq, event.event_type, _json(event.payload), event.created_at),
            )
        return event

    def list_events(self, session_id: str, after_seq: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM events WHERE session_id=? AND seq>? ORDER BY seq LIMIT ?",
                (session_id, after_seq, limit),
            ).fetchall()
        result = []
        for item in map(dict, rows):
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def checkpoint(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = new_id("cp")
        seq = self.next_seq(session_id)
        created_at = utc_now()
        with self._lock, self._connection:
            self._connection.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", (checkpoint_id, session_id, seq, _json(state), created_at))
        self.append_event(session_id, "checkpoint.created", {"checkpoint_id": checkpoint_id, "state": state})
        return {"id": checkpoint_id, "session_id": session_id, "seq": seq, "state": state, "created_at": created_at}

    def latest_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = _row(self._connection.execute("SELECT * FROM checkpoints WHERE session_id=? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone())
        if item:
            item["state"] = json.loads(item.pop("state_json"))
        return item

    def add_artifact(self, session_id: str, source: bytes, suffix: str, media_type: str, artifact_root: Path) -> dict[str, Any]:
        digest = hashlib.sha256(source).hexdigest()
        target = artifact_root / digest[:2] / f"{digest}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(source)
            temporary.replace(target)
        artifact_id = new_id("artifact")
        created_at = utc_now()
        with self._lock, self._connection:
            self._connection.execute("INSERT INTO artifacts VALUES (?,?,?,?,?,?,?)", (artifact_id, session_id, digest, str(target), media_type, len(source), created_at))
        return {"id": artifact_id, "session_id": session_id, "sha256": digest, "path": str(target), "media_type": media_type, "bytes": len(source), "created_at": created_at}

    def record_benchmark(self, suite_id: str, tool_name: str, status: str, metrics: dict[str, Any], artifact_path: str = "") -> str:
        run_id = new_id("bench")
        with self._lock, self._connection:
            self._connection.execute("INSERT INTO benchmark_runs VALUES (?,?,?,?,?,?,?)", (run_id, suite_id, tool_name, status, _json(metrics), artifact_path, utc_now()))
        return run_id

    def list_benchmarks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for item in map(dict, rows):
            item["metrics"] = json.loads(item.pop("metrics_json"))
            result.append(item)
        return result
