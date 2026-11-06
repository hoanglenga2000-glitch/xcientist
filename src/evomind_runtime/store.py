from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
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
CREATE TABLE IF NOT EXISTS assistant_runs (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
  prompt TEXT NOT NULL, task_root TEXT NOT NULL, status TEXT NOT NULL,
  plan_json TEXT NOT NULL, attachment_ids_json TEXT NOT NULL,
  retry_count INTEGER NOT NULL, error_class TEXT NOT NULL, error_message TEXT NOT NULL,
  model_provider TEXT NOT NULL, model TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS goal_records (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, goal_key TEXT NOT NULL,
  schema TEXT NOT NULL, allocation TEXT NOT NULL,
  spec_sha256 TEXT NOT NULL, spec_json TEXT NOT NULL,
  board_sha256 TEXT NOT NULL, human_baseline_sha256 TEXT NOT NULL, board_json TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(goal_key),
  FOREIGN KEY(run_id) REFERENCES assistant_runs(id)
);
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY, original_name TEXT NOT NULL, safe_name TEXT NOT NULL,
  media_type TEXT NOT NULL, total_bytes INTEGER NOT NULL, chunk_size INTEGER NOT NULL,
  expected_sha256 TEXT NOT NULL, status TEXT NOT NULL, chunk_count INTEGER NOT NULL,
  final_path TEXT NOT NULL, final_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS upload_chunks (
  upload_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL, path TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(upload_id, chunk_index), FOREIGN KEY(upload_id) REFERENCES uploads(id)
);
CREATE TABLE IF NOT EXISTS attachments (
  id TEXT PRIMARY KEY, upload_id TEXT NOT NULL, run_id TEXT NOT NULL,
  name TEXT NOT NULL, path TEXT NOT NULL, media_type TEXT NOT NULL,
  bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
  FOREIGN KEY(upload_id) REFERENCES uploads(id)
);
CREATE TABLE IF NOT EXISTS deliverables (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, session_id TEXT NOT NULL,
  name TEXT NOT NULL, path TEXT NOT NULL, media_type TEXT NOT NULL,
  bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, source_tool_call TEXT NOT NULL,
  preview_kind TEXT NOT NULL, verified_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_idempotency ON tool_calls(session_id, idempotency_key, created_at);
CREATE INDEX IF NOT EXISTS idx_assistant_runs_conversation ON assistant_runs(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_goal_records_run ON goal_records(run_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_attachments_run ON attachments(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_deliverables_run ON deliverables(run_id, verified_at);
CREATE TABLE IF NOT EXISTS tenant_usage (
  tenant_id TEXT PRIMARY KEY, bytes INTEGER, files INTEGER, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS upload_quota (
  upload_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, bytes INTEGER NOT NULL,
  files INTEGER NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_upload_quota_tenant ON upload_quota(tenant_id);
CREATE INDEX IF NOT EXISTS idx_attachments_upload ON attachments(upload_id, created_at);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status, updated_at);
CREATE TABLE IF NOT EXISTS upload_claims (
  upload_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS deletion_log (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, upload_id TEXT NOT NULL, attachment_id TEXT NOT NULL,
  bytes INTEGER NOT NULL, files INTEGER NOT NULL, deleted_at TEXT NOT NULL, actor TEXT NOT NULL,
  name_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deletion_log_upload ON deletion_log(upload_id);
CREATE TABLE IF NOT EXISTS health_probe (
  id INTEGER PRIMARY KEY, probed_at TEXT NOT NULL
);
"""

# Uploads in these states hold a quota reservation; anything else is released by reconcile.
_RESERVED_UPLOAD_STATUSES = ("uploading", "completing", "complete")
# Named columns: a DB opened by tip b244060 may carry a legacy, ignored uploads.claim_id column.
# The uploads schema itself stays identical to cc69213 so a rollback's positional INSERT works.
_UPLOAD_INSERT = (
    "INSERT INTO uploads (id, original_name, safe_name, media_type, total_bytes, chunk_size, expected_sha256, "
    "status, chunk_count, final_path, final_sha256, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


DEFAULT_BUSY_TIMEOUT_MS = 5000
APPEND_EVENT_LOCKED_RETRIES = 5


def _busy_timeout_ms() -> int:
    """`EVOMIND_SQLITE_BUSY_TIMEOUT_MS` (default 5000), clamped to [0, 600000]."""
    try:
        value = int(os.environ.get("EVOMIND_SQLITE_BUSY_TIMEOUT_MS", DEFAULT_BUSY_TIMEOUT_MS))
    except ValueError:
        value = DEFAULT_BUSY_TIMEOUT_MS
    return max(0, min(value, 600_000))


def _is_locked(error: sqlite3.OperationalError) -> bool:
    text = str(error).lower()
    return "database is locked" in text or "database table is locked" in text


class RuntimeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False, timeout=_busy_timeout_ms() / 1000)
        self._connection.row_factory = sqlite3.Row
        # t238 C7b A3: every connection waits for a competing writer instead of failing fast.
        self._connection.execute(f"PRAGMA busy_timeout={_busy_timeout_ms()}")
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
        assignments = ",".join(f"{key}=?" for key in updates)
        with self._lock, self._connection:
            if 'metadata_json' in updates:
                if not self._connection.in_transaction:
                    self._connection.execute('BEGIN IMMEDIATE')
                incoming = updates['metadata_json']
                incoming = json.loads(incoming) if isinstance(incoming, str) else dict(incoming)
                if not isinstance(incoming, dict):
                    raise ValueError('session_metadata_invalid')
                row = self._connection.execute('SELECT metadata_json FROM sessions WHERE id=?', (session_id,)).fetchone()
                if row is None:
                    raise KeyError(session_id)
                current = json.loads(row['metadata_json'])
                bound = current.get('model_execution_contract')
                if bound is not None:
                    if 'model_execution_contract' in incoming and incoming['model_execution_contract'] != bound:
                        raise ValueError('model_contract_drift')
                    incoming['model_execution_contract'] = bound
                if current.get('model_uncontracted_execution_started') is True:
                    if incoming.get('model_uncontracted_execution_started', True) is not True:
                        raise ValueError('model_execution_admission_drift')
                    incoming['model_uncontracted_execution_started'] = True
                updates['metadata_json'] = _json(incoming)
            self._connection.execute(f"UPDATE sessions SET {assignments} WHERE id=?", (*updates.values(), session_id))
        result = self.get_session(session_id)
        if result is None:
            raise KeyError(session_id)
        return result

    def set_user_pause_requested(self, session_id: str, requested: bool) -> bool:
        """Change only the pause flag; concurrent model/identity metadata survives."""
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute('BEGIN IMMEDIATE')
            row = self._connection.execute("SELECT metadata_json FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            metadata = json.loads(row["metadata_json"])
            changed = bool(metadata.get("user_pause_requested")) != requested
            if requested:
                metadata["user_pause_requested"] = True
            else:
                metadata.pop("user_pause_requested", None)
            if changed:
                self._connection.execute("UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?", (_json(metadata), utc_now(), session_id))
            return changed

    def bind_model_contract(self, session_id: str, contract: dict[str, Any]) -> dict[str, Any]:
        """Bind a new Run's model route atomically, without persisting credentials."""
        import re
        fields = {"schema", "provider", "model", "endpoint_sha256", "reasoning_effort", "service_tier"}
        schema = contract.get('schema')
        allowed_models = {'gpt-6-astra', 'gpt-5.6-sol'}
        if schema in {'evomind.model_execution_contract.v3', 'evomind.model_execution_contract.v4'}:
            allowed_models |= {'gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.5'}
        if schema == 'evomind.model_execution_contract.v4':
            allowed_models.update({'deepseek-v4-pro', 'deepseek-flash'})
            if contract.get('model') in {'deepseek-v4-pro', 'deepseek-flash'} and contract.get('wire_protocol') != 'chat_completions_v1':
                raise ValueError('model_contract_invalid')
            fields |= {'credential_binding_sha256', 'route_config_sha256'}
            if (not re.fullmatch(r'[a-f0-9]{64}', str(contract.get('credential_binding_sha256') or ''))
                    or (contract.get('route_config_sha256') is not None and not re.fullmatch(r'[a-f0-9]{64}', str(contract['route_config_sha256'])))):
                raise ValueError('model_contract_invalid')
        if schema in {'evomind.model_execution_contract.v2', 'evomind.model_execution_contract.v3', 'evomind.model_execution_contract.v4'}:
            fields |= {'wire_protocol', 'timeout_seconds', 'max_request_retries'}
            if schema in {'evomind.model_execution_contract.v3', 'evomind.model_execution_contract.v4'}:
                fields.add('retry_floor_seconds')
                if type(contract.get('retry_floor_seconds')) is not int or not 0 <= contract['retry_floor_seconds'] <= 30:
                    raise ValueError('model_contract_invalid')
            allowed_protocols = {'responses_stream_v1', 'chat_completions_v1'} if schema == 'evomind.model_execution_contract.v4' else {'responses_stream_v1'}
            if (contract.get('wire_protocol') not in allowed_protocols
                    or type(contract.get('timeout_seconds')) not in (int, float)
                    or not 0 < contract['timeout_seconds'] <= 3600
                    or type(contract.get('max_request_retries')) is not int
                    or not 0 <= contract['max_request_retries'] <= 2):
                raise ValueError('model_contract_invalid')
        if (set(contract) != fields or schema not in {"evomind.model_execution_contract.v1", "evomind.model_execution_contract.v2", "evomind.model_execution_contract.v3", "evomind.model_execution_contract.v4"}
                or contract.get("provider") != "openai"
                or contract.get("model") not in allowed_models
                or not re.fullmatch(r"[a-f0-9]{64}", str(contract.get("endpoint_sha256") or ""))):
            raise ValueError("model_contract_invalid")
        now = utc_now()
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute('BEGIN IMMEDIATE')
            row = self._connection.execute("SELECT metadata_json FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            metadata = json.loads(row["metadata_json"])
            previous = metadata.get("model_execution_contract")
            if previous is not None:
                if previous != contract:
                    raise ValueError("model_contract_drift")
                return previous
            if metadata.get('model_uncontracted_execution_started') is True:
                raise ValueError('legacy_model_contract_requires_review')
            observed = self._connection.execute(
                "SELECT 1 FROM events WHERE session_id=? AND event_type IN ('model.response','model_observed') LIMIT 1",
                (session_id,),
            ).fetchone()
            if observed:
                raise ValueError("legacy_model_contract_requires_review")
            metadata["model_execution_contract"] = dict(contract)
            self._connection.execute("UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?", (_json(metadata), now, session_id))
            seq = self._connection.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE session_id=?", (session_id,)).fetchone()[0]
            self._connection.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (
                new_id("event"), session_id, seq, "model.contract_bound", _json(contract), now,
            ))
        return dict(contract)

    def amend_siim_request_contract(self, session_id: str, expected: dict[str, Any], receipt: dict[str, Any]) -> None:
        """Internal deployment CAS for the explicitly confirmed SIIM amendment.

        Ordinary session updates still cannot modify a pinned model contract.
        This method is not exposed as a tool or HTTP endpoint.
        """
        if (expected.get('schema')!='evomind.model_execution_contract.v4'
                or expected.get('model')!='deepseek-v4-pro' or expected.get('timeout_seconds')!=180
                or receipt.get('schema')!='evomind.siim_request_amendment.v1'
                or receipt.get('authorization')!='explicit_user_confirmation_20260909'
                or receipt.get('max_output_tokens')!=16384 or receipt.get('total_request_seconds')!=600):
            raise ValueError('siim_request_amendment_invalid')
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute('BEGIN IMMEDIATE')
            row=self._connection.execute('SELECT status,metadata_json FROM sessions WHERE id=?',(session_id,)).fetchone()
            if not row or row['status'] not in {'paused','blocked','failed','completed'}:
                raise ValueError('siim_request_amendment_not_quiescent')
            if self._connection.execute("SELECT 1 FROM tool_calls WHERE session_id=? AND status='running'",(session_id,)).fetchone():
                raise ValueError('siim_request_amendment_tool_active')
            metadata=json.loads(row['metadata_json'])
            if metadata.get('model_execution_contract')!=expected:
                raise ValueError('siim_request_amendment_compare_failed')
            updated={**expected,'timeout_seconds':600}
            metadata['model_contract_history']=[*(metadata.get('model_contract_history') or []),
                {'contract':expected,'superseded_by':receipt['amendment_sha256'],'at':utc_now()}]
            metadata['model_execution_contract']=updated
            metadata['model_request_limits']=dict(receipt)
            self._connection.execute('UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?',(_json(metadata),utc_now(),session_id))
            seq=self._connection.execute('SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE session_id=?',(session_id,)).fetchone()[0]
            self._connection.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(new_id('event'),session_id,seq,
                'model.request_limits_amended',_json(receipt),utc_now()))

    def admit_uncontracted_model_client(self, session_id: str) -> None:
        """Serialize legacy/mock admission with the first governed binding.

        No write lock spans HTTP. The persistent admission marker prevents a
        peer from pinning a different model after an uncontracted request starts.
        """
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute('BEGIN IMMEDIATE')
            row = self._connection.execute('SELECT metadata_json FROM sessions WHERE id=?', (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            metadata = json.loads(row['metadata_json'])
            if metadata.get('model_execution_contract') is not None:
                raise ValueError('model_contract_missing_on_resume')
            if metadata.get('model_uncontracted_execution_started') is not True:
                metadata['model_uncontracted_execution_started'] = True
                self._connection.execute('UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?',
                    (_json(metadata), utc_now(), session_id))

    def rebind_session_managed_hpc_identity(
        self,
        session_id: str,
        *,
        expected_identity: dict[str, Any],
        next_identity: dict[str, Any],
        migration_record: dict[str, Any],
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically CAS one terminal Assistant Run's managed HPC identity.

        The service layer owns policy validation.  This storage primitive keeps
        the identity replacement and its immutable audit event in one SQLite
        transaction so a crash cannot leave an unrecorded rebind.
        """

        terminal_states = {"completed", "blocked", "failed", "cancelled"}
        now = utc_now()
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN IMMEDIATE")  # t238 C7b A3: seq allocation below
            session_row = self._connection.execute(
                "SELECT status,metadata_json FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            run_row = self._connection.execute(
                "SELECT session_id,status FROM assistant_runs WHERE id=?",
                (session_id,),
            ).fetchone()
            if session_row is None or run_row is None or str(run_row["session_id"]) != session_id:
                raise ValueError("managed_hpc_identity_rebind_run_binding_invalid")
            if str(session_row["status"]) not in terminal_states or str(run_row["status"]) not in terminal_states:
                raise ValueError("managed_hpc_identity_rebind_run_not_terminal")
            current_metadata_json = str(session_row["metadata_json"])
            metadata = json.loads(current_metadata_json)
            if not isinstance(metadata, dict):
                raise ValueError("managed_hpc_identity_rebind_metadata_invalid")
            current_identity = metadata.get("managed_hpc_identity")
            prior_migration = metadata.get("managed_hpc_identity_rebind")
            idempotency_sha = str(migration_record.get("idempotency_key_sha256") or "")
            target_sha = str(migration_record.get("to_identity_sha256") or "")
            historical = self._connection.execute(
                "SELECT payload_json FROM events WHERE session_id=? AND event_type=? ORDER BY seq",
                (session_id, "managed_hpc_identity.rebound"),
            ).fetchall()
            for row in historical:
                payload = json.loads(str(row["payload_json"]))
                if (
                    isinstance(payload, dict)
                    and str(payload.get("idempotency_key_sha256") or "") == idempotency_sha
                    and str(payload.get("to_identity_sha256") or "") != target_sha
                ):
                    raise ValueError("managed_hpc_identity_rebind_idempotency_key_reused")
            if current_identity == next_identity:
                if prior_migration == migration_record:
                    return {"updated": False, "metadata": metadata}
                raise ValueError("managed_hpc_identity_rebind_idempotency_mismatch")
            if current_identity != expected_identity:
                raise ValueError("managed_hpc_identity_rebind_compare_and_swap_failed")
            metadata["managed_hpc_identity"] = dict(next_identity)
            metadata["managed_hpc_identity_rebind"] = dict(migration_record)
            # Consumed refs can remain in legacy session metadata after their
            # envelopes have been tombstoned.  Never carry those opaque refs
            # across an allocation Generation boundary.
            metadata["run_secret_refs"] = []
            seq = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(seq),0)+1 AS value FROM events WHERE session_id=?",
                    (session_id,),
                ).fetchone()["value"]
            )
            event = RunEvent(
                new_id("evt"),
                session_id,
                seq,
                "managed_hpc_identity.rebound",
                dict(event_payload),
                now,
            )
            self._connection.execute(
                "UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=? AND metadata_json=?",
                (_json(metadata), now, session_id, current_metadata_json),
            )
            if self._connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise ValueError("managed_hpc_identity_rebind_compare_and_swap_failed")
            self._connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (event.id, event.session_id, event.seq, event.event_type, _json(event.payload), event.created_at),
            )
        return {"updated": True, "metadata": metadata}

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

    def create_assistant_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO assistant_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    payload["id"], payload["session_id"], payload["conversation_id"], payload["prompt"],
                    payload["task_root"], payload["status"], _json(payload.get("plan") or {}),
                    _json(payload.get("attachment_ids") or []), int(payload.get("retry_count") or 0),
                    payload.get("error_class", ""), payload.get("error_message", ""),
                    payload.get("model_provider", ""), payload.get("model", ""),
                    payload["created_at"], payload["updated_at"], payload.get("completed_at", ""),
                ),
            )
        return self.get_assistant_run(str(payload["id"])) or {}

    def update_assistant_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {
            "status", "plan_json", "attachment_ids_json", "retry_count", "error_class",
            "error_message", "model_provider", "model", "completed_at",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            mapped = "plan_json" if key == "plan" else "attachment_ids_json" if key == "attachment_ids" else key
            if mapped not in allowed:
                continue
            updates[mapped] = _json(value) if mapped.endswith("_json") and not isinstance(value, str) else value
        updates["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in updates)
        with self._lock, self._connection:
            self._connection.execute(
                f"UPDATE assistant_runs SET {assignments} WHERE id=?",
                (*updates.values(), run_id),
            )
        result = self.get_assistant_run(run_id)
        if result is None:
            raise KeyError(run_id)
        return result

    def settle_planning_if_unchanged(self, run_id: str, observed_updated_at: str, **fields: Any) -> bool:
        """Write a planning row only when status and updated_at are still the observed pair."""
        allowed = {
            "status", "plan_json", "attachment_ids_json", "retry_count", "error_class",
            "error_message", "model_provider", "model", "completed_at",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            mapped = "plan_json" if key == "plan" else "attachment_ids_json" if key == "attachment_ids" else key
            if mapped not in allowed:
                continue
            updates[mapped] = _json(value) if mapped.endswith("_json") and not isinstance(value, str) else value
        if not updates:
            return False
        updates["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in updates)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                f"UPDATE assistant_runs SET {assignments} WHERE id=? AND status='planning' AND updated_at=?",
                (*updates.values(), run_id, str(observed_updated_at)),
            )
            return cursor.rowcount == 1

    def get_assistant_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = _row(self._connection.execute("SELECT * FROM assistant_runs WHERE id=?", (run_id,)).fetchone())
        if item:
            item["plan"] = json.loads(item.pop("plan_json"))
            item["attachment_ids"] = json.loads(item.pop("attachment_ids_json"))
        return item

    def list_assistant_runs(self, conversation_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM assistant_runs"
        params: list[Any] = []
        if conversation_id:
            query += " WHERE conversation_id=?"
            params.append(conversation_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for item in map(dict, rows):
            item["plan"] = json.loads(item.pop("plan_json"))
            item["attachment_ids"] = json.loads(item.pop("attachment_ids_json"))
            result.append(item)
        return result

    def list_assistant_runs_with_status(self, status: str, limit: int = 5000) -> list[dict[str, Any]]:
        """Every run in one status, oldest first. The conversation list is capped at 200."""
        capped = max(1, min(int(limit), 5000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM assistant_runs WHERE status=? ORDER BY updated_at ASC, id ASC LIMIT ?",
                (str(status), capped),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for item in map(dict, rows):
            item["plan"] = json.loads(item.pop("plan_json"))
            item["attachment_ids"] = json.loads(item.pop("attachment_ids_json"))
            result.append(item)
        return result

    def list_assistant_runs_by_status(self, statuses: tuple[str, ...] | list[str], *, since: str = "",
                                      limit: int = 500) -> list[dict[str, Any]]:
        """Runs in any of ``statuses`` updated at or after ``since``, oldest created first (run supervision v1)."""
        wanted = [str(item) for item in statuses if str(item)]
        if not wanted:
            return []
        marks = ",".join("?" for _ in wanted)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM assistant_runs WHERE status IN ({marks}) AND updated_at >= ? "
                "ORDER BY created_at ASC, id ASC LIMIT ?",
                (*wanted, str(since or ""), max(1, min(int(limit), 5000))),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for item in map(dict, rows):
            item["plan"] = json.loads(item.pop("plan_json"))
            item["attachment_ids"] = json.loads(item.pop("attachment_ids_json"))
            result.append(item)
        return result

    @staticmethod
    def _decode_goal_row(row: sqlite3.Row | dict[str, Any] | None) -> dict[str, Any] | None:
        item = _row(row) if isinstance(row, sqlite3.Row) else dict(row) if row is not None else None
        if item is None:
            return None
        item["spec"] = json.loads(item.pop("spec_json"))
        item["board"] = json.loads(item.pop("board_json"))
        return item

    def get_goal_record(self, goal_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM goal_records WHERE id=?",
                (str(goal_id),),
            ).fetchone()
        return self._decode_goal_row(row)

    def get_goal_for_run(self, run_id: str, goal_key: str = "g21_five_competition") -> dict[str, Any] | None:
        query = "SELECT * FROM goal_records WHERE run_id=?"
        params: list[Any] = [str(run_id)]
        if goal_key:
            query += " AND goal_key=?"
            params.append(str(goal_key))
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        return self._decode_goal_row(row)

    def list_goal_records(self, run_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM goal_records"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id=?"
            params.append(str(run_id))
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [item for item in (self._decode_goal_row(row) for row in rows) if item is not None]

    def goal_records_schema_issues(self) -> tuple[str, ...]:
        """Return deterministic issues for legacy or weakened Goal tables."""

        issues: list[str] = []
        with self._lock:
            try:
                table_info = self._connection.execute("PRAGMA table_info(goal_records)").fetchall()
                columns = {str(row[1]) for row in table_info}
                required = {
                    "id", "run_id", "goal_key", "schema", "allocation", "spec_sha256", "spec_json",
                    "board_sha256", "human_baseline_sha256", "board_json", "status", "created_at", "updated_at",
                }
                if not columns:
                    return ("GOAL_RECORD_TABLE_MISSING",)
                if required - columns:
                    issues.append("GOAL_RECORD_COLUMNS_INCOMPLETE")
                by_name = {str(row[1]): row for row in table_info}
                if "id" in by_name and int(by_name["id"][5]) != 1:
                    issues.append("GOAL_RECORD_ID_NOT_PRIMARY_KEY")
                nullable = sorted(
                    name
                    for name in required - {"id"}
                    if name in by_name and int(by_name[name][3]) != 1
                )
                if nullable:
                    issues.append("GOAL_RECORD_REQUIRED_COLUMNS_NULLABLE")
                unique_goal_key = False
                for index in self._connection.execute("PRAGMA index_list(goal_records)").fetchall():
                    if int(index[2]) != 1:
                        continue
                    index_columns = [
                        str(row[2])
                        for row in self._connection.execute(f"PRAGMA index_info([{index[1]}])").fetchall()
                    ]
                    if index_columns == ["goal_key"]:
                        unique_goal_key = True
                if not unique_goal_key:
                    issues.append("GOAL_RECORD_GOAL_KEY_NOT_UNIQUE")
                foreign_keys = self._connection.execute("PRAGMA foreign_key_list(goal_records)").fetchall()
                if not any(
                    str(row[2]) == "assistant_runs"
                    and str(row[3]) == "run_id"
                    and str(row[4]) == "id"
                    for row in foreign_keys
                ):
                    issues.append("GOAL_RECORD_RUN_FOREIGN_KEY_MISSING")
            except sqlite3.Error:
                issues.append("GOAL_RECORD_SCHEMA_QUERY_FAILED")
        return tuple(dict.fromkeys(issues))

    def ensure_goal_record(
        self,
        payload: dict[str, Any],
        *,
        event_payload: dict[str, Any] | None = None,
        event_type: str = "goal.created",
        session_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically create or return an identity-matching Goal for an existing Run."""

        required = (
            "id", "run_id", "goal_key", "schema", "allocation", "spec_sha256",
            "spec_json", "board_sha256", "human_baseline_sha256", "board_json",
            "status", "created_at", "updated_at",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError("goal record fields missing: " + ",".join(missing))
        run_id = str(payload["run_id"])
        goal_key = str(payload["goal_key"])
        goal_id = str(payload["id"])
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN IMMEDIATE")  # t238 C7e A3: checks + seq under one write lock
            schema_issues = self.goal_records_schema_issues()
            if schema_issues:
                raise ValueError("goal_records schema invalid: " + ",".join(schema_issues))
            if self._connection.execute("SELECT 1 FROM assistant_runs WHERE id=?", (run_id,)).fetchone() is None:
                raise KeyError(run_id)
            session_row = self._connection.execute(
                "SELECT metadata_json FROM sessions WHERE id=?",
                (run_id,),
            ).fetchone()
            if session_row is None:
                raise KeyError(run_id)
            if session_metadata is not None:
                try:
                    current_metadata = json.loads(str(session_row[0] or "{}"))
                except json.JSONDecodeError as exc:
                    raise ValueError("session metadata JSON is invalid") from exc
                if not isinstance(current_metadata, dict):
                    raise ValueError("session metadata must be an object")
                for key, value in session_metadata.items():
                    if key in current_metadata and current_metadata[key] not in (None, value):
                        raise ValueError(f"existing session metadata conflicts with {key}")
                merged_metadata = {**current_metadata, **session_metadata}
                self._connection.execute(
                    "UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?",
                    (_json(merged_metadata), utc_now(), run_id),
                )
            existing_rows = self._connection.execute(
                "SELECT * FROM goal_records WHERE goal_key=? ORDER BY created_at",
                (goal_key,),
            ).fetchall()
            if len(existing_rows) > 1:
                raise ValueError("duplicate goal records")
            if existing_rows:
                existing_item = self._decode_goal_row(existing_rows[0])
                if (
                    str(existing_item.get("run_id")) != run_id
                    or str(existing_item.get("goal_key")) != goal_key
                    or str(existing_item.get("id")) != goal_id
                    or str(existing_item.get("schema")) != str(payload["schema"])
                    or str(existing_item.get("allocation")) != str(payload["allocation"])
                    or str(existing_item.get("spec_sha256")) != str(payload["spec_sha256"])
                    or str(existing_item.get("board_sha256")) != str(payload["board_sha256"])
                    or str(existing_item.get("human_baseline_sha256")) != str(payload["human_baseline_sha256"])
                ):
                    raise ValueError("goal identity conflict")
                from .competition_goal import validate_goal_record

                check = validate_goal_record(
                    {
                        "schema": existing_item["schema"],
                        "goal_id": existing_item["id"],
                        "run_id": existing_item["run_id"],
                        "allocation": existing_item["allocation"],
                        "goal_key": existing_item["goal_key"],
                        "status": existing_item["status"],
                        "spec_sha256": existing_item["spec_sha256"],
                        "spec": existing_item["spec"],
                        "board_sha256": existing_item["board_sha256"],
                        "human_baseline_sha256": existing_item["human_baseline_sha256"],
                        "board": existing_item["board"],
                    }
                )
                if not check.valid:
                    raise ValueError("existing goal record invalid: " + ",".join(item.code for item in check.issues))
                return existing_item, False
            try:
                spec = json.loads(str(payload["spec_json"]))
                board = json.loads(str(payload["board_json"]))
            except json.JSONDecodeError as exc:
                raise ValueError("goal record JSON is invalid") from exc
            from .competition_goal import validate_goal_record

            check = validate_goal_record(
                {
                    "schema": payload["schema"],
                    "goal_id": goal_id,
                    "run_id": run_id,
                    "allocation": payload["allocation"],
                    "goal_key": goal_key,
                    "status": payload["status"],
                    "spec_sha256": payload["spec_sha256"],
                    "spec": spec,
                    "board_sha256": payload["board_sha256"],
                    "human_baseline_sha256": payload["human_baseline_sha256"],
                    "board": board,
                }
            )
            if not check.valid:
                raise ValueError("goal record invalid: " + ",".join(item.code for item in check.issues))
            bound_id = self._connection.execute(
                "SELECT run_id,goal_key FROM goal_records WHERE id=? LIMIT 1",
                (goal_id,),
            ).fetchone()
            if bound_id is not None:
                raise ValueError("goal id is already bound to another record")
            if event_payload is None:
                event_payload = {
                    "goal_id": goal_id,
                    "goal_key": goal_key,
                    "allocation": str(payload["allocation"]),
                    "spec_sha256": str(payload["spec_sha256"]),
                    "board_sha256": str(payload["board_sha256"]),
                }
            self._connection.execute(
                "INSERT INTO goal_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    goal_id, run_id, goal_key, str(payload["schema"]), str(payload["allocation"]),
                    str(payload["spec_sha256"]), str(payload["spec_json"]), str(payload["board_sha256"]),
                    str(payload["human_baseline_sha256"]), str(payload["board_json"]), str(payload["status"]),
                    str(payload["created_at"]),
                    str(payload["updated_at"]),
                ),
            )
            if event_payload is not None:
                event = RunEvent(
                    new_id("evt"),
                    run_id,
                    self.next_seq(run_id),
                    event_type,
                    event_payload,
                )
                self._connection.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?)",
                    (event.id, event.session_id, event.seq, event.event_type, _json(event.payload), event.created_at),
                )
            created = self._connection.execute(
                "SELECT * FROM goal_records WHERE id=?",
                (goal_id,),
            ).fetchone()
        item = self._decode_goal_row(created)
        if item is None:
            raise RuntimeError("goal record insert was not readable")
        return item, True

    def update_goal_record(self, goal_id: str, *, status: str | None = None, board: Any = None,
                           board_sha256: str | None = None, updated_at: str | None = None,
                           human_baseline_sha256: str | None = None,
                           session_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = str(status)
        if board is not None:
            if board_sha256 is None:
                raise ValueError("board_sha256 is required when updating a goal board")
            from .competition_goal import sha256_json

            if sha256_json(board) != str(board_sha256).lower():
                raise ValueError("goal board hash mismatch")
            updates["board_json"] = _json(board)
        if board_sha256 is not None:
            updates["board_sha256"] = str(board_sha256)
        if human_baseline_sha256 is not None:
            updates["human_baseline_sha256"] = str(human_baseline_sha256)
        updates["updated_at"] = str(updated_at or utc_now())
        assignments = ",".join(f"{key}=?" for key in updates)
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN IMMEDIATE")  # t238 C7e A3: checks + seq under one write lock
            previous = self._connection.execute(
                "SELECT * FROM goal_records WHERE id=?",
                (str(goal_id),),
            ).fetchone()
            if previous is None:
                raise KeyError(goal_id)
            previous_item = self._decode_goal_row(previous)
            next_status = str(updates.get("status", previous_item["status"]))
            next_board = previous_item["board"]
            if "board_json" in updates:
                next_board = json.loads(str(updates["board_json"]))
            from .competition_goal import validate_goal_record

            check = validate_goal_record(
                {
                    "schema": previous_item["schema"],
                    "goal_id": previous_item["id"],
                    "run_id": previous_item["run_id"],
                    "allocation": previous_item["allocation"],
                    "goal_key": previous_item["goal_key"],
                    "status": next_status,
                    "spec_sha256": previous_item["spec_sha256"],
                    "spec": previous_item["spec"],
                    "board_sha256": str(updates.get("board_sha256", previous_item["board_sha256"])),
                    "human_baseline_sha256": str(
                        updates.get("human_baseline_sha256", previous_item["human_baseline_sha256"])
                    ),
                    "board": next_board,
                }
            )
            if not check.valid:
                raise ValueError("goal record invalid: " + ",".join(item.code for item in check.issues))
            record_changed = (
                str(previous_item["status"]) != next_status
                or str(previous_item["board_sha256"])
                != str(updates.get("board_sha256", previous_item["board_sha256"]))
                or str(previous_item["human_baseline_sha256"])
                != str(updates.get("human_baseline_sha256", previous_item["human_baseline_sha256"]))
            )
            if record_changed:
                self._connection.execute(
                    f"UPDATE goal_records SET {assignments} WHERE id=?",
                    (*updates.values(), str(goal_id)),
                )
            if session_metadata is not None:
                session_row = self._connection.execute(
                    "SELECT metadata_json FROM sessions WHERE id=?",
                    (str(previous_item["run_id"]),),
                ).fetchone()
                if session_row is None:
                    raise KeyError(str(previous_item["run_id"]))
                try:
                    current_metadata = json.loads(str(session_row[0] or "{}"))
                except json.JSONDecodeError as exc:
                    raise ValueError("session metadata JSON is invalid") from exc
                if not isinstance(current_metadata, dict):
                    raise ValueError("session metadata must be an object")
                for key, value in session_metadata.items():
                    replacing_board_hash = (
                        key == "goal_board_sha256"
                        and current_metadata.get(key) == previous_item["board_sha256"]
                    )
                    replacing_human_hash = (
                        key == "goal_human_baseline_sha256"
                        and current_metadata.get(key) == previous_item["human_baseline_sha256"]
                    )
                    if (
                        key in current_metadata
                        and current_metadata[key] not in (None, value)
                        and not replacing_board_hash
                        and not replacing_human_hash
                    ):
                        raise ValueError(f"existing session metadata conflicts with {key}")
                merged_metadata = {**current_metadata, **session_metadata}
                if merged_metadata != current_metadata:
                    self._connection.execute(
                        "UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?",
                        (_json(merged_metadata), utc_now(), str(previous_item["run_id"])),
                    )
            if record_changed:
                event = RunEvent(
                    new_id("evt"),
                    str(previous_item["run_id"]),
                    self.next_seq(str(previous_item["run_id"])),
                    "goal.updated",
                    {
                        "goal_id": str(goal_id),
                        "status": next_status,
                        "board_sha256": str(updates.get("board_sha256", previous_item["board_sha256"])),
                        "human_baseline_sha256": str(
                            updates.get("human_baseline_sha256", previous_item["human_baseline_sha256"])
                        ),
                    },
                )
                self._connection.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?)",
                    (event.id, event.session_id, event.seq, event.event_type, _json(event.payload), event.created_at),
                )
        result = self.get_goal_record(str(goal_id))
        if result is None:
            raise KeyError(goal_id)
        return result

    def migrate_goal_policy(
        self,
        goal_id: str,
        *,
        run_id: str,
        expected_spec_sha256: str,
        new_spec: dict[str, Any],
        policy_evidence: dict[str, Any],
        migration_id: str,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically migrate the one fixed Goal policy and its metadata/event.

        ``migration_id`` is the idempotency identity.  Only the exact initial
        production spec may be migrated; a completed migration can be replayed
        only with byte-equivalent canonical inputs.
        """

        from .competition_goal import (
            CONDITIONAL_BASELINE_MIGRATION_ID,
            CONDITIONAL_GOAL_SPEC_CANONICAL_SHA256,
            FIXED_ALLOCATION,
            FIXED_GOAL_ID,
            FIXED_GOAL_KEY,
            FIXED_RUN_ID,
            INITIAL_GOAL_SPEC_CANONICAL_SHA256,
            conditional_baseline_policy,
            sha256_json,
            validate_goal_record,
            validate_goal_spec,
            validate_policy_migration_evidence,
        )

        if (
            str(goal_id) != FIXED_GOAL_ID
            or str(run_id) != FIXED_RUN_ID
            or str(migration_id) != CONDITIONAL_BASELINE_MIGRATION_ID
        ):
            raise ValueError("goal policy migration identity mismatch")
        expected_spec_sha256 = str(expected_spec_sha256).lower()
        if expected_spec_sha256 != INITIAL_GOAL_SPEC_CANONICAL_SHA256:
            raise ValueError("goal policy migration old spec precondition mismatch")
        spec_ok, spec_issues = validate_goal_spec(new_spec)
        if not spec_ok or not conditional_baseline_policy(new_spec):
            raise ValueError(
                "goal policy migration new spec invalid: "
                + ",".join(item.code for item in spec_issues)
            )
        evidence_ok, evidence_issues = validate_policy_migration_evidence(
            policy_evidence,
            new_spec=new_spec,
        )
        if not evidence_ok:
            raise ValueError(
                "goal policy migration evidence invalid: "
                + ",".join(item.code for item in evidence_issues)
            )
        new_spec_sha256 = sha256_json(new_spec)
        if new_spec_sha256 != CONDITIONAL_GOAL_SPEC_CANONICAL_SHA256:
            raise ValueError("goal policy migration new spec immutability mismatch")
        policy_evidence_sha256 = sha256_json(policy_evidence)
        now = utc_now()
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN IMMEDIATE")  # t238 C7e A3: checks + seq under one write lock
            row = self._connection.execute(
                "SELECT * FROM goal_records WHERE id=?",
                (str(goal_id),),
            ).fetchone()
            if row is None:
                raise KeyError(goal_id)
            current = self._decode_goal_row(row)
            if (
                current is None
                or current.get("run_id") != FIXED_RUN_ID
                or current.get("goal_key") != FIXED_GOAL_KEY
                or str(current.get("allocation") or "").casefold()
                != FIXED_ALLOCATION.casefold()
            ):
                raise ValueError("goal policy migration cross-run binding rejected")
            if str(current.get("status") or "").casefold() != "blocked":
                raise ValueError("goal policy migration requires blocked Goal")
            event_rows = self._connection.execute(
                "SELECT * FROM events WHERE session_id=? AND event_type=? ORDER BY seq",
                (FIXED_RUN_ID, "goal.policy_migrated"),
            ).fetchall()
            migration_events: list[dict[str, Any]] = []
            for event_row in event_rows:
                event_item = dict(event_row)
                try:
                    event_payload = json.loads(str(event_item.get("payload_json") or "{}"))
                except json.JSONDecodeError as exc:
                    raise ValueError("goal policy migration event is invalid") from exc
                if event_payload.get("migration_id") == migration_id:
                    migration_events.append(event_payload)
            if str(current.get("spec_sha256") or "").lower() == new_spec_sha256:
                if len(migration_events) != 1 or len(event_rows) != 1:
                    raise ValueError("goal policy migration replay evidence invalid")
                replay = migration_events[0]
                if (
                    replay.get("old_spec_sha256") != expected_spec_sha256
                    or replay.get("new_spec_sha256") != new_spec_sha256
                    or replay.get("policy_evidence_sha256") != policy_evidence_sha256
                    or replay.get("policy_evidence_file_sha256")
                    != str(new_spec.get("policy_evidence_file_sha256") or "").lower()
                ):
                    raise ValueError("goal policy migration idempotency conflict")
                replay_session = self._connection.execute(
                    "SELECT metadata_json FROM sessions WHERE id=?",
                    (FIXED_RUN_ID,),
                ).fetchone()
                try:
                    replay_metadata = json.loads(str(replay_session[0] or "{}")) if replay_session else None
                except json.JSONDecodeError as exc:
                    raise ValueError("goal policy migration replay metadata invalid") from exc
                expected_replay_metadata = {
                    "goal_spec_sha256": new_spec_sha256,
                    "goal_previous_spec_sha256": expected_spec_sha256,
                    "goal_policy_migration_id": migration_id,
                    "goal_policy_evidence_sha256": policy_evidence_sha256,
                    "goal_policy_evidence_file_sha256": str(
                        new_spec.get("policy_evidence_file_sha256") or ""
                    ).lower(),
                }
                if not isinstance(replay_metadata, dict) or any(
                    replay_metadata.get(key) != value
                    for key, value in expected_replay_metadata.items()
                ):
                    raise ValueError("goal policy migration replay metadata invalid")
                return current, False
            if migration_events or event_rows:
                raise ValueError("goal policy migration event conflict")
            if str(current.get("spec_sha256") or "").lower() != expected_spec_sha256:
                raise ValueError("goal policy migration old spec precondition failed")
            if sha256_json(current.get("spec")) != INITIAL_GOAL_SPEC_CANONICAL_SHA256:
                raise ValueError("goal policy migration legacy spec bytes drift")
            candidate = {
                "schema": current.get("schema"),
                "goal_id": current.get("id"),
                "run_id": current.get("run_id"),
                "allocation": current.get("allocation"),
                "goal_key": current.get("goal_key"),
                "status": current.get("status"),
                "spec_sha256": new_spec_sha256,
                "spec": new_spec,
                "board_sha256": current.get("board_sha256"),
                "human_baseline_sha256": current.get("human_baseline_sha256"),
                "board": current.get("board"),
            }
            checked = validate_goal_record(candidate)
            if not checked.valid:
                raise ValueError(
                    "goal policy migration record invalid: "
                    + ",".join(item.code for item in checked.issues)
                )
            session_row = self._connection.execute(
                "SELECT metadata_json FROM sessions WHERE id=?",
                (FIXED_RUN_ID,),
            ).fetchone()
            if session_row is None:
                raise KeyError(FIXED_RUN_ID)
            try:
                metadata = json.loads(str(session_row[0] or "{}"))
            except json.JSONDecodeError as exc:
                raise ValueError("session metadata JSON is invalid") from exc
            if not isinstance(metadata, dict):
                raise ValueError("session metadata must be an object")
            if metadata.get("goal_spec_sha256") != expected_spec_sha256:
                raise ValueError("existing session metadata conflicts with goal_spec_sha256")
            migration_metadata = {
                "goal_spec_sha256": new_spec_sha256,
                "goal_previous_spec_sha256": expected_spec_sha256,
                "goal_policy_migration_id": migration_id,
                "goal_policy_evidence_sha256": policy_evidence_sha256,
                "goal_policy_evidence_file_sha256": str(
                    new_spec.get("policy_evidence_file_sha256") or ""
                ).lower(),
            }
            for key, value in migration_metadata.items():
                if key == "goal_spec_sha256":
                    continue
                if key in metadata and metadata[key] not in (None, value):
                    raise ValueError(f"existing session metadata conflicts with {key}")
            next_metadata = {**metadata, **migration_metadata}
            event_payload = {
                "goal_id": FIXED_GOAL_ID,
                "goal_key": FIXED_GOAL_KEY,
                "run_id": FIXED_RUN_ID,
                "allocation": FIXED_ALLOCATION,
                "migration_id": migration_id,
                "old_spec_sha256": expected_spec_sha256,
                "new_spec_sha256": new_spec_sha256,
                "policy_evidence_sha256": policy_evidence_sha256,
                "policy_evidence_file_sha256": migration_metadata[
                    "goal_policy_evidence_file_sha256"
                ],
                "board_sha256": str(current.get("board_sha256") or ""),
            }
            self._connection.execute(
                "UPDATE goal_records SET spec_sha256=?,spec_json=?,updated_at=? WHERE id=?",
                (new_spec_sha256, _json(new_spec), now, FIXED_GOAL_ID),
            )
            self._connection.execute(
                "UPDATE sessions SET metadata_json=?,updated_at=? WHERE id=?",
                (_json(next_metadata), now, FIXED_RUN_ID),
            )
            event = RunEvent(
                new_id("evt"),
                FIXED_RUN_ID,
                self.next_seq(FIXED_RUN_ID),
                "goal.policy_migrated",
                event_payload,
            )
            self._connection.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (
                    event.id,
                    event.session_id,
                    event.seq,
                    event.event_type,
                    _json(event.payload),
                    event.created_at,
                ),
            )
        result = self.get_goal_record(FIXED_GOAL_ID)
        if result is None:
            raise KeyError(FIXED_GOAL_ID)
        return result, True

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

    @staticmethod
    def _same_tool_request(current: dict[str, Any], call: ToolCall) -> bool:
        from .policy import argument_fingerprint

        return (
            current["session_id"] == call.session_id
            and current["tool_name"] == call.tool_name
            and current["idempotency_key"] == call.idempotency_key
            and current["arguments"] == call.arguments
            and argument_fingerprint(current["tool_name"], current["arguments"])
            == argument_fingerprint(call.tool_name, call.arguments)
        )

    def _write_tool_call(self, call: ToolCall, result: ToolResult | None = None) -> None:
        """Write only inside the caller's admission/settlement transaction."""
        self._connection.execute(
            "INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status,result_json=excluded.result_json,"
            "started_at=excluded.started_at,completed_at=excluded.completed_at,approval_id=excluded.approval_id",
            (call.id, call.session_id, call.tool_name, _json(call.arguments), call.status,
             _json(result.to_dict() if result else {}), call.created_at, call.started_at,
             call.completed_at, call.approval_id, call.idempotency_key),
        )

    def put_tool_call(self, call: ToolCall, result: ToolResult | None = None) -> None:
        """Persist evidence monotonically; a stale copy cannot reopen execution."""
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN IMMEDIATE")
            current = self.get_tool_call(call.id)
            if current:
                if not self._same_tool_request(current, call):
                    raise ValueError("tool_call_identity_immutable")
                if current["status"] in {"completed", "failed"}:
                    if (call.status != current["status"] or result is None
                            or result.to_dict() != current["result"]):
                        raise ValueError("tool_call_terminal_immutable")
                    return
                if current["status"] == "running" or current["started_at"]:
                    if call.status == "running" and result is None:
                        if call.started_at != current["started_at"] or call.approval_id != current["approval_id"]:
                            raise ValueError("tool_call_execution_identity_immutable")
                        return
                    if call.status not in {"completed", "failed"} or result is None:
                        raise ValueError("tool_call_execution_cannot_reopen")
                    if (call.started_at != current["started_at"] or call.approval_id != current["approval_id"]
                            or result.tool_call_id != call.id or result.ok != (call.status == "completed")):
                        raise ValueError("tool_call_settlement_identity_mismatch")
                elif current["status"] == "waiting_approval" and call.status in {"requested", "queued", "waiting_approval"}:
                    if call.status != "waiting_approval" or call.approval_id != current["approval_id"]:
                        raise ValueError("tool_call_approval_identity_immutable")
                    return
            self._write_tool_call(call, result)

    def admit_tool_call(
        self, call: ToolCall, result: ToolResult | None = None, *,
        approval: ApprovalRequest | None = None, approved_fingerprint: str = "", allow_paused: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        """Atomically reserve this logical action across processes, without TTL.

        SQLite's writer transaction ends before registry.invoke.  A durable
        running row is never expired/reclaimed: unknown execution needs manual
        reconciliation, not another executor.  Existing schemas need no rewrite.
        """
        from .policy import argument_fingerprint

        if call.status not in {"running", "waiting_approval", "failed"}:
            raise ValueError("tool_call_admission_state_invalid")
        if call.status == "running" and (not call.started_at or result is not None):
            raise ValueError("tool_call_admission_state_invalid")
        if call.status == "failed" and (result is None or result.ok or result.tool_call_id != call.id):
            raise ValueError("tool_call_admission_result_invalid")
        if approval is not None and not (
            call.status == "waiting_approval" and approval.status == "pending" and call.approval_id == approval.id
            and approval.session_id == call.session_id and approval.tool_call_id == call.id
            and approval.tool_name == call.tool_name and approval.normalized_arguments == call.arguments
            and approval.argument_fingerprint == argument_fingerprint(call.tool_name, call.arguments)
        ):
            raise ValueError("tool_call_admission_approval_invalid")
        with self._lock, self._connection:
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN IMMEDIATE")
            prior = self.get_idempotent_tool_call(call.session_id, call.idempotency_key)
            if prior and not self._same_tool_request(prior, call):
                raise ValueError("tool_call_idempotency_conflict")
            if prior and prior["id"] != call.id:
                old_approval = self.get_approval(prior["approval_id"]) if prior["approval_id"] else None
                try:
                    old_deadline = datetime.fromisoformat(str((old_approval or {}).get("expires_at") or "").replace("Z", "+00:00"))
                    if old_deadline.tzinfo is None:
                        old_deadline = old_deadline.replace(tzinfo=timezone.utc)
                except ValueError:
                    old_deadline = datetime.min.replace(tzinfo=timezone.utc)
                fresh_gate = (call.status == "waiting_approval" and prior["status"] == "waiting_approval"
                              and not prior["started_at"] and old_approval
                              and (old_approval["status"] in {"expired", "rejected"}
                                   or old_deadline <= datetime.now(timezone.utc)))
                if not fresh_gate:
                    return False, prior
            current = self.get_tool_call(call.id)
            if current:
                if not self._same_tool_request(current, call):
                    raise ValueError("tool_call_identity_immutable")
                if current["status"] in {"running", "completed", "failed"} or current["started_at"]:
                    return False, current
                if current["status"] == "waiting_approval" and call.status != "running":
                    return False, current
                call.created_at = current["created_at"]
            session = self.get_session(call.session_id)
            if not session:
                raise KeyError(call.session_id)
            if session["status"] == "cancelled":
                return False, {"status": "cancelled", "error": "user_cancel_requested", "dispatched": False}
            if session["metadata"].get("user_pause_requested") and not allow_paused:
                return False, {"status": "paused", "error": "user_pause_requested", "dispatched": False}
            if call.status == "running" and (call.approval_id or (current and current["status"] == "waiting_approval")):
                saved = self.get_approval(call.approval_id) if call.approval_id else None
                try:
                    expires = datetime.fromisoformat(str((saved or {}).get("expires_at") or "").replace("Z", "+00:00"))
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                except ValueError:
                    expires = datetime.min.replace(tzinfo=timezone.utc)
                if not (saved and current and current["approval_id"] == call.approval_id
                        and saved["status"] == "approved" and saved["session_id"] == call.session_id
                        and saved["tool_call_id"] == call.id and saved["tool_name"] == call.tool_name
                        and saved["normalized_arguments"] == call.arguments
                        and saved["argument_fingerprint"] == approved_fingerprint
                        == argument_fingerprint(call.tool_name, call.arguments)
                        and expires > datetime.now(timezone.utc)):
                    raise ValueError("tool_approval_reconciliation_required")
            self._write_tool_call(call, result)
            if approval is not None:
                self._connection.execute(
                    "INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (approval.id, approval.session_id, approval.tool_call_id, approval.tool_name,
                     approval.argument_fingerprint, _json(approval.normalized_arguments), _json(approval.impact_scope),
                     approval.risk_level, int(approval.reversible), approval.status, approval.created_at,
                     approval.expires_at, approval.decided_at, approval.decision_note),
                )
            return True, self.get_tool_call(call.id)

    def get_tool_call(self, call_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = _row(self._connection.execute("SELECT * FROM tool_calls WHERE id=?", (call_id,)).fetchone())
        if item:
            item["arguments"] = json.loads(item.pop("arguments_json"))
            item["result"] = json.loads(item.pop("result_json"))
        return item

    def list_tool_calls(self, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tool_calls WHERE session_id=? ORDER BY created_at LIMIT ?",
                (session_id, max(1, min(int(limit), 2000))),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for item in map(dict, rows):
            item["arguments"] = json.loads(item.pop("arguments_json"))
            item["result"] = json.loads(item.pop("result_json"))
            result.append(item)
        return result

    def list_failed_tool_calls(self, session_id: str, tool_name: str, *, since: str = "", limit: int = 400) -> list[dict[str, Any]]:
        """Failed calls of one tool in one session after ``since`` (run supervision v1 tool breaker)."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, session_id, tool_name, status, created_at, arguments_json, json_extract(result_json,'$.error') "
                "FROM tool_calls WHERE session_id=? AND tool_name=? AND status='failed' AND created_at > ? "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, tool_name, str(since or ""), max(1, min(int(limit), 2000))),
            ).fetchall()
        return [{"id": row[0], "session_id": row[1], "tool_name": row[2], "status": row[3], "created_at": row[4],
                 "arguments": json.loads(row[5]), "result": {"error": row[6] or ""}} for row in rows]

    def get_idempotent_tool_call(self, session_id: str, key: str) -> dict[str, Any] | None:
        if not key:
            return None
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tool_calls WHERE session_id=? AND idempotency_key=? ORDER BY created_at DESC, rowid DESC LIMIT 65",
                (session_id, key),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 64:
            raise ValueError("idempotency_history_requires_reconciliation")
        calls = []
        for row in rows:
            item = dict(row)
            item["arguments"] = json.loads(item.pop("arguments_json"))
            item["result"] = json.loads(item.pop("result_json"))
            calls.append(item)
        if any(item["tool_name"] != calls[0]["tool_name"] or item["arguments"] != calls[0]["arguments"] for item in calls):
            raise ValueError("idempotency_request_history_conflict")
        executed = [item for item in calls if item.get("started_at")]
        if len(executed) > 1:
            raise ValueError("idempotency_multiple_executions_require_reconciliation")
        # Expired gates are historical attempts, not new logical operations.
        if executed:
            return executed[0]
        return next((item for item in calls if item["status"] in {"completed", "failed"}), calls[0])

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

    def list_session_approvals(self, session_id: str, status: str = "", limit: int = 200) -> list[dict[str, Any]]:
        """Approvals for one session. Callers that already skipped a live run should not scan the table."""
        query = "SELECT * FROM approvals WHERE session_id=?"
        params: list[Any] = [session_id]
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 2000)))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        result = []
        for item in map(dict, rows):
            item["normalized_arguments"] = json.loads(item.pop("normalized_arguments_json"))
            item["impact_scope"] = json.loads(item.pop("impact_scope_json"))
            item["reversible"] = bool(item["reversible"])
            result.append(item)
        return result

    def has_open_child_run(self, parent_id: str) -> bool:
        """True when a child session or its assistant run is still outside a terminal status."""
        terminal = ("completed", "failed", "cancelled", "blocked")
        marks = ",".join("?" for _ in terminal)
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT 1 FROM sessions
                WHERE parent_session_id=? AND status NOT IN ({marks})
                UNION ALL
                SELECT 1 FROM assistant_runs AS runs
                JOIN sessions ON sessions.id = runs.session_id
                WHERE sessions.parent_session_id=? AND runs.status NOT IN ({marks})
                LIMIT 1
                """,
                (parent_id, *terminal, parent_id, *terminal),
            ).fetchone()
        return row is not None

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
        current = self.get_approval(approval_id)
        if current is None:
            raise KeyError(approval_id)
        if current["status"] != "pending":
            return current
        try:
            expires_at = datetime.fromisoformat(str(current.get("expires_at") or "").replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except ValueError:
            expires_at = datetime.min.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            with self._lock, self._connection:
                self._connection.execute(
                    "UPDATE approvals SET status='expired',decided_at=?,decision_note=? WHERE id=? AND status='pending'",
                    (utc_now(), "approval expired before decision", approval_id),
                )
            approval = self.get_approval(approval_id)
            if approval is None:
                raise KeyError(approval_id)
            self.append_event(approval["session_id"], "approval.expired", {"approval_id": approval_id})
            return approval
        status = "approved" if approved else "rejected"
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE approvals SET status=?,decided_at=?,decision_note=? WHERE id=? AND status='pending'",
                (status, utc_now(), note[:2000], approval_id),
            )
        approval = self.get_approval(approval_id)
        if approval is None:
            raise KeyError(approval_id)
        actual_status = str(approval["status"])
        self.append_event(approval["session_id"], f"approval.{actual_status}", {"approval_id": approval_id, "note": note[:500]})
        return approval

    def next_seq(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COALESCE(MAX(seq),0)+1 AS value FROM events WHERE session_id=?", (session_id,)).fetchone()
        return int(row["value"])

    def latest_event_seq(self, session_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(seq),0) AS value FROM events WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return int(row["value"])

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> RunEvent:
        """Append with `seq = MAX(seq)+1` under the database write lock (t238 C7b A3).

        BEGIN IMMEDIATE takes the write lock before the MAX read, so another connection (a
        second RuntimeStore on the same file) cannot allocate the same seq. A lock that
        outlives busy_timeout is retried a bounded number of times, then re-raised.
        """
        attempt = 0
        while True:
            try:
                with self._lock, self._connection:
                    if not self._connection.in_transaction:
                        self._connection.execute("BEGIN IMMEDIATE")
                    event = RunEvent(new_id("evt"), session_id, self.next_seq(session_id), event_type, payload)
                    self._connection.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?,?)",
                        (event.id, event.session_id, event.seq, event.event_type, _json(event.payload), event.created_at),
                    )
                return event
            except sqlite3.OperationalError as error:
                attempt += 1
                if not _is_locked(error) or attempt > APPEND_EVENT_LOCKED_RETRIES:
                    raise

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

    def latest_event_created_at(self, session_id: str) -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT MAX(created_at) AS created_at FROM events WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return ""
        return str(row["created_at"] or "")

    def latest_event(self, session_id: str, event_type: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                'SELECT * FROM events WHERE session_id=? AND event_type=? ORDER BY seq DESC LIMIT 1',
                (session_id, event_type),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item['payload'] = json.loads(item.pop('payload_json'))
        return item

    def checkpoint(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """t238 C7e A3: the checkpoint row and its `checkpoint.created` event share one seq, allocated
        under BEGIN IMMEDIATE (same bounded locked-retry as `append_event`)."""
        checkpoint_id = new_id("cp")
        attempt = 0
        while True:
            try:
                with self._lock, self._connection:
                    if not self._connection.in_transaction:
                        self._connection.execute("BEGIN IMMEDIATE")
                    event = RunEvent(new_id("evt"), session_id, self.next_seq(session_id), "checkpoint.created",
                                     {"checkpoint_id": checkpoint_id, "state": state})
                    self._connection.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                                             (checkpoint_id, session_id, event.seq, _json(state), event.created_at))
                    self._connection.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?,?)",
                        (event.id, event.session_id, event.seq, event.event_type, _json(event.payload), event.created_at),
                    )
                break
            except sqlite3.OperationalError as error:
                attempt += 1
                if not _is_locked(error) or attempt > APPEND_EVENT_LOCKED_RETRIES:
                    raise
        return {"id": checkpoint_id, "session_id": session_id, "seq": event.seq, "state": state, "created_at": event.created_at}

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

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Return one generic run artifact for internal verification.

        Generic artifacts back oversized tool results and are separate from
        user-facing deliverables.  Callers must not project the stored path;
        they must enforce the session binding and independently verify the
        stored bytes and SHA-256.
        """

        with self._lock:
            return _row(
                self._connection.execute(
                    "SELECT * FROM artifacts WHERE id=?",
                    (artifact_id,),
                ).fetchone()
            )

    def list_artifacts(self, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM artifacts WHERE session_id=? ORDER BY created_at LIMIT ?",
                (session_id, max(1, min(int(limit), 2000))),
            ).fetchall()
        return [dict(item) for item in rows]

    def put_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connection:
            self._connection.execute(
                _UPLOAD_INSERT,
                (
                    payload["id"], payload["original_name"], payload["safe_name"], payload["media_type"],
                    int(payload["total_bytes"]), int(payload["chunk_size"]), payload.get("expected_sha256", ""),
                    payload["status"], int(payload["chunk_count"]), payload.get("final_path", ""),
                    payload.get("final_sha256", ""), payload["created_at"], payload["updated_at"],
                ),
            )
        return self.get_upload(str(payload["id"])) or {}

    def update_upload(self, upload_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"status", "final_path", "final_sha256", "media_type", "updated_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in updates)
        with self._lock, self._connection:
            self._connection.execute(f"UPDATE uploads SET {assignments} WHERE id=?", (*updates.values(), upload_id))
        result = self.get_upload(upload_id)
        if result is None:
            raise KeyError(upload_id)
        return result

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        """The upload row plus `claim_id`, the live completion claim (from upload_claims, else None)."""
        with self._lock:
            row = _row(self._connection.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone())
            if row is None:
                return None
            claim = self._connection.execute(
                "SELECT claim_id FROM upload_claims WHERE upload_id=?", (upload_id,)
            ).fetchone()
        row["claim_id"] = str(claim[0]) if claim else None  # a legacy uploads.claim_id value is never trusted
        return row

    def _usage(self, tenant_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT bytes, files FROM tenant_usage WHERE tenant_id=?", (tenant_id,)
        ).fetchone()
        return {"tenant_id": tenant_id, "bytes": int(row[0] or 0) if row else 0, "files": int(row[1] or 0) if row else 0}

    def get_tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            return self._usage(tenant_id)

    def reserve_upload(
        self, payload: dict[str, Any], tenant_id: str, *, max_bytes: int, max_files: int
    ) -> tuple[bool, dict[str, Any]]:
        """Check the tenant quota and insert the upload + its reservation in one write transaction.

        BEGIN IMMEDIATE takes the database write lock before the read, so concurrent creators
        (threads or other connections) serialize and cannot over-commit. 0 = unlimited.
        Returns (accepted, usage) where usage is the state before a rejected reservation.
        """
        size = int(payload["total_bytes"])
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                usage = self._usage(tenant_id)
                if (max_bytes > 0 and usage["bytes"] + size > max_bytes) or (max_files > 0 and usage["files"] + 1 > max_files):
                    self._connection.rollback()
                    return False, usage
                self._connection.execute(
                    _UPLOAD_INSERT,
                    (
                        payload["id"], payload["original_name"], payload["safe_name"], payload["media_type"],
                        size, int(payload["chunk_size"]), payload.get("expected_sha256", ""),
                        payload["status"], int(payload["chunk_count"]), payload.get("final_path", ""),
                        payload.get("final_sha256", ""), payload["created_at"], payload["updated_at"],
                    ),
                )
                self._connection.execute(
                    "INSERT INTO upload_quota VALUES (?,?,?,?,?)",
                    (payload["id"], tenant_id, size, 1, payload["created_at"]),
                )
                self._connection.execute(
                    "INSERT INTO tenant_usage VALUES (?,?,?,?) ON CONFLICT(tenant_id) DO UPDATE SET "
                    "bytes=COALESCE(bytes,0)+excluded.bytes, files=COALESCE(files,0)+excluded.files, updated_at=excluded.updated_at",
                    (tenant_id, size, 1, utc_now()),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            return True, {"tenant_id": tenant_id, "bytes": usage["bytes"] + size, "files": usage["files"] + 1}

    def _release_locked(self, upload_id: str) -> bool:
        row = self._connection.execute(
            "SELECT tenant_id, bytes, files FROM upload_quota WHERE upload_id=?", (upload_id,)
        ).fetchone()
        if row is None:
            return False
        self._connection.execute("DELETE FROM upload_quota WHERE upload_id=?", (upload_id,))
        self._connection.execute(
            "UPDATE tenant_usage SET bytes=MAX(0,COALESCE(bytes,0)-?), files=MAX(0,COALESCE(files,0)-?), updated_at=? WHERE tenant_id=?",
            (int(row[1]), int(row[2]), utc_now(), row[0]),
        )
        return True

    def release_upload_reservation(self, upload_id: str) -> bool:
        with self._lock, self._connection:
            return self._release_locked(upload_id)

    def add_upload_reservation(self, upload_id: str, tenant_id: str, total_bytes: int) -> bool:
        """Backfill a ledger row for a legacy upload (no quota check). False if one already exists."""
        with self._lock, self._connection:
            inserted = self._connection.execute(
                "INSERT OR IGNORE INTO upload_quota VALUES (?,?,?,?,?)",
                (upload_id, tenant_id, int(total_bytes), 1, utc_now()),
            ).rowcount
        return bool(inserted)

    def uploads_without_reservation(self) -> list[dict[str, Any]]:
        marks = ",".join("?" for _ in _RESERVED_UPLOAD_STATUSES)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM uploads WHERE status IN ({marks}) AND id NOT IN (SELECT upload_id FROM upload_quota)",
                _RESERVED_UPLOAD_STATUSES,
            ).fetchall()
        return [dict(item) for item in rows]

    def reconcile_tenant_usage(self) -> dict[str, dict[str, int]]:
        """Recompute tenant_usage from the reservation ledger of live uploads; drop orphan rows."""
        marks = ",".join("?" for _ in _RESERVED_UPLOAD_STATUSES)
        with self._lock, self._connection:
            self._connection.execute(
                f"DELETE FROM upload_quota WHERE upload_id NOT IN (SELECT id FROM uploads WHERE status IN ({marks}))",
                _RESERVED_UPLOAD_STATUSES,
            )
            self._connection.execute("DELETE FROM tenant_usage")
            now = utc_now()
            self._connection.execute(
                "INSERT INTO tenant_usage SELECT tenant_id, SUM(bytes), SUM(files), ? FROM upload_quota GROUP BY tenant_id",
                (now,),
            )
            rows = self._connection.execute("SELECT tenant_id, bytes, files FROM tenant_usage").fetchall()
        return {str(row[0]): {"bytes": int(row[1] or 0), "files": int(row[2] or 0)} for row in rows}

    def transition_upload(self, upload_id: str, from_status: str, to_status: str) -> bool:
        """Compare-and-set on uploads.status; True only for the caller that made the transition."""
        with self._lock, self._connection:
            changed = self._connection.execute(
                "UPDATE uploads SET status=?, updated_at=? WHERE id=? AND status=?",
                (to_status, utc_now(), upload_id, from_status),
            ).rowcount
        return bool(changed)

    # Owner check used by every claim CAS: the row is completing AND upload_claims names this claim.
    _OWNED = "status='completing' AND EXISTS (SELECT 1 FROM upload_claims c WHERE c.upload_id=uploads.id AND c.claim_id=?)"

    def claim_upload(self, upload_id: str, claim_id: str) -> bool:
        """uploading → completing, owned by `claim_id` (the per-complete owner token)."""
        with self._lock, self._connection:
            changed = self._connection.execute(
                "UPDATE uploads SET status='completing', updated_at=? WHERE id=? AND status='uploading'",
                (utc_now(), upload_id),
            ).rowcount
            if changed:
                self._connection.execute(
                    "INSERT OR REPLACE INTO upload_claims VALUES (?,?,?)", (upload_id, claim_id, utc_now())
                )
        return bool(changed)

    def takeover_upload_claim(self, upload_id: str, old_claim: str | None, observed_updated_at: str, claim_id: str) -> bool:
        """Replace a stale claim only if it is exactly the one observed (owner and heartbeat unchanged)."""
        owner = (
            "NOT EXISTS (SELECT 1 FROM upload_claims c WHERE c.upload_id=uploads.id)" if old_claim is None
            else "EXISTS (SELECT 1 FROM upload_claims c WHERE c.upload_id=uploads.id AND c.claim_id=?)"
        )
        params: tuple[Any, ...] = (utc_now(), upload_id, observed_updated_at) + (() if old_claim is None else (old_claim,))
        with self._lock, self._connection:
            changed = self._connection.execute(
                f"UPDATE uploads SET updated_at=? WHERE id=? AND status='completing' AND updated_at=? AND {owner}", params
            ).rowcount
            if changed:
                self._connection.execute(
                    "INSERT OR REPLACE INTO upload_claims VALUES (?,?,?)", (upload_id, claim_id, utc_now())
                )
        return bool(changed)

    def heartbeat_upload_claim(self, upload_id: str, claim_id: str) -> bool:
        """Refresh a live claim; False means the claim was taken over or reset."""
        with self._lock, self._connection:
            return bool(self._connection.execute(
                f"UPDATE uploads SET updated_at=? WHERE id=? AND {self._OWNED}", (utc_now(), upload_id, claim_id)
            ).rowcount)

    def release_upload_claim(self, upload_id: str, claim_id: str) -> bool:
        """completing → uploading, only by the claim's owner."""
        with self._lock, self._connection:
            changed = self._connection.execute(
                f"UPDATE uploads SET status='uploading', updated_at=? WHERE id=? AND {self._OWNED}",
                (utc_now(), upload_id, claim_id),
            ).rowcount
            if changed:
                self._connection.execute("DELETE FROM upload_claims WHERE upload_id=?", (upload_id,))
        return bool(changed)

    def upload_claim_is(self, upload_id: str, claim_id: str) -> bool:
        with self._lock:
            return self._connection.execute(
                f"SELECT 1 FROM uploads WHERE id=? AND {self._OWNED}", (upload_id, claim_id)
            ).fetchone() is not None

    def commit_completed_upload(
        self, upload_id: str, claim_id: str, *, final_path: str, final_sha256: str, media_type: str, attachment: dict[str, Any]
    ) -> dict[str, Any] | None:
        """completing(claim_id) → complete and insert the library attachment in one transaction.

        None if this claim no longer owns the upload (taken over or reset); nothing is written then.
        """
        with self._lock, self._connection:
            changed = self._connection.execute(
                "UPDATE uploads SET status='complete', final_path=?, final_sha256=?, media_type=?, updated_at=? "
                f"WHERE id=? AND {self._OWNED}",
                (final_path, final_sha256, media_type, utc_now(), upload_id, claim_id),
            ).rowcount
            if not changed:
                return None
            self._connection.execute("DELETE FROM upload_claims WHERE upload_id=?", (upload_id,))
            self._connection.execute(
                "INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    attachment["id"], upload_id, attachment.get("run_id", ""), attachment["name"],
                    attachment["path"], attachment["media_type"], int(attachment["bytes"]), attachment["sha256"],
                    attachment["created_at"],
                ),
            )
        return self.get_upload(upload_id)

    def reset_completing_uploads(self) -> list[str]:
        with self._lock, self._connection:
            ids = [str(row[0]) for row in self._connection.execute("SELECT id FROM uploads WHERE status='completing'")]
            self._connection.execute(
                "UPDATE uploads SET status='uploading', updated_at=? WHERE status='completing'", (utc_now(),)
            )
            # No upload is completing any more: every claim row (incl. orphans from a rollback) is dead.
            self._connection.execute("DELETE FROM upload_claims")
        return ids

    def list_stale_uploads(self, cutoff: str, statuses: tuple[str, ...] = ("uploading", "completing")) -> list[dict[str, Any]]:
        """Incomplete uploads with no upload or chunk activity since `cutoff` (ISO-8601 UTC)."""
        marks = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM uploads u WHERE u.status IN ({marks}) AND u.updated_at < ? "
                "AND NOT EXISTS (SELECT 1 FROM upload_chunks c WHERE c.upload_id=u.id AND c.created_at >= ?)",
                (*statuses, cutoff, cutoff),
            ).fetchall()
        return [dict(item) for item in rows]

    def delete_abandoned_upload(
        self, upload_id: str, cutoff: str, statuses: tuple[str, ...] = ("uploading", "completing")
    ) -> bool:
        """Delete an incomplete upload's chunk rows, reservation and row in one transaction.

        Staleness is re-checked inside the transaction, so activity after the listing wins.
        """
        marks = ",".join("?" for _ in statuses)
        with self._lock, self._connection:
            row = self._connection.execute(
                f"SELECT 1 FROM uploads u WHERE u.id=? AND u.status IN ({marks}) AND u.updated_at < ? "
                "AND NOT EXISTS (SELECT 1 FROM upload_chunks c WHERE c.upload_id=u.id AND c.created_at >= ?)",
                (upload_id, *statuses, cutoff, cutoff),
            ).fetchone()
            if row is None or self._connection.execute(
                "SELECT 1 FROM attachments WHERE upload_id=? LIMIT 1", (upload_id,)
            ).fetchone():
                return False
            self._release_locked(upload_id)
            self._connection.execute("DELETE FROM upload_claims WHERE upload_id=?", (upload_id,))
            self._connection.execute("DELETE FROM upload_chunks WHERE upload_id=?", (upload_id,))
            self._connection.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
        return True

    def list_uploads_with_status(self, status: str, limit: int = 5000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM uploads WHERE status=? ORDER BY updated_at LIMIT ?", (status, max(1, int(limit)))
            ).fetchall()
        return [dict(item) for item in rows]

    def upload_tenant(self, upload_id: str) -> str:
        """The tenant key the upload's quota reservation is charged to ('' when none)."""
        with self._lock:
            row = self._connection.execute("SELECT tenant_id FROM upload_quota WHERE upload_id=?", (upload_id,)).fetchone()
        return str(row[0]) if row else ""

    def mark_upload_deleting(self, upload_id: str) -> str:
        """CAS complete/uploading → deleting (t238 C7b D6). Returns the resulting status ('' if no row).

        The reservation stays until the rows are removed, so a crash here keeps usage honest;
        reconcile releases it (`deleting` is not a reserved status) and start-up finishes the delete.
        """
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE uploads SET status='deleting',updated_at=? WHERE id=? AND status IN ('complete','uploading')",
                (utc_now(), upload_id),
            )
            row = self._connection.execute("SELECT status FROM uploads WHERE id=?", (upload_id,)).fetchone()
        return str(row[0]) if row else ""

    def unmark_upload_deleting(self, upload_id: str) -> bool:
        """CAS deleting → complete (t238 C7b D6-M1: a delete refused before any file was removed)."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE uploads SET status='complete',updated_at=? WHERE id=? AND status='deleting' AND final_path<>''",
                (utc_now(), upload_id),
            )
        return cursor.rowcount == 1

    def finish_upload_delete(self, upload_id: str, log_entry: dict[str, Any]) -> bool:
        """Remove every row of a `deleting` upload, release its usage and append deletion_log — one transaction."""
        with self._lock, self._connection:
            row = self._connection.execute("SELECT status FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if row is None or row[0] != "deleting":
                return False
            self._release_locked(upload_id)
            for table, column in (("attachments", "upload_id"), ("upload_chunks", "upload_id"),
                                  ("upload_claims", "upload_id"), ("uploads", "id")):
                self._connection.execute(f"DELETE FROM {table} WHERE {column}=?", (upload_id,))
            self._connection.execute(
                "INSERT INTO deletion_log VALUES (?,?,?,?,?,?,?,?,?)",
                (new_id("deletion"), log_entry["tenant_id"], upload_id, log_entry["attachment_id"],
                 int(log_entry["bytes"]), int(log_entry["files"]), utc_now(), log_entry["actor"], log_entry["name_sha256"]),
            )
        return True

    def deleted_upload_ids(self) -> list[tuple[str, str]]:
        """(upload_id, library attachment_id) of every logged delete (t238 C7b H2 owner-row repair)."""
        with self._lock:
            rows = self._connection.execute("SELECT DISTINCT upload_id, attachment_id FROM deletion_log").fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def deletion_logged(self, upload_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM deletion_log WHERE upload_id=? ORDER BY deleted_at DESC LIMIT 1", (upload_id,)
            ).fetchone()
        return _row(row)

    def health_probe(self) -> None:
        """Write + read inside a transaction that is always rolled back (t238 C7b O2); no business table touched."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("INSERT INTO health_probe(probed_at) VALUES (?)", (utc_now(),))
                self._connection.execute("SELECT COUNT(*) FROM health_probe").fetchone()
            finally:
                self._connection.rollback()

    def queue_depth(self) -> dict[str, int]:
        with self._lock:
            pending = self._connection.execute(
                "SELECT COUNT(*) FROM assistant_runs WHERE status NOT IN ('completed','blocked','failed','cancelled')"
            ).fetchone()[0]
            completing = self._connection.execute("SELECT COUNT(*) FROM uploads WHERE status='completing'").fetchone()[0]
            deleting = self._connection.execute("SELECT COUNT(*) FROM uploads WHERE status='deleting'").fetchone()[0]
        return {"active_runs": int(pending), "uploads_completing": int(completing), "uploads_deleting": int(deleting)}

    def put_upload_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO upload_chunks VALUES (?,?,?,?,?,?)",
                (
                    payload["upload_id"], int(payload["chunk_index"]), int(payload["bytes"]),
                    payload["sha256"], payload["path"], payload["created_at"],
                ),
            )
        return dict(payload)

    def get_upload_chunk(self, upload_id: str, chunk_index: int) -> dict[str, Any] | None:
        with self._lock:
            return _row(self._connection.execute(
                "SELECT * FROM upload_chunks WHERE upload_id=? AND chunk_index=?",
                (upload_id, int(chunk_index)),
            ).fetchone())

    def list_upload_chunks(self, upload_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM upload_chunks WHERE upload_id=? ORDER BY chunk_index",
                (upload_id,),
            ).fetchall()
        return [dict(item) for item in rows]

    def put_attachment(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO attachments VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    payload["id"], payload["upload_id"], payload.get("run_id", ""), payload["name"],
                    payload["path"], payload["media_type"], int(payload["bytes"]), payload["sha256"],
                    payload["created_at"],
                ),
            )
        return dict(payload)

    def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        with self._lock:
            return _row(self._connection.execute("SELECT * FROM attachments WHERE id=?", (attachment_id,)).fetchone())

    def list_attachments(self, run_id: str = "") -> list[dict[str, Any]]:
        query = "SELECT * FROM attachments"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id=?"
            params.append(run_id)
        query += " ORDER BY created_at"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [dict(item) for item in rows]

    def delete_run_attachments(self, run_id: str) -> int:
        """Drop the clone rows of one run (t238 C7b N-M1 rollback of a run that was never created)."""
        if not run_id:
            return 0
        with self._lock, self._connection:
            return self._connection.execute("DELETE FROM attachments WHERE run_id=?", (run_id,)).rowcount

    def get_attachment_by_upload_id(self, upload_id: str) -> dict[str, Any] | None:
        """The earliest attachment of an upload (the library row), via idx_attachments_upload."""
        with self._lock:
            return _row(self._connection.execute(
                "SELECT * FROM attachments WHERE upload_id=? ORDER BY created_at LIMIT 1", (upload_id,)
            ).fetchone())

    def list_attachments_by_upload_id(self, upload_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM attachments WHERE upload_id=? ORDER BY created_at LIMIT ?",
                (upload_id, max(1, min(int(limit), 5000))),
            ).fetchall()
        return [dict(item) for item in rows]

    def bind_attachment(self, attachment_id: str, run_id: str, path: str) -> dict[str, Any]:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE attachments SET run_id=?,path=? WHERE id=? AND (run_id='' OR run_id=?)",
                (run_id, path, attachment_id, run_id),
            )
        result = self.get_attachment(attachment_id)
        if result is None or result.get("run_id") != run_id:
            raise ValueError("attachment is already bound to another run")
        return result

    def publish_deliverable(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT * FROM deliverables WHERE run_id=? AND sha256=? AND name=? LIMIT 1",
                (payload["run_id"], payload["sha256"], payload["name"]),
            ).fetchone()
            if existing:
                return dict(existing)
            self._connection.execute(
                "INSERT INTO deliverables VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    payload["id"], payload["run_id"], payload["session_id"], payload["name"],
                    payload["path"], payload["media_type"], int(payload["bytes"]), payload["sha256"],
                    payload.get("source_tool_call", ""), payload["preview_kind"], payload["verified_at"],
                ),
            )
        return dict(payload)

    def get_deliverable(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            return _row(self._connection.execute("SELECT * FROM deliverables WHERE id=?", (artifact_id,)).fetchone())

    def list_deliverables(self, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM deliverables WHERE run_id=? ORDER BY verified_at LIMIT ?",
                (run_id, max(1, min(int(limit), 2000))),
            ).fetchall()
        return [dict(item) for item in rows]

    def remove_deliverables_by_source(self, run_id: str, source_tool_call: str) -> list[dict[str, Any]]:
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT * FROM deliverables WHERE run_id=? AND source_tool_call=? ORDER BY verified_at",
                (run_id, source_tool_call),
            ).fetchall()
            self._connection.execute(
                "DELETE FROM deliverables WHERE run_id=? AND source_tool_call=?",
                (run_id, source_tool_call),
            )
        return [dict(item) for item in rows]

    def remove_deliverables_by_ids(self, run_id: str, artifact_ids: list[str]) -> list[dict[str, Any]]:
        identifiers = list(dict.fromkeys(str(item) for item in artifact_ids if str(item)))
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        params = [run_id, *identifiers]
        with self._lock, self._connection:
            rows = self._connection.execute(
                f"SELECT * FROM deliverables WHERE run_id=? AND id IN ({placeholders}) ORDER BY verified_at",
                params,
            ).fetchall()
            self._connection.execute(
                f"DELETE FROM deliverables WHERE run_id=? AND id IN ({placeholders})",
                params,
            )
        return [dict(item) for item in rows]

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
