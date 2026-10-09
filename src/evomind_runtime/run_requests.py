"""Durable, principal-scoped idempotency for assistant Run admission."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from .execution_progress import resource_lease
from .models import new_id, utc_now
from .tenant_access import AccessError, AccessStore, Principal
from .user_projects import PROJECT_ID, ProjectStore
from .user_tasks import TASK_ID, UserTasks
from .model_profiles import PROFILE_ID, ModelProfiles


KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,128}")
ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,160}")


def request_key(body: dict, header: str = "") -> str:
    supplied = body.get("idempotency_key", "")
    if not isinstance(supplied, str) or (supplied and header and supplied != header):
        raise AccessError(400, "invalid_idempotency_key")
    key = supplied or header
    if key and not KEY_PATTERN.fullmatch(key):
        raise AccessError(400, "invalid_idempotency_key")
    return key


def normalized_request(body: dict) -> dict:
    result = {}
    if 'model_profile_id' in body or 'model_profile_version' in body:
        profile_id, version = body.get('model_profile_id'), body.get('model_profile_version')
        if not isinstance(profile_id, str) or not PROFILE_ID.fullmatch(profile_id) or type(version) is not int or version < 1:
            raise AccessError(400, 'invalid_model_profile_selection')
        result.update(model_profile_id=profile_id, model_profile_version=version)
    user_task_id = body.get('user_task_id', '')
    if not isinstance(user_task_id, str) or (user_task_id and not TASK_ID.fullmatch(user_task_id)):
        raise AccessError(400, 'invalid_user_task_id')
    if user_task_id:
        result['user_task_id'] = user_task_id
    project_id = body.get("project_id", "")
    if not isinstance(project_id, str) or (project_id and not PROJECT_ID.fullmatch(project_id)):
        raise AccessError(400, "invalid_project_id")
    # Preserve the exact legacy fingerprint when no project was supplied.
    if project_id:
        result["project_id"] = project_id
    for name in ("prompt", "conversation_id", "selected_task"):
        value = body.get(name, "")
        if not isinstance(value, str):
            raise AccessError(400, "invalid_run_request")
        result[name] = value.strip()
    if not result["prompt"] or len(result["prompt"]) > 20000:
        raise AccessError(400, "invalid_run_prompt")
    for name in ("conversation_id", "selected_task"):
        if result[name] and not ID_PATTERN.fullmatch(result[name]):
            raise AccessError(400, "invalid_run_request")
    for name in ("attachment_ids", "secret_refs"):
        values = body.get(name, [])
        if not isinstance(values, list) or len(values) > 50 or any(not isinstance(value, str) or not value for value in values):
            raise AccessError(400, "invalid_run_request")
        result[name] = list(dict.fromkeys(values))
    return result


def fingerprint(body: dict) -> str:
    return hashlib.sha256(json.dumps(normalized_request(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class RunRequests:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "run_requests.sqlite3"
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS run_requests (
                    tenant TEXT NOT NULL, owner TEXT NOT NULL, key_hash TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL, state TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(tenant, owner, key_hash)
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def scope(principal: Principal | None, key: str) -> tuple[str, str, str]:
        return (principal.tenant_id if principal else "_service", principal.owner_id if principal else "_service", hashlib.sha256(key.encode()).hexdigest())

    def find(self, scope: tuple[str, str, str]) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM run_requests WHERE tenant=? AND owner=? AND key_hash=?", scope).fetchone()
            return dict(row) if row else None

    def state(self, scope: tuple[str, str, str], value: str) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE run_requests SET state=?,updated_at=? WHERE tenant=? AND owner=? AND key_hash=?", (value, utc_now(), *scope))

    def lookup(self, runtime: Any, access: AccessStore, principal: Principal | None, key: str, body: dict | None = None) -> dict:
        if not KEY_PATTERN.fullmatch(key):
            raise AccessError(400, "invalid_idempotency_key")
        scope = self.scope(principal, key)
        row = self.find(scope)
        if row is None:
            raise AccessError(404, "run_request_not_found")
        if body is not None and row["fingerprint"] != fingerprint(body):
            raise AccessError(409, "idempotency_payload_changed")
        run = runtime.store.get_assistant_run(row["run_id"])
        if run is None:
            raise AccessError(409, "run_creation_in_progress" if row["state"] == "pending" else "run_creation_reconciliation_required")
        # Repair only the listing indexes of an already-owned durable session.
        # Reads never start or resume a worker.
        access.created_run(runtime, run, principal)
        snapshot = runtime.assistant.snapshot(row["run_id"])
        if snapshot.get('user_task_id'):
            UserTasks(self.root).link_run(principal, snapshot['user_task_id'], row['run_id'], runtime, access)
        return {**snapshot, "creation_request": {"replayed": True, "state": row["state"], "fingerprint": row["fingerprint"]}}

    def create(self, runtime: Any, access: AccessStore, principal: Principal | None, body: dict, header: str = "") -> dict:
        key = request_key(body, header)
        values = normalized_request(body)
        digest = fingerprint(body)
        scope = self.scope(principal, key)
        if key and self.find(scope):
            return self.lookup(runtime, access, principal, key, body)
        if values.get("project_id"):
            ProjectStore(self.root).get_owned(principal, values["project_id"])
        if values.get('model_profile_id'):
            ModelProfiles(self.root).binding(principal, values['model_profile_id'], values['model_profile_version'])
        if values.get('user_task_id'):
            task = UserTasks(self.root).get(principal, values['user_task_id'])
            if values['conversation_id'] and values['conversation_id'] != task['conversation_id']:
                raise AccessError(409, 'task_conversation_mismatch')
            if values['selected_task'] or re.match(r'^\[Selected task:', values['prompt']):
                raise AccessError(409, 'explicit_tool_task_link_required')
            values['conversation_id'] = task['conversation_id']
            selected_files = UserTasks(self.root).files(principal, task['id'], runtime, access)
            if sorted(values['attachment_ids']) != sorted(item['id'] for item in selected_files):
                raise AccessError(409, 'task_files_changed')
        with ExitStack() as stack:
            if values.get('user_task_id'):
                try:
                    stack.enter_context(resource_lease(self.root / 'task-admission-leases', values['user_task_id']))
                except RuntimeError:
                    raise AccessError(409, 'task_admission_in_progress') from None
                if key and self.find(scope):
                    return self.lookup(runtime, access, principal, key, body)
                running = UserTasks(self.root).runs(principal, values['user_task_id'], runtime, access)
                if any(item['status'] in {'queued', 'planning', 'running', 'verifying', 'pausing', 'waiting_approval'} for item in running):
                    raise AccessError(409, 'task_has_active_run')
            if key:
                try:
                    stack.enter_context(resource_lease(self.root / "request-leases", json.dumps(scope)))
                except RuntimeError as error:
                    if str(error) == "resource_busy_no_duplicate_worker":
                        raise AccessError(409, "run_creation_in_progress") from None
                    raise
                if self.find(scope):
                    return self.lookup(runtime, access, principal, key, body)
            conversation = values["conversation_id"] or (new_id("conversation") if principal else "conversation_default")
            guarded = {**values, "conversation_id": conversation, "managed_hpc_identity": body.get("managed_hpc_identity")}
            access.guard(runtime, principal, "POST", ["v1", "runs"], guarded)
            run_id = new_id("run")
            if key:
                with self.connect() as connection:
                    now = utc_now()
                    connection.execute("INSERT INTO run_requests VALUES(?,?,?,?,?,?,?,?,?)", (*scope, digest, run_id, conversation, "pending", now, now))
            try:
                result = runtime.assistant.create_run(
                    **values, managed_hpc_identity=body.get("managed_hpc_identity"),
                    start=False, _reserved_run_id=run_id,
                ) if values["conversation_id"] == conversation else runtime.assistant.create_run(
                    **{**values, "conversation_id": conversation}, managed_hpc_identity=body.get("managed_hpc_identity"),
                    start=False, _reserved_run_id=run_id,
                )
                access.created_run(runtime, result, principal)
                if values.get('model_profile_id'):
                    ModelProfiles(self.root).bind_run(principal, run_id, values['model_profile_id'], values['model_profile_version'])
                if values.get("project_id"):
                    ProjectStore(self.root).bind_run(principal, values["project_id"], run_id)
                if values.get('user_task_id'):
                    UserTasks(self.root).link_run(principal, values['user_task_id'], run_id, runtime, access)
                if key:
                    self.state(scope, "ready")
            except Exception:
                if key:
                    self.state(scope, "failed")
                persisted = runtime.store.get_assistant_run(run_id)
                if persisted and persisted.get("status") == "queued":
                    runtime.store.update_assistant_run(run_id, status="blocked", error_class="creation_interrupted", error_message="Creation was interrupted before worker admission; reconcile this same Run.")
                    runtime.store.update_session(run_id, status="blocked")
                raise
            try:
                runtime.assistant.start(run_id)
            except Exception:
                persisted = runtime.store.get_assistant_run(run_id)
                if persisted and persisted.get("status") == "queued":
                    runtime.store.update_assistant_run(run_id, status="blocked", error_class="worker_start_failed", error_message="Run is durable; worker start failed. Resume this same Run after reconciliation.")
                    runtime.store.update_session(run_id, status="blocked")
            snapshot = runtime.assistant.snapshot(run_id)
            return {**snapshot, "creation_request": {"replayed": False, "state": "ready", "fingerprint": digest}}
