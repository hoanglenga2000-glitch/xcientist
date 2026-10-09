"""Object authorization for the authenticated invitation-beta HTTP boundary.

The runtime service token is still required. Web requests additionally carry a
server-derived principal. Legacy service-token callers are administrative, not
an invitation-user fallback. Ownership is never inferred from client metadata.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SCOPE_HEADER = "X-EvoMind-Access-Scope"
TENANT_HEADER = "X-EvoMind-Tenant-Id"
PRINCIPAL_HEADER = "X-EvoMind-Principal-Id"


class AccessError(Exception):
    def __init__(self, status: int = 404, code: str = "not_found") -> None:
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    owner_id: str


current_principal: ContextVar[Principal | None] = ContextVar("evomind_http_principal", default=None)


def principal_from_headers(headers: Any) -> Principal | None:
    names = (SCOPE_HEADER, TENANT_HEADER, PRINCIPAL_HEADER)
    if hasattr(headers, "get_all") and any(len(headers.get_all(name, [])) > 1 for name in names):
        raise AccessError(400, "invalid_principal_scope")
    scope, tenant, owner = (headers.get(name, "") for name in names)
    if not any((scope, tenant, owner)):
        return None
    if (
        scope != "user.v1"
        or not re.fullmatch(r"tenant_[a-f0-9]{24}|local", tenant)
        or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", owner)
        or (tenant == "local" and owner != "local")
    ):
        raise AccessError(400, "invalid_principal_scope")
    return Principal(tenant, owner)


def scoped_session_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(metadata or {})
    principal = current_principal.get()
    if principal is not None:
        value.update(tenant_id=principal.tenant_id, owner_principal_id=principal.owner_id)
        value["http_access_scope"] = {
            "schema": "evomind.http_access.v1",
            "tenant_id": principal.tenant_id,
            "owner_principal_id": principal.owner_id,
        }
    return value


def register_scoped_session(runtime_root: Path, session_id: str) -> None:
    principal = current_principal.get()
    if principal is not None:
        AccessStore(runtime_root).bind("session", session_id, principal)


class AccessStore:
    """Separate additive ACL; existing records are not silently assigned owners."""

    def __init__(self, runtime_root: Path) -> None:
        self.path = Path(runtime_root) / "principal_access.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS resource_owners (
                    kind TEXT NOT NULL, resource_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    PRIMARY KEY(kind, resource_id)
                );
                CREATE INDEX IF NOT EXISTS resource_owners_principal
                ON resource_owners(tenant_id, owner_id, kind);
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def bind(self, kind: str, resource_id: str, principal: Principal) -> None:
        if not isinstance(resource_id, str) or not resource_id or len(resource_id) > 256:
            raise AccessError(409, "resource_identity_missing")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO resource_owners VALUES (?, ?, ?, ?)",
                (kind, resource_id, principal.tenant_id, principal.owner_id),
            )
            row = connection.execute(
                "SELECT tenant_id, owner_id FROM resource_owners WHERE kind=? AND resource_id=?",
                (kind, resource_id),
            ).fetchone()
            if row != (principal.tenant_id, principal.owner_id):
                raise AccessError()

    def owns(self, kind: str, resource_id: str, principal: Principal) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM resource_owners WHERE kind=? AND resource_id=? AND tenant_id=? AND owner_id=?",
                (kind, resource_id, principal.tenant_id, principal.owner_id),
            ).fetchone() is not None

    def owner_for(self, kind: str, resource_id: str) -> Principal | None:
        with self._connect() as connection:
            row = connection.execute("SELECT tenant_id,owner_id FROM resource_owners WHERE kind=? AND resource_id=?", (kind, resource_id)).fetchone()
            return Principal(*row) if row else None

    def require(self, kind: str, resource_id: str, principal: Principal) -> None:
        if not self.owns(kind, resource_id, principal):
            raise AccessError()

    def resources(self, kind: str, principal: Principal) -> list[str]:
        with self._connect() as connection:
            return [row[0] for row in connection.execute(
                "SELECT resource_id FROM resource_owners WHERE kind=? AND tenant_id=? AND owner_id=? ORDER BY rowid DESC",
                (kind, principal.tenant_id, principal.owner_id),
            )]

    def require_run(self, runtime: Any, run_id: str, principal: Principal) -> None:
        run = runtime.store.get_assistant_run(run_id)
        if not run:
            raise AccessError()
        self.require("session", str(run.get("session_id") or ""), principal)

    def owns_run(self, runtime: Any, run_id: str, principal: Principal) -> bool:
        try:
            self.require_run(runtime, run_id, principal)
            return True
        except AccessError:
            return False

    def list_runs(self, runtime: Any, principal: Principal, conversation_id: str, limit: int) -> list[dict]:
        rows = []
        for run_id in self.resources("run", principal):
            run = runtime.store.get_assistant_run(run_id)
            if run and (not conversation_id or run.get("conversation_id") == conversation_id):
                rows.append(run)
        rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
        return rows[:max(1, min(limit, 100))]

    def list_sessions(self, runtime: Any, principal: Principal) -> list[dict]:
        return [session for session_id in self.resources("session", principal)[:100]
                if (session := runtime.store.get_session(session_id)) is not None]

    def created_upload(self, payload: dict, principal: Principal | None) -> None:
        if principal is not None:
            self.bind("upload", str(payload.get("upload_id") or payload.get("id") or ""), principal)

    def completed_upload(self, payload: dict, principal: Principal | None) -> None:
        if principal is None:
            return
        attachment = payload.get("attachment") if isinstance(payload.get("attachment"), dict) else payload
        self.bind("attachment", str(attachment.get("attachment_id") or attachment.get("id") or ""), principal)

    def created_run(self, runtime: Any, payload: dict, principal: Principal | None) -> None:
        if principal is None:
            return
        run_id = str(payload.get("id") or "")
        self.require_run(runtime, run_id, principal)
        self.bind("run", run_id, principal)
        conversation = str(payload.get("conversation_id") or "")
        if conversation:
            self.bind("conversation", conversation, principal)

    def _prepare_run(self, runtime: Any, body: dict, principal: Principal) -> None:
        attachments = body.get("attachment_ids")
        if isinstance(attachments, list):
            for attachment_id in attachments:
                self.require("attachment", str(attachment_id), principal)
        conversation = str(body.get("conversation_id") or "")
        if conversation:
            owner = self.owner_for("conversation", conversation)
            if owner is not None and owner != principal:
                raise AccessError()
            if owner is None:
                for existing in runtime.store.list_assistant_runs(conversation, 1):
                    self.require_run(runtime, str(existing["id"]), principal)
            self.bind("conversation", conversation, principal)

    def guard(self, runtime: Any, principal: Principal | None, method: str, parts: list[str], body: dict | None = None) -> None:
        if principal is None:
            return
        body = body or {}
        identity = body.get("managed_hpc_identity")
        if identity is not None and (
            not isinstance(identity, dict)
            or identity.get("tenant_id") != principal.tenant_id
            or identity.get("owner_principal_id") != principal.owner_id
        ):
            raise AccessError(403, "managed_identity_scope_mismatch")
        if parts[:1] != ["v1"]:
            raise AccessError()
        if parts[:2] == ['v1', 'model-profiles'] and (
            (method == 'GET' and len(parts) in {2, 3}) or
            (method == 'POST' and (len(parts) in {2, 3} or (len(parts) == 4 and parts[3] in {'default', 'disable', 'test'})))
        ):
            return  # ModelProfiles checks object ownership even for service callers.
        if parts[:2] == ["v1", "user-tasks"] and (
            (method == "GET" and len(parts) in {2, 3}) or (method == "POST" and (len(parts) == 2 or (len(parts) == 4 and parts[3] in {"draft", "links", "files"})))
        ):
            # UserTasks checks the server-derived principal on every object.
            return
        if method == 'GET' and parts[:2] == ['v1', 'user-files'] and len(parts) in {2, 3}:
            return  # user_files checks each resource against the authenticated owner.
        if parts == ["v1", "execution-budget"] and method in {"GET", "POST"}:
            # budget_web additionally requires the server-owned policy owner.
            return
        if parts[:2] == ["v1", "projects"] and (
            (method == "GET" and len(parts) in {2, 3}) or (method == "POST" and len(parts) == 2)
        ):
            # ProjectStore applies the tenant and owner checks to every object.
            return
        if method == "GET" and parts in (
            ["v1", "health"], ["v1", "tools"], ["v1", "super-agent", "capabilities"],
            ["v1", "sessions"], ["v1", "runs"], ["v1", "approvals"], ["v1", "goals"],
        ):
            return
        if len(parts) >= 3 and parts[1] == "runs":
            self.require_run(runtime, parts[2], principal)
            if method == "GET" or (method == "POST" and parts[3:] in (["actions"], ["reports"])):
                return
        if len(parts) >= 3 and parts[1] == "sessions" and method == "GET":
            self.require("session", parts[2], principal)
            return
        if len(parts) >= 3 and parts[1] == "artifacts" and method == "GET":
            getter = getattr(runtime.store, "get_deliverable", None)
            artifact = getter(parts[2]) if getter else None
            artifact = artifact or runtime.store.get_artifact(parts[2])
            if not artifact:
                raise AccessError()
            if artifact.get("run_id"):
                self.require_run(runtime, str(artifact["run_id"]), principal)
            else:
                self.require("session", str(artifact.get("session_id") or ""), principal)
            return
        if len(parts) == 3 and parts[1] == "goals" and method == "GET":
            goal = runtime.store.get_goal_record(parts[2])
            self.require_run(runtime, str((goal or {}).get("run_id") or ""), principal)
            return
        if method == "POST" and parts == ["v1", "runs"]:
            self._prepare_run(runtime, body, principal)
            return
        if method == "POST" and parts == ["v1", "uploads"]:
            return
        if len(parts) >= 3 and parts[1] == "uploads":
            self.require("upload", parts[2], principal)
            if (method == "POST" and parts[3:] == ["complete"]) or (
                method == "PUT" and len(parts) == 5 and parts[3] == "chunks"
            ):
                return
        if method == "POST" and len(parts) == 4 and parts[1] == "approvals" and parts[3] == "decision":
            approval = runtime.store.get_approval(parts[2])
            self.require("session", str((approval or {}).get("session_id") or ""), principal)
            return
        raise AccessError(403, "scoped_endpoint_not_available")
