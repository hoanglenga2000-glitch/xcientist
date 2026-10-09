"""Private project registry; no worker admission or compute enrollment side effects."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .models import new_id, utc_now
from .tenant_access import AccessError, Principal


PROJECT_ID = re.compile(r"project_[a-f0-9]{32}")
KEY = re.compile(r"[A-Za-z0-9_-]{8,128}")


class ProjectStore:
    def __init__(self, runtime_root: Path):
        self.path = Path(runtime_root) / "projects.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    name TEXT NOT NULL, key_hash TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(tenant_id, owner_id, key_hash)
                );
                CREATE INDEX IF NOT EXISTS projects_owner ON projects(tenant_id, owner_id, created_at);
                CREATE TABLE IF NOT EXISTS project_runs (
                    run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS project_runs_owner ON project_runs(tenant_id, owner_id, project_id);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def scope(principal: Principal | None) -> tuple[str, str]:
        if principal is None:
            raise AccessError(403, "project_principal_required")
        return principal.tenant_id, principal.owner_id

    @staticmethod
    def public(row) -> dict:
        return {key: row[key] for key in ("id", "name", "created_at", "updated_at")}

    def list_owned(self, principal: Principal | None) -> list[dict]:
        scope = self.scope(principal)
        with self.connect() as connection:
            return [self.public(row) for row in connection.execute(
                "SELECT * FROM projects WHERE tenant_id=? AND owner_id=? ORDER BY created_at DESC,id DESC LIMIT 100", scope)]

    def get_owned(self, principal: Principal | None, project_id: str) -> dict:
        scope = self.scope(principal)
        if not isinstance(project_id, str) or not PROJECT_ID.fullmatch(project_id):
            raise AccessError(404, "project_not_found")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id=? AND tenant_id=? AND owner_id=?", (project_id, *scope)).fetchone()
            if row is None:
                raise AccessError(404, "project_not_found")
            return self.public(row)

    def create(self, principal: Principal | None, body: dict) -> dict:
        scope = self.scope(principal)
        if set(body) - {"name", "idempotency_key"}:
            raise AccessError(400, "invalid_project_request")
        name, key = body.get("name"), body.get("idempotency_key")
        if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 120
                or re.search(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", name)
                or not isinstance(key, str) or not KEY.fullmatch(key)):
            raise AccessError(400, "invalid_project_request")
        name = name.strip()
        fingerprint = hashlib.sha256(json.dumps({"name": name}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM projects WHERE tenant_id=? AND owner_id=? AND key_hash=?", (*scope, key_hash)).fetchone()
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise AccessError(409, "project_idempotency_conflict")
                return {"project": self.public(previous), "replayed": True}
            if connection.execute("SELECT count(*) FROM projects WHERE tenant_id=? AND owner_id=?", scope).fetchone()[0] >= 1000:
                raise AccessError(409, "project_limit_reached")
            project_id, now = new_id("project"), utc_now()
            connection.execute("INSERT INTO projects VALUES(?,?,?,?,?,?,?,?)", (project_id, *scope, name, key_hash, fingerprint, now, now))
            return {"project": {"id": project_id, "name": name, "created_at": now, "updated_at": now}, "replayed": False}

    def bind_run(self, principal: Principal | None, project_id: str, run_id: str) -> None:
        self.get_owned(principal, project_id)
        scope = self.scope(principal)
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO project_runs VALUES(?,?,?,?,?)", (run_id, project_id, *scope, utc_now()))
            row = connection.execute("SELECT project_id,tenant_id,owner_id FROM project_runs WHERE run_id=?", (run_id,)).fetchone()
            if tuple(row) != (project_id, *scope):
                raise AccessError(409, "run_project_is_immutable")

    def runs(self, runtime, access, principal: Principal | None, project_id: str = "") -> list[dict]:
        scope = self.scope(principal)
        if project_id:
            self.get_owned(principal, project_id)
        with self.connect() as connection:
            links = {row["run_id"]: row["project_id"] for row in connection.execute(
                "SELECT run_id,project_id FROM project_runs WHERE tenant_id=? AND owner_id=?", scope)}
            if project_id:
                ids = [row[0] for row in connection.execute(
                    "SELECT run_id FROM project_runs WHERE tenant_id=? AND owner_id=? AND project_id=? ORDER BY created_at DESC LIMIT 100", (*scope, project_id))]
                rows = [row for run_id in ids if access.owns_run(runtime, run_id, principal) and (row := runtime.store.get_assistant_run(run_id))]
            else:
                rows = access.list_runs(runtime, principal, "", 100)
        result = []
        for row in rows:
            prompt = str(row.get("prompt") or "")
            sensitive = re.search(r"password|passwd|secret|token|credential|api[_ -]?key|\u767b\u5f55|\u8d26\u53f7|\u5bc6\u7801|\u53e3\u4ee4|\u51ed\u636e", prompt, re.I)
            title = " ".join(prompt.split())[:100] if not sensitive else "Experiment " + str(row.get("created_at") or "")[:10]
            result.append({**{key: row.get(key) for key in ("id", "conversation_id", "status", "created_at", "updated_at", "model", "model_provider")}, "title": title or row["id"], "project_id": links.get(row["id"], "")})
        return sorted(result, key=lambda row: str(row.get("updated_at") or ""), reverse=True)
