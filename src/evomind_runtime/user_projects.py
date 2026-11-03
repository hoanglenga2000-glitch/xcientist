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
INSTRUCTIONS_MAX = 8000
# Additive C2 columns; defaults keep every pre-existing row readable (never dropped or rewritten).
# ``deleted_instructions`` holds the instructions of a soft-deleted project so restore can bring them back.
META_COLUMNS = (("instructions", "TEXT NOT NULL DEFAULT ''"), ("deleted_at", "TEXT"),
                ("version", "INTEGER NOT NULL DEFAULT 1"), ("deleted_instructions", "TEXT"))


def valid_name(name) -> str:
    if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 120
            or re.search(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]", name)):
        raise AccessError(400, "invalid_project_request")
    return name.strip()


def valid_instructions(text) -> str:
    if not isinstance(text, str) or len(text) > INSTRUCTIONS_MAX or "\x00" in text:
        raise AccessError(400, "invalid_project_request")
    return text


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
                CREATE TABLE IF NOT EXISTS project_task_snapshots (
                    project_id TEXT NOT NULL REFERENCES projects(id), task_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, task_id)
                );
                -- Binds an existing 资料 file (an owned attachment) to a project. Unbinding deletes the row, never the file;
                -- a project delete only stamps ``detached_at`` so restore can bring every binding back.
                CREATE TABLE IF NOT EXISTS project_files (
                    project_id TEXT NOT NULL REFERENCES projects(id), file_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, created_at TEXT NOT NULL, detached_at TEXT,
                    PRIMARY KEY(project_id, file_id)
                );
            """)
            present = {row[1] for row in connection.execute("PRAGMA table_info(projects)")}
            for column, kind in META_COLUMNS:
                if column not in present:
                    connection.execute(f"ALTER TABLE projects ADD COLUMN {column} {kind}")

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
        return {key: row[key] for key in ("id", "name", "instructions", "version", "created_at", "updated_at")}

    def list_owned(self, principal: Principal | None, counts: dict | None = None) -> list[dict]:
        """Live projects. ``counts`` (project_id -> live conversations) adds ``conversation_count``."""
        scope = self.scope(principal)
        with self.connect() as connection:
            rows = [self.public(row) for row in connection.execute(
                "SELECT * FROM projects WHERE tenant_id=? AND owner_id=? AND deleted_at IS NULL ORDER BY created_at DESC,id DESC LIMIT 100", scope)]
        return rows if counts is None else [{**row, "conversation_count": counts.get(row["id"], 0)} for row in rows]

    def live_ids(self, principal: Principal | None) -> set[str]:
        with self.connect() as connection:
            return {row[0] for row in connection.execute(
                "SELECT id FROM projects WHERE tenant_id=? AND owner_id=? AND deleted_at IS NULL", self.scope(principal))}

    def get_owned(self, principal: Principal | None, project_id: str) -> dict:
        scope = self.scope(principal)
        if not isinstance(project_id, str) or not PROJECT_ID.fullmatch(project_id):
            raise AccessError(404, "project_not_found")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id=? AND tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                                     (project_id, *scope)).fetchone()
            if row is None:
                raise AccessError(404, "project_not_found")
            return self.public(row)

    def _get_any(self, principal: Principal | None, project_id: str):
        """Owned row, deleted or not; other owners get the same 404 as a missing id."""
        scope = self.scope(principal)
        if not isinstance(project_id, str) or not PROJECT_ID.fullmatch(project_id):
            raise AccessError(404, "project_not_found")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE id=? AND tenant_id=? AND owner_id=?", (project_id, *scope)).fetchone()
        if row is None:
            raise AccessError(404, "project_not_found")
        return row

    def patch(self, principal: Principal | None, project_id: str, body: dict) -> dict:
        self.get_owned(principal, project_id)
        fields = set(body) - {"version"}
        if not fields or fields - {"name", "instructions"} or type(body.get("version")) is not int:
            raise AccessError(400, "invalid_project_request")
        sets, values = [], []
        if "name" in body:
            sets.append("name=?")
            values.append(valid_name(body["name"]))
        if "instructions" in body:
            sets.append("instructions=?")
            values.append(valid_instructions(body["instructions"]))
        with self.connect() as connection:
            updated = connection.execute(
                f"UPDATE projects SET {','.join(sets)},updated_at=?,version=version+1 WHERE id=? AND tenant_id=? AND owner_id=? AND version=? AND deleted_at IS NULL",
                (*values, utc_now(), project_id, *self.scope(principal), body["version"]))
            if updated.rowcount != 1:
                raise AccessError(409, "project_version_conflict")
        return {"project": self.get_owned(principal, project_id)}

    def delete(self, principal: Principal | None, project_id: str, tasks) -> dict:
        """Soft delete. Idempotent; a repeat sweeps any member that is still attached.

        Order (two databases, no shared transaction): mark deleted + keep the instructions, then snapshot
        each member before detaching it. A crash in between leaves a deleted project that a repeat DELETE finishes.
        """
        self._get_any(principal, project_id)
        scope = self.scope(principal)
        with self.connect() as connection:
            connection.execute(
                "UPDATE projects SET deleted_at=?,deleted_instructions=instructions,instructions='',version=version+1,updated_at=? "
                "WHERE id=? AND tenant_id=? AND owner_id=? AND deleted_at IS NULL", (utc_now(), utc_now(), project_id, *scope))
            connection.execute("UPDATE project_files SET detached_at=? WHERE project_id=? AND tenant_id=? AND owner_id=? AND detached_at IS NULL",
                               (utc_now(), project_id, *scope))
        members = tasks.project_members(principal, project_id)
        with self.connect() as connection:
            connection.executemany("INSERT OR IGNORE INTO project_task_snapshots VALUES(?,?,?,?,?)",
                                   [(project_id, task_id, *scope, utc_now()) for task_id in members])
        detached = tasks.set_project(principal, members, project_id, None)
        return {"id": project_id, "deleted": True, "detached_task_count": detached}

    def restore(self, principal: Principal | None, project_id: str, tasks) -> dict:
        """Undo a delete: project, instructions, file bindings and every snapshotted membership.

        A member is re-attached only while it is still in no project; one moved to another project since the
        delete stays there (restore never steals it back). Idempotent; the snapshot is cleared once applied.
        """
        self._get_any(principal, project_id)
        scope = self.scope(principal)
        with self.connect() as connection:
            connection.execute(
                "UPDATE projects SET deleted_at=NULL,instructions=COALESCE(deleted_instructions,''),deleted_instructions=NULL,"
                "version=version+1,updated_at=? WHERE id=? AND tenant_id=? AND owner_id=? AND deleted_at IS NOT NULL",
                (utc_now(), project_id, *scope))
            connection.execute("UPDATE project_files SET detached_at=NULL WHERE project_id=? AND tenant_id=? AND owner_id=?", (project_id, *scope))
            members = [row[0] for row in connection.execute(
                "SELECT task_id FROM project_task_snapshots WHERE project_id=? AND tenant_id=? AND owner_id=? ORDER BY task_id", (project_id, *scope))]
        restored = tasks.set_project(principal, members, None, project_id)
        with self.connect() as connection:
            connection.execute("DELETE FROM project_task_snapshots WHERE project_id=? AND tenant_id=? AND owner_id=?", (project_id, *scope))
        return {"project": self.get_owned(principal, project_id), "restored_task_count": restored}

    def create(self, principal: Principal | None, body: dict) -> dict:
        scope = self.scope(principal)
        if set(body) - {"name", "idempotency_key"}:
            raise AccessError(400, "invalid_project_request")
        name, key = valid_name(body.get("name")), body.get("idempotency_key")
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise AccessError(400, "invalid_project_request")
        fingerprint = hashlib.sha256(json.dumps({"name": name}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM projects WHERE tenant_id=? AND owner_id=? AND key_hash=?", (*scope, key_hash)).fetchone()
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise AccessError(409, "project_idempotency_conflict")
                return {"project": self.public(previous), "replayed": True}
            if connection.execute("SELECT count(*) FROM projects WHERE tenant_id=? AND owner_id=? AND deleted_at IS NULL", scope).fetchone()[0] >= 1000:
                raise AccessError(409, "project_limit_reached")
            project_id, now = new_id("project"), utc_now()
            connection.execute("INSERT INTO projects(id,tenant_id,owner_id,name,key_hash,fingerprint,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                               (project_id, *scope, name, key_hash, fingerprint, now, now))
            return {"project": {"id": project_id, "name": name, "instructions": "", "version": 1, "created_at": now, "updated_at": now}, "replayed": False}

    def _bound_files(self, runtime, access, principal: Principal | None, project_id: str) -> list[dict]:
        """Owned 资料 metadata of the live bindings, oldest first; ``bound_at`` is the binding time."""
        from .user_files import metadata
        self.get_owned(principal, project_id)
        with self.connect() as connection:
            rows = connection.execute("SELECT file_id,created_at FROM project_files WHERE project_id=? AND tenant_id=? AND owner_id=? "
                                      "AND detached_at IS NULL ORDER BY created_at,file_id LIMIT 200", (project_id, *self.scope(principal))).fetchall()
        files = []
        for row in rows:
            try:
                files.append({**metadata(runtime, access, principal, row["file_id"]), "bound_at": row["created_at"]})
            except AccessError:
                continue  # no longer readable by this owner: never listed, never offered to a run
        return files

    def files(self, runtime, access, principal: Principal | None, project_id: str) -> dict:
        """UI rows: no hashes, no paths."""
        return {"files": [{"id": item["id"], "name": item["name"], "size": item["bytes"], "created_at": item["bound_at"]}
                          for item in self._bound_files(runtime, access, principal, project_id)]}

    def file_ids(self, runtime, access, principal: Principal | None, project_id: str) -> set[str]:
        return {item["id"] for item in self._bound_files(runtime, access, principal, project_id)}

    def bind_file(self, runtime, access, principal: Principal | None, project_id: str, body: dict) -> tuple[int, dict]:
        """Bind an owned 资料 file. Idempotent (200 when already bound); re-binding after an unbind is the undo."""
        from .user_files import metadata
        self.get_owned(principal, project_id)
        if not isinstance(body, dict) or set(body) != {"file_id"} or not isinstance(body["file_id"], str):
            raise AccessError(400, "invalid_project_file_request")
        item = metadata(runtime, access, principal, body["file_id"])
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT count(*) FROM project_files WHERE project_id=?", (project_id,)).fetchone()[0] >= 200:
                raise AccessError(409, "project_file_limit_reached")
            added = connection.execute("INSERT OR IGNORE INTO project_files VALUES(?,?,?,?,?,NULL)",
                                       (project_id, item["id"], *self.scope(principal), utc_now())).rowcount
        row = next(row for row in self.files(runtime, access, principal, project_id)["files"] if row["id"] == item["id"])
        return (201 if added else 200), {"file": row}

    def unbind_file(self, principal: Principal | None, project_id: str, file_id: str) -> dict:
        """Remove the binding only; the file stays in 资料. Idempotent."""
        self.get_owned(principal, project_id)
        with self.connect() as connection:
            connection.execute("DELETE FROM project_files WHERE project_id=? AND file_id=? AND tenant_id=? AND owner_id=?",
                               (project_id, file_id, *self.scope(principal)))
        return {"id": file_id, "unbound": True}

    def run_snapshot(self, runtime, principal: Principal | None, project_id: str) -> dict:
        """What a run of this project sees, frozen at run creation (instructions bound by sha256, file names only)."""
        from .tenant_access import AccessStore
        project = self.get_owned(principal, project_id)
        files = self._bound_files(runtime, AccessStore(runtime.runtime_root), principal, project_id)
        return {"project_id": project_id, "instructions": project["instructions"],
                "instructions_sha256": hashlib.sha256(project["instructions"].encode()).hexdigest(),
                "files": [{key: item[key] for key in ("id", "name", "bytes")} for item in files]}

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
                "SELECT r.run_id,r.project_id FROM project_runs r JOIN projects p ON p.id=r.project_id "
                "WHERE r.tenant_id=? AND r.owner_id=? AND p.deleted_at IS NULL", scope)}
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
