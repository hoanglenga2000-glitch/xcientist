"""Long-term user memory (t237 C3.2): owner-scoped rows, soft delete + single restore, clear-all without undo.

Scope is USER (c3_contracts.PRODUCT_MEMORY_SCOPE): ``project_id`` is provenance only and never filters reads —
except rows the agent saved with ``scope='project'`` (t237 C6c), which inject only into runs of that project.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3

from .c3_contracts import clamp_memory_text, resolve_memory_enabled
from .models import new_id, utc_now
from .tenant_access import AccessError, Principal


MEMORY_ID = re.compile(r"memory_[a-f0-9]{32}")
LIST_LIMIT = 500
PUBLIC_FIELDS = ("id", "text", "source_task_id", "source_task_title", "project_id", "created_at", "updated_at",
                 "kind", "scope", "pinned", "last_used_at", "use_count")
INSERT_FIELDS = PUBLIC_FIELDS[1:7]
MEMORY_KINDS = ("profile", "preference", "project", "fact", "other")
MEMORY_SCOPES = ("user", "project")
PROMPT_TOP_K = 20
# t237 C6c additive migration (L20): columns added to a C3 database on open; NULLs read as these defaults.
MEMORY_COLUMNS = {"kind": "TEXT", "scope": "TEXT", "pinned": "INTEGER", "last_used_at": "TEXT", "use_count": "INTEGER"}
MEMORY_DEFAULTS = {"kind": "other", "scope": "user", "pinned": 0, "use_count": 0}


def valid_text(text) -> tuple[str, bool]:
    """(clamped text, truncated?). Non-string, empty or NUL-bearing text is refused."""
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise AccessError(400, "invalid_memory_request")
    return clamp_memory_text(text.strip())


def principal_from_metadata(metadata) -> Principal | None:
    """Owner bound to a run's session at creation (tenant_access.scoped_session_metadata); None for service sessions."""
    scope = (metadata or {}).get("http_access_scope") if isinstance(metadata, dict) else None
    if not isinstance(scope, dict) or not scope.get("tenant_id") or not scope.get("owner_principal_id"):
        return None
    return Principal(str(scope["tenant_id"]), str(scope["owner_principal_id"]))


def optional_str(value, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def valid_kind(kind) -> str:
    if kind not in MEMORY_KINDS:
        raise AccessError(400, "invalid_memory_kind")
    return kind


def normalized(text: str) -> str:
    return " ".join(str(text).split()).casefold().rstrip("。.!！;；")


def near_duplicate(a: str, b: str) -> bool:
    """Same fact: equal after whitespace/case folding, or one contains the other at ≥ 0.9 length ratio."""
    a, b = normalized(a), normalized(b)
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return bool(short) and short in long and len(short) / len(long) >= 0.9


def tokens(text: str) -> set[str]:
    """Latin words (case-folded) plus CJK character bigrams: enough overlap signal without an embedder."""
    text = str(text).casefold()
    words = set(re.findall(r"[a-z0-9]{2,}", text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(run[index:index + 2] for index in range(max(1, len(run) - 1)))
    return words


def strict_bools(body, allowed: set[str], error: str) -> dict:
    if not isinstance(body, dict) or not body or set(body) - allowed or any(type(value) is not bool for value in body.values()):
        raise AccessError(400, error)
    return body


class UserMemoryStore:
    def __init__(self, runtime_root: Path):
        self.path = Path(runtime_root) / "user_memory.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            # ``seq`` orders rows created within the same timestamp; ``cleared_at`` marks rows swept by
            # clear-all, which (unlike a single delete) can never be restored.
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, text TEXT NOT NULL,
                    source_task_id TEXT, source_task_title TEXT, project_id TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT, cleared_at TEXT,
                    source_run_id TEXT
                );
                CREATE INDEX IF NOT EXISTS user_memories_run ON user_memories(source_run_id);
                CREATE INDEX IF NOT EXISTS user_memories_owner ON user_memories(tenant_id, owner_id, deleted_at);
                -- NULL / missing row = never set: reads as the product default (resolve_memory_enabled).
                CREATE TABLE IF NOT EXISTS user_settings (
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, memory_enabled INTEGER, updated_at TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, owner_id)
                );
                CREATE TABLE IF NOT EXISTS memory_project_settings (
                    tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    use_account_memory INTEGER, project_only INTEGER, updated_at TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, owner_id, project_id)
                );
            """)
            self._ensure_columns(connection)

    @staticmethod
    def _ensure_columns(connection) -> None:
        """Additive + idempotent (t237 C6c, L20): add what a C3 database lacks, backfill NULLs, never drop or rewrite rows."""
        existing = {row[1] for row in connection.execute("PRAGMA table_info(user_memories)")}
        for name, kind in MEMORY_COLUMNS.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE user_memories ADD COLUMN {name} {kind}")
        for name, value in MEMORY_DEFAULTS.items():
            connection.execute(f"UPDATE user_memories SET {name}=? WHERE {name} IS NULL", (value,))
        if "paused" not in {row[1] for row in connection.execute("PRAGMA table_info(user_settings)")}:
            connection.execute("ALTER TABLE user_settings ADD COLUMN paused INTEGER DEFAULT 0")

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
    def scope(principal: Principal | None) -> tuple[str, str]:
        if principal is None:
            raise AccessError(403, "memory_principal_required")
        return principal.tenant_id, principal.owner_id

    @staticmethod
    def public(row) -> dict:
        result = {key: row[key] for key in PUBLIC_FIELDS}
        result["pinned"] = bool(result["pinned"])
        result["use_count"] = int(result["use_count"] or 0)
        return result

    def _row(self, principal: Principal | None, memory_id: str, *, live: bool):
        """Owned row (live only, or any); other owners get the same 404 as a missing id."""
        scope = self.scope(principal)
        if not isinstance(memory_id, str) or not MEMORY_ID.fullmatch(memory_id):
            raise AccessError(404, "memory_not_found")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM user_memories WHERE id=? AND tenant_id=? AND owner_id=?"
                                     + (" AND deleted_at IS NULL" if live else ""), (memory_id, *scope)).fetchone()
        if row is None:
            raise AccessError(404, "memory_not_found")
        return row

    def add(self, principal: Principal | None, text, *, source_task_id: str | None = None,
            source_task_title: str | None = None, project_id: str | None = None, source_run_id: str | None = None,
            kind: str = "other", scope: str = "user") -> dict:
        """Insert one row, or refresh the owner's live near-duplicate instead (t237 C6c). Refused while paused (L18)."""
        owner = self.scope(principal)
        clean, _ = valid_text(text)
        kind = valid_kind(kind)
        project_id = optional_str(project_id, 128)
        if scope not in MEMORY_SCOPES or (scope == "project" and not project_id):
            raise AccessError(400, "invalid_memory_scope")
        if self.paused(principal):
            raise AccessError(409, "memory_paused")
        now = utc_now()
        with self.connect() as connection:
            for row in connection.execute("SELECT * FROM user_memories WHERE tenant_id=? AND owner_id=? AND deleted_at IS NULL "
                                          "ORDER BY seq DESC LIMIT ?", (*owner, LIST_LIMIT)).fetchall():
                if near_duplicate(row["text"], clean):
                    connection.execute("UPDATE user_memories SET updated_at=? WHERE id=?", (now, row["id"]))
                    return self.public({**dict(row), "updated_at": now})
            memory_id = new_id("memory")
            row = {"id": memory_id, "text": clean, "source_task_id": optional_str(source_task_id, 128),
                   "source_task_title": optional_str(source_task_title, 200), "project_id": project_id,
                   "created_at": now, "updated_at": now, "kind": kind, "scope": scope, "pinned": False, "last_used_at": None, "use_count": 0}
            connection.execute("INSERT INTO user_memories(id,tenant_id,owner_id,text,source_task_id,source_task_title,project_id,"
                               "created_at,updated_at,source_run_id,kind,scope,pinned,use_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0,0)",
                               (memory_id, *owner, *(row[key] for key in INSERT_FIELDS), optional_str(source_run_id, 128), kind, scope))
        return row

    def list_owned(self, principal: Principal | None, title_of: Callable[[str], str | None] | None = None, *,
                   kind: str | None = None, scope: str | None = None, q: str | None = None, pinned: bool | None = None) -> list[dict]:
        """Live rows, newest first, across every project (USER scope).

        t237 C6c filters: ``kind`` (one kind or a comma list), ``scope``, ``pinned`` and ``q`` (case-folded substring of the text).

        ``title_of`` refreshes ``source_task_title`` from the owner's live conversation at read time (03 §2):
        renamed conversations show the current title; a deleted or unknown one yields ``None`` (来自已删除的对话),
        overriding any title stored at write time. Without ``title_of``, the write-time title is returned as-is (legacy / unit tests).
        """
        owner = self.scope(principal)
        kinds = [valid_kind(item) for item in kind.split(",")] if kind else []
        if scope and scope not in MEMORY_SCOPES:
            raise AccessError(400, "invalid_memory_scope")
        needle = normalized(q) if isinstance(q, str) and q.strip() else ""
        with self.connect() as connection:
            rows = [self.public(row) for row in connection.execute(
                "SELECT * FROM user_memories WHERE tenant_id=? AND owner_id=? AND deleted_at IS NULL "
                "ORDER BY created_at DESC,seq DESC LIMIT ?", (*owner, LIST_LIMIT))]
        rows = [row for row in rows if (not kinds or row["kind"] in kinds) and (not scope or row["scope"] == scope)
                and (pinned is None or row["pinned"] is pinned) and (not needle or needle in normalized(row["text"]))]
        if title_of is not None:
            titles: dict[str, str | None] = {}
            for row in rows:
                task_id = row.get("source_task_id")
                if task_id and task_id not in titles:
                    titles[task_id] = title_of(task_id)
                row["source_task_title"] = titles.get(task_id) if task_id else None
        return rows

    def saved_by_run(self, run_id: str) -> list[dict]:
        """Live rows a run recorded, oldest first ({id, text}: MemorySavedEvent). The caller has already authorized the run."""
        with self.connect() as connection:
            return [{"id": row["id"], "text": row["text"]} for row in connection.execute(
                "SELECT id,text FROM user_memories WHERE source_run_id=? AND deleted_at IS NULL ORDER BY seq LIMIT 50", (str(run_id),))]

    def get(self, principal: Principal | None, memory_id: str) -> dict:
        return self.public(self._row(principal, memory_id, live=True))

    def update(self, principal: Principal | None, memory_id: str, text) -> dict:
        """Replace the text of a live row; over-long text is clamped to MEMORY_TEXT_MAX_CHARS."""
        self._row(principal, memory_id, live=True)
        clean, truncated = valid_text(text)
        with self.connect() as connection:
            updated = connection.execute(
                "UPDATE user_memories SET text=?,updated_at=? WHERE id=? AND tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                (clean, utc_now(), memory_id, *self.scope(principal)))
            if updated.rowcount != 1:
                raise AccessError(404, "memory_not_found")
        return {"memory": self.get(principal, memory_id), "truncated": truncated}

    def patch(self, principal: Principal | None, memory_id: str, body) -> dict:
        """t237 C6c: any non-empty subset of {text, kind, pinned}; the next prompt reads it (L19)."""
        if not isinstance(body, dict) or not body or set(body) - {"text", "kind", "pinned"}:
            raise AccessError(400, "invalid_memory_request")
        if "pinned" in body and type(body["pinned"]) is not bool:
            raise AccessError(400, "invalid_memory_request")
        kind = valid_kind(body["kind"]) if "kind" in body else None
        clean, truncated = valid_text(body["text"]) if "text" in body else (None, False)
        self._row(principal, memory_id, live=True)
        updates = {**({"text": clean} if clean is not None else {}), **({"kind": kind} if kind else {}),
                   **({"pinned": int(body["pinned"])} if "pinned" in body else {}), "updated_at": utc_now()}
        with self.connect() as connection:
            updated = connection.execute(
                f"UPDATE user_memories SET {','.join(f'{key}=?' for key in updates)} WHERE id=? AND tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                (*updates.values(), memory_id, *self.scope(principal)))
            if updated.rowcount != 1:
                raise AccessError(404, "memory_not_found")
        return {"memory": self.get(principal, memory_id), "truncated": truncated}

    def select_for_prompt(self, principal: Principal | None, query: str, *, project_id: str = "", limit: int = PROMPT_TOP_K) -> list[dict]:
        """Rows to inject: pinned first, then by token overlap with ``query``, then newest; at most ``limit``.

        ``scope='project'`` rows join only runs of their own project (L17); the project's memory settings
        (``project_only`` / ``use_account_memory=0``) drop account-wide rows from that project's runs.
        """
        rows = self.list_owned(principal)
        settings = self.project_settings(principal, project_id) if project_id else None
        account = settings is None or (settings["use_account_memory"] and not settings["project_only"])
        rows = [row for row in rows if (row["scope"] == "project" and project_id and row["project_id"] == project_id)
                or (row["scope"] != "project" and account)]
        wanted = tokens(query)
        ranked = sorted(enumerate(rows), key=lambda item: (not item[1]["pinned"], -len(wanted & tokens(item[1]["text"])), item[0]))
        return [row for _, row in ranked[:limit]]

    def mark_used(self, memory_ids: list[str]) -> None:
        """Bump ``last_used_at`` / ``use_count`` for rows a prompt injected. Callers pass ids they already selected for the owner."""
        if not memory_ids:
            return
        with self.connect() as connection:
            connection.executemany("UPDATE user_memories SET last_used_at=?,use_count=COALESCE(use_count,0)+1 WHERE id=?",
                                   [(utc_now(), memory_id) for memory_id in memory_ids])

    def delete(self, principal: Principal | None, memory_id: str) -> dict:
        """Soft delete. Idempotent."""
        self._row(principal, memory_id, live=False)
        with self.connect() as connection:
            connection.execute("UPDATE user_memories SET deleted_at=? WHERE id=? AND tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                               (utc_now(), memory_id, *self.scope(principal)))
        return {"id": memory_id, "deleted": True}

    def restore(self, principal: Principal | None, memory_id: str) -> dict:
        """Undo a single delete. Rows swept by clear-all are gone for good (404). Idempotent on a live row."""
        if self._row(principal, memory_id, live=False)["cleared_at"] is not None:
            raise AccessError(404, "memory_not_found")
        with self.connect() as connection:
            connection.execute("UPDATE user_memories SET deleted_at=NULL WHERE id=? AND tenant_id=? AND owner_id=? AND cleared_at IS NULL",
                               (memory_id, *self.scope(principal)))
        return {"memory": self.get(principal, memory_id)}

    def clear_all(self, principal: Principal | None) -> int:
        """Soft-delete every live row and seal every deleted one against restore (no undo). Returns rows cleared."""
        scope, now = self.scope(principal), utc_now()
        with self.connect() as connection:
            cleared = connection.execute("UPDATE user_memories SET deleted_at=? WHERE tenant_id=? AND owner_id=? AND deleted_at IS NULL",
                                         (now, *scope)).rowcount
            connection.execute("UPDATE user_memories SET cleared_at=? WHERE tenant_id=? AND owner_id=? AND cleared_at IS NULL", (now, *scope))
        return cleared

    def settings(self, principal: Principal | None) -> dict:
        """``memory_enabled`` (read + write) and ``paused`` (writes only; L18 pause ≠ reset) are independent."""
        scope = self.scope(principal)
        with self.connect() as connection:
            row = connection.execute("SELECT memory_enabled,paused FROM user_settings WHERE tenant_id=? AND owner_id=?", scope).fetchone()
        return {"memory_enabled": resolve_memory_enabled(None if row is None else row[0]), "paused": bool(row is not None and row[1])}

    def memory_enabled(self, principal: Principal | None) -> bool:
        return self.settings(principal)["memory_enabled"]

    def paused(self, principal: Principal | None) -> bool:
        return self.settings(principal)["paused"]

    def patch_settings(self, principal: Principal | None, body) -> dict:
        scope = self.scope(principal)
        body = strict_bools(body, {"memory_enabled", "paused"}, "invalid_settings_request")
        current = self.settings(principal)
        merged = {**current, **body}
        with self.connect() as connection:
            connection.execute("INSERT INTO user_settings(tenant_id,owner_id,memory_enabled,updated_at,paused) VALUES(?,?,?,?,?) "
                               "ON CONFLICT(tenant_id,owner_id) DO UPDATE SET memory_enabled=excluded.memory_enabled,"
                               "paused=excluded.paused,updated_at=excluded.updated_at",
                               (*scope, int(merged["memory_enabled"]), utc_now(), int(merged["paused"])))
        return self.settings(principal)

    def project_settings(self, principal: Principal | None, project_id: str) -> dict:
        """Per-project memory switches; the caller has already checked the project is the principal's own."""
        scope = self.scope(principal)
        with self.connect() as connection:
            row = connection.execute("SELECT use_account_memory,project_only FROM memory_project_settings "
                                     "WHERE tenant_id=? AND owner_id=? AND project_id=?", (*scope, str(project_id))).fetchone()
        return {"use_account_memory": row is None or row[0] is None or bool(row[0]), "project_only": bool(row is not None and row[1])}

    def patch_project_settings(self, principal: Principal | None, project_id: str, body) -> dict:
        scope = self.scope(principal)
        body = strict_bools(body, {"use_account_memory", "project_only"}, "invalid_memory_settings_request")
        merged = {**self.project_settings(principal, project_id), **body}
        with self.connect() as connection:
            connection.execute("INSERT INTO memory_project_settings VALUES(?,?,?,?,?,?) ON CONFLICT(tenant_id,owner_id,project_id) DO UPDATE SET "
                               "use_account_memory=excluded.use_account_memory,project_only=excluded.project_only,updated_at=excluded.updated_at",
                               (*scope, str(project_id), int(merged["use_account_memory"]), int(merged["project_only"]), utc_now()))
        return self.project_settings(principal, project_id)
