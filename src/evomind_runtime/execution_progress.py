"""Executor-owned progress, freshness and cross-process resource leases.

The control-plane heartbeat is deliberately not proof that a model is fitting.
Only managed adapters may report a verified work kind. Legacy runs remain unknown.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


SCHEMA = "evomind.execution_progress.v1"
HEARTBEAT_SECONDS = 15
STALE_SECONDS = 45
STALLED_SECONDS = 180
KINDS = {"preparing", "downloading", "installing", "training", "inference", "evaluating", "executing", "waiting_approval", "idle"}
TERMINAL = {"completed", "failed", "cancelled", "blocked"}
_SECRET = re.compile(r"(?i)(password|passwd|authorization|bearer|cookie|secret|api[_ -]?key|密码|口令|凭据)\s*[:=]\s*\S+")
_QUERY = re.compile(r"(https?://[^\s?<>\"']+)\?[^\s<>\"']+")


def safe_text(value: Any, limit: int = 300) -> str:
    text = _QUERY.sub(r"\1?[redacted]", str(value or ""))
    return _SECRET.sub("[redacted]", text)[:limit]


def project_progress(record: dict[str, Any] | None, *, status: str = "running", now: float | None = None) -> dict[str, Any]:
    clock = time.time() if now is None else now
    row = dict(record or {})
    heartbeat = row.get("heartbeat_at")
    age = max(0.0, clock - float(heartbeat)) if isinstance(heartbeat, (int, float)) else None
    terminal = status in TERMINAL
    source = row.get("source", "unknown")
    trusted_kind = source == "managed_adapter" and row.get("work_kind") in KINDS
    kind = row.get("work_kind", "executing") if trusted_kind else "executing"
    if status == "waiting_approval":
        kind = "waiting_approval"
    paused = status == "paused"
    if paused:
        kind = "idle"
    freshness = "terminal" if terminal else "unknown" if age is None else "stale" if age > STALE_SECONDS else "fresh"
    done, total = row.get("completed_units"), row.get("total_units")
    counts_valid = (
        isinstance(done, (int, float)) and not isinstance(done, bool)
        and isinstance(total, (int, float)) and not isinstance(total, bool)
        and 0 <= done <= total and total > 0 and trusted_kind and not paused
    )
    last_progress = row.get("last_progress_at")
    stalled = bool(not terminal and freshness == "fresh" and isinstance(last_progress, (int, float)) and clock - last_progress > STALLED_SECONDS)
    return {
        "schema": SCHEMA,
        "work_kind": kind,
        "phase": "user_paused" if paused else row.get("phase", "awaiting_executor_evidence"),
        "source": "executor" if paused else source,
        "worker_state": "idle" if paused else row.get("worker_state", "unverified"),
        "heartbeat_at": heartbeat,
        "heartbeat_age_seconds": age,
        "last_progress_at": last_progress,
        "freshness": "paused" if paused else freshness,
        "stalled": stalled and not paused,
        "completed_units": done if counts_valid else None,
        "total_units": total if counts_valid else None,
        "unit": row.get("unit") if counts_valid else None,
        "percent": round(100 * done / total, 2) if counts_valid else None,
        "failure_class": safe_text(row.get("failure_class")),
        "detail": safe_text(row.get("detail")),
        "tool_call_id": row.get("tool_call_id", ""),
        "execution_terminal": terminal,
        "objective_outcome": row.get("objective_outcome", "not_verified"),
        "budget": row.get("budget", {}),
    }


class ProgressStore:
    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS execution_progress (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=15)

    def update(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        # The writer is called by the runtime/adapter, never exposed as a model tool.
        with self._connect() as connection:
            old = connection.execute("SELECT payload FROM execution_progress WHERE run_id=?", (run_id,)).fetchone()
            row = json.loads(old[0]) if old else {}
            if payload.get("tool_call_id") and payload.get("tool_call_id") != row.get("tool_call_id"):
                row = {"budget": row["budget"]} if "budget" in row else {}
            if "completed_units" in payload and payload["completed_units"] != row.get("completed_units"):
                payload = {**payload, "last_progress_at": time.time()}
            row.update(payload)
            row["detail"] = safe_text(row.get("detail"))
            connection.execute("INSERT INTO execution_progress VALUES (?,?) ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload", (run_id, json.dumps(row, ensure_ascii=True)))
        return row

    def read(self, run_id: str, *, status: str = "running") -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM execution_progress WHERE run_id=?", (run_id,)).fetchone()
        return project_progress(json.loads(row[0]) if row else None, status=status)


class ExecutionHeartbeat:
    def __init__(self, store: ProgressStore, run_id: str, call_id: str, emit: Callable[[dict[str, Any]], None], *, interval: float = HEARTBEAT_SECONDS) -> None:
        self.store, self.run_id, self.call_id, self.emit = store, run_id, call_id, emit
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.RLock()

    def report(self, **values: Any) -> None:
        with self.lock:
            payload = self.store.update(self.run_id, {"tool_call_id": self.call_id, "heartbeat_at": time.time(), **values})
            self.emit(project_progress(payload))

    def __enter__(self) -> ExecutionHeartbeat:
        self.report(source="executor", work_kind="executing", phase="tool_execution", worker_state="unverified", last_progress_at=time.time())
        def beat() -> None:
            while not self.stop.wait(self.interval):
                try:
                    self.report()
                except (sqlite3.Error, RuntimeError):
                    self.stop.set()
        self.thread = threading.Thread(target=beat, name="evomind-executor-heartbeat", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)


@contextmanager
def resource_lease(root: Path, resource: str) -> Iterator[None]:
    """Kernel-held advisory lock; never steal an expired lease from a live owner."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / (hashlib.sha256(resource.encode()).hexdigest() + ".lock")
    if path.is_symlink():
        raise ValueError("resource lease cannot be a symlink")
    handle = path.open("a+b")
    locked = False
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except (BlockingIOError, PermissionError) as exc:
        if not locked:
            raise RuntimeError("resource_busy_no_duplicate_worker") from exc
        raise
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def queued_resource(root: Path, resource: str, owner: str, *, wait_seconds: float = 300, notify: Callable[[], None] | None = None) -> Iterator[None]:
    """Persistent FIFO plus a kernel lock. Stale active tickets fail closed.

    Queue repair after a host crash needs independent remote-worker settlement;
    neither a browser reconnect nor a timer silently steals the previous slot.
    """
    root.mkdir(parents=True, exist_ok=True)
    database = root / "queue.sqlite3"
    with sqlite3.connect(database, timeout=15) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS resource_queue (ticket INTEGER PRIMARY KEY AUTOINCREMENT, resource TEXT NOT NULL, owner TEXT NOT NULL UNIQUE, state TEXT NOT NULL)")
        cursor = connection.execute("INSERT INTO resource_queue(resource,owner,state) VALUES (?,?,'waiting')", (resource, owner))
        ticket = cursor.lastrowid
    deadline = time.monotonic() + wait_seconds
    retain_uncertain = False
    try:
        while True:
            with sqlite3.connect(database, timeout=15) as connection:
                head = connection.execute("SELECT ticket FROM resource_queue WHERE resource=? ORDER BY ticket LIMIT 1", (resource,)).fetchone()
            if head and head[0] == ticket:
                with resource_lease(root, resource):
                    with sqlite3.connect(database, timeout=15) as connection:
                        connection.execute("UPDATE resource_queue SET state='active' WHERE ticket=?", (ticket,))
                    try:
                        yield
                    except RuntimeError as error:
                        if str(error) == "remote_settlement_unconfirmed":
                            retain_uncertain = True
                            with sqlite3.connect(database, timeout=15) as connection:
                                connection.execute("UPDATE resource_queue SET state='uncertain' WHERE ticket=?", (ticket,))
                        raise
                return
            if notify:
                notify()
            if time.monotonic() >= deadline:
                raise TimeoutError("resource_queue_budget_exhausted")
            time.sleep(0.25)
    finally:
        if not retain_uncertain:
            with sqlite3.connect(database, timeout=15) as connection:
                connection.execute("DELETE FROM resource_queue WHERE ticket=? AND owner=?", (ticket, owner))
