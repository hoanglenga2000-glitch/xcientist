"""Fair time-sharing of the pool's GPU cards between tenants.

Supersedes the host-local hot-patch of the same module name. State lives in one
SQLite file (WAL, busy_timeout) next to the runtime so every runtime thread
sees the same queue:

* ``pool_nodes``        - GPU cards. Only the cards named in gpu-pool.json (the
                          primary ``node_id`` plus the optional ``nodes`` list) are
                          granted; every other row (retired allocations) is ignored.
* ``quotas``            - optional per-tenant override of the rolling budget.
* ``train_jobs``        - queued / running / finished training requests.
* ``gpu_leases``        - the active grant, kept alive by heartbeats.
* ``tenant_gpu_usage``  - booked GPU seconds, one row per job (never twice).
* ``gpu_audit``         - operator audit trail (grants, kills, reconcile, path
                          confinement violations).

Policy (see docs/gpu-timeshare-c7c.md):

* Concurrency 1 per tenant, enforced by SQLite triggers, plus one UNIQUE lease
  per card, so N cards run up to N tenants at once. No preemption of a running lease.
* Card choice: the best idle card that fits the job now (lowest ``priority``,
  then the most free memory); see ``schedule``.
* Queue order = ``fair_order``: per tenant only the oldest queued job (FIFO in
  tenant) competes; the tenant with the lowest ``recent usage - head wait``
  wins, ties by submission order. Waiting lowers the score 1:1, so a tenant
  that already ran long cannot block a fresh tenant, and it is never starved:
  after waiting as long as it has used, it beats a fresh tenant.
* Every lease has a deadline = grant + effective max run, where effective max
  run = min(request or default, platform ceiling, remaining budget).
* Budget = daily (host-local calendar day) and rolling window, in GPU seconds.
* ETA (estimate) = remaining of the current lease (bounded by its deadline)
  + the requested run lengths of the live queued jobs ahead. Never negative.
* A queued job is live while its waiter (the tool call holding it) refreshed
  ``waiter_at`` within ``waiter_live_seconds``. A job without a live waiter keeps
  its place until ``queue_hold_seconds`` but never blocks anyone: position, ETA
  and dispatch all count the first live job.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

def _service_node_job_id() -> int:
    """2026-11-05: the one schedulable card follows the operator-owned pool file (job ids change when an
    allocation expires). Read once at import; a restart applies a new card. Default keeps job 98270."""

    import os
    path = Path(os.environ.get("EVOMIND_GPU_POOL_CONFIG", r"C:\ProgramData\EvoMind\state\gpu-pool.json"))
    try:
        block = json.loads(path.read_text(encoding="utf-8-sig")).get("gpu_pool") or {}
        job = block.get("hpc_job_id")
        if isinstance(job, int) and not isinstance(job, bool) and 10_000 <= job <= 9_999_999 and block.get("node_id") == f"job{job}":
            return job
    except (OSError, ValueError, AttributeError):
        pass
    return 98270


ALLOWED_HPC_JOB_ID = _service_node_job_id()
ALLOWED_NODE_ID = f"job{ALLOWED_HPC_JOB_ID}"


def _service_node_display() -> tuple[str, int]:
    """Display label / VRAM of the card (admin pool view only; admission uses the live free-memory probe)."""

    import os
    try:
        path = Path(os.environ.get("EVOMIND_GPU_POOL_CONFIG", r"C:\ProgramData\EvoMind\state\gpu-pool.json"))
        block = json.loads(path.read_text(encoding="utf-8-sig")).get("gpu_pool") or {}
        label = str(block.get("gpu_model") or "A800")[:32]
        vram = block.get("gpu_vram_mb")
        return label, int(vram) if isinstance(vram, int) and not isinstance(vram, bool) and vram > 0 else 81920
    except (OSError, ValueError, AttributeError):
        return "A800", 81920


SERVICE_NODE_LABEL, SERVICE_NODE_VRAM_MB = _service_node_display()


def _extra_pool_nodes() -> tuple[dict, ...]:
    """2026-11-05: more cards in the same pool. ``gpu_pool.nodes`` = [{node_id, hpc_job_id, gpu_model, gpu_vram_mb, priority}].

    The primary card (``node_id``/``hpc_job_id``) is always first. An entry that is malformed, duplicates a card or comes
    with an invalid primary is ignored, so it is never schedulable. Read once at import, like the primary.
    """

    import os
    path = Path(os.environ.get("EVOMIND_GPU_POOL_CONFIG", r"C:\ProgramData\EvoMind\state\gpu-pool.json"))
    try:
        block = json.loads(path.read_text(encoding="utf-8-sig")).get("gpu_pool") or {}
        if block.get("hpc_job_id") != ALLOWED_HPC_JOB_ID or block.get("node_id") != ALLOWED_NODE_ID:
            return ()
        items = block.get("nodes") or []
    except (OSError, ValueError, AttributeError):
        return ()
    nodes: list[dict] = []
    seen = {ALLOWED_NODE_ID}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        job, vram, priority = item.get("hpc_job_id"), item.get("gpu_vram_mb"), item.get("priority", 1)
        valid_int = lambda value: isinstance(value, int) and not isinstance(value, bool)
        if (not valid_int(job) or not 10_000 <= job <= 9_999_999 or item.get("node_id") != f"job{job}" or f"job{job}" in seen
                or not valid_int(vram) or vram <= 0 or not valid_int(priority) or priority < 0):
            continue
        seen.add(f"job{job}")
        nodes.append({"node_id": f"job{job}", "hpc_job_id": job, "label": str(item.get("gpu_model") or "GPU")[:32],
                      "vram_mb": vram, "priority": priority})
    return tuple(nodes)


EXTRA_POOL_NODES = _extra_pool_nodes()


def _primary_priority() -> int:
    """2026-11-05: optional ``gpu_pool.priority`` of the primary card (default 0), so another card can be preferred."""

    import os
    path = Path(os.environ.get("EVOMIND_GPU_POOL_CONFIG", r"C:\ProgramData\EvoMind\state\gpu-pool.json"))
    try:
        block = json.loads(path.read_text(encoding="utf-8-sig")).get("gpu_pool") or {}
        value = block.get("priority", 0) if block.get("node_id") == ALLOWED_NODE_ID else 0
    except (OSError, ValueError, AttributeError):
        return 0
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


POOL_NODE_IDS = (ALLOWED_NODE_ID, *(node["node_id"] for node in EXTRA_POOL_NODES))
POOL_HPC_JOB_IDS = frozenset({ALLOWED_HPC_JOB_ID, *(node["hpc_job_id"] for node in EXTRA_POOL_NODES)})
# Lower = preferred when several idle cards fit the job; ties go to the card with more free memory.
NODE_PRIORITY = {ALLOWED_NODE_ID: _primary_priority(), **{node["node_id"]: node["priority"] for node in EXTRA_POOL_NODES}}

SCHEMA = """
CREATE TABLE IF NOT EXISTS pool_nodes (
  id TEXT PRIMARY KEY, label TEXT NOT NULL, gpu_uuid TEXT NOT NULL, gpu_model TEXT NOT NULL,
  vram_mb INTEGER NOT NULL, profile TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quotas (
  tenant_id TEXT PRIMARY KEY, gpu_hours REAL NOT NULL, period_days INTEGER NOT NULL DEFAULT 30,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS train_jobs (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
  tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, run_id TEXT NOT NULL,
  vram_mb INTEGER NOT NULL, max_hours REAL NOT NULL, status TEXT NOT NULL,
  node_id TEXT, required_node TEXT, detail TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL, started_at REAL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS train_jobs_status ON train_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS train_jobs_tenant ON train_jobs(tenant_id, finished_at);
CREATE TABLE IF NOT EXISTS gpu_leases (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE, node_id TEXT NOT NULL UNIQUE,
  granted_at REAL NOT NULL, heartbeat_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tenant_gpu_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE,
  lease_id TEXT, run_id TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL NOT NULL,
  seconds REAL NOT NULL, reason TEXT NOT NULL, booked_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tenant_gpu_usage_tenant ON tenant_gpu_usage(tenant_id, finished_at);
CREATE TABLE IF NOT EXISTS gpu_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, kind TEXT NOT NULL,
  tenant_id TEXT, job_id TEXT, detail TEXT NOT NULL DEFAULT '{}'
);
"""

# Additive columns on the hot-patch tables (an old DB opens unchanged; NULL = legacy row).
_ADDED_COLUMNS = {
    "train_jobs": (
        ("requested_seconds", "INTEGER"),
        ("max_run_seconds", "INTEGER"),
        ("deadline_at", "REAL"),
        ("cancel_requested_at", "REAL"),
        ("waiter_at", "REAL"),
        ("reason", "TEXT"),
        ("budget_exempt", "INTEGER"),  # operator run: its own timeout, no tenant budget
        ("orphan_kill_at", "REAL"),  # next attempt to kill a remote process left behind by a lost job
        ("dispatched_at", "REAL"),  # C7c-FIX4: when the work really started on the card (NULL = legacy: billed from started_at)
    ),
    "gpu_leases": (
        ("tenant_id", "TEXT"),
        ("deadline_at", "REAL"),
        ("kill_requested_at", "REAL"),
    ),
}

# Concurrency = 1 per tenant, enforced by the database for every writer (not just this module).
# Triggers never re-validate existing rows, so an old DB keeps opening.
_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS train_jobs_one_running_per_tenant_ins
BEFORE INSERT ON train_jobs WHEN NEW.status = 'running'
  AND EXISTS (SELECT 1 FROM train_jobs WHERE tenant_id = NEW.tenant_id AND status = 'running')
BEGIN SELECT RAISE(ABORT, 'tenant_gpu_concurrency'); END;
CREATE TRIGGER IF NOT EXISTS train_jobs_one_running_per_tenant_upd
BEFORE UPDATE OF status ON train_jobs WHEN NEW.status = 'running' AND OLD.status <> 'running'
  AND EXISTS (SELECT 1 FROM train_jobs WHERE tenant_id = NEW.tenant_id AND status = 'running' AND id <> NEW.id)
BEGIN SELECT RAISE(ABORT, 'tenant_gpu_concurrency'); END;
CREATE TRIGGER IF NOT EXISTS gpu_leases_one_per_tenant
BEFORE INSERT ON gpu_leases WHEN NEW.tenant_id IS NOT NULL
  AND EXISTS (SELECT 1 FROM gpu_leases WHERE tenant_id = NEW.tenant_id)
BEGIN SELECT RAISE(ABORT, 'tenant_gpu_concurrency'); END;
"""

QUEUED, RUNNING = "queued", "running"
SUCCEEDED, FAILED, CANCELLED, LOST, TIMEOUT, REJECTED = "succeeded", "failed", "cancelled", "lost", "timeout", "rejected"
TERMINAL = {SUCCEEDED, FAILED, CANCELLED, LOST, TIMEOUT, REJECTED}
PREPARING = "preparing"  # running-job reason while dependencies are fetched: nothing is dispatched, nothing is billed
ENV_NOT_READY = "environment_not_ready"  # the reason kept when a job failed while it was still preparing
FREEING = "freeing_gpu_memory"  # queued-job reason while the memory broker clears space (ADDENDUM 2)
ACTIVE = {QUEUED, RUNNING}
# Legacy hot-patch default (used only to fill the old NOT NULL column).
DEFAULT_QUOTA_HOURS = 24.0

FreeVramProbe = Callable[[dict], "int | None"]


class SchedulerError(ValueError):
    """``str(error)`` is always a stable code (see gpu_messages)."""


@dataclass(frozen=True)
class GpuPolicy:
    max_run_seconds: int = 1800
    max_run_ceiling_seconds: int = 3600
    min_grant_seconds: int = 60
    daily_budget_seconds: int = 120 * 60
    rolling_budget_seconds: int = 600 * 60
    rolling_window_seconds: int = 7 * 86400
    # Host-local calendar day for the daily budget (EvoMind host = UTC+08:00).
    day_utc_offset_seconds: int = 8 * 3600
    lease_ttl_seconds: int = 300
    kill_grace_seconds: int = 30
    kill_timeout_seconds: int = 120
    queue_hold_seconds: int = 1800
    max_queued_per_tenant: int = 3
    max_queue_length: int = 50
    default_vram_mb: int = 4096
    node_vram_mb: int = SERVICE_NODE_VRAM_MB
    broker_stop_seconds: int = 60  # ADDENDUM 2: estimate of the memory broker's stop time (vLLM ≈ 30–60 s)
    waiter_live_seconds: int = 90  # waiters refresh every poll (5 s); a probe can take up to 50 s
    orphan_kill_retry_seconds: int = 60

    @classmethod
    def from_mapping(cls, value: dict | None) -> "GpuPolicy":
        value = value or {}
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        for name, item in known.items():
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise SchedulerError("gpu_policy_invalid")
        policy = cls(**known)
        if (policy.max_run_seconds < policy.min_grant_seconds or policy.max_run_ceiling_seconds < policy.max_run_seconds
                or policy.lease_ttl_seconds < 30 or policy.max_queued_per_tenant < 1 or policy.default_vram_mb < 1):
            raise SchedulerError("gpu_policy_invalid")
        return policy


@dataclass(frozen=True)
class Grant:
    job_id: str
    lease_id: str
    node_id: str
    profile: str
    deadline_at: float = 0.0
    max_run_seconds: int = 0


@dataclass
class QueueEntry:
    """Input row for ``fair_order`` (pure, unit-testable)."""

    job_id: str
    tenant_id: str
    seq: int
    created_at: float
    requested_seconds: int


def fair_order(entries: Iterable[QueueEntry], usage_seconds: dict[str, float], now: float) -> list[QueueEntry]:
    """Order the queue: FIFO within a tenant, fair across tenants, no starvation.

    Repeatedly pick, among each tenant's oldest queued job, the one with the lowest
    ``score = tenant usage - head wait`` (ties: lower seq). After a pick that
    tenant's usage grows by the picked job's requested seconds, so a tenant with
    several queued jobs interleaves with others instead of taking consecutive slots.
    """

    per_tenant: dict[str, list[QueueEntry]] = {}
    for entry in sorted(entries, key=lambda item: item.seq):
        per_tenant.setdefault(entry.tenant_id, []).append(entry)
    used = {tenant: float(usage_seconds.get(tenant, 0.0)) for tenant in per_tenant}
    ordered: list[QueueEntry] = []
    while per_tenant:
        def score(tenant: str) -> tuple[float, int]:
            head = per_tenant[tenant][0]
            return (used[tenant] - max(0.0, now - head.created_at), head.seq)

        tenant = min(per_tenant, key=score)
        head = per_tenant[tenant].pop(0)
        ordered.append(head)
        used[tenant] += head.requested_seconds
        if not per_tenant[tenant]:
            del per_tenant[tenant]
    return ordered


def eta_seconds(now: float, running_deadline: float | None, ahead_requested: Iterable[float]) -> int:
    """Estimated wait: rest of the current lease + requested lengths of the jobs ahead."""

    total = 0.0
    if running_deadline is not None:
        total += max(0.0, float(running_deadline) - now)
    for value in ahead_requested:
        total += max(0.0, float(value))
    return max(0, int(round(total))) if total == total else 0  # NaN guard


class GpuScheduler:
    def __init__(self, path: Path, *, policy: GpuPolicy | None = None, lease_ttl_seconds: float | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or GpuPolicy()
        self.lease_ttl = float(lease_ttl_seconds if lease_ttl_seconds is not None else self.policy.lease_ttl_seconds)
        self.waiter_ttl = min(self.lease_ttl, float(self.policy.waiter_live_seconds))
        self.clock = clock
        # (job row) -> confirmed gone. Runs outside any transaction, before the next grant.
        self.orphan_killer: Callable[[dict], bool] | None = None
        self._local = threading.local()
        # executescript manages its own transaction; it must not run inside _tx.
        db = self._db()
        db.executescript(SCHEMA)
        for table, columns in _ADDED_COLUMNS.items():
            present = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns:
                if name not in present:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
        db.executescript(_TRIGGERS)

    # -- connection handling: one connection per thread, WAL, bounded waits --
    def _db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
        except BaseException:
            db.execute("ROLLBACK")
            raise
        db.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def audit(self, kind: str, *, tenant_id: str = "", job_id: str = "", detail: dict | None = None,
              db: sqlite3.Connection | None = None) -> None:
        values = (self.clock(), kind, tenant_id or None, job_id or None, json.dumps(detail or {}, sort_keys=True))
        sql = "INSERT INTO gpu_audit(at,kind,tenant_id,job_id,detail) VALUES(?,?,?,?,?)"
        if db is not None:
            db.execute(sql, values)
        else:
            with self._tx() as tx:
                tx.execute(sql, values)

    # -- pool and quotas --
    def upsert_node(self, node_id: str, *, label: str, gpu_uuid: str, gpu_model: str, vram_mb: int, profile: str, enabled: bool = True) -> None:
        now = self.clock()
        with self._tx() as db:
            db.execute(
                "INSERT INTO pool_nodes(id,label,gpu_uuid,gpu_model,vram_mb,profile,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET label=excluded.label,gpu_uuid=excluded.gpu_uuid,gpu_model=excluded.gpu_model,"
                "vram_mb=excluded.vram_mb,profile=excluded.profile,enabled=excluded.enabled,updated_at=excluded.updated_at",
                (node_id, label, gpu_uuid, gpu_model, int(vram_mb), profile, int(enabled), now, now),
            )

    def ensure_node(self, *, profile: str, vram_mb: int | None = None) -> None:
        """Register the one schedulable node; other rows are left untouched (and ignored)."""

        db = self._db()
        row = db.execute("SELECT * FROM pool_nodes WHERE id=?", (ALLOWED_NODE_ID,)).fetchone()
        if row is not None and row["profile"] == profile and row["gpu_model"] == SERVICE_NODE_LABEL:
            return
        self.upsert_node(ALLOWED_NODE_ID, label=SERVICE_NODE_LABEL, gpu_uuid=row["gpu_uuid"] if row else "",
                         gpu_model=SERVICE_NODE_LABEL, vram_mb=int(vram_mb or self.policy.node_vram_mb), profile=profile,
                         enabled=bool(row["enabled"]) if row else True)

    def ensure_extra_nodes(self, profiles: dict[str, str]) -> None:
        """Register the additional pool cards that have a verified credential profile (others stay unregistered)."""

        db = self._db()
        for node in EXTRA_POOL_NODES:
            profile = str(profiles.get(node["node_id"]) or "")
            if not profile:
                continue
            row = db.execute("SELECT * FROM pool_nodes WHERE id=?", (node["node_id"],)).fetchone()
            if row is not None and row["profile"] == profile and row["gpu_model"] == node["label"] and int(row["vram_mb"]) == node["vram_mb"]:
                continue
            self.upsert_node(node["node_id"], label=node["label"], gpu_uuid=row["gpu_uuid"] if row else "", gpu_model=node["label"],
                             vram_mb=node["vram_mb"], profile=profile, enabled=bool(row["enabled"]) if row else True)

    def set_node_enabled(self, node_id: str, enabled: bool) -> None:
        with self._tx() as db:
            db.execute("UPDATE pool_nodes SET enabled=?, updated_at=? WHERE id=?", (int(enabled), self.clock(), node_id))
            self.audit("node_enabled" if enabled else "node_disabled", detail={"node": node_id}, db=db)

    def _node(self, db: sqlite3.Connection) -> sqlite3.Row | None:
        return db.execute("SELECT * FROM pool_nodes WHERE id=? AND enabled=1", (ALLOWED_NODE_ID,)).fetchone()

    def _nodes(self, db: sqlite3.Connection) -> list[sqlite3.Row]:
        """Enabled, schedulable cards in preference order (the primary first)."""

        rows = [row for row in db.execute("SELECT * FROM pool_nodes WHERE enabled=1") if row["id"] in POOL_NODE_IDS]
        return sorted(rows, key=lambda row: (NODE_PRIORITY.get(row["id"], 99), POOL_NODE_IDS.index(row["id"])))

    def idle_nodes(self) -> list[dict]:
        """Schedulable cards that hold no lease right now."""

        db = self._db()
        leased = {row["node_id"] for row in db.execute("SELECT node_id FROM gpu_leases")}
        return [dict(row) for row in self._nodes(db) if row["id"] not in leased]

    def set_quota(self, tenant_id: str, gpu_hours: float, period_days: int = 30) -> None:
        if gpu_hours < 0 or period_days < 1:
            raise SchedulerError("gpu_policy_invalid")
        with self._tx() as db:
            db.execute(
                "INSERT INTO quotas(tenant_id,gpu_hours,period_days,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(tenant_id) DO UPDATE SET gpu_hours=excluded.gpu_hours,period_days=excluded.period_days,updated_at=excluded.updated_at",
                (tenant_id, float(gpu_hours), int(period_days), self.clock()),
            )

    # -- budget --
    def _rolling(self, db: sqlite3.Connection, tenant_id: str) -> tuple[float, float]:
        """(budget seconds, window seconds); a quotas row overrides the configured rolling budget."""

        row = db.execute("SELECT gpu_hours, period_days FROM quotas WHERE tenant_id=?", (tenant_id,)).fetchone()
        if row is not None:
            return float(row["gpu_hours"]) * 3600.0, float(row["period_days"]) * 86400.0
        return float(self.policy.rolling_budget_seconds), float(self.policy.rolling_window_seconds)

    def _day_start(self, now: float) -> float:
        offset = self.policy.day_utc_offset_seconds
        return ((now + offset) // 86400) * 86400 - offset

    def _used_since(self, db: sqlite3.Connection, tenant_id: str, since: float, now: float) -> float:
        total = 0.0
        for row in db.execute("SELECT started_at, finished_at FROM tenant_gpu_usage WHERE tenant_id=? AND finished_at > ?", (tenant_id, since)):
            total += max(0.0, float(row["finished_at"]) - max(float(row["started_at"]), since))
        for row in db.execute("SELECT started_at, dispatched_at, reason FROM train_jobs WHERE tenant_id=? AND status=? AND started_at IS NOT NULL", (tenant_id, RUNNING)):
            if row["reason"] == PREPARING:
                continue  # dependency preparation is not GPU use
            total += max(0.0, now - max(float(row["dispatched_at"] or row["started_at"]), since))
        return total

    def _budget(self, db: sqlite3.Connection, tenant_id: str, now: float) -> dict[str, float]:
        rolling_budget, window = self._rolling(db, tenant_id)
        daily_used = self._used_since(db, tenant_id, self._day_start(now), now)
        rolling_used = self._used_since(db, tenant_id, now - window, now)
        daily_left = max(0.0, self.policy.daily_budget_seconds - daily_used)
        rolling_left = max(0.0, rolling_budget - rolling_used)
        return {"daily_used": daily_used, "rolling_used": rolling_used, "daily_left": daily_left,
                "rolling_left": rolling_left, "remaining": min(daily_left, rolling_left)}

    def _require_budget(self, db: sqlite3.Connection, tenant_id: str, now: float) -> None:
        budget = self._budget(db, tenant_id, now)
        if budget["remaining"] < self.policy.min_grant_seconds:
            # Say which limit it is: "tomorrow" is false while the rolling window is the one that ran out.
            raise SchedulerError("quota_exhausted" if budget["daily_left"] < self.policy.min_grant_seconds else "quota_exhausted_rolling")

    def usage(self, tenant_id: str) -> dict[str, Any]:
        budget = self._budget(self._db(), tenant_id, self.clock())
        # C7c-FIX5g: used rounds UP and remaining rounds DOWN, so the whole-minute figures always add up (used + remaining = limit
        # while the daily limit binds) and the tenant is never shown more time than is left.
        floor = lambda value: int(value // 60)
        ceil = lambda value: int(-(-float(value) // 60))
        return {"daily_used_minutes": ceil(budget["daily_used"]), "daily_limit_minutes": self.policy.daily_budget_seconds // 60,
                "rolling_used_minutes": ceil(budget["rolling_used"]), "remaining_minutes": floor(budget["remaining"])}

    def job_billed_seconds(self, job_id: Any) -> float:
        """Seconds booked to this job (0 when it never reached the card). The ledger decides what the card may promise."""
        row = self._db().execute("SELECT seconds FROM tenant_gpu_usage WHERE job_id=?", (job_id,)).fetchone()
        return float(row["seconds"]) if row else 0.0

    def usage_rows(self, tenant_id: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM tenant_gpu_usage" + (" WHERE tenant_id=?" if tenant_id else "") + " ORDER BY id DESC LIMIT ?"
        return [dict(row) for row in self._db().execute(sql, ((tenant_id, limit) if tenant_id else (limit,)))]

    def _book(self, db: sqlite3.Connection, job: sqlite3.Row | dict, end: float, reason: str, lease_id: str | None) -> float:
        """Book a finished lease once (UNIQUE job_id); a second call is a no-op."""

        if job["started_at"] is None or job["reason"] == PREPARING:
            return 0.0  # never dispatched (dependency preparation only): not charged
        start = float(job["dispatched_at"] or job["started_at"])
        if job["deadline_at"] is not None:
            end = min(end, float(job["deadline_at"]) + self.policy.kill_grace_seconds)
        end = max(start, end)
        seconds = end - start
        db.execute(
            "INSERT OR IGNORE INTO tenant_gpu_usage(tenant_id,job_id,lease_id,run_id,started_at,finished_at,seconds,reason,booked_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (job["tenant_id"], job["id"], lease_id, job["run_id"], start, end, seconds, reason, self.clock()),
        )
        return seconds

    # -- jobs --
    def submit(self, *, tenant_id: str, owner_id: str, run_id: str, vram_mb: int | None = None,
               requested_seconds: int | None = None, max_hours: float | None = None,
               required_node: str | None = None, budget_exempt: bool = False) -> dict:
        now = self.clock()
        vram = int(vram_mb or self.policy.default_vram_mb)
        if requested_seconds is None and max_hours is not None:
            requested_seconds = int(float(max_hours) * 3600)
        requested = int(requested_seconds or self.policy.max_run_seconds)
        if not tenant_id or not run_id or vram <= 0 or requested <= 0:
            raise SchedulerError("gpu_request_invalid")
        if not budget_exempt:
            requested = min(requested, self.policy.max_run_ceiling_seconds)
        if required_node not in (None, "", *POOL_NODE_IDS):
            raise SchedulerError("pool_unavailable")
        with self._tx() as db:
            existing = db.execute(
                "SELECT * FROM train_jobs WHERE run_id=? AND tenant_id=? AND status IN (?,?) ORDER BY seq LIMIT 1",
                (run_id, tenant_id, QUEUED, RUNNING)).fetchone()
            if existing:
                db.execute("UPDATE train_jobs SET waiter_at=? WHERE id=? AND status=?", (now, existing["id"], QUEUED))
                return dict(db.execute("SELECT * FROM train_jobs WHERE id=?", (existing["id"],)).fetchone())
            nodes = self._nodes(db)
            if not nodes:
                raise SchedulerError("pool_unavailable")
            if vram > max(int(node["vram_mb"]) for node in nodes):
                raise SchedulerError("insufficient_gpu_memory")
            if not budget_exempt:
                self._require_budget(db, tenant_id, now)
            # Held jobs without a waiter do not count: they cannot fill the queue for a whole hold window.
            queued = [row for row in db.execute("SELECT tenant_id, waiter_at FROM train_jobs WHERE status=?", (QUEUED,)).fetchall()
                      if self._live(row, now)]
            if (len(queued) >= self.policy.max_queue_length
                    or sum(1 for row in queued if row["tenant_id"] == tenant_id) >= self.policy.max_queued_per_tenant):
                raise SchedulerError("queue_full")
            job_id = "gjob_" + uuid.uuid4().hex
            db.execute(
                "INSERT INTO train_jobs(id,tenant_id,owner_id,run_id,vram_mb,max_hours,status,required_node,created_at,"
                "requested_seconds,waiter_at,budget_exempt) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, tenant_id, owner_id or tenant_id, run_id, vram, requested / 3600.0, QUEUED,
                 required_node or (ALLOWED_NODE_ID if not EXTRA_POOL_NODES else None), now, requested, now, int(budget_exempt)),
            )
            self.audit("submitted", tenant_id=tenant_id, job_id=job_id, detail={"requested_seconds": requested, "vram_mb": vram}, db=db)
            return dict(db.execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone())

    def job(self, job_id: str) -> dict:
        row = self._db().execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def lease_for(self, job_id: str) -> dict | None:
        row = self._db().execute("SELECT * FROM gpu_leases WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def touch_waiter(self, job_id: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE train_jobs SET waiter_at=? WHERE id=? AND status=?", (self.clock(), job_id, QUEUED))

    def _requested(self, row: sqlite3.Row | dict) -> int:
        value = row["requested_seconds"]
        if value is None:  # legacy hot-patch row
            value = int(float(row["max_hours"] or 0) * 3600) or self.policy.max_run_seconds
        return int(value) if row["budget_exempt"] else min(int(value), self.policy.max_run_ceiling_seconds)

    def _live(self, row: sqlite3.Row | dict, now: float) -> bool:
        """Someone is still waiting for this job: its tool call refreshed the waiter recently."""

        return row["waiter_at"] is not None and float(row["waiter_at"]) >= now - self.waiter_ttl

    def _ordered_queue(self, db: sqlite3.Connection, now: float) -> list[sqlite3.Row]:
        rows = {row["id"]: row for row in db.execute("SELECT * FROM train_jobs WHERE status=? ORDER BY seq", (QUEUED,))}
        tenants = {row["tenant_id"] for row in rows.values()}
        usage = {tenant: self._used_since(db, tenant, now - self.policy.rolling_window_seconds, now) for tenant in tenants}
        entries = [QueueEntry(row["id"], row["tenant_id"], int(row["seq"]), float(row["created_at"]), self._requested(row)) for row in rows.values()]
        return [rows[entry.job_id] for entry in fair_order(entries, usage, now)]

    def next_queued_request(self, *, excluding_tenant: str | None = None) -> int | None:
        """VRAM MiB of the job that would be granted next (fair order), for the broker's linger decision."""

        db = self._db()
        running = {row["tenant_id"] for row in db.execute("SELECT tenant_id FROM train_jobs WHERE status=?", (RUNNING,))}
        now = self.clock()
        for row in self._ordered_queue(db, now):
            if row["tenant_id"] in running or row["tenant_id"] == excluding_tenant or not self._live(row, now):
                continue
            return int(row["vram_mb"])
        return None

    def _live_order(self, db: sqlite3.Connection, now: float, job_id: str) -> list[sqlite3.Row]:
        """The fair order without dead waiters; ``job_id`` itself always stays in it."""

        return [row for row in self._ordered_queue(db, now) if row["id"] == job_id or self._live(row, now)]

    def queue_position(self, job_id: str) -> int | None:
        ordered = [row["id"] for row in self._live_order(self._db(), self.clock(), job_id)]
        return ordered.index(job_id) + 1 if job_id in ordered else None

    def queue_status(self, job_id: str) -> dict[str, Any]:
        """1-based position among live jobs and an ETA estimate for one queued job."""

        db = self._db()
        now = self.clock()
        ordered = self._live_order(db, now, job_id)
        ids = [row["id"] for row in ordered]
        if job_id not in ids:
            return {"position": None, "eta_seconds": None}
        index = ids.index(job_id)
        leases = {row["node_id"]: row for row in db.execute("SELECT node_id, deadline_at, granted_at FROM gpu_leases")}
        cards = [row["id"] for row in self._nodes(db)] or [ALLOWED_NODE_ID]
        deadline = None
        if all(card in leases for card in cards):
            # Every card is busy: the first one to free up starts the queue again.
            deadline = min(lease["deadline_at"] if lease["deadline_at"] is not None else float(lease["granted_at"]) + self.policy.max_run_seconds
                           for lease in (leases[card] for card in cards))
        # ADDENDUM 2: a job the broker is freeing memory for adds its stop estimate; restore after a job never does.
        freeing = [self.policy.broker_stop_seconds for row in ordered[: index + 1] if row["reason"] == FREEING]
        ahead = [self._requested(row) / len(cards) for row in ordered[:index]]  # the jobs ahead spread over the cards
        status = {"position": index + 1,
                  "eta_seconds": eta_seconds(now, deadline, [*ahead, *freeing]),
                  "eta_is_estimate": True}
        return status if self._live(ordered[index], now) else {**status, "held": True}

    def expire_leases(self) -> list[str]:
        """Heartbeat TTL expiry (kept for the hot-patch API); full repair lives in ``reconcile``."""

        return [item["job_id"] for item in self.reconcile()["expired"]]

    def mark_reason(self, job_id: str, reason: str | None) -> None:
        with self._tx() as db:
            db.execute("UPDATE train_jobs SET reason=? WHERE id=? AND status=?", (reason, job_id, QUEUED))

    def mark_preparing(self, job_id: str) -> None:
        """C7c-FIX4: the lease is held but nothing runs yet (dependency preparation); the budget is not charged."""

        with self._tx() as db:
            db.execute("UPDATE train_jobs SET reason=? WHERE id=? AND status=? AND dispatched_at IS NULL", (PREPARING, job_id, RUNNING))

    def clear_preparing(self, job_id: str) -> None:
        """Completed work is never left unbilled: drop the prep mark (billing then runs from the lease start, as before)."""

        with self._tx() as db:
            db.execute("UPDATE train_jobs SET reason=NULL WHERE id=? AND status=? AND reason=?", (job_id, RUNNING, PREPARING))

    def mark_dispatched(self, job_id: str) -> bool:
        """C7c-FIX4: the work is handed to the card; billing starts here.

        C7c-FIX5: atomic with the tenant's cancel. A job whose cancel is already requested (or that is no longer
        running) is NOT marked: ``False`` tells the runtime not to dispatch. Either the cancel lands first (no dispatch,
        nothing billed) or this lands first (a normal running job the cancel then stops).
        """

        with self._tx() as db:
            cursor = db.execute("UPDATE train_jobs SET reason=NULL, dispatched_at=? WHERE id=? AND status=? AND cancel_requested_at IS NULL",
                                (self.clock(), job_id, RUNNING))
            return bool(cursor.rowcount)

    def mark_freeing(self, job_id: str, freeing: bool) -> None:
        """ADDENDUM 2: the head job waits while the broker frees memory; it holds the card meanwhile."""

        with self._tx() as db:
            db.execute("UPDATE train_jobs SET reason=?, waiter_at=? WHERE id=? AND status=?",
                       (FREEING if freeing else None, self.clock(), job_id, QUEUED))

    def schedule(self, free_vram: "FreeVramProbe | int | None" = None, *, require_waiter: bool = True,
                 only_job: str | None = None) -> list[Grant]:
        """Grant the idle cards to the eligible queued jobs in fair order; never preempts.

        ``free_vram`` is a callable ``(node) -> free MB | None``, a ``{node_id: free MB | None}`` map (a missing card is
        unknown), or a number from an earlier probe (applied to every idle card). ``None`` means unknown/unhealthy: no grant.
        Jobs whose waiter has gone away keep their place (held) but are not granted and
        do not block the first live job.

        2026-11-05 (multi-card): each job goes to the best idle card that fits it now: the lowest ``priority`` first, then
        the one with the most free memory. When the head job fits no idle card it keeps its claim on every card big
        enough for it (no skipping ahead, so large requests are never starved); only cards too small to ever hold it
        stay open for the jobs behind it.
        """

        self.reconcile()
        now = self.clock()
        with self._tx() as db:
            leased = {row["node_id"] for row in db.execute("SELECT node_id FROM gpu_leases")}
            idle = [node for node in self._nodes(db) if node["id"] not in leased]
            if not idle:
                return []
            running_tenants = {row["tenant_id"] for row in db.execute("SELECT tenant_id FROM train_jobs WHERE status=?", (RUNNING,))}
            if only_job is None and db.execute(
                    "SELECT 1 FROM train_jobs WHERE status=? AND reason=? AND waiter_at >= ?", (QUEUED, FREEING, now - self.waiter_ttl)).fetchone():
                return []  # memory is being freed for the head job; nobody else takes the card meanwhile
            probed: dict[str, int | None] = {}

            def free_of(node: sqlite3.Row) -> int | None:
                if node["id"] not in probed:
                    if callable(free_vram):
                        value = free_vram(dict(node))
                    elif isinstance(free_vram, dict):
                        value = free_vram.get(node["id"])
                    else:
                        value = int(node["vram_mb"]) if free_vram is None else free_vram
                    probed[node["id"]] = None if value is None else int(value)
                return probed[node["id"]]

            grants: list[Grant] = []
            for job in self._ordered_queue(db, now):
                if not idle:
                    break
                if only_job is not None and job["id"] != only_job:
                    continue
                if job["tenant_id"] in running_tenants:
                    continue
                if require_waiter and not self._live(job, now):
                    continue
                if job["budget_exempt"]:
                    max_run = self._requested(job)
                else:
                    budget = self._budget(db, job["tenant_id"], now)
                    max_run = int(min(self._requested(job), self.policy.max_run_ceiling_seconds, budget["remaining"]))
                    if max_run < self.policy.min_grant_seconds:
                        limit = "quota_exhausted" if budget["daily_left"] < self.policy.min_grant_seconds else "quota_exhausted_rolling"
                        self._finish_row(db, job, REJECTED, {"reason": limit}, now)
                        continue
                need = int(job["vram_mb"])
                frees = [(node, free_of(node)) for node in idle]
                fitting = [(node, free) for node, free in frees if free is not None and free >= need]
                if not fitting:
                    # Head-of-line: no skipping ahead on any card this job could use, so large requests are never starved.
                    known = any(free is not None for _node, free in frees)
                    db.execute("UPDATE train_jobs SET reason=? WHERE id=?", ("insufficient_gpu_memory" if known else "pool_unavailable", job["id"]))
                    idle = [node for node in idle if int(node["vram_mb"]) < need]
                    continue
                node = min(fitting, key=lambda item: (NODE_PRIORITY.get(item[0]["id"], 99), -int(item[1]), POOL_NODE_IDS.index(item[0]["id"])))[0]
                lease_id = "glease_" + uuid.uuid4().hex
                deadline = now + max_run
                db.execute("INSERT INTO gpu_leases(id,job_id,node_id,granted_at,heartbeat_at,tenant_id,deadline_at) VALUES(?,?,?,?,?,?,?)",
                           (lease_id, job["id"], node["id"], now, now, job["tenant_id"], deadline))
                db.execute("UPDATE train_jobs SET status=?, node_id=?, started_at=?, max_run_seconds=?, deadline_at=?, reason=NULL WHERE id=?",
                           (RUNNING, node["id"], now, max_run, deadline, job["id"]))
                detail = {"max_run_seconds": max_run} if len(POOL_NODE_IDS) == 1 else {"max_run_seconds": max_run, "node": node["id"]}
                self.audit("granted", tenant_id=job["tenant_id"], job_id=job["id"], detail=detail, db=db)
                grants.append(Grant(job["id"], lease_id, node["id"], str(node["profile"]), deadline, max_run))
                idle = [item for item in idle if item["id"] != node["id"]]
                running_tenants.add(job["tenant_id"])
        return grants

    def heartbeat(self, lease_id: str) -> bool:
        with self._tx() as db:
            return db.execute("UPDATE gpu_leases SET heartbeat_at=? WHERE id=?", (self.clock(), lease_id)).rowcount == 1

    def lease_state(self, job_id: str) -> dict[str, Any]:
        """What the lease supervisor must do now: keep running, stop for cancel, or stop at the deadline."""

        db = self._db()
        job = db.execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or job["status"] != RUNNING:
            return {"action": "released"}
        now = self.clock()
        if job["cancel_requested_at"] is not None:
            return {"action": "cancel"}
        if job["deadline_at"] is not None and now >= float(job["deadline_at"]):
            return {"action": "timeout"}
        return {"action": "run", "remaining_seconds": max(0.0, float(job["deadline_at"]) - now) if job["deadline_at"] is not None else None}

    def mark_kill_requested(self, job_id: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE gpu_leases SET kill_requested_at=COALESCE(kill_requested_at, ?) WHERE job_id=?", (self.clock(), job_id))

    def _finish_row(self, db: sqlite3.Connection, job: sqlite3.Row, status: str, detail: dict | None, end: float) -> None:
        lease = db.execute("SELECT * FROM gpu_leases WHERE job_id=?", (job["id"],)).fetchone()
        db.execute("DELETE FROM gpu_leases WHERE job_id=?", (job["id"],))
        reason = (detail or {}).get("reason")
        if status == FAILED and not reason and job["status"] == RUNNING and job["reason"] == PREPARING and job["dispatched_at"] is None:
            reason = ENV_NOT_READY  # never handed over: a preparation failure, which the tenant UI words as such
        db.execute("UPDATE train_jobs SET status=?, finished_at=?, detail=?, reason=? WHERE id=?",
                   (status, end, json.dumps(detail or {}, sort_keys=True), reason, job["id"]))
        if job["status"] == RUNNING:
            self._book(db, job, end, status, lease["id"] if lease else None)
        self.audit("finished", tenant_id=job["tenant_id"], job_id=job["id"], detail={"status": status, **(detail or {})}, db=db)

    def finish(self, job_id: str, status: str, detail: dict | None = None, *, ended_at: float | None = None) -> dict:
        if status not in TERMINAL:
            raise SchedulerError("gpu_request_invalid")
        with self._tx() as db:
            row = db.execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in TERMINAL:
                self._finish_row(db, row, status, detail, self.clock() if ended_at is None else ended_at)
        return self.job(job_id)

    def cancel(self, job_id: str, tenant_id: str) -> dict:
        """Tenant-scoped and idempotent. Queued: cancelled now. Preparing or already dispatched: settle at once (bill only if dispatched)."""

        with self._tx() as db:
            row = db.execute("SELECT * FROM train_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["tenant_id"] != tenant_id:
                raise KeyError(job_id)
            if row["status"] == QUEUED:
                self._finish_row(db, row, CANCELLED, {"reason": "cancelled_by_owner"}, self.clock())
            elif row["status"] == RUNNING and row["cancel_requested_at"] is None:
                db.execute("UPDATE train_jobs SET cancel_requested_at=? WHERE id=?", (self.clock(), job_id))
                self.audit("cancel_requested", tenant_id=tenant_id, job_id=job_id, db=db)
                if not self._settle_prep_cancel(db, row, "cancelled_by_owner"):
                    self._settle_post_dispatch_cancel(db, row, "cancelled_by_owner")
        return self.job(job_id)

    def _settle_prep_cancel(self, db: sqlite3.Connection, row: sqlite3.Row, reason: str) -> bool:
        """C7c-FIX5c: a cancel while the environment is still being prepared settles the job at once.

        Nothing was handed to the card and nothing is billed, so there is nothing to wait for. Waiting for the in-flight
        preparation step (a dependency install can run for minutes; the cancel is only honoured between steps) left the
        card on 「正在取消…」 for minutes after the run was already cancelled. The preparing thread still ends by itself:
        its dispatch report is refused, and its release finds the job settled.
        """

        if row["status"] != RUNNING or row["reason"] != PREPARING or row["dispatched_at"] is not None:
            return False
        self._finish_row(db, row, CANCELLED, {"reason": reason, "while": "preparing"}, self.clock())
        self._mark_orphan(db, row["id"], self.clock())  # a preparation step still in flight is cleaned up before the next grant
        return True

    def _settle_post_dispatch_cancel(self, db: sqlite3.Connection, row: sqlite3.Row, reason: str) -> bool:
        """C7e: a cancel after the work was handed to the card (「已发出」) settles at once, same shape as prep-cancel.

        LIVE C2 on FIX5c tip 8f29426 spent 21.056 s on 「正在取消…」: cancel only set ``cancel_requested_at``, then
        LeaseHandle waited up to ``heartbeat_seconds`` (30) before kill+release, and reconcile's cancel hard-stop was
        ``kill_grace + kill_timeout`` (150 s). The tenant card must reach terminal 「已取消」 within ≤10 s (FE poll
        ~1.5 s). Billing still runs for dispatch→cancel via ``_finish_row``/``_book``; the remote process is cleaned
        by the orphan killer the way a settled prep step is.
        """

        if row["status"] != RUNNING or row["dispatched_at"] is None:
            return False
        self._finish_row(db, row, CANCELLED, {"reason": reason, "while": "dispatched"}, self.clock())
        self._mark_orphan(db, row["id"], self.clock())
        return True

    def cancel_run(self, run_id: str) -> list[str]:
        """The Run was stopped: same effect as the tenant's cancel, for whatever job it has."""

        cancelled = []
        with self._tx() as db:
            for row in db.execute("SELECT * FROM train_jobs WHERE run_id=? AND status IN (?,?)", (run_id, QUEUED, RUNNING)).fetchall():
                if row["status"] == QUEUED:
                    self._finish_row(db, row, CANCELLED, {"reason": "run_stopped"}, self.clock())
                elif row["cancel_requested_at"] is None:
                    db.execute("UPDATE train_jobs SET cancel_requested_at=? WHERE id=?", (self.clock(), row["id"]))
                    self.audit("cancel_requested", tenant_id=row["tenant_id"], job_id=row["id"], detail={"reason": "run_stopped"}, db=db)
                    if not self._settle_prep_cancel(db, row, "run_stopped"):
                        self._settle_post_dispatch_cancel(db, row, "run_stopped")
                cancelled.append(row["id"])
        return cancelled

    def jobs_for_tenant(self, tenant_id: str, *, run_id: str | None = None, limit: int = 20) -> list[dict]:
        sql = "SELECT * FROM train_jobs WHERE tenant_id=?" + (" AND run_id=?" if run_id else "") + " ORDER BY seq DESC LIMIT ?"
        params = (tenant_id, run_id, limit) if run_id else (tenant_id, limit)
        return [dict(row) for row in self._db().execute(sql, params)]

    # -- reconcile (C7c-2) --
    def reconcile(self, process_alive: Callable[[dict], "bool | None"] | None = None) -> dict[str, list]:
        """Idempotent repair after a service restart or a runner crash.

        * lease heartbeat older than TTL, or ``process_alive(job)`` is False → job ``lost``,
          usage booked up to the last heartbeat (once).
        * lease past deadline + grace + kill timeout (a stuck remote kill) → released as ``timeout``
          or ``cancelled``; the lease never leaks.
        * ``running`` job without a lease → ``lost`` (booked up to its start; nothing provable ran).
        * lease without a running job → deleted.
        * queued job whose waiter is gone longer than the hold window → ``cancelled``.
        The queue itself is preserved.
        """

        now = self.clock()
        report: dict[str, list] = {"expired": [], "stuck": [], "orphan_jobs": [], "orphan_leases": [], "holds_expired": []}
        with self._tx() as db:
            for lease in db.execute("SELECT * FROM gpu_leases").fetchall():
                job = db.execute("SELECT * FROM train_jobs WHERE id=?", (lease["job_id"],)).fetchone()
                if job is None or job["status"] != RUNNING:
                    db.execute("DELETE FROM gpu_leases WHERE id=?", (lease["id"],))
                    report["orphan_leases"].append(lease["id"])
                    continue
                deadline = lease["deadline_at"]
                hard_stop = None if deadline is None else float(deadline) + self.policy.kill_grace_seconds + self.policy.kill_timeout_seconds
                if job["cancel_requested_at"] is not None:
                    # C7e: card SLA ≤10 s for cancel (prep/dispatch settle at once; this caps any leftover RUNNING+cancel_requested).
                    cancel_stop = float(job["cancel_requested_at"]) + 10.0
                    hard_stop = cancel_stop if hard_stop is None else min(hard_stop, cancel_stop)
                if float(lease["heartbeat_at"]) < now - self.lease_ttl:
                    self._finish_row(db, job, LOST, {"reason": "lease_heartbeat_expired"}, float(lease["heartbeat_at"]))
                    self._mark_orphan(db, job["id"], now)
                    report["expired"].append({"job_id": job["id"], "tenant_id": job["tenant_id"]})
                elif hard_stop is not None and now >= hard_stop:
                    status = CANCELLED if job["cancel_requested_at"] is not None else TIMEOUT
                    self._finish_row(db, job, status, {"reason": "remote_kill_unconfirmed"}, now)
                    self._mark_orphan(db, job["id"], now)
                    report["stuck"].append({"job_id": job["id"], "tenant_id": job["tenant_id"]})
            for job in db.execute("SELECT * FROM train_jobs WHERE status=? AND id NOT IN (SELECT job_id FROM gpu_leases)", (RUNNING,)).fetchall():
                self._finish_row(db, job, LOST, {"reason": "running_without_lease"}, float(job["started_at"] or now))
                self._mark_orphan(db, job["id"], now)
                report["orphan_jobs"].append(job["id"])
            hold = now - self.policy.queue_hold_seconds
            for job in db.execute("SELECT * FROM train_jobs WHERE status=? AND COALESCE(waiter_at, created_at) < ?", (QUEUED, hold)).fetchall():
                self._finish_row(db, job, CANCELLED, {"reason": "queue_hold_expired"}, now)
                report["holds_expired"].append(job["id"])
        if process_alive is not None:
            for lease in [dict(row) for row in self._db().execute("SELECT * FROM gpu_leases")]:
                job = self.job(lease["job_id"])
                if job["status"] == RUNNING and process_alive(job) is False:
                    with self._tx() as db:
                        current = db.execute("SELECT * FROM train_jobs WHERE id=?", (job["id"],)).fetchone()
                        if current is not None and current["status"] == RUNNING:
                            self._finish_row(db, current, LOST, {"reason": "remote_process_gone"}, float(lease["heartbeat_at"]))
                            report["expired"].append({"job_id": job["id"], "tenant_id": job["tenant_id"]})
        killed = self._kill_orphans()
        if killed:
            report["orphans_killed"] = killed
        return report

    def clear_orphan(self, job_id: str) -> None:
        """C7e-R1: the remote worker of a settled job was confirmed gone; reconcile need not retry the kill."""

        with self._tx() as db:
            db.execute("UPDATE train_jobs SET orphan_kill_at=NULL WHERE id=?", (job_id,))

    def mark_orphan_pending(self, job_id: str, tenant_id: str = "") -> None:
        """2026-11-03: a kill that was not confirmed by release time is queued for the reconcile kill (pid marker)
        and audited as an alert, so the remote worker is never silently left running on the card."""

        with self._tx() as db:
            db.execute("UPDATE train_jobs SET orphan_kill_at=COALESCE(orphan_kill_at, ?) WHERE id=?", (self.clock(), job_id))
            self.audit("orphan_kill_pending", tenant_id=tenant_id, job_id=job_id, db=db)

    @staticmethod
    def _mark_orphan(db: sqlite3.Connection, job_id: str, now: float) -> None:
        db.execute("UPDATE train_jobs SET orphan_kill_at=? WHERE id=?", (now, job_id))

    def _kill_orphans(self) -> list[str]:
        """Stop remote processes of lost jobs before the next grant; otherwise it hits the remote slot lock.

        A failed attempt is retried after ``orphan_kill_retry_seconds``. Once the remote ``timeout``
        must have ended the process anyway, the record is dropped.
        """

        killer = self.orphan_killer
        if killer is None:
            return []
        now = self.clock()
        killed = []
        rows = self._db().execute("SELECT * FROM train_jobs WHERE orphan_kill_at IS NOT NULL AND orphan_kill_at <= ?", (now,)).fetchall()
        for row in rows:
            job = dict(row)
            if job["deadline_at"] is not None and now > float(job["deadline_at"]) + self.lease_ttl + self.policy.max_run_ceiling_seconds:
                with self._tx() as db:
                    db.execute("UPDATE train_jobs SET orphan_kill_at=NULL WHERE id=?", (job["id"],))
                continue
            with self._tx() as db:  # concurrent waiters do not repeat the same attempt
                db.execute("UPDATE train_jobs SET orphan_kill_at=? WHERE id=?", (now + self.policy.orphan_kill_retry_seconds, job["id"]))
            try:
                confirmed = bool(killer(job))
            except Exception:  # noqa: BLE001 - unreachable host: retried, never blocks the queue
                confirmed = False
            with self._tx() as db:
                if confirmed:
                    db.execute("UPDATE train_jobs SET orphan_kill_at=NULL WHERE id=?", (job["id"],))
                self.audit("orphan_killed" if confirmed else "orphan_kill_failed", tenant_id=job["tenant_id"], job_id=job["id"], db=db)
            if confirmed:
                killed.append(job["id"])
        return killed

    # -- admin view --
    def snapshot(self) -> dict:
        db = self._db()
        now = self.clock()
        leases = {row["node_id"]: dict(row) for row in db.execute("SELECT * FROM gpu_leases")}
        nodes = []
        for row in db.execute("SELECT * FROM pool_nodes ORDER BY id"):
            lease = leases.get(row["id"])
            nodes.append({**dict(row), "schedulable": row["id"] in POOL_NODE_IDS and bool(row["enabled"]),
                          "priority": NODE_PRIORITY.get(row["id"]),
                          "busy": bool(lease), "job_id": lease["job_id"] if lease else None,
                          "deadline_in_seconds": round(float(lease["deadline_at"]) - now, 1) if lease and lease["deadline_at"] is not None else None,
                          "heartbeat_age_seconds": round(now - lease["heartbeat_at"], 1) if lease else None})
        queue = [{"job_id": r["id"], "tenant_id": r["tenant_id"], "vram_mb": r["vram_mb"], "requested_seconds": self._requested(r),
                  "waiting_seconds": round(now - r["created_at"], 1), "reason": r["reason"]}
                 for r in self._ordered_queue(db, now)]
        tenants = sorted({r["tenant_id"] for r in db.execute("SELECT tenant_id FROM train_jobs UNION SELECT tenant_id FROM quotas")})
        usage = sorted(({"tenant_id": t, **self.usage(t)} for t in tenants), key=lambda u: -u["rolling_used_minutes"])
        return {"schema": "evomind.gpu_pool.v2", "nodes": nodes, "queue": queue, "usage": usage}
