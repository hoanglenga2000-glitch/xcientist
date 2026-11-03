"""t238 C7e O4: capacity watermarks for the single 4 vCPU / 8 GB / 180 GB host.

Gauges: ``runtime_db_bytes`` (runtime.sqlite3 + WAL), ``uploads_bytes``, ``data_free_bytes``,
``growth_bytes_per_day`` (DB + uploads, from a small persisted history), ``active_users_7d``,
``growth_bytes_per_active_user_day`` and ``days_to_full``. They feed the O3 alert rules in
``CAPACITY_RULES``; the model and thresholds are documented in ``docs/CAPACITY_MODEL_C7E.md``.

Walking the uploads tree can be slow, so a sample is taken at most every ``min_interval``
seconds (monitor tick) and health/metrics read the cached value; the walk itself is capped at
``MAX_WALK_FILES`` files (the result is then marked ``partial``).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .ops_alerts import Rule

GIB = 1024 ** 3
MAX_WALK_FILES = 200_000
HISTORY_MAX_POINTS = 60
HISTORY_MIN_SPACING_SECONDS = 6 * 3600
GROWTH_WINDOW_DAYS = 14
MIN_INTERVAL_SECONDS = 600.0

CAPACITY_RULES: tuple[Rule, ...] = (
    Rule("capacity_runtime_db_large", "runtime_db_bytes", ">=", 2 * GIB, "warning", 0, "运行时数据库超过 2 GiB"),
    Rule("capacity_runtime_db_critical", "runtime_db_bytes", ">=", 8 * GIB, "critical", 0, "运行时数据库超过 8 GiB"),
    Rule("capacity_uploads_large", "uploads_bytes", ">=", 100 * GIB, "warning", 0, "上传文件总量超过 100 GiB"),
    Rule("capacity_uploads_critical", "uploads_bytes", ">=", 140 * GIB, "critical", 0, "上传文件总量超过 140 GiB"),
    Rule("capacity_days_to_full_warning", "days_to_full", "<", 30, "warning", 0, "按当前增长，30 天内磁盘会满"),
    Rule("capacity_days_to_full_critical", "days_to_full", "<", 7, "critical", 0, "按当前增长，7 天内磁盘会满"),
    # t238 C7e O1: the scheduled backup's own result (scripts/ops/evomind_backup.py writes last_run.json).
    Rule("backup_skipped", "backup_last_failed", "==", 1, "critical", 0, "定时备份被跳过或校验失败（多半是磁盘空间不够）"),
    Rule("backup_stale", "backup_age_hours", ">", 36, "warning", 0, "超过 36 小时没有成功的定时备份"),
)
DEFAULT_BACKUP_ROOT = r"C:\ProgramData\EvoMind\backups"
CAPACITY_METRICS = ("runtime_db_bytes", "uploads_bytes", "data_free_bytes", "growth_bytes_per_day",
                    "active_users_7d", "growth_bytes_per_active_user_day", "days_to_full",
                    "backup_last_failed", "backup_age_hours")


def backup_status(root: str | os.PathLike | None = None, *, now: float | None = None) -> dict[str, Any]:
    """``backup_last_failed`` (1/0) and ``backup_age_hours`` since the last successful run; ``None`` = unknown
    (no backup root on this host, e.g. before the scheduled task is registered)."""
    base = Path(root or os.environ.get("EVOMIND_BACKUP_ROOT") or DEFAULT_BACKUP_ROOT) / "scheduled"
    status: dict[str, Any] = {"backup_last_failed": None, "backup_age_hours": None}
    try:
        record = json.loads((base / "last_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return status
    if not isinstance(record, dict):
        return status
    clock = time.time() if now is None else now
    status["backup_last_failed"] = 0 if record.get("status") == "ok" else 1
    success = None
    for marker in sorted((base / "daily").glob("*/verified.json")) if (base / "daily").is_dir() else []:
        try:
            item = json.loads(marker.read_text(encoding="utf-8"))
            if item.get("status") == "ok":
                success = max(success or 0.0, float(item.get("at_epoch") or 0))
        except (OSError, ValueError, TypeError):
            continue
    if success:
        status["backup_age_hours"] = round(max(0.0, clock - success) / 3600, 2)
    return status


def _tree_bytes(root: Path, limit: int = MAX_WALK_FILES) -> tuple[int, bool]:
    total, seen = 0, 0
    if not root.is_dir():
        return 0, False
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in files:
            seen += 1
            if seen > limit:
                return total, True
            try:
                info = (Path(current) / name).lstat()
            except OSError:
                continue
            total += info.st_size
    return total, False


def _db_bytes(runtime_root: Path) -> int:
    total = 0
    for suffix in ("", "-wal"):
        try:
            total += (runtime_root / f"runtime.sqlite3{suffix}").stat().st_size
        except OSError:
            pass
    return total


def active_users(runtime_root: Path, store, *, days: int = 7, now: datetime | None = None) -> int | None:
    """Distinct (tenant, owner) that created a run in the last ``days`` days; ``None`` if unknown."""
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
    try:
        with store._lock:
            ids = [row[0] for row in store._connection.execute(
                "SELECT id FROM assistant_runs WHERE created_at >= ? ORDER BY created_at DESC LIMIT 2000", (cutoff,))]
        path = Path(runtime_root) / "principal_access.sqlite3"
        if not ids:
            return 0
        if not path.is_file():
            return None
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1)
        try:
            marks = ",".join("?" * len(ids))
            row = connection.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT tenant_id, owner_id FROM resource_owners WHERE kind='run' AND resource_id IN ({marks}))",
                ids).fetchone()
        finally:
            connection.close()
        return int(row[0])
    except (sqlite3.Error, OSError):
        return None


class CapacityTracker:
    def __init__(self, runtime_root: Path, upload_root: Path, *, store=None,
                 disk_usage: Callable[[Any], Any] | None = None, clock: Callable[[], float] = time.time,
                 min_interval: float = MIN_INTERVAL_SECONDS, walk_limit: int = MAX_WALK_FILES) -> None:
        self.runtime_root, self.upload_root = Path(runtime_root), Path(upload_root)
        self.store = store
        self.disk_usage = disk_usage or shutil.disk_usage
        self.clock = clock
        self.min_interval = float(min_interval)
        self.walk_limit = walk_limit
        self.history_path = self.runtime_root / "ops" / "capacity_history.json"
        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._sampled_at = float("-inf")

    def _history(self) -> list[list[float]]:
        try:
            points = json.loads(self.history_path.read_text(encoding="utf-8"))
            return [[float(t), float(b)] for t, b in points][-HISTORY_MAX_POINTS:]
        except (OSError, ValueError, TypeError):
            return []

    def _save_history(self, points: list[list[float]]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_name(f".{self.history_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(points[-HISTORY_MAX_POINTS:]), encoding="utf-8")
        os.replace(temporary, self.history_path)

    def sample(self, *, force: bool = False) -> dict[str, Any]:
        """Measure now (rate limited by ``min_interval`` unless ``force``) and return the snapshot."""
        now = self.clock()
        with self._lock:
            if not force and self._cached is not None and now - self._sampled_at < self.min_interval:
                return dict(self._cached)
        db = _db_bytes(self.runtime_root)
        uploads, partial = _tree_bytes(self.upload_root, self.walk_limit)
        try:
            free = int(self.disk_usage(self.runtime_root).free)
        except (OSError, AttributeError, ValueError):
            free = None
        used = db + uploads
        points = self._history()
        if not points or now - points[-1][0] >= HISTORY_MIN_SPACING_SECONDS:
            points.append([now, float(used)])
            try:
                self._save_history(points)
            except OSError:
                pass
        window = [p for p in points if now - p[0] <= GROWTH_WINDOW_DAYS * 86400]
        growth = None
        if len(window) >= 2 and window[-1][0] - window[0][0] >= 86400:
            growth = max(0.0, (window[-1][1] - window[0][1]) / ((window[-1][0] - window[0][0]) / 86400))
        users = active_users(self.runtime_root, self.store) if self.store is not None else None
        snapshot = {
            "runtime_db_bytes": db,
            "uploads_bytes": uploads,
            "uploads_partial": partial,
            "data_free_bytes": free,
            "growth_bytes_per_day": None if growth is None else round(growth, 1),
            "active_users_7d": users,
            "growth_bytes_per_active_user_day": None if growth is None or not users else round(growth / users, 1),
            "days_to_full": None if growth is None or not growth or free is None else round(free / growth, 1),
            **backup_status(now=now),
        }
        with self._lock:
            self._cached, self._sampled_at = dict(snapshot), now
        return snapshot

    def cached(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._cached is None else dict(self._cached)


_trackers: dict[str, CapacityTracker] = {}
_trackers_lock = threading.Lock()


def tracker_for(runtime) -> CapacityTracker:
    key = str(Path(runtime.runtime_root).resolve())
    with _trackers_lock:
        tracker = _trackers.get(key)
        if tracker is None or tracker.store is not runtime.store:
            tracker = CapacityTracker(runtime.runtime_root, runtime.assistant.upload_root, store=runtime.store,
                                      disk_usage=getattr(runtime.assistant, "disk_usage", None))
            _trackers[key] = tracker
        return tracker


def health_component(runtime) -> dict[str, Any]:
    """Cached capacity for deep health (never walks the tree inside the health budget)."""
    tracker = tracker_for(runtime)
    snapshot = tracker.cached()
    if snapshot is None:
        threading.Thread(target=lambda: _safe_sample(tracker), name="evomind-capacity-sample", daemon=True).start()
        return {"ok": True, "state": "pending"}
    firing = [rule.name for rule in CAPACITY_RULES if rule.severity == "critical" and rule.breached(snapshot.get(rule.metric), None)]
    warn = [rule.name for rule in CAPACITY_RULES if rule.severity == "warning" and rule.breached(snapshot.get(rule.metric), None)]
    return {"ok": not firing, "critical": firing, "warning": warn,
            **{key: snapshot.get(key) for key in (*CAPACITY_METRICS, "uploads_partial")}}


def _safe_sample(tracker: CapacityTracker) -> None:
    try:
        tracker.sample()
    except Exception:  # noqa: BLE001 - best effort background refresh
        pass
