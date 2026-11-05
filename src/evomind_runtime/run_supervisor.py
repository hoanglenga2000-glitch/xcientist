"""t238 C7e A4: what happens after a run is STALLED (no progress for 180 s while the heartbeat says it is live).

    STALLED → probe → recoverable  → reattach (restart the run's worker with resume=True)
                    → unrecoverable → mark the run failed (`stalled_unrecoverable`) + run_failed event
                                      (the UI's bell shows terminal runs: that is the user notice)

Long-running training / queued GPU jobs are exempt from STALLED-kill and from the 30-minute
orphan settle whenever the scheduler/lease provider (`ops_providers` "job_liveness", plugged in by
C7c) reports the job as queued/running with a live heartbeat. With no provider registered the
exemption never applies and the orphan settle keeps its existing behaviour; the stall supervisor
itself only runs when the monitor ticks it (``EVOMIND_STALL_SUPERVISOR`` != "0").

What counts as a stall (FIX1 M2):
- an in-flight tool call whose ExecutionHeartbeat is fresh is NOT a stall, however long it runs
  without unit counts (a long download / launcher is normal work); nothing is counted or alerted;
- a fresh heartbeat with no progress for 180 s that belongs to no in-flight call is a stall;
- a heartbeat that has been silent for 180 s while no local thread owns the run is a stall
  (the executor died: the genuine no-heartbeat case).

Ownership (FIX1 H1): a run is locally owned when its worker thread, its resume waiter, or ANY
approval worker executing one of its calls is alive (same threads ``recover_incomplete`` checks).
An owned run is never failed or reattached; it is only counted. Otherwise: no tool in flight and
fewer than ``MAX_REATTACH`` reattaches → reattach; else fail visibly.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from .execution_progress import HEARTBEAT_SECONDS, STALE_SECONDS, STALLED_SECONDS

MAX_REATTACH = 2
LIVE_STATES = frozenset({"queued", "running", "leased", "pending"})
ACTIVE_RUN_STATUSES = frozenset({"queued", "planning", "running", "verifying", "recovering"})
# A provider heartbeat older than this is no longer "live" (4 missed scheduler beats).
JOB_HEARTBEAT_LIVE_SECONDS = max(STALE_SECONDS, 4 * HEARTBEAT_SECONDS)


def job_is_live(run_id: str) -> bool:
    """True only when the scheduler/lease provider vouches for a queued/running job with a live heartbeat."""
    from . import ops_providers

    value = ops_providers.call("job_liveness", run_id)
    if not isinstance(value, dict):
        return False
    state = str(value.get("state") or "").lower()
    try:
        age = float(value.get("heartbeat_age_seconds"))
    except (TypeError, ValueError):
        return False
    return state in LIVE_STATES and 0 <= age <= JOB_HEARTBEAT_LIVE_SECONDS



def _usable_timestamp(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _running_reference_time(job: dict, lease: dict | None, installed_at: float) -> float:
    """Epoch the stall age of a RUNNING job is measured from: lease heartbeat, else started_at, else created_at,
    else the supervisor's install time. Never "now" (that made age 0 and the exemption permanent, ACCEPT7 F1)."""
    for candidate in ((lease or {}).get("heartbeat_at"), job.get("started_at"), job.get("created_at")):
        value = _usable_timestamp(candidate)
        if value is not None:
            return value
    return float(installed_at)


def install_job_liveness(runtime_root, *, clock=None):
    """t238 C7e m-2: plug the C7c train_jobs / lease heartbeat into the stall supervisor.

    Reads only the local ``gpu_scheduler.sqlite3`` (no SSH). A queued or running job —
    including FIX4 ``preparing`` (running + reason=preparing, lease held, not billed) and a
    dispatched running job — reports ``state`` + ``heartbeat_age_seconds`` so a service
    restart with no local owner does not stall-fail an in-flight GPU job. Returns the
    unregister callable. Idempotent: replaces any previous registration.
    """
    import time as _time
    from pathlib import Path as _Path
    from . import ops_providers
    from .gpu_admission import DB_NAME
    from .gpu_scheduler import QUEUED, RUNNING, GpuScheduler

    root = _Path(runtime_root)
    # Anchor for a row with no usable timestamp: the moment this supervisor/service started. It is fixed (never
    # re-read as "now"), so such a row ages out of the live window instead of being exempt forever (ACCEPT7 F1).
    installed_at = float(clock() if clock is not None else _time.time())

    def provider(run_id: str):
        if not run_id:
            return None
        path = root / DB_NAME
        if not path.is_file():
            return None
        sched = GpuScheduler(path, clock=clock) if clock is not None else GpuScheduler(path)
        try:
            row = sched._db().execute(
                "SELECT * FROM train_jobs WHERE run_id=? AND status IN (?,?) ORDER BY seq DESC LIMIT 1",
                (run_id, QUEUED, RUNNING),
            ).fetchone()
            if row is None:
                return None
            job = dict(row)
            state = str(job["status"])
            now = float(sched.clock())
            if state == QUEUED:
                hb = _usable_timestamp(job["waiter_at"]) or _usable_timestamp(job["created_at"]) or installed_at
            else:
                hb = _running_reference_time(job, sched.lease_for(job["id"]), installed_at)
            return {"state": state, "heartbeat_age_seconds": max(0.0, now - hb)}
        finally:
            sched.close()

    return ops_providers.register("job_liveness", provider)



class StallSupervisor:
    def __init__(self, runtime, *, clock: Callable[[], float] = time.time, max_reattach: int = MAX_REATTACH) -> None:
        self.runtime = runtime
        self.clock = clock
        self.max_reattach = max_reattach
        self._lock = threading.Lock()
        self._seen: set[tuple[str, float]] = set()  # (run_id, last_progress_at) already counted as one stall

    def _progress(self, run_id: str, status: str) -> dict[str, Any] | None:
        """Same projection as ProgressStore.read, but on the injectable clock; None without any record."""
        from .execution_progress import project_progress
        import json
        with self.runtime.progress._connect() as connection:
            row = connection.execute("SELECT payload FROM execution_progress WHERE run_id=?", (run_id,)).fetchone()
        return project_progress(json.loads(row[0]), status=status, now=self.clock()) if row else None

    def _locally_owned(self, run_id: str, in_flight: list[dict[str, Any]]) -> bool:
        """FIX1 H1: the run's worker, its resume waiter, or an approval worker for one of its calls is alive."""
        assistant = self.runtime.assistant
        call_approvals = {str(call.get("approval_id") or "") for call in in_flight} - {""}
        try:
            stored = self.runtime.store.list_tool_calls(run_id, limit=2000)
        except Exception:  # noqa: BLE001 - the in-flight view is still checked
            stored = []
        call_approvals |= {str(call.get("approval_id") or "") for call in stored
                           if str(call.get("status") or "") in {"running", "waiting_approval"}} - {""}
        with assistant._lock:
            threads = [assistant._threads.get(run_id), getattr(assistant, "_resume_waiters", {}).get(run_id)]
            workers = dict(getattr(assistant, "_approval_workers", {}))
        threads += [workers.get(approval_id) for approval_id in call_approvals]
        if any(thread is not None and thread.is_alive() for thread in threads):
            return True
        # An approval worker whose approval belongs to this run, even if no call row names it (duplicate-turn guard).
        for approval_id, thread in workers.items():
            if thread is None or not thread.is_alive() or approval_id in call_approvals:
                continue
            try:
                approval = self.runtime.store.get_approval(approval_id) or {}
            except Exception:  # noqa: BLE001
                approval = {}
            if str(approval.get("session_id") or "") == run_id:
                return True
        return False

    def _is_stall(self, progress: dict[str, Any] | None, in_flight: list[dict[str, Any]], owned: bool) -> bool:
        if progress is None or progress.get("execution_terminal"):
            return False
        live_call = str(progress.get("tool_call_id") or "")
        if progress.get("freshness") == "fresh" and live_call and live_call in {str(call.get("id") or "") for call in in_flight}:
            return False  # FIX1 M2: an in-flight tool under a fresh ExecutionHeartbeat is working, not stalled
        if progress.get("stalled"):
            return True
        age = progress.get("heartbeat_age_seconds")
        return isinstance(age, (int, float)) and age > STALLED_SECONDS and not owned

    def tick(self) -> dict[str, list[str]]:
        """One pass over active runs. Returns run ids per outcome (for logs/tests)."""
        from .ops_metrics import METRICS

        outcome: dict[str, list[str]] = {"exempt": [], "reattached": [], "failed": [], "left_running": []}
        assistant = self.runtime.assistant
        for run in self.runtime.store.list_assistant_runs(limit=500):
            run_id, status = str(run.get("id") or ""), str(run.get("status") or "")
            if status not in ACTIVE_RUN_STATUSES:
                continue
            in_flight = assistant._effective_active_tool_calls(run_id)
            owned = self._locally_owned(run_id, in_flight)
            progress = self._progress(run_id, status)
            if not self._is_stall(progress, in_flight, owned):
                continue
            key = (run_id, float(progress.get("last_progress_at") or progress.get("heartbeat_at") or 0))
            with self._lock:
                first = key not in self._seen
                self._seen.add(key)
            if first:
                METRICS.inc_stalled()
            if job_is_live(run_id):
                outcome["exempt"].append(run_id)
                continue
            if owned:
                outcome["left_running"].append(run_id)
                continue
            metadata = dict((self.runtime.get_session(run_id) or {}).get("metadata") or {})
            attempts = int(metadata.get("stall_reattach_count") or 0)
            if not in_flight and attempts < self.max_reattach:
                metadata["stall_reattach_count"] = attempts + 1
                self.runtime.store.update_session(run_id, metadata_json=metadata)
                self.runtime.store.append_event(run_id, "run_stall_reattach", {"run_id": run_id, "attempt": attempts + 1,
                                                                               "stalled_seconds": STALLED_SECONDS})
                self.runtime.store.update_assistant_run(run_id, status="recovering")
                if assistant.start(run_id, resume=True):
                    outcome["reattached"].append(run_id)
                    continue
            self._fail(run_id, attempts=attempts, in_flight=len(in_flight))
            outcome["failed"].append(run_id)
        return outcome

    def _fail(self, run_id: str, *, attempts: int, in_flight: int) -> None:
        message = "任务长时间没有进展，已停止。可以点「再试一次」重新开始。"
        self.runtime.store.update_assistant_run(run_id, status="failed", error_class="stalled_unrecoverable",
                                                error_message=message)
        try:
            self.runtime.store.update_session(run_id, status="failed")
        except KeyError:
            pass
        self.runtime.store.append_event(run_id, "run_failed", {
            "run_id": run_id, "status": "failed", "error_class": "stalled_unrecoverable", "recoverable": True,
            "reattach_attempts": attempts, "tool_calls_in_flight": in_flight, "message": message,
        })


def enabled(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("EVOMIND_STALL_SUPERVISOR", "1")).strip() != "0"
