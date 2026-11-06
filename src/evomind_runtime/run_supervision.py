"""Run supervision v1: the dsh-superagent execution model ported to EvoMind runs.

One module owns the run lifecycle record and the rules around it:

- ``RunLedger`` writes every execution attempt (dsh: Assignment) to ``execution_attempts_v2``,
  every classified failure to ``failure_observations_v2`` and every delivery to
  ``operation_receipts_v2``. Reports use the dsh vocabulary:
  progress / waiting / ready / blocked / failed / dropped.
- ``derive_stage`` turns a run into one stage, first match wins (dsh 003 "Ticket stages"):
  attention (待你处理) > blocked (受阻/失败) > queued (排队) > waiting_gpu (等待训练) > working (执行中).
- ``model_retry_decision`` retries passing model failures (unavailable / timeout / rate limit /
  empty answer) three times on the same model, 2 s, 8 s, 30 s apart (Retry-After wins, capped at 60 s).
  Auth, request and identity failures never retry. The model is never switched here.
- ``RunSlots`` bounds concurrent model work: a global limit, a per-tenant limit and one reserved
  slot that only a person's resume or an approved action may take. GPU tools give their slot back
  while the scheduler owns the card.
- ``tool_breaker_verdict`` stops loops: an identical call that already failed three times is refused,
  and a tool that failed ten times with the same error in one run is closed until a person resumes.
- ``compact_events`` keeps the last 50 high-volume events of runs that ended 7+ days ago.
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

REPORTS = ("progress", "waiting", "ready", "blocked", "failed", "dropped")
LIVE_STATUSES = frozenset({"queued", "planning", "running", "verifying", "recovering", "pausing"})
TERMINAL_STATUSES = frozenset({"completed", "cancelled", "delivered"})
STAGE_LABELS = {
    "attention": "待你处理",
    "blocked": "受阻",
    "failed": "失败",
    "queued": "排队中",
    "waiting_gpu": "等待训练",
    "working": "执行中",
    "paused": "已暂停",
    "done": "已结束",
}

# ---------------------------------------------------------------- model retry

RETRY_DELAYS_SECONDS = (2.0, 8.0, 30.0)
RETRY_AFTER_CAP_SECONDS = 60.0
MODEL_RETRY_LIMIT = len(RETRY_DELAYS_SECONDS)
RETRYABLE_MODEL_CATEGORIES = frozenset({"unavailable", "timeout", "rate_limited"})
EMPTY_ANSWER_CODES = frozenset({"empty_model_response", "model_empty_response"})
_LLM_CODE = re.compile(r"LLMError/([A-Za-z0-9_]{1,80})")


class ModelTurnFailed(RuntimeError):
    """A model turn that ended without an answer; ``str()`` keeps the runtime's public text."""

    def __init__(self, text: str, *, code: str = "", retry_after_seconds: float | None = None) -> None:
        super().__init__(text)
        self.code = code
        self.retry_after_seconds = retry_after_seconds


def model_failure(text: str, code: str = "") -> tuple[str, str] | None:
    """``(public code, category)`` for a model-unavailable text, or None when it is not a model failure."""
    from .llm_backends import category_for_code

    match = _LLM_CODE.search(str(text or ""))
    found = code or (match.group(1) if match else "")
    if not found:
        return None
    if found in EMPTY_ANSWER_CODES:
        return found, "unavailable"
    return found, category_for_code(found)


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float
    code: str
    category: str


def model_retry_decision(text: str, retries_done: int, *, code: str = "",
                         retry_after_seconds: float | None = None) -> RetryDecision | None:
    """None when ``text`` is not a model failure; else whether to retry the same model and how long to wait."""
    found = model_failure(text, code)
    if found is None:
        return None
    public, category = found
    if category not in RETRYABLE_MODEL_CATEGORIES or retries_done >= MODEL_RETRY_LIMIT:
        return RetryDecision(False, 0.0, public, category)
    delay = RETRY_DELAYS_SECONDS[retries_done]
    if isinstance(retry_after_seconds, (int, float)) and retry_after_seconds > 0:
        delay = float(retry_after_seconds)
    return RetryDecision(True, min(delay, RETRY_AFTER_CAP_SECONDS), public, category)


# ---------------------------------------------------------------- attention cards

MODEL_GROUP_LABELS = {"anthropic": "Claude", "openai": "GPT", "deepseek": "国模"}


def group_default_model(failed_platform_id: str, provider: str = "") -> str:
    """The listed default of the failed model's group (first listed id of that provider), or ''.

    When the failed model already is its group's default, '' (the card then offers retry only and the
    UI keeps its own alternate list). Never a hidden id: only ``available_platform_models`` is read.
    """
    try:
        from .model_router import available_platform_models
        listed = available_platform_models()
    except Exception:  # noqa: BLE001 - a card without a switch target is still a valid card
        return ""
    group = str(provider or (failed_platform_id.split(":", 1)[0] if ":" in failed_platform_id else ""))
    for item in listed:
        if str(item.get("provider") or "") == group:
            candidate = str(item.get("id") or "")
            return "" if candidate == failed_platform_id else candidate
    return ""


def model_unavailable_card(*, code: str, category: str, platform_model_id: str, provider: str,
                           retries: int) -> dict[str, Any]:
    target = group_default_model(platform_model_id, provider)
    options: list[dict[str, Any]] = [{"action": "resume", "label": "重试原模型"}]
    if target:
        options.append({"action": "new_run_with_model", "platform_model_id": target, "label": "换成同组默认模型"})
    return {
        "kind": "model_unavailable",
        "reason": "这个模型多次没有给出回复，对话和已完成的步骤都已保留。",
        "code": code,
        "category": category,
        "retries": retries,
        "platform_model_id": platform_model_id or None,
        "options": options,
        "created_at": _utc_now(),
    }


def approval_card(kind: str) -> dict[str, Any]:
    reason = "确认已过期，任务停在这一步。" if kind == "approval_expired" else "你拒绝了这一步，任务停在这里。"
    return {"kind": kind, "reason": reason, "options": [{"action": "resume", "label": "重新申请确认"}], "created_at": _utc_now()}


def tool_card(tool: str, code: str, count: int) -> dict[str, Any]:
    return {
        "kind": "tool_failing",
        "reason": "同一个步骤反复出现同样的错误，已暂停这个工具，避免空耗。",
        "tool": tool,
        "code": code,
        "count": count,
        "options": [{"action": "resume", "label": "调整后继续"}],
        "created_at": _utc_now(),
    }


def public_attention(status: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
    """The card a person must act on, or None. Tool cards only show once the run is no longer live."""
    card = metadata.get("attention")
    if isinstance(card, dict) and card.get("kind") and status not in TERMINAL_STATUSES:
        if status in LIVE_STATUSES and card.get("kind") != "tool_failing":
            return None  # a live run is already acting on it (e.g. resumed): stale card
        return _strip_card(card)
    circuit = metadata.get("tool_circuit")
    if isinstance(circuit, dict) and circuit and status not in LIVE_STATUSES:
        tool, info = sorted(circuit.items(), key=lambda item: str((item[1] or {}).get("opened_at") or ""))[-1]
        info = info if isinstance(info, dict) else {}
        return _strip_card(tool_card(str(tool), str(info.get("code") or ""), int(info.get("count") or 0)))
    return None


def _strip_card(card: dict[str, Any]) -> dict[str, Any]:
    keep = ("kind", "reason", "code", "category", "retries", "platform_model_id", "tool", "count", "options", "created_at")
    return {key: card[key] for key in keep if key in card}


# ---------------------------------------------------------------- stage


def derive_stage(status: str, *, attention: dict[str, Any] | None = None, queue_position: int | None = None,
                 slot_state: str = "", gpu_tool_running: bool = False) -> str:
    status = str(status or "")
    if status in TERMINAL_STATUSES:
        return "done"
    if status == "waiting_approval" or (attention and status not in LIVE_STATUSES):
        return "attention"
    if status in {"blocked", "failed"}:
        return status
    if status in {"paused"}:
        return "paused"
    if queue_position or slot_state == "queued" or status == "queued":
        return "queued"
    if slot_state == "gpu" or gpu_tool_running:
        return "waiting_gpu"
    return "working"


# ---------------------------------------------------------------- ledger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLedger:
    """The only writer of run attempts, failures and receipts (shared SuperAgentStore connection)."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self._lock = threading.Lock()

    def begin_attempt(self, run_id: str, *, reason: str, model: str = "") -> str:
        with self._lock:
            previous = self.store.latest_attempt(run_id)
            if previous and previous.get("status") == "running":
                # dsh 003 "Restart": an Assignment still open when a new one starts ended without a report.
                previous.update({"status": "ended", "report": "failed", "interrupted": True, "reported_at": _utc_now()})
                self.store.put_attempt(previous)
            number = self.store.count_attempts(run_id) + 1
            attempt_id = f"{run_id}:a{number}"
            self.store.put_attempt({
                "attempt_id": attempt_id, "run_id": run_id, "node_id": "run",
                "idempotency_key": f"attempt-{number}", "status": "running", "report": "progress",
                "reason": str(reason)[:60], "model": str(model)[:120], "number": number, "started_at": _utc_now(),
            })
        return attempt_id

    def report(self, attempt_id: str, report: str, **detail: Any) -> None:
        if report not in REPORTS:
            raise ValueError(f"unknown report {report}")
        run_id = attempt_id.rsplit(":a", 1)[0]
        existing = self.store.get_attempt(attempt_id) or {
            "attempt_id": attempt_id, "run_id": run_id, "node_id": "run", "idempotency_key": attempt_id.rsplit(":", 1)[-1]}
        existing.update({key: value for key, value in detail.items() if value is not None})
        existing.update({"status": "ended" if report != "progress" else "running", "report": report, "reported_at": _utc_now()})
        self.store.put_attempt(existing)

    def failure(self, run_id: str, *, signature: str, category: str, node_id: str = "run", **detail: Any) -> None:
        self.store.append_failure({
            "failure_id": "failure_" + hashlib.sha256(f"{run_id}|{signature}|{time.time_ns()}".encode()).hexdigest()[:32],
            "run_id": run_id, "node_id": node_id, "signature": str(signature)[:200], "category": str(category)[:60],
            **{key: value for key, value in detail.items() if value is not None},
        })

    def receipt(self, run_id: str, operation: str, ok: bool, **detail: Any) -> None:
        self.store.append_receipt({
            "receipt_id": "receipt_" + hashlib.sha256(f"{run_id}|{operation}|{time.time_ns()}".encode()).hexdigest()[:32],
            "run_id": run_id, "node_id": "run", "operation": operation, "ok": bool(ok),
            **{key: value for key, value in detail.items() if value is not None},
        })

    def latest(self, run_id: str) -> dict[str, Any] | None:
        return self.store.latest_attempt(run_id)


# ---------------------------------------------------------------- slots

PRIORITY_RANK = {"human": 0, "continue": 1, "normal": 2}


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(os.environ.get(name, default)), high))
    except (TypeError, ValueError):
        return default


@dataclass
class _Waiter:
    seq: int
    run_id: str
    tenant: str
    priority: str


@dataclass
class _Held:
    tenant: str
    kind: str  # normal | reserve


class RunSlots:
    """Concurrency slots for model work (dsh 003 "Concurrency"), in-process.

    ``total`` normal slots, at most ``per_tenant`` of them per tenant (an empty tenant, the
    operator, has no tenant cap), plus ``reserve`` slots only a person's resume or an approved
    action may take. Waiters are served in priority then arrival order; a waiter that cannot run
    (its tenant is at the cap) does not hold up the ones behind it.
    """

    def __init__(self, total: int | None = None, per_tenant: int | None = None, reserve: int | None = None) -> None:
        self.total = total if total is not None else _int_env("EVOMIND_RUN_SLOTS", 6, 1, 64)
        self.per_tenant = per_tenant if per_tenant is not None else _int_env("EVOMIND_RUN_SLOTS_PER_TENANT", 2, 1, 64)
        self.reserve = reserve if reserve is not None else _int_env("EVOMIND_RUN_SLOTS_RESERVE", 1, 0, 8)
        self._cond = threading.Condition()
        self._held: dict[str, _Held] = {}
        self._waiting: list[_Waiter] = []
        self._gpu: dict[str, str] = {}
        self._seq = 0

    # -- grant rule
    def _kind_for(self, waiter: _Waiter, held: dict[str, _Held]) -> str | None:
        normal = sum(1 for item in held.values() if item.kind == "normal")
        tenant_count = sum(1 for item in held.values() if waiter.tenant and item.tenant == waiter.tenant)
        if normal < self.total and (not waiter.tenant or tenant_count < self.per_tenant):
            return "normal"
        if waiter.priority == "human" and sum(1 for item in held.values() if item.kind == "reserve") < self.reserve:
            return "reserve"
        return None

    def _granted_kind(self, me: _Waiter) -> str | None:
        simulated = dict(self._held)
        for waiter in sorted(self._waiting, key=lambda item: (PRIORITY_RANK.get(item.priority, 2), item.seq)):
            kind = self._kind_for(waiter, simulated)
            if kind is None:
                continue
            if waiter is me:
                return kind
            simulated[waiter.run_id] = _Held(waiter.tenant, kind)
        return None

    def acquire(self, run_id: str, tenant: str = "", *, priority: str = "normal",
                should_abort: Callable[[], bool] = lambda: False,
                on_queued: Callable[[int], None] | None = None, poll_seconds: float = 1.0) -> bool:
        """Block until ``run_id`` holds a slot. False when ``should_abort`` turned true while waiting."""
        with self._cond:
            if run_id in self._held:
                return True
            self._seq += 1
            me = _Waiter(self._seq, run_id, str(tenant or ""), priority if priority in PRIORITY_RANK else "normal")
            self._waiting.append(me)
            announced = None
            try:
                while True:
                    kind = self._granted_kind(me)
                    if kind is not None:
                        self._held[run_id] = _Held(me.tenant, kind)
                        self._gpu.pop(run_id, None)
                        return True
                    position = self._position_locked(run_id)
                    if on_queued is not None and position != announced:
                        announced = position
                        try:
                            on_queued(position or 1)
                        except Exception:  # noqa: BLE001 - a status write never blocks the queue
                            pass
                    self._cond.wait(poll_seconds)
                    if should_abort():
                        return False
            finally:
                if me in self._waiting:
                    self._waiting.remove(me)
                self._cond.notify_all()

    def release(self, run_id: str) -> None:
        with self._cond:
            self._held.pop(run_id, None)
            self._gpu.pop(run_id, None)
            self._cond.notify_all()

    @contextmanager
    def released_for_gpu(self, run_id: str, *, should_abort: Callable[[], bool] = lambda: False) -> Iterator[None]:
        """Give the slot back while the GPU scheduler owns the work; take one again (ahead of new runs) after."""
        with self._cond:
            held = self._held.pop(run_id, None)
            if held is not None:
                self._gpu[run_id] = held.tenant
                self._cond.notify_all()
        try:
            yield
        finally:
            if held is not None:
                self.acquire(run_id, held.tenant, priority="continue", should_abort=should_abort)

    def _position_locked(self, run_id: str) -> int | None:
        ordered = sorted(self._waiting, key=lambda item: (PRIORITY_RANK.get(item.priority, 2), item.seq))
        for index, waiter in enumerate(ordered, start=1):
            if waiter.run_id == run_id:
                return index
        return None

    def position(self, run_id: str) -> int | None:
        with self._cond:
            return self._position_locked(run_id)

    def state(self, run_id: str) -> str:
        with self._cond:
            if run_id in self._held:
                return "held"
            if run_id in self._gpu:
                return "gpu"
            if any(item.run_id == run_id for item in self._waiting):
                return "queued"
            return ""

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "total": self.total, "per_tenant": self.per_tenant, "reserve": self.reserve,
                "held": len(self._held), "held_reserve": sum(1 for item in self._held.values() if item.kind == "reserve"),
                "queued": len(self._waiting), "gpu_released": len(self._gpu),
            }


GPU_SLOT_RELEASE_TOOLS = frozenset({"managed_tensor_train", "hpc_execute_solution", "hpc_calibration_run"})


# ---------------------------------------------------------------- tool breaker

IDENTICAL_FAILURE_LIMIT = 3
TOOL_FAILURE_LIMIT = 10


def error_code(result: Any) -> str:
    """A stable error signature: the part before the first ':' (no paths, no messages)."""
    raw = str((result or {}).get("error") or "") if isinstance(result, dict) else ""
    return raw.split(":", 1)[0].strip()[:80]


@dataclass(frozen=True)
class BreakerVerdict:
    refuse: bool
    reason: str = ""          # identical_call | tool_open
    code: str = ""
    count: int = 0


def tool_breaker_verdict(calls: Iterable[dict[str, Any]], tool_name: str, arguments: Any, *,
                         since: str = "", open_tools: Iterable[str] = ()) -> BreakerVerdict:
    """Whether a new call of ``tool_name(arguments)`` must be refused before it runs.

    Counts only this run's failed calls after ``since`` (a person's resume resets the count).
    Our own refusals carry their own codes and never feed the count.
    """
    if tool_name in set(open_tools):
        return BreakerVerdict(True, "tool_open")
    identical = 0
    per_code: dict[str, int] = {}
    for call in calls:
        if str(call.get("tool_name") or "") != tool_name or str(call.get("status") or "") != "failed":
            continue
        if since and str(call.get("created_at") or "") <= since:
            continue
        code = error_code(call.get("result"))
        if not code or code in {"repeated_identical_failure", "tool_circuit_open"}:
            continue
        per_code[code] = per_code.get(code, 0) + 1
        if call.get("arguments") == arguments:
            identical += 1
    if identical >= IDENTICAL_FAILURE_LIMIT:
        return BreakerVerdict(True, "identical_call", count=identical)
    for code, count in per_code.items():
        if count >= TOOL_FAILURE_LIMIT:
            return BreakerVerdict(True, "tool_open", code=code, count=count)
    return BreakerVerdict(False)


# ---------------------------------------------------------------- stall rule

NO_PROGRESS_STALL_SECONDS = 30 * 60
ABANDONED_SECONDS = 24 * 3600


def parse_utc(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def no_progress_stall(run: dict[str, Any], now: float) -> str:
    """'' | 'stalled' | 'abandoned' for an active run that has no progress record at all."""
    updated = parse_utc(run.get("updated_at"))
    if updated is None:
        return ""
    age = now - updated
    if age > ABANDONED_SECONDS:
        return "abandoned"
    if age > NO_PROGRESS_STALL_SECONDS:
        return "stalled"
    return ""


# ---------------------------------------------------------------- recovery note

RECOVERY_NOTES = {
    "host_restarted": "The service restarted while this run was active.",
    "stall_reattach": "This run stopped making progress and was picked up again.",
    "model_retry": "The previous model request failed without an answer and is being retried on the same model.",
}


def recovery_note(reason: str) -> str:
    lead = RECOVERY_NOTES.get(reason)
    if not lead:
        return ""
    return ("Recovery note: " + lead + " Before repeating any action with side effects (training, uploads, "
            "downloads, submissions, file publishing), check outputs/, the published artifacts and the settled "
            "tool receipts above; reuse what already exists and repeat an action only when its receipt shows it did not happen.")


# ---------------------------------------------------------------- event compaction

COMPACT_EVENT_TYPES = ("run_progress", "assistant.delta")


def compact_events(db_path: str | Path, *, older_than_days: int = 7, keep: int = 50,
                   event_types: Iterable[str] = COMPACT_EVENT_TYPES, batch_sessions: int = 200,
                   checkpoint: bool = True) -> dict[str, int]:
    """Keep the last ``keep`` events of each high-volume type for runs that ended ``older_than_days`` ago."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    types = tuple(event_types)
    deleted = sessions = 0
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        rows = connection.execute(
            "SELECT id FROM assistant_runs WHERE status IN ('completed','cancelled','failed','blocked') AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [row[0] for row in rows]
        for start in range(0, len(ids), batch_sessions):
            chunk = ids[start:start + batch_sessions]
            with connection:
                for session_id in chunk:
                    for event_type in types:
                        cursor = connection.execute(
                            "DELETE FROM events WHERE session_id=? AND event_type=? AND seq NOT IN "
                            "(SELECT seq FROM events WHERE session_id=? AND event_type=? ORDER BY seq DESC LIMIT ?)",
                            (session_id, event_type, session_id, event_type, int(keep)),
                        )
                        deleted += cursor.rowcount or 0
            sessions += len(chunk)
        if checkpoint:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    return {"sessions": sessions, "deleted": deleted}


def compaction_enabled(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("EVOMIND_EVENT_COMPACTION", "1")).strip() != "0"


def shadow_plans_enabled(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("EVOMIND_SUPER_AGENT_SHADOW_PLANS", "0")).strip() == "1"


# ---------------------------------------------------------------- operator view


def supervision_view(runtime: Any, *, days: int = 14, limit: int = 100) -> dict[str, Any]:
    """Operator list of what waits for a person (dsh panel: a healthy state takes no space)."""
    assistant = runtime.assistant
    store = runtime.store
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = store.list_assistant_runs_by_status(
        ("waiting_approval", "blocked", "failed", "queued", "planning", "running", "verifying", "recovering"),
        since=since, limit=500)
    items: list[dict[str, Any]] = []
    now = time.time()
    for run in rows:
        run_id = str(run.get("id") or "")
        session = store.get_session(run_id) or {}
        metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        status = str(run.get("status") or "")
        attention = public_attention(status, metadata)
        slot_state = assistant.slots.state(run_id)
        stage = derive_stage(status, attention=attention, queue_position=assistant.slots.position(run_id), slot_state=slot_state)
        updated = parse_utc(run.get("updated_at"))
        stale = status in LIVE_STATUSES and slot_state == "" and updated is not None and now - updated > NO_PROGRESS_STALL_SECONDS
        if stage not in {"attention", "blocked", "failed"} and not stale:
            continue
        latest = assistant.ledger.latest(run_id) or {}
        items.append({
            "run_id": run_id, "status": status, "stage": "stale" if stale and stage == "working" else stage,
            "error_class": str(run.get("error_class") or ""), "model": str(run.get("model") or ""),
            "tenant": str(((metadata.get("http_access_scope") or {}) if isinstance(metadata.get("http_access_scope"), dict) else {}).get("tenant_id") or ""),
            "attention_kind": (attention or {}).get("kind"), "updated_at": run.get("updated_at"),
            "last_report": latest.get("report"), "attempts": latest.get("number"),
        })
        if len(items) >= limit:
            break
    return {"schema": "evomind.run_supervision.v1", "slots": assistant.slots.snapshot(),
            "attention": items, "failure_categories": runtime.super_agent.store.failure_summary(since=since)}


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


__all__ = [
    "BreakerVerdict", "GPU_SLOT_RELEASE_TOOLS", "ModelTurnFailed", "RetryDecision", "RunLedger", "RunSlots",
    "STAGE_LABELS", "approval_card", "compact_events", "compaction_enabled", "derive_stage", "model_failure",
    "model_retry_decision", "model_unavailable_card", "no_progress_stall", "public_attention", "recovery_note",
    "shadow_plans_enabled", "supervision_view", "tool_breaker_verdict", "tool_card",
]
