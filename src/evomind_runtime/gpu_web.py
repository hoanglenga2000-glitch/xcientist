"""t238 C7c-6: tenant GPU state API (U2/U3). Only the caller's own jobs; no node, job id, host or UUID.

States (``state``): queued / running / succeeded / cancelled / timeout / unavailable / failed.
``label`` is the ready-made Chinese line; numbers are whole minutes, ETA is an estimate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import gpu_admission
from . import gpu_scheduler as gs
from .tenant_access import AccessError

_STATE = {gs.QUEUED: "queued", gs.RUNNING: "running", gs.SUCCEEDED: "succeeded", gs.CANCELLED: "cancelled",
          gs.TIMEOUT: "timeout", gs.LOST: "failed", gs.FAILED: "failed", gs.REJECTED: "unavailable"}


ENV_NOT_READY = gs.ENV_NOT_READY
PREP_COLD_SECONDS = 90  # warm wheel-cache resume finishes in ~12 s; past 90 s means the cold download path (not a pure wall-clock stand-in for "first use")
PREP_SLOW_SECONDS = 600  # a preparation that takes longer than this gets the "slower than usual" sentence
BREAKER_NOTICE = "你的排队位置还在，恢复后在这个任务里重新发起就行。"
PREP_NOTICE = "第一次使用要多花一点时间。你可以先离开这个页面，回来就能看到最新状态。"
PREP_SLOW_NOTICE = "这次比平时慢，我们还在继续。"
CANCELLED_FREE_NOTICE = "没有占用你的训练时长。"
FAILED_FREE_NOTICE = "这次没有占用你的训练时长。"
FAILED_BILLED_NOTICE = "训练中途出错了，这次的用时已计入。"
PREP_CANCEL_NOTICE = "正在停止准备。不会开始训练，也不占用你的训练时长。"
PREP_LEDGER_NOTICE = "准备期间不占用你的训练时长。"
ENV_FAILED_NOTICE = "已经准备好的部分会保留，重试会接着来，不用从头开始。这次没有开始训练，没有占用你的训练时长。"
# A rejected request names why in its title (the generic 「暂不可用」 hid the cause).
_UNAVAILABLE_TITLE = {"quota_exhausted": "今天的训练时长已用完", "quota_exhausted_rolling": "最近的训练时长已用完",
                      "request_too_large": "这次训练太大", "insufficient_gpu_memory": "现在训练排满了",
                      "queue_full": "现在排队的训练太多"}


def _minutes(seconds: float | None, *, up: bool = False) -> int:
    value = max(0.0, float(seconds or 0.0))
    return int(-(-value // 60)) if up else int(value // 60)


def _badge(item: dict, state: str) -> str:
    """The one source of the title chip: it always says what the card says (the FE shows it as is)."""
    if state == "queued":
        return "需要你重新发起" if item.get("retry_label") else "训练暂时用不了" if item.get("paused") else "排队中"
    if state == "running":
        return "正在取消" if item.get("label") == "正在取消…" else "准备中" if item.get("preparing") else "训练中"
    return str(item["label"])


def project_job(sched: gs.GpuScheduler, row: dict, breaker_state: str) -> dict[str, Any]:
    now = sched.clock()
    state = _STATE.get(str(row["status"]), "failed")
    item: dict[str, Any] = {"run_id": row["run_id"], "state": state, "can_cancel": state in {"queued", "running"}}
    reason = str(row.get("reason") or "")
    if state == "queued":
        status = sched.queue_status(row["id"])
        item.update(position=status["position"], eta_minutes=_minutes(status["eta_seconds"], up=True), eta_is_estimate=True)
        if breaker_state == "open":
            item["notice"] = BREAKER_NOTICE
        elif reason in {"insufficient_gpu_memory", "gpu_memory_short"}:
            item["notice"] = gpu_admission.message("gpu_memory_short")
        position = item["position"] or 1
        item["label"] = f"排队中（第 {position} 位，预计等待约 {max(1, item['eta_minutes'])} 分钟）"
        if status["eta_seconds"] is None:  # unknown: never print an invented number
            item.pop("eta_minutes")
            item.pop("eta_is_estimate")
            item["label"] = f"排队中（第 {position} 位）"
        if breaker_state == "open":  # the title only names the state: no position or minutes next to "temporarily unavailable"
            item["label"] = "训练暂时用不了"
            item["paused"] = True  # state stays queued (the place is kept); the title chip must not say 排队中 beside this card
            item.pop("eta_minutes", None)
            item.pop("eta_is_estimate", None)
        if status.get("held"):  # no tool call is waiting: the place is kept, but nothing starts it
            item.update(label="需要你重新发起", notice=gpu_admission.message("queue_held"), retry_label="重新发起")
            item.pop("eta_minutes", None)
            item.pop("eta_is_estimate", None)
        elif reason == gs.FREEING:
            # ADDENDUM 2: never names the paused services.
            item["label"] = f"正在腾出位置，预计 {max(1, item['eta_minutes'])} 分钟内开始。" if "eta_minutes" in item else "正在腾出位置…"
    elif state == "running":
        started = float(row["started_at"] or now)
        max_run = int(row["max_run_seconds"] or sched.policy.max_run_seconds)
        if row.get("cancel_requested_at") is not None:
            item.update(can_cancel=False, label="正在取消…")
            if reason == gs.PREPARING:  # honest about the wait: the in-flight preparation step ends first; nothing is started or billed
                item["notice"] = PREP_CANCEL_NOTICE
        elif reason == gs.PREPARING:
            # C7c-FIX4: the card holds the place but nothing was handed over yet (dependencies are being prepared): not training, not billed.
            waited = now - started
            cold, slow = waited >= PREP_COLD_SECONDS, waited >= PREP_SLOW_SECONDS
            head = PREP_SLOW_NOTICE if slow else (PREP_NOTICE if cold else "")
            # concatenate with no extra space: both sentences already end with 。
            item.update(label="正在准备训练环境", preparing=True,
                        notice=(head + PREP_LEDGER_NOTICE) if head else PREP_LEDGER_NOTICE)
        else:
            item["label"] = f"训练中（已运行 {_minutes(now - float(row.get('dispatched_at') or started))} 分，最长 {_minutes(max_run)} 分）"
        shown = float(row.get("dispatched_at") or started)
        item.update(elapsed_minutes=0 if reason == gs.PREPARING else _minutes(now - shown), max_minutes=_minutes(max_run))
    else:
        code = {"cancelled": "cancelled", "timeout": "timeout", "unavailable": "rejected"}.get(state, "")
        if state == "cancelled" and reason == "broker_window_expired":
            code = "broker_window_expired"
        if reason == "quota_exhausted_rolling":
            code = reason
        if state == "failed":
            code = "lost" if row["status"] == gs.LOST else ""
        item["label"] = {"succeeded": "已完成", "cancelled": "已取消", "timeout": "到时间了，已停止", "unavailable": "暂不可用", "failed": "没有完成"}[state]
        if state == "unavailable" and reason in _UNAVAILABLE_TITLE:
            item["label"] = _UNAVAILABLE_TITLE[reason]
            code = reason if reason in gpu_admission.MESSAGES else code
        if state == "failed" and reason == ENV_NOT_READY:  # failed while still preparing: never started, never billed
            item.update(label="环境没有准备好", notice=ENV_FAILED_NOTICE, retry_label="重新准备")
        elif code:
            item["notice"] = gpu_admission.message(code)
        billed = sched.job_billed_seconds(row["id"])
        if state == "cancelled" and reason != "broker_window_expired":  # the ledger, not the words of the dialog, decides what is promised
            # C7c-FIX5c C5: under a full minute the whole-minute ledger line may not move (ceil absorbs the seconds). Say so; never claim 「没有占用」 when seconds were booked.
            if billed <= 0:
                item["notice"] = CANCELLED_FREE_NOTICE
            elif billed < 60:
                item["notice"] = "已取消。这次用了不到 1 分钟，已计入今天的训练时长。"
            else:
                item["notice"] = f"已取消。这次用了 {_minutes(billed, up=True)} 分钟，已计入今天的训练时长。"
            item["billed_minutes"] = _minutes(billed, up=True)
        elif state == "failed" and not (reason == ENV_NOT_READY) and code == "":
            item["notice"] = FAILED_BILLED_NOTICE if billed > 0 else FAILED_FREE_NOTICE
    item["badge"] = _badge(item, state)
    return item


def _open(runtime_root: Path) -> gs.GpuScheduler | None:
    config = gpu_admission.active()
    if config is None or not (Path(runtime_root) / gpu_admission.DB_NAME).exists():
        return None
    return gpu_admission.scheduler(runtime_root, config=config)


def tenant_status(runtime_root: Path, principal: Any, *, run_id: str = "") -> dict[str, Any]:
    if principal is None:
        raise AccessError(403, "principal_required")
    sched = _open(runtime_root)
    if sched is None:
        return {"schema": "evomind.gpu_tenant.v1", "enabled": False, "jobs": [], "budget": None}
    try:
        breaker = gpu_admission.CircuitBreaker(sched).state()
        rows = sched.jobs_for_tenant(principal.tenant_id, run_id=run_id or None, limit=10)
        usage = sched.usage(principal.tenant_id)
        jobs = [project_job(sched, row, breaker) for row in rows]
        return {"schema": "evomind.gpu_tenant.v1", "enabled": True, "available": breaker != "open", "jobs": jobs,
                "budget": {"remaining_minutes": usage["remaining_minutes"], "daily_limit_minutes": usage["daily_limit_minutes"],
                           "daily_used_minutes": usage["daily_used_minutes"],
                           "max_run_minutes": _minutes(sched.policy.max_run_seconds)}}
    finally:
        sched.close()


def tenant_cancel(runtime_root: Path, principal: Any, run_id: str) -> dict[str, Any]:
    """Cancel this tenant's active job for one of its runs. Idempotent; others' runs → 404."""

    if principal is None:
        raise AccessError(403, "principal_required")
    sched = _open(runtime_root)
    if sched is None:
        raise AccessError(404, "not_found")
    try:
        rows = sched.jobs_for_tenant(principal.tenant_id, run_id=run_id, limit=1)
        if not rows:
            raise AccessError(404, "not_found")
        row = rows[0]
        if row["status"] in gs.ACTIVE:
            row = sched.cancel(row["id"], principal.tenant_id)
        return {"schema": "evomind.gpu_tenant.v1", "job": project_job(sched, row, gpu_admission.CircuitBreaker(sched).state())}
    finally:
        sched.close()


def broker_status(runtime_root: Path, broker: Any = None) -> dict[str, Any]:
    """Operator-only broker section (service names, memory, window). Never served to tenants."""

    from . import gpu_vram

    try:
        vram = gpu_vram.load(runtime_root)
    except gpu_vram.VramConfigError as error:
        return {"enabled": False, "config_error": str(error)}
    if broker is None:
        broker = gpu_admission._startup_broker(runtime_root) if vram.gpu_broker_enabled else None
    if broker is None:
        return {"enabled": False, "configured": bool(vram.gpu_broker_config_path)}
    try:
        return {"enabled": True, **broker.status()}
    except Exception as error:  # noqa: BLE001
        return {"enabled": True, "status_error": type(error).__name__}


def operator_pool(runtime_root: Path, principal: Any) -> dict[str, Any]:
    if principal is not None:
        raise AccessError(403, "operator_required")
    sched = gpu_admission.scheduler(runtime_root)
    try:
        snapshot = sched.snapshot()
        snapshot["breaker"] = gpu_admission.CircuitBreaker(sched).state()
        snapshot["active"] = gpu_admission.active() is not None
        snapshot["gpu_broker"] = broker_status(runtime_root)  # ADDENDUM 2: operator-only
        return snapshot
    finally:
        sched.close()
