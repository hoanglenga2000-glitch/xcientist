"""Shared train/no-train decision helpers for EvoMind Scientist gates."""
from __future__ import annotations

import re
from typing import Any


SENSITIVE_GATE_RE = re.compile(
    r"(?i)(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]\s*\S+"
)


def safe_gate_text(value: Any, *, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    text = SENSITIVE_GATE_RE.sub(r"\1=[redacted]", text)
    return text[:limit]


def build_execution_gate_decision(
    contract: dict[str, Any],
    *,
    require_model_ready: bool = True,
) -> dict[str, Any]:
    """Normalize a Scientist contract into a reusable train/no-train decision."""
    root_causes = [
        safe_gate_text(item, limit=120)
        for item in (contract.get("root_causes") or [])
        if safe_gate_text(item).strip()
    ]
    setup_blockers = [
        safe_gate_text(item)
        for item in (contract.get("setup_blockers") or [])
        if safe_gate_text(item).strip()
    ]
    blocked_by: list[str] = []
    if not contract.get("ok", True):
        blocked_by.append("contract_error")
    if str(contract.get("go_no_go") or "").lower() == "no_go":
        blocked_by.append("execution_contract_no_go")
    if require_model_ready and not bool(contract.get("model_training_ready")):
        blocked_by.append("model_training_not_ready")
    if require_model_ready and not bool(contract.get("agent_session_ready")):
        blocked_by.append("agent_session_not_ready")
    if require_model_ready and str(contract.get("data_contract_status") or "").lower() != "ready":
        blocked_by.append("data_contract_not_ready")

    blocked = bool(blocked_by)
    safe_next_commands = (
        ["evomind repair", "evomind workplan", "evomind ready"]
        if blocked
        else [str(contract.get("execution_command") or "").strip() or "evomind run <task>"]
    )
    message = (
        "Training is blocked by the Scientist execution gate; clear the listed setup/data/resource blockers first."
        if blocked
        else "Training may proceed only through the audited AgentSession/workstation path."
    )
    return {
        "ok": True,
        "blocked": blocked,
        "status": "blocked" if blocked else "ready_for_gated_training",
        "require_model_ready": require_model_ready,
        "blocked_by": blocked_by,
        "root_causes": root_causes,
        "setup_blockers": setup_blockers,
        "safe_next_commands": safe_next_commands,
        "message": message,
        "artifact_path": str(contract.get("artifact_path") or ""),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
