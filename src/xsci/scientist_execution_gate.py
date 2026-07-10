"""Shared Scientist execution gate for CLI run entry points.

The gate is deliberately small: it builds the same execution contract used by
the EvoMind terminal, renders a compact human summary, and tells direct training
entry points when to stop.  It never starts training and never submits to
Kaggle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import Config, find_project_dir, load_config
from .scientist_gate_decision import build_execution_gate_decision
from .kaggle_session import SessionState
from .terminal_tools import TerminalTools


def _project_root(root: Optional[Path] = None) -> Path:
    return Path(root or find_project_dir() or Path.cwd())


def build_execution_contract_for_task(
    task: str,
    *,
    root: Optional[Path] = None,
    cfg: Optional[Config] = None,
    compute: Optional[str] = None,
    goal: str = "",
) -> dict[str, Any]:
    """Build and persist a Scientist execution contract for a task.

    ``task`` may be a registered slug or a direct task JSON path.  The selected
    task is assigned after SessionState restore so the caller's explicit task
    wins over any previous console session.
    """
    project_root = _project_root(root)
    state = SessionState.from_root(project_root, cfg=cfg or load_config(project_root))
    state.selected_task = task
    state.last_goal = goal or state.last_goal
    if compute:
        state.current_compute_override = compute
    state.refresh_task_brief(project_root)
    contract = TerminalTools.dispatch("scientist_execution_contract", state, project_root)
    contract["execution_gate_decision"] = build_execution_gate_decision(contract)
    artifact = str(contract.get("artifact_path") or "")
    if artifact:
        try:
            Path(artifact).write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
    return contract


def contract_blocks_training(contract: dict[str, Any], *, require_model_ready: bool = True) -> bool:
    """Return True when a direct training loop must not start."""
    return bool(
        build_execution_gate_decision(
            contract,
            require_model_ready=require_model_ready,
        )["blocked"]
    )


def render_execution_contract_lines(contract: dict[str, Any]) -> list[str]:
    """Compact CLI rendering with no secrets and no leaderboard overclaim."""
    gate_decision = build_execution_gate_decision(contract)
    lines = [
        "Scientist execution contract:",
        f"  go/no-go        : {contract.get('go_no_go', 'unknown')}",
        f"  agent session   : {'ready' if contract.get('agent_session_ready') else 'blocked'}",
        f"  model training  : {'ready' if contract.get('model_training_ready') else 'blocked'}",
        f"  data contract   : {contract.get('data_contract_status', 'unknown')}",
        f"  gate decision   : {gate_decision['status']}",
    ]
    roots = [str(item) for item in (contract.get("root_causes") or []) if str(item)]
    if roots:
        lines.append("  root causes     : " + ", ".join(roots[:6]))
    blocked_by = gate_decision.get("blocked_by") or []
    if blocked_by:
        lines.append("  blocked by      : " + ", ".join(str(item) for item in blocked_by[:6]))
    safe_next = gate_decision.get("safe_next_commands") or []
    if safe_next:
        lines.append("  safe next       : " + " | ".join(str(item) for item in safe_next[:3]))
    artifact = str(contract.get("artifact_path") or "")
    if artifact:
        lines.append("  artifact        : " + artifact)
    lines.append("  official submit : blocked_until_explicit_human_approval")
    return lines
