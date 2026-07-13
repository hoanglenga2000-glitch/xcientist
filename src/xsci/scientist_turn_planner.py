"""Per-turn planning for the EvoMind AI Scientist terminal.

The terminal feels intelligent only when each natural-language turn has an
observable control decision: what the user asked for, which tools should be
used first, which gates must stop execution, and which artifacts should prove
progress.  This module builds that durable plan without starting training or
submitting to Kaggle.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .kaggle_intent import EXECUTION, PLANNING, TOOL_QUERY, classify
from .kaggle_session import SessionState

SENSITIVE_RE = re.compile(
    r"(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _safe_text(value: Any, *, limit: int = 1200) -> str:
    text = "" if value is None else str(value)
    text = SENSITIVE_RE.sub(r"\1=[redacted]", text)
    return text[:limit]


def _safe_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[truncated]"
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:80]:
            key_text = str(key)
            if re.search(r"(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)", key_text, re.I):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = _safe_json(item, depth=depth + 1)
        return result
    return _safe_text(value)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_presence(root: Path) -> dict[str, dict[str, Any]]:
    xsci = root / ".xsci"
    names = {
        "context_packet": "scientist_context_packet.json",
        "situation_model": "scientist_situation_model.json",
        "autopilot": "scientist_autopilot.json",
        "action_queue": "scientist_action_queue.json",
        "workplan": "scientist_workplan.json",
        "execution_contract": "scientist_execution_contract.json",
        "loop": "scientist_loop.json",
        "recovery": "scientist_recovery_snapshot.json",
    }
    result: dict[str, dict[str, Any]] = {}
    for key, name in names.items():
        path = xsci / name
        payload = _read_json(path)
        result[key] = {
            "path": str(path),
            "present": isinstance(payload, dict),
            "tool": payload.get("tool") if isinstance(payload, dict) else "",
            "mode": payload.get("mode") if isinstance(payload, dict) else "",
            "status": (
                payload.get("status")
                or payload.get("situation_status")
                or payload.get("go_no_go")
                or ""
            ) if isinstance(payload, dict) else "",
        }
    return result


def _tool_item(tool: str, why: str, *, confidence: float,
               expected_artifacts: list[str] | None = None,
               gate: str = "read_only") -> dict[str, Any]:
    return {
        "tool": tool,
        "why": why,
        "confidence": round(float(confidence), 2),
        "gate": gate,
        "expected_artifacts": expected_artifacts or [],
    }


def _dedup_tools(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        tool = str(item.get("tool") or "")
        if not tool or tool in seen:
            continue
        seen.add(tool)
        output.append(item)
    return output


def _has_any_text(text: str, needles: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(needle in low for needle in needles)


def _is_meta_scientist_goal(text: str, payload: str = "") -> bool:
    """Detect requests about improving EvoMind's own Scientist ability."""
    if payload in {
        "scientist_autopilot",
        "scientist_self_audit",
        "scientist_loop",
        "scientist_memory_consolidation",
    }:
        return True
    meta_needles = (
        "不够智能",
        "像ai scientist",
        "像 ai scientist",
        "真正的ai scientist",
        "真正 ai scientist",
        "复杂问题",
        "超级终端",
        "像claude code",
        "像 claude code",
        "像codex",
        "像 codex",
        "self evolution",
        "self-evolution",
        "self audit",
        "scientist ability",
        "agent capability",
        "claude code",
        "codex",
    )
    return _has_any_text(text, meta_needles)


def _artifact_present(artifact_state: dict[str, dict[str, Any]], key: str) -> bool:
    item = artifact_state.get(key) if isinstance(artifact_state, dict) else None
    return bool(isinstance(item, dict) and item.get("present"))


def _build_scientific_critique(
    *,
    prompt: str,
    intent_kind: str,
    payload: str,
    selected_task: bool,
    can_execute: bool,
    blocking_gates: list[str],
    advisory_gaps: list[str],
    artifact_state: dict[str, dict[str, Any]],
    selected_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the per-turn Scientist critique that drives tool choice.

    This is the small "scientist brain" in front of the tool loop: it says what
    evidence is missing, why the current answer is uncertain, which claims are
    forbidden, and whether the next action is actionable or must remain
    observational.
    """
    evidence_gaps: list[dict[str, Any]] = []
    uncertainty_drivers: list[str] = []
    recommended_tool_inserts: list[str] = []

    def add_gap(severity: str, gap: str, why: str, tool: str) -> None:
        evidence_gaps.append({
            "severity": severity,
            "gap": gap,
            "why_it_matters": why,
            "suggested_tool": tool,
        })
        if tool and tool not in recommended_tool_inserts:
            recommended_tool_inserts.append(tool)

    if not selected_task:
        add_gap(
            "blocking",
            "no_selected_task",
            "Task-specific modeling and benchmark reasoning require a selected Kaggle competition.",
            "task_list",
        )
    if blocking_gates:
        add_gap(
            "blocking",
            "blocking_setup_gates",
            "Execution requests cannot be trusted until hard setup gates are repaired.",
            "scientist_repair_plan",
        )
        uncertainty_drivers.append("Hard readiness gates are still blocking execution.")
    if advisory_gaps:
        add_gap(
            "advisory",
            "advisory_setup_gaps",
            "Non-blocking setup gaps can still reduce research quality or prevent official evidence.",
            "system_status",
        )
    if intent_kind == EXECUTION and not _artifact_present(artifact_state, "execution_contract"):
        add_gap(
            "blocking",
            "missing_execution_contract",
            "Training-like requests need a go/no-go contract before any candidate run.",
            "scientist_execution_contract",
        )
    if intent_kind in {EXECUTION, PLANNING} and not _artifact_present(artifact_state, "situation_model"):
        add_gap(
            "medium",
            "missing_situation_model",
            "A complex research turn needs an Observe-Orient-Decide-Act state snapshot.",
            "scientist_situation_model",
        )
    if intent_kind in {EXECUTION, PLANNING} and not _artifact_present(artifact_state, "workplan"):
        add_gap(
            "medium",
            "missing_workplan",
            "Multi-step research should be recoverable as a workplan with gates and artifacts.",
            "scientist_workplan",
        )
    if _is_meta_scientist_goal(prompt, payload):
        add_gap(
            "high",
            "agent_capability_audit_needed",
            "Requests about being smarter like Claude Code/Codex need self-audit before execution advice.",
            "scientist_self_audit",
        )
        add_gap(
            "high",
            "memory_consolidation_needed",
            "A self-evolving agent must convert recent traces and failures into reusable memory.",
            "scientist_memory_consolidation",
        )
        for tool in ("scientist_autopilot", "scientist_workplan"):
            if tool not in recommended_tool_inserts:
                recommended_tool_inserts.append(tool)
        uncertainty_drivers.append("The user is asking about EvoMind's own capability, so tool strategy must inspect the agent state.")
    if not _artifact_present(artifact_state, "action_queue") and intent_kind == EXECUTION:
        add_gap(
            "medium",
            "missing_action_queue",
            "The next executable step should be visible as a gated queue item, not implied by prose.",
            "scientist_action_queue",
        )

    if not evidence_gaps:
        uncertainty_drivers.append("No major artifact gap was detected, but leaderboard/rank evidence is still absent unless Kaggle response artifacts exist.")

    hard_gaps = sum(1 for gap in evidence_gaps if gap.get("severity") == "blocking")
    medium_or_high = sum(1 for gap in evidence_gaps if gap.get("severity") in {"high", "medium"})
    actionability_score = 100
    actionability_score -= hard_gaps * 35
    actionability_score -= medium_or_high * 12
    if not selected_task:
        actionability_score -= 20
    if not can_execute:
        actionability_score -= 15
    actionability_score = max(0, min(100, actionability_score))

    if _is_meta_scientist_goal(prompt, payload):
        decision = "self_audit_then_consolidate_memory"
    elif hard_gaps:
        decision = "repair_or_observe_before_execution"
    elif actionability_score >= 70 and intent_kind == EXECUTION:
        decision = "ready_for_gated_execution_plan"
    elif actionability_score >= 55:
        decision = "observe_then_plan"
    else:
        decision = "insufficient_evidence"

    return {
        "decision": decision,
        "actionability_score": actionability_score,
        "evidence_gaps": evidence_gaps[:8],
        "uncertainty_drivers": uncertainty_drivers[:6],
        "recommended_tool_inserts": recommended_tool_inserts[:8],
        "tool_rationale": [
            {
                "tool": str(item.get("tool") or ""),
                "why": str(item.get("why") or ""),
                "gate": str(item.get("gate") or ""),
            }
            for item in selected_tools[:8]
        ],
        "claim_boundaries": [
            "Do not claim official Kaggle rank, medal, or top30 without a Kaggle response artifact.",
            "Do not treat a plan, local CV, or proxy score as leaderboard proof.",
            "Do not start training from this turn plan; execution must pass AgentSession/workstation gates.",
        ],
    }


def _apply_critique_to_tools(
    selected_tools: list[dict[str, Any]],
    critique: dict[str, Any],
) -> list[dict[str, Any]]:
    inserts = [str(item) for item in (critique.get("recommended_tool_inserts") or []) if item]
    if not inserts:
        return selected_tools
    existing = {str(item.get("tool") or "") for item in selected_tools}
    prepend: list[dict[str, Any]] = []
    for tool in inserts:
        if tool in existing:
            continue
        prepend.append(_tool_item(
            tool,
            "Inserted by scientific critique to close a concrete evidence or capability gap.",
            confidence=0.89,
            expected_artifacts=[f".xsci/{tool}.json"] if tool.startswith("scientist_") else [],
            gate="critique_gap_gate",
        ))
    return _dedup_tools(prepend + selected_tools)


def _build_tool_budget_policy(
    selected_tools: list[dict[str, Any]],
    critique: dict[str, Any],
) -> dict[str, Any]:
    """Recommend the minimum tool budget needed to close critique gaps."""
    tool_sequence = [str(item.get("tool") or "") for item in selected_tools if item.get("tool")]
    recommended = [
        str(item)
        for item in (critique.get("recommended_tool_inserts") or [])
        if str(item) in tool_sequence
    ]
    must_run_tools: list[str] = []
    for tool in recommended:
        if tool not in must_run_tools:
            must_run_tools.append(tool)
    last_required_index = -1
    for tool in must_run_tools:
        try:
            last_required_index = max(last_required_index, tool_sequence.index(tool))
        except ValueError:
            continue
    recommended_min = 4
    if last_required_index >= 0:
        recommended_min = max(recommended_min, last_required_index + 1)
    recommended_min = max(1, min(8, recommended_min, len(tool_sequence) or 1))
    expansion_reason = ""
    if recommended_min > 4:
        expansion_reason = "scientific_critique_requires_multi_tool_closure"
    elif must_run_tools:
        expansion_reason = "scientific_critique_required_tools_fit_default_budget"
    else:
        expansion_reason = "default_budget_sufficient"
    return {
        "default_max_tools": 4,
        "recommended_min_tools": recommended_min,
        "max_allowed_tools": 8,
        "must_run_tools": must_run_tools[:8],
        "expansion_reason": expansion_reason,
        "completion_gate": (
            "attempt_all_must_run_tools_before_declaring_scientist_turn_complete"
            if must_run_tools
            else "default_read_only_turn_completion"
        ),
    }


def _build_requirement_ledger(
    *,
    prompt: str,
    intent_kind: str,
    payload: str,
    selected_task: bool,
    can_execute: bool,
    blocking_gates: list[str],
    advisory_gaps: list[str],
    artifact_state: dict[str, dict[str, Any]],
    selected_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive a durable requirement ledger for the current Scientist turn.

    Strong coding agents do not just answer a prompt; they preserve a checklist
    of what the user asked for, what evidence would prove completion, and which
    gates still block action.  This ledger makes that checklist explicit and
    machine-readable for terminal, UI, recovery, and later memory writeback.
    """
    tool_names = {str(item.get("tool") or "") for item in selected_tools if item.get("tool")}
    requirements: list[dict[str, Any]] = []

    def artifact_ready(key: str) -> bool:
        return _artifact_present(artifact_state, key)

    def tool_planned(tool: str) -> bool:
        return tool in tool_names

    def add(
        req_id: str,
        description: str,
        *,
        status: str,
        evidence_needed: list[str],
        gate: str,
        reason: str = "",
        mapped_tools: list[str] | None = None,
    ) -> None:
        requirements.append({
            "id": req_id,
            "description": description,
            "status": status,
            "gate": gate,
            "reason": reason,
            "evidence_needed": evidence_needed,
            "mapped_tools": mapped_tools or [],
        })

    add(
        "selected_task_context",
        "A concrete Kaggle/MLE task must be selected before task-specific research or modeling.",
        status="satisfied" if selected_task else "blocked",
        evidence_needed=[".xsci/session.json", ".xsci/tasks/*.json"],
        gate="task_selection_gate",
        reason="" if selected_task else "No selected task is available in session state.",
        mapped_tools=["task_list"],
    )
    add(
        "setup_gate_clearance",
        "Hard setup gates must be clear before execution-like work can start.",
        status="satisfied" if not blocking_gates else "blocked",
        evidence_needed=["evomind ready", ".xsci/scientist_repair_plan.json"],
        gate="setup_gate",
        reason="; ".join(blocking_gates[:3]),
        mapped_tools=["system_status", "scientist_repair_plan"],
    )
    if advisory_gaps:
        add(
            "advisory_setup_review",
            "Advisory setup gaps should be visible so the answer does not hide degraded capability.",
            status="pending",
            evidence_needed=["evomind ready", ".xsci/scientist_step_trace.jsonl"],
            gate="advisory_gate",
            reason="; ".join(advisory_gaps[:3]),
            mapped_tools=["system_status", "scientist_step_trace"],
        )
    add(
        "situation_model",
        "The turn should observe current evidence, blockers, uncertainty, memory, and strategy before acting.",
        status="satisfied" if artifact_ready("situation_model") else ("planned" if tool_planned("scientist_situation_model") else "pending"),
        evidence_needed=[".xsci/scientist_situation_model.json"],
        gate="observe_orient_gate",
        mapped_tools=["scientist_situation_model"],
    )
    add(
        "context_packet",
        "Every AI Scientist turn should materialize a compact context packet before answering.",
        status="satisfied" if artifact_ready("context_packet") else ("planned" if tool_planned("scientist_context_packet") else "pending"),
        evidence_needed=[".xsci/scientist_context_packet.json", ".xsci/scientist_context_packet.md"],
        gate="context_grounding_gate",
        mapped_tools=["scientist_context_packet"],
    )
    add(
        "recoverable_workplan",
        "Complex work should leave a recoverable workplan rather than only prose.",
        status="satisfied" if artifact_ready("workplan") else ("planned" if tool_planned("scientist_workplan") else "pending"),
        evidence_needed=[".xsci/scientist_workplan.json"],
        gate="workplan_gate",
        mapped_tools=["scientist_workplan"],
    )
    if intent_kind == EXECUTION:
        add(
            "execution_contract",
            "Training-like requests require a go/no-go execution contract before any run starts.",
            status="satisfied" if artifact_ready("execution_contract") and can_execute else ("blocked" if blocking_gates else "planned"),
            evidence_needed=[".xsci/scientist_execution_contract.json"],
            gate="execution_contract_gate",
            reason="" if can_execute else "Execution is not currently ready.",
            mapped_tools=["scientist_execution_contract"],
        )
        add(
            "data_and_validation_contract",
            "Execution must produce or verify data/schema/validation contracts before model training claims.",
            status="planned" if tool_planned("scientist_execution_contract") else "pending",
            evidence_needed=["validation_contract", "metrics.json", "artifact_manifest"],
            gate="data_validation_gate",
            mapped_tools=["scientist_execution_contract", "data_check"],
        )
    if intent_kind == PLANNING:
        add(
            "memory_guided_hypotheses",
            "Planning should generate or reuse hypotheses from retrospective memory before choosing a branch.",
            status="planned" if tool_planned("scientist_innovation_backlog") else "pending",
            evidence_needed=[".xsci/scientist_innovation_backlog.json", "experiments/evolution/retrospective_memory.json"],
            gate="memory_reuse_gate",
            mapped_tools=["scientist_innovation_backlog", "evolution_status"],
        )
        add(
            "hypothesis_review",
            "Candidate ideas should be ranked by evidence, risk, readiness, and expected impact.",
            status="planned" if tool_planned("scientist_hypothesis_review") else "pending",
            evidence_needed=[".xsci/scientist_hypothesis_review.json"],
            gate="hypothesis_review_gate",
            mapped_tools=["scientist_hypothesis_review"],
        )
    if _is_meta_scientist_goal(prompt, payload):
        add(
            "agent_self_audit",
            "Requests to make EvoMind smarter require a capability self-audit before claiming parity.",
            status="planned" if tool_planned("scientist_self_audit") else "pending",
            evidence_needed=[".xsci/scientist_self_audit.json"],
            gate="capability_audit_gate",
            mapped_tools=["scientist_self_audit"],
        )
        add(
            "memory_consolidation",
            "A self-evolving Scientist must write recent lessons into reusable memory.",
            status="planned" if tool_planned("scientist_memory_consolidation") else "pending",
            evidence_needed=[".xsci/scientist_memory_consolidation.json", "experiments/evolution/retrospective_memory.json"],
            gate="memory_writeback_gate",
            mapped_tools=["scientist_memory_consolidation"],
        )
        add(
            "parity_lifecycle",
            "The turn must preserve an observe-plan-act-reflect-improve lifecycle artifact.",
            status="planned",
            evidence_needed=[".xsci/scientist_parity_loop.jsonl"],
            gate="scientist_parity_gate",
            mapped_tools=["scientist_turn_plan"],
        )
    add(
        "claim_boundary",
        "Rank, medal, top30, and official-submit claims require Kaggle response artifacts.",
        status="satisfied",
        evidence_needed=["Kaggle response artifact", "claim_audit"],
        gate="claim_audit_gate",
        reason="No leaderboard claim is made by a turn plan.",
        mapped_tools=["scientist_execution_contract"],
    )
    add(
        "secret_safety",
        "The turn must not read, print, or persist plaintext credentials.",
        status="satisfied",
        evidence_needed=["secret scan", "redacted turn artifacts"],
        gate="secret_safety_gate",
        mapped_tools=[],
    )
    add(
        "no_unapproved_training_or_submit",
        "The turn plan must not start training or official Kaggle submission by itself.",
        status="satisfied",
        evidence_needed=["no_training_started=true", "official_submit=blocked_until_explicit_human_approval"],
        gate="human_and_agent_session_gate",
        mapped_tools=[],
    )

    open_requirements = [
        item["id"] for item in requirements
        if item.get("status") not in {"satisfied"}
    ]
    blocked_requirements = [
        item["id"] for item in requirements
        if item.get("status") == "blocked"
    ]
    return {
        "schema": "evomind.ai_scientist.requirement_ledger.v1",
        "goal": prompt,
        "intent": {"kind": intent_kind, "payload": payload},
        "requirements": requirements,
        "satisfied_requirements": [item["id"] for item in requirements if item.get("status") == "satisfied"],
        "open_requirements": open_requirements,
        "blocked_requirements": blocked_requirements,
        "next_evidence_to_collect": [
            evidence
            for item in requirements
            if item.get("status") in {"blocked", "pending", "planned"}
            for evidence in item.get("evidence_needed", [])[:2]
        ][:10],
        "completion_gate": {
            "all_blocking_requirements_must_be_satisfied": True,
            "must_record_requirement_ledger": True,
            "must_record_artifact": ".xsci/scientist_turn_plan.json",
            "must_preserve_no_training": True,
            "must_preserve_submit_block": True,
        },
    }


def _build_parity_lifecycle(
    *,
    prompt: str,
    intent_kind: str,
    payload: str,
    selected_tools: list[dict[str, Any]],
    critique: dict[str, Any],
    requirement_ledger: dict[str, Any],
    artifact_state: dict[str, dict[str, Any]],
    next_safe_command: str,
) -> dict[str, Any]:
    """Create an explicit observe-plan-act-reflect-improve lifecycle contract."""
    tool_sequence = [str(item.get("tool") or "") for item in selected_tools if item.get("tool")]
    gaps = critique.get("evidence_gaps") if isinstance(critique.get("evidence_gaps"), list) else []
    gap_names = [
        str(gap.get("gap") or "")[:160]
        for gap in gaps
        if isinstance(gap, dict) and gap.get("gap")
    ]
    present_artifacts = [
        key for key, item in artifact_state.items()
        if isinstance(item, dict) and item.get("present")
    ]
    missing_artifacts = [
        key for key, item in artifact_state.items()
        if isinstance(item, dict) and not item.get("present")
    ]
    return {
        "schema": "evomind.ai_scientist.parity_lifecycle.v1",
        "loop_name": "observe_plan_act_reflect_improve",
        "goal": prompt,
        "intent": {"kind": intent_kind, "payload": payload},
        "phases": [
            {
                "phase": "observe",
                "status": "planned",
                "purpose": "Collect current task, setup, artifact, memory, and gate state before answering.",
                "evidence": {
                    "present_artifacts": present_artifacts[:12],
                    "missing_artifacts": missing_artifacts[:12],
                    "open_requirements": requirement_ledger.get("open_requirements", [])[:12],
                    "blocked_requirements": requirement_ledger.get("blocked_requirements", [])[:12],
                },
            },
            {
                "phase": "plan",
                "status": "planned",
                "purpose": "Choose tools, budget, stop conditions, and expected artifacts before acting.",
                "tool_sequence": tool_sequence,
                "requirement_count": len(requirement_ledger.get("requirements", []) or []),
            },
            {
                "phase": "act",
                "status": "planned",
                "purpose": "Execute only bounded safe tools in order; stop before training, download, or official submit gates.",
                "gate": "safe_terminal_tools_only",
            },
            {
                "phase": "reflect",
                "status": "planned",
                "purpose": "Convert evidence gaps, uncertainty, and blocked gates into explicit critique.",
                "evidence_gaps": gap_names[:8],
                "decision": critique.get("decision"),
            },
            {
                "phase": "improve",
                "status": "planned",
                "purpose": "Persist next safe command and improvement hooks for the next turn.",
                "next_safe_command": next_safe_command,
                "memory_writeback_required": bool(gap_names or critique.get("recommended_tool_inserts")),
            },
        ],
        "completion_gate": {
            "required_phases": ["observe", "plan", "act", "reflect", "improve"],
            "must_record_artifact": ".xsci/scientist_parity_loop.jsonl",
            "must_preserve_no_training": True,
            "must_preserve_submit_block": True,
        },
    }


def _select_tool_sequence(intent_kind: str, payload: str, *,
                          selected_task: bool,
                          blocking_gates: list[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    has_blocker = bool(blocking_gates)
    if intent_kind == EXECUTION:
        tools.extend([
            _tool_item(
                "scientist_situation_model",
                "Orient before compute: synthesize evidence, blockers, uncertainty, and memory.",
                confidence=0.96,
                expected_artifacts=[".xsci/scientist_situation_model.json"],
            ),
            _tool_item(
                "scientist_execution_contract",
                "Training requests must pass a go/no-go contract before any run starts.",
                confidence=0.98,
                expected_artifacts=[".xsci/scientist_execution_contract.json"],
                gate="execution_gate",
            ),
            _tool_item(
                "scientist_strategy_optimizer",
                "Rank safe interventions before spending compute or choosing a gated action.",
                confidence=0.93,
                expected_artifacts=[".xsci/scientist_strategy_optimizer.json"],
                gate="strategy_gate",
            ),
            _tool_item(
                "scientist_experiment_blueprint",
                "Convert the chosen hypothesis into branch, resource, artifacts, rollback, and memory writeback.",
                confidence=0.91,
                expected_artifacts=[".xsci/scientist_experiment_blueprint.json"],
            ),
            _tool_item(
                "scientist_action_queue",
                "Expose the next command and stop at training/download/submit gates.",
                confidence=0.90,
                expected_artifacts=[".xsci/scientist_action_queue.json"],
                gate="human_or_run_gate",
            ),
        ])
        if has_blocker:
            tools.insert(1, _tool_item(
                "scientist_repair_plan",
                "Setup blockers dominate the request, so produce repair steps before training.",
                confidence=0.94,
                expected_artifacts=[".xsci/scientist_repair_plan.json"],
            ))
    elif intent_kind == PLANNING:
        tools.extend([
            _tool_item("scientist_situation_model", "Ground the plan in the current evidence state.", confidence=0.94, expected_artifacts=[".xsci/scientist_situation_model.json"]),
            _tool_item("scientist_innovation_backlog", "Generate memory-guided research hypotheses before selecting branches.", confidence=0.92, expected_artifacts=[".xsci/scientist_innovation_backlog.json"]),
            _tool_item("scientist_hypothesis_review", "Rank hypotheses by evidence, risk, readiness, and expected impact.", confidence=0.91, expected_artifacts=[".xsci/scientist_hypothesis_review.json"]),
            _tool_item("scientist_strategy_optimizer", "Rank interventions and next commands by impact, evidence, cost, risk, and gates.", confidence=0.90, expected_artifacts=[".xsci/scientist_strategy_optimizer.json"]),
            _tool_item("scientist_workplan", "Materialize a recoverable multi-step plan for terminal and UI.", confidence=0.90, expected_artifacts=[".xsci/scientist_workplan.json"]),
        ])
    elif intent_kind == TOOL_QUERY:
        if payload == "scientist_autopilot":
            tools.extend([
                _tool_item("scientist_self_audit", "Capability-oriented requests need an agent self-audit before broad autopilot advice.", confidence=0.93, expected_artifacts=[".xsci/scientist_self_audit.json"]),
                _tool_item("scientist_situation_model", "Ground the capability diagnosis in current evidence, blockers, memory, and gates.", confidence=0.92, expected_artifacts=[".xsci/scientist_situation_model.json"]),
                _tool_item("scientist_memory_consolidation", "Check whether trace lessons are being written back into reusable memory.", confidence=0.90, expected_artifacts=[".xsci/scientist_memory_consolidation.json"]),
                _tool_item("scientist_autopilot", "Run the high-level diagnosis after self-audit and memory checks.", confidence=0.88, expected_artifacts=[".xsci/scientist_autopilot.json"]),
                _tool_item("scientist_strategy_optimizer", "Convert diagnosis and action queue into a ranked next-step strategy.", confidence=0.87, expected_artifacts=[".xsci/scientist_strategy_optimizer.json"]),
                _tool_item("scientist_workplan", "Turn the diagnosis into a recoverable multi-step plan.", confidence=0.86, expected_artifacts=[".xsci/scientist_workplan.json"]),
            ])
        elif payload:
            tools.append(_tool_item(
                payload,
                "User asked for this specific read-only capability.",
                confidence=0.95,
                expected_artifacts=[f".xsci/{payload}.json"] if payload.startswith("scientist_") else [],
            ))
        tools.append(_tool_item(
            "scientist_step_trace",
            "Attach recent tool events so the answer is grounded in observable work.",
            confidence=0.82,
            expected_artifacts=[".xsci/scientist_step_trace.jsonl"],
        ))
    else:
        tools.extend([
            _tool_item("scientist_situation_model", "Give broad questions a stateful scientific orientation first.", confidence=0.88, expected_artifacts=[".xsci/scientist_situation_model.json"]),
            _tool_item("system_status", "Check setup/readiness so advice does not ignore gates.", confidence=0.84),
            _tool_item("scientist_step_trace", "Use recent evidence instead of a stateless chat reply.", confidence=0.80, expected_artifacts=[".xsci/scientist_step_trace.jsonl"]),
        ])
    if not selected_task:
        tools.insert(0, _tool_item(
            "task_list",
            "No selected task: discover registered competitions before task-specific reasoning.",
            confidence=0.97,
        ))
    if not any(str(item.get("tool") or "") == "scientist_context_packet" for item in tools):
        insert_at = 1 if tools and tools[0].get("tool") == "task_list" else 0
        tools.insert(insert_at, _tool_item(
            "scientist_context_packet",
            "Build the per-turn context packet so reasoning is grounded in task, gates, memory, strategy, and artifacts.",
            confidence=0.96,
            expected_artifacts=[".xsci/scientist_context_packet.json", ".xsci/scientist_context_packet.md"],
            gate="context_grounding_gate",
        ))
    return _dedup_tools(tools)


def build_scientist_turn_plan(
    session: SessionState,
    root: Path | str,
    user_text: str = "",
    *,
    persist: bool = True,
    record_turn: bool = False,
) -> dict[str, Any]:
    """Build and optionally persist a per-turn AI Scientist control plan.

    The plan is intentionally read-only. It may recommend a gated training
    command, but it never executes one.
    """
    root_path = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_path = root_path / ".xsci" / "scientist_turn_plan.json"
    prompt = _safe_text(user_text or session.last_goal or "", limit=900)
    intent = classify(user_text or session.last_goal or "")
    payload = str(intent.payload or "")
    selected_task = bool(session.selected_task)
    blocking_gates = session.blocking_setup()
    advisory_gaps = session.missing_setup()
    can_execute = session.can_execute()
    selected_tools = _select_tool_sequence(
        intent.kind,
        payload,
        selected_task=selected_task,
        blocking_gates=blocking_gates,
    )
    artifact_state = _artifact_presence(root_path)
    critique = _build_scientific_critique(
        prompt=prompt,
        intent_kind=intent.kind,
        payload=payload,
        selected_task=selected_task,
        can_execute=can_execute,
        blocking_gates=blocking_gates,
        advisory_gaps=advisory_gaps,
        artifact_state=artifact_state,
        selected_tools=selected_tools,
    )
    selected_tools = _apply_critique_to_tools(selected_tools, critique)
    critique["tool_rationale"] = [
        {
            "tool": str(item.get("tool") or ""),
            "why": str(item.get("why") or ""),
            "gate": str(item.get("gate") or ""),
        }
        for item in selected_tools[:8]
    ]
    requirement_ledger = _build_requirement_ledger(
        prompt=prompt,
        intent_kind=intent.kind,
        payload=payload,
        selected_task=selected_task,
        can_execute=can_execute,
        blocking_gates=blocking_gates,
        advisory_gaps=advisory_gaps,
        artifact_state=artifact_state,
        selected_tools=selected_tools,
    )
    tool_budget = _build_tool_budget_policy(selected_tools, critique)
    autonomy_level = (
        "gated_executor_pending" if intent.kind == EXECUTION and can_execute else
        "repair_first" if blocking_gates else
        "planner_observer" if intent.kind == PLANNING else
        "read_only_tool_loop"
    )
    stop_conditions = [
        "Stop before any model training unless execution goes through evomind run / AgentSession gates.",
        "Stop before official Kaggle submit unless explicit human approval and Kaggle response artifact exist.",
        "Stop and repair if required artifacts, validation contract, or claim audit are missing.",
    ]
    if blocking_gates:
        stop_conditions.insert(0, "Stop now because setup gates are blocking execution: " + "; ".join(blocking_gates[:3]))
    if not selected_task:
        stop_conditions.insert(0, "Stop task-specific planning until a competition is selected or registered.")

    expected_artifacts: list[str] = []
    for item in selected_tools:
        expected_artifacts.extend(str(x) for x in item.get("expected_artifacts") or [])
    expected_artifacts = list(dict.fromkeys(expected_artifacts + [str(artifact_path)]))

    next_safe_command = "evomind situation"
    if intent.kind == PLANNING and selected_task:
        next_safe_command = "evomind innovate-plan"
    elif intent.kind == EXECUTION:
        next_safe_command = "evomind repair" if blocking_gates else "evomind contract"
    elif intent.kind == TOOL_QUERY and payload:
        next_safe_command = f"evomind {payload.replace('scientist_', '').replace('_', '-')}"

    parity_lifecycle = _build_parity_lifecycle(
        prompt=prompt,
        intent_kind=intent.kind,
        payload=payload,
        selected_tools=selected_tools,
        critique=critique,
        requirement_ledger=requirement_ledger,
        artifact_state=artifact_state,
        next_safe_command=next_safe_command,
    )

    plan: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_turn_plan",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "user_goal": prompt,
        "intent": {
            "kind": intent.kind,
            "payload": payload,
            "args": list(intent.args or []),
        },
        "goal_interpretation": {
            "requested_mode": intent.kind,
            "task_bound": selected_task,
            "needs_compute_gate": intent.kind == EXECUTION,
            "needs_llm_reasoning": intent.kind in {EXECUTION, PLANNING} or not payload,
        },
        "autonomy_level": autonomy_level,
        "readiness": {
            "llm_ready": session.llm_ready,
            "kaggle_ready": session.kaggle_ready,
            "compute_backend": session.compute_backend,
            "gpu_ready": session.gpu_ready,
            "gpu_blocked": session.gpu_blocked,
            "can_execute": can_execute,
            "blocking_gates": blocking_gates,
            "advisory_gaps": advisory_gaps,
        },
        "selected_tools": selected_tools,
        "tool_sequence": [item["tool"] for item in selected_tools],
        "scientific_critique": critique,
        "requirement_ledger": requirement_ledger,
        "tool_budget": tool_budget,
        "parity_lifecycle": parity_lifecycle,
        "expected_artifacts": expected_artifacts,
        "stop_conditions": stop_conditions,
        "next_safe_command": next_safe_command,
        "artifact_state": artifact_state,
        "response_contract": {
            "must_reference_evidence": True,
            "must_name_blockers": True,
            "must_report_open_requirements": True,
            "must_not_claim_rank_or_medal_without_kaggle_response": True,
            "must_not_print_or_read_secrets": True,
        },
        "artifact_path": str(artifact_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    plan = _safe_json(plan)

    if persist:
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = artifact_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(artifact_path)
        except OSError as exc:
            plan["ok"] = False
            plan["message"] = f"Could not write turn-plan artifact: {exc}"

        try:
            from .scientist_trace import record_scientist_step_event

            record_scientist_step_event(root_path, {
                "trace_run_id": f"turn_plan_{generated_at.replace(':', '').replace('+', 'Z')}",
                "source": "scientist_turn_plan",
                "task": session.selected_task or "",
                "phase": "turn_planning",
                "status": "passed" if plan.get("ok", True) else "blocked",
                "tool": "scientist_turn_plan",
                "message": f"intent={intent.kind}; autonomy={autonomy_level}; tools={len(selected_tools)}",
                "artifact_path": str(artifact_path),
                "details": {
                    "intent": intent.kind,
                    "payload": payload,
                    "tool_sequence": plan.get("tool_sequence", []),
                    "blocking_gates": blocking_gates,
                    "open_requirements": requirement_ledger.get("open_requirements", []),
                    "blocked_requirements": requirement_ledger.get("blocked_requirements", []),
                },
                "no_training_started": True,
            })
        except Exception:
            pass

    if record_turn:
        try:
            from .scientist_turns import record_scientist_turn

            record_scientist_turn(root_path, {
                "task": session.selected_task or "",
                "route": "scientist_turn_plan",
                "user": prompt or "scientist_turn_plan",
                "forced_tools": plan.get("tool_sequence", []),
                "executed_tools": [{"tool": "scientist_turn_plan", "ok": plan.get("ok", True)}],
                "mode": autonomy_level,
                "decision": {
                    "intent": intent.kind,
                    "next_safe_command": next_safe_command,
                    "tool_sequence": plan.get("tool_sequence", []),
                    "requirement_ledger": plan.get("requirement_ledger", {}),
                    "parity_lifecycle": plan.get("parity_lifecycle", {}),
                },
                "blockers": blocking_gates,
                "next_actions": [next_safe_command],
                "artifacts": [str(artifact_path)],
                "answer_preview": f"turn plan intent={intent.kind}; autonomy={autonomy_level}",
                "parity_lifecycle": plan.get("parity_lifecycle", {}),
                "no_training_started": True,
            })
        except Exception:
            pass

    return plan
