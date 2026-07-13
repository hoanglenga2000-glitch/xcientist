"""Provider-neutral, bounded tool loop for complex EvoMind Scientist turns.

The deterministic turn planner remains the safety authority.  This module adds
the missing Codex/Claude-style behavior between planning and final synthesis:
the configured model can inspect a tool result, choose a different next tool,
recover from a failed observation, and stop with an evidence-backed answer.
Only the read-only/gated ``TerminalTools`` registry is exposed and every call is
bounded, deduplicated, traced, and persisted without raw secret-bearing values.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research_os.agent.messaging import AgentMessageClient, ToolResult, ToolSpec

_SENSITIVE_KEY = re.compile(r"(api[_-]?key|token|password|passwd|secret|cookie|authorization)", re.I)

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "model_status": "Inspect the configured model/provider without reading credentials.",
    "system_status": "Inspect model, Kaggle, compute, task, and dashboard readiness gates.",
    "task_list": "List registered research tasks when no task is selected.",
    "inspect_task": "Inspect the selected task, metric, modality, and schema.",
    "data_check": "Check declared train/test/sample data availability without downloading.",
    "recent_run": "Inspect the latest run and best-so-far evidence.",
    "gpu_status": "Inspect GPU/HPC evidence and blockers without opening a connection.",
    "kaggle_status": "Inspect Kaggle configuration state without reading the token.",
    "next_steps": "Summarize current blockers and the safest next action.",
    "evolution_status": "Inspect retrospective memory, evolution tracking, and prior lessons.",
    "scientist_checkpoint": "Build a structured research checkpoint from current evidence.",
    "scientist_context_packet": "Build the compact task, gate, memory, and artifact context packet.",
    "research_decision": "Choose the next evidence-backed research branch and rollback condition.",
    "scientist_workplan": "Build a recoverable multi-step workplan with artifacts and gates.",
    "scientist_step_trace": "Inspect recent tool, gate, and artifact events.",
    "scientist_self_audit": "Audit EvoMind capability using current behavioral evidence.",
    "scientist_readiness_report": "Separate local capability from execution and claim readiness.",
    "scientist_causal_diagnosis": "Map symptoms to root causes, evidence, and safe interventions.",
    "scientist_strategy_optimizer": "Rank candidate interventions by evidence, impact, cost, and risk.",
    "scientist_upgrade_plan": "Turn capability gaps into a scoped engineering plan.",
    "scientist_self_upgrade_loop": "Create a gated Code Agent work order for the next capability gap.",
    "scientist_patch_work_order": "Create or inspect the current patch work order.",
    "scientist_engineering_loop": "Validate an existing patch in an isolated Git worktree.",
    "scientist_memory_consolidation": "Write sanitized outcomes and failures into retrospective memory.",
    "scientist_innovation_backlog": "Generate memory-guided, auditable research hypotheses.",
    "scientist_hypothesis_review": "Critique and rank hypotheses using evidence and risk gates.",
    "scientist_experiment_blueprint": "Create an auditable experiment blueprint without training.",
    "scientist_innovation_trial_feedback": "Record the result of a completed innovation trial.",
    "scientist_situation_model": "Synthesize evidence, uncertainty, blockers, strategy, and memory.",
    "scientist_recovery": "Build a long-horizon recovery snapshot and resume commands.",
    "scientist_repair_plan": "Diagnose blockers and produce safe repair steps.",
    "scientist_execution_contract": "Build the pre-execution go/no-go and claim-boundary contract.",
    "scientist_action_queue": "Expose the next gated action and expected artifacts.",
    "scientist_continuation_status": "Inspect unfinished must-run tools and safe continuation state.",
}

_ALWAYS_AVAILABLE = (
    "system_status",
    "task_list",
    "inspect_task",
    "data_check",
    "recent_run",
    "evolution_status",
    "scientist_checkpoint",
    "scientist_situation_model",
    "scientist_repair_plan",
    "scientist_execution_contract",
    "scientist_step_trace",
    "scientist_recovery",
    "scientist_continuation_status",
)

_META_TOOLS = (
    "scientist_self_audit",
    "scientist_readiness_report",
    "scientist_causal_diagnosis",
    "scientist_strategy_optimizer",
    "scientist_upgrade_plan",
    "scientist_self_upgrade_loop",
    "scientist_patch_work_order",
    "scientist_engineering_loop",
    "scientist_memory_consolidation",
)


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "[nested]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1200]
    if isinstance(value, list):
        return [_safe_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            key_text = str(key)
            if _SENSITIVE_KEY.search(key_text):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = _safe_value(item, depth=depth + 1)
        return result
    return str(value)[:1200]


def _result_text(tool: str, result: dict[str, Any]) -> str:
    safe = _safe_value(result)
    rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    return f"[{tool}] {rendered[:7000]}"


def _is_meta_goal(goal: str) -> bool:
    low = (goal or "").lower()
    return any(token in low for token in (
        "claude code", "codex", "ai scientist", "agent capability",
        "自动进化", "自我进化", "自进化", "不够智能", "复杂问题",
        "工程闭环", "修复系统", "优化系统", "提升能力",
    ))


def _allowed_tools(turn_plan: dict[str, Any], goal: str) -> list[str]:
    from .terminal_tools import TerminalTools

    registered = set(TerminalTools.list_tool_names())
    planned = [str(item) for item in (turn_plan.get("tool_sequence") or []) if item]
    candidates = [*planned, *_ALWAYS_AVAILABLE]
    if _is_meta_goal(goal):
        candidates.extend(_META_TOOLS)
    # Nested autonomous loops and browser-opening tools are deliberately absent.
    blocked = {
        "dashboard",
        "scientist_autopilot",
        "scientist_loop",
        "scientist_next_action",
        "scientist_continuation_resume",
        "scientist_context_packet",  # the caller already builds it deterministically
        "scientist_turn_plan",
    }
    return list(dict.fromkeys(
        name for name in candidates
        if name in registered and name not in blocked
    ))


def _tool_specs(names: list[str]) -> list[ToolSpec]:
    schema = {"type": "object", "properties": {}, "required": []}
    return [
        ToolSpec(
            name,
            _TOOL_DESCRIPTIONS.get(name, "Inspect one bounded EvoMind Scientist evidence source."),
            schema,
        )
        for name in names
    ]


def _requirement_resolution(
    turn_plan: dict[str, Any],
    calls: list[dict[str, Any]],
    runtime_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from .terminal_agent import _tool_result_blocking_signals

    ledger = turn_plan.get("requirement_ledger")
    items = (
        ledger.get("requirements")
        if isinstance(ledger, dict) and isinstance(ledger.get("requirements"), list)
        else ledger.get("items") if isinstance(ledger, dict) else []
    )
    if not isinstance(items, list):
        items = []
    passed = {
        str(item.get("tool") or "")
        for item in calls
        if isinstance(item, dict) and item.get("ok") is True and item.get("executed") is True
    }
    blocking_signals = {
        tool: _tool_result_blocking_signals(runtime_results.get(tool))
        for tool in passed
    }
    resolved: list[str] = []
    unresolved: list[str] = []
    blocked: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        requirement_id = str(item.get("id") or "")
        status = str(item.get("status") or "pending")
        mapped = {str(tool) for tool in (item.get("mapped_tools") or []) if tool}
        mapped_hits = mapped & passed
        clear_hits = {tool for tool in mapped_hits if not blocking_signals.get(tool)}
        blocked_hits = {tool: blocking_signals[tool] for tool in mapped_hits if blocking_signals.get(tool)}
        if status in {"satisfied", "complete", "completed"} or clear_hits:
            resolved.append(requirement_id)
        elif status == "blocked" or blocked_hits:
            blocked.append(requirement_id)
        else:
            unresolved.append(requirement_id)
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "blocked": blocked,
        "tool_blocking_signals": blocking_signals,
        "closure_ratio": round(len(resolved) / max(1, len(resolved) + len(unresolved) + len(blocked)), 3),
    }


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_adaptive_scientist_tool_loop(
    session: Any,
    root: Path | str,
    *,
    goal: str,
    turn_plan: dict[str, Any],
    initial_evidence: dict[str, Any] | None = None,
    max_rounds: int = 4,
    max_tool_calls: int = 6,
    client: Any | None = None,
    dispatch: Callable[[str, Any, Path], dict[str, Any]] | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run a bounded model-directed observe/act/replan loop.

    ``client`` and ``dispatch`` are injectable so failure recovery and anti-loop
    behavior can be tested without a network, credentials, or real side effects.
    """
    from .terminal_tools import TerminalTools

    root_path = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_path = root_path / ".xsci" / "scientist_adaptive_tool_loop.json"
    history_path = root_path / ".xsci" / "scientist_adaptive_tool_loop_history.jsonl"
    allowed = _allowed_tools(turn_plan, goal)
    max_rounds = max(1, min(8, int(max_rounds)))
    max_tool_calls = max(1, min(10, int(max_tool_calls)))
    dispatcher = dispatch or TerminalTools.dispatch
    calls: list[dict[str, Any]] = []
    runtime_results: dict[str, dict[str, Any]] = {}
    messages: list[dict[str, Any]] = []
    final_text = ""
    stop_reason = "provider_unavailable"
    provider = ""
    model = ""
    input_tokens = 0
    output_tokens = 0
    failure_round: int | None = None
    failed_tools: set[str] = set()
    replanned_after_failure = False

    def emit(phase: str, status: str, message: str, **details: Any) -> None:
        if observer is None:
            return
        try:
            observer({
                "source": "scientist_adaptive_tool_loop",
                "phase": phase,
                "status": status,
                "message": message,
                "details": _safe_value(details),
                "no_training_started": True,
            })
        except Exception:
            pass

    if client is None:
        client = AgentMessageClient(max_retries=1, timeout=120)
    available = bool(getattr(client, "is_available", lambda: False)())
    if not available or not allowed:
        payload = {
            "ok": True,
            "available": available,
            "used": False,
            "tool": "scientist_adaptive_tool_loop",
            "generated_at": generated_at,
            "selected_task": getattr(session, "selected_task", "") or "",
            "goal": goal[:1600],
            "allowed_tools": allowed,
            "stop_reason": "provider_unavailable" if not available else "no_allowed_tools",
            "tool_calls": [],
            "requirement_resolution": _requirement_resolution(turn_plan, [], {}),
            "artifact_path": str(artifact_path),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
        if persist:
            _write_artifact(artifact_path, payload)
        payload["runtime_results"] = runtime_results
        return payload

    compact_plan = {
        "intent": turn_plan.get("intent"),
        "autonomy_level": turn_plan.get("autonomy_level"),
        "tool_sequence": turn_plan.get("tool_sequence"),
        "scientific_critique": turn_plan.get("scientific_critique"),
        "requirement_ledger": turn_plan.get("requirement_ledger"),
        "readiness": turn_plan.get("readiness"),
        "stop_conditions": turn_plan.get("stop_conditions"),
    }
    system = (
        "You are EvoMind's adaptive tool orchestrator. Close the user's requirements by observing evidence, "
        "calling one or more listed tools, reading each result, and replanning when evidence changes. If a tool "
        "fails, do not repeat it blindly: select a complementary diagnosis or repair-plan tool on the next round. "
        "Do not request or reveal credentials. Never train, download data, modify the main worktree, merge code, "
        "or submit to Kaggle in this loop. Stop at those gates and name them. Do not claim rank or success without "
        "an artifact. Prefer one focused tool per round so the next choice can use the new result. Finish with a "
        "direct evidence-backed answer and explicit remaining requirements."
    )
    messages.append({
        "role": "user",
        "content": (
            f"[USER GOAL]\n{goal[:3000]}\n\n"
            f"[TURN PLAN]\n{json.dumps(_safe_value(compact_plan), ensure_ascii=False)[:18000]}\n\n"
            f"[INITIAL EVIDENCE]\n{json.dumps(_safe_value(initial_evidence or {}), ensure_ascii=False)[:9000]}"
        ),
    })
    specs = _tool_specs(allowed)
    emit("adaptive_start", "running", "Starting model-directed tool selection.", allowed_tools=allowed)

    for round_index in range(1, max_rounds + 1):
        if sum(1 for item in calls if item.get("executed")) >= max_tool_calls:
            stop_reason = "tool_budget_exhausted"
            break
        try:
            turn = client.send(
                messages,
                system=system,
                tools=specs,
                max_tokens=1200,
                temperature=0.15,
            )
        except Exception as exc:
            stop_reason = f"model_error:{type(exc).__name__}"
            emit("adaptive_model_error", "blocked", stop_reason)
            break
        provider = str(getattr(turn, "provider", "") or provider)
        model = str(getattr(turn, "model", "") or model)
        input_tokens += int(getattr(turn, "input_tokens", 0) or 0)
        output_tokens += int(getattr(turn, "output_tokens", 0) or 0)
        if getattr(turn, "text", ""):
            final_text = str(turn.text)[:12000]
        messages.append({"role": "assistant", "content": turn.raw_content})
        if not turn.wants_tool:
            stop_reason = str(turn.stop_reason or "end_turn")
            emit("adaptive_complete", "passed", "Model completed the adaptive turn.", round=round_index)
            break

        results: list[dict[str, Any]] = []
        for call in turn.tool_calls:
            name = str(call.name or "")
            already_executed = any(item.get("tool") == name and item.get("executed") for item in calls)
            budget_used = sum(1 for item in calls if item.get("executed"))
            record: dict[str, Any] = {
                "round": round_index,
                "tool": name,
                "rationale": str(turn.text or "Model selected this tool after reviewing current evidence.")[:800],
                "executed": False,
                "ok": False,
                "artifact_path": "",
                "summary": "",
            }
            if name not in allowed:
                out = f"Tool '{name}' is not allowed in the bounded Scientist loop. Choose a listed read-only tool."
                record["blocked_reason"] = "tool_not_allowed"
            elif already_executed:
                out = f"Tool '{name}' already ran in this turn. Replan from its result instead of repeating it."
                record["blocked_reason"] = "duplicate_tool_call"
            elif budget_used >= max_tool_calls:
                out = "The bounded Scientist tool budget is exhausted. Stop and summarize remaining requirements."
                record["blocked_reason"] = "tool_budget_exhausted"
            else:
                try:
                    result = dispatcher(name, session, root_path)
                except Exception as exc:
                    result = {"ok": False, "tool": name, "message": f"{type(exc).__name__}: {exc}"}
                if not isinstance(result, dict):
                    result = {"ok": False, "tool": name, "message": "tool returned a non-object result"}
                runtime_results[name] = result
                record["executed"] = True
                record["ok"] = bool(result.get("ok", True))
                record["artifact_path"] = str(result.get("artifact_path") or result.get("path") or "")
                record["summary"] = str(result.get("message") or result.get("status") or result.get("mode") or "")[:700]
                out = _result_text(name, result)
                if not record["ok"] and failure_round is None:
                    failure_round = round_index
                if not record["ok"]:
                    failed_tools.add(name)
                elif (
                    failure_round is not None
                    and round_index > failure_round
                    and name not in failed_tools
                ):
                    replanned_after_failure = True
                emit(
                    "adaptive_tool_completed" if record["ok"] else "adaptive_tool_failed",
                    "passed" if record["ok"] else "blocked",
                    f"{name} {'completed' if record['ok'] else 'failed'}; model must re-evaluate next action.",
                    round=round_index,
                    tool=name,
                    artifact_path=record["artifact_path"],
                )
            calls.append(record)
            results.append(ToolResult(
                tool_use_id=call.id,
                content=out,
                is_error=not record.get("ok", False),
            ).to_wire())
        messages.append({"role": "user", "content": results})
    else:
        stop_reason = "round_budget_exhausted"

    planned = [
        str(item) for item in (turn_plan.get("tool_sequence") or [])
        if item not in {"scientist_context_packet", "scientist_turn_plan"}
    ]
    executed_names = [str(item.get("tool") or "") for item in calls if item.get("executed")]
    dynamic_selection = bool(executed_names) and executed_names != planned[:len(executed_names)]
    requirement_resolution = _requirement_resolution(turn_plan, calls, runtime_results)
    open_requirements = [
        *requirement_resolution["unresolved"],
        *requirement_resolution["blocked"],
    ]
    if stop_reason.startswith("model_error"):
        status = "blocked"
    elif open_requirements:
        status = "needs_continuation"
    else:
        status = "completed"
    persisted_calls = [{key: value for key, value in item.items()} for item in calls]
    payload = {
        "ok": status != "blocked",
        "available": True,
        "used": True,
        "tool": "scientist_adaptive_tool_loop",
        "schema": "evomind.ai_scientist.adaptive_tool_loop.v1",
        "generated_at": generated_at,
        "selected_task": getattr(session, "selected_task", "") or "",
        "goal": goal[:1600],
        "provider": provider,
        "model": model,
        "rounds": max((int(item.get("round") or 0) for item in calls), default=0),
        "max_rounds": max_rounds,
        "max_tool_calls": max_tool_calls,
        "allowed_tools": allowed,
        "planned_tools": planned,
        "executed_tools": executed_names,
        "tool_calls": persisted_calls,
        "dynamic_tool_selection": dynamic_selection,
        "failure_observed": failure_round is not None,
        "replanned_after_failure": replanned_after_failure,
        "duplicate_calls_blocked": sum(1 for item in calls if item.get("blocked_reason") == "duplicate_tool_call"),
        "unsafe_or_unknown_calls_blocked": sum(1 for item in calls if item.get("blocked_reason") == "tool_not_allowed"),
        "requirement_resolution": requirement_resolution,
        "status": status,
        "open_requirements": open_requirements,
        "stop_reason": stop_reason,
        "final_text": final_text,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "artifact_path": str(artifact_path),
        "history_path": str(history_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "main_worktree_merge": "blocked_until_human_review",
        },
        "next_safe_action": (
            "evomind continuation-status"
            if status == "needs_continuation"
            else ""
        ),
    }
    if persist:
        _write_artifact(artifact_path, payload)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    emit(
        "adaptive_finish",
        "passed" if payload["ok"] else "blocked",
        f"stop={stop_reason}; tools={len(executed_names)}; unresolved={len(requirement_resolution['unresolved'])}",
        replanned_after_failure=replanned_after_failure,
    )
    payload["runtime_results"] = runtime_results
    return payload
