"""TerminalAgent — the control layer for EvoMind's Claude Code-like interaction.

This module bridges the command shell (kaggle.py) and the tool ecosystem.  It
receives raw user text, classifies intent, decides whether the user wants a
tool query, training, chat, or planning, and orchestrates the result — complete
with streaming stage events and structured output.

It is intentionally *not* a full research agent (that is ``research_os.agent.
AgentSession``).  It handles the terminal conversation surface: tool inspection,
model queries, gate checks, and the decision of whether to enter training.
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .kaggle_intent import (
    CAPABILITY, CHAT, EXECUTION, GREETING, OFFICIAL, PLANNING, REPORT, MEMORY,
    STATUS, TASK_ADD, TASK_USE, TOOL_QUERY, classify,
)
from .kaggle_session import SessionState, MODE_CHAT, MODE_EXECUTING, MODE_PLANNING
from .recovery_guard import RecoveryGuard
from .terminal_events import (
    TerminalEventEmitter,
    render_scientist_autopilot_summary,
    render_scientist_causal_diagnosis_summary,
    render_scientist_context_packet_summary,
    render_scientist_continuation_resume_summary,
    render_scientist_continuation_status_summary,
    render_scientist_experiment_blueprint_summary,
    render_scientist_hypothesis_review_summary,
    render_scientist_innovation_backlog_summary,
    render_scientist_innovation_trial_feedback_summary,
    render_scientist_loop_summary,
    render_scientist_memory_consolidation_summary,
    render_scientist_recovery_summary,
    render_scientist_readiness_report_summary,
    render_scientist_self_audit_summary,
    render_scientist_strategy_optimizer_summary,
    render_scientist_patch_work_order_summary,
    render_scientist_self_upgrade_loop_summary,
    render_scientist_situation_model_summary,
    render_scientist_step_trace_timeline,
    render_scientist_turn_plan_summary,
    render_scientist_upgrade_plan_summary,
    render_tool_result_as_lines,
)
from .terminal_tools import TerminalTools, run_scientist_autopilot, run_scientist_continuation_resume, run_scientist_loop
from .tool_ledger import ToolLedger


@dataclass
class TerminalResult:
    """The outcome of one user turn handled by the TerminalAgent."""
    rc: int
    should_exit: bool
    selected_task: Optional[str] = None
    action: str = ""           # "tool_call" | "training" | "planning" | "chat" | "greeting" | "report"
    summary: str = ""
    artifacts: list[str] = field(default_factory=list)
    blocked: bool = False


def _ansi(code: str, text: str) -> str:
    import os, sys
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _strong(text: str) -> str:
    return _ansi("97;1", text)


def _dim(text: str) -> str:
    return _ansi("90", text)


def _resolve_requirement_ledger_after_tools(
    ledger: dict[str, Any],
    *,
    executed: list[dict[str, Any]],
    artifacts: list[str],
    blockers: list[str],
    no_training_started: bool = True,
    official_submit: str = "blocked_until_explicit_human_approval",
) -> dict[str, Any]:
    """Update a planned requirement ledger with post-tool execution evidence.

    The turn planner creates the ledger before tools run.  This resolver closes
    requirements when their mapped safe tools actually executed and keeps hard
    gates blocked when live blockers remain.  The result is still read-only:
    it never trains and never submits; it only makes the Scientist turn's
    reasoning auditable after action.
    """
    if not isinstance(ledger, dict):
        return {}
    executed_ok = {
        str(item.get("tool") or "")
        for item in executed
        if item.get("ok") and str(item.get("tool") or "") != "scientist_turn_plan"
    }
    artifact_set = {
        candidate
        for item in artifacts
        if str(item)
        for candidate in {str(item), str(item).replace("\\", "/")}
    }
    requirements: list[dict[str, Any]] = []
    for raw_item in ledger.get("requirements") or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        req_id = str(item.get("id") or "")
        current = str(item.get("status") or "pending")
        mapped_tools = [str(tool) for tool in (item.get("mapped_tools") or []) if str(tool)]
        mapped_tool_hits = [tool for tool in mapped_tools if tool in executed_ok]
        evidence_needed = [str(value) for value in (item.get("evidence_needed") or []) if str(value)]
        evidence_hits = [
            evidence
            for evidence in evidence_needed
            if any(evidence in artifact or evidence.replace("\\", "/") in artifact for artifact in artifact_set)
        ]

        status = current
        reason = str(item.get("reason") or "")
        if req_id == "setup_gate_clearance":
            status = "blocked" if blockers else "satisfied"
            reason = "; ".join(blockers[:3]) if blockers else "No blocking setup gates remain in this turn."
        elif req_id == "no_unapproved_training_or_submit":
            status = (
                "satisfied"
                if no_training_started and official_submit == "blocked_until_explicit_human_approval"
                else "blocked"
            )
            reason = "Training and official submit remained blocked for this Scientist turn."
        elif req_id in {"secret_safety", "claim_boundary", "selected_task_context"} and current == "satisfied":
            status = "satisfied"
        elif mapped_tool_hits:
            status = "satisfied"
            reason = f"Closed by executed safe tool(s): {', '.join(mapped_tool_hits[:4])}."
        elif evidence_hits and current != "blocked":
            status = "satisfied"
            reason = f"Closed by artifact evidence: {', '.join(evidence_hits[:3])}."
        elif current == "planned":
            status = "pending"
            reason = reason or "Planned tool did not run in the current tool budget."

        item["status"] = status
        item["reason"] = reason
        item["execution_evidence"] = {
            "mapped_tool_hits": mapped_tool_hits,
            "artifact_hits": evidence_hits,
        }
        requirements.append(item)

    satisfied = [str(item.get("id")) for item in requirements if item.get("status") == "satisfied"]
    open_requirements = [str(item.get("id")) for item in requirements if item.get("status") != "satisfied"]
    blocked_requirements = [str(item.get("id")) for item in requirements if item.get("status") == "blocked"]
    next_evidence = [
        evidence
        for item in requirements
        if item.get("status") != "satisfied"
        for evidence in (item.get("evidence_needed") or [])[:2]
    ][:10]
    return {
        **ledger,
        "requirements": requirements,
        "satisfied_requirements": satisfied,
        "open_requirements": open_requirements,
        "blocked_requirements": blocked_requirements,
        "next_evidence_to_collect": next_evidence,
        "resolution": {
            "mode": "post_tool_execution",
            "executed_tools": sorted(executed_ok),
            "artifact_count": len(artifact_set),
            "blocker_count": len(blockers),
            "no_training_started": no_training_started,
            "official_submit": official_submit,
        },
    }


def _scientist_continuation_command(goal: str, recommended_budget: int) -> str:
    safe_goal = " ".join(str(goal or "").replace('"', "'").split())
    safe_goal = re.sub(
        r"(?i)(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        safe_goal,
    )
    safe_goal = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}\b", "[redacted-key]", safe_goal)
    if len(safe_goal) > 220:
        safe_goal = safe_goal[:217] + "..."
    budget = max(1, min(8, int(recommended_budget or 4)))
    return f'evomind ask --json --max-tools {budget} "{safe_goal}"'


def _agent_reply(text: str, *, title: str = "EvoMind") -> None:
    """Print a paragraph-wrapped agent reply."""
    print()
    print(_strong(title))
    for paragraph in text.strip().split("\n"):
        if not paragraph.strip():
            print()
            continue
        wrapped = textwrap.wrap(paragraph, width=88, replace_whitespace=False) or [""]
        for line in wrapped:
            print(f"  {line}")


class TerminalAgent:
    """Control layer: intent → action → tool/stream → result.

    Usage::

        agent = TerminalAgent()
        result = agent.handle("你现在使用的什么模型", session, root)
        # result.action == "tool_call", result.summary contains model info
    """

    def __init__(self, *, colour: bool = True) -> None:
        self._colour = colour
        self._emitter: Optional[TerminalEventEmitter] = None
        self._guard = RecoveryGuard()
        self._ledger: Optional[ToolLedger] = None

    def _get_emitter(self, root: Path) -> TerminalEventEmitter:
        if self._emitter is None or self._emitter.workspace_root != root:
            self._emitter = TerminalEventEmitter(root, colour=self._colour)
        return self._emitter

    def handle(self, text: str, session: SessionState, root: Path) -> TerminalResult:
        """Main dispatch for one user turn."""
        raw = (text or "").strip()

        # ── Recovery: wire guard + ledger on every turn ───────────
        self._ledger = ToolLedger(root)
        self._guard.set_state_file(Path(root) / ".xsci" / "recovery_guard.md")
        self._guard.emit(session, event="UserPromptSubmit")

        if not raw:
            return self._empty_turn(session)

        intent = classify(raw)
        result: TerminalResult

        # ── Greetings ──────────────────────────────────────────────
        if intent.kind == GREETING:
            result = TerminalResult(
                rc=0, should_exit=False, action="greeting",
                summary="你好，我是 EvoMind 对话终端。我可以帮你浏览比赛、检查数据、规划实验、启动训练。输入 `help` 查看命令。",
                selected_task=session.selected_task,
            )

        # ── TOOL_QUERY ────────────────────────────────────────────
        elif intent.kind == TOOL_QUERY:
            result = self._handle_tool_query(intent, session, root)

        # ── EXECUTION (training) ──────────────────────────────────
        elif intent.kind == EXECUTION:
            result = self._handle_execution(raw, intent, session, root)

        # ── PLANNING ──────────────────────────────────────────────
        elif intent.kind == PLANNING:
            result = self._handle_planning(raw, session, root)

        # ── CHAT: make broad natural-language turns observable ──────
        elif intent.kind == CHAT:
            result = self.handle_scientist_turn(raw, session, root)

        # ── Other intents that the main dispatcher handles directly ─
        else:
            result = TerminalResult(
                rc=0, should_exit=False, action="passthrough",
                summary="", selected_task=session.selected_task,
            )

        # ── Record the turn in the tool ledger ────────────────────
        self._ledger.record(
            result.action or "unknown",
            {"summary": result.summary[:200]},
            ok=(result.rc == 0 and not result.blocked),
            summary=result.summary[:200],
        )

        # ── Update recovery guard after the turn ──────────────────
        self._guard.record_tool(f"{result.action}: {'ok' if result.rc == 0 else 'rc=' + str(result.rc)} — {result.summary[:100]}")
        self._guard.emit(session, event="PostToolCall")

        return result

    def handle_scientist_turn(self, text: str, session: SessionState,
                              root: Path, *, max_tools: int = 4) -> TerminalResult:
        """Run one safe AI Scientist turn for natural-language requests.

        This is the terminal equivalent of a Claude/Codex turn: plan the tools,
        execute bounded read-only observations, persist artifacts, and stop
        before any training/download/official-submit gate.  It gives broad
        prompts a visible research loop instead of a stateless chat answer.
        """
        raw = (text or "").strip()
        self._ledger = ToolLedger(root)
        self._guard.set_state_file(Path(root) / ".xsci" / "recovery_guard.md")
        self._guard.emit(session, event="ScientistTurnSubmit")
        emitter = self._get_emitter(root)
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        from .scientist_turn_planner import build_scientist_turn_plan
        from .scientist_trace import record_scientist_step_event
        from .scientist_turns import record_scientist_parity_loop, record_scientist_turn

        session.last_goal = raw
        emitter.emit(
            "AI Scientist turn",
            "planning tools, gates, evidence, and stop conditions",
            status="running",
        )
        plan = build_scientist_turn_plan(session, root, raw, persist=True)
        plan_artifact = str(plan.get("artifact_path") or "")
        emitter.emit(
            "AI Scientist turn",
            f"plan ready: autonomy={plan.get('autonomy_level', 'unknown')}",
            status="passed" if plan.get("ok", True) else "blocked",
            artifact=plan_artifact or None,
        )

        try:
            record_scientist_step_event(root, {
                "trace_run_id": f"terminal_turn_{generated_at.replace(':', '').replace('+', 'Z')}",
                "source": "terminal_scientist_turn",
                "task": session.selected_task or "",
                "phase": "terminal_turn_plan",
                "status": "passed" if plan.get("ok", True) else "blocked",
                "tool": "scientist_turn_plan",
                "message": f"tool_sequence={plan.get('tool_sequence', [])}",
                "artifact_path": plan_artifact,
                "no_training_started": True,
            })
        except Exception:
            pass

        safe_tools = set(TerminalTools.list_tool_names())
        tool_sequence = [str(item) for item in (plan.get("tool_sequence") or []) if item]
        effective_max_tools = self._effective_scientist_tool_budget(plan, max_tools)
        executed: list[dict[str, Any]] = [{
            "tool": "scientist_turn_plan",
            "ok": bool(plan.get("ok", True)),
            "artifact_path": plan_artifact,
            "message": f"autonomy={plan.get('autonomy_level', 'unknown')}",
        }]
        rendered_sections = ["\n".join(render_scientist_turn_plan_summary(plan))]
        artifacts = [plan_artifact] if plan_artifact else []

        emitter.emit("Scientist context", "building per-turn context packet", status="running")
        context_packet = TerminalTools.dispatch("scientist_context_packet", session, root)
        context_ok = bool(context_packet.get("ok", True))
        context_artifact = str(context_packet.get("artifact_path") or "")
        context_message = (
            f"quality={((context_packet.get('context_quality') or {}) if isinstance(context_packet.get('context_quality'), dict) else {}).get('score', 'n/a')}; "
            f"next={context_packet.get('next_safe_command') or '(none)'}"
        )
        self._ledger.record("scientist_context_packet", context_packet, ok=context_ok, summary=context_message)
        emitter.emit(
            "Scientist context",
            context_message,
            status="passed" if context_ok else "blocked",
            artifact=context_artifact or None,
        )
        executed.append({
            "tool": "scientist_context_packet",
            "ok": context_ok,
            "artifact_path": context_artifact,
            "message": context_message,
        })
        if context_artifact:
            artifacts.append(context_artifact)
        if context_packet.get("markdown_artifact_path"):
            artifacts.append(str(context_packet.get("markdown_artifact_path")))
        rendered_sections.append("\n".join(render_scientist_context_packet_summary(context_packet)))

        for tool_name in tool_sequence:
            if tool_name in {"scientist_turn_plan", "scientist_context_packet"}:
                continue
            budgeted_executed = [
                item for item in executed
                if item.get("tool") not in {"scientist_turn_plan", "scientist_context_packet"}
            ]
            if len(budgeted_executed) >= effective_max_tools:
                break
            if tool_name not in safe_tools:
                executed.append({
                    "tool": tool_name,
                    "ok": False,
                    "message": "not in safe terminal tool registry",
                })
                continue

            emitter.emit("Scientist tool", f"calling {tool_name}", status="running")
            try:
                record_scientist_step_event(root, {
                    "trace_run_id": f"terminal_turn_{generated_at.replace(':', '').replace('+', 'Z')}",
                    "source": "terminal_scientist_turn",
                    "task": session.selected_task or "",
                    "phase": "terminal_tool_started",
                    "status": "running",
                    "tool": tool_name,
                    "message": f"calling {tool_name}",
                    "no_training_started": True,
                })
            except Exception:
                pass

            result = TerminalTools.dispatch(tool_name, session, root)
            ok = bool(result.get("ok", True))
            artifact = str(result.get("artifact_path") or "")
            message = str(result.get("message") or result.get("mode") or result.get("status") or "")[:260]
            self._ledger.record(tool_name, result, ok=ok, summary=message)
            emitter.emit(
                "Scientist tool",
                f"{tool_name} {'completed' if ok else 'blocked'}",
                status="passed" if ok else "blocked",
                artifact=artifact or None,
            )
            try:
                record_scientist_step_event(root, {
                    "trace_run_id": f"terminal_turn_{generated_at.replace(':', '').replace('+', 'Z')}",
                    "source": "terminal_scientist_turn",
                    "task": session.selected_task or "",
                    "phase": "terminal_tool_completed" if ok else "terminal_tool_blocked",
                    "status": "passed" if ok else "blocked",
                    "tool": tool_name,
                    "message": message or f"{tool_name} completed",
                    "artifact_path": artifact,
                    "no_training_started": True,
                })
            except Exception:
                pass

            executed.append({
                "tool": tool_name,
                "ok": ok,
                "artifact_path": artifact,
                "message": message,
            })
            if artifact:
                artifacts.append(artifact)
            rendered_sections.append("\n".join(self._render_scientist_tool_summary(tool_name, result)))

        readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
        blockers = [str(item) for item in (readiness.get("blocking_gates") or [])] if readiness else []
        executed_tool_names = {
            str(item.get("tool") or "")
            for item in executed
            if item.get("tool") and item.get("tool") != "scientist_turn_plan"
        }
        deferred_tools = [
            tool
            for tool in tool_sequence
            if tool and tool != "scientist_turn_plan" and tool not in executed_tool_names
        ]
        plan_tool_budget = plan.get("tool_budget") if isinstance(plan.get("tool_budget"), dict) else {}
        must_run_tools = [str(tool) for tool in (plan_tool_budget.get("must_run_tools") or []) if tool]
        must_run_tool_set = set(must_run_tools)
        must_run_deferred_tools = [tool for tool in deferred_tools if tool in must_run_tool_set]
        terminal_tool_budget = {
            **plan_tool_budget,
            "requested_max_tools": max_tools,
            "effective_max_tools": effective_max_tools,
            "executed_tool_count": len([tool for tool in executed_tool_names if tool != "scientist_context_packet"]),
            "context_packet_auto_executed": "scientist_context_packet" in executed_tool_names,
            "must_run_deferred_count": len(must_run_deferred_tools),
        }
        try:
            recommended_budget = int(terminal_tool_budget.get("recommended_min_tools") or effective_max_tools or 4)
        except (TypeError, ValueError):
            recommended_budget = effective_max_tools or 4
        continuation_path = root / ".xsci" / "scientist_continuation.json"
        continuation_status = "needs_more_tools" if must_run_deferred_tools else "closed"
        continuation_payload = {
            "schema": "evomind.ai_scientist.continuation.v1",
            "tool": "scientist_continuation",
            "generated_at": generated_at,
            "selected_task": session.selected_task or "",
            "status": continuation_status,
            "reason": (
                "The Scientist turn hit an explicit or bounded tool budget before all must-run read-only tools completed."
                if must_run_deferred_tools else
                "All must-run read-only tools completed in the current Scientist turn."
            ),
            "requested_max_tools": max_tools,
            "effective_max_tools": effective_max_tools,
            "recommended_min_tools": recommended_budget,
            "explicit_user_budget_cap": bool(max_tools < 4),
            "executed_tools": sorted(executed_tool_names),
            "deferred_tools": deferred_tools,
            "must_run_deferred_tools": must_run_deferred_tools,
            "remaining_safe_tools": must_run_deferred_tools,
            "safe_next_command": (
                _scientist_continuation_command(raw, max(recommended_budget, len(tool_sequence) or recommended_budget))
                if must_run_deferred_tools else ""
            ),
            "action_queue_hint": [
                {
                    "id": f"continue_required_tool_{index + 1}",
                    "title": f"Continue required read-only Scientist tool: {tool}",
                    "status": "ready",
                    "command": f"evomind {tool.replace('scientist_', '').replace('_', '-')}",
                    "safe_tool": tool,
                    "gate": "continuation_read_only_gate",
                    "autonomy": "read_only",
                    "no_training_started": True,
                }
                for index, tool in enumerate(must_run_deferred_tools)
            ],
            "artifact_path": str(continuation_path),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
        try:
            continuation_path.parent.mkdir(parents=True, exist_ok=True)
            continuation_path.write_text(json.dumps(continuation_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts.append(str(continuation_path))
        except OSError:
            pass
        parity_lifecycle = plan.get("parity_lifecycle") if isinstance(plan.get("parity_lifecycle"), dict) else {}
        updated_phases = []
        for item in parity_lifecycle.get("phases") or []:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase") or "")
            updated = dict(item)
            if phase in {"observe", "plan"}:
                updated["status"] = "passed" if plan.get("ok", True) else "blocked"
            elif phase == "act":
                updated["status"] = "passed" if executed_tool_names else "blocked"
                updated["executed_tools"] = sorted(executed_tool_names)
                updated["deferred_tools"] = deferred_tools
            elif phase == "reflect":
                updated["status"] = "passed" if plan.get("scientific_critique") else "blocked"
                updated["budget_exhausted"] = bool(must_run_deferred_tools)
            elif phase == "improve":
                updated["status"] = "needs_more_tools" if must_run_deferred_tools else "passed"
                updated["next_safe_command"] = plan.get("next_safe_command")
                updated["improvement_record"] = {
                        "must_run_deferred_tools": must_run_deferred_tools,
                        "deferred_tools": deferred_tools,
                        "next_safe_command": plan.get("next_safe_command"),
                        "continuation_artifact_path": str(continuation_path),
                        "continuation_safe_next_command": continuation_payload.get("safe_next_command", ""),
                        "lesson": (
                            "Tool budget was too small for the planned Scientist turn; rerun with more tools."
                            if must_run_deferred_tools else
                        "Observe-plan-act-reflect-improve loop completed within the safe terminal budget."
                    ),
                }
            updated_phases.append(updated)
        parity_lifecycle = {
            **parity_lifecycle,
            "phases": updated_phases,
            "phase_status": {str(item.get("phase")): str(item.get("status")) for item in updated_phases if isinstance(item, dict)},
            "executed_tools": sorted(executed_tool_names),
            "deferred_tools": deferred_tools,
            "must_run_deferred_tools": must_run_deferred_tools,
            "budget_exhausted": bool(must_run_deferred_tools),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
        parity_loop_artifact = str(root / ".xsci" / "scientist_parity_loop.jsonl")
        artifacts.append(parity_loop_artifact)
        resolved_requirement_ledger = _resolve_requirement_ledger_after_tools(
            plan.get("requirement_ledger") if isinstance(plan.get("requirement_ledger"), dict) else {},
            executed=executed,
            artifacts=artifacts,
            blockers=blockers,
            no_training_started=True,
            official_submit="blocked_until_explicit_human_approval",
        )
        if resolved_requirement_ledger:
            for phase_item in parity_lifecycle.get("phases") or []:
                if not isinstance(phase_item, dict):
                    continue
                if phase_item.get("phase") == "observe":
                    evidence = phase_item.get("evidence") if isinstance(phase_item.get("evidence"), dict) else {}
                    phase_item["evidence"] = {
                        **evidence,
                        "open_requirements": resolved_requirement_ledger.get("open_requirements", [])[:12],
                        "blocked_requirements": resolved_requirement_ledger.get("blocked_requirements", [])[:12],
                    }
                elif phase_item.get("phase") == "improve":
                    improvement = phase_item.get("improvement_record") if isinstance(phase_item.get("improvement_record"), dict) else {}
                    phase_item["improvement_record"] = {
                        **improvement,
                        "requirement_ledger_resolution": resolved_requirement_ledger.get("resolution", {}),
                    }
        artifact_path = root / ".xsci" / "scientist_terminal_turn.json"
        payload = {
            "ok": True,
            "tool": "scientist_terminal_turn",
            "generated_at": generated_at,
            "selected_task": session.selected_task or "",
            "user_goal": raw,
            "plan_artifact_path": plan_artifact,
            "autonomy_level": plan.get("autonomy_level"),
            "tool_sequence": tool_sequence,
            "executed_tools": executed,
            "scientific_critique": plan.get("scientific_critique") if isinstance(plan.get("scientific_critique"), dict) else {},
            "requirement_ledger": resolved_requirement_ledger,
            "tool_budget": terminal_tool_budget,
            "continuation": continuation_payload,
            "continuation_artifact_path": str(continuation_path),
            "deferred_tools": deferred_tools,
            "must_run_deferred_tools": must_run_deferred_tools,
            "budget_exhausted": bool(must_run_deferred_tools),
            "parity_lifecycle": parity_lifecycle,
            "parity_loop_artifact": parity_loop_artifact,
            "next_safe_command": plan.get("next_safe_command"),
            "stop_conditions": plan.get("stop_conditions", []),
            "execution_ready": not blockers,
            "execution_blocked": bool(blockers),
            "blocking_gates": blockers,
            "artifacts": list(dict.fromkeys([path for path in artifacts if path])),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["artifact_path"] = str(artifact_path)
            artifacts.append(str(artifact_path))
        except OSError as exc:
            payload["ok"] = False
            payload["message"] = f"Could not write scientist terminal turn artifact: {exc}"

        summary_plan = {
            **plan,
            "requirement_ledger": resolved_requirement_ledger,
        }
        summary = self._build_scientist_turn_summary(raw, session, summary_plan, executed, artifacts, parity_lifecycle)
        try:
            record_scientist_turn(root, {
                "task": session.selected_task or "",
                "route": "scientist_terminal_turn",
                "user": raw,
                "forced_tools": ["scientist_turn_plan", *tool_sequence],
                "executed_tools": executed,
                "mode": str(plan.get("autonomy_level") or ""),
                "decision": {
                    "next_safe_command": plan.get("next_safe_command"),
                    "tool_sequence": tool_sequence,
                    "scientific_critique": plan.get("scientific_critique") if isinstance(plan.get("scientific_critique"), dict) else {},
                    "requirement_ledger": resolved_requirement_ledger,
                    "tool_budget": terminal_tool_budget,
                    "deferred_tools": deferred_tools,
                    "must_run_deferred_tools": must_run_deferred_tools,
                    "budget_exhausted": bool(must_run_deferred_tools),
                    "continuation": continuation_payload,
                },
                "blockers": (plan.get("readiness") or {}).get("blocking_gates", [])
                if isinstance(plan.get("readiness"), dict) else [],
                "next_actions": [str(plan.get("next_safe_command"))] if plan.get("next_safe_command") else [],
                "artifacts": list(dict.fromkeys([path for path in artifacts if path])),
                "parity_lifecycle": parity_lifecycle,
                "continuation": continuation_payload,
                "answer_preview": summary,
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            })
        except Exception:
            pass
        try:
            record_scientist_parity_loop(root, {
                "task": session.selected_task or "",
                "route": "scientist_terminal_turn",
                "goal": raw,
                "lifecycle": parity_lifecycle,
                "phase_status": parity_lifecycle.get("phase_status") if isinstance(parity_lifecycle, dict) else {},
                "executed_tools": executed,
                "deferred_tools": deferred_tools,
                "next_safe_command": plan.get("next_safe_command"),
                "improvement_record": (
                    (next(
                        (item.get("improvement_record") for item in updated_phases if isinstance(item, dict) and item.get("phase") == "improve"),
                        {},
                    ))
                ),
                "artifacts": list(dict.fromkeys([path for path in artifacts if path])),
                "no_training_started": True,
            })
        except Exception:
            pass
        try:
            record_scientist_step_event(root, {
                "trace_run_id": f"terminal_turn_{generated_at.replace(':', '').replace('+', 'Z')}",
                "source": "terminal_scientist_turn",
                "task": session.selected_task or "",
                "phase": "terminal_turn_complete",
                "status": "passed" if payload.get("ok", True) else "blocked",
                "tool": "scientist_terminal_turn",
                "message": f"executed_tools={len(executed)}; next={plan.get('next_safe_command')}",
                "artifact_path": str(artifact_path),
                "no_training_started": True,
            })
        except Exception:
            pass
        self._guard.record_tool(f"scientist_turn: tools={len(executed)} next={plan.get('next_safe_command')}")
        self._guard.emit(session, event="PostScientistTurn")
        return TerminalResult(
            rc=0 if payload.get("ok", True) else 1,
            should_exit=False,
            selected_task=session.selected_task,
            action="scientist_turn",
            summary=summary + "\n\n" + "\n\n".join(rendered_sections[:3]),
            artifacts=list(dict.fromkeys([path for path in artifacts if path])),
            blocked=not payload.get("ok", True),
        )

    def _effective_scientist_tool_budget(self, plan: dict[str, Any], requested_max_tools: int) -> int:
        """Expand default Scientist turns enough to close critique gaps.

        User-supplied small budgets are respected: values below 4 are treated as
        an intentional cap.  The default/UI budget of 4 may expand up to 8 when
        the turn plan says critical critique tools would otherwise be skipped.
        """
        try:
            requested = max(1, min(8, int(requested_max_tools)))
        except (TypeError, ValueError):
            requested = 4
        if requested < 4:
            return requested
        budget = plan.get("tool_budget") if isinstance(plan.get("tool_budget"), dict) else {}
        try:
            recommended = int(budget.get("recommended_min_tools") or requested)
        except (TypeError, ValueError):
            recommended = requested
        return max(requested, min(8, max(1, recommended)))

    def _render_scientist_tool_summary(self, tool_name: str, result: dict[str, Any]) -> list[str]:
        if tool_name == "scientist_step_trace":
            return render_scientist_step_trace_timeline(result)
        if tool_name == "scientist_turn_plan":
            return render_scientist_turn_plan_summary(result)
        if tool_name == "scientist_context_packet":
            return render_scientist_context_packet_summary(result)
        if tool_name == "scientist_continuation_status":
            return render_scientist_continuation_status_summary(result)
        if tool_name == "scientist_continuation_resume":
            return render_scientist_continuation_resume_summary(result)
        if tool_name == "scientist_situation_model":
            return render_scientist_situation_model_summary(result)
        if tool_name == "scientist_autopilot":
            return render_scientist_autopilot_summary(result)
        if tool_name == "scientist_loop":
            return render_scientist_loop_summary(result)
        if tool_name == "scientist_recovery":
            return render_scientist_recovery_summary(result)
        if tool_name == "scientist_self_audit":
            return render_scientist_self_audit_summary(result)
        if tool_name == "scientist_readiness_report":
            return render_scientist_readiness_report_summary(result)
        if tool_name == "scientist_causal_diagnosis":
            return render_scientist_causal_diagnosis_summary(result)
        if tool_name == "scientist_strategy_optimizer":
            return render_scientist_strategy_optimizer_summary(result)
        if tool_name == "scientist_patch_work_order":
            return render_scientist_patch_work_order_summary(result)
        if tool_name == "scientist_upgrade_plan":
            return render_scientist_upgrade_plan_summary(result)
        if tool_name == "scientist_self_upgrade_loop":
            return render_scientist_self_upgrade_loop_summary(result)
        if tool_name == "scientist_memory_consolidation":
            return render_scientist_memory_consolidation_summary(result)
        if tool_name == "scientist_innovation_backlog":
            return render_scientist_innovation_backlog_summary(result)
        if tool_name == "scientist_hypothesis_review":
            return render_scientist_hypothesis_review_summary(result)
        if tool_name == "scientist_experiment_blueprint":
            return render_scientist_experiment_blueprint_summary(result)
        if tool_name == "scientist_innovation_trial_feedback":
            return render_scientist_innovation_trial_feedback_summary(result)
        return render_tool_result_as_lines(result)

    def _build_scientist_turn_summary(self, raw: str, session: SessionState,
                                      plan: dict[str, Any],
                                      executed: list[dict[str, Any]],
                                      artifacts: list[str],
                                      parity_lifecycle: dict[str, Any] | None = None) -> str:
        readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
        blockers = [str(item) for item in (readiness.get("blocking_gates") or [])] if readiness else []
        ok_tools = [item for item in executed if item.get("ok")]
        blocked_tools = [item for item in executed if not item.get("ok")]
        budget = plan.get("tool_budget") if isinstance(plan.get("tool_budget"), dict) else {}
        executed_names = {
            str(item.get("tool") or "")
            for item in executed
            if item.get("tool") and item.get("tool") != "scientist_turn_plan"
        }
        deferred_tools = [
            str(tool)
            for tool in (plan.get("tool_sequence") or [])
            if str(tool) and str(tool) != "scientist_turn_plan" and str(tool) not in executed_names
        ]
        lines = [
            "[AI Scientist Turn] OK",
            f"  goal: {raw or '(empty)'}",
            f"  selected_task: {session.selected_task or '(none)'}",
            f"  autonomy_level: {plan.get('autonomy_level') or 'unknown'}",
            f"  tools_executed: {len(ok_tools)} ok / {len(blocked_tools)} blocked",
        ]
        if budget:
            lines.append(
                "  tool_budget: "
                f"recommended_min={budget.get('recommended_min_tools', 'n/a')}; "
                f"reason={budget.get('expansion_reason', 'default')}"
            )
        if blockers:
            lines.append("  scientific_judgment: execution is blocked; repair gates before any training.")
            lines.append("  blockers:")
            for item in blockers[:5]:
                lines.append(f"    - {item}")
        elif session.selected_task:
            lines.append("  scientific_judgment: evidence is sufficient for the next gated planning step, not for rank/medal claims.")
        else:
            lines.append("  scientific_judgment: select or register a competition before task-specific modeling.")
        critique = plan.get("scientific_critique") if isinstance(plan.get("scientific_critique"), dict) else {}
        if critique:
            lines.append(
                "  scientific_critique: "
                f"{critique.get('decision', 'unknown')} "
                f"(actionability={critique.get('actionability_score', 'n/a')})"
            )
            gaps = critique.get("evidence_gaps") if isinstance(critique.get("evidence_gaps"), list) else []
            if gaps:
                lines.append("  evidence_gaps:")
                for gap in gaps[:4]:
                    if isinstance(gap, dict):
                        lines.append(
                            "    - "
                            f"{gap.get('severity', 'unknown')}: "
                            f"{gap.get('gap', '')} "
                            f"-> {gap.get('suggested_tool', '')}"
                        )
        requirement_ledger = plan.get("requirement_ledger") if isinstance(plan.get("requirement_ledger"), dict) else {}
        open_requirements = [
            str(item)
            for item in (requirement_ledger.get("open_requirements") or [])
            if str(item)
        ]
        blocked_requirements = [
            str(item)
            for item in (requirement_ledger.get("blocked_requirements") or [])
            if str(item)
        ]
        if requirement_ledger:
            lines.append(
                "  requirement_ledger: "
                f"open={len(open_requirements)}; blocked={len(blocked_requirements)}"
            )
            if blocked_requirements:
                lines.append("  blocked_requirements:")
                for item in blocked_requirements[:5]:
                    lines.append(f"    - {item}")
            elif open_requirements:
                lines.append("  open_requirements:")
                for item in open_requirements[:5]:
                    lines.append(f"    - {item}")
        if plan.get("next_safe_command"):
            lines.append(f"  next_safe_command: {plan.get('next_safe_command')}")
        parity = parity_lifecycle if isinstance(parity_lifecycle, dict) else (
            plan.get("parity_lifecycle") if isinstance(plan.get("parity_lifecycle"), dict) else {}
        )
        phases = parity.get("phases") if isinstance(parity.get("phases"), list) else []
        if phases:
            lines.append("  parity_lifecycle:")
            for item in phases[:5]:
                if isinstance(item, dict):
                    lines.append(f"    - {item.get('phase')}: {item.get('status', 'planned')}")
        if isinstance(parity, dict) and parity.get("budget_exhausted"):
            improve_record = next(
                (
                    item.get("improvement_record")
                    for item in phases
                    if isinstance(item, dict) and item.get("phase") == "improve" and isinstance(item.get("improvement_record"), dict)
                ),
                {},
            )
            lines.append("  continuation_required: True")
            if isinstance(improve_record, dict) and improve_record.get("continuation_safe_next_command"):
                lines.append(f"  continuation_safe_next_command: {improve_record.get('continuation_safe_next_command')}")
        if deferred_tools:
            lines.append("  deferred_tools:")
            for tool in deferred_tools[:5]:
                lines.append(f"    - {tool}")
        if artifacts:
            lines.append("  artifacts:")
            for path in list(dict.fromkeys([path for path in artifacts if path]))[:8]:
                lines.append(f"    - {path}")
        lines.append("  no_training_started: True")
        lines.append("  official_submit: blocked_until_explicit_human_approval")
        return "\n".join(lines)

    # ── Private handlers ───────────────────────────────────────────────

    def _empty_turn(self, session: SessionState) -> TerminalResult:
        if session.selected_task:
            msg = f"Current task: {session.selected_task}. Describe your research goal or type /help."
        else:
            msg = "No task selected. Use `competitions` to browse, or `task add <url>` to register one. Type /help."
        return TerminalResult(rc=0, should_exit=False, action="chat",
                              summary=msg, selected_task=session.selected_task)

    def _handle_tool_query(self, intent, session: SessionState, root: Path) -> TerminalResult:
        """Handle deterministic tool queries (no LLM needed)."""
        payload = intent.payload or ""
        emitter = self._get_emitter(root)

        # Map intent payload to tool name
        tool_map = {
            "model_status": "model_status",
            "tool_status": "tool_list",
            "task_list": "task_list",
            "data_check": "data_check",
            "recent_run": "recent_run",
            "progress": "recent_run",
            "gpu_status": "gpu_status",
            "kaggle_status": "kaggle_status",
            "system_status": "system_status",
            "evolution_status": "evolution_status",
            "scientist_checkpoint": "scientist_checkpoint",
            "scientist_context_packet": "scientist_context_packet",
            "research_decision": "research_decision",
            "scientist_workplan": "scientist_workplan",
            "scientist_turn_plan": "scientist_turn_plan",
            "scientist_repair_plan": "scientist_repair_plan",
            "scientist_execution_contract": "scientist_execution_contract",
            "scientist_step_trace": "scientist_step_trace",
            "scientist_recovery": "scientist_recovery",
            "scientist_action_queue": "scientist_action_queue",
            "scientist_continuation_status": "scientist_continuation_status",
            "scientist_continuation_resume": "scientist_continuation_resume",
            "scientist_next_action": "scientist_next_action",
            "scientist_autopilot": "scientist_autopilot",
            "scientist_loop": "scientist_loop",
            "scientist_self_audit": "scientist_self_audit",
            "scientist_readiness_report": "scientist_readiness_report",
            "scientist_causal_diagnosis": "scientist_causal_diagnosis",
            "scientist_strategy_optimizer": "scientist_strategy_optimizer",
            "scientist_upgrade_plan": "scientist_upgrade_plan",
            "scientist_self_upgrade_loop": "scientist_self_upgrade_loop",
            "scientist_patch_work_order": "scientist_patch_work_order",
            "scientist_memory_consolidation": "scientist_memory_consolidation",
            "scientist_innovation_backlog": "scientist_innovation_backlog",
            "scientist_situation_model": "scientist_situation_model",
        }
        tool_name = tool_map.get(payload, payload) if payload else "system_status"

        # Special: "tool_status" → list available tools, not a single tool result
        if tool_name == "tool_list" or payload in ("tool_status",):
            return self._show_tool_list(session, root)

        # Special: "task_list" needs session aware rendering
        if tool_name == "task_list":
            return self._show_task_list(session, root)

        # Special: Scientist autopilot streams every internal tool call.
        if tool_name == "scientist_autopilot":
            result = run_scientist_autopilot(
                session,
                root,
                observer=lambda event: self._emit_scientist_autopilot_event(emitter, event),
            )
            lines = render_scientist_autopilot_summary(result)
            artifacts = [
                str(path)
                for path in [
                    result.get("artifact_path"),
                    result.get("action_queue_artifact_path"),
                    (result.get("workplan") or {}).get("artifact_path") if isinstance(result.get("workplan"), dict) else None,
                    (result.get("repair_plan") or {}).get("artifact_path") if isinstance(result.get("repair_plan"), dict) else None,
                    (result.get("execution_contract") or {}).get("artifact_path") if isinstance(result.get("execution_contract"), dict) else None,
                    (result.get("step_trace") or {}).get("artifact_path") if isinstance(result.get("step_trace"), dict) else None,
                ]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        # Special: Scientist loop runs bounded safe next-action cycles and
        # streams the loop phases.
        if tool_name == "scientist_loop":
            result = run_scientist_loop(
                session,
                root,
                observer=lambda event: self._emit_scientist_loop_event(emitter, event),
            )
            lines = render_scientist_loop_summary(result)
            artifacts = [
                str(path)
                for path in [
                    result.get("artifact_path"),
                    result.get("lessons_path"),
                    (result.get("final_autopilot") or {}).get("artifact_path") if isinstance(result.get("final_autopilot"), dict) else None,
                    (result.get("final_autopilot") or {}).get("action_queue_artifact_path") if isinstance(result.get("final_autopilot"), dict) else None,
                ]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        # Special: Scientist recovery builds a compact restart/compaction
        # snapshot from guard + turn ledger + trace + latest plan artifacts.
        if tool_name == "scientist_recovery":
            emitter.emit("Recovery", "building Scientist recovery snapshot", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Recovery",
                "recovery snapshot ready" if result.get("ok", True) else "recovery snapshot blocked",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_recovery_summary(result)
            artifacts = [
                str(path)
                for path in [
                    result.get("artifact_path"),
                    result.get("guard_path"),
                    result.get("latest_workplan_artifact"),
                    result.get("latest_repair_artifact"),
                    result.get("latest_contract_artifact"),
                    result.get("action_queue_artifact"),
                ]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        # Special: Scientist self-audit scores EvoMind itself and writes a
        # system-upgrade backlog without starting training.
        if tool_name == "scientist_self_audit":
            emitter.emit("Self audit", "auditing EvoMind agent capability", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Self audit",
                f"capability score={result.get('overall_score', 0)}; backlog ready",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_self_audit_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("backlog_artifact_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_readiness_report":
            emitter.emit("Readiness report", "building unified Scientist capability, gate, and claim report", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            claim = result.get("claim_readiness") if isinstance(result.get("claim_readiness"), dict) else {}
            emitter.emit(
                "Readiness report",
                f"launch={result.get('launch_readiness', 'unknown')}; training={claim.get('training_readiness_claim', 'unknown')}",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_readiness_report_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("markdown_artifact_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_causal_diagnosis":
            emitter.emit("Causal diagnosis", "linking symptoms, root causes, evidence, and safe interventions", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Causal diagnosis",
                f"posture={result.get('posture', 'unknown')}; next={result.get('next_safe_command', 'evomind autopilot')}",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_causal_diagnosis_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("markdown_artifact_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_context_packet":
            emitter.emit("Context packet", "compacting task, gates, memory, strategy, and requirements", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            quality = result.get("context_quality") if isinstance(result.get("context_quality"), dict) else {}
            emitter.emit(
                "Context packet",
                f"quality={quality.get('score', 'n/a')}; next={result.get('next_safe_command', 'evomind strategy')}",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_context_packet_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("markdown_artifact_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_strategy_optimizer":
            emitter.emit("Strategy optimizer", "ranking safe interventions by impact, evidence, cost, risk, and gate status", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            selected = result.get("selected_strategy") if isinstance(result.get("selected_strategy"), dict) else {}
            emitter.emit(
                "Strategy optimizer",
                f"selected={selected.get('id', 'none')}; next={result.get('next_safe_command', 'evomind causal-diagnosis')}",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_strategy_optimizer_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("markdown_artifact_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_upgrade_plan":
            emitter.emit("Upgrade plan", "planning self-audit backlog closure", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Upgrade plan",
                f"planned_steps={result.get('open_backlog_count', 0)}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_upgrade_plan_summary(result)
            artifacts = [
                str(path)
                for path in [
                    result.get("artifact_path"),
                    result.get("source_backlog_path"),
                    result.get("source_self_audit_path"),
                ]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_self_upgrade_loop":
            emitter.emit("Self-upgrade loop", "creating self-upgrade work order", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Self-upgrade loop",
                f"selected={result.get('selected_backlog_id') or 'none'}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_self_upgrade_loop_summary(result)
            artifacts = [
                str(path)
                for path in [
                    result.get("artifact_path"),
                    result.get("work_order_path"),
                    result.get("trials_path"),
                    result.get("source_upgrade_plan_path"),
                    result.get("source_self_audit_path"),
                ]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_memory_consolidation":
            emitter.emit("Memory consolidation", "writing safe lessons into retrospective memory", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Memory consolidation",
                f"records_added={result.get('records_added', 0)}; records_total={result.get('records_total', 0)}",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_memory_consolidation_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("memory_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        # Special: Scientist innovation backlog mines memory and writes
        # proposal-only hypotheses. It does not start training.
        if tool_name == "scientist_innovation_backlog":
            emitter.emit("Innovation backlog", "mining memory for proposal-only hypotheses", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            hypotheses = result.get("innovation_hypotheses") or []
            emitter.emit(
                "Innovation backlog",
                f"hypotheses={len(hypotheses) if isinstance(hypotheses, list) else 0}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_innovation_backlog_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("innovation_log_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_hypothesis_review":
            emitter.emit("Hypothesis review", "ranking proposal-only hypotheses against gates", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            selected = result.get("selected_hypothesis") if isinstance(result.get("selected_hypothesis"), dict) else {}
            emitter.emit(
                "Hypothesis review",
                f"reviewed={result.get('hypotheses_reviewed', 0)}; selected={selected.get('strategy_name', 'none')}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_hypothesis_review_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("source_backlog_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_experiment_blueprint":
            emitter.emit("Experiment blueprint", "building gated plan from reviewed hypothesis", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            blueprint = result.get("experiment_blueprint") if isinstance(result.get("experiment_blueprint"), dict) else {}
            emitter.emit(
                "Experiment blueprint",
                f"status={result.get('blueprint_status', 'unknown')}; blueprint={blueprint.get('blueprint_id', 'none')}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_experiment_blueprint_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("source_review_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_innovation_trial_feedback":
            emitter.emit("Innovation feedback", "writing hypothesis and blueprint gate outcome to memory", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Innovation feedback",
                f"outcome={result.get('outcome', 'unknown')}; gate={result.get('gate_status', 'unknown')}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_innovation_trial_feedback_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("innovation_log_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_situation_model":
            emitter.emit("Situation model", "synthesizing evidence, uncertainty, blockers, strategy, and memory", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Situation model",
                f"posture={result.get('situation_status', 'unknown')}; readiness={result.get('readiness_score', 0)}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_situation_model_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), *(result.get("source_artifacts") or [])]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_turn_plan":
            emitter.emit("Turn plan", "building per-turn tool plan and stop conditions", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            intent_info = result.get("intent") if isinstance(result.get("intent"), dict) else {}
            emitter.emit(
                "Turn plan",
                f"intent={intent_info.get('kind', 'unknown')}; autonomy={result.get('autonomy_level', 'unknown')}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_turn_plan_summary(result)
            artifacts = [str(result.get("artifact_path"))] if result.get("artifact_path") else []
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=artifacts,
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_continuation_status":
            emitter.emit("Continuation", "checking incomplete Scientist turn state", status="running")
            result = TerminalTools.dispatch(tool_name, session, root)
            emitter.emit(
                "Continuation",
                f"status={result.get('status', 'unknown')}; remaining={result.get('remaining_count', 0)}; no training started",
                status="passed" if result.get("ok", True) else "blocked",
                artifact=str(result.get("artifact_path") or "") or None,
            )
            lines = render_scientist_continuation_status_summary(result)
            artifacts = [
                str(path)
                for path in [result.get("artifact_path"), result.get("continuation_artifact_path")]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        if tool_name == "scientist_continuation_resume":
            result = run_scientist_continuation_resume(
                session,
                root,
                observer=lambda event: self._emit_scientist_continuation_resume_event(emitter, event),
            )
            lines = render_scientist_continuation_resume_summary(result)
            artifacts = [
                str(path)
                for path in [
                    result.get("artifact_path"),
                    result.get("continuation_status_artifact_path"),
                    result.get("continuation_artifact_path"),
                ]
                if path
            ]
            return TerminalResult(
                rc=0 if result.get("ok", True) else 1,
                should_exit=False,
                action="tool_call",
                summary="\n".join(lines),
                selected_task=session.selected_task,
                artifacts=list(dict.fromkeys(artifacts)),
                blocked=not result.get("ok", True),
            )

        # Run the tool
        emitter.emit("Tool call", f"calling {tool_name}", status="running")
        result = TerminalTools.dispatch(tool_name, session, root)
        status = "passed" if result.get("ok") else "blocked"
        emitter.emit("Tool call", f"{tool_name} completed", status=status)

        lines = render_tool_result_as_lines(result)
        return TerminalResult(
            rc=0, should_exit=False, action="tool_call",
            summary="\n".join(lines),
            selected_task=session.selected_task,
            blocked=not result.get("ok"),
        )

    def _emit_scientist_autopilot_event(self, emitter: TerminalEventEmitter, event: dict) -> None:
        """Render one autopilot event as a Claude-Code-like terminal stage."""
        stage_labels = {
            "system_status": "Observe system",
            "inspect_task": "Inspect task",
            "data_check": "Check data",
            "recent_run": "Read latest run",
            "evolution_status": "Read memory",
            "scientist_checkpoint": "Checkpoint",
            "research_decision": "Choose branch",
            "scientist_workplan": "Build workplan",
            "scientist_repair_plan": "Repair plan",
            "scientist_execution_contract": "Execution contract",
        }
        phase = str(event.get("phase") or "")
        tool = str(event.get("tool") or "")
        if phase == "autopilot_start":
            emitter.emit(
                "AI Scientist",
                "starting bounded multi-tool diagnosis; no training or Kaggle submit will start",
                status="running",
            )
            return
        if phase == "tool_started":
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            confidence = details.get("confidence")
            rationale = str(details.get("rationale") or "calling tool")
            if isinstance(confidence, (int, float)):
                rationale = f"{rationale} confidence={float(confidence):.2f}"
            emitter.emit(stage_labels.get(tool, tool), rationale[:260], status="running")
            return
        if phase in {"tool_completed", "tool_blocked"}:
            status = "passed" if phase == "tool_completed" else "blocked"
            message = str(event.get("message") or "completed").replace("\n", " ")[:220]
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            confidence = details.get("confidence")
            if isinstance(confidence, (int, float)):
                message = f"{message} confidence={float(confidence):.2f}"
            artifact = str(event.get("artifact_path") or "") or None
            emitter.emit(stage_labels.get(tool, tool), message, status=status, artifact=artifact)
            return
        if phase == "autopilot_complete":
            status = "passed" if event.get("status") == "completed" else "blocked"
            artifact = str(event.get("artifact_path") or "") or None
            emitter.emit("AI Scientist", "diagnosis complete; artifacts persisted", status=status, artifact=artifact)

    def _emit_scientist_continuation_resume_event(self, emitter: TerminalEventEmitter, event: dict) -> None:
        """Render one continuation resume event as a staged terminal update."""
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "running")
        message = str(event.get("message") or "").replace("\n", " ")[:260]
        artifact = str(event.get("artifact_path") or "") or None
        stage = {
            "continuation_resume_start": "Continuation resume",
            "continuation_resume_step_started": "Safe continuation step",
            "continuation_resume_step_completed": "Safe continuation step",
            "continuation_resume_complete": "Continuation resume",
        }.get(phase, "Continuation resume")
        rendered_status = (
            "passed"
            if status in {"passed", "closed", "completed"}
            else "blocked"
            if status in {"blocked", "blocked_by_gate", "stalled"}
            else "running"
        )
        emitter.emit(stage, message or phase, status=rendered_status, artifact=artifact)

    def _emit_scientist_loop_event(self, emitter: TerminalEventEmitter, event: dict) -> None:
        """Render one bounded autonomous loop event."""
        phase = str(event.get("phase") or "")
        status = str(event.get("status") or "running")
        message = str(event.get("message") or "").replace("\n", " ")[:260]
        artifact = str(event.get("artifact_path") or "") or None
        stage = {
            "loop_start": "AI Scientist loop",
            "loop_observe": "Observe and decide",
            "loop_next_action": "Safe next action",
            "loop_refresh": "Refresh evidence",
            "loop_repetition_escalation": "Escalate repeated action",
            "loop_complete": "Learn and stop",
        }.get(phase, "AI Scientist loop")
        rendered_status = "passed" if status in {"passed", "completed"} else "blocked" if status == "blocked" else "running"
        emitter.emit(stage, message or phase, status=rendered_status, artifact=artifact)

    def _show_tool_list(self, session: SessionState, root: Path) -> TerminalResult:
        """Show available terminal tools."""
        names = TerminalTools.list_tool_names()
        tool_descriptions = {
            "model_status": "当前 LLM 模型、provider、就绪状态",
            "system_status": "完整系统就绪状态（LLM/Kaggle/GPU/任务）",
            "task_list": "已注册的任务列表",
            "inspect_task": "当前选中任务的详细信息",
            "data_check": "检查数据文件是否就绪",
            "recent_run": "最近一次训练的结果",
            "gpu_status": "GPU/HPC 配置和阻塞状态",
            "kaggle_status": "Kaggle API 配置状态",
            "dashboard": "打开工作站面板",
            "next_steps": "下一步应该做什么",
            "evolution_status": "自进化统计、经验记忆和创新尝试证据",
            "scientist_checkpoint": "AI Scientist 研究检查点（观察/分析/提案/门禁/行动）",
            "research_decision": "下一轮实验决策 artifact（branch/code mode/gate/rollback）",
            "scientist_workplan": "多步 AI Scientist 工作计划（步骤/门禁/证据/恢复命令）",
            "scientist_repair_plan": "自我修复计划（阻塞归因/root cause/修复步骤/safe next command）",
            "scientist_execution_contract": "执行前合同（go/no-go、branch、rollback、required artifacts）",
            "scientist_step_trace": "步骤级工具轨迹（每次工具调用、门禁、证据文件）",
            "scientist_recovery": "长程恢复快照（guard/turn ledger/step trace/下一步命令）",
            "scientist_action_queue": "AI Scientist 行动队列（下一步命令、门禁、风险、证据和回滚）",
            "scientist_next_action": "执行下一个安全只读动作；训练/提交/下载会停在门禁",
            "scientist_continuation_resume": "自动续跑上轮未完成的安全只读工具，直到闭环、门禁或停滞",
            "scientist_autopilot": "多工具 AI Scientist 自动诊断链（状态/数据/记忆/决策/下一步）",
            "scientist_loop": "有界 AI Scientist 自主循环（诊断→安全下一步→复盘→经验 artifact）",
            "scientist_readiness_report": "统一输出能力、执行门禁、声明边界和下一步命令的上线就绪报告",
            "scientist_causal_diagnosis": "把症状、根因、证据和安全干预动作组织成可复盘因果图",
            "scientist_strategy_optimizer": "把根因、行动队列和就绪报告转成按影响/证据/成本/风险排序的下一步策略",
            "scientist_context_packet": "生成每轮 AI Scientist 上下文包，汇总任务、门禁、记忆、策略和下一步命令",
            "scientist_upgrade_plan": "把自审计升级 backlog 转成工程计划（文件/验收/门禁）",
            "scientist_self_upgrade_loop": "把最高优先级能力缺口转成可审计自升级工单；不改代码、不训练、不提交 Kaggle",
            "scientist_patch_work_order": "把最近失败/阻塞证据转成代码 Agent 补丁工单；不自动改代码、不训练、不提交 Kaggle",
        }
        lines = ["EvoMind 可用终端工具："]
        for name in names:
            desc = tool_descriptions.get(name, "")
            lines.append(f"  • {name}" + (f" — {desc}" if desc else ""))
        lines.append("")
        lines.append("你也可以直接描述需求，例如：“检查数据”、“最近训练结果怎么样”、“现在用的什么模型”、“它有没有学到经验”、“全面诊断下一步怎么做”。")
        return TerminalResult(
            rc=0, should_exit=False, action="tool_call",
            summary="\n".join(lines), selected_task=session.selected_task,
        )

    def _show_task_list(self, session: SessionState, root: Path) -> TerminalResult:
        """Show registered tasks in a friendly format."""
        result = TerminalTools.dispatch("task_list", session, root)
        lines = []
        tasks = result.get("tasks", [])
        if not tasks:
            lines.append("还没有注册任何比赛。")
            lines.append("运行 `competitions` 浏览 Kaggle 比赛，然后用 `task add <url>` 注册。")
        else:
            lines.append(f"已注册 {len(tasks)} 个任务：")
            for t in tasks:
                mark = "→" if t["slug"] == session.selected_task else " "
                lines.append(f"  {mark} {t['slug']}" + (f"  ({t['brief']})" if t.get("brief") else ""))
        return TerminalResult(
            rc=0, should_exit=False, action="tool_call",
            summary="\n".join(lines), selected_task=session.selected_task,
        )

    def _handle_execution(self, raw: str, intent, session: SessionState,
                          root: Path) -> TerminalResult:
        """Handle execution intent: check gates, run preflight, maybe start training.

        Returns ``blocked=True`` if gates prevent execution; the caller (kaggle.py)
        will then print the blocker message instead of calling _run_agent.
        """
        # Detect compute override from natural language
        from .kaggle import _infer_compute_override
        compute = _infer_compute_override(raw)

        # Resume flag
        resume = (intent.payload == "resume" or
                  any(w in raw.lower() for w in ("继续", "接着", "resume", "continue")))

        if not session.selected_task:
            return TerminalResult(
                rc=1, should_exit=False, action="training",
                selected_task=session.selected_task,
                summary="还没有选中比赛，所以不会启动训练。请先用 `competitions` 浏览，或用 `task add <kaggle-url>` 注册并选择一个任务。",
                blocked=True,
            )

        # Run preflight stages with real checks
        emitter = self._get_emitter(root)
        effective_compute = compute or session.compute_backend

        # Stage 1: Inspect task
        task_result = TerminalTools.dispatch("inspect_task", session, root)
        task_ok = task_result.get("ok")
        emitter.emit("Inspecting task",
                     f"task={session.selected_task}, "
                     f"metric={task_result.get('metric', '?')}, "
                     f"modality={task_result.get('modality', '?')}",
                     status="passed" if task_ok else "blocked")
        if not task_ok:
            return TerminalResult(
                rc=1, should_exit=False, action="training",
                selected_task=session.selected_task,
                summary=f"Task inspection failed: {task_result.get('message', '')}",
                blocked=True,
            )

        # Stage 2: Check data
        data_result = TerminalTools.dispatch("data_check", session, root)
        data_ok = data_result.get("ok") and data_result.get("train_csv")
        emitter.emit("Checking data",
                     f"train.csv={'found' if data_result.get('train_csv') else 'missing'}, "
                     f"test.csv={'found' if data_result.get('test_csv') else 'missing'}",
                     status="passed" if data_ok else "blocked")

        # Stage 3: Check config
        model_result = TerminalTools.dispatch("model_status", session, root)
        model_ready = model_result.get("ready") and model_result.get("ok")
        emitter.emit("Checking config",
                     f"provider={model_result.get('provider')}, "
                     f"model={model_result.get('model')}, "
                     f"ready={'yes' if model_ready else 'no'}",
                     status="passed" if model_ready else "blocked")

        # Stage 4: Select compute
        if effective_compute == "gpu" and session.gpu_blocked:
            emitter.emit("Selecting compute", f"compute=gpu BLOCKED: {session.gpu_blocker or session.gpu_status}",
                         status="blocked")
        else:
            emitter.emit("Selecting compute",
                         f"compute={effective_compute}" +
                         (" (GPU blocker ignored — using local)" if effective_compute == "local" and session.gpu_blocked else ""),
                         status="passed")

        # Stage 5: Check gates
        blockers = session.blocking_setup(compute_override=effective_compute)
        if blockers:
            emitter.emit("Planning experiment", f"blocked: {', '.join(blockers[:3])}",
                         status="blocked")
            return TerminalResult(
                rc=1, should_exit=False, action="training",
                selected_task=session.selected_task,
                summary="Setup needed before execution:\n" + "\n".join(f"- {b}" for b in blockers),
                blocked=True,
            )
        emitter.emit("Planning experiment",
                     f"goal={raw[:80]}, resume={'yes' if resume else 'no'}",
                     status="passed")

        # Stage 6: Entering agent
        emitter.emit("Entering workstation agent",
                     f"compute={effective_compute}, events → events.jsonl, dashboard → :8088",
                     status="passed")

        # All gates clear — signal the caller to call _run_agent.
        return TerminalResult(
            rc=0, should_exit=False, action="training",
            selected_task=session.selected_task,
            summary=f"Preflight passed. Starting {effective_compute} training for {session.selected_task}.",
            artifacts=[f"compute={effective_compute}", f"resume={resume}"],
            blocked=False,
        )

    def _handle_planning(self, raw: str, session: SessionState,
                         root: Path) -> TerminalResult:
        """Handle planning intent."""
        if not session.selected_task:
            return TerminalResult(
                rc=0, should_exit=False, action="planning",
                selected_task=session.selected_task,
                summary="I can plan, but no competition is selected. Browse with `competitions` or `task add <url>` first.",
            )
        return TerminalResult(
            rc=0, should_exit=False, action="planning",
            selected_task=session.selected_task,
            summary="",  # caller runs ConversationAgent.plan()
        )
