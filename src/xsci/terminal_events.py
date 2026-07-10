"""Structured streaming events for the EvoMind research terminal.

Each event is a lightweight ``dict`` that carries a ``type`` field and a
human-readable ``message``.  The caller decides whether to print the event
immediately (terminal renderer), write it to a JSONL file (dashboard sink), or
both.

This module mirrors the CLAUDE CODE terminal experience: staged output,
tool-call visibility, immediate flush, and a durable event log for the frontend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Preflight stage names — shown BEFORE the deep agent starts.
PREFLIGHT_STAGES = (
    "Inspecting task",
    "Checking data",
    "Checking config",
    "Selecting compute",
    "Planning experiment",
    "Entering workstation agent",
)


@dataclass
class TerminalStage:
    """One discrete step in the terminal agent's execution."""
    stage: str            # e.g. "Inspecting task"
    message: str          # human-readable detail
    status: str = "running"   # "running" | "passed" | "blocked" | "failed"
    artifact: Optional[str] = None  # path to a produced file, if any


def emit_stage(stage: str, message: str, *, status: str = "running",
               artifact: Optional[str] = None) -> dict:
    """Return a structured event dict.  The caller decides what to do with it."""
    return {
        "type": "terminal_stage",
        "stage": stage,
        "message": message,
        "status": status,
        "artifact": artifact,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


class TerminalEventEmitter:
    """Fan-out emitter: prints to stdout and writes to a JSONL file.

    Usage::

        emitter = TerminalEventEmitter(root)
        emitter.emit("Inspecting task", "task=titanic, metric=accuracy")
        emitter.emit("Checking data", "local_data_dir=..., train.csv found", status="passed")
    """

    def __init__(self, workspace_root: Path, *, colour: bool = True,
                 jsonl_path: Optional[Path] = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.colour = colour
        self._jsonl_path = jsonl_path or (self.workspace_root / "terminal_events.jsonl")
        self._idx = 0

    def _ensure_jsonl(self) -> None:
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, message: str, *, status: str = "running",
             artifact: Optional[str] = None) -> dict:
        """Emit one stage event to both stdout and the JSONL sink."""
        self._idx += 1
        event = emit_stage(stage, message, status=status, artifact=artifact)
        event["seq"] = self._idx

        # Print immediately to stdout (like Claude Code's staged output).
        glyph = ">"
        if status == "passed":
            mark = "\033[92m[OK]\033[0m" if self.colour else "[OK]"
        elif status == "blocked":
            mark = "\033[93m[BLOCKED]\033[0m" if self.colour else "[BLOCKED]"
        elif status == "failed":
            mark = "\033[91m[FAIL]\033[0m" if self.colour else "[FAIL]"
        else:
            mark = "\033[96m[...]\033[0m" if self.colour else "[...]"

        try:
            print(f"  {mark} {glyph} {stage}: {message}", flush=True)
        except (UnicodeEncodeError, OSError):
            print(f"  [... ] > {stage}: {message}", flush=True)

        # Append to JSONL for the dashboard.
        try:
            self._ensure_jsonl()
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass  # never crash because of a log write failure

        return event


def render_tool_result_as_lines(result: dict) -> list[str]:
    """Turn a tool result dict into a list of display lines suitable for the
    terminal or for LLM context injection."""
    lines: list[str] = []
    ok = result.get("ok", True)
    tool = result.get("tool", "?")
    status = "OK" if ok else "BLOCKED"
    lines.append(f"[tool:{tool}] {status}")

    for key, value in result.items():
        if key in ("ok", "tool"):
            continue
        if isinstance(value, list):
            if not value:
                lines.append(f"  {key}: (empty)")
            else:
                lines.append(f"  {key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"    - " + ", ".join(
                            f"{k}={v}" for k, v in item.items() if k != "path"))
                    else:
                        lines.append(f"    - {item}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {key}: {value}")

    return lines


def _short(value: object, *, limit: int = 220) -> str:
    text = "" if value is None else str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _append_memory_reuse_plan(lines: list[str], plan: object, *, indent: str = "  ") -> None:
    if not isinstance(plan, dict) or not plan:
        return
    rules = plan.get("reuse_rules") or []
    avoids = plan.get("avoid_patterns") or []
    ids = plan.get("supporting_memory_ids") or []
    lines.append(f"{indent}memory_reuse_plan:")
    lines.append(f"{indent}  - gate: {plan.get('gate') or 'memory_reuse_before_execution'}")
    lines.append(f"{indent}  - status: {plan.get('status') or 'unknown'}")
    if ids:
        lines.append(f"{indent}  - supporting_memory_ids: " + ", ".join(_short(item, limit=40) for item in ids[:6]))
    if rules:
        lines.append(f"{indent}  reuse_rules:")
        for item in rules[:3]:
            if isinstance(item, dict):
                lines.append(f"{indent}    - {_short(item.get('strategy'), limit=140)}")
            else:
                lines.append(f"{indent}    - {_short(item, limit=140)}")
    if avoids:
        lines.append(f"{indent}  avoid_patterns:")
        for item in avoids[:3]:
            if isinstance(item, dict):
                lines.append(f"{indent}    - {_short(item.get('pattern'), limit=140)}")
            else:
                lines.append(f"{indent}    - {_short(item, limit=140)}")


def _trace_status_mark(status: str) -> str:
    normalized = (status or "info").lower()
    if normalized in {"passed", "completed", "ok", "ready"}:
        return "OK"
    if normalized in {"blocked", "failed", "error"}:
        return "BLOCKED" if normalized == "blocked" else "FAIL"
    if normalized in {"running", "pending"}:
        return "RUN"
    return "INFO"


def render_scientist_step_trace_timeline(result: dict) -> list[str]:
    """Render recent Scientist events as a Claude-Code-like timeline.

    The raw JSON remains available in ``.xsci/scientist_step_trace.jsonl``.  This
    terminal view is intentionally compact: it shows phase, tool, gate, message,
    evidence, and artifacts without dumping nested payloads or secrets.
    """
    ok = result.get("ok", True)
    events = result.get("recent") or []
    if not isinstance(events, list):
        events = []

    lines = [f"[tool:scientist_step_trace] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  events: {result.get('count', len(events))}")
    lines.append(f"  artifact: {result.get('artifact_path') or '.xsci/scientist_step_trace.jsonl'}")
    lines.append("  live_command: evomind live")

    if not events:
        lines.append("  timeline: (empty)")
        lines.append("  next_safe_command: evomind autopilot")
    else:
        lines.append("  timeline:")
        tail = events[-20:]
        start_index = max(1, len(events) - len(tail) + 1)
        for index, event in enumerate(tail, start=start_index):
            if not isinstance(event, dict):
                lines.append(f"    {index:02d}. [INFO] {_short(event)}")
                continue
            status = str(event.get("status") or "info")
            mark = _trace_status_mark(status)
            phase = _short(event.get("phase") or "step", limit=48)
            tool = _short(event.get("tool") or event.get("source") or "scientist", limit=48)
            message = _short(event.get("message") or "", limit=260)
            gate = _short(event.get("gate") or "", limit=80)
            ts = _short(event.get("ts") or "", limit=32)
            header = f"    {index:02d}. [{mark}] {phase} / {tool}"
            if gate:
                header += f" gate={gate}"
            if ts:
                header += f" @ {ts}"
            lines.append(header)
            if message:
                lines.append(f"        {message}")
            evidence = event.get("evidence") or []
            if isinstance(evidence, list) and evidence:
                lines.append("        evidence: " + ", ".join(_short(item, limit=70) for item in evidence[:5]))
            artifact = _short(event.get("artifact_path") or "", limit=180)
            if artifact:
                lines.append(f"        artifact: {artifact}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_autopilot_summary(result: dict) -> list[str]:
    """Render the multi-tool Scientist run as a compact terminal summary.

    The full nested payload is still persisted as JSON.  This view is meant for
    human terminal use: what was inspected, what the Scientist decided, what is
    blocked, and which artifacts can be audited next.
    """
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_autopilot] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  mode: {result.get('mode', 'unknown')}")
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    if result.get("trace_run_id"):
        lines.append(f"  trace_run_id: {result.get('trace_run_id')}")

    summary_lines = result.get("summary_lines") or []
    if summary_lines:
        lines.append("  summary:")
        for item in summary_lines[:8]:
            lines.append(f"    - {item}")

    tool_trace = result.get("tool_trace") or []
    lines.append("  tool_trace:")
    if tool_trace:
        for item in tool_trace:
            if not isinstance(item, dict):
                lines.append(f"    - {item}")
                continue
            mark = "OK" if item.get("ok", True) else "BLOCKED"
            tool = item.get("tool") or "?"
            msg = str(item.get("message") or "").strip()
            confidence = item.get("confidence")
            confidence_text = ""
            if isinstance(confidence, (int, float)):
                confidence_text = f" confidence={float(confidence):.2f}"
            rationale = str(item.get("rationale") or "").strip()
            signal = str(item.get("evidence_signal") or "").strip()
            lines.append(f"    - [{mark}] {tool}{confidence_text}: {msg[:180]}")
            if rationale:
                suffix = f" evidence={signal}" if signal else ""
                lines.append(f"       why: {rationale[:220]}{suffix}")
    else:
        lines.append("    - (empty)")

    selected_hypothesis = result.get("selected_hypothesis")
    if isinstance(selected_hypothesis, dict) and selected_hypothesis:
        lines.append("  selected_hypothesis:")
        lines.append(
            f"    - {selected_hypothesis.get('strategy_name') or selected_hypothesis.get('hypothesis_id') or 'unknown'} "
            f"score={selected_hypothesis.get('score', 'n/a')} "
            f"status={selected_hypothesis.get('status', 'unknown')} "
            f"branch={selected_hypothesis.get('branch_type', 'unknown')} "
            f"mode={selected_hypothesis.get('code_generation_mode', 'unknown')}"
        )

    _append_memory_reuse_plan(lines, result.get("memory_reuse_plan"), indent="  ")

    blockers = result.get("blockers") or []
    if blockers:
        lines.append("  blockers:")
        for item in blockers[:8]:
            lines.append(f"    - {item}")

    next_actions = result.get("next_actions") or []
    if next_actions:
        lines.append("  next_actions:")
        for item in next_actions[:8]:
            lines.append(f"    - {item}")

    action_queue = result.get("action_queue") or []
    if action_queue:
        lines.append("  action_queue:")
        for index, item in enumerate(action_queue[:6], start=1):
            if not isinstance(item, dict):
                lines.append(f"    {index}. {item}")
                continue
            lines.append(
                f"    {index}. [{item.get('status', 'unknown')}] "
                f"{item.get('title', 'action')}"
            )
            if item.get("command"):
                lines.append(f"       command: {item.get('command')}")
            if item.get("gate"):
                lines.append(f"       gate: {item.get('gate')}")
            if item.get("why"):
                lines.append(f"       why: {str(item.get('why'))[:180]}")

    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    if decision:
        lines.append(
            "  decision: "
            f"action={decision.get('selected_action') or 'none'}, "
            f"branch={decision.get('selected_branch') or 'none'}, "
            f"mode={decision.get('code_generation_mode') or 'none'}"
        )

    artifact_paths = [result.get("artifact_path")]
    artifact_paths.append(result.get("action_queue_artifact_path"))
    for key in ("workplan", "repair_plan", "execution_contract", "step_trace"):
        value = result.get(key)
        if isinstance(value, dict):
            artifact_paths.append(value.get("artifact_path"))
    artifact_paths = [str(path) for path in artifact_paths if path]
    if artifact_paths:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifact_paths):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_loop_summary(result: dict) -> list[str]:
    """Render the bounded autonomous Scientist loop for terminal users."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_loop] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  mode: {result.get('mode', 'unknown')}")
    lines.append(f"  stop_reason: {result.get('stop_reason', 'unknown')}")
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    if result.get("trace_run_id"):
        lines.append(f"  trace_run_id: {result.get('trace_run_id')}")

    steps = result.get("steps") or []
    lines.append("  loop_steps:")
    if steps:
        for index, step in enumerate(steps[:10], start=1):
            if not isinstance(step, dict):
                lines.append(f"    {index}. {step}")
                continue
            label = step.get("step") or step.get("tool") or "step"
            status = step.get("status") or "unknown"
            tool = step.get("tool") or ""
            lines.append(f"    {index}. [{status}] {label}" + (f" via {tool}" if tool else ""))
            if step.get("selected_action"):
                lines.append(f"       selected_action: {step.get('selected_action')}")
            if step.get("gate"):
                lines.append(f"       gate: {step.get('gate')}")
            if step.get("artifact_path"):
                lines.append(f"       artifact: {step.get('artifact_path')}")
    else:
        lines.append("    - (empty)")

    final_next = result.get("final_next_action")
    if isinstance(final_next, dict):
        selected = final_next.get("selected_action")
        lines.append("  final_next_action:")
        lines.append(f"    status: {final_next.get('status')}")
        if isinstance(selected, dict):
            lines.append(f"    id: {selected.get('id')}")
            lines.append(f"    command: {selected.get('command')}")
            lines.append(f"    gate: {selected.get('gate')}")

    lesson = result.get("lesson")
    if isinstance(lesson, dict):
        lines.append("  learned:")
        if lesson.get("lesson"):
            lines.append(f"    - {lesson.get('lesson')}")
        if lesson.get("stop_reason"):
            lines.append(f"    - stop_reason={lesson.get('stop_reason')}")

    memory = result.get("memory_consolidation")
    if isinstance(memory, dict):
        lines.append("  memory_writeback:")
        lines.append(f"    status: {'OK' if memory.get('ok', True) else 'BLOCKED'}")
        lines.append(f"    records_added: {memory.get('records_added', 0)}")
        lines.append(f"    records_total: {memory.get('records_total', 0)}")
        if memory.get("memory_path"):
            lines.append(f"    memory_path: {memory.get('memory_path')}")
        if memory.get("artifact_path"):
            lines.append(f"    artifact: {memory.get('artifact_path')}")

    artifacts = [result.get("artifact_path"), result.get("lessons_path")]
    final_auto = result.get("final_autopilot")
    if isinstance(final_auto, dict):
        artifacts.append(final_auto.get("artifact_path"))
        artifacts.append(final_auto.get("action_queue_artifact_path"))
    if isinstance(memory, dict):
        artifacts.append(memory.get("artifact_path"))
        artifacts.append(memory.get("memory_path"))
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_recovery_summary(result: dict) -> list[str]:
    """Render long-horizon recovery state without dumping the full JSON."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_recovery] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  recovery_decision: {result.get('recovery_decision') or 'unknown'}")
    if result.get("last_goal"):
        lines.append(f"  last_goal: {result.get('last_goal')}")
    lines.append(f"  recent_turns: {result.get('recent_turn_count', 0)}")
    lines.append(f"  recent_steps: {result.get('recent_step_count', 0)}")

    latest_loop = result.get("latest_loop") if isinstance(result.get("latest_loop"), dict) else {}
    if latest_loop:
        lines.append(
            "  latest_loop: "
            f"mode={latest_loop.get('mode') or 'unknown'}, "
            f"stop_reason={latest_loop.get('stop_reason') or 'unknown'}"
        )

    selected = result.get("selected_resume_action")
    if isinstance(selected, dict):
        lines.append("  selected_resume_action:")
        lines.append(f"    id: {selected.get('id')}")
        lines.append(f"    command: {selected.get('command')}")
        lines.append(f"    gate: {selected.get('gate')}")
        if selected.get("why"):
            lines.append(f"    why: {str(selected.get('why'))[:220]}")

    blockers = result.get("blockers") or []
    if blockers:
        lines.append("  blockers:")
        for blocker in blockers[:8]:
            lines.append(f"    - {blocker}")

    commands = result.get("resume_commands") or []
    if commands:
        lines.append("  resume_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [
        result.get("artifact_path"),
        result.get("guard_path"),
        result.get("latest_workplan_artifact"),
        result.get("latest_repair_artifact"),
        result.get("latest_contract_artifact"),
        result.get("action_queue_artifact"),
    ]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_self_audit_summary(result: dict) -> list[str]:
    """Render the agent capability self-audit as a compact terminal report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_self_audit] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  overall_score: {result.get('overall_score', 0)}")
    if result.get("capability_readiness"):
        lines.append(f"  capability_readiness: {result.get('capability_readiness')}")
    lines.append(f"  launch_readiness: {result.get('launch_readiness') or 'unknown'}")
    claim_readiness = result.get("claim_readiness")
    if isinstance(claim_readiness, dict):
        lines.append("  claim_readiness:")
        for key in (
            "capability_claim",
            "training_readiness_claim",
            "ai_scientist_parity_claim",
            "rank_or_medal_claim",
            "official_submit_claim",
        ):
            if key in claim_readiness:
                lines.append(f"    - {key}: {claim_readiness.get(key)}")
    trend = result.get("capability_trend")
    if isinstance(trend, dict):
        delta = trend.get("score_delta")
        delta_text = "first_record" if delta is None else f"{delta:+d}" if isinstance(delta, int) else str(delta)
        lines.append(
            "  capability_trend: "
            f"previous={trend.get('previous_score')}; "
            f"current={trend.get('current_score')}; "
            f"delta={delta_text}; "
            f"records={trend.get('records_after', 0)}"
        )
    execution_readiness = result.get("execution_readiness")
    if isinstance(execution_readiness, dict):
        lines.append(f"  execution_readiness: {execution_readiness.get('status') or 'unknown'}")
        lines.append(
            "    runtime_execution_ready: "
            + str(bool(execution_readiness.get("runtime_execution_ready")))
        )
        lines.append(
            "    gate_enforced: "
            + str(bool(execution_readiness.get("gate_enforced")))
        )

    capabilities = result.get("capabilities") or []
    if capabilities:
        lines.append("  capability_scores:")
        for item in capabilities[:10]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"    - {item.get('name')}: "
                f"{item.get('score')} ({item.get('status')})"
            )

    gaps = result.get("gaps") or []
    if gaps:
        lines.append("  gaps:")
        for gap in gaps[:8]:
            if not isinstance(gap, dict):
                lines.append(f"    - {gap}")
                continue
            missing = gap.get("missing_checks") or []
            lines.append(
                f"    - [{gap.get('severity')}] {gap.get('capability')} "
                f"score={gap.get('score')}"
            )
            if missing:
                lines.append(f"      missing: {', '.join(str(x) for x in missing[:3])}")

    backlog = result.get("upgrade_backlog") or []
    if backlog:
        lines.append("  upgrade_backlog:")
        for item in backlog[:8]:
            if not isinstance(item, dict):
                lines.append(f"    - {item}")
                continue
            lines.append(
                f"    - [{item.get('priority', 'P?')}] {item.get('title')} "
                f"({item.get('id')})"
            )
            if item.get("safe_next_command"):
                lines.append(f"      safe_next_command: {item.get('safe_next_command')}")

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    trend_path = trend.get("path") if isinstance(trend, dict) else None
    artifacts = [result.get("artifact_path"), result.get("backlog_artifact_path"), trend_path]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_readiness_report_summary(result: dict) -> list[str]:
    """Render the unified AI Scientist readiness report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_readiness_report] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  overall_score: {result.get('overall_score', 0)}")
    lines.append(f"  capability_readiness: {result.get('capability_readiness') or 'unknown'}")
    lines.append(f"  launch_readiness: {result.get('launch_readiness') or 'unknown'}")

    claim = result.get("claim_readiness")
    if isinstance(claim, dict):
        lines.append("  claim_readiness:")
        for key in (
            "training_readiness_claim",
            "ai_scientist_parity_claim",
            "rank_or_medal_claim",
            "official_submit_claim",
        ):
            if key in claim:
                lines.append(f"    - {key}: {claim.get(key)}")

    matrix = result.get("readiness_matrix") if isinstance(result.get("readiness_matrix"), list) else []
    if matrix:
        lines.append("  readiness_matrix:")
        for item in matrix[:10]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "    - "
                f"{item.get('name')}: {item.get('status')} "
                f"(next={item.get('next_action')})"
            )
            if item.get("evidence"):
                lines.append(f"      evidence: {_short(item.get('evidence'), limit=180)}")

    blockers = result.get("blocking_reasons") if isinstance(result.get("blocking_reasons"), list) else []
    if blockers:
        lines.append("  blocking_reasons:")
        for item in blockers[:8]:
            lines.append(f"    - {_short(item, limit=220)}")

    commands = result.get("recommended_next_commands") if isinstance(result.get("recommended_next_commands"), list) else []
    if commands:
        lines.append("  recommended_next_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [
        result.get("artifact_path"),
        result.get("markdown_artifact_path"),
    ]
    for item in result.get("artifact_evidence") or []:
        if isinstance(item, dict) and item.get("path"):
            artifacts.append(item.get("path"))
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines


def render_scientist_causal_diagnosis_summary(result: dict) -> list[str]:
    """Render the causal diagnosis graph as a concise terminal report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_causal_diagnosis] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  posture: {result.get('posture') or 'unknown'}")
    lines.append(f"  next_safe_command: {result.get('next_safe_command') or 'evomind autopilot'}")

    symptoms = result.get("symptoms") if isinstance(result.get("symptoms"), list) else []
    if symptoms:
        lines.append("  symptoms:")
        for item in symptoms[:8]:
            if isinstance(item, dict):
                lines.append(f"    - [{item.get('severity')}] {item.get('id')}: {_short(item.get('summary'), limit=180)}")

    causes = result.get("root_causes") if isinstance(result.get("root_causes"), list) else []
    if causes:
        lines.append("  root_causes:")
        for item in causes[:8]:
            if isinstance(item, dict):
                lines.append(
                    "    - "
                    f"{item.get('id')} "
                    f"confidence={item.get('confidence')} "
                    f"gate={item.get('gate')}"
                )
                if item.get("summary"):
                    lines.append(f"      {_short(item.get('summary'), limit=180)}")

    interventions = result.get("interventions") if isinstance(result.get("interventions"), list) else []
    if interventions:
        lines.append("  interventions:")
        for item in interventions[:8]:
            if isinstance(item, dict):
                lines.append(
                    "    - "
                    f"{item.get('id')}: {_short(item.get('title'), limit=150)} "
                    f"command={item.get('safe_next_command')}"
                )

    graph = result.get("causal_graph") if isinstance(result.get("causal_graph"), dict) else {}
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    if edges:
        lines.append("  causal_edges:")
        for edge in edges[:10]:
            if isinstance(edge, dict):
                lines.append(f"    - {edge.get('from')} -> {edge.get('to')} ({edge.get('relation')})")

    artifacts = [result.get("artifact_path"), result.get("markdown_artifact_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines


def render_scientist_strategy_optimizer_summary(result: dict) -> list[str]:
    """Render the strategy optimizer as a concise terminal decision report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_strategy_optimizer] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  strategy_posture: {result.get('strategy_posture') or 'unknown'}")
    lines.append(f"  source_posture: {result.get('source_posture') or 'unknown'}")
    lines.append(f"  next_safe_command: {result.get('next_safe_command') or 'evomind causal-diagnosis'}")

    selected = result.get("selected_strategy") if isinstance(result.get("selected_strategy"), dict) else {}
    if selected:
        lines.append("  selected_strategy:")
        lines.append(f"    - id: {selected.get('id')}")
        lines.append(f"      title: {_short(selected.get('title'), limit=160)}")
        lines.append(f"      score: {selected.get('total_score')} gate={selected.get('gate_status')}")
        if selected.get("rationale"):
            lines.append(f"      rationale: {_short(selected.get('rationale'), limit=220)}")

    ranking = result.get("intervention_ranking") if isinstance(result.get("intervention_ranking"), list) else []
    if ranking:
        lines.append("  intervention_ranking:")
        for item in ranking[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "    - "
                f"#{item.get('rank')} {item.get('id')}: "
                f"score={item.get('total_score')} "
                f"impact={item.get('expected_impact')} "
                f"evidence={item.get('evidence_strength')} "
                f"risk={item.get('risk_level')}"
            )
            lines.append(f"      command: {item.get('safe_next_command')}")

    matrix = result.get("decision_matrix") if isinstance(result.get("decision_matrix"), dict) else {}
    if matrix:
        lines.append(
            "  decision_matrix: "
            f"candidates={matrix.get('candidate_count', 0)} "
            f"readiness_blocked={matrix.get('readiness_blocked')}"
        )

    artifacts = [result.get("artifact_path"), result.get("markdown_artifact_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines


def render_scientist_context_packet_summary(result: dict) -> list[str]:
    """Render the per-turn Scientist context packet as a compact briefing."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_context_packet] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    quality = result.get("context_quality") if isinstance(result.get("context_quality"), dict) else {}
    if quality:
        lines.append(
            "  context_quality: "
            f"score={quality.get('score')} "
            f"mode={quality.get('interpretation') or 'unknown'} "
            f"missing={len(quality.get('missing_sources') or [])}"
        )
    readiness = result.get("readiness") if isinstance(result.get("readiness"), dict) else {}
    if readiness:
        lines.append(
            "  readiness: "
            f"llm={readiness.get('llm_ready')} "
            f"kaggle={readiness.get('kaggle_ready')} "
            f"compute={readiness.get('compute_backend')} "
            f"can_execute={readiness.get('can_execute')}"
        )
        blockers = readiness.get("blocking_gates") if isinstance(readiness.get("blocking_gates"), list) else []
        if blockers:
            lines.append("  blocking_gates:")
            for item in blockers[:6]:
                lines.append(f"    - {_short(item, limit=160)}")
    strategy = result.get("active_strategy") if isinstance(result.get("active_strategy"), dict) else {}
    if strategy:
        lines.append(
            "  active_strategy: "
            f"action={strategy.get('selected_action') or '(none)'} "
            f"gate={strategy.get('gate_status') or '(unknown)'}"
        )
        if strategy.get("selected_command"):
            lines.append(f"    command: {strategy.get('selected_command')}")
    memory = result.get("memory_digest") if isinstance(result.get("memory_digest"), dict) else {}
    if memory:
        lines.append(
            "  memory_digest: "
            f"records={memory.get('retrospective_records', 0)} "
            f"scientist_total={memory.get('scientist_memory_records_total')}"
        )
        lessons = memory.get("recent_lessons") if isinstance(memory.get("recent_lessons"), list) else []
        if lessons:
            lines.append("  recent_lessons:")
            for lesson in lessons[:4]:
                lines.append(f"    - {_short(lesson, limit=180)}")
    req = result.get("requirement_context") if isinstance(result.get("requirement_context"), dict) else {}
    if req:
        lines.append(
            "  requirement_context: "
            f"open={len(req.get('open_requirements') or [])} "
            f"blocked={len(req.get('blocked_requirements') or [])}"
        )
    if result.get("next_safe_command"):
        lines.append(f"  next_safe_command: {result.get('next_safe_command')}")
    artifacts = [result.get("artifact_path"), result.get("markdown_artifact_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines


def render_scientist_reasoning_synthesis_summary(result: dict) -> list[str]:
    """Render the evidence-grounded answer contract for one Scientist turn."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_reasoning_synthesis] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  reasoning_mode: {result.get('reasoning_mode') or 'unknown'}")
    quality = result.get("reasoning_quality") if isinstance(result.get("reasoning_quality"), dict) else {}
    if quality:
        lines.append(
            "  reasoning_quality: "
            f"score={quality.get('score', 0)} "
            f"status={quality.get('status') or 'unknown'} "
            f"hypotheses={quality.get('hypotheses_produced', 0)}/"
            f"{quality.get('hypotheses_requested', 0)}"
        )
        missing = quality.get("missing_contract_items") if isinstance(quality.get("missing_contract_items"), list) else []
        if missing:
            lines.append(f"  missing_contract_items: {', '.join(str(item) for item in missing[:8])}")
    if result.get("direct_answer"):
        lines.append(f"  direct_answer: {_short(result.get('direct_answer'), limit=280)}")
    hypotheses = result.get("hypotheses") if isinstance(result.get("hypotheses"), list) else []
    if hypotheses:
        lines.append("  hypotheses:")
        for item in hypotheses[:6]:
            if isinstance(item, dict):
                lines.append(
                    f"    - {item.get('id')}: {_short(item.get('title'), limit=100)} "
                    f"[risk={item.get('risk')}; cost={item.get('cost')}]"
                )
    if result.get("selected_hypothesis_id"):
        lines.append(f"  selected_hypothesis: {result.get('selected_hypothesis_id')}")
    next_action = result.get("next_safe_action") if isinstance(result.get("next_safe_action"), dict) else {}
    if next_action:
        lines.append(
            f"  next_safe_action: {next_action.get('command') or '(none)'} "
            f"gate={next_action.get('gate') or '(unknown)'}"
        )
    llm = result.get("llm") if isinstance(result.get("llm"), dict) else {}
    if llm:
        lines.append(
            "  llm: "
            f"used={llm.get('used')} provider={llm.get('provider') or '(none)'} "
            f"model={llm.get('model') or '(none)'} cache_read={llm.get('cache_read_tokens', 0)}"
        )
    artifacts = [result.get("artifact_path"), result.get("markdown_artifact_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines


def render_scientist_upgrade_plan_summary(result: dict) -> list[str]:
    """Render the self-audit upgrade backlog as an executable engineering plan."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_upgrade_plan] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  readiness: {result.get('readiness') or 'unknown'}")
    lines.append(f"  open_backlog_count: {result.get('open_backlog_count', 0)}")
    if result.get("overall_score") is not None:
        lines.append(f"  self_audit_score: {result.get('overall_score')}")

    steps = result.get("planned_steps") or []
    if steps:
        lines.append("  planned_steps:")
        for item in steps[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"    - {item.get('step_id')}: [{item.get('priority', 'P?')}] "
                f"{item.get('title')} ({item.get('backlog_id')})"
            )
            files = item.get("files_to_inspect") or []
            if files:
                lines.append("      files: " + ", ".join(str(x) for x in files[:5]))
            checks = item.get("acceptance_checks") or []
            if checks:
                lines.append("      acceptance: " + " | ".join(str(x) for x in checks[:3]))
            if item.get("safe_next_command"):
                lines.append(f"      safe_next_command: {item.get('safe_next_command')}")

    policy = result.get("execution_policy") or {}
    if isinstance(policy, dict):
        lines.append(
            "  execution_policy: "
            f"mode={policy.get('mode', 'engineering_plan_only')}; "
            f"training={policy.get('training', 'blocked')}; "
            f"submit={policy.get('official_kaggle_submit', 'blocked')}"
        )

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [
        result.get("artifact_path"),
        result.get("source_backlog_path"),
        result.get("source_self_audit_path"),
    ]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_self_upgrade_loop_summary(result: dict) -> list[str]:
    """Render the safe self-upgrade bridge as a compact work-order report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_self_upgrade_loop] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  status: {result.get('status') or 'unknown'}")
    lines.append(f"  selected_backlog_id: {result.get('selected_backlog_id') or '(none)'}")
    if result.get("selected_title"):
        lines.append(f"  selected_title: {result.get('selected_title')}")
    if result.get("overall_score_before") is not None:
        lines.append(f"  self_audit_score_before: {result.get('overall_score_before')}")
    lines.append(f"  open_backlog_count: {result.get('open_backlog_count', 0)}")

    work_order = result.get("work_order") or {}
    if isinstance(work_order, dict):
        files = work_order.get("files_to_edit") or []
        checks = work_order.get("acceptance_checks") or []
        artifacts = work_order.get("expected_artifacts") or []
        if files:
            lines.append("  files_to_edit:")
            for path in files[:8]:
                lines.append(f"    - {path}")
        if checks:
            lines.append("  acceptance_checks:")
            for check in checks[:6]:
                lines.append(f"    - {check}")
        if artifacts:
            lines.append("  expected_artifacts:")
            for artifact in artifacts[:6]:
                lines.append(f"    - {artifact}")
        if work_order.get("human_gate"):
            lines.append(f"  human_gate: {work_order.get('human_gate')}")

    phases = result.get("loop_phases") or []
    if phases:
        lines.append("  loop_phases:")
        for item in phases[:6]:
            if isinstance(item, dict):
                lines.append(
                    f"    - {item.get('phase')}: "
                    f"{item.get('status') or 'recorded'} -> {item.get('artifact') or '(none)'}"
                )

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [
        result.get("artifact_path"),
        result.get("work_order_path"),
        result.get("trials_path"),
        result.get("source_upgrade_plan_path"),
        result.get("source_self_audit_path"),
    ]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_patch_work_order_summary(result: dict) -> list[str]:
    """Render a code-agent patch work order from Scientist evidence."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_patch_work_order] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  status: {result.get('status') or 'unknown'}")
    lines.append(f"  selected_issue_id: {result.get('selected_issue_id') or '(none)'}")
    if result.get("selected_title"):
        lines.append(f"  selected_title: {result.get('selected_title')}")

    work_order = result.get("work_order") or {}
    if isinstance(work_order, dict):
        if work_order.get("rationale"):
            lines.append(f"  rationale: {work_order.get('rationale')}")
        files = work_order.get("files_to_edit") or []
        checks = work_order.get("acceptance_checks") or []
        expected = work_order.get("expected_artifacts") or []
        if files:
            lines.append("  files_to_edit:")
            for path in files[:10]:
                lines.append(f"    - {path}")
        if checks:
            lines.append("  acceptance_checks:")
            for check in checks[:8]:
                lines.append(f"    - {check}")
        if expected:
            lines.append("  expected_artifacts:")
            for artifact in expected[:8]:
                lines.append(f"    - {artifact}")
        if work_order.get("safe_next_command"):
            lines.append(f"  safe_next_command: {work_order.get('safe_next_command')}")
        if work_order.get("human_gate"):
            lines.append(f"  human_gate: {work_order.get('human_gate')}")

    evidence = result.get("evidence") or {}
    if isinstance(evidence, dict):
        terminal_turn = evidence.get("terminal_turn") if isinstance(evidence.get("terminal_turn"), dict) else {}
        parity = evidence.get("parity") if isinstance(evidence.get("parity"), dict) else {}
        repair = evidence.get("repair_plan") if isinstance(evidence.get("repair_plan"), dict) else {}
        lines.append(
            "  evidence_summary: "
            f"terminal_turn={bool(terminal_turn.get('present'))}; "
            f"budget_exhausted={bool(terminal_turn.get('budget_exhausted'))}; "
            f"parity={bool(parity.get('present'))}; "
            f"root_causes={','.join(str(x) for x in (repair.get('root_causes') or [])[:4]) or '(none)'}"
        )
        phase_status = parity.get("phase_status") if isinstance(parity.get("phase_status"), dict) else {}
        if phase_status:
            lines.append("  parity_phase_status:")
            for key, value in list(phase_status.items())[:6]:
                lines.append(f"    - {key}: {value}")

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:6]:
            lines.append(f"    - {command}")

    artifacts = [
        result.get("artifact_path"),
        result.get("action_queue_path"),
        result.get("trials_path"),
    ]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_engineering_loop_summary(result: dict) -> list[str]:
    """Render an isolated patch validation run."""
    ok = result.get("ok", False)
    lines = [f"[tool:scientist_engineering_loop] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  status: {result.get('status') or 'unknown'}")
    if result.get("message"):
        lines.append(f"  message: {_short(result.get('message'), limit=260)}")
    work_order = result.get("work_order") if isinstance(result.get("work_order"), dict) else {}
    if work_order:
        lines.append(f"  work_order: {work_order.get('id') or '(none)'}")
        lines.append(f"  human_gate: {work_order.get('human_gate') or result.get('human_gate') or '(none)'}")
    changed = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
    if changed:
        lines.append("  changed_files:")
        for path in changed[:10]:
            lines.append(f"    - {path}")
    checks = result.get("acceptance_checks") if isinstance(result.get("acceptance_checks"), list) else []
    if checks:
        passed = sum(1 for item in checks if isinstance(item, dict) and item.get("passed"))
        lines.append(f"  acceptance_checks: {passed}/{len(checks)} passed")
        for item in checks[:8]:
            if isinstance(item, dict):
                lines.append(
                    f"    - {'PASS' if item.get('passed') else 'FAIL'} "
                    f"{_short(item.get('command'), limit=180)}"
                )
    lines.append(f"  main_worktree_modified: {result.get('main_worktree_modified', False)}")
    lines.append(f"  merge_ready: {result.get('merge_ready', False)}")
    if result.get("candidate_diff_path"):
        lines.append(f"  candidate_diff: {result.get('candidate_diff_path')}")
    if result.get("run_manifest_path"):
        lines.append(f"  manifest: {result.get('run_manifest_path')}")
    if result.get("next_safe_command"):
        lines.append(f"  next_safe_command: {result.get('next_safe_command')}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_memory_consolidation_summary(result: dict) -> list[str]:
    """Render Scientist memory writeback as a compact terminal report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_memory_consolidation] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  records_before: {result.get('records_before', 0)}")
    lines.append(f"  candidate_records: {result.get('candidate_records', 0)}")
    lines.append(f"  records_added: {result.get('records_added', 0)}")
    lines.append(f"  records_total: {result.get('records_total', 0)}")

    sources = result.get("source_counts") or {}
    if isinstance(sources, dict):
        lines.append("  sources:")
        for key in (
            "loop_present",
            "lessons",
            "step_events",
            "blocked_step_events",
            "hypothesis_review_present",
            "experiment_blueprint_present",
            "execution_contract_present",
            "patch_work_order_present",
            "patch_action_queue_present",
            "patch_trials",
            "continuation_resume_present",
        ):
            lines.append(f"    - {key}: {sources.get(key, 0)}")

    added = result.get("added_memory_ids") or []
    if added:
        lines.append("  added_memory_ids:")
        for item in added[:8]:
            lines.append(f"    - {item}")

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [result.get("artifact_path"), result.get("memory_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_innovation_backlog_summary(result: dict) -> list[str]:
    """Render memory-guided innovation hypotheses as a compact terminal report."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_innovation_backlog] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")

    task = result.get("task_profile") or {}
    if isinstance(task, dict):
        lines.append(
            "  task_profile: "
            f"{task.get('modality', '?')}/{task.get('task_type', '?')} "
            f"metric={task.get('metric', '?')}({task.get('metric_direction', '?')})"
        )

    memory = result.get("memory_summary") or {}
    if isinstance(memory, dict):
        lines.append("  memory_reuse:")
        for key in (
            "retrospective_memory_records",
            "matched_task_type_records",
            "loop_lessons",
            "turns_considered",
            "step_events_considered",
            "strategy_components_considered",
        ):
            lines.append(f"    - {key}: {memory.get(key, 0)}")

    _append_memory_reuse_plan(lines, result.get("memory_reuse_plan"), indent="  ")

    hypotheses = result.get("innovation_hypotheses") or []
    if hypotheses:
        lines.append("  innovation_hypotheses:")
        for item in hypotheses[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"    - {item.get('id')}: {item.get('strategy_name')} "
                f"[{item.get('proposed_branch_type')}/{item.get('code_generation_mode')}]"
            )
            components = item.get("components") or []
            if components:
                lines.append("      components: " + ", ".join(str(x) for x in components[:5]))
            if item.get("gate"):
                lines.append(f"      gate: {item.get('gate')}")

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [result.get("artifact_path"), result.get("innovation_log_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_hypothesis_review_summary(result: dict) -> list[str]:
    """Render the proposal critique/ranking board for terminal users."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_hypothesis_review] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  recommendation: {result.get('recommendation') or 'unknown'}")
    lines.append(f"  hypotheses_reviewed: {result.get('hypotheses_reviewed', 0)}")

    gate = result.get("gate_summary") or {}
    if isinstance(gate, dict):
        lines.append("  gate_summary:")
        for key in ("data_ready", "execution_contract", "memory_records", "matched_task_type_records"):
            lines.append(f"    - {key}: {gate.get(key)}")

    selected = result.get("selected_hypothesis")
    if isinstance(selected, dict):
        lines.append("  selected_hypothesis:")
        lines.append(
            f"    #{selected.get('rank')} {selected.get('strategy_name')} "
            f"score={selected.get('score')} status={selected.get('status')} risk={selected.get('risk_level')}"
        )
        if selected.get("next_gate"):
            lines.append(f"    next_gate: {selected.get('next_gate')}")
        blockers = selected.get("blockers") or []
        if blockers:
            lines.append("    blockers:")
            for item in blockers[:5]:
                lines.append(f"      - {item}")

    reviews = result.get("reviews") or []
    if reviews:
        lines.append("  ranked_reviews:")
        for item in reviews[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"    - #{item.get('rank')} {item.get('strategy_name')} "
                f"score={item.get('score')} status={item.get('status')} "
                f"risk={item.get('risk_level')} branch={item.get('branch_type')}"
            )
            reasons = item.get("reasons") or []
            if reasons:
                lines.append("      reasons: " + "; ".join(str(x) for x in reasons[:5]))

    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")

    artifacts = [result.get("artifact_path"), result.get("source_backlog_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")

    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_experiment_blueprint_summary(result: dict) -> list[str]:
    """Render the reviewed-hypothesis execution blueprint."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_experiment_blueprint] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  blueprint_status: {result.get('blueprint_status') or 'unknown'}")
    selected = result.get("selected_hypothesis")
    if isinstance(selected, dict) and selected:
        lines.append(
            "  selected_hypothesis: "
            f"{selected.get('strategy_name') or selected.get('hypothesis_id') or 'unknown'} "
            f"score={selected.get('score', 'n/a')} "
            f"branch={selected.get('branch_type', 'unknown')} "
            f"mode={selected.get('code_generation_mode', 'unknown')}"
        )
    blueprint = result.get("experiment_blueprint")
    if isinstance(blueprint, dict):
        lines.append("  experiment_blueprint:")
        for key in ("blueprint_id", "branch_type", "code_generation_mode", "resource_mode", "run_command", "dry_run_command"):
            lines.append(f"    - {key}: {blueprint.get(key)}")
        required = blueprint.get("required_artifacts") or []
        if required:
            lines.append("    required_artifacts:")
            for item in required[:8]:
                lines.append(f"      - {item}")
        memory = blueprint.get("memory_writeback_plan")
        if isinstance(memory, dict):
            lines.append(f"    memory_writeback: {memory.get('target')} when {memory.get('write_when')}")
        _append_memory_reuse_plan(lines, blueprint.get("memory_reuse_plan") or result.get("memory_reuse_plan"), indent="    ")
    gate = result.get("gate_summary") or {}
    if isinstance(gate, dict):
        lines.append("  gate_summary:")
        for key in ("hypothesis_review", "execution_contract", "ready_for_gated_execution"):
            lines.append(f"    - {key}: {gate.get(key)}")
        blockers = gate.get("blockers") or []
        if blockers:
            lines.append("    blockers:")
            for item in blockers[:5]:
                lines.append(f"      - {item}")
    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")
    artifacts = [result.get("artifact_path"), result.get("source_review_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_innovation_trial_feedback_summary(result: dict) -> list[str]:
    """Render the innovation gate-feedback writeback artifact."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_innovation_trial_feedback] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  outcome: {result.get('outcome') or 'unknown'}")
    lines.append(f"  gate_status: {result.get('gate_status') or 'unknown'}")
    lines.append(f"  strategy: {result.get('strategy_name') or 'unknown'}")
    lines.append(f"  hypothesis_id: {result.get('hypothesis_id') or 'unknown'}")
    lines.append(f"  blueprint_id: {result.get('blueprint_id') or 'unknown'}")
    lines.append(f"  branch: {result.get('branch_type') or 'unknown'}")
    lines.append(f"  code_generation_mode: {result.get('code_generation_mode') or 'unknown'}")
    lines.append(
        "  memory_reuse: "
        f"rules={result.get('memory_reuse_rule_count', 0)}, "
        f"avoid={result.get('avoid_pattern_count', 0)}"
    )
    feedback = result.get("trial_feedback")
    if isinstance(feedback, dict):
        blockers = feedback.get("blockers") or []
        if blockers:
            lines.append("  blockers:")
            for item in blockers[:6]:
                lines.append(f"    - {item}")
    if result.get("lesson"):
        lines.append(f"  lesson: {result.get('lesson')}")
    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")
    artifacts = [result.get("artifact_path"), result.get("innovation_log_path")]
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_situation_model_summary(result: dict) -> list[str]:
    """Render the high-level AI Scientist situation model."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_situation_model] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  situation_status: {result.get('situation_status') or 'unknown'}")
    lines.append(f"  readiness_score: {result.get('readiness_score', 0)}")
    model = result.get("situation_model")
    if isinstance(model, dict):
        lines.append(f"  research_question: {model.get('research_question') or '(none)'}")
        lines.append(f"  reasoning_mode: {model.get('reasoning_mode') or 'unknown'}")
        checks = model.get("readiness_checks")
        if isinstance(checks, dict):
            lines.append("  readiness_checks:")
            for key, value in checks.items():
                lines.append(f"    - {key}: {value}")
        blockers = model.get("blocker_model")
        if isinstance(blockers, list) and blockers:
            lines.append("  blocker_model:")
            for item in blockers[:8]:
                if isinstance(item, dict):
                    lines.append(
                        "    - "
                        f"{item.get('category', 'unknown')} "
                        f"severity={item.get('severity', 'unknown')} "
                        f"repair={item.get('repair_command', 'evomind repair')}: "
                        f"{item.get('blocker', '')}"
                    )
        uncertainties = model.get("uncertainties")
        if isinstance(uncertainties, list) and uncertainties:
            lines.append("  uncertainties:")
            for item in uncertainties[:8]:
                lines.append(f"    - {item}")
        strategy = model.get("strategy_model")
        if isinstance(strategy, dict):
            selected = strategy.get("selected_hypothesis")
            if isinstance(selected, dict) and selected:
                lines.append(
                    "  selected_hypothesis: "
                    f"{selected.get('strategy_name') or selected.get('hypothesis_id') or 'unknown'} "
                    f"score={selected.get('score', 'n/a')}"
                )
            blueprint = strategy.get("experiment_blueprint")
            if isinstance(blueprint, dict) and blueprint:
                lines.append(
                    "  experiment_blueprint: "
                    f"{blueprint.get('blueprint_id') or 'not_generated'} "
                    f"branch={blueprint.get('branch_type', 'unknown')} "
                    f"mode={blueprint.get('code_generation_mode', 'unknown')}"
                )
        evolution = model.get("self_evolution_model")
        if isinstance(evolution, dict):
            lines.append(
                "  self_evolution: "
                f"skill={evolution.get('skill_level', 'unknown')}; "
                f"runs={evolution.get('total_runs', 0)}; "
                f"lessons={evolution.get('lessons_recorded', 0)}; "
                f"memory={evolution.get('memory_records', 0)}"
            )
    commands = result.get("next_safe_commands") or []
    if commands:
        lines.append("  next_safe_commands:")
        for command in commands[:8]:
            lines.append(f"    - {command}")
    artifacts = [result.get("artifact_path")]
    artifacts.extend(result.get("source_artifacts") or [])
    artifacts = [str(path) for path in artifacts if path]
    if artifacts:
        lines.append("  artifacts:")
        for path in dict.fromkeys(artifacts):
            lines.append(f"    - {path}")
    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_turn_plan_summary(result: dict) -> list[str]:
    """Render the per-turn Scientist control plan."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_turn_plan] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    lines.append(f"  intent: {intent.get('kind') or 'unknown'}")
    if intent.get("payload"):
        lines.append(f"  intent_payload: {intent.get('payload')}")
    lines.append(f"  autonomy_level: {result.get('autonomy_level') or 'unknown'}")
    readiness = result.get("readiness") if isinstance(result.get("readiness"), dict) else {}
    if readiness:
        lines.append(
            "  readiness: "
            f"llm={readiness.get('llm_ready')}; "
            f"kaggle={readiness.get('kaggle_ready')}; "
            f"compute={readiness.get('compute_backend')}; "
            f"can_execute={readiness.get('can_execute')}"
        )
        blockers = readiness.get("blocking_gates") or []
        if blockers:
            lines.append("  blocking_gates:")
            for item in blockers[:6]:
                lines.append(f"    - {item}")
    critique = result.get("scientific_critique") if isinstance(result.get("scientific_critique"), dict) else {}
    if critique:
        lines.append(
            "  scientific_critique: "
            f"decision={critique.get('decision') or 'unknown'}; "
            f"actionability={critique.get('actionability_score', 'n/a')}"
        )
        gaps = critique.get("evidence_gaps") or []
        if gaps:
            lines.append("  evidence_gaps:")
            for gap in gaps[:5]:
                if isinstance(gap, dict):
                    lines.append(
                        "    - "
                        f"{gap.get('severity', 'unknown')} "
                        f"{gap.get('gap', '')}: "
                        f"{gap.get('why_it_matters', '')} "
                        f"(tool={gap.get('suggested_tool', '')})"
                    )
                else:
                    lines.append(f"    - {gap}")
        uncertainty = critique.get("uncertainty_drivers") or []
        if uncertainty:
            lines.append("  uncertainty:")
            for item in uncertainty[:4]:
                lines.append(f"    - {item}")
        boundaries = critique.get("claim_boundaries") or []
        if boundaries:
            lines.append("  claim_boundaries:")
            for item in boundaries[:3]:
                lines.append(f"    - {item}")
    selected_tools = result.get("selected_tools") or []
    if selected_tools:
        lines.append("  selected_tools:")
        for item in selected_tools[:8]:
            if isinstance(item, dict):
                lines.append(
                    "    - "
                    f"{item.get('tool')} "
                    f"confidence={item.get('confidence')} "
                    f"gate={item.get('gate')}: "
                    f"{item.get('why')}"
                )
    parity = result.get("parity_lifecycle") if isinstance(result.get("parity_lifecycle"), dict) else {}
    phases = parity.get("phases") if isinstance(parity.get("phases"), list) else []
    if phases:
        lines.append("  parity_lifecycle:")
        for item in phases[:5]:
            if isinstance(item, dict):
                lines.append(
                    "    - "
                    f"{item.get('phase')}: "
                    f"{item.get('status', 'planned')} "
                    f"({item.get('purpose', '')})"
                )
    stops = result.get("stop_conditions") or []
    if stops:
        lines.append("  stop_conditions:")
        for item in stops[:6]:
            lines.append(f"    - {item}")
    expected = result.get("expected_artifacts") or []
    if expected:
        lines.append("  expected_artifacts:")
        for item in expected[:8]:
            lines.append(f"    - {item}")
    if result.get("next_safe_command"):
        lines.append(f"  next_safe_command: {result.get('next_safe_command')}")
    if result.get("artifact_path"):
        lines.append(f"  artifact: {result.get('artifact_path')}")
    if result.get("no_training_started", True):
        lines.append("  no_training_started: True")
    lines.append(
        "  official_submit: "
        + str(result.get("official_submit") or "blocked_until_explicit_human_approval")
    )
    return lines


def render_scientist_continuation_status_summary(result: dict) -> list[str]:
    """Render the current AI Scientist continuation checkpoint."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_continuation_status] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  status: {result.get('status') or 'unknown'}")
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(
        "  progress: "
        f"{result.get('completed_required_tools', 0)}/"
        f"{result.get('total_required_tools', 0)} "
        f"({result.get('completion_ratio', 0)})"
    )
    remaining = result.get("remaining_safe_tools") if isinstance(result.get("remaining_safe_tools"), list) else []
    if remaining:
        lines.append("  remaining_safe_tools:")
        for tool in remaining[:8]:
            lines.append(f"    - {_short(tool, limit=120)}")
    completed = result.get("executed_or_completed_tools") if isinstance(result.get("executed_or_completed_tools"), list) else []
    if completed:
        lines.append("  completed_tools:")
        for tool in completed[:8]:
            lines.append(f"    - {_short(tool, limit=120)}")
    history = result.get("progress_history") if isinstance(result.get("progress_history"), list) else []
    if history:
        lines.append("  recent_progress:")
        for item in history[-5:]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "    - "
                f"{_short(item.get('safe_tool'), limit=80)}: "
                f"ok={item.get('tool_ok')}; "
                f"status={_short(item.get('status'), limit=60)}"
            )
    if result.get("next_safe_action_command"):
        lines.append(f"  next_safe_action: {result.get('next_safe_action_command')}")
    elif result.get("safe_next_command"):
        lines.append(f"  safe_next_command: {result.get('safe_next_command')}")
    if result.get("message"):
        lines.append(f"  message: {_short(result.get('message'), limit=260)}")
    if result.get("continuation_artifact_path"):
        lines.append(f"  continuation_artifact: {result.get('continuation_artifact_path')}")
    if result.get("artifact_path"):
        lines.append(f"  status_artifact: {result.get('artifact_path')}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines


def render_scientist_continuation_resume_summary(result: dict) -> list[str]:
    """Render bounded automatic completion of remaining continuation tools."""
    ok = result.get("ok", True)
    lines = [f"[tool:scientist_continuation_resume] {'OK' if ok else 'BLOCKED'}"]
    lines.append(f"  status: {result.get('status') or 'unknown'}")
    lines.append(f"  stop_reason: {result.get('stop_reason') or 'unknown'}")
    lines.append(f"  selected_task: {result.get('selected_task') or '(none)'}")
    lines.append(f"  steps_executed: {result.get('steps_executed', 0)}/{result.get('max_steps', 0)}")
    final_status = result.get("final_status") if isinstance(result.get("final_status"), dict) else {}
    if final_status:
        lines.append(
            "  final_progress: "
            f"{final_status.get('completed_required_tools', 0)}/"
            f"{final_status.get('total_required_tools', 0)} "
            f"({final_status.get('completion_ratio', 0)})"
        )
    steps = result.get("steps") if isinstance(result.get("steps"), list) else []
    if steps:
        lines.append("  executed_steps:")
        for item in steps[-8:]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "    - "
                f"#{item.get('index')}: "
                f"{_short(item.get('executed_tool'), limit=80)} "
                f"status={_short(item.get('status'), limit=60)} "
                f"remaining={len(item.get('after_remaining_safe_tools') or [])}"
            )
    remaining = result.get("remaining_safe_tools") if isinstance(result.get("remaining_safe_tools"), list) else []
    if remaining:
        lines.append("  remaining_safe_tools:")
        for tool in remaining[:8]:
            lines.append(f"    - {_short(tool, limit=120)}")
    if result.get("message"):
        lines.append(f"  message: {_short(result.get('message'), limit=260)}")
    if result.get("artifact_path"):
        lines.append(f"  resume_artifact: {result.get('artifact_path')}")
    if result.get("continuation_status_artifact_path"):
        lines.append(f"  status_artifact: {result.get('continuation_status_artifact_path')}")
    lines.append(f"  no_training_started: {result.get('no_training_started', True)}")
    lines.append(f"  official_submit: {result.get('official_submit') or 'blocked_until_explicit_human_approval'}")
    return lines
