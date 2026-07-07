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

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .kaggle_intent import (
    CAPABILITY, CHAT, EXECUTION, GREETING, OFFICIAL, PLANNING, REPORT, MEMORY,
    STATUS, TASK_ADD, TASK_USE, TOOL_QUERY, classify,
)
from .kaggle_session import SessionState, MODE_CHAT, MODE_EXECUTING, MODE_PLANNING
from .recovery_guard import RecoveryGuard
from .terminal_events import TerminalEventEmitter, render_tool_result_as_lines
from .terminal_tools import TerminalTools
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
        }
        tool_name = tool_map.get(payload, payload) if payload else "system_status"

        # Special: "tool_status" → list available tools, not a single tool result
        if tool_name == "tool_list" or payload in ("tool_status",):
            return self._show_tool_list(session, root)

        # Special: "task_list" needs session aware rendering
        if tool_name == "task_list":
            return self._show_task_list(session, root)

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
        }
        lines = ["EvoMind 可用终端工具："]
        for name in names:
            desc = tool_descriptions.get(name, "")
            lines.append(f"  • {name}" + (f" — {desc}" if desc else ""))
        lines.append("")
        lines.append("你也可以直接描述需求，例如：“检查数据”、“最近训练结果怎么样”、“现在用的什么模型”。")
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
