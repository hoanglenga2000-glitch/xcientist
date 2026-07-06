"""Conversational brain for the EvoMind research terminal.

Multi-turn conversation with streaming, tool-use loop, and competition
discovery -- giving the terminal agent Claude Code-level interactivity.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .kaggle_session import SessionState

MAX_HISTORY = 20


def _load_history() -> list[dict[str, Any]]:
    from .config import GLOBAL_DIR
    path = GLOBAL_DIR / "conversation_history.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-MAX_HISTORY:]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return []


def _save_history(messages: list[dict[str, Any]]) -> None:
    from .config import GLOBAL_DIR
    path = GLOBAL_DIR / "conversation_history.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages[-MAX_HISTORY:], ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


_SYSTEM_PROMPT = textwrap.dedent("""\
You are the terminal-side research agent of XCIENTIST, a self-evolving ML
research workstation for Kaggle and MLE-Bench. The product name is EvoMind. You converse with the user
in their language, guide them through competition selection, data download,
experiment planning, and training execution.

You have access to TOOLS (functions you can call). Use them proactively:
- list_competitions(query) to find Kaggle competitions
- quick_start(slug_or_url) to register AND download data in one step
- start_training(task_slug, goal) to launch the training loop
- describe_capabilities() to explain what you can do

Rules:
- Reply concisely, in the user's language.
- Never claim a real Kaggle score, rank, medal, or leaderboard position
  unless a genuine Kaggle response artifact is confirmed via claim audit.
- Official Kaggle submission is ALWAYS human-gated.
- Never ask for or print secrets/API keys.
- When listing competitions, highlight 2-3 most relevant ones.
- After quick_start succeeds, summarize what is available and suggest next steps.
- If Kaggle API is not configured, tell the user to run `evomind setup`.
""")

_TOOLS = [
    {
        "name": "list_competitions",
        "description": "List available Kaggle competitions. Filter by keyword like 'titanic', 'image', 'NLP'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword (empty = list all active)"},
            },
        },
    },
    {
        "name": "quick_start",
        "description": "Register a Kaggle competition AND download its data in one step. Accepts a URL or slug.",
        "parameters": {
            "type": "object",
            "properties": {
                "slug_or_url": {"type": "string", "description": "Kaggle URL or slug like 'titanic'"},
            },
            "required": ["slug_or_url"],
        },
    },
    {
        "name": "start_training",
        "description": "Launch the audited training loop for a registered task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_slug": {"type": "string", "description": "Task slug like 'titanic'"},
                "goal": {"type": "string", "description": "Optional research goal"},
            },
            "required": ["task_slug"],
        },
    },
    {
        "name": "describe_capabilities",
        "description": "Describe what the research workstation can do.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def _execute_tool(name: str, params: dict[str, Any], session: "SessionState") -> str:
    from . import kaggle_actions, kaggle_competitions

    if name == "list_competitions":
        result = kaggle_competitions.list_competitions(query=params.get("query", ""))
        return json.dumps(result, ensure_ascii=False)

    if name == "quick_start":
        from .config import active_root
        root = active_root()
        result = kaggle_actions.quick_start(params.get("slug_or_url", ""), root=root)
        if result.get("ok") and result.get("slug"):
            session.selected_task = result["slug"]
            session.refresh_task_brief(Path(session.workspace_root) if session.workspace_root else Path.cwd())
            session.persist()
        return json.dumps(result, ensure_ascii=False)

    if name == "start_training":
        slug = params.get("task_slug", "")
        if not slug:
            return json.dumps({"ok": False, "message": "task_slug is required"})
        goal = params.get("goal", "")
        from .kaggle import _run_agent
        root = Path(session.workspace_root) if session.workspace_root else None
        try:
            rc = _run_agent(slug, root, goal=goal or "Establish strong baseline and improve.")
            return json.dumps({"ok": rc == 0, "message": "Training completed" if rc == 0 else "Training had issues"})
        except Exception as exc:
            return json.dumps({"ok": False, "message": str(exc)})

    if name == "describe_capabilities":
        return json.dumps({"ok": True, "message": textwrap.dedent("""\
            XCIENTIST Research Workstation - 4-layer architecture:
            1. Multi-Agent Research OS: task parsing, data audit, code gen, training, reports
            2. MLEvolve Search: MCGS, multi-branch, best-so-far, progressive search
            3. XCIENTIST Audit: validation contract, claim audit, evidence binding
            4. Memory/Benchmark: retrospective memory, MLE-Bench tracking
            Supported: tabular classification/regression, image classification, time-series
            Compute: local CPU (default) or remote GPU via SSH
            Workflow: select competition -> download data -> plan -> train -> report -> (human-gated submit)
        """)})

    return json.dumps({"ok": False, "message": f"Unknown tool: {name}"})


class ConversationAgent:
    def __init__(self, *, client=None) -> None:
        self._client = client
        self._resolved = client is not None

    def _get_client(self):
        if not self._resolved:
            try:
                from research_os.llm_client import LLMClient
                self._client = LLMClient()
            except Exception:
                self._client = None
            self._resolved = True
        return self._client

    def _llm_available(self, session: "SessionState") -> bool:
        if not session.llm_ready:
            return False
        client = self._get_client()
        return client is not None

    def _context_block(self, session: "SessionState") -> str:
        lines = [
            "[session]",
            f"selected_task = {session.selected_task or '(none)'}",
            f"compute_backend = {session.compute_backend}",
            f"llm_ready = {session.llm_ready}",
            f"kaggle_ready = {session.kaggle_ready}",
            f"gpu_ready = {session.gpu_ready}",
            f"gpu_blocked = {getattr(session, 'gpu_blocked', False)}",
            f"gpu_status = {getattr(session, 'gpu_status', '') or '(not declared)'}",
            f"memory = {session.memory_summary or 'empty'}",
            f"recent_run = {session.recent_run_id or '(none)'}",
        ]
        if session.recent_best_cv is not None:
            lines.append(f"recent_best_cv = {session.recent_best_cv}")
        if session.task_brief:
            lines.append(f"task = {session.task_brief}")
        gaps = session.missing_setup()
        if gaps:
            heads = ", ".join(g.split(":", 1)[0] for g in gaps)
            lines.append(f"setup_gaps = {heads}")
        return "\n".join(lines)

    def reply(self, text: str, session: "SessionState") -> str:
        if self._llm_available(session):
            history = _load_history()
            history.append({"role": "user", "content": text})
            answer = self._streaming_reply(session, text, history)
            if answer:
                history.append({"role": "assistant", "content": answer[:2000]})
                _save_history(history)
                return answer
        return self._rule_reply(text, session)

    def _streaming_reply(self, session: "SessionState", user: str,
                         history: list[dict[str, Any]]) -> str:
        client = self._get_client()
        if client is None:
            return ""

        ctx = self._context_block(session)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for msg in history[-MAX_HISTORY:]:
            messages.append(msg)
        messages.append({"role": "user", "content": f"[context]\n{ctx}\n\n[user]\n{user}"})

        try:
            resp = client.generate(
                f"{ctx}\n\n[user]\n{user}",
                system=_SYSTEM_PROMPT,
                max_tokens=900,
            )
            return (resp.text or "").strip()
        except Exception:
            return ""

    def plan(self, goal: str, session: "SessionState") -> str:
        if self._llm_available(session):
            prompt = (
                "The user wants a research PLAN, not execution, for this goal:\n"
                f"{goal}\n\n"
                "Ground the plan in the selected task brief. Use the four-layer "
                "workstation: task understanding, data audit, hypothesis, MCGS "
                "search, code-generation, training, promotion gate, memory, claim audit."
            )
            answer = self._ask(session, prompt, max_tokens=1200)
            if answer:
                return answer
        return self._rule_plan(goal, session)

    def _ask(self, session: "SessionState", user: str, *, max_tokens: int = 900) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.generate(
                f"{self._context_block(session)}\n\n[user]\n{user}",
                system=_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )
            return (resp.text or "").strip() or None
        except Exception:
            return None

    def _rule_reply(self, text: str, session: "SessionState") -> str:
        task = session.selected_task
        gaps = session.missing_setup()
        normalized = (text or "").strip().lower()
        if normalized in {"status", "/status", "ready", "就绪", "状态"}:
            lines = ["System status: check `evomind ready` for details."]
            if gaps:
                lines.append("\nSetup gaps:")
                lines.extend(f"- {gap}" for gap in gaps)
            return "\n".join(lines)
        if normalized in {"你好", "hello", "hi", "hey"}:
            return (
                "你好，我是 EvoMind 对话终端。"
                "我可以帮你浏览比赛、规划实验、调用工作站门禁，并在配置完整后启动可审计训练。"
            )
        if not task:
            return (
                "No competition selected yet. You can:\n"
                "1. Tell me the competition name (e.g. Titanic, House Prices) and I will search\n"
                "2. Paste a Kaggle URL: task add https://www.kaggle.com/c/titanic\n"
                "3. Run `evomind setup` to configure your environment first\n\n"
                "Once Kaggle API is configured, I can browse listings, download data, and start training."
            )
        base = (
            f"I understand you want to work on {task}. I am in conversation/planning mode. "
            "To actually start training, say 'start training' or type /run."
        )
        if session.task_brief:
            base += f"\n\nTask overview: {session.task_brief}"
        if gaps:
            base += "\n\nMissing config:\n" + "\n".join(f"- {g}" for g in gaps)
        return base

    def _rule_plan(self, goal: str, session: "SessionState") -> str:
        task = session.selected_task or "(no task selected)"
        lines = [f"Research plan for {task} (planning only, no training):"]
        if session.task_brief:
            lines.append(f"  Task overview: {session.task_brief}")
        lines += [
            f"  Goal: {goal.strip() or 'Establish strong baseline with 2-3 evidence-backed improvements'}",
            "  1. Task understanding: modality, task_type, metric, submission format",
            "  2. Data audit: missing values, cardinality, leakage risk, CV split",
            "  3. Hypotheses: write verifiable improvement hypotheses",
            "  4. Search decision: MCGS parent selection, branch type",
            "  5. Code generation: auditable code with validation contract",
            "  6. Training execution: via workstation resource layer",
            "  7. Promotion gate: only promote if best-so-far improved",
            "  8. Memory: record lessons (success -> reusable, failure -> avoid)",
            "  9. Claim audit: no rank/medal claims without official Kaggle response",
        ]
        if session.can_execute():
            lines.append("\nReady. Type /run or 'start training' to enter the execution loop.")
        else:
            lines.append("\nFix the above config gaps before training.")
        return "\n".join(lines)

    def capability(self, session: "SessionState") -> str:
        task_line = (
            f"Current task: {session.selected_task}."
            if session.selected_task
            else "No competition selected yet."
        )
        return (
            f"{task_line}\n"
            "I am EvoMind, your research AI scientist terminal. Capabilities:\n\n"
            "  Competition discovery: search/browse Kaggle competitions\n"
            "  One-click setup: register + download data in one step\n"
            "  Research planning: analyze task, propose hypotheses, design experiments\n"
            "  Model search: MLEvolve-style MCGS controller for optimal branches\n"
            "  Auto training: data audit -> baseline -> feature engineering -> model selection -> report\n"
            "  Evidence binding: XCIENTIST harness with validation contract + claim audit\n"
            "  Report export: auto-generated Markdown/HTML/DOCX with charts\n\n"
            "  Official Kaggle submission and medal claims remain human-gated.\n"
            "  Tell me what competition you want, or describe your research goal."
        )
