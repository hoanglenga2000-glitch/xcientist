"""EvoMind recovery hook — Claude Code context-guard distillation.

Mirrors Claude Code's ``claude-context-guard-hook.js``: writes a durable
recovery section to the persistent state file on every lifecycle event so
EvoMind can recover from compaction, restart, or confusing memory shifts
without asking the user to restate the goal.

Architecture:
  - ``RecoveryGuard`` is called by the terminal agent on each user turn.
  - It writes a ``<!-- RECOVERY_GUARD_AUTO -->`` section into the persistent
    state file (session.json or a dedicated recovery.md).
  - The section captures: session ID, workspace, selected task, last action,
    last goal, git status, recent tool outcomes, and a transcript tail.
  - Recovery rules are embedded so the agent (or LLM) knows to prefer the
    guard over an incomplete conversation summary.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


GUARD_START = "<!-- RECOVERY_GUARD_AUTO -->"
GUARD_END = "<!-- /RECOVERY_GUARD_AUTO -->"


@dataclass
class RecoveryContext:
    """Snapshot of the terminal agent's state at one point in time."""
    profile: str = "EvoMind"
    event: str = "UserPromptSubmit"
    session_id: str = ""
    cwd: str = ""
    workspace_root: str = ""
    selected_task: str = ""
    last_action: str = ""
    last_goal: str = ""
    compute_override: str = ""
    tool_outcomes: list[str] = field(default_factory=list)
    git_status: str = ""
    updated_at: str = ""

    def to_guard_section(self) -> str:
        lines = [GUARD_START]
        lines.append(f"Updated: {self.updated_at}")
        lines.append(f"Profile: {self.profile}")
        lines.append(f"Event: {self.event}")
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Workspace: {self.workspace_root}")
        lines.append(f"Task: {self.selected_task or '(none)'}")
        lines.append(f"Last action: {self.last_action or '(none)'}")
        lines.append(f"Last goal: {self.last_goal or '(none)'}")
        lines.append(f"Compute override: {self.compute_override or 'default'}")
        lines.append("")
        lines.append("Recovery rules:")
        lines.append("- After compaction, resume from this guard before asking the user to restate.")
        lines.append("- Prefer this guard over any incomplete conversation summary.")
        lines.append("- Recover: active goal, touched files, constraints, verification status.")
        lines.append("- Do not store API keys, tokens, or passwords.")
        lines.append("")
        if self.tool_outcomes:
            lines.append("Recent tool outcomes:")
            for outcome in self.tool_outcomes[-8:]:
                lines.append(f"  - {outcome}")
            lines.append("")
        if self.git_status:
            lines.append("Git status:")
            lines.append(self.git_status)
        lines.append(GUARD_END)
        return "\n".join(lines)


def _run_git_status(root: Path) -> str:
    """Run ``git status --short --branch`` and return a redacted summary."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--branch"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        if result.returncode != 0:
            return ""
        lines = result.stdout.strip().split("\n")[:40]
        # Redact any lines that look like they contain keys
        safe = []
        for line in lines:
            if any(secret in line.lower() for secret in ("api_key", "token", "secret", "password", "sk-")):
                safe.append("  (line redacted — may contain secret)")
            else:
                safe.append(line)
        return "\n".join(safe)
    except Exception:
        return ""


def build_recovery_context(
    *,
    profile: str = "EvoMind",
    event: str = "UserPromptSubmit",
    session_id: str = "",
    workspace_root: Path | str = "",
    selected_task: str = "",
    last_action: str = "",
    last_goal: str = "",
    compute_override: str = "",
    tool_outcomes: Optional[list[str]] = None,
) -> RecoveryContext:
    workspace = Path(workspace_root) if workspace_root else Path.cwd()
    return RecoveryContext(
        profile=profile,
        event=event,
        session_id=session_id or os.environ.get("EVOMIND_SESSION_ID", ""),
        cwd=str(Path.cwd()),
        workspace_root=str(workspace),
        selected_task=selected_task,
        last_action=last_action,
        last_goal=last_goal,
        compute_override=compute_override,
        tool_outcomes=tool_outcomes or [],
        git_status=_run_git_status(workspace),
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )


class RecoveryGuard:
    """Write a recovery guard section into the persistent state file on each turn.

    Usage::

        guard = RecoveryGuard()
        guard.emit(session, event="UserPromptSubmit")
        # ... after tool call ...
        guard.record_tool("model_status: ok — provider=anthropic, model=claude-opus-4-8")
        guard.emit(session, event="PostToolCall")
    """

    def __init__(self, state_file: Optional[Path] = None) -> None:
        self._state_file: Optional[Path] = Path(state_file) if state_file else None
        self._outcomes: list[str] = []

    def set_state_file(self, path: Path) -> None:
        self._state_file = Path(path)

    def record_tool(self, summary: str) -> None:
        # Redact any API key patterns before storing
        import re
        safe = summary
        safe = re.sub(r'sk-[A-Za-z0-9_-]{10,}', '[redacted-key]', safe)
        safe = re.sub(r'agt_codex_[A-Za-z0-9_-]{10,}', '[redacted-token]', safe)
        self._outcomes.append(f"[{datetime.now().strftime('%H:%M:%S')}] {safe}")

    def emit(self, session, *, event: str = "UserPromptSubmit") -> Optional[Path]:
        """Write the recovery guard into the persistent state file.

        Call this on SessionStart, UserPromptSubmit, and after tool calls.
        """
        if self._state_file is None:
            # Default: write to the session.json path in the workspace
            ws = session.workspace_root or str(Path.cwd())
            self._state_file = Path(ws) / ".xsci" / "recovery_guard.md"

        ctx = build_recovery_context(
            profile=getattr(session, "memory_profile", "EvoMind"),
            event=event,
            workspace_root=session.workspace_root or "",
            selected_task=session.selected_task or "",
            last_action=getattr(session, "last_action", ""),
            last_goal=session.last_goal or "",
            compute_override=getattr(session, "current_compute_override", ""),
            tool_outcomes=list(self._outcomes),
        )
        section = ctx.to_guard_section()

        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            if self._state_file.exists():
                existing = self._state_file.read_text(encoding="utf-8")

            # Replace or append the auto guard section
            if GUARD_START in existing and GUARD_END in existing:
                before = existing[:existing.index(GUARD_START)]
                after = existing[existing.index(GUARD_END) + len(GUARD_END):]
                new_content = before + section + after
            else:
                new_content = existing.rstrip() + "\n\n" + section + "\n"

            self._state_file.write_text(new_content, encoding="utf-8")
            return self._state_file
        except OSError:
            return None


# ── Compaction recovery block (mirrors Claude Code's buildCompactionRecoveryBlock) ──

def build_compaction_recovery_block(state_file: Path) -> str:
    """Return a system-prompt block that survives conversation compaction.

    When the conversation history is trimmed, this block remains as a stable
    anchor — the LLM reads it and knows the current task, goal, and recent
    outcomes without needing to recover them from the truncated transcript.
    """
    if not state_file.exists():
        return ""

    try:
        text = state_file.read_text(encoding="utf-8")
    except OSError:
        return ""

    # Extract only the guard section
    guard_text = text
    if GUARD_START in text and GUARD_END in text:
        start = text.index(GUARD_START)
        end = text.index(GUARD_END) + len(GUARD_END)
        guard_text = text[start:end]

    # Cap at 16KB to keep the prompt lean
    guard_text = guard_text[:16000]

    return f"""
<evomind_compaction_recovery>
The conversation may have been compacted. This recovery block is the durable
source of truth — prefer it over any incomplete summary.

RECOVERY RULES:
1. If the conversation summary conflicts with this block, trust this block.
2. Recover: selected task, last goal, tool outcomes, compute setting, git status.
3. Do NOT restart completed work. Read recent tool outcomes to understand what
   was already done.
4. Never store or print API keys, tokens, or passwords.

{guard_text}
</evomind_compaction_recovery>
""".strip()
