"""Tool call ledger — Claude Code's messages.jsonl distillation.

Every tool call + its result is appended to a JSONL file in the workspace
so that:
  1. After a crash, the agent can see what was already done and skip it.
  2. After compaction, the stable recovery block references recent outcomes.
  3. The dashboard (:8088) can show a chronological tool-call timeline.

Mirrors Claude Code's ``_ledger`` (MessageLedger) in research_os/agent/session.py
but operates at the TERMINAL level — tracking lightweight tools like model_status,
data_check, task_list, and the gates that block or pass training.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ToolLedger:
    """Append-only JSONL log of terminal tool calls.

    Usage::

        ledger = ToolLedger(workspace_root)
        ledger.record("model_status", {"provider": "anthropic"}, ok=True)
        ledger.record("start_training", {"task": "titanic", "compute": "local"}, ok=False,
                      summary="blocked: no LLM key")

        # Recover after crash:
        recent = ledger.recent(limit=10)
        for entry in recent:
            print(entry["tool"], entry["ok"], entry["summary"])
    """

    def __init__(self, workspace_root: Path | str) -> None:
        self._path = Path(workspace_root) / "tool_ledger.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, tool: str, result: dict[str, Any], *,
               ok: bool = True, summary: str = "") -> str:
        """Append one tool call entry to the ledger.

        Returns the ISO timestamp of the recorded entry.
        """
        ts = datetime.now().isoformat(timespec="seconds")
        entry = {
            "ts": ts,
            "tool": tool,
            "ok": ok,
            "result_keys": sorted(k for k in result if k not in ("ok", "tool")),
            "summary": summary or result.get("message", "")[:300],
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return ts

    def recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent entries (newest last)."""
        entries: list[dict[str, Any]] = []
        try:
            if not self._path.exists():
                return entries
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return entries[-limit:]

    def last_outcome_for(self, tool: str) -> Optional[dict[str, Any]]:
        """Return the most recent entry for a specific tool, or None."""
        entries = self.recent(limit=200)
        for entry in reversed(entries):
            if entry.get("tool") == tool:
                return entry
        return None

    def summary_lines(self, limit: int = 8) -> list[str]:
        """Return human-readable summary lines for the recovery context."""
        entries = self.recent(limit=limit)
        lines = []
        for e in entries:
            mark = "✓" if e.get("ok") else "✗"
            lines.append(f"[{e.get('ts', '?')}] {mark} {e.get('tool', '?')}: {e.get('summary', '')[:120]}")
        return lines
