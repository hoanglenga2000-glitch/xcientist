"""Message ledger — incremental, crash-survivable conversation persistence.

A long research run can be killed (GPU blip, host exit) mid-flight. The ledger
appends each message to ``messages.jsonl`` as it is added, so the conversation
survives a crash and can be resumed. It is deliberately simple (JSONL, stdlib
only, UTF-8) — the search graph / summary / events already hold the auditable
research state; this only preserves the raw dialogue needed to continue.

Design:
  * ``append`` writes one message per line, flushed — a reader/resumer sees turns
    as they happen and a crash loses at most the in-flight line.
  * ``load`` reconstructs the message list, tolerating a half-written trailing line
    (same forgiving policy as events.read_events).
  * secrets never appear here (messages carry task/goal/code/results, not keys).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MessageLedger:
    """Append-only JSONL log of the conversation, for crash-survival + resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a half-written trailing line from a crash
        return out

    def reset(self) -> None:
        """Truncate the ledger (used when a compaction rewrites the history)."""
        self.path.write_text("", encoding="utf-8")

    def rewrite(self, messages: list[dict[str, Any]]) -> None:
        """Replace the whole ledger (e.g. after compaction shrinks the history)."""
        with self.path.open("w", encoding="utf-8") as fh:
            for message in messages:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")
            fh.flush()
