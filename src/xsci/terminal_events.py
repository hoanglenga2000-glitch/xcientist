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
        glyph = "▸" if self.colour else ">"
        if status == "passed":
            mark = "\033[92m✓\033[0m" if self.colour else "[OK]"
        elif status == "blocked":
            mark = "\033[93m⊘\033[0m" if self.colour else "[BLOCKED]"
        elif status == "failed":
            mark = "\033[91m✗\033[0m" if self.colour else "[FAIL]"
        else:
            mark = "\033[96m●\033[0m" if self.colour else "[...]"

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
    status = "✓" if ok else "✗"
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
