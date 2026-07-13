"""Durable AI Scientist turn ledger for EvoMind.

The terminal can feel smart only when its reasoning is observable and
recoverable. This module stores each high-level scientist turn as sanitized
JSONL so the CLI, dashboard, and recovery context can show what was inspected,
which tools were used, what decision was reached, and which gates stayed
blocked.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        return [_safe_json(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:60]:
            key_text = str(key)
            if re.search(r"(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)", key_text, re.I):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = _safe_json(item, depth=depth + 1)
        return result
    return _safe_text(value)


def scientist_turns_path(root: Path | str) -> Path:
    return Path(root) / ".xsci" / "scientist_turns.jsonl"


def scientist_latest_turn_path(root: Path | str) -> Path:
    return Path(root) / ".xsci" / "scientist_latest_turn.json"


def scientist_parity_loop_path(root: Path | str) -> Path:
    return Path(root) / ".xsci" / "scientist_parity_loop.jsonl"


def record_scientist_turn(root: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one sanitized scientist turn and update latest-turn JSON."""
    root_path = Path(root)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "ts": ts,
        "turn_id": payload.get("turn_id") or f"turn_{ts.replace(':', '').replace('+', 'Z')}",
        "task": payload.get("task") or "",
        "route": payload.get("route") or "unknown",
        "user": _safe_text(payload.get("user"), limit=500),
        "forced_tools": _safe_json(payload.get("forced_tools") or []),
        "executed_tools": _safe_json(payload.get("executed_tools") or []),
        "mode": payload.get("mode") or "",
        "decision": _safe_json(payload.get("decision") or {}),
        "blockers": _safe_json(payload.get("blockers") or []),
        "next_actions": _safe_json(payload.get("next_actions") or []),
        "artifacts": _safe_json(payload.get("artifacts") or []),
        "parity_lifecycle": _safe_json(payload.get("parity_lifecycle") or {}),
        "answer_preview": _safe_text(payload.get("answer_preview"), limit=900),
        "no_training_started": bool(payload.get("no_training_started", True)),
        "official_submit": "blocked_until_explicit_human_approval",
    }

    jsonl_path = scientist_turns_path(root_path)
    latest_path = scientist_latest_turn_path(root_path)
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        latest_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return entry


def record_scientist_parity_loop(root: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one observe-plan-act-reflect-improve lifecycle record."""
    root_path = Path(root)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {
        "ts": ts,
        "turn_id": payload.get("turn_id") or f"parity_{ts.replace(':', '').replace('+', 'Z')}",
        "task": payload.get("task") or "",
        "route": payload.get("route") or "scientist_terminal_turn",
        "goal": _safe_text(payload.get("goal"), limit=500),
        "lifecycle": _safe_json(payload.get("lifecycle") or {}),
        "phase_status": _safe_json(payload.get("phase_status") or {}),
        "executed_tools": _safe_json(payload.get("executed_tools") or []),
        "deferred_tools": _safe_json(payload.get("deferred_tools") or []),
        "next_safe_command": _safe_text(payload.get("next_safe_command"), limit=240),
        "improvement_record": _safe_json(payload.get("improvement_record") or {}),
        "artifacts": _safe_json(payload.get("artifacts") or []),
        "no_training_started": bool(payload.get("no_training_started", True)),
        "official_submit": "blocked_until_explicit_human_approval",
    }
    path = scientist_parity_loop_path(root_path)
    latest_path = root_path / ".xsci" / "scientist_latest_parity_loop.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        latest_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return entry


def load_recent_scientist_turns(root: Path | str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Read recent turns, newest last."""
    path = scientist_turns_path(root)
    entries: list[dict[str, Any]] = []
    try:
        if not path.exists():
            return entries
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    entries.append(_safe_json(data))
    except OSError:
        return []
    return entries[-limit:]


def load_recent_scientist_parity_loops(root: Path | str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Read recent observe-plan-act-reflect-improve lifecycle records."""
    path = scientist_parity_loop_path(root)
    entries: list[dict[str, Any]] = []
    try:
        if not path.exists():
            return entries
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    entries.append(_safe_json(data))
    except OSError:
        return []
    return entries[-limit:]
