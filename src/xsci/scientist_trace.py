"""Durable step-level trace for the EvoMind AI Scientist.

The scientist turn ledger stores a high-level conversation summary.  This
module stores the lower-level execution trace that makes the agent feel
observable: each checkpoint, tool call, gate decision, workplan step, artifact,
and blocker is appended as sanitized JSONL for the terminal and dashboard.

It never reads secrets, never starts training, and never submits to Kaggle.
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
SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)",
    re.IGNORECASE,
)


def _safe_text(value: Any, *, limit: int = 1600) -> str:
    text = "" if value is None else str(value)
    text = SENSITIVE_RE.sub(r"\1=[redacted]", text)
    return text[:limit]


def _safe_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json(item, depth=depth + 1) for item in value[:40]]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:80]:
            key_text = str(key)
            if SENSITIVE_KEY_RE.search(key_text):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = _safe_json(item, depth=depth + 1)
        return result
    return _safe_text(value)


def scientist_step_trace_path(root: Path | str) -> Path:
    return Path(root) / ".xsci" / "scientist_step_trace.jsonl"


def scientist_latest_step_event_path(root: Path | str) -> Path:
    return Path(root) / ".xsci" / "scientist_latest_step_event.json"


def record_scientist_step_event(root: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one sanitized scientist step event and update latest event JSON."""
    root_path = Path(root)
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    event = {
        "ts": ts,
        "event_id": payload.get("event_id") or f"evt_{ts.replace(':', '').replace('+', 'Z')}",
        "trace_run_id": payload.get("trace_run_id") or "",
        "source": payload.get("source") or "evomind",
        "task": payload.get("task") or "",
        "phase": payload.get("phase") or payload.get("event") or "step",
        "step_id": payload.get("step_id") or "",
        "status": payload.get("status") or "info",
        "tool": payload.get("tool") or "",
        "message": _safe_text(payload.get("message"), limit=900),
        "artifact_path": _safe_text(payload.get("artifact_path"), limit=600),
        "gate": payload.get("gate") or "",
        "evidence": _safe_json(payload.get("evidence") or []),
        "details": _safe_json(payload.get("details") or {}),
        "no_training_started": bool(payload.get("no_training_started", True)),
        "official_submit": "blocked_until_explicit_human_approval",
    }

    jsonl_path = scientist_step_trace_path(root_path)
    latest_path = scientist_latest_step_event_path(root_path)
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        latest_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return event


def load_recent_scientist_step_events(root: Path | str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Read recent step trace events, newest last."""
    path = scientist_step_trace_path(root)
    events: list[dict[str, Any]] = []
    try:
        if not path.exists():
            return events
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
                    events.append(_safe_json(data))
    except OSError:
        return []
    return events[-limit:]
