"""Automatic context rescue — Claude Code's autoRescueContext distillation.

When conversation history grows past a byte threshold, this module trims the
oldest turns while keeping the current goal, task state, and recent tool
results intact.  The algorithm is DETERMINISTIC (no LLM call needed for the
compaction itself).

Mirrors the router's ``autoRescueContext()`` which runs BEFORE the request
hits the upstream API, so the model never sees the oversized body.
"""
from __future__ import annotations

from typing import Any


def estimate_message_chars(messages: list[dict[str, Any]]) -> int:
    """Conservative estimate of total character count in a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for val in block.values():
                        if isinstance(val, str):
                            total += len(val)
    return total


def estimate_body_bytes(messages: list[dict[str, Any]], system: Any = None,
                        tools: Any = None) -> int:
    """Rough byte estimate of the full request body (messages + system + tools)."""
    msg_chars = estimate_message_chars(messages)
    sys_chars = 0
    if isinstance(system, str):
        sys_chars = len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                sys_chars += len(str(block.get("text", "")))
    tool_chars = len(str(tools)) if tools else 0
    # UTF-8: assume 1.5 bytes per char for CJK safety
    return int((msg_chars + sys_chars + tool_chars) * 1.5)


def auto_rescue_context(
    messages: list[dict[str, Any]],
    *,
    target_bytes: int = 680_000,
    min_keep_messages: int = 4,
    max_drop_iterations: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trim oldest messages until the body fits under ``target_bytes``.

    Returns ``(trimmed_messages, report_dict)``.  The report describes what
    was dropped so the caller can log it.

    The algorithm:
      1. Always keep the system prompt / first message.
      2. Drop the oldest user+assistant pair that is NOT the system prompt.
      3. Stop when under ``target_bytes`` or ``min_keep_messages`` remain.
    """
    if len(messages) <= min_keep_messages:
        return messages, {"dropped": 0, "before": len(messages), "after": len(messages),
                          "reason": "below_min_keep"}

    before_count = len(messages)
    before_bytes = estimate_body_bytes(messages)

    if before_bytes <= target_bytes:
        return messages, {"dropped": 0, "before": before_count, "after": before_count,
                          "before_bytes": before_bytes, "reason": "under_target"}

    # Keep the first message (system context / seed) and the last N
    kept = list(messages)
    dropped = 0
    for _ in range(max_drop_iterations):
        if len(kept) <= min_keep_messages:
            break
        current_bytes = estimate_body_bytes(kept)
        if current_bytes <= target_bytes:
            break
        # Drop the oldest non-system message (index 1, after the seed)
        if len(kept) > 1:
            del kept[1]
            dropped += 1
        else:
            break

    after_bytes = estimate_body_bytes(kept)
    return kept, {
        "dropped": dropped,
        "before": before_count,
        "after": len(kept),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "reason": "rescued" if after_bytes <= target_bytes else "truncated_best_effort",
        "target_bytes": target_bytes,
        "min_kept": min_keep_messages,
    }


def build_context_rescue_system_block(report: dict[str, Any]) -> str:
    """Build a system prompt block that tells the model context was trimmed.

    The model sees this and knows its job is to recover the active goal from
    the persistent recovery guard, not to ask the user to restate everything.
    """
    if report.get("dropped", 0) == 0:
        return ""

    return (
        f"<context_rescue_notice>\n"
        f"The conversation history was automatically trimmed to stay within the\n"
        f"context budget. {report.get('dropped', 0)} older turns were dropped.\n"
        f"Before: {report.get('before', '?')} messages ({report.get('before_bytes', '?')} bytes)\n"
        f"After:  {report.get('after', '?')} messages ({report.get('after_bytes', '?')} bytes)\n"
        f"\n"
        f"Review the recovery guard in your system context for the active goal,\n"
        f"selected task, and recent tool outcomes. Do NOT restart completed work.\n"
        f"</context_rescue_notice>"
    )
