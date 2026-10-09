"""Read-only, message-scoped projection of durable assistant turns."""
from __future__ import annotations

import hashlib
from typing import Any


ACTIVE_STATUSES = {"queued", "planning", "running", "verifying", "recovering"}
EXECUTION_HEADER = "You are executing one durable EvoMind assistant run.\n"
CONTINUATION_HEADER = "Original user objective (historical context only; overridden wherever it conflicts with the active continuation):\n"


def project_message_history(run: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    run_id = str(run["id"])
    session_id = str(run.get("session_id") or run_id)
    prompt = str(run.get("prompt") or "")
    messages = [{"id": f"user_{run_id}", "role": "user", "content": prompt,
                 "created_at": str(run.get("created_at") or ""), "kind": "original_request"}]
    known_instructions = set((run.get("plan") or {}).get("continuation_instructions") or [])
    last_user = -1
    last_assistant = -1
    latest_reply = None
    for index, turn in enumerate(turns):
        if turn.get("session_id") and str(turn["session_id"]) != session_id:
            continue
        role, content = turn.get("role"), turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        identity = str(turn.get("id") or hashlib.sha256(f"{index}:{role}:{content}".encode()).hexdigest())
        row = {"id": f"history_{run_id}_{identity}", "role": role, "content": content,
               "created_at": str(turn.get("created_at") or ""), "kind": "stored_reply"}
        if role == "user":
            last_user = index
            row["kind"] = "user_request"
            if content.startswith(EXECUTION_HEADER):
                # Never expose the model's infrastructure prompt, tool receipts,
                # credential availability, or injected conversation context.
                instruction = next((value for value in known_instructions
                                    if content.endswith(CONTINUATION_HEADER + prompt + "\nActive continuation objective:\n" + value)), "")
                row.update(content=instruction, kind="user_request" if instruction else "continued_execution")
                if not instruction and last_assistant < 0:
                    continue
            elif content == prompt and last_assistant < 0:
                continue
        else:
            last_assistant = index
            latest_reply = row
        messages.append(row)
    current = bool(latest_reply is not None and last_assistant > last_user
                   and str(run.get("status")) not in ACTIVE_STATUSES)
    answer = str(latest_reply["content"]) if current else ""
    active_id = str(latest_reply["id"]) if current else f"assistant_{run_id}"
    if not current:
        messages.append({"id": active_id, "role": "assistant", "content": "", "kind": "pending_response"})
    return {"schema": "evomind.assistant_message_history.v1", "messages": messages,
            "active_message_id": active_id, "answer": answer, "answer_is_current": current}
