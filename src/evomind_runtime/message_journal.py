"""Private, durable native-tool turns for the ordinary assistant loop."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


class MessageJournal:
    def __init__(self, runtime, session_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", session_id):
            raise ValueError("message_journal_session_id_invalid")
        self.runtime, self.session_id = runtime, session_id
        self.root = runtime.runtime_root / "message_journals" / session_id
        self.history_path = self.root / "history.json"
        self.pending_path = self.root / "pending.json"

    def write(self, path: Path, value):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError("message_journal_symlink")
        temporary = path.with_suffix(".tmp")
        if temporary.is_symlink():
            raise ValueError("message_journal_symlink")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def history(self, fallback):
        if not self.history_path.exists():
            return fallback
        value = json.loads(self.history_path.read_text(encoding="utf-8"))
        if value.get("session_id") != self.session_id:
            raise ValueError("message_journal_identity_invalid")
        return value["messages"]

    def save_history(self, messages):
        self.write(self.history_path, {"session_id": self.session_id, "messages": messages})

    def begin(self, messages, calls, offered):
        if self.pending_path.exists():
            raise ValueError("message_journal_unsettled_turn")
        self.write(self.pending_path, {"schema": "evomind.message_tool_batch.v1", "session_id": self.session_id,
            "messages": messages, "offered": sorted(offered), "results": [],
            "calls": [{"id": call.id, "name": call.name, "arguments": call.input} for call in calls]})

    def settle(self):
        from research_os.agent.messaging import ToolResult
        if not self.pending_path.exists():
            return None
        value = json.loads(self.pending_path.read_text(encoding="utf-8"))
        if value.get("session_id") != self.session_id:
            raise ValueError("message_journal_identity_invalid")
        while value["calls"]:
            if self.runtime.user_pause_requested(self.session_id):
                return {"status": "paused", "text": "User pause retained the unfinished tool batch."}
            call = value["calls"][0]
            if call["name"] not in value["offered"]:
                raise ValueError("message_journal_tool_not_offered")
            outcome = self.runtime.invoke_tool(self.session_id, call["name"], call["arguments"], idempotency_key=call["id"])
            if outcome["status"] in {"waiting_approval", "running", "paused", "cancelled"}:
                return {"text": "Existing tool turn awaits settlement; no duplicate was dispatched.", **outcome}
            result = outcome.get("result") or {}
            value["results"].append(ToolResult(call["id"], json.dumps(result, ensure_ascii=True), not bool(result.get("ok"))).to_wire())
            value["calls"].pop(0)
            self.write(self.pending_path, value)
        messages = value["messages"] + [{"role": "user", "content": value["results"]}]
        self.save_history(messages)
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]
        archived = self.root / (digest + ".settled.json")
        if archived.exists():
            if json.loads(archived.read_text(encoding="utf-8")) != value:
                raise ValueError("message_journal_replay_drift")
            self.pending_path.unlink()
        else:
            self.pending_path.rename(archived)
        return {"status": "settled", "messages": messages}
