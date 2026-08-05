from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .model_router import ModelRouter
from .models import ApprovalRequest, PermissionLevel, Session, SessionStatus, ToolCall, ToolResult, new_id, utc_now
from .policy import PolicyEngine, argument_fingerprint
from .store import RuntimeStore
from .tools import ToolContext, ToolRegistry, build_default_registry


class AgentRuntime:
    def __init__(
        self, workspace_root: str | Path, runtime_root: str | Path | None = None, registry: ToolRegistry | None = None
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.runtime_root = Path(runtime_root or self.workspace_root / "workspace" / "runtime").resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.runtime_root / "artifacts"
        self.store = RuntimeStore(self.runtime_root / "runtime.sqlite3")
        self.registry = registry or build_default_registry()
        self.policy = PolicyEngine()
        self.router = ModelRouter()
        # ThreadingHTTPServer can deliver duplicate retries concurrently.  Keep
        # the idempotency lookup and the initial durable state transition
        # atomic so only one request is allowed to execute a side effect.
        self._invoke_lock = threading.RLock()

    def close(self) -> None:
        self.store.close()

    def create_session(
        self,
        *,
        objective: str = "",
        title: str = "",
        permission_level: str = PermissionLevel.WORKSPACE_WRITE.value,
        workspace_root: str | None = None,
        parent_session_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        PermissionLevel(permission_level)
        session = Session(
            session_id or new_id("session"),
            str(Path(workspace_root or self.workspace_root).resolve()),
            permission_level,
            title=title or objective[:80],
            objective=objective,
            parent_session_id=parent_session_id,
        )
        return self.store.create_session(session).to_dict()

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if not session:
            raise KeyError(session_id)
        return session

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_sessions(limit)

    def tools(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.registry.specs()]

    def invoke_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str = "",
        approved_fingerprint: str = "",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        decision = self.policy.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            permission_level=session["permission_level"],
            workspace_root=session["workspace_root"],
            approved_fingerprint=approved_fingerprint,
        )
        if tool_call_id:
            call_id = tool_call_id
        elif idempotency_key:
            stable_key = argument_fingerprint("idempotency", {"session_id": session_id, "key": idempotency_key})
            call_id = f"call_{stable_key[:32]}"
        else:
            call_id = new_id("call")

        with self._invoke_lock:
            existing = self.store.get_tool_call(call_id)
            if existing:
                same_request = (
                    existing["session_id"] == session_id
                    and existing["tool_name"] == tool_name
                    and existing["arguments"] == decision.normalized_arguments
                )
                if not same_request:
                    conflict = ToolResult(
                        call_id,
                        False,
                        {"existing_tool_name": existing["tool_name"], "existing_arguments": existing["arguments"]},
                        "idempotency key is already bound to a different tool request",
                        error="idempotency_conflict",
                    )
                    return {"status": "failed", "result": conflict.to_dict(), "replayed": True}
                if existing["status"] in {"completed", "failed"}:
                    return {
                        "status": existing["status"],
                        "tool_call": existing,
                        "result": existing["result"],
                        "replayed": True,
                    }
                if existing["status"] == "waiting_approval" and not approved_fingerprint:
                    approval = self.store.get_approval(existing["approval_id"])
                    return {"status": "waiting_approval", "tool_call": existing, "approval": approval, "replayed": True}
                if existing["status"] == "running":
                    return {"status": "running", "tool_call": existing, "replayed": True}

            call = ToolCall(
                call_id, session_id, tool_name, decision.normalized_arguments, idempotency_key=idempotency_key
            )
            if decision.requires_approval:
                call.status = "waiting_approval"
                approval = ApprovalRequest(
                    new_id("approval"),
                    session_id,
                    call.id,
                    tool_name,
                    argument_fingerprint(tool_name, decision.normalized_arguments),
                    decision.normalized_arguments,
                    decision.scope,
                    decision.risk_level,
                    decision.reversible,
                    expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                )
                call.approval_id = approval.id
                self.store.put_tool_call(call)
                self.store.put_approval(approval)
                self.store.update_session(session_id, status=SessionStatus.WAITING_APPROVAL.value)
                self.store.append_event(session_id, "approval.requested", approval.to_dict())
                self.store.checkpoint(
                    session_id, {"phase": "waiting_approval", "tool_call_id": call.id, "approval_id": approval.id}
                )
                return {"status": "waiting_approval", "tool_call": call.to_dict(), "approval": approval.to_dict()}
            if not decision.allowed:
                result = ToolResult(call.id, False, {}, decision.reason, error="permission_denied")
                call.status, call.completed_at = "failed", utc_now()
                self.store.put_tool_call(call, result)
                self.store.append_event(session_id, "tool.failed", result.to_dict())
                return {"status": "failed", "result": result.to_dict(), "decision": decision.to_dict()}
            call.status, call.started_at = "running", utc_now()
            self.store.put_tool_call(call)
            self.store.update_session(session_id, status=SessionStatus.RUNNING.value)
            self.store.append_event(session_id, "tool.started", call.to_dict())
        context = ToolContext(
            session_id, Path(session["workspace_root"]), self.runtime_root, self.artifact_root, self.store
        )
        result = self.registry.invoke(tool_name, decision.normalized_arguments, context)
        result.tool_call_id = call.id
        call.status, call.completed_at = ("completed" if result.ok else "failed"), utc_now()
        self.store.put_tool_call(call, result)
        self.store.append_event(session_id, f"tool.{call.status}", result.to_dict())
        self.store.checkpoint(session_id, {"phase": "observation_confirmed", "tool_call_id": call.id, "ok": result.ok})
        return {"status": call.status, "result": result.to_dict(), "decision": decision.to_dict()}

    def decide_approval(self, approval_id: str, approved: bool, note: str = "") -> dict[str, Any]:
        approval = self.store.decide_approval(approval_id, approved, note)
        call = self.store.get_tool_call(approval["tool_call_id"])
        if not call:
            raise KeyError(approval["tool_call_id"])
        if not approved:
            self.store.update_session(approval["session_id"], status=SessionStatus.PAUSED.value)
            return {"status": "rejected", "approval": approval}
        return self.invoke_tool(
            approval["session_id"],
            call["tool_name"],
            call["arguments"],
            tool_call_id=call["id"],
            approved_fingerprint=approval["argument_fingerprint"],
            idempotency_key=call["idempotency_key"],
        )

    def resume(self, session_id: str) -> dict[str, Any]:
        pending = self.store.pending_tool_call(session_id)
        if pending:
            approval = self.store.get_approval(pending["approval_id"])
            if approval and approval["status"] == "approved":
                return self.invoke_tool(
                    session_id,
                    pending["tool_name"],
                    pending["arguments"],
                    tool_call_id=pending["id"],
                    approved_fingerprint=approval["argument_fingerprint"],
                )
            return {"status": "waiting_approval", "tool_call": pending, "approval": approval}
        checkpoint = self.store.latest_checkpoint(session_id)
        self.store.update_session(session_id, status=SessionStatus.RUNNING.value)
        self.store.append_event(session_id, "session.resumed", {"checkpoint": checkpoint})
        return {"status": "resumed", "checkpoint": checkpoint}

    def cancel(self, session_id: str) -> dict[str, Any]:
        session = self.store.update_session(session_id, status=SessionStatus.CANCELLED.value)
        self.store.append_event(session_id, "session.cancelled", {})
        return session

    def message(self, session_id: str, content: str, *, max_steps: int = 12) -> dict[str, Any]:
        self.store.add_turn(session_id, "user", content)
        session = self.get_session(session_id)
        decision = self.router.route("execution", session["selected_model_policy"])
        self.store.append_event(session_id, "model.routed", decision.to_dict())
        try:
            from research_os.agent.messaging import AgentMessageClient
            from research_os.agent.messaging import ToolResult as MessageToolResult
            from research_os.agent.messaging import ToolSpec as MessageToolSpec

            client = AgentMessageClient()
            if not client.is_available():
                raise RuntimeError("no configured model provider")
            messages = [
                {"role": item["role"], "content": item["content"]} for item in self.store.list_turns(session_id)
            ]
            specs = [
                MessageToolSpec(item.name, item.description, item.input_schema)
                for item in self.registry.specs()
                if item.available
            ]
            system = (
                "You are EvoMind. Plan, use tools, verify observations, preserve evidence, and stop at approval gates."
            )
            model_execution: dict[str, Any] = {
                "provider": "",
                "model": "",
                "native_tool_loop": True,
                "native_tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "turns": 0,
            }
            for step in range(max_steps):
                turn = client.send(messages, system=system, tools=specs)
                model_execution.update(
                    {
                        "provider": turn.provider,
                        "model": turn.model,
                        "turns": step + 1,
                    }
                )
                model_execution["native_tool_calls"] += len(turn.tool_calls)
                model_execution["input_tokens"] += turn.input_tokens
                model_execution["output_tokens"] += turn.output_tokens
                self.store.append_event(
                    session_id,
                    "model.response",
                    {
                        "provider": turn.provider,
                        "model": turn.model,
                        "stop_reason": turn.stop_reason,
                        "input_tokens": turn.input_tokens,
                        "output_tokens": turn.output_tokens,
                        "tool_names": [requested.name for requested in turn.tool_calls],
                    },
                )
                messages.append({"role": "assistant", "content": turn.raw_content})
                if turn.text:
                    self.store.add_turn(session_id, "assistant", turn.text)
                    self.store.append_event(session_id, "assistant.delta", {"text": turn.text})
                if not turn.tool_calls:
                    self.store.update_session(session_id, status=SessionStatus.COMPLETED.value)
                    return {
                        "status": "completed",
                        "text": turn.text,
                        "model": decision.to_dict(),
                        "model_execution": model_execution,
                    }
                tool_results = []
                for requested in turn.tool_calls:
                    outcome = self.invoke_tool(
                        session_id, requested.name, requested.input, idempotency_key=requested.id
                    )
                    if outcome["status"] == "waiting_approval":
                        return {
                            "status": "waiting_approval",
                            "text": turn.text,
                            "model": decision.to_dict(),
                            "model_execution": model_execution,
                            **outcome,
                        }
                    result = outcome.get("result", {})
                    tool_results.append(
                        MessageToolResult(
                            requested.id,
                            json.dumps(result, ensure_ascii=False, default=str),
                            not bool(result.get("ok")),
                        )
                    )
                messages.append({"role": "user", "content": [item.to_wire() for item in tool_results]})
            self.store.update_session(session_id, status=SessionStatus.PAUSED.value)
            return {
                "status": "paused",
                "text": "tool step budget reached",
                "model": decision.to_dict(),
                "model_execution": model_execution,
            }
        except Exception as exc:
            text = f"Runtime session is durable. Model execution is unavailable: {type(exc).__name__}."
            self.store.add_turn(session_id, "assistant", text)
            self.store.append_event(session_id, "assistant.completed", {"text": text, "degraded": True})
            self.store.update_session(session_id, status=SessionStatus.PAUSED.value)
            return {"status": "paused", "text": text, "model": decision.to_dict()}
