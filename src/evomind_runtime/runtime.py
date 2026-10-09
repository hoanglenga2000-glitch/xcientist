from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .model_router import ModelRouter
from .models import ApprovalRequest, PermissionLevel, Session, SessionStatus, ToolCall, ToolResult, new_id, utc_now
from .policy import PolicyEngine, argument_fingerprint
from .store import RuntimeStore
from .tools import ToolContext, ToolRegistry, build_default_registry
from .assistant_runs import AssistantRunService
from .super_agent_runtime import SuperAgentMode, SuperAgentRuntime
from .tenant_access import register_scoped_session, scoped_session_metadata
from .execution_progress import ExecutionHeartbeat, ProgressStore, safe_text


def _message_max_output_tokens(session: dict[str, Any], client: Any) -> int:
    """Bound ordinary Flash replies without changing calibration allowances.

    A confirmed per-Run request amendment still takes precedence in the
    governed transport. This does not change the pinned model contract.
    """
    if {"official_calibration", "siim_calibration"} & set(session.get("metadata") or {}):
        return 8192
    # Hotfix D9: raise the ordinary Flash output budget. A long planning turn for a
    # large-data training objective still hit model_output_truncated at 16384 tokens
    # (the calibration path already uses 32768, so the provider supports it).
    return 32768 if getattr(client, "model", "") == "deepseek-flash" else 4096


class AgentRuntime:
    def __init__(
        self, workspace_root: str | Path, runtime_root: str | Path | None = None, registry: ToolRegistry | None = None
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.runtime_root = Path(runtime_root or self.workspace_root / "workspace" / "runtime").resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.runtime_root / "artifacts"
        self.store = RuntimeStore(self.runtime_root / "runtime.sqlite3")
        self.progress = ProgressStore(self.runtime_root / "runtime.sqlite3")
        self.super_agent = SuperAgentRuntime(
            workspace_root=self.workspace_root,
            runtime_root=self.runtime_root,
            tenant_id=os.getenv("EVOMIND_TENANT_ID", "local"),
            project_id=os.getenv("EVOMIND_PROJECT_ID", self.workspace_root.name or "default"),
        )
        self.registry = registry or build_default_registry()
        self.super_agent.ingest_runtime_tools(self.registry.specs())
        self.policy = PolicyEngine()
        self.router = ModelRouter()
        # ThreadingHTTPServer can deliver duplicate retries concurrently.  Keep
        # the idempotency lookup and the initial durable state transition
        # atomic so only one request is allowed to execute a side effect.
        self._invoke_lock = threading.RLock()
        self.assistant = AssistantRunService(self)
        from .report_jobs import ReportJobs
        self.reports = ReportJobs(self)
        self._close_lock = threading.RLock()
        self._closed = False

    def close(self, timeout: float = 2.0) -> bool:
        if self._closed:
            return True
        if not self.assistant.shutdown(timeout):
            return False
        if not self.reports.shutdown(timeout):
            return False
        with self._close_lock:
            if not self._closed:
                self.super_agent.close()
                self.store.close()
                self._closed = True
        return True

    def create_session(
        self,
        *,
        objective: str = "",
        title: str = "",
        permission_level: str = PermissionLevel.WORKSPACE_WRITE.value,
        workspace_root: str | None = None,
        parent_session_id: str = "",
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        PermissionLevel(permission_level)
        metadata = scoped_session_metadata(metadata)
        from .aibuild_engine import use_aibuild
        session_metadata = dict(metadata or {})
        if use_aibuild(objective, session_metadata, self.runtime_root):
            session_metadata["research_orchestrator"] = "aibuildai2"
            from .training_control import load_policy
            control = load_policy(self.runtime_root)
            if control is not None:
                session_metadata["training_control_policy_sha256"] = control["policy_sha256"]
        session = Session(
            session_id or new_id("session"),
            str(Path(workspace_root or self.workspace_root).resolve()),
            permission_level,
            title=title or objective[:80],
            objective=objective,
            parent_session_id=parent_session_id,
            metadata=session_metadata,
        )
        created = self.store.create_session(session).to_dict()
        register_scoped_session(self.runtime_root, created["id"])
        session_tenant_id, session_project_id = self.super_agent.session_identity(created.get("metadata"))
        directory_id = self.super_agent.ensure_session_workspace(
            session_id=created["id"],
            workspace_root=created["workspace_root"],
            tenant_id=session_tenant_id,
            project_id=session_project_id,
        )
        next_metadata = dict(created.get("metadata") or {})
        next_metadata["super_agent_directory_ids"] = [directory_id]
        return self.store.update_session(created["id"], metadata_json=next_metadata)

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if not session:
            raise KeyError(session_id)
        metadata = dict(session.get("metadata") or {})
        session_tenant_id, session_project_id = self.super_agent.session_identity(metadata)
        directory_id = self.super_agent.ensure_session_workspace(
            session_id=session["id"],
            workspace_root=session["workspace_root"],
            tenant_id=session_tenant_id,
            project_id=session_project_id,
        )
        if metadata.get("super_agent_directory_ids") != [directory_id]:
            metadata["super_agent_directory_ids"] = [directory_id]
            session = self.store.update_session(session["id"], metadata_json=metadata)
        return session

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_sessions(limit)

    def ensure_fixed_goal(
        self,
        *,
        run_id: str,
        spec: dict[str, Any],
        board: dict[str, Any],
        status: str = "blocked",
    ) -> dict[str, Any]:
        """Bind the fixed G21 Goal to an existing Run without creating a Run."""

        from .competition_goal import FIXED_RUN_ID, sha256_json
        from .goal_board import GoalRecordError, ensure_fixed_goal

        if run_id != FIXED_RUN_ID:
            raise GoalRecordError("goal run must be the fixed G21 Run")
        session = self.store.get_session(run_id)
        if session is None:
            raise KeyError(run_id)
        expected_metadata = {
            "goal_id": "goal_g21_five_competition",
            "goal_key": "g21_five_competition",
            "goal_spec_sha256": sha256_json(spec),
            "goal_board_sha256": sha256_json(board),
        }
        metadata = dict(session.get("metadata") or {})
        for key, value in expected_metadata.items():
            existing_value = metadata.get(key)
            if existing_value not in (None, value):
                raise GoalRecordError(f"existing session metadata conflicts with {key}")

        result = ensure_fixed_goal(
            self.store,
            run_id=run_id,
            spec=spec,
            board=board,
            status=status,
        )
        record = result["record"]
        return result

    def get_goal(self, run_id: str, goal_key: str = "") -> dict[str, Any] | None:
        return self.store.get_goal_for_run(run_id, goal_key or "g21_five_competition")

    def update_fixed_goal(self, *, goal_id: str, board: dict[str, Any], status: str | None = None) -> dict[str, Any]:
        """Update the fixed Goal board through its validated, evented path."""

        from .competition_goal import FIXED_GOAL_ID, sha256_json
        from .goal_board import GoalRecordError, update_fixed_goal_board

        if goal_id != FIXED_GOAL_ID:
            raise GoalRecordError("goal identity is not the fixed G21 goal")
        current = self.store.get_goal_record(goal_id)
        if current is None:
            raise KeyError(goal_id)
        session = self.store.get_session(str(current["run_id"]))
        if session is None:
            raise KeyError(str(current["run_id"]))
        metadata = dict(session.get("metadata") or {})
        for key, expected in (("goal_id", goal_id), ("goal_key", current["goal_key"])):
            if metadata.get(key) not in (None, expected):
                raise GoalRecordError(f"existing session metadata conflicts with {key}")
        if metadata.get("goal_board_sha256") not in (None, current["board_sha256"]):
            raise GoalRecordError("existing session metadata conflicts with goal_board_sha256")
        return update_fixed_goal_board(
            self.store,
            goal_id=goal_id,
            board=board,
            status=status,
            session_metadata={
                "goal_id": goal_id,
                "goal_key": current["goal_key"],
                "goal_spec_sha256": current["spec_sha256"],
                "goal_board_sha256": sha256_json(board),
                "goal_human_baseline_sha256": str(
                    board.get("human_baseline_evidence_sha256")
                    or current.get("human_baseline_sha256")
                    or ""
                ),
            },
        )

    def migrate_fixed_goal_policy(
        self,
        *,
        goal_id: str,
        run_id: str,
        expected_spec_sha256: str,
        new_spec: dict[str, Any],
        policy_evidence: dict[str, Any],
        migration_id: str,
    ) -> dict[str, Any]:
        """Migrate only the fixed Goal policy through the atomic store path."""

        from .competition_goal import FIXED_GOAL_ID, FIXED_RUN_ID
        from .goal_board import GoalRecordError

        if goal_id != FIXED_GOAL_ID or run_id != FIXED_RUN_ID:
            raise GoalRecordError("goal policy migration identity mismatch")
        current = self.store.get_goal_record(goal_id)
        if current is None:
            raise KeyError(goal_id)
        if current.get("run_id") != run_id:
            raise GoalRecordError("goal policy migration cross-run binding rejected")
        record, migrated = self.store.migrate_goal_policy(
            goal_id,
            run_id=run_id,
            expected_spec_sha256=expected_spec_sha256,
            new_spec=new_spec,
            policy_evidence=policy_evidence,
            migration_id=migration_id,
        )
        return {"record": record, "migrated": migrated}

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
        pre_decision = self.policy.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            permission_level=session["permission_level"],
            workspace_root=session["workspace_root"],
            approved_fingerprint="",
        )
        if tool_call_id:
            call_id = tool_call_id
        elif idempotency_key:
            stable_key = argument_fingerprint("idempotency", {"session_id": session_id, "key": idempotency_key})
            call_id = f"call_{stable_key[:32]}"
        else:
            call_id = new_id("call")

        if idempotency_key:
            try:
                with self._invoke_lock:
                    prior = self.store.get_idempotent_tool_call(session_id, idempotency_key)
                if prior:
                    call_id = str(prior["id"])
            except ValueError:
                result = ToolResult(call_id, False, {}, "Idempotency history requires reconciliation; no action was started", error="idempotency_conflict")
                return {"status": "failed", "result": result.to_dict(), "replayed": True}

        approval_verified = False
        approval_id = ""
        verified_fingerprint = ""
        if approved_fingerprint:
            candidate_call = self.store.get_tool_call(call_id)
            candidate_approval = (
                self.store.get_approval(str(candidate_call.get("approval_id") or ""))
                if candidate_call and candidate_call.get("approval_id")
                else None
            )
            expires_at = datetime.min.replace(tzinfo=timezone.utc)
            if candidate_approval:
                try:
                    expires_at = datetime.fromisoformat(
                        str(candidate_approval.get("expires_at") or "").replace("Z", "+00:00")
                    )
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            approval_verified = bool(
                candidate_call
                and candidate_approval
                and candidate_call.get("session_id") == session_id
                and candidate_call.get("tool_name") == tool_name
                and candidate_call.get("arguments") == pre_decision.normalized_arguments
                and candidate_approval.get("session_id") == session_id
                and candidate_approval.get("tool_call_id") == call_id
                and candidate_approval.get("tool_name") == tool_name
                and candidate_approval.get("normalized_arguments") == pre_decision.normalized_arguments
                and candidate_approval.get("status") == "approved"
                and candidate_approval.get("argument_fingerprint") == approved_fingerprint
                and expires_at > datetime.now(timezone.utc)
            )
            if approval_verified:
                approval_id = str(candidate_approval["id"])
                verified_fingerprint = approved_fingerprint

        decision = self.policy.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            permission_level=session["permission_level"],
            workspace_root=session["workspace_root"],
            approved_fingerprint=verified_fingerprint,
        )
        from .personal_tool_boundary import constrain
        pre_decision = constrain(session, tool_name, pre_decision)
        decision = constrain(session, tool_name, decision)

        def admit(call, result=None, approval=None):
            spec = self.registry.get(tool_name)
            try:
                claimed, saved = self.store.admit_tool_call(
                    call, result, approval=approval, approved_fingerprint=verified_fingerprint,
                    allow_paused=bool(spec and spec.read_only),
                )
            except ValueError:
                failure = ToolResult(call.id, False, {}, "Tool admission requires reconciliation; no action was started",
                                     error="idempotency_conflict")
                return {"status": "failed", "result": failure.to_dict(), "replayed": True, "dispatched": False}
            if claimed:
                return None
            if "id" not in saved:
                return saved
            replay = {"status": saved["status"], "tool_call": saved, "replayed": True}
            if saved["status"] in {"completed", "failed"}:
                replay["result"] = saved["result"]
            elif saved["status"] == "waiting_approval":
                replay["approval"] = self.store.get_approval(saved["approval_id"])
            return replay

        with self._invoke_lock:
            if idempotency_key:
                try:
                    latest = self.store.get_idempotent_tool_call(session_id, idempotency_key)
                except ValueError:
                    result = ToolResult(call_id, False, {}, "Idempotency history requires reconciliation; no action was started", error="idempotency_conflict")
                    return {"status": "failed", "result": result.to_dict(), "replayed": True}
                if latest and latest["id"] != call_id:
                    call_id = str(latest["id"])
                    approval_verified, approval_id, verified_fingerprint = False, "", ""
                    decision = pre_decision
            existing = self.store.get_tool_call(call_id)
            if existing:
                same_request = (
                    existing["session_id"] == session_id
                    and existing["tool_name"] == tool_name
                    and existing["arguments"] == decision.normalized_arguments
                    and (not idempotency_key or existing.get("idempotency_key") == idempotency_key)
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
                if existing["status"] == "waiting_approval" and not approval_verified:
                    approval = self.store.get_approval(existing["approval_id"])
                    approval_status = str((approval or {}).get("status") or "")
                    try:
                        deadline = datetime.fromisoformat(str((approval or {}).get("expires_at") or "").replace("Z", "+00:00"))
                        if deadline.tzinfo is None:
                            deadline = deadline.replace(tzinfo=timezone.utc)
                    except ValueError:
                        deadline = datetime.min.replace(tzinfo=timezone.utc)
                    if approval_status in {"pending", "approved"} and deadline > datetime.now(timezone.utc):
                        return {"status": "waiting_approval", "tool_call": existing, "approval": approval, "replayed": True}
                    if approval_status == "pending":
                        self.store.decide_approval(existing["approval_id"], False, "expired before replay")
                    # Expired/rejected gates are immutable evidence. A
                    # same-request retry must get a fresh call/approval.
                    call_id = new_id("call")
                    existing = None
                if existing and existing["status"] == "running":
                    return {"status": "running", "tool_call": existing, "replayed": True}

            if self.get_session(session_id)["status"] == "cancelled":
                return {"status": "cancelled", "error": "user_cancel_requested", "dispatched": False}
            if self.user_pause_requested(session_id):
                spec = self.registry.get(tool_name)
                if spec is None or not spec.read_only:
                    return {"status": "paused", "error": "user_pause_requested", "dispatched": False}

            call = ToolCall(
                call_id, session_id, tool_name, decision.normalized_arguments, idempotency_key=idempotency_key
            )
            if approval_verified:
                call.approval_id = approval_id
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
                replay = admit(call, approval=approval)
                if replay is not None:
                    return replay
                self.store.update_session(session_id, status=SessionStatus.WAITING_APPROVAL.value)
                self.store.append_event(session_id, "approval.requested", approval.to_dict())
                self.store.checkpoint(
                    session_id, {"phase": "waiting_approval", "tool_call_id": call.id, "approval_id": approval.id}
                )
                return {"status": "waiting_approval", "tool_call": call.to_dict(), "approval": approval.to_dict()}
            if not decision.allowed:
                error_code = "personal_tool_boundary" if decision.scope.get("personal_tool_boundary") else "cross_run_path" if decision.scope.get("cross_run_workspace") else "permission_denied"
                result = ToolResult(call.id, False, {}, decision.reason, error=error_code)
                call.status, call.completed_at = "failed", utc_now()
                replay = admit(call, result)
                if replay is not None:
                    return replay
                self.store.append_event(session_id, "tool.failed", result.to_dict())
                return {"status": "failed", "result": result.to_dict(), "decision": decision.to_dict()}
            call.status, call.started_at = "running", utc_now()
            replay = admit(call)
            if replay is not None:
                return replay
            self.store.update_session(session_id, status=SessionStatus.RUNNING.value)
            self.store.append_event(session_id, "tool.started", call.to_dict())
        context = ToolContext(
            session_id,
            Path(session["workspace_root"]),
            self.workspace_root,
            self.runtime_root,
            self.artifact_root,
            self.store,
            dict(session.get("metadata") or {}),
            self.super_agent,
            approval_verified,
            approval_id,
            verified_fingerprint,
        )
        context.reports = self.reports
        with ExecutionHeartbeat(
            self.progress, session_id, call.id,
            lambda payload: self.store.append_event(session_id, "run_progress", payload),
        ) as heartbeat:
            context.progress = heartbeat.report
            result = self.registry.invoke(tool_name, decision.normalized_arguments, context)
            heartbeat.report(phase="tool_completed" if result.ok else "tool_failed", worker_state="unverified", failure_class=safe_text(result.error))
        result.tool_call_id = call.id
        call.status, call.completed_at = ("completed" if result.ok else "failed"), utc_now()
        self.store.put_tool_call(call, result)
        self.store.append_event(session_id, f"tool.{call.status}", result.to_dict())
        self.store.checkpoint(session_id, {"phase": "observation_confirmed", "tool_call_id": call.id, "ok": result.ok})
        self.settle_user_pause(session_id)
        return {"status": call.status, "result": result.to_dict(), "decision": decision.to_dict()}

    def decide_approval(self, approval_id: str, approved: bool, note: str = "", *, execute: bool = True) -> dict[str, Any]:
        with self._invoke_lock:
            approval = self.store.decide_approval(approval_id, approved, note)
            call = self.store.get_tool_call(approval["tool_call_id"])
            if not call:
                raise KeyError(approval["tool_call_id"])
            if not approved or approval["status"] != "approved":
                self.store.update_session(approval["session_id"], status=SessionStatus.PAUSED.value)
                return {"status": approval["status"], "approval": approval}
            if not execute:
                return {"status": "approved", "approval": approval, "tool_call": call}
        # Never hold the global admission lock during a long external action.
        return self.invoke_tool(
            approval["session_id"], call["tool_name"], call["arguments"],
            tool_call_id=call["id"], approved_fingerprint=approval["argument_fingerprint"],
            idempotency_key=call["idempotency_key"],
        )

    def user_pause_requested(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        return bool((session.get("metadata") or {}).get("user_pause_requested"))

    def pause(self, session_id: str) -> dict[str, Any]:
        with self._invoke_lock:
            session = self.get_session(session_id)
            if session["status"] in {"completed", "cancelled"}:
                return session
            if self.store.set_user_pause_requested(session_id, True):
                self.store.append_event(session_id, "session.pause_requested", {"signals_sent": 0, "new_work_blocked": True})
        self.settle_user_pause(session_id)
        return self.get_session(session_id)

    def settle_user_pause(self, session_id: str) -> None:
        with self._invoke_lock:
            if not self.user_pause_requested(session_id):
                return
            run = self.store.get_assistant_run(session_id)
            if run and str(run.get("error_class") or "") in {
                "execution_reconciliation_required", "approval_execution_unconfirmed",
            }:
                return
            pending = any(call["status"] == "running" for call in self.store.list_tool_calls(session_id, limit=2000))
            worker = self.assistant._threads.get(session_id)
            # Inspect handles without acquiring the assistant lock (lock order).
            control_alive = bool(worker and worker.is_alive())
            state = "pausing" if pending or control_alive else "paused"
            previous_state = self.get_session(session_id)["status"]
            self.store.update_session(session_id, status=state)
            if self.store.get_assistant_run(session_id):
                self.store.update_assistant_run(session_id, status=state, completed_at="")
            if state == "paused" and previous_state != "paused":
                self.store.append_event(session_id, "run_paused", {"status": "paused", "signals_sent": 0, "budget_reset": False})

    def clear_user_pause(self, session_id: str) -> None:
        with self._invoke_lock:
            if any(call["status"] == "running" for call in self.store.list_tool_calls(session_id, limit=2000)):
                raise ValueError("pause_resume_requires_tool_settlement")
            if self.store.set_user_pause_requested(session_id, False):
                self.store.update_session(session_id, status="recovering")
                self.store.append_event(session_id, "session.pause_released", {"budget_reset": False, "tools_replayed": False})

    def resume(self, session_id: str) -> dict[str, Any]:
        if self.user_pause_requested(session_id):
            self.clear_user_pause(session_id)
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
                    idempotency_key=pending.get("idempotency_key", ""),
                )
            return {"status": "waiting_approval", "tool_call": pending, "approval": approval}
        checkpoint = self.store.latest_checkpoint(session_id)
        self.store.update_session(session_id, status=SessionStatus.RUNNING.value)
        self.store.append_event(session_id, "session.resumed", {"checkpoint": checkpoint})
        return {"status": "resumed", "checkpoint": checkpoint}

    def cancel(self, session_id: str) -> dict[str, Any]:
        with self._invoke_lock:
            self.store.set_user_pause_requested(session_id, False)
            session = self.store.update_session(session_id, status=SessionStatus.CANCELLED.value)
        self.store.append_event(session_id, "session.cancelled", {})
        return session

    # Statuses that claim work is in flight.  A session may only rest in one of
    # them while some piece of live evidence supports it (an active tool call, a
    # fresh approval gate, or an assistant Run that still owns the lifecycle).
    ACTIVE_SESSION_STATUSES = (
        SessionStatus.QUEUED.value,
        SessionStatus.PLANNING.value,
        SessionStatus.RUNNING.value,
        SessionStatus.VERIFYING.value,
        SessionStatus.WAITING_APPROVAL.value,
        SessionStatus.RECOVERING.value,
    )
    ACTIVE_RUN_STATUSES = (
        SessionStatus.CREATED.value,
        SessionStatus.QUEUED.value,
        SessionStatus.PLANNING.value,
        SessionStatus.RUNNING.value,
        SessionStatus.VERIFYING.value,
        SessionStatus.WAITING_APPROVAL.value,
        SessionStatus.RECOVERING.value,
    )

    @staticmethod
    def _timestamp(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def settle_orphan_sessions(self, *, idle_seconds: int = 1800, limit: int = 5000) -> dict[str, Any]:
        """Settle sessions that still claim activity but hold no live work.

        A session is reconciled only when every piece of evidence agrees that
        nothing is in flight:

        * its status claims activity (running / planning / verifying / queued /
          recovering / waiting_approval);
        * no tool call is effectively active (the assistant's own definition
          already ignores gates whose approval expired or was rejected);
        * no approval gate is pending and unexpired;
        * the assistant Run that owns it, when one exists, has already stopped;
        * the newest recorded activity is older than ``idle_seconds``.

        The settled status mirrors the evidence: the owning Run's resting status
        when the Run exists, otherwise the latest tool-call outcome, otherwise
        ``blocked`` for an undecided gate.  Nothing is deleted; every change is
        recorded as a ``session.reconciled`` event so the repair stays auditable.
        """

        now = datetime.now(timezone.utc)
        reconciled: list[dict[str, Any]] = []
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        # Approval rows use "pending" for an undecided gate; only an unexpired one
        # still blocks its session from being settled.
        fresh_gates: set[str] = set()
        for approval in self.store.list_approvals(status="pending", limit=100_000):
            deadline = self._timestamp(approval.get("expires_at"))
            if deadline is None or deadline > now:
                fresh_gates.add(str(approval.get("session_id") or ""))

        for session in self.store.list_sessions(limit=limit):
            session_id = str(session.get("id") or "")
            status = str(session.get("status") or "")
            if not session_id or status not in self.ACTIVE_SESSION_STATUSES:
                continue
            calls = self.store.list_tool_calls(session_id, limit=2000)
            if self.assistant._effective_active_tool_calls(session_id, calls):
                skip("tool_call_active")
                continue
            if session_id in fresh_gates:
                skip("approval_pending")
                continue
            run = self.store.get_assistant_run(session_id)
            run_status = str((run or {}).get("status") or "")
            if run and run_status in self.ACTIVE_RUN_STATUSES:
                skip("run_still_owns_lifecycle")
                continue
            # ``updated_at`` already covers creation (every create writes it),
            # so activity is measured from it plus the tool-call timeline.
            stamps = [self._timestamp(session.get("updated_at"))]
            for call in calls:
                stamps.append(self._timestamp(call.get("completed_at")))
                stamps.append(self._timestamp(call.get("started_at")))
                stamps.append(self._timestamp(call.get("created_at")))
            newest = max((stamp for stamp in stamps if stamp is not None), default=None)
            if newest is not None and (now - newest).total_seconds() < idle_seconds:
                skip("recent_activity")
                continue
            target = self._settled_session_status(run_status, calls)
            if target == status:
                skip("already_settled")
                continue
            self.store.update_session(session_id, status=target)
            self.store.append_event(
                session_id,
                "session.reconciled",
                {
                    "previous_status": status,
                    "status": target,
                    "reason": "no_live_work",
                    "run_status": run_status,
                    "tool_call_count": len(calls),
                    "last_activity_at": newest.isoformat() if newest else "",
                    "idle_seconds": idle_seconds,
                },
            )
            reconciled.append({"session_id": session_id, "from": status, "to": target})
        return {
            "checked_at": now.isoformat(),
            "reconciled_count": len(reconciled),
            "reconciled": reconciled,
            "skipped": dict(sorted(skipped.items())),
        }

    def _settled_session_status(self, run_status: str, calls: list[dict[str, Any]]) -> str:
        """Return the status that current evidence supports for a resting session."""

        if run_status:
            return run_status
        if calls:
            latest = max(
                calls,
                key=lambda item: (str(item.get("created_at") or ""), str(item.get("id") or "")),
            )
            latest_status = str(latest.get("status") or "")
            if latest_status == "completed":
                return SessionStatus.COMPLETED.value
            if latest_status == "failed":
                return SessionStatus.FAILED.value
            if latest_status == "waiting_approval":
                return SessionStatus.BLOCKED.value
            return SessionStatus.FAILED.value
        return SessionStatus.BLOCKED.value

    def _message_tool_specs(
        self,
        content: str,
        session: dict[str, Any],
        extra_names: set[str] | None = None,
    ) -> list[Any]:
        if self.super_agent.mode is SuperAgentMode.OFF:
            specs = [item for item in self.registry.specs_for_prompt(content) if item.available]
        else:
            essential = {
                "super_agent_status", "capability_discover", "ecosystem_refresh", "connector_health",
                "directory_list", "directory_stat", "directory_read", "directory_hash",
                "directory_mkdir", "directory_write_atomic", "directory_copy", "directory_sync", "directory_delete",
                "tool_synthesize", "tool_test", "tool_canary", "tool_promote", "tool_reject",
                "transfer_fetch", "job_execute", "job_status", "job_cancel",
                "file_list", "file_search", "file_read", "file_write", "file_patch",
                "shell_exec", "process_start", "process_list", "process_poll", "process_log", "process_stdin", "process_cancel",
                "attachment_list", "attachment_read", "artifact_publish", "artifact_list", "artifact_preview", "artifact_bundle",
                "runtime_health", "research_capabilities", "verified_context", "skill_list", "training_route",
                "managed_model_prepare", "artifact_import",
                "model_protocols", "managed_tensor_train",
                "report_generate", "report_status",
            }
            discovered = {
                item["descriptor"]["capability_id"]
                for item in self.super_agent.discover(
                    content,
                    healthy_only=True,
                    run_id=str(session.get("id") or "") or None,
                    workspace_root=session.get("workspace_root"),
                    metadata=session.get("metadata") if isinstance(session.get("metadata"), dict) else {},
                    limit=80,
                )
            }
            selected = essential | discovered | set(extra_names or ())
            specs = [item for item in self.registry.specs() if item.available and item.name in selected]
        metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
        allowed = {
            str(value)
            for value in (metadata.get("run_allowed_tool_names") or [])
            if str(value)
        }
        if allowed:
            specs = [item for item in specs if item.name in allowed]
        return specs

    def message(self, session_id: str, content: str, *, max_steps: int = 12) -> dict[str, Any]:
        if self.get_session(session_id)["status"] == "cancelled":
            return {"status": "cancelled", "text": "Task cancelled; no new work was dispatched."}
        if self.user_pause_requested(session_id):
            return {"status": "paused", "text": "Task paused by the user; no new work was dispatched."}
        self.store.add_turn(session_id, "user", content)
        session = self.get_session(session_id)
        if (session.get("metadata") or {}).get("research_orchestrator") == "aibuildai2":
            from .aibuild_engine import AIBuildEngine
            try:
                return AIBuildEngine(self, session).execute()
            except Exception as exc:
                import re
                detail = str(exc)
                code = detail if re.fullmatch(r"[A-Za-z0-9_:-]{1,160}", detail) else safe_text(type(exc).__name__)
                self.store.append_event(session_id, "research_outcome", {"status": "blocked", "error_class": code, "objective_outcome": "not_verified"})
                self.store.update_session(session_id, status="blocked")
                return {"status": "blocked", "text": f"Research state retained; orchestration requires attention ({code}).", "error_class": code}
        if self.super_agent.mode is not SuperAgentMode.OFF:
            try:
                kernel = self.super_agent.shadow_plan(
                    run_id=session_id,
                    objective=str(session.get("objective") or content),
                    workspace_root=session.get("workspace_root"),
                    metadata=session.get("metadata") if isinstance(session.get("metadata"), dict) else {},
                )
                self.store.append_event(
                    session_id,
                    "super_agent.shadow_planned",
                    {
                        "schema": "evomind.super_agent_shadow.v1",
                        "mode": self.super_agent.mode.value,
                        "graph_id": kernel.graph.id,
                        "ready_node_ids": [item.id for item in kernel.graph.ready_nodes()],
                        "capability_count": len(self.super_agent.catalog.all(run_id=session_id)),
                    },
                )
            except Exception as exc:
                self.store.append_event(
                    session_id,
                    "super_agent.shadow_failed",
                    {"error_class": type(exc).__name__, "mode": self.super_agent.mode.value},
                )
        personal_binding = (session.get('metadata') or {}).get('model_profile')
        if not personal_binding:
            decision = self.router.route("execution", session["selected_model_policy"])
            self.store.append_event(session_id, "model.routed", decision.to_dict())
        else:
            from .models import ModelDecision
            decision = ModelDecision('execution', personal_binding['provider'], personal_binding['model'], 'Explicit personal profile revision; no fallback.', [], False)
        try:
            from research_os.agent.messaging import AgentMessageClient
            from research_os.agent.messaging import ToolResult as MessageToolResult
            from research_os.agent.messaging import ToolSpec as MessageToolSpec

            from .model_transport import governed_client, bind_run_client
            def guard_model_stream():
                session_state = self.get_session(session_id)
                if session_state['status'] == 'cancelled':
                    raise RuntimeError('user_cancel_requested')
                if (session_state.get('metadata') or {}).get('user_pause_requested'):
                    from .run_control import RunPaused
                    raise RunPaused()

            def guard_model_dispatch():
                if "siim_calibration" in (self.get_session(session_id).get("metadata") or {}):
                    from .siim_calibration_control import before_model_attempt as siim_model_gate
                    siim_model_gate(self, session_id)
                if "official_calibration" in (self.get_session(session_id).get("metadata") or {}):
                    from .ev_calibration_control import before_model_attempt
                    before_model_attempt(self, session_id)
                guard_model_stream()
            from .chat_stream_transport import PublicTextStream
            text_stream = PublicTextStream(lambda value: self.store.append_event(session_id, 'assistant.stream', value), guard_model_stream)
            if personal_binding:
                from .personal_model_client import client_for_run
                client = client_for_run(self, session, before_attempt=guard_model_dispatch)
                self.store.append_event(session_id, 'model.personal_profile_bound', personal_binding)
            else:
                client = governed_client(AgentMessageClient(), lambda value: self.store.append_event(session_id, "model.transport_attempt", value),
                                         before_attempt=guard_model_dispatch, text_observer=text_stream, during_request=guard_model_stream)
                bind_run_client(self.store, session_id, client)
            if not client.is_available():
                raise RuntimeError("no configured model provider")
            from .message_journal import MessageJournal
            journal = MessageJournal(self, session_id)
            pending = journal.settle()
            if pending and pending["status"] != "settled":
                return pending
            fallback_messages = [
                {"role": item["role"], "content": item["content"]} for item in self.store.list_turns(session_id)
            ]
            messages = journal.history(fallback_messages)
            if messages is not fallback_messages:
                messages.append({"role": "user", "content": content})
            expanded_tool_names: set[str] = set()
            system = (
                "You are EvoMind, a durable task-completion agent. Plan, use tools, verify observations, preserve evidence, "
                "and continue until the objective is complete or an exact approval/input gate is reached. Work only inside "
                "the configured task workspace. Put final user-facing files in outputs/ and call artifact_publish for every "
                "deliverable. A file write is not completion until the file is re-read or tested and published. Stop at "
                "approval gates for external or irreversible effects. To request approval, issue the exact intended tool call; "
                "the runtime policy will create the approval card. Never replace a required tool call with a prose-only approval request."
                " Discover authorized capabilities before guessing a specialized tool. Address files as directory_id plus a "
                "capability-relative path; never invent connector roots or absolute remote paths."
                " Before fitting any model, call training_route with the attachment/path and the complete training request. "
                "Obey its decision: local_cpu may use CPU only; hpc_gpu must call hpc_verify and pass all five current "
                "identity samples before hpc_execute_solution. Never use a local GPU and never silently fall back to local "
                "training when an HPC route is blocked. Collect, publish, and independently verify the resulting artifacts."
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
            empty_terminal_turns = 0
            for step in range(max_steps):
                if self.user_pause_requested(session_id):
                    return {"status": "paused", "text": "Task paused at the next model-call boundary."}
                specs = [
                    MessageToolSpec(item.name, item.description, item.input_schema)
                    for item in self._message_tool_specs(content, session, expanded_tool_names)
                ]
                turn = client.send(messages, system=system, tools=specs,
                                   max_tokens=_message_max_output_tokens(session, client))
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
                if not turn.text and not turn.tool_calls:
                    published_artifacts = self.store.list_deliverables(session_id)
                    if model_execution["native_tool_calls"] and published_artifacts:
                        fallback_text = (
                            "The model returned no final prose after verified tool execution. "
                            "Use the published artifacts and run evidence as the authoritative result."
                        )
                        model_execution["completion_mode"] = "verified_artifact_fallback"
                        model_execution["verified_artifact_count"] = len(published_artifacts)
                        self.store.add_turn(session_id, "assistant", fallback_text)
                        self.store.append_event(
                            session_id,
                            "assistant.completed",
                            {
                                "text": fallback_text,
                                "degraded": True,
                                "reason": "empty_terminal_after_published_artifact",
                                "artifact_count": len(published_artifacts),
                            },
                        )
                        self.store.update_session(session_id, status=SessionStatus.COMPLETED.value)
                        return {
                            "status": "completed",
                            "text": fallback_text,
                            "model": decision.to_dict(),
                            "model_execution": model_execution,
                        }
                    empty_terminal_turns += 1
                    self.store.append_event(
                        session_id,
                        "model.empty_response",
                        {
                            "provider": turn.provider,
                            "model": turn.model,
                            "stop_reason": turn.stop_reason,
                            "attempt": empty_terminal_turns,
                        },
                    )
                    if empty_terminal_turns >= 3:
                        raise RuntimeError("model returned an empty terminal response after 3 attempts")
                    continue
                empty_terminal_turns = 0
                messages.append({"role": "assistant", "content": turn.raw_content})
                if turn.text:
                    self.store.add_turn(session_id, "assistant", turn.text)
                    self.store.append_event(session_id, "assistant.delta", {"text": turn.text})
                if not turn.tool_calls:
                    journal.save_history(messages)
                    self.store.update_session(session_id, status=SessionStatus.COMPLETED.value)
                    return {
                        "status": "completed",
                        "text": turn.text,
                        "model": decision.to_dict(),
                        "model_execution": model_execution,
                    }
                journal.begin(messages, turn.tool_calls, {spec.name for spec in specs})
                settled_batch = journal.settle()
                if settled_batch["status"] != "settled":
                    return {**settled_batch, "model": decision.to_dict(), "model_execution": model_execution}
                messages = settled_batch["messages"]
                # Discovery expands capabilities only after its persisted result.
                for requested in turn.tool_calls:
                    observed_call = self.store.get_idempotent_tool_call(session_id, requested.id)
                    result = (observed_call or {}).get("result", {})
                    if requested.name == "capability_discover" and bool(result.get("ok")):
                        content_payload = result.get("content") if isinstance(result.get("content"), dict) else {}
                        for match in content_payload.get("matches") or []:
                            descriptor = match.get("descriptor") if isinstance(match, dict) else None
                            capability_id = str(descriptor.get("capability_id") or "") if isinstance(descriptor, dict) else ""
                            if self.registry.get(capability_id) is not None:
                                expanded_tool_names.add(capability_id)
            self.store.update_session(session_id, status=SessionStatus.PAUSED.value)
            return {
                "status": "paused",
                "text": "tool step budget reached",
                "model": decision.to_dict(),
                "model_execution": model_execution,
            }
        except Exception as exc:
            if self.get_session(session_id)["status"] == "cancelled":
                return {"status": "cancelled", "text": "Task cancelled; existing receipts are preserved."}
            from .model_transport import ModelTransportError
            detail = exc.code if isinstance(exc, ModelTransportError) else type(exc).__name__
            text = f"Runtime session is durable. Model execution is unavailable: LLMError/{detail}. No tool was executed by the failed model request."
            self.store.add_turn(session_id, "assistant", text)
            contract_error = isinstance(exc, ValueError) and str(exc) in {
                'model_contract_drift', 'model_contract_invalid', 'legacy_model_contract_requires_review'}
            if isinstance(exc, ModelTransportError) or contract_error:
                detail = str(exc) if contract_error else exc.code
                self.store.append_event(session_id, 'assistant.blocked', {'error_code': detail, 'degraded': True})
                self.store.update_session(session_id, status=SessionStatus.BLOCKED.value)
                return {'status': 'blocked', 'text': text, 'error_code': detail, 'model': decision.to_dict()}
            self.store.append_event(session_id, "assistant.completed", {"text": text, "degraded": True})
            self.store.update_session(session_id, status=SessionStatus.PAUSED.value)
            return {"status": "paused", "text": text, "model": decision.to_dict()}
