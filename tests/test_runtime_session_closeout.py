"""A finished direct tool call must never leave its session "running" forever.

Regression cover for the 2026-09-16 closeout: raw sessions (``POST /v1/sessions``)
have no assistant Run row, so the direct-tool path is the only lifecycle writer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from evomind_runtime.http_server import _invoke_direct_tool
from evomind_runtime.models import ApprovalRequest, ToolCall, ToolResult, new_id, utc_now
from evomind_runtime.runtime import AgentRuntime

TOOL = "hpc_asset_probe"


def _raw_session(runtime: AgentRuntime, title: str = "closeout probe") -> str:
    session = runtime.create_session(objective="Read-only closeout probe.", title=title)
    return str(session["id"])


def _stub_tool(runtime: AgentRuntime, ok: bool) -> None:
    def handler(arguments: dict, _context) -> ToolResult:
        if ok:
            return ToolResult("", True, {"scope": arguments.get("scope")}, "stub complete")
        return ToolResult("", False, {"read_only": True}, "stub failed", error="fixture_gate")

    runtime.registry._handlers[TOOL] = handler


def _backdate(runtime: AgentRuntime, session_id: str, stamp: str) -> None:
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (stamp, session_id)
        )
        runtime.store._connection.execute(
            "UPDATE tool_calls SET created_at=?, started_at=?, completed_at=? WHERE session_id=?",
            (stamp, stamp, stamp, session_id),
        )
        runtime.store._connection.execute(
            "UPDATE events SET created_at=? WHERE session_id=?", (stamp, session_id)
        )


def test_direct_tool_completion_settles_raw_session(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=True)
        session_id = _raw_session(runtime)
        assert runtime.store.get_session(session_id)["status"] == "created"

        outcome = _invoke_direct_tool(
            runtime, session_id, TOOL, {"scope": "cure_mindgames"}, "closeout-success-1"
        )

        assert outcome["status"] == "completed"
        assert runtime.store.get_session(session_id)["status"] == "completed"
        events = [item["event_type"] for item in runtime.store.list_events(session_id)]
        assert "session.direct_tool_settled" in events
    finally:
        runtime.close()


def test_direct_tool_failure_settles_raw_session_as_failed(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=False)
        session_id = _raw_session(runtime)

        outcome = _invoke_direct_tool(
            runtime, session_id, TOOL, {"scope": "cure_mindgames"}, "closeout-failure-1"
        )

        assert outcome["status"] == "failed"
        assert runtime.store.get_session(session_id)["status"] == "failed"
    finally:
        runtime.close()


def test_direct_tool_never_resurrects_a_cancelled_session(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=True)
        session_id = _raw_session(runtime)
        runtime.cancel(session_id)

        _invoke_direct_tool(
            runtime, session_id, TOOL, {"scope": "cure_mindgames"}, "closeout-cancelled-1"
        )

        assert runtime.store.get_session(session_id)["status"] == "cancelled"
    finally:
        runtime.close()


def test_settle_orphan_sessions_reconciles_stale_running_session(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=True)
        session_id = _raw_session(runtime)
        _invoke_direct_tool(
            runtime, session_id, TOOL, {"scope": "cure_mindgames"}, "closeout-stale-1"
        )
        stale = "2026-08-01T00:00:00+00:00"
        runtime.store.update_session(session_id, status="running")
        _backdate(runtime, session_id, stale)

        receipt = runtime.settle_orphan_sessions(idle_seconds=1800)

        assert receipt["reconciled_count"] == 1
        assert receipt["reconciled"][0] == {
            "session_id": session_id,
            "from": "running",
            "to": "completed",
        }
        assert runtime.store.get_session(session_id)["status"] == "completed"
        payloads = [
            item
            for item in runtime.store.list_events(session_id)
            if item["event_type"] == "session.reconciled"
        ]
        assert payloads and payloads[-1]["payload"]["previous_status"] == "running"
    finally:
        runtime.close()


def test_settle_orphan_sessions_skips_live_and_owned_work(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=True)

        live_id = _raw_session(runtime, "live tool call")
        runtime.store.put_tool_call(
            ToolCall(new_id("call"), live_id, TOOL, {"scope": "cure_mindgames"}, status="running",
                     started_at=utc_now())
        )
        runtime.store.update_session(live_id, status="running")
        _backdate(runtime, live_id, "2026-08-01T00:00:00+00:00")

        recent_id = _raw_session(runtime, "recent activity")
        _invoke_direct_tool(runtime, recent_id, TOOL, {"scope": "cure_mindgames"}, "closeout-recent-1")
        runtime.store.update_session(recent_id, status="running")

        owned_run = runtime.assistant.create_run(
            prompt="Closeout probe that keeps its Run active.",
            conversation_id="closeout_owned",
            start=False,
        )
        owned_id = str(owned_run["session_id"])
        runtime.store.update_session(owned_id, status="running")

        receipt = runtime.settle_orphan_sessions(idle_seconds=1800)

        assert receipt["reconciled_count"] == 0
        assert receipt["skipped"].get("tool_call_active") == 1
        assert receipt["skipped"].get("recent_activity", 0) >= 1
        assert receipt["skipped"].get("run_still_owns_lifecycle", 0) >= 1
        assert runtime.store.get_session(live_id)["status"] == "running"
        assert runtime.store.get_session(recent_id)["status"] == "running"
        assert runtime.store.get_session(owned_id)["status"] == "running"
    finally:
        runtime.close()


def test_settle_orphan_sessions_respects_a_fresh_approval_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=True)
        session_id = _raw_session(runtime, "open gate")
        call = ToolCall(
            new_id("call"),
            session_id,
            TOOL,
            {"scope": "cure_mindgames"},
            status="waiting_approval",
            started_at=utc_now(),
        )
        approval = ApprovalRequest(
            new_id("approval"),
            session_id,
            call.id,
            TOOL,
            "fixture-fingerprint",
            {"scope": "cure_mindgames"},
            {},
            "low",
            True,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        )
        call.approval_id = approval.id
        runtime.store.put_tool_call(call)
        runtime.store.put_approval(approval)
        runtime.store.update_session(session_id, status="waiting_approval")
        _backdate(runtime, session_id, "2026-08-01T00:00:00+00:00")

        receipt = runtime.settle_orphan_sessions(idle_seconds=1800)

        assert receipt["reconciled_count"] == 0
        assert receipt["skipped"] == {"tool_call_active": 1}
        assert runtime.store.get_session(session_id)["status"] == "waiting_approval"
    finally:
        runtime.close()


def test_settle_orphan_sessions_blocks_on_an_expired_gate(tmp_path: Path, monkeypatch) -> None:
    """An expired gate is history, not live work: the session settles as blocked."""

    runtime = AgentRuntime(tmp_path)
    try:
        _stub_tool(runtime, ok=True)
        session_id = _raw_session(runtime, "expired gate")
        call = ToolCall(
            new_id("call"),
            session_id,
            TOOL,
            {"scope": "cure_mindgames"},
            status="waiting_approval",
            started_at=utc_now(),
        )
        approval = ApprovalRequest(
            new_id("approval"),
            session_id,
            call.id,
            TOOL,
            "fixture-fingerprint",
            {"scope": "cure_mindgames"},
            {},
            "low",
            True,
            status="expired",
            decided_at=utc_now(),
            expires_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        )
        call.approval_id = approval.id
        runtime.store.put_tool_call(call)
        runtime.store.put_approval(approval)
        runtime.store.update_session(session_id, status="waiting_approval")
        _backdate(runtime, session_id, "2026-08-01T00:00:00+00:00")

        receipt = runtime.settle_orphan_sessions(idle_seconds=1800)

        assert receipt["reconciled_count"] == 1
        assert receipt["reconciled"][0]["to"] == "blocked"
        assert runtime.store.get_session(session_id)["status"] == "blocked"
    finally:
        runtime.close()
