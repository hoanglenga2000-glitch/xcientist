import contextvars
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from evomind_runtime.models import ToolCall, ToolResult, new_id
from evomind_runtime.runtime import AgentRuntime


def pending(runtime, run, key="one-action"):
    return runtime.invoke_tool(run["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key=key)


def test_expired_approval_chain_reuses_completed_execution_and_preserves_history(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Expiry replay fixture", start=False)
        first = pending(runtime, run)
        with sqlite3.connect(runtime.runtime_root / "runtime.sqlite3") as c:
            c.execute("UPDATE approvals SET expires_at=? WHERE id=?", ("2000-01-01T00:00:00+00:00", first["approval"]["id"]))
        second = pending(runtime, run)
        assert runtime.store.get_approval(first["approval"]["id"])["status"] == "expired"
        runtime.registry.invoke = lambda *_: ToolResult("", True, {}, "isolated fixture")
        assert runtime.decide_approval(second["approval"]["id"], True)["status"] == "completed"
        extra = ToolCall(new_id("call"), run["id"], "file_delete", second["tool_call"]["arguments"], idempotency_key="one-action")
        extra.status = "waiting_approval"
        runtime.store.put_tool_call(extra)
        replay = pending(runtime, run)
        assert replay["status"] == "completed" and replay["replayed"] is True
        assert len(runtime.store.list_approvals()) == 2
        assert len(runtime.store.list_tool_calls(run["id"])) == 3
        conflict = runtime.invoke_tool(run["id"], "file_delete", {"path": "different.txt"}, idempotency_key="one-action")
        assert conflict["result"]["error"] == "idempotency_conflict"
    finally:
        runtime.close()


def test_background_approval_returns_before_tool_settles_and_preserves_context(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    started, release, resumed = threading.Event(), threading.Event(), threading.Event()
    marker = contextvars.ContextVar("approval-fixture-principal", default="missing")
    seen = []
    try:
        run = runtime.assistant.create_run(prompt="Async approval fixture", start=False)
        gate = pending(runtime, run)
        def invoke(_name, _args, context):
            seen.append((marker.get(), context.approval_verified))
            started.set()
            assert release.wait(5)
            return ToolResult("", True, {}, "isolated fixture")
        monkeypatch.setattr(runtime.registry, "invoke", invoke)
        monkeypatch.setattr(runtime.assistant, "_start_or_defer", lambda *_args, **_kwargs: resumed.set())
        token = marker.set("fixture-owner")
        try:
            before = time.monotonic()
            response = runtime.assistant.decide_approval(gate["approval"]["id"], True, background=True)
            assert time.monotonic() - before < 1
        finally:
            marker.reset(token)
        assert response["approval"]["status"] == "approved"
        assert started.wait(1) and not resumed.is_set()
        duplicate = runtime.assistant.decide_approval(gate["approval"]["id"], True, background=True)
        assert duplicate["approval"]["status"] == "approved"
        assert len(seen) == 1 and seen[0] == ("fixture-owner", True)
        release.set()
        assert resumed.wait(2)
    finally:
        release.set()
        runtime.close(timeout=3)


def test_approved_long_tool_does_not_hold_the_global_admission_lock(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    started, release = threading.Event(), threading.Event()
    try:
        run = runtime.assistant.create_run(prompt="Admission isolation fixture", start=False)
        gate = pending(runtime, run)
        def invoke(name, *_args):
            if name == "file_delete":
                started.set()
                assert release.wait(5)
            return ToolResult("", True, {}, "isolated fixture")
        monkeypatch.setattr(runtime.registry, "invoke", invoke)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runtime.decide_approval, gate["approval"]["id"], True)
            try:
                assert started.wait(1)
                other = runtime.create_session(objective="Independent read fixture")
                result = pool.submit(runtime.invoke_tool, other["id"], "file_read", {"path": "fixture.txt"}).result(timeout=1)
                assert result["status"] == "completed"
            finally:
                release.set()
            assert first.result(timeout=2)["status"] == "completed"
    finally:
        release.set()
        runtime.close()


def test_shutdown_drains_approved_workers_before_closing_store(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    started, release = threading.Event(), threading.Event()
    try:
        run = runtime.assistant.create_run(prompt="Shutdown fixture", start=False)
        gate = pending(runtime, run)
        def invoke(*_args):
            started.set()
            assert release.wait(5)
            return ToolResult("", True, {}, "isolated fixture")
        monkeypatch.setattr(runtime.registry, "invoke", invoke)
        runtime.assistant.decide_approval(gate["approval"]["id"], True, background=True)
        assert started.wait(1)
        assert runtime.close(timeout=0.01) is False
        assert runtime.store.get_approval(gate["approval"]["id"])["status"] == "approved"
    finally:
        release.set()
        assert runtime.close(timeout=3) is True


def test_recorded_unstarted_approval_recovers_once_without_new_gate(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    resumed = threading.Event()
    count = []
    try:
        run = runtime.assistant.create_run(prompt="Durable approval recovery fixture", start=False)
        gate = pending(runtime, run)
        runtime.store.update_assistant_run(run["id"], status="waiting_approval")
        assert runtime.decide_approval(gate["approval"]["id"], True, execute=False)["status"] == "approved"
        def invoke(*_args):
            count.append(1)
            return ToolResult("", True, {}, "isolated fixture")
        monkeypatch.setattr(runtime.registry, "invoke", invoke)
        monkeypatch.setattr(runtime.assistant, "_start_or_defer", lambda *_args, **_kwargs: resumed.set())
        assert run["id"] in runtime.assistant.recover_incomplete()
        assert resumed.wait(2)
        assert count == [1]
        assert len(runtime.store.list_approvals()) == 1
        assert pending(runtime, run)["status"] == "completed"
    finally:
        runtime.close(timeout=3)
