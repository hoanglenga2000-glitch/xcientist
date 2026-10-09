import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.aibuild_engine import AIBuildEngine
from evomind_runtime.run_control import RunPaused


def test_pause_waits_for_existing_tool_and_blocks_new_writes(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    entered, release = threading.Event(), threading.Event()
    try:
        run = runtime.assistant.create_run(prompt="pause fixture", start=False)
        invoke = runtime.registry.invoke
        def delayed(name, args, context):
            if name == "file_write":
                entered.set()
                assert release.wait(5)
            return invoke(name, args, context)
        monkeypatch.setattr(runtime.registry, "invoke", delayed)
        results = []
        worker = threading.Thread(target=lambda: results.append(runtime.invoke_tool(run["id"], "file_write", {"path": "one.txt", "content": "once"}, idempotency_key="one")))
        worker.start()
        assert entered.wait(5)
        state = runtime.assistant.action(run["id"], "pause")
        assert state["status"] == "pausing"
        blocked = runtime.invoke_tool(run["id"], "file_write", {"path": "two.txt", "content": "not yet"}, idempotency_key="two")
        assert blocked == {"status": "paused", "error": "user_pause_requested", "dispatched": False}
        assert runtime.invoke_tool(run["id"], "file_write", {"path": "one.txt", "content": "once"}, idempotency_key="one")["status"] == "running"
        with pytest.raises(ValueError, match="requires_tool_settlement"):
            runtime.clear_user_pause(run["id"])
        release.set()
        worker.join(5)
        assert not worker.is_alive() and results[0]["status"] == "completed"
        assert runtime.assistant.snapshot(run["id"])["status"] == "paused"
        assert runtime.assistant.snapshot(run["id"])["progress"]["phase"] == "user_paused"
        assert runtime.invoke_tool(run["id"], "file_write", {"path": "one.txt", "content": "once"}, idempotency_key="one")["replayed"] is True
        assert len(runtime.store.list_tool_calls(run["id"])) == 1
        seen = []
        runtime.store.update_assistant_run(run["id"], retry_count=2)
        monkeypatch.setattr(runtime.assistant, "start", lambda *args, **kwargs: seen.append((args, kwargs)) or True)
        runtime.assistant.action(run["id"], "resume")
        assert not runtime.user_pause_requested(run["id"])
        assert runtime.store.get_assistant_run(run["id"])["retry_count"] == 2
        assert len(seen) == 1
    finally:
        release.set()
        runtime.close()


def test_pause_survives_runtime_reopen_and_no_model_call(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="pause across restart", start=False)
    runtime.assistant.action(run["id"], "pause")
    runtime.close()
    runtime = AgentRuntime(tmp_path)
    try:
        from research_os.agent import messaging
        monkeypatch.setattr(messaging, "_post_json", lambda *_: pytest.fail("paused Run must not call a model"))
        assert runtime.assistant.start(run["id"]) is False
        assert runtime.message(run["id"], "continue automatically")["status"] == "paused"
        assert runtime.assistant.snapshot(run["id"])["status"] == "paused"
    finally:
        runtime.close()


def test_approval_during_pause_is_recorded_without_dispatch(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="approval pause fixture", start=False)
        pending = runtime.invoke_tool(run["id"], "file_delete", {"path": "nonexistent-fixture.txt"}, idempotency_key="approval")
        assert pending["status"] == "waiting_approval"
        runtime.assistant.action(run["id"], "pause")
        monkeypatch.setattr(runtime.registry, "invoke", lambda *_: pytest.fail("paused approval must not execute"))
        answer = runtime.assistant.decide_approval(pending["approval"]["id"], True, background=True)
        assert answer["status"] == "paused" and answer["execution_enqueued"] is False
        assert runtime.store.get_approval(pending["approval"]["id"])["status"] == "approved"
        assert runtime.assistant.snapshot(run["id"])["status"] == "paused"
    finally:
        runtime.close()


def test_aibuild_batch_resume_keeps_all_results_and_never_rewrites_first_file(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="batch journal fixture", start=False)
        engine = AIBuildEngine(runtime, runtime.get_session(run["id"]))
        original = runtime.registry.invoke
        writes = []
        def invoke(name, args, context):
            result = original(name, args, context)
            if name == "file_write":
                writes.append(args["path"])
                if len(writes) == 1:
                    runtime.pause(run["id"])
            return result
        monkeypatch.setattr(runtime.registry, "invoke", invoke)
        pending = {"messages": [{"role": "assistant", "content": []}], "remaining_calls": [
            {"id": "first", "name": "file_write", "input": {"path": "first.txt", "content": "first"}, "key": "first"},
            {"id": "second", "name": "file_write", "input": {"path": "second.txt", "content": "second"}, "key": "second"},
        ], "completed_results": [], "tool_outcomes": []}
        with pytest.raises(RunPaused):
            engine._settle_tool_batch("coder", "CoderAgent", pending)
        journal_path = engine.path / "coder.pending.json"
        journal = json.loads(journal_path.read_text())
        assert [item["id"] for item in journal["remaining_calls"]] == ["second"]
        assert len(journal["completed_results"]) == 1
        first = Path(runtime.get_session(run["id"])["workspace_root"]) / "first.txt"
        before = first.stat().st_mtime_ns
        runtime.clear_user_pause(run["id"])
        messages = engine._settle_tool_batch("coder", "CoderAgent", journal)
        assert len(messages[-1]["content"]) == 2
        assert [Path(value).name for value in writes] == ["first.txt", "second.txt"]
        assert first.stat().st_mtime_ns == before
        assert not journal_path.exists()
        assert len(json.loads((engine.path / "coder.tool-ledger.json").read_text())) == 2
    finally:
        runtime.close()


def test_cooperative_role_pause_does_not_complete_or_fail_the_graph(tmp_path):
    from research_os.agent.multi_agent import AgentTask, AgentRoleSpec, AgentResult, MultiAgentStore, MultiAgentSupervisor, create_run
    paused = [True]
    calls = []
    def execute(task, _handoff, _run):
        if paused[0]:
            raise RunPaused()
        calls.append(task.task_id)
        return AgentResult(task.task_id, "done")
    graph = create_run(run_id="pause_graph", objective="pause graph fixture",
        tasks=[AgentTask(task_id="a", role="Coder", goal="a"), AgentTask(task_id="b", role="Coder", goal="b", dependencies=["a"])],
        roles=[AgentRoleSpec(role="Coder", capabilities=("coding",))], max_concurrency=1)
    supervisor = MultiAgentSupervisor(graph, MultiAgentStore(tmp_path), {"Coder": execute})
    outcome = supervisor.run_until_blocked()
    assert outcome.status == "paused"
    assert outcome.tasks["a"].status == "ready" and not calls
    assert not any(task.status == "failed" for task in outcome.tasks.values())
    paused[0] = False
    supervisor.resume()
    assert supervisor.run_until_blocked().status == "completed"
    assert calls == ["a", "b"]


def test_model_failures_do_not_retry_the_entire_run(monkeypatch):
    from evomind_runtime.assistant_runs import _failure_policy
    monkeypatch.setenv("EVOMIND_ASSISTANT_AUTO_RETRY", "1")
    assert _failure_policy(RuntimeError("LLMError/http_503")) == ("recoverable", False)


def test_cancelled_paused_run_does_not_revert_to_paused_or_dispatch(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="pause then cancel", start=False)
        runtime.assistant.action(run["id"], "pause")
        state = runtime.assistant.action(run["id"], "cancel")
        assert state["status"] == "cancelled"
        assert not runtime.user_pause_requested(run["id"])
        assert runtime.message(run["id"], "continue")["status"] == "cancelled"
        result = runtime.invoke_tool(run["id"], "file_write", {"path": "unexpected.txt", "content": "no"})
        assert result["status"] == "cancelled" and result["dispatched"] is False
    finally:
        runtime.close()


def test_generic_journal_resumes_remaining_native_calls_without_repeating(tmp_path, monkeypatch):
    from evomind_runtime.message_journal import MessageJournal
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective="ordinary tool batch", workspace_root=str(tmp_path / "task"))
        Path(session["workspace_root"]).mkdir(exist_ok=True)
        journal = MessageJournal(runtime, session["id"])
        original = runtime.registry.invoke
        writes = []
        def invoke(name, arguments, context):
            result = original(name, arguments, context)
            if name == "file_write":
                writes.append(Path(arguments["path"]).name)
                if len(writes) == 1:
                    runtime.pause(session["id"])
            return result
        monkeypatch.setattr(runtime.registry, "invoke", invoke)
        calls = [SimpleNamespace(id=name, name="file_write", input={"path": name+".txt", "content": name}) for name in ("one", "two")]
        journal.begin([{"role": "assistant", "content": []}], calls, {"file_write"})
        assert journal.settle()["status"] == "paused"
        runtime.clear_user_pause(session["id"])
        resumed = MessageJournal(runtime, session["id"]).settle()
        assert resumed["status"] == "settled"
        assert len(resumed["messages"][-1]["content"]) == 2
        assert writes == ["one.txt", "two.txt"]
        assert journal.settle() is None
    finally:
        runtime.close()
