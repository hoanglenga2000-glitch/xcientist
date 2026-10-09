from copy import deepcopy

from evomind_runtime.assistant_history import CONTINUATION_HEADER, EXECUTION_HEADER, project_message_history
from evomind_runtime.runtime import AgentRuntime


def test_history_preserves_old_replies_and_hides_previous_answer_during_resume():
    run = {"id": "run_a", "session_id": "run_a", "prompt": "Fit one model", "status": "running", "plan": {"continuation_instructions": ["Continue with the fixed protocol"]}}
    turns = [
        {"id": "u1", "role": "user", "content": EXECUTION_HEADER + "internal fixture context\nUser objective:\nFit one model"},
        {"id": "a1", "role": "assistant", "content": "Previous permission failure"},
        {"id": "u2", "role": "user", "content": EXECUTION_HEADER + "private fixture context\n" + CONTINUATION_HEADER + "Fit one model\nActive continuation objective:\nContinue with the fixed protocol"},
    ]
    before = deepcopy(turns)
    history = project_message_history(run, turns)
    assert turns == before
    assert history["answer"] == "" and history["answer_is_current"] is False
    assert [row["content"] for row in history["messages"]] == ["Fit one model", "Previous permission failure", "Continue with the fixed protocol", ""]
    assert not any("fixture context" in row["content"] for row in history["messages"])
    turns.append({"id": "a2", "role": "assistant", "content": "Independent verification returned a result"})
    run["status"] = "completed"
    completed = project_message_history(run, turns)
    assert completed["answer_is_current"] is True
    assert completed["messages"][1] == history["messages"][1]
    assert completed["active_message_id"].endswith("a2")


def test_unknown_framework_context_and_foreign_or_tool_turns_are_not_exposed():
    run = {"id": "run_a", "prompt": "request", "status": "waiting_approval"}
    turns = [{"id": "a1", "role": "assistant", "content": "Old reply"},
             {"id": "u2", "role": "user", "content": EXECUTION_HEADER + "private-fixture-value"},
             {"id": "other", "session_id": "run_b", "role": "assistant", "content": "foreign-fixture-value"},
             {"id": "tool", "role": "tool", "content": "private-tool-fixture-value"}]
    history = project_message_history(run, turns)
    assert history["answer_is_current"] is False
    assert history["messages"][-2]["kind"] == "continued_execution"
    assert all("fixture-value" not in row["content"] for row in history["messages"])


def test_run_snapshot_projects_history_without_rewriting_durable_turns(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="History projection fixture", start=False)
        runtime.store.add_turn(run["id"], "user", "History projection fixture")
        runtime.store.add_turn(run["id"], "assistant", "Old failure")
        runtime.store.add_turn(run["id"], "user", "Continue the same task")
        runtime.store.update_assistant_run(run["id"], status="waiting_approval")
        before = runtime.store.list_turns(run["id"])
        snapshot = runtime.assistant.snapshot(run["id"])
        assert snapshot["answer"] == ""
        assert snapshot["answer_is_current"] is False
        assert any(row["content"] == "Old failure" for row in snapshot["message_history"]["messages"])
        assert runtime.store.list_turns(run["id"]) == before
    finally:
        runtime.close()


def test_executor_progress_is_forwarded_as_a_public_run_progress_event(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Progress forwarding fixture", start=False)
        value = {"schema": "evomind.execution_progress.v1", "source": "executor", "phase": "research_model_call", "heartbeat_at": 100.0}
        runtime.store.append_event(run["id"], "execution_progress", value)
        events = runtime.assistant.events(run["id"])
        event = next(row for row in events if row["event_type"] == "run_progress")
        assert event["payload"] == {"run_id": run["id"], "progress": value}
    finally:
        runtime.close()
