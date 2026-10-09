from __future__ import annotations

from pathlib import Path
import sys

import pytest

from evomind_runtime import tools
from evomind_runtime.benchmark import parity_status, parity_suite
from evomind_runtime.evolution import evaluate_candidate
from evomind_runtime.policy import PolicyEngine, argument_fingerprint
from evomind_runtime.runtime import AgentRuntime


def test_suite_has_required_60_task_distribution():
    suite = parity_suite()
    assert len(suite) == 60
    assert sum(item["category"] == "coding" for item in suite) == 20
    empty = parity_status([])
    assert empty["status"] == "implemented_not_parity_verified"
    assert empty["expected_records"] == 60 * 3 * 3
    assert empty["qualifying_records"] == 0
    assert empty["release_gate"] == "NO-GO"


def test_parity_gate_rejects_three_generic_completed_rows():
    runs = [
        {"tool_name": tool_name, "status": "completed", "metrics": {}}
        for tool_name in ("evomind", "claude-code", "codex")
    ]
    result = parity_status(runs)

    assert result["compared_tools"] == ["claude-code", "codex", "evomind"]
    assert result["qualifying_records"] == 0
    assert result["invalid_completed_records"] == 3
    assert result["status"] == "implemented_not_parity_verified"
    assert result["release_gate"] == "NO-GO"


def test_parity_gate_requires_complete_repeated_evidence_and_computes_advantage():
    runs = []
    scores = {"evomind": 0.9, "claude-code": 0.8, "codex": 0.85}
    for task in parity_suite():
        for tool_name, score in scores.items():
            for repetition in range(1, 4):
                runs.append(
                    {
                        "tool_name": tool_name,
                        "status": "completed",
                        "metrics": {
                            "task_id": task["id"],
                            "repetition": repetition,
                            "score": score,
                            "success": True,
                            "artifact_complete": True,
                            "unsupported_claims": 0,
                            "duration_seconds": 1.0,
                            "tool_calls_total": 2,
                            "tool_calls_succeeded": 2,
                        },
                    }
                )

    result = parity_status(runs)

    assert result["qualifying_records"] == result["expected_records"] == 540
    assert result["comparison_complete"] is True
    assert result["status"] == "parity_verified"
    assert result["advantage_status"] == "verified"
    assert result["release_gate"] == "GO"


def test_runtime_file_round_trip_and_checkpoint(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="edit")
    written = runtime.invoke_tool(session["id"], "file_write", {"path": "中文.txt", "content": "hello"})
    assert written["status"] == "completed"
    read = runtime.invoke_tool(session["id"], "file_read", {"path": "中文.txt"})
    assert read["result"]["content"]["lines"][0]["text"] == "hello"
    assert runtime.store.latest_checkpoint(session["id"])["state"]["phase"] == "observation_confirmed"
    runtime.close()


def test_exact_approval_resumes_same_tool_call(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()
    target = tmp_path / "remove.txt"
    target.write_text("data", encoding="utf-8")
    waiting = runtime.invoke_tool(session["id"], "file_delete", {"path": str(target)})
    assert waiting["status"] == "waiting_approval"
    call_id = waiting["tool_call"]["id"]
    completed = runtime.decide_approval(waiting["approval"]["id"], True, "approved exact path")
    assert completed["status"] == "completed"
    assert completed["result"]["tool_call_id"] == call_id
    assert not target.exists()
    replay = runtime.invoke_tool(session["id"], "file_delete", {"path": str(target)}, tool_call_id=call_id)
    assert replay["replayed"] is True
    runtime.close()


def test_idempotency_key_replays_completed_side_effect(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()
    arguments = {"path": "once.txt", "content": "written once"}

    first = runtime.invoke_tool(session["id"], "file_write", arguments, idempotency_key="write-once")
    second = runtime.invoke_tool(session["id"], "file_write", arguments, idempotency_key="write-once")

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert second["replayed"] is True
    assert second["result"]["tool_call_id"] == first["result"]["tool_call_id"]
    assert (tmp_path / "once.txt").read_text(encoding="utf-8") == "written once"
    runtime.close()


def test_idempotency_key_rejects_parameter_drift(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()
    first = runtime.invoke_tool(
        session["id"],
        "file_write",
        {"path": "stable.txt", "content": "original"},
        idempotency_key="stable-write",
    )
    conflict = runtime.invoke_tool(
        session["id"],
        "file_write",
        {"path": "stable.txt", "content": "changed"},
        idempotency_key="stable-write",
    )

    assert first["status"] == "completed"
    assert conflict["status"] == "failed"
    assert conflict["result"]["error"] == "idempotency_conflict"
    assert (tmp_path / "stable.txt").read_text(encoding="utf-8") == "original"
    runtime.close()


def test_workspace_write_cannot_escape_without_approval(tmp_path: Path):
    decision = PolicyEngine().evaluate(
        tool_name="file_write",
        arguments={"path": str(tmp_path.parent / "outside.txt"), "content": "x"},
        permission_level="workspace-write",
        workspace_root=tmp_path,
    )
    assert decision.requires_approval and not decision.allowed


def test_sibling_assistant_run_path_is_rejected_without_human_approval(tmp_path: Path):
    task_parent = tmp_path / "workspace" / "runtime" / "assistant_tasks"
    current_root = task_parent / "run_0123456789abcdef0123456789abcdef"
    sibling_root = task_parent / "run_0123456789abcdef0123456789abcde0"
    runtime = AgentRuntime(tmp_path / "project")
    session = runtime.create_session(
        session_id="run_0123456789abcdef0123456789abcdef",
        workspace_root=str(current_root),
        permission_level="workspace-write",
    )

    outcome = runtime.invoke_tool(
        session["id"],
        "file_write",
        {"path": str(sibling_root / "outputs" / "result.json"), "content": "{}"},
    )

    assert outcome["status"] == "failed"
    assert outcome["result"]["error"] == "cross_run_path"
    assert outcome["decision"]["scope"]["cross_run_workspace"] is True
    assert runtime.store.list_approvals(status="pending") == []
    assert not sibling_root.exists()
    runtime.close()


def test_expired_approval_retry_creates_a_fresh_gate_for_same_idempotency_key(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(permission_level="workspace-write")
    first = runtime.invoke_tool(
        session["id"],
        "browser_open",
        {"url": "https://example.com"},
        idempotency_key="browser-open-once",
    )
    first_approval_id = first["approval"]["id"]
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute(
            "UPDATE approvals SET expires_at=? WHERE id=?",
            ("2000-01-01T00:00:00+00:00", first_approval_id),
        )
    expired = runtime.decide_approval(first_approval_id, True, "late approval")
    assert expired["approval"]["status"] == "expired"

    second = runtime.invoke_tool(
        session["id"],
        "browser_open",
        {"url": "https://example.com"},
        idempotency_key="browser-open-once",
    )
    assert second["status"] == "waiting_approval"
    assert second["approval"]["status"] == "pending"
    assert second["approval"]["id"] != first_approval_id
    assert second["tool_call"]["id"] != first["tool_call"]["id"]
    runtime.close()


def test_approved_file_tool_still_cannot_escape_workspace(tmp_path: Path):
    runtime = AgentRuntime(tmp_path / "workspace")
    session = runtime.create_session(permission_level="workspace-write")
    outside = tmp_path / "outside.txt"
    waiting = runtime.invoke_tool(session["id"], "file_write", {"path": str(outside), "content": "x"})
    assert waiting["status"] == "waiting_approval"
    completed = runtime.decide_approval(waiting["approval"]["id"], True, "approve exact call")
    assert completed["status"] == "failed"
    assert "escapes the configured workspace" in completed["result"]["error"]
    assert not outside.exists()
    runtime.close()


def test_file_glob_cannot_traverse_above_workspace(tmp_path: Path):
    runtime = AgentRuntime(tmp_path / "workspace")
    session = runtime.create_session(permission_level="observe")
    result = runtime.invoke_tool(session["id"], "file_list", {"path": ".", "glob": "../*"})
    assert result["status"] == "failed"
    assert "glob pattern escapes" in result["result"]["error"]
    runtime.close()


def test_fingerprint_is_stable_and_parameter_exact():
    first = argument_fingerprint("shell_exec", {"argv": ["python", "task.py"], "cwd": "x"})
    assert first == argument_fingerprint("shell_exec", {"cwd": "x", "argv": ["python", "task.py"]})
    assert first != argument_fingerprint("shell_exec", {"argv": ["python", "other.py"], "cwd": "x"})


def test_process_execution_requires_allowlisted_argument_arrays(tmp_path: Path):
    script = tmp_path / "task.py"
    script.write_text("print('allowlisted-array-ok')\nprint('TO' + 'KEN=' + 'sensitive-test-value')\n", encoding="utf-8")
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(permission_level="workspace-write")

    completed = runtime.invoke_tool(
        session["id"],
        "shell_exec",
        {"argv": [sys.executable, str(script)], "cwd": str(tmp_path), "timeout_seconds": 30},
    )
    assert completed["status"] == "completed"
    assert "allowlisted-array-ok" in completed["result"]["content"]["output"]
    assert "sensitive-test-value" not in completed["result"]["content"]["output"]
    assert "TOKEN=[redacted]" in completed["result"]["content"]["output"]

    inline = runtime.invoke_tool(session["id"], "shell_exec", {"argv": [sys.executable, "-c", "print(1)"]})
    assert inline["status"] == "failed"
    assert "inline Python code is not allowed" in inline["result"]["error"]

    untrusted = runtime.invoke_tool(session["id"], "shell_exec", {"argv": ["powershell.exe", "Get-Date"]})
    assert untrusted["status"] == "failed"
    assert "not allowlisted" in untrusted["result"]["error"]
    runtime.close()


def test_managed_processes_are_visible_only_to_their_own_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = tools.ProcessManager()
    monkeypatch.setattr(tools, "PROCESSES", manager)
    first_root = tmp_path / "run_first"
    second_root = tmp_path / "run_second"
    first_root.mkdir()
    second_root.mkdir()
    script = first_root / "wait.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    runtime = AgentRuntime(tmp_path / "project")
    first = runtime.create_session(workspace_root=str(first_root), permission_level="workspace-write")
    second = runtime.create_session(workspace_root=str(second_root), permission_level="workspace-write")

    started = runtime.invoke_tool(
        first["id"],
        "process_start",
        {"argv": [sys.executable, str(script)], "cwd": str(first_root)},
    )
    assert started["status"] == "completed"
    process_id = started["result"]["content"]["process_id"]
    assert [item["id"] for item in runtime.invoke_tool(first["id"], "process_list", {})["result"]["content"]["processes"]] == [process_id]
    assert runtime.invoke_tool(second["id"], "process_list", {})["result"]["content"]["processes"] == []

    denied = runtime.invoke_tool(second["id"], "process_poll", {"process_id": process_id})
    assert denied["status"] == "failed"
    assert "KeyError" in denied["result"]["error"]

    manager.cancel(manager.get(process_id, first["id"]))
    runtime.close()


def test_identical_running_process_start_is_reused_within_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = tools.ProcessManager()
    monkeypatch.setattr(tools, "PROCESSES", manager)
    script = tmp_path / "wait.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    runtime = AgentRuntime(tmp_path / "project")
    run = runtime.create_session(workspace_root=str(tmp_path), permission_level="workspace-write")
    args = {"argv": [sys.executable, str(script)], "cwd": str(tmp_path)}

    first = runtime.invoke_tool(run["id"], "process_start", args)
    second = runtime.invoke_tool(run["id"], "process_start", args)

    first_content = first["result"]["content"]
    second_content = second["result"]["content"]
    assert second_content["process_id"] == first_content["process_id"]
    assert second_content["pid"] == first_content["pid"]
    assert first_content["reused_running_equivalent"] is False
    assert second_content["reused_running_equivalent"] is True
    assert len(runtime.invoke_tool(run["id"], "process_list", {})["result"]["content"]["processes"]) == 1

    manager.cancel(manager.get(first_content["process_id"], run["id"]))
    runtime.close()


def _tool_context(tmp_path: Path) -> tools.ToolContext:
    return tools.ToolContext(
        session_id="run_fixture",
        workspace_root=tmp_path,
        project_root=tmp_path,
        runtime_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        store=None,
        metadata={},
    )


def test_python_venv_is_allowed_only_for_one_workspace_local_destination(tmp_path: Path):
    context = _tool_context(tmp_path)

    trusted = tools._trusted_argv(
        {"argv": [sys.executable, "-m", "venv", "work/mindgames-venv"]},
        context,
    )

    assert trusted[1:] == ["-m", "venv", "work/mindgames-venv"]
    with pytest.raises(ValueError, match="escapes the configured workspace"):
        tools._trusted_argv(
            {"argv": [sys.executable, "-m", "venv", str(tmp_path.parent / "outside-venv")]},
            context,
        )
    with pytest.raises(ValueError, match="no mutation flags"):
        tools._trusted_argv(
            {"argv": [sys.executable, "-m", "venv", "--clear", "work/mindgames-venv"]},
            context,
        )


def test_pip_is_allowed_only_through_verified_workspace_venv_with_bounded_requirements(tmp_path: Path):
    context = _tool_context(tmp_path)
    venv_root = tmp_path / "work" / "mindgames-venv"
    interpreter = venv_root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"fixture")
    (venv_root / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")

    trusted = tools._trusted_argv(
        {
            "argv": [
                str(interpreter), "-m", "pip", "install", "--no-input",
                "--timeout=120", "--retries=8", "--resume-retries=8", "--prefer-binary",
                "textarena>=0.7.2,<0.8", "scikit-learn", "joblib", "trueskill",
            ]
        },
        context,
    )

    assert trusted[0] == str(interpreter.resolve())
    assert trusted[3:5] == ["install", "--no-input"]
    with pytest.raises(ValueError, match="workspace-local virtual environment"):
        tools._trusted_argv(
            {"argv": [sys.executable, "-m", "pip", "install", "textarena"]},
            context,
        )
    with pytest.raises(ValueError, match="option is not allowlisted"):
        tools._trusted_argv(
            {"argv": [str(interpreter), "-m", "pip", "install", "--target", "vendor", "textarena"]},
            context,
        )
    with pytest.raises(ValueError, match="outside the allowed range"):
        tools._trusted_argv(
            {"argv": [str(interpreter), "-m", "pip", "install", "--timeout=999", "textarena"]},
            context,
        )
    with pytest.raises(ValueError, match="direct URL"):
        tools._trusted_argv(
            {"argv": [str(interpreter), "-m", "pip", "install", "https://example.invalid/package.whl"]},
            context,
        )


def test_shell_exec_preserves_nonzero_exit_code_and_file_round_trip(tmp_path: Path):
    script = tmp_path / "fails.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('agent-output.txt').write_text('research-data-ok', encoding='utf-8')\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(permission_level="workspace-write")

    executed = runtime.invoke_tool(
        session["id"],
        "shell_exec",
        {"argv": [sys.executable, str(script)], "cwd": str(tmp_path), "timeout_seconds": 30},
    )
    assert executed["status"] == "failed"
    assert executed["result"]["content"]["exit_code"] == 7
    assert executed["result"]["error"] == "nonzero_exit"

    read = runtime.invoke_tool(session["id"], "file_read", {"path": "agent-output.txt"})
    assert read["status"] == "completed"
    assert read["result"]["content"]["lines"][0]["text"] == "research-data-ok"
    runtime.close()


def test_store_persists_across_runtime_restart(tmp_path: Path):
    first = AgentRuntime(tmp_path)
    session = first.create_session(objective="durable")
    first.close()
    second = AgentRuntime(tmp_path)
    assert second.get_session(session["id"])["objective"] == "durable"
    second.close()


def test_observe_permission_rejects_mutation(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(permission_level="observe")
    result = runtime.invoke_tool(session["id"], "file_write", {"path": "x", "content": "y"})
    assert result["result"]["error"] == "permission_denied"
    runtime.close()


def test_tool_schemas_are_valid(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    assert len(runtime.tools()) >= 25
    assert all(item["input_schema"]["type"] == "object" for item in runtime.tools())
    runtime.close()


def test_evolution_never_auto_promotes():
    decision = evaluate_candidate(
        {
            "target_fixed": True,
            "regression_green": True,
            "critical_subset_no_regression": True,
            "success_rate_delta_pp": 2.1,
        }
    )
    assert decision.status == "awaiting_human_promotion"


def test_artifact_content_addressing_for_large_result(tmp_path: Path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()
    path = tmp_path / "large.txt"
    path.write_text("a" * 100_000, encoding="utf-8")
    result = runtime.invoke_tool(session["id"], "file_read", {"path": str(path), "end_line": 1})
    assert result["status"] == "completed"
    runtime.close()


def test_message_reports_served_model_and_native_tool_calls(tmp_path: Path, monkeypatch):
    from research_os.agent import messaging

    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"ready": true}\n', encoding="utf-8")

    class StubClient:
        def __init__(self):
            self.turn = 0

        def is_available(self):
            return True

        def send(self, messages, *, system, tools, max_tokens):
            assert max_tokens == 4096
            self.turn += 1
            if self.turn == 1:
                call = messaging.ToolCall(
                    "tool-read-1",
                    "file_read",
                    {"path": str(evidence), "start_line": 1, "end_line": 20},
                )
                return messaging.AssistantTurn(
                    text="",
                    tool_calls=[call],
                    stop_reason="tool_use",
                    raw_content=[
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.input,
                        }
                    ],
                    provider="openai",
                    model="gpt-5.6-sol",
                    input_tokens=11,
                    output_tokens=7,
                )
            return messaging.AssistantTurn(
                text='{"status":"done"}',
                tool_calls=[],
                stop_reason="stop",
                raw_content=[{"type": "text", "text": '{"status":"done"}'}],
                provider="openai",
                model="gpt-5.6-sol",
                input_tokens=19,
                output_tokens=5,
            )

    monkeypatch.setattr(messaging, "AgentMessageClient", StubClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-present-only-in-process")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")

    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(permission_level="observe")
    result = runtime.message(session["id"], "Read the evidence and decide.")

    assert result["status"] == "completed"
    assert result["model_execution"] == {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "native_tool_loop": True,
        "native_tool_calls": 1,
        "input_tokens": 30,
        "output_tokens": 12,
        "turns": 2,
    }
    responses = [event for event in runtime.store.list_events(session["id"]) if event["event_type"] == "model.response"]
    assert [item["payload"]["model"] for item in responses] == [
        "gpt-5.6-sol",
        "gpt-5.6-sol",
    ]
    runtime.close()


def test_message_retries_empty_terminal_turn_before_completing(tmp_path: Path, monkeypatch):
    from research_os.agent import messaging

    calls = {"count": 0}

    class StubClient:
        def is_available(self):
            return True

        def send(self, messages, *, system, tools, max_tokens):
            assert max_tokens == 4096
            del messages, system, tools
            calls["count"] += 1
            if calls["count"] <= 2:
                return messaging.AssistantTurn(
                    text="",
                    tool_calls=[],
                    stop_reason="stop",
                    raw_content=[],
                    provider="openai",
                    model="gpt-5.6-sol",
                )
            return messaging.AssistantTurn(
                text="recovered after empty terminal responses",
                tool_calls=[],
                stop_reason="stop",
                raw_content=[{"type": "text", "text": "recovered after empty terminal responses"}],
                provider="openai",
                model="gpt-5.6-sol",
            )

    monkeypatch.setattr(messaging, "AgentMessageClient", StubClient)
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()

    result = runtime.message(session["id"], "complete a durable task")

    assert result["status"] == "completed"
    assert result["text"] == "recovered after empty terminal responses"
    assert result["model_execution"]["turns"] == 3
    assert calls["count"] == 3
    assert all(
        turn["content"]
        for turn in runtime.store.list_turns(session["id"])
        if turn["role"] == "assistant"
    )
    runtime.close()


def test_message_never_completes_after_persistent_empty_terminal_turns(tmp_path: Path, monkeypatch):
    from research_os.agent import messaging

    calls = {"count": 0}

    class StubClient:
        def is_available(self):
            return True

        def send(self, messages, *, system, tools, max_tokens):
            assert max_tokens == 4096
            del messages, system, tools
            calls["count"] += 1
            return messaging.AssistantTurn(
                text="",
                tool_calls=[],
                stop_reason="stop",
                raw_content=[],
                provider="openai",
                model="gpt-5.6-sol",
            )

    monkeypatch.setattr(messaging, "AgentMessageClient", StubClient)
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()

    result = runtime.message(session["id"], "complete a durable task")

    assert result["status"] == "paused"
    assert calls["count"] == 3
    assert runtime.get_session(session["id"])["status"] == "paused"
    assert not any(turn["role"] == "assistant" and not turn["content"] for turn in runtime.store.list_turns(session["id"]))
    runtime.close()


def test_message_uses_verified_artifact_fallback_after_tool_loop_empty_terminal(
    tmp_path: Path, monkeypatch
):
    from research_os.agent import messaging

    calls = {"count": 0}

    class StubClient:
        def is_available(self):
            return True

        def send(self, messages, *, system, tools, max_tokens):
            assert max_tokens == 4096
            del messages, system, tools
            calls["count"] += 1
            if calls["count"] == 1:
                call = messaging.ToolCall("artifact-list-1", "artifact_list", {})
                return messaging.AssistantTurn(
                    text="",
                    tool_calls=[call],
                    stop_reason="tool_use",
                    raw_content=[
                        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                    ],
                    provider="openai",
                    model="gpt-5.6-sol",
                )
            return messaging.AssistantTurn(
                text="",
                tool_calls=[],
                stop_reason="stop",
                raw_content=[],
                provider="openai",
                model="gpt-5.6-sol",
            )

    monkeypatch.setattr(messaging, "AgentMessageClient", StubClient)
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="verify the published artifact",
        conversation_id="verified_artifact_empty_terminal",
        start=False,
    )
    output = Path(run["task_root"]) / "outputs" / "proof.json"
    output.write_text('{"verified":true}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], output, source_tool_call="fixture_publish")

    result = runtime.message(run["id"], "verify the published artifact")

    assert result["status"] == "completed"
    assert result["model_execution"]["native_tool_calls"] == 1
    assert result["model_execution"]["completion_mode"] == "verified_artifact_fallback"
    assert "published artifacts" in result["text"]
    assert calls["count"] == 2
    assert runtime.store.list_deliverables(run["id"])[0]["name"] == "proof.json"
    runtime.close()
