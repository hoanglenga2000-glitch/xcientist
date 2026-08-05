from __future__ import annotations

from pathlib import Path
import sys

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

        def send(self, messages, *, system, tools):
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
