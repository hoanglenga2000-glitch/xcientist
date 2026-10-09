from __future__ import annotations

import http.client
import hashlib
import json
import time
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import evomind_runtime.assistant_runs as assistant_runs_module
from evomind_runtime.competition_goal import FIXED_RUN_ID
from evomind_runtime.http_server import (
    MAX_BODY_BYTES,
    RequestContractError,
    _invoke_direct_tool,
    ensure_token,
    make_handler,
)
from evomind_runtime.models import ApprovalRequest, ToolCall, ToolResult
from evomind_runtime.runtime import AgentRuntime


def request(port: int, method: str, path: str, token: str, *, body: bytes | None = None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    merged = {"Authorization": f"Bearer {token}", **(headers or {})}
    connection.request(method, path, body=body, headers=merged)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def request_bytes(port: int, method: str, path: str, token: str, *, body: bytes | None = None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    merged = {"Authorization": f"Bearer {token}", **(headers or {})}
    connection.request(method, path, body=body, headers=merged)
    response = connection.getresponse()
    payload = response.read()
    result = response.status, {key.lower(): value for key, value in response.getheaders()}, payload
    connection.close()
    return result


def cancelled_fixed_run(runtime: AgentRuntime, monkeypatch) -> tuple[dict, tuple[str, str, str, str]]:
    original_new_id = assistant_runs_module.new_id
    monkeypatch.setattr(
        assistant_runs_module,
        "new_id",
        lambda prefix: FIXED_RUN_ID if prefix == "run" else original_new_id(prefix),
    )
    run = runtime.assistant.create_run(
        prompt="Probe fixed G21 assets read-only",
        conversation_id="terminal_preserving_direct_probe",
        start=False,
    )
    lifecycle = (
        "cancelled",
        "preexisting_terminal_gate",
        "Preserve the fixed terminal Gate.",
        "2026-09-01T00:00:00.000+00:00",
    )
    runtime.store.update_assistant_run(
        FIXED_RUN_ID,
        status=lifecycle[0],
        error_class=lifecycle[1],
        error_message=lifecycle[2],
        completed_at=lifecycle[3],
    )
    runtime.store.update_session(FIXED_RUN_ID, status="cancelled")
    return run, lifecycle


def assistant_lifecycle(runtime: AgentRuntime) -> tuple[str, str, str, str]:
    run = runtime.store.get_assistant_run(FIXED_RUN_ID) or {}
    return (
        str(run.get("status") or ""),
        str(run.get("error_class") or ""),
        str(run.get("error_message") or ""),
        str(run.get("completed_at") or ""),
    )


def managed_hpc_identity_fixture(job_id: int, generation: int) -> dict:
    tenant_id = "tenant_aaaaaaaaaaaaaaaaaaaaaaaa"
    return {
        "tenant_id": tenant_id,
        "owner_principal_id": "admin",
        "job_id": job_id,
        "credential_profile": f"{tenant_id}_job{job_id}_g{generation}",
        "allocation_generation": generation,
        "profile_instance_id": (
            "35279c5f-6a99-4dd0-b53a-789b7f03b376"
            if generation == 21
            else "8f247e8a-1450-4f8b-8c77-535ea25e2f55"
        ),
        "allocation_binding_id": f"aimslab-job{job_id}-fixture-binding",
    }


def test_assistant_action_route_propagates_rebind_idempotency_key(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    original_new_id = assistant_runs_module.new_id
    monkeypatch.setattr(
        assistant_runs_module,
        "new_id",
        lambda prefix: FIXED_RUN_ID if prefix == "run" else original_new_id(prefix),
    )
    old_identity = managed_hpc_identity_fixture(92257, 21)
    next_identity = managed_hpc_identity_fixture(93015, 24)
    run = runtime.assistant.create_run(
        prompt="Fixed terminal Run identity rebind fixture",
        conversation_id="http-managed-hpc-rebind",
        managed_hpc_identity=old_identity,
        start=False,
    )
    metadata = dict(runtime.get_session(run["id"])["metadata"])
    metadata["run_secret_refs"] = ["rs_consumed_fixture"]
    runtime.store.update_session(run["id"], status="cancelled", metadata_json=metadata)
    runtime.store.update_assistant_run(run["id"], status="cancelled")
    before_seq = runtime.store.latest_event_seq(run["id"])

    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    body = json.dumps(
        {
            "action": "rebind_managed_hpc_identity",
            "managed_hpc_identity": next_identity,
            "idempotency_key": "http-rebind-job93015-g24-once",
        }
    ).encode("utf-8")
    try:
        status, first = request(
            port,
            "POST",
            f"/v1/runs/{run['id']}/actions",
            token,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert first["status"] == "cancelled"
        assert first["managed_hpc_identity_rebind"]["updated"] is True
        assert first["managed_hpc_identity_rebind"]["run_resumed"] is False
        assert first["managed_hpc_identity_rebind"]["hpc_accessed"] is False
        assert first["last_event_seq"] == before_seq + 1

        status, replay = request(
            port,
            "POST",
            f"/v1/runs/{run['id']}/actions",
            token,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert replay["managed_hpc_identity_rebind"]["updated"] is False
        assert replay["last_event_seq"] == before_seq + 1
        session = runtime.get_session(run["id"])
        assert session["status"] == "cancelled"
        assert session["metadata"]["managed_hpc_identity"] == next_identity
        assert session["metadata"]["run_secret_refs"] == []
        assert sum(
            event["event_type"] == "managed_hpc_identity.rebound"
            for event in runtime.store.list_events(run["id"], after_seq=0, limit=2000)
        ) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_runtime_rotates_a_malformed_token_atomically(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "runtime.token").write_text("", encoding="ascii")
    token = ensure_token(root)
    assert 32 <= len(token) <= 128
    assert token == (root / "runtime.token").read_text(encoding="ascii").strip()
    assert ensure_token(root) == token


def test_runtime_http_contract_is_authenticated_and_body_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("EVOMIND_BUILD_COMMIT_HASH", "b" * 40)
    monkeypatch.setenv("EVOMIND_SOURCE_TREE_SHA256", "a" * 64)
    runtime = AgentRuntime(tmp_path)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        status, payload = request(port, "GET", "/v1/health", "wrong-token")
        assert status == 401 and payload["error"] == "unauthorized"
        status, payload = request(port, "GET", "/v1/health", token)
        assert status == 200 and payload["status"] == "ready"
        assert payload["backend_version"] == "0.3.0"
        assert payload["commit_hash"] == "b" * 40
        assert payload["source_tree_sha256"] == "a" * 64

        status, payload = request(port, "GET", "/v1/super-agent/status", token)
        assert status == 200
        assert payload["schema"] == "evomind.super_agent_runtime.v1"
        assert payload["mode"] == "shadow"
        assert payload["migration_applied"] is True

        status, payload = request(
            port,
            "GET",
            "/v1/super-agent/capabilities?operation=write&objective=authorized%20directory",
            token,
        )
        assert status == 200
        assert "workspace" in {
            item["descriptor"]["capability_id"] for item in payload["matches"]
        }

        status, payload = request(
            port,
            "POST",
            "/v1/sessions",
            token,
            body=b"{}",
            headers={"Content-Type": "text/plain"},
        )
        assert status == 415 and payload["error"] == "unsupported_content_type"

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.putrequest("POST", "/v1/sessions")
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        assert response.status == 413 and payload["error"] == "body_too_large"

        status, payload = request(
            port,
            "POST",
            "/v1/sessions",
            token,
            body=b"[]",
            headers={"Content-Type": "application/json"},
        )
        assert status == 400 and payload["error"] == "json_object_required"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_assistant_http_upload_run_event_and_artifact_round_trip(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(tmp_path)

    def completed_message(session_id: str, _content: str, *, max_steps: int = 24):
        del max_steps
        task_root = Path(runtime.get_session(session_id)["workspace_root"])
        output = task_root / "outputs" / "结果.md"
        output.write_text("# HTTP fixture\n\nverified\n", encoding="utf-8")
        runtime.store.add_turn(session_id, "assistant", "HTTP fixture completed.")
        return {
            "status": "completed",
            "text": "HTTP fixture completed.",
            "model_execution": {"provider": "fixture-provider", "model": "served-http-model", "native_tool_calls": 1},
        }

    monkeypatch.setattr(runtime, "message", completed_message)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    data = b"column,value\nfixture,1\n"
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        status, upload = request(
            port,
            "POST",
            "/v1/uploads",
            token,
            body=json.dumps({"name": "输入.csv", "total_bytes": len(data), "media_type": "text/csv", "sha256": sha256}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 201
        status, chunk = request(
            port,
            "PUT",
            f"/v1/uploads/{upload['id']}/chunks/0",
            token,
            body=data,
            headers={"Content-Type": "application/octet-stream", "X-Chunk-SHA256": sha256},
        )
        assert status == 200 and chunk["sha256"] == sha256
        status, completed_upload = request(
            port,
            "POST",
            f"/v1/uploads/{upload['id']}/complete",
            token,
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert status == 200 and completed_upload["attachment"]["sha256"] == sha256

        status, run = request(
            port,
            "POST",
            "/v1/runs",
            token,
            body=json.dumps({
                "prompt": "分析附件并生成结果",
                "conversation_id": "http_fixture",
                "attachment_ids": [completed_upload["attachment"]["id"]],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 201
        deadline = time.monotonic() + 5
        snapshot = run
        while snapshot["status"] != "completed" and time.monotonic() < deadline:
            time.sleep(0.02)
            status, snapshot = request(port, "GET", f"/v1/runs/{run['id']}", token)
            assert status == 200
        assert snapshot["status"] == "completed"
        assert snapshot["model"] == "served-http-model"
        assert len(snapshot["artifacts"]) == 1

        status, events = request(port, "GET", f"/v1/runs/{run['id']}/events?after=0", token)
        assert status == 200
        event_types = [item["event_type"] for item in events["events"]]
        assert "model_observed" in event_types
        assert "artifact_published" in event_types
        assert event_types[-1] == "run_completed"

        artifact = snapshot["artifacts"][0]
        status, headers, body = request_bytes(port, "GET", f"/v1/artifacts/{artifact['id']}?download=1", token)
        assert status == 200
        assert body == Path(artifact["path"]).read_bytes()
        assert headers["x-artifact-bytes"] == str(len(body))
        assert headers["x-artifact-sha256"] == hashlib.sha256(body).hexdigest()
        assert "filename*=UTF-8''" in headers["content-disposition"]
        assert "%E7%BB%93%E6%9E%9C.md" in headers["content-disposition"]

        status, preview = request(port, "GET", f"/v1/artifacts/{artifact['id']}/preview", token)
        assert status == 200
        assert "HTTP fixture" in preview["preview"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_approval_decision_requires_a_json_boolean(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session()
    target = tmp_path / "delete-only-after-boolean-approval.txt"
    target.write_text("preserve", encoding="utf-8")
    waiting = runtime.invoke_tool(session["id"], "file_delete", {"path": str(target)})
    approval_id = waiting["approval"]["id"]
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        for payload in ({"approved": "false"}, {"approved": 1}, {"approved": None}, {}):
            status, error = request(
                port,
                "POST",
                f"/v1/approvals/{approval_id}/decision",
                token,
                body=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            assert status == 400
            assert error["error"] == "approved_boolean_required"
            assert runtime.store.get_approval(approval_id)["status"] == "pending"
            assert target.exists()

        assistant_run = runtime.assistant.create_run(
            prompt="Verify approval boolean rejection",
            conversation_id="http_boolean_rejection",
            start=False,
        )
        rejected_target = Path(assistant_run["task_root"]) / "outputs" / "reject.txt"
        rejected_target.write_text("preserve", encoding="utf-8")
        rejected_waiting = runtime.invoke_tool(
            assistant_run["id"], "file_delete", {"path": str(rejected_target)},
        )
        status, rejected = request(
            port,
            "POST",
            f"/v1/approvals/{rejected_waiting['approval']['id']}/decision",
            token,
            body=json.dumps({"approved": False, "note": "reject exact deletion"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert rejected["approval"]["status"] == "rejected"
        assert rejected_target.exists()

        approved_run = runtime.assistant.create_run(
            prompt="Verify approval boolean acceptance",
            conversation_id="http_boolean_approval",
            start=False,
        )
        approved_target = Path(approved_run["task_root"]) / "outputs" / "approve.txt"
        approved_target.write_text("delete", encoding="utf-8")
        approved_waiting = runtime.invoke_tool(
            approved_run["id"], "file_delete", {"path": str(approved_target)},
        )
        status, approved = request(
            port,
            "POST",
            f"/v1/approvals/{approved_waiting['approval']['id']}/decision",
            token,
            body=json.dumps({"approved": True, "note": "approve exact deletion"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert approved["approval"]["status"] == "approved"
        # Approval dispatch is asynchronous; wait for this exact approved call,
        # not for an unrelated task or an unbounded sleep.
        approved_call_id = approved_waiting["tool_call"]["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            call = runtime.store.get_tool_call(approved_call_id)
            if call["status"] not in {"waiting_approval", "running", "queued"}:
                break
            time.sleep(0.01)
        assert runtime.store.get_tool_call(approved_call_id)["status"] == "completed"
        assert not approved_target.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_direct_managed_tool_projects_running_and_recovering_into_assistant_run(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="训练并交付模型",
        conversation_id="direct_tool_status_projection",
        start=False,
    )
    runtime.store.update_assistant_run(run["id"], status="blocked", error_class="llm_error", error_message="fixture")
    runtime.store.update_session(run["id"], status="blocked")
    observed: dict[str, str] = {}

    def completed_tool(session_id: str, tool_name: str, arguments: dict, **_kwargs):
        del tool_name, arguments
        observed["assistant_status"] = runtime.store.get_assistant_run(session_id)["status"]
        observed["session_status"] = runtime.store.get_session(session_id)["status"]
        return {
            "status": "completed",
            "result": {"ok": True, "summary": "managed tool completed", "error": "", "content": {}},
        }

    monkeypatch.setattr(runtime, "invoke_tool", completed_tool)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        status, outcome = request(
            port,
            "POST",
            f"/v1/sessions/{run['id']}/tools",
            token,
            body=json.dumps({"tool_name": "runtime_health", "arguments": {}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200 and outcome["status"] == "completed"
        assert observed == {"assistant_status": "running", "session_status": "running"}
        projected = runtime.store.get_assistant_run(run["id"])
        assert projected["status"] == "recovering"
        assert projected["error_class"] == ""
        assert projected["error_message"] == ""
        assert projected["completed_at"] == ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_cancelled_fixed_hpc_asset_probe_http_restores_terminal_lifecycle(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    _run, original_lifecycle = cancelled_fixed_run(runtime, monkeypatch)
    observed: dict[str, str] = {}

    def harmless_probe(arguments: dict, context) -> ToolResult:
        observed["assistant_status"] = runtime.store.get_assistant_run(context.session_id)["status"]
        observed["session_status"] = runtime.store.get_session(context.session_id)["status"]
        return ToolResult("", True, {"scope": arguments.get("scope")}, "probe complete")

    monkeypatch.setitem(runtime.registry._handlers, "hpc_asset_probe", harmless_probe)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    body = json.dumps(
        {
            "tool_name": "hpc_asset_probe",
            "arguments": {"scope": "cure_mindgames"},
            "idempotency_key": "r120-terminal-probe-success-once",
        }
    ).encode()
    try:
        status, outcome = request(
            port,
            "POST",
            f"/v1/sessions/{FIXED_RUN_ID}/tools",
            token,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert status == 200 and outcome["status"] == "completed"
        assert observed == {"assistant_status": "cancelled", "session_status": "running"}
        assert assistant_lifecycle(runtime) == original_lifecycle
        assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"
        snapshot = runtime.assistant.snapshot(FIXED_RUN_ID)
        assert snapshot["status"] == "cancelled" and snapshot["terminal"] is True
        assert snapshot["active_tool_calls"] == []
        assert [item for item in snapshot["approvals"] if item["status"] == "pending"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_cancelled_fixed_hpc_asset_probe_restores_after_failure_and_replay(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    _run, original_lifecycle = cancelled_fixed_run(runtime, monkeypatch)
    calls = 0

    def failed_probe(_arguments: dict, _context) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult("", False, {"read_only": True}, "probe failed closed", error="fixture_gate")

    monkeypatch.setitem(runtime.registry._handlers, "hpc_asset_probe", failed_probe)
    arguments = {"scope": "cure_mindgames"}
    first = _invoke_direct_tool(
        runtime,
        FIXED_RUN_ID,
        "hpc_asset_probe",
        arguments,
        "r120-terminal-probe-failure-once",
    )
    replay = _invoke_direct_tool(
        runtime,
        FIXED_RUN_ID,
        "hpc_asset_probe",
        arguments,
        "r120-terminal-probe-failure-once",
    )

    assert first["status"] == "failed" and first["result"]["error"] == "fixture_gate"
    assert replay["status"] == "failed" and replay["replayed"] is True
    assert calls == 1
    assert assistant_lifecycle(runtime) == original_lifecycle
    assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"
    runtime.close()


def test_cancelled_fixed_hpc_asset_probe_restores_session_after_exception(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    _run, original_lifecycle = cancelled_fixed_run(runtime, monkeypatch)

    def explode(session_id: str, *_args, **_kwargs):
        runtime.store.update_session(session_id, status="running")
        assert assistant_lifecycle(runtime) == original_lifecycle
        raise RuntimeError("fixture transport exception")

    monkeypatch.setattr(runtime, "invoke_tool", explode)
    try:
        _invoke_direct_tool(
            runtime,
            FIXED_RUN_ID,
            "hpc_asset_probe",
            {"scope": "cure_mindgames"},
            "r120-terminal-probe-exception-once",
        )
    except RuntimeError as exc:
        assert str(exc) == "fixture transport exception"
    else:
        raise AssertionError("fixture exception was not propagated")
    assert assistant_lifecycle(runtime) == original_lifecycle
    assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"
    runtime.close()


def test_cancelled_fixed_hpc_asset_probe_concurrent_replay_executes_once(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    _run, original_lifecycle = cancelled_fixed_run(runtime, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_probe(_arguments: dict, _context) -> ToolResult:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return ToolResult("", True, {"read_only": True}, "probe complete")

    monkeypatch.setitem(runtime.registry._handlers, "hpc_asset_probe", blocked_probe)
    outcomes: list[dict] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            outcomes.append(
                _invoke_direct_tool(
                    runtime,
                    FIXED_RUN_ID,
                    "hpc_asset_probe",
                    {"scope": "cure_mindgames"},
                    "r120-terminal-probe-concurrent-once",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports the captured failure
            failures.append(exc)

    first = threading.Thread(target=invoke, daemon=True)
    second = threading.Thread(target=invoke, daemon=True)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert calls == 1 and second.is_alive()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert failures == [] and len(outcomes) == 2
    assert calls == 1
    assert sorted(bool(item.get("replayed")) for item in outcomes) == [False, True]
    assert assistant_lifecycle(runtime) == original_lifecycle
    assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"
    runtime.close()


def test_cancelled_fixed_hpc_asset_probe_fails_closed_on_approval_or_assistant_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    _run, original_lifecycle = cancelled_fixed_run(runtime, monkeypatch)

    def waiting_approval(session_id: str, *_args, **_kwargs) -> dict:
        runtime.store.update_session(session_id, status="waiting_approval")
        return {"status": "waiting_approval", "approval": {"status": "pending"}}

    monkeypatch.setattr(runtime, "invoke_tool", waiting_approval)
    try:
        _invoke_direct_tool(
            runtime,
            FIXED_RUN_ID,
            "hpc_asset_probe",
            {"scope": "cure_mindgames"},
            "r120-terminal-probe-approval-drift-once",
        )
    except RequestContractError as exc:
        assert exc.status == 409 and exc.code == "terminal_read_only_approval_drift"
    else:
        raise AssertionError("approval drift was accepted")
    assert assistant_lifecycle(runtime) == original_lifecycle
    assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"

    def assistant_drift(session_id: str, *_args, **_kwargs) -> dict:
        runtime.store.update_session(session_id, status="running")
        runtime.store.update_assistant_run(session_id, status="recovering")
        return {"status": "completed", "result": {"ok": True}}

    monkeypatch.setattr(runtime, "invoke_tool", assistant_drift)
    try:
        _invoke_direct_tool(
            runtime,
            FIXED_RUN_ID,
            "hpc_asset_probe",
            {"scope": "cure_mindgames"},
            "r120-terminal-probe-assistant-drift-once",
        )
    except RequestContractError as exc:
        assert exc.status == 409 and exc.code == "terminal_read_only_assistant_drift"
    else:
        raise AssertionError("assistant lifecycle drift was accepted")
    assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"
    runtime.close()


def test_cancelled_fixed_hpc_asset_probe_rejects_empty_key_or_existing_activity(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    cancelled_fixed_run(runtime, monkeypatch)
    for key, code in (("", "terminal_read_only_idempotency_required"), ("   ", "terminal_read_only_idempotency_required")):
        try:
            _invoke_direct_tool(runtime, FIXED_RUN_ID, "hpc_asset_probe", {}, key)
        except RequestContractError as exc:
            assert exc.status == 409 and exc.code == code
        else:
            raise AssertionError("empty idempotency key was accepted")

    runtime.store.put_tool_call(
        ToolCall(
            id="call_existing_terminal_probe",
            session_id=FIXED_RUN_ID,
            tool_name="hpc_asset_probe",
            arguments={"scope": "cure_mindgames"},
            status="running",
        )
    )
    try:
        _invoke_direct_tool(
            runtime,
            FIXED_RUN_ID,
            "hpc_asset_probe",
            {},
            "r120-terminal-probe-active-rejected",
        )
    except RequestContractError as exc:
        assert exc.status == 409 and exc.code == "terminal_read_only_active_tool_call"
    else:
        raise AssertionError("active probe precondition was accepted")
    runtime.store.put_tool_call(
        ToolCall(
            id="call_existing_terminal_probe",
            session_id=FIXED_RUN_ID,
            tool_name="hpc_asset_probe",
            arguments={"scope": "cure_mindgames"},
            status="completed",
            completed_at="2026-09-01T00:00:01.000+00:00",
        ),
        ToolResult("call_existing_terminal_probe", True, {}, "settled fixture"),
    )
    runtime.store.put_approval(
        ApprovalRequest(
            id="approval_existing_terminal_probe",
            session_id=FIXED_RUN_ID,
            tool_call_id="call_settled_other_tool",
            tool_name="file_delete",
            argument_fingerprint="a" * 64,
            normalized_arguments={},
            impact_scope={},
            risk_level="high",
            reversible=True,
            expires_at="2099-01-01T00:00:00+00:00",
        )
    )
    try:
        _invoke_direct_tool(
            runtime,
            FIXED_RUN_ID,
            "hpc_asset_probe",
            {},
            "r120-terminal-probe-pending-rejected",
        )
    except RequestContractError as exc:
        assert exc.status == 409 and exc.code == "terminal_read_only_pending_approval"
    else:
        raise AssertionError("pending approval precondition was accepted")
    assert runtime.store.get_assistant_run(FIXED_RUN_ID)["status"] == "cancelled"
    assert runtime.store.get_session(FIXED_RUN_ID)["status"] == "cancelled"
    runtime.close()
