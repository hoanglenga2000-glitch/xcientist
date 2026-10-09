"""GPT-5.5 fail-closed acceptance: synthetic fixtures and loopback HTTP only.

Hypothesis: an injected request failure never dispatches an incomplete tool turn;
an admitted external action is dispatched at most once across lost responses and
store reopen; new submission/approval acknowledgements have local P95 <= 1 s.
Blast radius: fresh pytest directories and ephemeral 127.0.0.1 ports, zero users,
zero model-provider/HPC/GPU/production traffic.  Every held action has a bounded
release and every server is shut down in finally.  These are NOT Chrome, actual
model stability, process-kill, GPU-worker identity or deployment acceptance.
"""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import multiprocessing
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import urllib.request

import pytest

from evomind_runtime import model_transport, responses_transport
from evomind_runtime.http_server import make_handler
from evomind_runtime.message_journal import MessageJournal
from evomind_runtime.models import ToolCall, ToolResult, utc_now
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.store import RuntimeStore
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig

# Reuse the existing boundary/approval/ownership fixtures, not production data.
from invitation_offline_plugin import invitation_offline_boundary  # noqa: F401
from test_approval_async_http import approval_fixture
from test_invitation_tenant_access import ALICE, TOKEN


MODEL = "gpt-5.5"
SENTINEL = "private-fixture-error-must-not-enter-observation"
WRITE = messaging.ToolSpec("file_write", "Write one synthetic fixture", {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"], "additionalProperties": False,
})


def _reply(protocol, fault="ok"):
    name = "not_offered" if fault == "unoffered" else WRITE.name
    arguments = "[]" if fault == "invalid_arguments" else json.dumps({"path": "proof.txt", "content": "once"})
    if fault == "invalid_schema":
        arguments = json.dumps({"path": "proof.txt", "content": {"not": "text"}})
    model = "unrequested-model" if fault == "wrong_model" else MODEL
    if fault == "error_envelope":
        return {"error": {"type": "upstream_error", "message": SENTINEL}}
    if protocol == "responses":
        return {
            "id": "response_fixture", "model": model,
            "status": "incomplete" if fault == "truncated" else "completed", "error": None,
            "output": [] if fault == "empty" else [{"type": "function_call", "id": "function_fixture",
                "call_id": "call_fixture_once", "name": name, "arguments": arguments, "status": "completed"}],
            "usage": {"input_tokens": 7, "output_tokens": 9},
        }
    return {"model": model, "choices": [{"finish_reason": "length" if fault == "truncated" else "tool_calls",
        "message": {"content": "", "tool_calls": [] if fault == "empty" else [{"id": "call_fixture_once",
            "type": "function", "function": {"name": name, "arguments": arguments}}]}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 9}}


@contextlib.contextmanager
def _scripted_model_http(protocol, faults):
    """A real HTTP/SSE reader against an in-memory, non-provider endpoint."""
    seen, lock = [], threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                index = len(seen)
                seen.append({"model": payload.get("model"), "path": self.path, "stream": payload.get("stream"),
                             "tool_count": len(payload.get("tools", []))})
            fault = faults[index] if index < len(faults) else "unexpected_extra_attempt"
            if fault == "timeout":
                # Real socket timeout; not a monkeypatched exception or clock.
                threading.Event().wait(0.65)
            status = int(fault[5:]) if fault.startswith("http_") else 200
            body = _reply(protocol, fault)
            if status != 200:
                body = {"error": {"message": SENTINEL}}
            if protocol == "responses" and status == 200:
                if fault == "missing_terminal":
                    body_bytes = b'data: {"type":"response.output_item.done"}\n\ndata: [DONE]\n\n'
                elif "error" in body and body["error"]:
                    body_bytes = ("data: " + json.dumps(body) + "\n\n").encode()
                else:
                    body_bytes = ("data: " + json.dumps({"type": "response." + body["status"], "response": body}) + "\n\n").encode()
                media = "text/event-stream"
            else:
                body_bytes, media = json.dumps(body).encode(), "application/json"
            try:
                self.send_response(status)
                self.send_header("Content-Type", media)
                self.send_header("Content-Length", str(len(body_bytes)))
                if status == 429:
                    self.send_header("Retry-After", "3")
                self.end_headers()
                self.wfile.write(body_bytes)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # A bounded timeout/closed client intentionally abandons bytes.
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def _client(monkeypatch, endpoint, protocol, events, delays, retries=1):
    monkeypatch.delenv("EVOMIND_MODEL_ACCEPTANCE_EXTRAS", raising=False)
    monkeypatch.delenv("EVOMIND_MODEL_ROUTE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("EVOMIND_MODEL_WIRE_PROTOCOL", protocol)
    # Capture only the retry wait; preserve real HTTP timeouts and unrelated time.
    monkeypatch.setattr(model_transport, "time", SimpleNamespace(monotonic=time.monotonic, sleep=delays.append))
    monkeypatch.setattr(responses_transport, "_responses_opener", urllib.request.build_opener(
        urllib.request.ProxyHandler({}), responses_transport._NoRedirectHandler()))
    base = messaging.AgentMessageClient(timeout=0.2, max_retries=retries, transports=[
        messaging.OpenAITransport(ProviderConfig("openai", endpoint, MODEL, "synthetic-unused-credential"))])
    guarded = model_transport.governed_client(base, events.append)
    assert guarded is not base, "gpt-5.5 must be governed without an acceptance-only environment switch"
    assert guarded.contract["model"] == MODEL
    return guarded


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("fault,code", [
    ("http_429", "http_429"), ("http_502", "http_502"), ("http_503", "http_503"),
    ("timeout", "model_network_error"), ("empty", "empty_model_response"),
    ("invalid_arguments", "tool_arguments_invalid"), ("error_envelope", "upstream_error"),
])
def test_gpt55_loopback_fault_retry_dispatches_only_valid_terminal_once(tmp_path, monkeypatch, protocol, fault, code):
    events, delays = [], []
    with _scripted_model_http(protocol, [fault, "ok"]) as (endpoint, seen):
        client = _client(monkeypatch, endpoint, protocol, events, delays)
        turn = client.send([{"role": "user", "content": "Write proof once."}], system="Synthetic fixture only", tools=[WRITE])
    assert [item["status"] for item in events] == ["failed", "completed"]
    assert events[0]["error_code"] == code
    # HTTP-200 faults still carry provider usage (fixture _reply); failed attempts report it
    # as known, consistent with test_model_transport_reliability.py truncated-response case.
    usage_reported = fault in {"empty", "invalid_arguments"}
    assert events[0]["token_usage_known"] is usage_reported
    if usage_reported:
        assert events[0]["input_tokens"] == 7 and events[0]["output_tokens"] == 9
    else:
        assert events[0]["input_tokens"] is None and events[0]["output_tokens"] is None
    assert events[1]["input_tokens"] == 7 and events[1]["output_tokens"] == 9
    assert client.last_attempt_count == len(seen) == 2
    assert {item["model"] for item in seen} == {MODEL}
    assert all(item["tool_count"] == 1 for item in seen)
    assert delays == [3 if fault == "http_429" else 2]
    assert SENTINEL not in json.dumps(events)
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Synthetic fault matrix output", start=False)
        journal = MessageJournal(runtime, run["id"])
        journal.begin([{"role": "assistant", "content": turn.raw_content}], turn.tool_calls, {WRITE.name})
        assert journal.settle()["status"] == "settled"
        assert journal.settle() is None
        calls = runtime.store.list_tool_calls(run["id"])
        assert len(calls) == 1 and calls[0]["status"] == "completed" and calls[0]["result"]["ok"] is True
        assert (Path(runtime.get_session(run["id"])["workspace_root"]) / "proof.txt").read_text() == "once"
    finally:
        assert runtime.close()


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("fault,code", [
    ("http_401", "http_401"), ("http_403", "http_403"),
    ("unoffered", "unoffered_tool_call"), ("wrong_model", "model_identity_unconfirmed"),
    ("truncated", "model_output_truncated"),
])
def test_gpt55_loopback_hard_failure_has_no_retry_or_dispatch(tmp_path, monkeypatch, protocol, fault, code):
    events, delays = [], []
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Synthetic rejection evidence", start=False)
        with _scripted_model_http(protocol, [fault]) as (endpoint, seen):
            client = _client(monkeypatch, endpoint, protocol, events, delays, retries=2)
            with pytest.raises(model_transport.ModelTransportError) as caught:
                client.send([], system="Synthetic fixture only", tools=[WRITE])
        assert caught.value.code == code and caught.value.retryable is False
        assert len(seen) == len(events) == 1 and delays == []
        assert runtime.store.list_tool_calls(run["id"]) == []
        assert runtime.store.get_assistant_run(run["id"])["status"] != "completed"
        assert SENTINEL not in str(caught.value) + json.dumps(events)
    finally:
        assert runtime.close()


def test_gpt55_sse_missing_terminal_exhausts_only_model_attempts(tmp_path, monkeypatch):
    events, delays = [], []
    with _scripted_model_http("responses", ["missing_terminal"] * 3) as (endpoint, seen):
        client = _client(monkeypatch, endpoint, "responses", events, delays, retries=20)
        with pytest.raises(model_transport.ModelTransportError, match="model_response_protocol_error"):
            client.send([], system="Synthetic fixture only", tools=[WRITE])
    assert len(seen) == client.last_attempt_count == 3
    assert delays == [2, 4]
    assert [item["status"] for item in events] == ["failed"] * 3
    assert not list(tmp_path.rglob("proof.txt"))


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("fault,code", [("http_503", "http_503"), ("empty", "empty_model_response"),
                                      ("invalid_arguments", "tool_arguments_invalid")])
def test_gpt55_repeated_failure_cannot_escape_request_retry_limit(monkeypatch, protocol, fault, code):
    events, delays = [], []
    with _scripted_model_http(protocol, [fault] * 3) as (endpoint, seen):
        client = _client(monkeypatch, endpoint, protocol, events, delays, retries=2)
        with pytest.raises(model_transport.ModelTransportError) as caught:
            client.send([], system="Synthetic fixture only", tools=[WRITE])
    assert caught.value.code == code
    assert len(seen) == len(events) == client.last_attempt_count == 3
    assert all(item["status"] == "failed" for item in events)
    assert delays == [2, 4]


def test_gpt55_semantic_argument_failure_is_a_failed_receipt_not_execution(tmp_path, monkeypatch):
    events, delays = [], []
    with _scripted_model_http("responses", ["invalid_schema"]) as (endpoint, seen):
        client = _client(monkeypatch, endpoint, "responses", events, delays)
        turn = client.send([], system="Synthetic fixture only", tools=[WRITE])
    assert len(seen) == 1 and delays == []
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Semantic argument rejection fixture", start=False)
        monkeypatch.setitem(runtime.registry._handlers, "file_write", lambda *_: pytest.fail("Invalid schema must never reach a handler"))
        journal = MessageJournal(runtime, run["id"])
        journal.begin([{"role": "assistant", "content": turn.raw_content}], turn.tool_calls, {"file_write"})
        settled = journal.settle()
        assert settled["status"] == "settled"
        result = settled["messages"][-1]["content"][0]
        assert result["is_error"] is True
        receipts = runtime.store.list_tool_calls(run["id"])
        assert len(receipts) == 1 and receipts[0]["status"] == "failed" and receipts[0]["result"]["ok"] is False
        assert runtime.assistant.snapshot(run["id"])["status"] != "completed"
        assert not (Path(runtime.get_session(run["id"])["workspace_root"]) / "proof.txt").exists()
    finally:
        assert runtime.close()


def test_gpt55_aibuild_budget_counts_actual_http_attempts_without_outer_retries(tmp_path, monkeypatch):
    from evomind_runtime.aibuild_engine import AIBuildEngine

    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Synthetic AIBuild request accounting", start=False)
        events, delays = [], []
        with _scripted_model_http("responses", ["http_503", "http_502", "ok"]) as (endpoint, seen):
            client = _client(monkeypatch, endpoint, "responses", events, delays, retries=2)
            engine = AIBuildEngine(runtime, runtime.get_session(run["id"]), client_factory=lambda: client)
            assert engine._send(engine._client(), [], "Synthetic fixture only", [WRITE]).model == MODEL
        assert engine.calls == len(seen) == 3 and delays == [2, 4]
        assert runtime.store.list_tool_calls(run["id"]) == []
    finally:
        assert runtime.close()


def _p95(samples):
    return sorted(samples)[math.ceil(0.95 * len(samples)) - 1]


def test_local_http_new_approval_ack_p95_does_not_wait_for_actions(tmp_path, monkeypatch, record_property):
    with approval_fixture(tmp_path, monkeypatch) as (runtime, run, first, started, release, resumed, calls, resumes, request):
        gates = [first]
        for index in range(1, 20):
            gates.append(runtime.invoke_tool(run["id"], "file_delete", {"path": f"fixture-{index}.txt"}, idempotency_key=f"action-{index}"))
        samples = []
        for gate in gates:
            before = time.monotonic()
            status, reply = request(f"/v1/approvals/{gate['approval']['id']}/decision", {"approved": True})
            samples.append(time.monotonic() - before)
            assert status == 200 and reply["approval"]["status"] == "approved"
            assert reply["run"]["status"] != "completed"
            assert not release.is_set() and not resumed.is_set()
        assert started.wait(1)
        record_property("evidence_scope", "local_loopback_http_synthetic_actions_not_chrome")
        record_property("approval_ack_samples", len(samples))
        record_property("approval_ack_p95_seconds", _p95(samples))
        assert len(samples) == 20 and _p95(samples) <= 1.0
        deadline = time.monotonic() + 2
        while len(calls) < 20 and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert calls == [True] * 20 and resumes == []


@contextlib.contextmanager
def _runtime_http(runtime, principal=ALICE):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, TOKEN))
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()

    def request(path, body=None, *, abandon_response=False):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        headers = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}
        if principal is not None:
            headers.update({"X-EvoMind-Access-Scope": "user.v1", "X-EvoMind-Tenant-Id": principal.tenant_id,
                            "X-EvoMind-Principal-Id": principal.owner_id})
        try:
            connection.request("GET" if body is None else "POST", path,
                               body=None if body is None else json.dumps(body).encode(), headers=headers)
            if abandon_response:
                # Complete request sent, zero response bytes read: browser-like
                # connection loss, not a monkeypatch of RunRequests or dispatch.
                connection.sock.shutdown(socket.SHUT_RDWR)
                return None
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        yield request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@contextlib.contextmanager
def _held_run_executor(runtime, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    calls, failures = [], []

    def execute(run_id, *, resume):
        calls.append(run_id)
        runtime.store.update_assistant_run(run_id, status="running")
        entered.set()
        if not release.wait(15):
            failures.append("synthetic_action_release_timeout")

    monkeypatch.setattr(runtime.assistant, "_execute", execute)
    try:
        yield entered, release, calls
    finally:
        release.set()
        assert runtime.assistant.shutdown(timeout=3)
        assert failures == []


def test_local_http_new_submission_ack_p95_does_not_wait_for_run(tmp_path, monkeypatch, record_property):
    runtime = AgentRuntime(tmp_path)
    try:
        with _held_run_executor(runtime, monkeypatch) as (entered, release, calls), _runtime_http(runtime) as request:
            samples, run_ids = [], []
            for index in range(20):
                before = time.monotonic()
                status, reply = request("/v1/runs", {"prompt": "Held synthetic task", "idempotency_key": f"http_latency_{index:02d}"})
                samples.append(time.monotonic() - before)
                assert status == 201 and reply["status"] != "completed"
                assert reply["creation_request"]["replayed"] is False
                run_ids.append(reply["id"])
                assert not release.is_set()
            assert entered.wait(1)
            record_property("evidence_scope", "local_loopback_http_synthetic_executor_not_chrome")
            record_property("submission_ack_samples", len(samples))
            record_property("submission_ack_p95_seconds", _p95(samples))
            assert len(samples) == 20 and _p95(samples) <= 1.0
            assert sorted(calls) == sorted(run_ids) and len(set(run_ids)) == 20
            assert all(runtime.assistant._threads[run_id].is_alive() for run_id in run_ids)
    finally:
        assert runtime.close(timeout=3)


def test_http_disconnect_replay_and_service_reopen_do_not_dispatch_twice(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    body = {"prompt": "Dropped HTTP response fixture", "idempotency_key": "browser_disconnect_fixture"}
    try:
        with _held_run_executor(runtime, monkeypatch) as (entered, release, calls), _runtime_http(runtime) as request:
            request("/v1/runs", body, abandon_response=True)
            assert entered.wait(2) and len(calls) == 1
            run_id = calls[0]
            with ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(lambda _: request("/v1/runs", body), range(8)))
            assert all(status == 201 and reply["id"] == run_id and reply["creation_request"]["replayed"] for status, reply in replies)
            assert calls == [run_id] and not release.is_set()
    finally:
        assert runtime.close(timeout=3)
    restored = AgentRuntime(tmp_path)
    try:
        monkeypatch.setattr(restored.assistant, "start", lambda *_a, **_k: pytest.fail("Lost ACK replay must not start a Run"))
        with _runtime_http(restored) as request:
            status, reply = request("/v1/runs", body)
            assert status == 201 and reply["id"] == run_id and reply["creation_request"]["replayed"]
            assert len(restored.store.list_assistant_runs()) == 1
    finally:
        assert restored.close()


def test_http_approval_disconnect_and_reopen_keep_one_execution_receipt(tmp_path, monkeypatch):
    with approval_fixture(tmp_path, monkeypatch) as (runtime, run, gate, started, release, resumed, calls, resumes, request):
        path = f"/v1/approvals/{gate['approval']['id']}/decision"
        with _runtime_http(runtime, principal=None) as disconnecting_request:
            disconnecting_request(path, {"approved": True}, abandon_response=True)
            assert started.wait(2) and calls == [True]
            status, reply = request(path, {"approved": True})
            assert status == 200 and reply["tool_result"]["execution_enqueued"] is False
            assert calls == [True] and resumes == [] and not release.is_set()
        release.set()
        assert resumed.wait(2)
        deadline = time.monotonic() + 2
        while runtime.assistant._approval_workers and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        receipt = runtime.store.get_tool_call(gate["tool_call"]["id"])
        assert receipt["status"] == "completed" and receipt["result"]["ok"] is True
        assert calls == [True] and resumes == [True]
    restored = AgentRuntime(tmp_path)
    try:
        monkeypatch.setattr(restored.registry, "invoke", lambda *_: pytest.fail("Approval replay cannot execute after reopen"))
        monkeypatch.setattr(restored.assistant, "_start_or_defer", lambda *_a, **_k: pytest.fail("Approval replay cannot resume after reopen"))
        with _runtime_http(restored, principal=None) as request:
            status, reply = request(path, {"approved": True})
            assert status == 200 and reply["approval"]["status"] == "approved"
            assert reply["tool_result"]["execution_enqueued"] is False
            assert restored.store.get_tool_call(gate["tool_call"]["id"]) == receipt
    finally:
        assert restored.close()


def test_store_reopen_recovers_lost_journal_ack_from_exact_tool_receipt(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="Journal commit interruption fixture", start=False)
    journal = MessageJournal(runtime, run["id"])
    calls = [SimpleNamespace(id=name, name="file_write", input={"path": name + ".txt", "content": name}) for name in ("first", "second")]
    journal.begin([{"role": "assistant", "content": []}], calls, {"file_write"})
    original = journal.write

    def interrupt_after_receipt(path, value):
        if path == journal.pending_path and len(value.get("results", [])) == 1:
            raise RuntimeError("synthetic_journal_commit_interruption")
        return original(path, value)

    monkeypatch.setattr(journal, "write", interrupt_after_receipt)
    try:
        with pytest.raises(RuntimeError, match="synthetic_journal_commit_interruption"):
            journal.settle()
        root = Path(runtime.get_session(run["id"])["workspace_root"])
        stamp = (root / "first.txt").stat().st_mtime_ns
        receipt = runtime.store.get_idempotent_tool_call(run["id"], "first")
        assert receipt["status"] == "completed" and receipt["result"]["ok"]
        assert len(json.loads(journal.pending_path.read_text())["calls"]) == 2
    finally:
        assert runtime.close()
    restored = AgentRuntime(tmp_path)
    try:
        result = MessageJournal(restored, run["id"]).settle()
        assert result["status"] == "settled" and len(result["messages"][-1]["content"]) == 2
        assert (root / "first.txt").stat().st_mtime_ns == stamp
        assert (root / "second.txt").read_text() == "second"
        assert restored.store.get_idempotent_tool_call(run["id"], "first") == receipt
        assert len(restored.store.list_tool_calls(run["id"])) == 2
        assert all(row["status"] == "completed" for row in restored.store.list_tool_calls(run["id"]))
        first_result = json.loads(result["messages"][-1]["content"][0]["content"])
        assert first_result == receipt["result"]
        assert first_result["content"]["sha256"] == hashlib.sha256((root / "first.txt").read_bytes()).hexdigest()
    finally:
        assert restored.close()


@pytest.mark.parametrize("was_paused", [False, True])
def test_unknown_execution_after_store_reopen_is_reconciliation_not_replay(tmp_path, monkeypatch, was_paused):
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="Unknown external execution fixture", start=False)
    monkeypatch.setattr(runtime.registry, "invoke", lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic_unconfirmed_exit")))
    try:
        runtime.store.update_assistant_run(run["id"], status="running")
        with pytest.raises(RuntimeError, match="synthetic_unconfirmed_exit"):
            runtime.invoke_tool(run["id"], "file_write", {"path": "unknown.txt", "content": "never confirmed"}, idempotency_key="unknown_worker")
        call = runtime.store.get_idempotent_tool_call(run["id"], "unknown_worker")
        assert call["status"] == "running" and not call["result"]
        if was_paused:
            runtime.pause(run["id"])
    finally:
        assert runtime.close()
    restored = AgentRuntime(tmp_path)
    try:
        monkeypatch.setattr(restored.assistant, "start", lambda *_a, **_k: pytest.fail("Unknown worker cannot be auto-restarted"))
        monkeypatch.setattr(restored.registry, "invoke", lambda *_a, **_k: pytest.fail("Unknown action cannot be replayed"))
        assert restored.assistant.recover_incomplete() == []
        persisted = restored.store.get_assistant_run(run["id"])
        assert persisted["status"] == "blocked" and persisted["error_class"] == "execution_reconciliation_required"
        state = restored.assistant.snapshot(run["id"])
        assert state["status"] == "blocked" and state["error_class"] == "execution_reconciliation_required"
        assert state["progress"]["freshness"] == "terminal"
        assert restored.assistant.recover_incomplete() == []
        restored.settle_user_pause(run["id"])
        assert restored.store.get_assistant_run(run["id"])["status"] == "blocked"
        assert restored.store.get_tool_call(call["id"]) == call
        answer = restored.invoke_tool(run["id"], "file_write", call["arguments"], idempotency_key="unknown_worker")
        assert answer["status"] == "running" and answer["replayed"] is True
        assert not (Path(restored.get_session(run["id"])["workspace_root"]) / "unknown.txt").exists()
    finally:
        assert restored.close()


def test_cancel_unknown_execution_preserves_cancelled_lifecycle_and_reconciliation_evidence(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt='Cancel unknown execution fixture', start=False)
        runtime.store.update_assistant_run(run['id'], status='running')
        monkeypatch.setattr(runtime.registry, 'invoke', lambda *_: (_ for _ in ()).throw(RuntimeError('unknown_fixture')))
        with pytest.raises(RuntimeError, match='unknown_fixture'):
            runtime.invoke_tool(run['id'], 'file_write', {'path': 'unknown.txt', 'content': 'not executed'}, idempotency_key='unknown_cancel')
        runtime.assistant.recover_incomplete()
        before = runtime.store.get_idempotent_tool_call(run['id'], 'unknown_cancel')
        assert runtime.assistant.snapshot(run['id'])['status'] == 'blocked'
        result = runtime.assistant.action(run['id'], 'cancel')
        after = runtime.assistant.snapshot(run['id'])
        assert result['status'] == after['status'] == 'cancelled'
        assert after['completed_at']
        assert after['error_class'] == 'execution_reconciliation_required'
        assert runtime.store.get_idempotent_tool_call(run['id'], 'unknown_cancel') == before
    finally:
        assert runtime.close()


def test_two_runtime_store_approval_overlap_dispatches_at_most_once(tmp_path, monkeypatch):
    first = AgentRuntime(tmp_path)
    run = first.assistant.create_run(prompt="Two store approval race fixture", start=False)
    gate = first.invoke_tool(run["id"], "file_delete", {"path": "nonexistent.txt"}, idempotency_key="two_store_approval")
    approval = first.decide_approval(gate["approval"]["id"], True, execute=False)["approval"]
    second = AgentRuntime(tmp_path)
    barrier, release, two_entered = threading.Barrier(2), threading.Event(), threading.Event()
    effects, effect_lock = [], threading.Lock()

    def effect(_name, _arguments, _context):
        with effect_lock:
            effects.append("external_action")
            if len(effects) == 2:
                two_entered.set()
        assert release.wait(3)
        return ToolResult("", True, {}, "synthetic receipt")

    # Force a stale-read interleaving at the old INSERT/REPLACE boundary.  A
    # proper admission lease/CAS may serialize or bypass this boundary: the
    # bounded barrier then times out without requiring a second dispatcher.
    for runtime in (first, second):
        original = runtime.store.put_tool_call

        def put(call, result=None, *, original=original):
            if call.status == "running" and result is None:
                try:
                    barrier.wait(timeout=0.5)
                except threading.BrokenBarrierError:
                    pass
            return original(call, result)

        monkeypatch.setattr(runtime.store, "put_tool_call", put)
        monkeypatch.setattr(runtime.registry, "invoke", effect)

    def invoke(runtime):
        return runtime.invoke_tool(run["id"], gate["tool_call"]["tool_name"], gate["tool_call"]["arguments"],
            tool_call_id=gate["tool_call"]["id"], approved_fingerprint=approval["argument_fingerprint"],
            idempotency_key="two_store_approval")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(invoke, runtime) for runtime in (first, second)]
            two_entered.wait(0.9)
            release.set()
            outcomes = [future.result(timeout=3) for future in futures]
        assert effects == ["external_action"], "Two RuntimeStore instances must share atomic execution admission"
        assert all(outcome["status"] in {"completed", "running"} for outcome in outcomes)
        assert any(outcome.get("replayed") for outcome in outcomes)
        first_receipt = first.store.get_tool_call(gate["tool_call"]["id"])
        assert first_receipt == second.store.get_tool_call(gate["tool_call"]["id"])
        assert first_receipt["status"] == "completed" and first_receipt["result"]["ok"]
        kinds = [item["event_type"] for item in first.store.list_events(run["id"])]
        assert kinds.count("tool.started") == kinds.count("tool.completed") == 1
    finally:
        release.set()
        assert second.close(timeout=3)
        assert first.close(timeout=3)


def test_two_runtime_stores_keep_one_exact_approval_decision(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Durable double decision fixture", start=False)
        gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "not-there.txt"}, idempotency_key="double_decision")
        other = RuntimeStore(runtime.runtime_root / "runtime.sqlite3")
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(lambda store: store.decide_approval(gate["approval"]["id"], True), [runtime.store, other]))
            assert all(reply["status"] == "approved" for reply in replies)
            assert runtime.store.get_approval(gate["approval"]["id"]) == other.get_approval(gate["approval"]["id"])
            assert runtime.store.get_tool_call(gate["tool_call"]["id"])["status"] == "waiting_approval"
        finally:
            other.close()
    finally:
        assert runtime.close()


def test_direct_resume_keeps_the_persisted_idempotency_key(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    calls = []
    try:
        session = runtime.create_session(objective="Direct resume identity fixture")
        gate = runtime.invoke_tool(session["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key="resume_once")
        runtime.decide_approval(gate["approval"]["id"], True, execute=False)
        monkeypatch.setattr(runtime.registry, "invoke", lambda *_: calls.append(True) or ToolResult("", True, {}, "Synthetic action"))
        result = runtime.resume(session["id"])
        assert result["status"] == "completed" and calls == [True]
        receipt = runtime.store.get_tool_call(gate["tool_call"]["id"])
        assert receipt["idempotency_key"] == "resume_once"
        replay = runtime.invoke_tool(session["id"], "file_delete", receipt["arguments"], idempotency_key="resume_once")
        assert replay["replayed"] is True and calls == [True]
    finally:
        assert runtime.close()


def test_approval_worker_exception_during_pause_keeps_reconciliation_visible(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Paused unconfirmed approval fixture", start=False)
        gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key="paused_exception")

        def interrupt(*_args):
            runtime.pause(run["id"])
            raise RuntimeError("synthetic_execution_exit_without_receipt")

        monkeypatch.setattr(runtime.registry, "invoke", interrupt)
        runtime.assistant.decide_approval(gate["approval"]["id"], True, background=True)
        deadline = time.monotonic() + 2
        while runtime.assistant._approval_workers and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert not runtime.assistant._approval_workers
        state = runtime.assistant.snapshot(run["id"])
        assert state["status"] == "blocked" and state["error_class"] == "approval_execution_unconfirmed"
        assert runtime.store.get_assistant_run(run["id"])["status"] == "blocked"
        assert runtime.assistant.recover_incomplete() == []
        assert runtime.assistant.snapshot(run["id"])["status"] == "blocked"
    finally:
        assert runtime.close()


@pytest.mark.parametrize("stale_status", ["requested", "queued", "waiting_approval", "running", "completed", "failed"])
def test_stale_tool_copy_cannot_replace_terminal_receipt_or_reopen_execution(tmp_path, stale_status):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Terminal stale copy fixture", start=False)
        first = runtime.invoke_tool(run["id"], "file_write", {"path": "once.txt", "content": "once"}, idempotency_key="immutable_once")
        receipt = runtime.store.get_tool_call(first["result"]["tool_call_id"])
        stale = ToolCall(**{name: value for name, value in receipt.items() if name != "result"})
        stale.status, stale.started_at, stale.completed_at = stale_status, "", ""
        second = RuntimeStore(runtime.runtime_root / "runtime.sqlite3")
        try:
            with pytest.raises(ValueError, match="tool_call_terminal_immutable"):
                second.put_tool_call(stale)
            assert second.get_tool_call(stale.id) == receipt
        finally:
            second.close()
        replay = runtime.invoke_tool(run["id"], "file_write", receipt["arguments"], idempotency_key="immutable_once")
        assert replay["replayed"] is True and replay["result"] == receipt["result"]
        assert len(runtime.store.list_tool_calls(run["id"])) == 1
    finally:
        assert runtime.close()


@pytest.mark.parametrize("stale_status", ["requested", "queued", "waiting_approval", "running"])
def test_running_receipt_cannot_be_reset_by_another_store(tmp_path, stale_status):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Running stale copy fixture", start=False)
        call = ToolCall("call_persisted_running", run["id"], "file_write", {"path": "pending.txt", "content": "x"},
                        status="running", started_at=utc_now(), idempotency_key="running_once")
        runtime.store.put_tool_call(call)
        receipt = runtime.store.get_tool_call(call.id)
        stale = ToolCall(**{**call.to_dict(), "status": stale_status, "started_at": ""})
        second = RuntimeStore(runtime.runtime_root / "runtime.sqlite3")
        try:
            with pytest.raises(ValueError, match="tool_call_execution_"):
                second.put_tool_call(stale)
            assert second.get_tool_call(call.id) == receipt
            claimed, saved = second.admit_tool_call(call)
            assert claimed is False and saved == receipt
        finally:
            second.close()
    finally:
        assert runtime.close()


@pytest.mark.parametrize("different_arguments", [False, True])
def test_new_call_id_cannot_bypass_existing_logical_action_claim(tmp_path, different_arguments):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Call alias fixture", start=False)
        result = runtime.invoke_tool(run["id"], "file_write", {"path": "alias.txt", "content": "once"}, idempotency_key="logical_once")
        receipt = runtime.store.get_tool_call(result["result"]["tool_call_id"])
        arguments = {**receipt["arguments"], **({"content": "changed"} if different_arguments else {})}
        alias = ToolCall("call_alias", run["id"], "file_write", arguments, status="running", started_at=utc_now(), idempotency_key="logical_once")
        other = RuntimeStore(runtime.runtime_root / "runtime.sqlite3")
        try:
            if different_arguments:
                with pytest.raises(ValueError, match="tool_call_idempotency_conflict"):
                    other.admit_tool_call(alias)
            else:
                claimed, saved = other.admit_tool_call(alias)
                assert claimed is False and saved == receipt
            assert other.get_tool_call(alias.id) is None
            assert other.get_tool_call(receipt["id"]) == receipt
        finally:
            other.close()
    finally:
        assert runtime.close()


def _process_admission_fixture(database, descriptor, fingerprint, barrier, queue, marker):
    """Spawn-safe local Store/CAS proof; never constructs a model or GPU client."""
    store = RuntimeStore(database)
    try:
        call = ToolCall(**{key: value for key, value in descriptor.items() if key != "result"})
        call.status, call.started_at = "running", utc_now()
        barrier.wait(timeout=8)
        claimed, saved = store.admit_tool_call(call, approved_fingerprint=fingerprint)
        if claimed:
            with Path(marker).open("a", encoding="utf-8") as handle:
                handle.write(call.id + "\n")
                handle.flush()
            call.status, call.completed_at = "completed", utc_now()
            store.put_tool_call(call, ToolResult(call.id, True, {"fixture": "subprocess_only"}, "Local subprocess receipt"))
        queue.put({"claimed": claimed, "call_id": saved["id"]})
    except Exception as error:
        queue.put({"error_type": type(error).__name__, "error": str(error)})
    finally:
        store.close()


def test_two_actual_processes_share_one_non_expiring_tool_claim(tmp_path):
    runtime = AgentRuntime(tmp_path)
    children = []
    process_context = multiprocessing.get_context("spawn")
    queue, barrier = process_context.Queue(), process_context.Barrier(2)
    try:
        run = runtime.assistant.create_run(prompt="Process isolation claim fixture", start=False)
        gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "unused.txt"}, idempotency_key="process_once")
        approved = runtime.decide_approval(gate["approval"]["id"], True, execute=False)["approval"]
        descriptor = runtime.store.get_tool_call(gate["tool_call"]["id"])
        marker = tmp_path / "subprocess-effects.txt"
        args = (str(runtime.runtime_root / "runtime.sqlite3"), descriptor, approved["argument_fingerprint"], barrier, queue, str(marker))
        children = [process_context.Process(target=_process_admission_fixture, args=args) for _ in range(2)]
        for child in children:
            child.start()
        replies = [queue.get(timeout=12) for _ in children]
        for child in children:
            child.join(timeout=3)
            assert child.exitcode == 0
        assert all("error" not in reply for reply in replies), replies
        assert sorted(reply["claimed"] for reply in replies) == [False, True]
        assert {reply["call_id"] for reply in replies} == {descriptor["id"]}
        assert marker.read_text().splitlines() == [descriptor["id"]]
        receipt = runtime.store.get_tool_call(descriptor["id"])
        assert receipt["status"] == "completed" and receipt["result"]["ok"] is True
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=2)
        queue.close()
        queue.join_thread()
        assert runtime.close()
