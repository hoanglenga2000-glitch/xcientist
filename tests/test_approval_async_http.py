import contextlib
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import sqlite3
import threading
import time
import urllib.request

from evomind_runtime.http_server import make_handler
from evomind_runtime.models import ToolCall, ToolResult
from evomind_runtime.runtime import AgentRuntime


@contextlib.contextmanager
def approval_fixture(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="Isolated HTTP approval fixture", start=False)
    gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key="same-action")
    started, release, resumed = threading.Event(), threading.Event(), threading.Event()
    calls, resumes = [], []
    def invoke(_name, _arguments, context):
        calls.append(context.approval_verified)
        started.set()
        if not release.wait(5):
            raise RuntimeError("fixture_release_timeout")
        return ToolResult("", True, {}, "isolated fixture only")
    def resume(*_args, **_kwargs):
        resumes.append(True)
        resumed.set()
    monkeypatch.setattr(runtime.registry, "invoke", invoke)
    monkeypatch.setattr(runtime.assistant, "_start_or_defer", resume)
    token = "isolated-approval-http-fixture"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, data=None):
        encoded = None if data is None else json.dumps(data).encode()
        query = urllib.request.Request(f"http://127.0.0.1:{server.server_port}"+path, data=encoded,
                                       headers={"Authorization": "Bearer "+token, "Content-Type": "application/json"})
        with opener.open(query, timeout=1.5) as response:
            return response.status, json.load(response)
    try:
        yield runtime, run, gate, started, release, resumed, calls, resumes, request
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert runtime.close(timeout=3)


def test_real_http_approval_acknowledges_before_external_action_finishes(tmp_path, monkeypatch):
    with approval_fixture(tmp_path, monkeypatch) as (runtime, run, gate, started, release, resumed, calls, resumes, request):
        before = time.monotonic()
        status, reply = request('/v1/approvals/'+gate['approval']['id']+'/decision', {"approved": True})
        assert status == 200 and time.monotonic()-before < 1
        assert reply["approval"]["status"] == "approved"
        assert reply["run"]["id"] == run["id"]
        assert started.wait(1) and not release.is_set() and not resumed.is_set()
        status, state = request('/v1/runs/'+run['id'])
        assert status == 200 and state["status"] == "running"
        assert state["active_tool_calls"] == [gate["tool_call"]["id"]]
        assert calls == [True]


def test_http_concurrent_and_terminal_replays_do_not_redispatch(tmp_path, monkeypatch):
    with approval_fixture(tmp_path, monkeypatch) as (runtime, run, gate, started, release, resumed, calls, resumes, request):
        path = '/v1/approvals/'+gate['approval']['id']+'/decision'
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: request(path, {"approved": True}), range(8)))
        assert all(status == 200 and reply["approval"]["status"] == "approved" for status, reply in replies)
        assert started.wait(1) and calls == [True]
        release.set()
        assert resumed.wait(2)
        deadline = time.monotonic()+2
        while runtime.assistant._approval_workers and time.monotonic() < deadline:
            time.sleep(0.01)
        status, reply = request(path, {"approved": True})
        assert status == 200 and reply["tool_result"]["execution_enqueued"] is False
        assert calls == [True] and resumes == [True]
        assert not runtime.assistant._approval_workers


def test_recovery_does_not_restart_an_unsettled_external_action(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Unsettled execution fixture", start=False)
        gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key="same-action")
        runtime.decide_approval(gate["approval"]["id"], True, execute=False)
        saved = gate["tool_call"]
        call = ToolCall(saved["id"], run["id"], saved["tool_name"], saved["arguments"], idempotency_key="same-action")
        call.status, call.approval_id = "running", gate["approval"]["id"]
        runtime.store.put_tool_call(call)
        runtime.store.update_assistant_run(run["id"], status="running")
        monkeypatch.setattr(runtime.assistant, "start", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must_not_restart")))
        assert runtime.assistant.recover_incomplete() == []
        state = runtime.store.get_assistant_run(run["id"])
        assert state["status"] == "blocked" and state["error_class"] == "execution_reconciliation_required"
        assert runtime.store.get_tool_call(call.id)["status"] == "running"
        assert not runtime.assistant._approval_workers
    finally:
        runtime.close()


def test_recovery_preserves_the_live_local_approval_worker(tmp_path, monkeypatch):
    with approval_fixture(tmp_path, monkeypatch) as (runtime, run, gate, started, release, resumed, calls, resumes, request):
        request('/v1/approvals/'+gate['approval']['id']+'/decision', {"approved": True})
        assert started.wait(1)
        assert runtime.assistant.recover_incomplete() == []
        assert runtime.store.get_assistant_run(run["id"])["status"] == "running"
        assert calls == [True]


def test_expired_recorded_approval_cannot_execute_after_restart(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run = runtime.assistant.create_run(prompt="Expired dispatch fixture", start=False)
        gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key="same-action")
        runtime.decide_approval(gate["approval"]["id"], True, execute=False)
        with sqlite3.connect(runtime.runtime_root / "runtime.sqlite3") as connection:
            connection.execute("UPDATE approvals SET expires_at=? WHERE id=?", ("2000-01-01T00:00:00+00:00", gate["approval"]["id"]))
        called = []
        monkeypatch.setattr(runtime.registry, "invoke", lambda *_args: called.append(True))
        runtime.store.update_assistant_run(run["id"], status="waiting_approval")
        runtime.assistant.recover_incomplete()
        deadline = time.monotonic()+2
        while runtime.assistant._approval_workers and time.monotonic()<deadline:
            time.sleep(0.01)
        assert called == []
        pending = runtime.store.list_approvals(status="pending")
        assert len(pending) == 1 and pending[0]["id"] != gate["approval"]["id"]
        assert runtime.store.get_assistant_run(run["id"])["status"] == "waiting_approval"
    finally:
        runtime.close(timeout=3)
