from __future__ import annotations

import hashlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from evomind_runtime.http_server import make_handler
from evomind_runtime.models import ApprovalRequest
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tenant_access import AccessError, AccessStore, Principal


ALICE = Principal("tenant_" + "a" * 24, "alice")
BOB = Principal("tenant_" + "b" * 24, "bob")
COLLEAGUE = Principal(ALICE.tenant_id, "colleague")
TOKEN = "fixture-runtime-service-token-only"


@pytest.fixture
def api(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    create = runtime.assistant.create_run
    monkeypatch.setattr(runtime.assistant, "create_run", lambda **kw: create(**{**kw, "start": False}))
    monkeypatch.setattr(runtime.assistant, "start", lambda *_args, **_kwargs: False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, TOKEN))
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()

    def request(method, path, principal=ALICE, body=None, raw=None, extra=None):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        if principal is not None:
            headers.update({
                "X-EvoMind-Access-Scope": "user.v1", "X-EvoMind-Tenant-Id": principal.tenant_id,
                "X-EvoMind-Principal-Id": principal.owner_id,
            })
        payload = raw
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        headers.update(extra or {})
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            data = response.read()
            content = json.loads(data) if "application/json" in response.getheader("Content-Type", "") else data
            return response.status, content
        finally:
            connection.close()

    try:
        yield runtime, request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def new_run(request, principal=ALICE, **extra):
    status, run = request("POST", "/v1/runs", principal, {"prompt": "Summarize this local invitation test.", **extra})
    assert status == 201, run
    return run


@pytest.mark.parametrize("other", [BOB, COLLEAGUE])
def test_run_session_events_actions_and_lists_are_owner_scoped(api, other):
    runtime, request = api
    run = new_run(request, conversation_id="conversation-alice")
    second = new_run(request, other, conversation_id="conversation-other")
    for path in [f"/v1/runs/{run['id']}", f"/v1/runs/{run['id']}/events", f"/v1/sessions/{run['id']}", f"/v1/sessions/{run['id']}/events"]:
        assert request("GET", path)[0] == 200
        assert request("GET", path, other) == (404, {"error": "not_found"})
    assert request("POST", f"/v1/runs/{run['id']}/actions", other, {"action": "cancel"})[0] == 404
    assert {row["id"] for row in request("GET", "/v1/runs")[1]["runs"]} == {run["id"]}
    assert {row["id"] for row in request("GET", "/v1/runs", other)[1]["runs"]} == {second["id"]}
    assert {row["id"] for row in request("GET", "/v1/sessions")[1]["sessions"]} == {run["id"]}
    assert request("POST", "/v1/runs", other, {"prompt": "Do not inherit another conversation", "conversation_id": "conversation-alice"})[0] == 404
    assert runtime.store.get_assistant_run(run["id"])["status"] != "cancelled"


def test_upload_chunks_completion_and_attachment_reuse_require_same_owner(api):
    _, request = api
    data = b"name,value\none,1\n"
    digest = hashlib.sha256(data).hexdigest()
    status, upload = request("POST", "/v1/uploads", body={"name": "fixture.txt", "total_bytes": len(data), "media_type": "text/plain", "sha256": digest})
    assert status == 201, upload
    upload_id = upload.get("upload_id") or upload.get("id")
    chunk_path = f"/v1/uploads/{upload_id}/chunks/0"
    chunk_headers = {"Content-Type": "application/octet-stream", "X-Chunk-SHA256": digest}
    assert request("PUT", chunk_path, BOB, raw=data, extra=chunk_headers)[0] == 404
    assert request("PUT", chunk_path, raw=data, extra=chunk_headers)[0] == 200
    complete_path = f"/v1/uploads/{upload_id}/complete"
    assert request("POST", complete_path, BOB, {})[0] == 404
    status, completed = request("POST", complete_path, body={})
    assert status == 200, completed
    attachment = completed.get("attachment", completed)
    attachment_id = attachment.get("attachment_id") or attachment.get("id")
    assert attachment_id
    assert request("POST", "/v1/runs", BOB, {"prompt": "Do not attach another user's input", "attachment_ids": [attachment_id]})[0] == 404
    assert new_run(request, attachment_ids=[attachment_id])["id"]


def test_approvals_and_artifact_bytes_are_authorized_before_use(api, monkeypatch, tmp_path):
    runtime, request = api
    run = new_run(request)
    approval = ApprovalRequest("approval-owner-fixture", run["id"], "call-fixture", "managed_model_prepare", "a" * 64, {}, {}, "high", False)
    runtime.store.put_approval(approval)
    assert request("GET", "/v1/approvals", BOB)[1]["approvals"] == []
    assert request("GET", "/v1/approvals")[1]["approvals"][0]["id"] == approval.id
    assert request("POST", f"/v1/approvals/{approval.id}/decision", BOB, {"approved": True})[0] == 404
    assert runtime.store.get_approval(approval.id)["status"] == "pending"

    artifact = {"id": "artifact-scope-fixture", "run_id": run["id"], "name": "fixture.txt", "media_type": "text/plain", "bytes": 5, "sha256": hashlib.sha256(b"owned").hexdigest()}
    path = tmp_path / "owned-fixture.txt"
    path.write_bytes(b"owned")
    monkeypatch.setattr(runtime.store, "get_deliverable", lambda key: artifact if key == artifact["id"] else None, raising=False)
    calls = []

    def artifact_file(key, derived=""):
        calls.append(key)
        return artifact, path

    monkeypatch.setattr(runtime.assistant, "artifact_file", artifact_file)
    monkeypatch.setattr(runtime.assistant, "preview", lambda key: {"text": "owned"})
    for suffix in ["", "/preview"]:
        assert request("GET", f"/v1/artifacts/{artifact['id']}{suffix}", BOB)[0] == 404
    assert calls == []
    assert request("GET", f"/v1/artifacts/{artifact['id']}") == (200, b"owned")


def test_legacy_unowned_runs_and_browser_supplied_scope_do_not_grant_access(api):
    runtime, request = api
    legacy = runtime.assistant.create_run(prompt="Legacy administrative fixture")
    assert request("GET", f"/v1/runs/{legacy['id']}")[0] == 404
    assert request("GET", f"/v1/runs/{legacy['id']}", None)[0] == 200
    assert request("GET", "/v1/health", extra={"Authorization": "Bearer wrong"})[0] == 401
    assert request("GET", "/v1/health", extra={"X-EvoMind-Tenant-Id": ""})[0] == 400
    assert request("POST", "/v1/sessions", body={"permission_level": "full_access"})[0] == 403
    assert request("GET", "/v1/benchmarks")[0] == 403
    assert request("POST", "/v1/runs", body={"prompt": "Reject forged managed identity", "managed_hpc_identity": {"tenant_id": BOB.tenant_id, "owner_principal_id": "bob"}})[0] == 403


def test_acl_survives_reopen_and_cannot_be_reassigned(tmp_path):
    first = AccessStore(tmp_path)
    first.bind("upload", "upload-persisted", ALICE)
    second = AccessStore(tmp_path)
    second.require("upload", "upload-persisted", ALICE)
    with pytest.raises(AccessError):
        second.bind("upload", "upload-persisted", BOB)
    assert second.owns("upload", "upload-persisted", ALICE)
    assert not second.owns("upload", "upload-persisted", BOB)
