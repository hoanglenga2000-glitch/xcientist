from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from evomind_runtime.http_server import MAX_BODY_BYTES, ensure_token, make_handler
from evomind_runtime.runtime import AgentRuntime


def request(port: int, method: str, path: str, token: str, *, body: bytes | None = None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    merged = {"Authorization": f"Bearer {token}", **(headers or {})}
    connection.request(method, path, body=body, headers=merged)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


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
