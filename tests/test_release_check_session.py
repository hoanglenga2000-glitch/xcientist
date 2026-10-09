"""Release checks sign in through the product's one-time bootstrap flow.

There is no test-only auth hook in product code: the harness exchanges the
server's own single-use, loopback-only bootstrap URL for the ordinary local
session. These tests pin that contract.
"""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB_SRC = ROOT / "web" / "research-agent-workstation" / "src"


def load(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load("release_check_session_auth", "scripts/workstation_local_auth.py")
    monkeypatch.setattr(module, "runtime_dir", lambda: tmp_path)
    return module


def _serve(token: str, seen: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D401 - silence test server
            return

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "origin": self.headers.get("Origin"), "token_ok": body.get("token") == token})
            if body.get("token") != token or seen[:-1]:
                self.send_response(409)
                self.end_headers()
                return
            payload = json.dumps({"ok": True, "csrf_token": "csrf-value-ffffffffffffffffffff"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", "evomind_local_session=session-value-gggggggggggggggg; Path=/; HttpOnly; SameSite=strict")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_establish_exchanges_one_time_bootstrap_and_stores_owner_session(auth, tmp_path, capsys) -> None:
    token = "bootstrap-token-hhhhhhhhhhhhhhhh"
    seen: list[dict] = []
    server = _serve(token, seen)
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        auth.bootstrap_url_path(port).write_text(f"{base}/#bootstrap={token}", encoding="utf-8")
        assert auth.establish_local_test_session(base) == "bootstrapped"
        # Reuse instead of a second (rejected) bootstrap exchange.
        assert auth.establish_local_test_session(base) == "reused"
    finally:
        server.shutdown()
    assert seen == [{"path": "/api/session/bootstrap", "origin": base, "token_ok": True}]
    assert not auth.bootstrap_url_path(port).exists()
    stored = json.loads(auth.test_session_path(port).read_text(encoding="utf-8"))
    assert stored == {"port": port, "cookie": "session-value-gggggggggggggggg", "csrf": "csrf-value-ffffffffffffffffffff"}
    out = capsys.readouterr().out
    assert token not in out and "session-value" not in out


def test_session_is_used_only_for_read_only_requests(auth, tmp_path) -> None:
    base = "http://127.0.0.1:18088"
    (tmp_path / "dashboard.18088.automation.token").write_text("automation-token-iiiiiiiiiiiiii", encoding="ascii")
    auth.test_session_path(18088).write_text(
        json.dumps({"port": 18088, "cookie": "session-value-jjjjjjjjjjjjjjjj", "csrf": "csrf-value-kkkkkkkkkkkkkkkkkk"}),
        encoding="utf-8",
    )
    read = auth.authenticated_headers(base, {"Accept": "application/json"}, method="GET")
    assert read == {"Accept": "application/json", "Cookie": "evomind_local_session=session-value-jjjjjjjjjjjjjjjj"}
    # Legacy callers and mutations keep the principal-free automation header.
    assert auth.authenticated_headers(base) == {auth.AUTOMATION_HEADER: "automation-token-iiiiiiiiiiiiii"}
    assert auth.authenticated_headers(base, method="POST") == {auth.AUTOMATION_HEADER: "automation-token-iiiiiiiiiiiiii"}


@pytest.mark.parametrize("base", ["https://127.0.0.1:18088", "http://evomind.example.com:18088", "http://10.0.0.5:18088"])
def test_session_never_leaves_loopback_http(auth, tmp_path, base) -> None:
    auth.test_session_path(18088).write_text(
        json.dumps({"port": 18088, "cookie": "session-value-llllllllllllllll", "csrf": "csrf-value-mmmmmmmmmmmmmmmmmm"}),
        encoding="utf-8",
    )
    assert auth.local_test_session(base) == {}
    assert "Cookie" not in auth.authenticated_headers(base, method="GET")
    with pytest.raises(RuntimeError, match="test_session_requires_loopback_http"):
        auth.establish_local_test_session(base)


def test_session_file_is_bound_to_its_port_and_value_shape(auth, tmp_path) -> None:
    path = auth.test_session_path(18088)
    path.write_text(json.dumps({"port": 18089, "cookie": "session-value-nnnnnnnnnnnnnnnn", "csrf": "c" * 32}), encoding="utf-8")
    assert auth.local_test_session("http://127.0.0.1:18088") == {}
    path.write_text(json.dumps({"port": 18088, "cookie": "bad value; Path=/", "csrf": "c" * 32}), encoding="utf-8")
    assert auth.local_test_session("http://127.0.0.1:18088") == {}


def test_manager_stop_removes_release_check_session(tmp_path, monkeypatch) -> None:
    manager = load("release_check_session_manager", "scripts/manage_workstation_dashboard.py")
    monkeypatch.setattr(manager, "runtime_dir", lambda: tmp_path)
    session = manager.test_session_path(18088)
    session.write_text("{}", encoding="utf-8")
    manager.remove_local_auth_files(18088)
    assert not session.exists()


def test_product_code_has_no_test_auth_hook_and_bootstrap_stays_loopback_only() -> None:
    """Production safety: the harness adds no product switch to enable."""

    offenders = [
        str(path.relative_to(ROOT))
        for path in WEB_SRC.rglob("*")
        if path.suffix in {".ts", ".tsx", ".mjs", ".js"}
        and ("test-session" in (text := path.read_text(encoding="utf-8", errors="replace"))
             or "WORKSTATION_RELEASE_CHECK" in text)
    ]
    assert offenders == []
    proxy = (WEB_SRC / "proxy.ts").read_text(encoding="utf-8")
    assert 'pathname === "/api/session/bootstrap" && !isLoopbackHostHeader(host)' in proxy
    route = (WEB_SRC / "app" / "api" / "session" / "bootstrap" / "route.ts").read_text(encoding="utf-8")
    assert "bootstrap_consumed" in route and "validBootstrapToken" in route


def test_acceptance_signs_in_before_live_gates_and_cleans_up() -> None:
    text = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")
    establish = text.index('Run-Check "establish_release_check_session"')
    assert text.index('Run-Check "start_production_workstation_frontend"') < establish
    assert establish < text.index('Run-Check "new_user_release_readiness_live"')
    assert "workstation_local_auth.py remove --base-url $BaseUrl" in text
    for script in ("verify_workstation_click_smoke.mjs", "verify_workstation_interactive_controls.mjs"):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8-sig")
        assert "applyReleaseCheckAuth(client, root, baseUrl)" in source
