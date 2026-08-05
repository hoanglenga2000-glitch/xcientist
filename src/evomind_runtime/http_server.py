from __future__ import annotations

import argparse
import json
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .benchmark import parity_status, parity_suite
from .runtime import AgentRuntime

MAX_BODY_BYTES = 1024 * 1024
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


class RequestContractError(ValueError):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


def ensure_token(root: Path) -> str:
    path = root / "runtime.token"
    current = ""
    try:
        current = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        pass
    if TOKEN_PATTERN.fullmatch(current):
        return current

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(token, encoding="ascii", newline="\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return token


def make_handler(runtime: AgentRuntime, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EvoMindRuntime/0.3"

        def _json(self, status: int, value):
            data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            raw_size = self.headers.get("Content-Length")
            if raw_size is None:
                raise RequestContractError(411, "content_length_required")
            if not raw_size.isdigit():
                raise RequestContractError(400, "invalid_content_length")
            size = int(raw_size)
            if size > MAX_BODY_BYTES:
                raise RequestContractError(413, "body_too_large")
            data = self.rfile.read(size)
            if len(data) != size:
                raise RequestContractError(400, "incomplete_body")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise RequestContractError(415, "unsupported_content_type")
            try:
                value = json.loads(data or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RequestContractError(400, "invalid_json") from exc
            if not isinstance(value, dict):
                raise RequestContractError(400, "json_object_required")
            return value

        def _authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {token}"):
                self._json(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            try:
                if parsed.path == "/v1/health":
                    return self._json(200, {"status": "ready", "version": "0.3.0"})
                if parsed.path == "/v1/tools":
                    return self._json(200, {"tools": runtime.tools()})
                if parsed.path == "/v1/sessions":
                    return self._json(200, {"sessions": runtime.list_sessions()})
                if parsed.path == "/v1/approvals":
                    return self._json(
                        200, {"approvals": runtime.store.list_approvals(parse_qs(parsed.query).get("status", [""])[0])}
                    )
                if parsed.path == "/v1/benchmarks":
                    return self._json(
                        200,
                        {
                            "suite": parity_suite(),
                            "runs": runtime.store.list_benchmarks(),
                            "gate": parity_status(runtime.store.list_benchmarks()),
                        },
                    )
                if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
                    return self._json(200, runtime.get_session(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "events":
                    after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                    events = runtime.store.list_events(parts[2], after)
                    if "text/event-stream" in self.headers.get("Accept", ""):
                        data = "".join(
                            f"id: {e['seq']}\nevent: {e['event_type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                            for e in events
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    return self._json(200, {"events": events})
                self._json(404, {"error": "not_found"})
            except KeyError as exc:
                self._json(404, {"error": "not_found", "id": str(exc)})
            except Exception as exc:
                self._json(500, {"error": type(exc).__name__, "message": str(exc)})

        def do_POST(self):
            if not self._authorized():
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                body = self._body()
                if parts == ["v1", "sessions"]:
                    return self._json(201, runtime.create_session(**body))
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":
                    return self._json(
                        200,
                        runtime.message(
                            parts[2], str(body.get("content", "")), max_steps=int(body.get("max_steps", 12))
                        ),
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "tools":
                    return self._json(
                        200,
                        runtime.invoke_tool(
                            parts[2],
                            str(body["tool_name"]),
                            dict(body.get("arguments") or {}),
                            idempotency_key=str(body.get("idempotency_key") or ""),
                        ),
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "resume":
                    return self._json(200, runtime.resume(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "cancel":
                    return self._json(200, runtime.cancel(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "approvals"] and parts[3] == "decision":
                    return self._json(
                        200, runtime.decide_approval(parts[2], bool(body.get("approved")), str(body.get("note", "")))
                    )
                self._json(404, {"error": "not_found"})
            except RequestContractError as exc:
                self._json(exc.status, {"error": exc.code})
            except KeyError as exc:
                self._json(404, {"error": "not_found", "id": str(exc)})
            except Exception as exc:
                self._json(500, {"error": type(exc).__name__, "message": str(exc)})

        def log_message(self, _format, *_args):
            pass

    return Handler


def serve(workspace: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    runtime = AgentRuntime(workspace)
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer((host, port), make_handler(runtime, token))
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        runtime.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(Path(args.workspace).resolve(), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
