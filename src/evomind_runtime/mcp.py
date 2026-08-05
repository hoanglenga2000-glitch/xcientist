from __future__ import annotations

import json
import subprocess
import threading
from typing import Any


class MCPClient:
    """Small stdio JSON-RPC client supporting newline and Content-Length framing."""

    def __init__(self, command: list[str], timeout: float = 30, framing: str = "newline") -> None:
        self.command = command
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        if framing not in {"newline", "content-length"}:
            raise ValueError("framing must be newline or content-length")
        self.framing = framing

    def __enter__(self) -> "MCPClient":
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "evomind", "version": "0.3.0"}})
        self.notify("notifications/initialized", {})
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _write(self, payload: dict[str, Any]) -> None:
        assert self.process and self.process.stdin
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.framing == "content-length":
            self.process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
        else:
            self.process.stdin.write(data + b"\n")
        self.process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self.process and self.process.stdout
        result: list[dict[str, Any]] = []
        error: list[BaseException] = []
        def worker() -> None:
            try:
                first = self.process.stdout.readline()
                if first.lower().startswith(b"content-length:"):
                    length = int(first.split(b":", 1)[1].strip())
                    while self.process.stdout.readline() not in {b"\n", b"\r\n", b""}:
                        pass
                    data = self.process.stdout.read(length)
                else:
                    data = first
                result.append(json.loads(data.decode("utf-8")))
            except BaseException as exc:
                error.append(exc)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            raise TimeoutError("MCP response timed out")
        if error:
            raise RuntimeError(f"MCP response failed: {error[0]}")
        return result[0]

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            response = self._read()
            if response.get("id") == request_id:
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                return dict(response.get("result") or {})

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})
