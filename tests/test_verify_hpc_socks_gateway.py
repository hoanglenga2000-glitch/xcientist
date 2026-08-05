from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_hpc_socks_gateway.py"
_SPEC = importlib.util.spec_from_file_location("verify_hpc_socks_gateway", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
gateway = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gateway)


class _SocketFixture:
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    def sendall(self, _payload: bytes) -> None:
        return None

    def recv(self, _size: int) -> bytes:
        return next(self._responses)

    def close(self) -> None:
        return None


def _responses(banner: bytes) -> list[bytes]:
    return [
        b"\x05\x00",
        b"\x05\x00\x00\x01",
        b"\x00\x00\x00\x00",
        b"\x00\x00",
        banner,
    ]


def test_socks5_banner_requires_ssh_protocol_banner(monkeypatch):
    monkeypatch.setattr(gateway.socket, "create_connection", lambda *_args, **_kwargs: _SocketFixture(_responses(b"")))

    with pytest.raises(RuntimeError, match="SSH protocol banner"):
        gateway.socks5_banner("127.0.0.1", 7890, "10.120.18.240", 6988, "", "")


def test_socks5_banner_accepts_ssh_banner(monkeypatch):
    monkeypatch.setattr(
        gateway.socket,
        "create_connection",
        lambda *_args, **_kwargs: _SocketFixture(_responses(b"SSH-2.0-test\r\n")),
    )

    assert gateway.socks5_banner("127.0.0.1", 7890, "10.120.18.240", 6988, "", "") == "SSH-2.0-test"
