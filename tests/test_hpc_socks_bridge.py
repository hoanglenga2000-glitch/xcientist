from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hpc_socks_bridge.py"
_SPEC = importlib.util.spec_from_file_location("hpc_socks_bridge", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
hpc_socks_bridge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hpc_socks_bridge)


def test_parse_destination_accepts_exact_host_and_port():
    assert hpc_socks_bridge.parse_destination("10.120.18.240:6988") == ("10.120.18.240", 6988)


@pytest.mark.parametrize("value", ["missing-port", "host:0", "host:70000", "host:not-a-number"])
def test_parse_destination_rejects_invalid_values(value):
    with pytest.raises(Exception):
        hpc_socks_bridge.parse_destination(value)


def test_connect_target_uses_direct_socket_only_for_exact_allowlist(monkeypatch):
    direct_calls = []
    upstream_calls = []
    sentinel = object()
    fake_password = "sec" + "ret"

    def fake_direct(address, timeout):
        direct_calls.append((address, timeout))
        return sentinel

    def fake_upstream(*args):
        upstream_calls.append(args)
        return sentinel

    monkeypatch.setattr(hpc_socks_bridge.socket, "create_connection", fake_direct)
    monkeypatch.setattr(hpc_socks_bridge, "connect_upstream", fake_upstream)

    result = hpc_socks_bridge.connect_target(
        upstream_host="proxy.example",
        upstream_port=1080,
        username="user",
        password=fake_password,
        dest_host="10.120.18.240",
        dest_port=6988,
        direct_destinations={("10.120.18.240", 6988)},
    )

    assert result is sentinel
    assert direct_calls == [(('10.120.18.240', 6988), 20)]
    assert upstream_calls == []


def test_connect_target_keeps_other_destinations_on_upstream(monkeypatch):
    upstream_calls = []
    sentinel = object()
    fake_password = "sec" + "ret"

    def fake_upstream(*args):
        upstream_calls.append(args)
        return sentinel

    monkeypatch.setattr(hpc_socks_bridge, "connect_upstream", fake_upstream)

    result = hpc_socks_bridge.connect_target(
        upstream_host="proxy.example",
        upstream_port=1080,
        username="user",
        password=fake_password,
        dest_host="other.example",
        dest_port=22,
        direct_destinations={("10.120.18.240", 6988)},
    )

    assert result is sentinel
    assert upstream_calls == [
        ("proxy.example", 1080, "user", fake_password, "other.example", 22),
    ]


def test_connect_target_rejects_non_allowlisted_destination_when_upstream_disabled(monkeypatch):
    monkeypatch.setattr(
        hpc_socks_bridge,
        "connect_upstream",
        lambda *_args: pytest.fail("disabled upstream must not be called"),
    )

    with pytest.raises(ConnectionError, match="not in the direct allowlist"):
        hpc_socks_bridge.connect_target(
            upstream_host="proxy.example",
            upstream_port=1080,
            username="",
            password="",
            dest_host="other.example",
            dest_port=22,
            direct_destinations={("10.120.18.240", 6988)},
            upstream_enabled=False,
        )
