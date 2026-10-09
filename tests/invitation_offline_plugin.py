"""Network boundary for local invitation acceptance, not production execution."""
from __future__ import annotations

import ipaddress
import socket

import pytest


PROTECTED_PORTS = {7890, 8088, 8765, 18795, 65068}


def _allow(host, port) -> None:
    if host not in {None, "", "localhost", "0.0.0.0", "::"}:
        try:
            loopback = ipaddress.ip_address(str(host)).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise RuntimeError("acceptance_external_network_forbidden")
    if str(port).isdigit() and int(port) in PROTECTED_PORTS:
        raise RuntimeError("acceptance_live_service_port_forbidden")


@pytest.fixture(autouse=True)
def invitation_offline_boundary(monkeypatch):
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        if not isinstance(address, tuple) or len(address) < 2:
            raise RuntimeError("acceptance_socket_target_forbidden")
        _allow(address[0], address[1])
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        if not isinstance(address, tuple) or len(address) < 2:
            raise RuntimeError("acceptance_socket_target_forbidden")
        _allow(address[0], address[1])
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        _allow(host, port)
        return getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
