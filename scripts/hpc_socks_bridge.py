from __future__ import annotations

import argparse
import ipaddress
import os
import select
import socket
import socketserver
import struct
from pathlib import Path


def secret_value(name: str) -> str:
    direct = os.environ.get(name, "")
    if direct:
        return direct
    file_path = os.environ.get(f"{name}_FILE", "")
    if file_path:
        try:
            return Path(file_path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    secret_dir = os.environ.get("WORKSTATION_SECRET_DIR", "")
    if secret_dir:
        try:
            return (Path(secret_dir) / name).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("unexpected socket close")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def parse_client_request(sock: socket.socket) -> tuple[str, int]:
    header = recv_exact(sock, 2)
    if header[0] != 5:
        raise ConnectionError("only SOCKS5 is supported")
    methods = recv_exact(sock, header[1])
    if 0 not in methods:
        sock.sendall(b"\x05\xff")
        raise ConnectionError("client does not support no-auth SOCKS5")
    sock.sendall(b"\x05\x00")

    version, command, _reserved, address_type = recv_exact(sock, 4)
    if version != 5 or command != 1:
        sock.sendall(b"\x05\x07\x00\x01\x00\x00\x00\x00\x00\x00")
        raise ConnectionError("only SOCKS5 CONNECT is supported")
    if address_type == 1:
        host = socket.inet_ntoa(recv_exact(sock, 4))
    elif address_type == 3:
        length = recv_exact(sock, 1)[0]
        host = recv_exact(sock, length).decode("utf-8")
    elif address_type == 4:
        host = socket.inet_ntop(socket.AF_INET6, recv_exact(sock, 16))
    else:
        sock.sendall(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")
        raise ConnectionError("unsupported SOCKS5 address type")
    port = struct.unpack("!H", recv_exact(sock, 2))[0]
    return host, port


def connect_upstream(upstream_host: str, upstream_port: int, username: str, password: str, dest_host: str, dest_port: int) -> socket.socket:
    upstream = socket.create_connection((upstream_host, upstream_port), timeout=20)
    if username:
        upstream.sendall(b"\x05\x01\x02")
        if recv_exact(upstream, 2) != b"\x05\x02":
            raise ConnectionError("upstream SOCKS5 rejected username/password auth")
        user = username.encode("utf-8")
        password_bytes = password.encode("utf-8")
        upstream.sendall(b"\x01" + bytes([len(user)]) + user + bytes([len(password_bytes)]) + password_bytes)
        if recv_exact(upstream, 2) != b"\x01\x00":
            raise ConnectionError("upstream SOCKS5 username/password auth failed")
    else:
        upstream.sendall(b"\x05\x01\x00")
        if recv_exact(upstream, 2) != b"\x05\x00":
            raise ConnectionError("upstream SOCKS5 rejected no-auth mode")

    host = dest_host.encode("utf-8")
    upstream.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host + struct.pack("!H", dest_port))
    response = recv_exact(upstream, 4)
    if response[1] != 0:
        raise ConnectionError(f"upstream SOCKS5 connect failed with code {response[1]}")
    if response[3] == 1:
        recv_exact(upstream, 4)
    elif response[3] == 3:
        recv_exact(upstream, recv_exact(upstream, 1)[0])
    elif response[3] == 4:
        recv_exact(upstream, 16)
    recv_exact(upstream, 2)
    return upstream


def relay(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 120)
            if not readable:
                continue
            for source in readable:
                data = source.recv(32768)
                if not data:
                    return
                target = right if source is left else left
                target.sendall(data)
    finally:
        for item in sockets:
            try:
                item.close()
            except OSError:
                pass


class SocksBridgeHandler(socketserver.BaseRequestHandler):
    upstream_host: str
    upstream_port: int
    username: str
    password: str

    def handle(self) -> None:
        try:
            dest_host, dest_port = parse_client_request(self.request)
            upstream = connect_upstream(self.upstream_host, self.upstream_port, self.username, self.password, dest_host, dest_port)
            self.request.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            relay(self.request, upstream)
        except Exception:
            try:
                self.request.sendall(b"\x05\x01\x00\x01\x00\x00\x00\x00\x00\x00")
            except OSError:
                pass


class ThreadedSocksBridge(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Local no-auth SOCKS5 bridge for HKUST(GZ) HPC resources.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=7890)
    parser.add_argument("--upstream-host", default=os.environ.get("HPC_SOCKS_HOST") or "")
    parser.add_argument("--upstream-port", type=int, default=int(os.environ.get("HPC_SOCKS_PORT") or "0"))
    args = parser.parse_args()

    try:
        if not ipaddress.ip_address(args.listen_host).is_loopback:
            raise ValueError("listen host must be loopback")
    except ValueError as exc:
        raise SystemExit(f"HPC_SOCKS_BRIDGE_FAILED: {exc}") from exc
    if not args.upstream_host or not 1 <= args.upstream_port <= 65535:
        raise SystemExit("HPC_SOCKS_BRIDGE_FAILED: upstream host and port must be configured explicitly")

    username = secret_value("GPU_SSH_SOCKS_USER") or secret_value("HPC_SOCKS_USER")
    password = secret_value("GPU_SSH_SOCKS_PASSWORD") or secret_value("HPC_SOCKS_PASSWORD")
    if not username or not password:
        raise SystemExit("HPC_SOCKS_BRIDGE_FAILED: set HPC_SOCKS_USER and HPC_SOCKS_PASSWORD or *_FILE")
    if len(username.encode("utf-8")) > 255 or len(password.encode("utf-8")) > 255:
        raise SystemExit("HPC_SOCKS_BRIDGE_FAILED: SOCKS credentials exceed protocol limits")

    SocksBridgeHandler.upstream_host = args.upstream_host
    SocksBridgeHandler.upstream_port = args.upstream_port
    SocksBridgeHandler.username = username
    SocksBridgeHandler.password = password
    with ThreadedSocksBridge((args.listen_host, args.listen_port), SocksBridgeHandler) as server:
        print(f"HPC SOCKS bridge listening on {args.listen_host}:{args.listen_port} -> {args.upstream_host}:{args.upstream_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
