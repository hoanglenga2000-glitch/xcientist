from __future__ import annotations

import os
import socket
import struct
import sys
import threading


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def secret_value(name: str) -> str:
    direct = os.environ.get(name, "")
    if direct:
        return direct
    file_path = os.environ.get(f"{name}_FILE", "")
    if file_path:
        try:
            return open(file_path, "r", encoding="utf-8").read().strip()
        except OSError:
            return ""
    secret_dir = os.environ.get("WORKSTATION_SECRET_DIR", "")
    if secret_dir:
        candidate = os.path.join(secret_dir, name)
        try:
            return open(candidate, "r", encoding="utf-8").read().strip()
        except OSError:
            return ""
    return ""


def socks5_connect(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, username: str = "", password: str = "") -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=20)
    if username:
        sock.sendall(b"\x05\x01\x02")
        if sock.recv(2) != b"\x05\x02":
            fail("SOCKS5 proxy rejected username/password mode")
        user = username.encode("utf-8")
        password_bytes = password.encode("utf-8")
        if len(user) > 255 or len(password_bytes) > 255:
            fail("SOCKS5 username/password is too long")
        sock.sendall(b"\x01" + bytes([len(user)]) + user + bytes([len(password_bytes)]) + password_bytes)
        if sock.recv(2) != b"\x01\x00":
            fail("SOCKS5 username/password authentication failed")
    else:
        sock.sendall(b"\x05\x01\x00")
        if sock.recv(2) != b"\x05\x00":
            fail("SOCKS5 proxy rejected no-auth mode")
    host = dest_host.encode("utf-8")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host + struct.pack("!H", dest_port))
    head = sock.recv(4)
    if len(head) < 4 or head[1] != 0:
        fail(f"SOCKS5 connect failed: {head!r}")
    if head[3] == 1:
        sock.recv(4)
    elif head[3] == 3:
        sock.recv(sock.recv(1)[0])
    elif head[3] == 4:
        sock.recv(16)
    sock.recv(2)
    sock.settimeout(None)
    return sock


def main() -> None:
    if len(sys.argv) not in {5, 6}:
        fail("usage: hpc_socks_proxy.py <proxy_host> <proxy_port> <dest_host> <dest_port> [proxy_user]")
    proxy_host, proxy_port, dest_host, dest_port = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    proxy_user = sys.argv[5] if len(sys.argv) == 6 else os.environ.get("GPU_SSH_SOCKS_USER", "") or os.environ.get("HPC_SOCKS_USER", "")
    proxy_password = secret_value("GPU_SSH_SOCKS_PASSWORD") or secret_value("HPC_SOCKS_PASSWORD")
    sock = socks5_connect(proxy_host, proxy_port, dest_host, dest_port, proxy_user, proxy_password)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    def stdin_to_socket() -> None:
        while True:
            data = os.read(sys.stdin.fileno(), 32768)
            if not data:
                try:
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            sock.sendall(data)

    pump = threading.Thread(target=stdin_to_socket, daemon=True)
    pump.start()
    while True:
        data = sock.recv(32768)
        if not data:
            break
        stdout.write(data)
        stdout.flush()


if __name__ == "__main__":
    main()
