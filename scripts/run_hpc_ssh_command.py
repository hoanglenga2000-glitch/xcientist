from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import ipaddress

import paramiko


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("SOCKS5 connection closed before the full response was read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def socks5_connect(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, timeout: float = 15.0) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    sock.settimeout(timeout)
    sock.sendall(b"\x05\x01\x00")
    if recv_exact(sock, 2) != b"\x05\x00":
        raise RuntimeError("SOCKS5 method negotiation failed")

    try:
        ipv4 = ipaddress.IPv4Address(dest_host)
        request = b"\x05\x01\x00\x01" + ipv4.packed + struct.pack("!H", dest_port)
    except ipaddress.AddressValueError:
        host_bytes = dest_host.encode("ascii")
        request = b"\x05\x01\x00\x03" + bytes([len(host_bytes)]) + host_bytes + struct.pack("!H", dest_port)
    sock.sendall(request)
    header = recv_exact(sock, 4)
    if len(header) != 4 or header[0] != 5 or header[1] != 0:
        raise RuntimeError(f"SOCKS5 connect failed with response {header!r}")

    address_type = header[3]
    if address_type == 1:
        recv_exact(sock, 4)
    elif address_type == 3:
        recv_exact(sock, recv_exact(sock, 1)[0])
    elif address_type == 4:
        recv_exact(sock, 16)
    else:
        raise RuntimeError(f"Unsupported SOCKS5 address type {address_type}")
    recv_exact(sock, 2)
    return sock


def connect_with_password_or_keyboard_interactive(
    client: paramiko.SSHClient,
    host: str,
    port: int,
    username: str,
    password: str,
    sock: socket.socket | None,
) -> None:
    client.connect(
        host,
        port=port,
        username=username,
        password=password,
        sock=sock,
        allow_agent=False,
        look_for_keys=False,
        timeout=15,
        banner_timeout=15,
        auth_timeout=25,
    )

    def handler(_title: str, _instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
        return [password for _prompt, _echo in prompt_list]

    transport = client.get_transport()
    if transport is None:
        raise paramiko.ssh_exception.AuthenticationException("SSH transport was not established.")
    if transport.is_authenticated():
        return
    transport.auth_interactive(username, handler)


def connect_transport_password(
    client: paramiko.SSHClient,
    host: str,
    port: int,
    username: str,
    password: str,
    sock: socket.socket | None,
) -> None:
    transport = paramiko.Transport(sock if sock is not None else (host, port))
    transport.banner_timeout = 60
    transport.auth_timeout = 30
    transport.connect(username=username, password=password)
    if not transport.is_authenticated():
        transport.close()
        raise paramiko.ssh_exception.AuthenticationException("Password authentication did not complete.")
    client._transport = transport


def connect_keyboard_interactive(
    client: paramiko.SSHClient,
    host: str,
    port: int,
    username: str,
    password: str,
    sock: socket.socket | None,
) -> None:
    transport = paramiko.Transport(sock if sock is not None else (host, port))
    transport.banner_timeout = 15
    transport.auth_timeout = 25
    transport.start_client(timeout=15)

    def handler(_title: str, _instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
        return [password for _prompt, _echo in prompt_list]

    transport.auth_interactive(username, handler)
    if not transport.is_authenticated():
        transport.close()
        raise paramiko.ssh_exception.AuthenticationException("Keyboard-interactive authentication did not complete.")
    client._transport = transport


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one whitelisted command on the HPC SSH gateway without printing secrets.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--proxy-host", default="")
    parser.add_argument("--proxy-port", type=int, default=0)
    parser.add_argument("--command", default="")
    parser.add_argument("--command-file", default="")
    parser.add_argument("--password-env", default="GPU_SSH_PASSWORD")
    args = parser.parse_args()

    command = args.command
    if args.command_file:
        with open(args.command_file, "r", encoding="utf-8") as handle:
            command = handle.read()
    if not command:
        print("--command or --command-file is required.", file=sys.stderr)
        return 2

    password = os.environ.get(args.password_env, "")
    if not password:
        print("GPU SSH password env is not configured.", file=sys.stderr)
        return 2

    sock: socket.socket | None = None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if args.proxy_host:
            sock = socks5_connect(args.proxy_host, args.proxy_port, args.host, args.port)
        try:
            connect_transport_password(
                client,
                args.host,
                args.port,
                args.user,
                password,
                sock,
            )
        except paramiko.ssh_exception.AuthenticationException:
            client.close()
            if sock:
                sock.close()
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            sock = socks5_connect(args.proxy_host, args.proxy_port, args.host, args.port) if args.proxy_host else None
            connect_with_password_or_keyboard_interactive(
                client,
                args.host,
                args.port,
                args.user,
                password,
                sock,
            )
        transport = client.get_transport()
        if not transport or not transport.is_authenticated():
            print(
                "GPU SSH authentication did not fully complete. The login node may require publickey after password partial success.",
                file=sys.stderr,
            )
            return 3
        _, stdout, stderr = client.exec_command(command, timeout=60 * 30)
        exit_status = stdout.channel.recv_exit_status()
        sys.stdout.write(stdout.read().decode("utf-8", "replace"))
        sys.stderr.write(stderr.read().decode("utf-8", "replace"))
        return int(exit_status)
    finally:
        client.close()
        if sock:
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
