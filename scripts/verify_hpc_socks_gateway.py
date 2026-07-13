from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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


def socks5_banner(proxy_host: str, proxy_port: int, dest_host: str, dest_port: int, username: str, password: str) -> str:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    if username:
        sock.sendall(b"\x05\x01\x02")
        if sock.recv(2) != b"\x05\x02":
            raise RuntimeError("SOCKS5 proxy did not accept username/password mode")
        user = username.encode("utf-8")
        password_bytes = password.encode("utf-8")
        sock.sendall(b"\x01" + bytes([len(user)]) + user + bytes([len(password_bytes)]) + password_bytes)
        if sock.recv(2) != b"\x01\x00":
            raise RuntimeError("SOCKS5 username/password authentication failed")
    else:
        sock.sendall(b"\x05\x01\x00")
        if sock.recv(2) != b"\x05\x00":
            raise RuntimeError("SOCKS5 proxy rejected no-auth mode")

    host = dest_host.encode("utf-8")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(host)]) + host + struct.pack("!H", dest_port))
    head = sock.recv(4)
    if len(head) < 4 or head[1] != 0:
        raise RuntimeError(f"SOCKS5 connect failed: {head!r}")
    if head[3] == 1:
        sock.recv(4)
    elif head[3] == 3:
        sock.recv(sock.recv(1)[0])
    elif head[3] == 4:
        sock.recv(16)
    sock.recv(2)
    banner = sock.recv(64).decode("ascii", "replace").strip()
    sock.close()
    return banner


def run_windows_bridge_manager(action: str, *arguments: str) -> dict[str, Any]:
    manager = ROOT / "scripts" / "manage_hpc_proxy_bridge.ps1"
    powershell = "powershell.exe" if os.name != "nt" else "powershell"
    completed = subprocess.run(
        [powershell, "-ExecutionPolicy", "Bypass", "-File", str(manager), action, *arguments],
        text=True,
        capture_output=True,
        timeout=30,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"raw_stdout": completed.stdout.strip(), "raw_stderr": completed.stderr.strip()}
    payload["returncode"] = completed.returncode
    return payload


def verified_dpapi_bridge(dest_host: str, dest_port: int) -> dict[str, Any] | None:
    try:
        status = run_windows_bridge_manager("status")
        if status.get("returncode") != 0 or status.get("status") != "running" or not status.get("credential_installed"):
            return None
        test = run_windows_bridge_manager(
            "test",
            "-DestinationHost",
            dest_host,
            "-DestinationPort",
            str(dest_port),
        )
        if test.get("returncode") != 0 or test.get("status") != "passed":
            return None
        return {
            "status": "passed",
            "proxy": "127.0.0.1:7890",
            "destination": f"{dest_host}:{dest_port}",
            "auth_mode": "windows_dpapi_bridge",
            "banner": test.get("banner"),
            "bridge_status": {
                "status": status.get("status"),
                "credential_installed": status.get("credential_installed"),
                "listen_port": status.get("listen_port"),
                "pid": status.get("pid"),
            },
        }
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an HPC SOCKS5 proxy can reach the SSH gateway banner.")
    parser.add_argument("--proxy-host", default=os.environ.get("GPU_SSH_SOCKS_HOST") or os.environ.get("HPC_SOCKS_HOST") or "")
    parser.add_argument("--proxy-port", type=int, default=int(os.environ.get("GPU_SSH_SOCKS_PORT") or os.environ.get("HPC_SOCKS_PORT") or "0"))
    parser.add_argument("--dest-host", default=os.environ.get("GPU_SSH_HOST") or "")
    parser.add_argument("--dest-port", type=int, default=int(os.environ.get("GPU_SSH_PORT") or "0"))
    parser.add_argument("--require-auth", action="store_true")
    args = parser.parse_args()

    missing_endpoint = []
    if not args.proxy_host or not 1 <= args.proxy_port <= 65535:
        missing_endpoint.append("proxy_host_port")
    if not args.dest_host or not 1 <= args.dest_port <= 65535:
        missing_endpoint.append("destination_host_port")
    if missing_endpoint:
        print(json.dumps({"status": "not_configured", "missing": missing_endpoint}, indent=2))
        return

    username = secret_value("GPU_SSH_SOCKS_USER") or secret_value("HPC_SOCKS_USER")
    password = secret_value("GPU_SSH_SOCKS_PASSWORD") or secret_value("HPC_SOCKS_PASSWORD")
    if args.require_auth and (not username or not password):
        dpapi_bridge = verified_dpapi_bridge(args.dest_host, args.dest_port)
        if dpapi_bridge:
            print(json.dumps(dpapi_bridge, ensure_ascii=False, indent=2))
            return
        print(json.dumps({
            "status": "not_configured",
            "missing_env": [name for name, value in {"GPU_SSH_SOCKS_USER": username, "GPU_SSH_SOCKS_PASSWORD": password}.items() if not value],
        }, ensure_ascii=False, indent=2))
        return

    try:
        banner = socks5_banner(args.proxy_host, args.proxy_port, args.dest_host, args.dest_port, username, password)
        print(json.dumps({
            "status": "passed",
            "proxy": f"{args.proxy_host}:{args.proxy_port}",
            "destination": f"{args.dest_host}:{args.dest_port}",
            "auth_mode": "username_password" if username else "none",
            "banner": banner,
        }, ensure_ascii=False, indent=2))
    except Exception as error:
        raise SystemExit(json.dumps({
            "status": "failed",
            "proxy": f"{args.proxy_host}:{args.proxy_port}",
            "destination": f"{args.dest_host}:{args.dest_port}",
            "auth_mode": "username_password" if username else "none",
            "error": str(error),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
