#!/usr/bin/env python3
"""SSH to HPC container via SOCKS5 proxy using paramiko with interactive shell.

All credentials are resolved from the environment via ``gpu_credentials`` —
never hardcoded. See .env.example for the supported variables:
  GPU_SSH_HOST / GPU_SSH_PORT / GPU_SSH_USER / GPU_SSH_PASSWORD[_FILE]
  GPU_SSH_KEY_PATH[_FILE]
  GPU_SSH_SOCKS_HOST / GPU_SSH_SOCKS_PORT / GPU_SSH_SOCKS_USER[_FILE] / GPU_SSH_SOCKS_PASSWORD[_FILE]
  HPC_LOGIN_HOST / HPC_LOGIN_PORT / HPC_LOGIN_USER / HPC_LOGIN_PASSWORD[_FILE]  (jump node)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko
import socks

# Resolve the shared credential helper from src/ without hardcoding secrets.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    CredentialError,
    _read_value,
)


def _require_env(name: str) -> str:
    value = _read_value(name)
    if not value:
        raise CredentialError(
            f"Missing required credential {name!r}. Set it in your environment or a .env file. "
            "Never hardcode it in source."
        )
    return value


def main():
    target_host = sys.argv[1] if len(sys.argv) > 1 else _require_env("GPU_SSH_HOST")
    target_port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("GPU_SSH_PORT", "22"))
    ssh_user = sys.argv[3] if len(sys.argv) > 3 else _require_env("GPU_SSH_USER")
    ssh_pass = sys.argv[4] if len(sys.argv) > 4 else _read_value("GPU_SSH_PASSWORD")

    socks_host = os.environ.get("GPU_SSH_SOCKS_HOST", "127.0.0.1")
    socks_port = int(os.environ.get("GPU_SSH_SOCKS_PORT", "7890"))

    # Connect via local SOCKS5 bridge
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, socks_host, socks_port)
    print(f"Connecting via SOCKS5 proxy to {target_host}:{target_port}...")
    try:
        sock.connect((target_host, target_port))
        print("Tunnel established!")
    except Exception as e:
        print(f"Direct SOCKS5 to {target_host} failed: {e}")
        print("Trying via HPC login node jump...")
        login_host = _require_env("HPC_LOGIN_HOST")
        login_port = int(os.environ.get("HPC_LOGIN_PORT", "22"))
        login_user = _require_env("HPC_LOGIN_USER")
        login_pass = _read_value("HPC_LOGIN_PASSWORD")

        sock2 = socks.socksocket()
        sock2.set_proxy(socks.SOCKS5, socks_host, socks_port)
        sock2.connect((login_host, login_port))

        # Auth to login node
        login_transport = paramiko.Transport(sock2)
        login_transport.connect(username=login_user, password=login_pass)

        # Open a direct-tcpip channel through login node to container
        dest_addr = (target_host, target_port)
        local_addr = ("127.0.0.1", 0)
        channel = login_transport.open_channel("direct-tcpip", dest_addr, local_addr, timeout=20)
        print(f"Jump channel to {target_host}:{target_port} opened!")

        # Now SSH through this channel
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client._transport = paramiko.Transport(channel)
        client._transport.connect(username=ssh_user, password=ssh_pass)
        print(f"SSH authenticated to container as {ssh_user}!")

        run_commands(client, login_transport)
        return

    # Direct SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("ignored", sock=sock, username=ssh_user, password=ssh_pass, timeout=20)
    print(f"SSH authenticated to {target_host} as {ssh_user}!")

    run_commands(client, None)


def run_commands(client: paramiko.SSHClient, jump_transport: paramiko.Transport | None):
    cmds = [
        "whoami",
        "hostname",
        "pwd",
        "ls -la /",
        "ls -la ~",
        "df -h",
        "cat /etc/os-release | head -5",
        "nvidia-smi 2>/dev/null | head -20 || echo 'No GPU'",
    ]
    for cmd in cmds:
        print(f"\n=== {cmd} ===")
        try:
            stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
            out = stdout.read().decode("utf-8", errors="replace").strip()
            err = stderr.read().decode("utf-8", errors="replace").strip()
            if out:
                print(out[:2000])
            if err:
                print(f"STDERR: {err[:500]}")
        except Exception as e:
            print(f"ERROR: {e}")

    client.close()
    if jump_transport:
        jump_transport.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
