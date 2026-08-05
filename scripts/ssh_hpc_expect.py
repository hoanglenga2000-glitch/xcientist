#!/usr/bin/env python3
"""Interactive SSH to HPC via SOCKS5 proxy using pexpect."""
from __future__ import annotations
import sys
import subprocess
import time


def ssh_via_proxy(proxy_cmd: str, host: str, port: int, user: str, password: str, remote_cmd: str = "") -> str:
    """Run SSH via SOCKS5 proxy, auto-inputting password."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=20",
        "-o", f"ProxyCommand={proxy_cmd}",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        "-p", str(port),
        f"{user}@{host}",
    ]
    if remote_cmd:
        ssh_cmd.append(remote_cmd)

    print(f"Running: {' '.join(ssh_cmd)}")

    # Use Python to drive the interactive SSH
    import pty
    import os
    import select

    pid, fd = pty.fork()
    if pid == 0:
        # Child - exec ssh
        os.execvp("ssh", ssh_cmd)
    else:
        # Parent - interact with child
        output = b""
        password_sent = False
        start = time.time()
        while time.time() - start < 60:
            try:
                r, _, _ = select.select([fd], [], [], 5)
                if r:
                    data = os.read(fd, 4096)
                    output += data
                    decoded = data.decode("utf-8", errors="replace")

                    if "password" in decoded.lower() and not password_sent:
                        time.sleep(0.5)
                        os.write(fd, (password + "\n").encode())
                        password_sent = True

                    if password_sent and ("$" in decoded or "#" in decoded or "welcome" in decoded.lower()):
                        # Got a shell or welcome
                        time.sleep(1)
                        # Send exit or command
                        if remote_cmd:
                            # Wait for command output
                            time.sleep(2)
                            r2, _, _ = select.select([fd], [], [], 10)
                            if r2:
                                output += os.read(fd, 4096)
                        os.write(fd, b"exit\n")
                        time.sleep(1)
                        break
            except OSError:
                break

        try:
            os.waitpid(pid, 0)
        except:
            pass

        return output.decode("utf-8", errors="replace")


def ssh_paramiko_jump(proxy_host: str, proxy_port: int, proxy_user: str, proxy_pass: str,
                       dest_host: str, dest_port: int, dest_user: str, dest_pass: str,
                       commands: list[str]) -> dict[str, str]:
    """SSH via jump host using paramiko."""
    import socks
    import socket
    import paramiko

    results = {}

    # Step 1: Connect to jump host via SOCKS5
    print(f"Step 1: Connecting to jump host {proxy_host}:{proxy_port} via SOCKS5...")
    jump_sock = socks.socksocket()
    jump_sock.set_proxy(socks.SOCKS5, "127.0.0.1", 7890)
    jump_sock.connect((proxy_host, proxy_port))
    print("  SOCKS5 tunnel established!")

    # Step 2: SSH to jump host
    jump_client = paramiko.SSHClient()
    jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Use Transport for more control
    jump_transport = paramiko.Transport(jump_sock)
    jump_transport.connect(username=proxy_user, password=proxy_pass)
    print(f"  SSH authenticated to jump host as {proxy_user}!")

    # Step 3: Port forward through jump host to destination
    print(f"Step 3: Opening direct-tcpip channel to {dest_host}:{dest_port}...")
    try:
        dest_channel = jump_transport.open_channel(
            "direct-tcpip",
            (dest_host, dest_port),
            ("127.0.0.1", 0),
            timeout=20
        )
        print("  Channel opened!")

        # Step 4: SSH through the channel to destination
        dest_transport = paramiko.Transport(dest_channel)
        dest_transport.connect(username=dest_user, password=dest_pass)
        print(f"  SSH authenticated to container as {dest_user}!")

        dest_client = paramiko.SSHClient()
        dest_client._transport = dest_transport

        for cmd in commands:
            print(f"\n=== {cmd} ===")
            try:
                session = dest_transport.open_session()
                session.exec_command(cmd)
                session.settimeout(30)
                out = b""
                while True:
                    try:
                        chunk = session.recv(4096)
                        if not chunk:
                            break
                        out += chunk
                    except:
                        break
                result = out.decode("utf-8", errors="replace").strip()
                print(result[:2000])
                results[cmd] = result
            except Exception as e:
                print(f"  ERROR: {e}")
                results[cmd] = f"ERROR: {e}"

        dest_transport.close()
    except Exception as e:
        print(f"  Port forward/channel failed: {e}")
        print("  Trying SSH from jump host directly...")

        # Fallback: execute ssh command on jump host
        ssh_cmd = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {dest_user}@{dest_host} -p {dest_port}'
        try:
            session = jump_transport.open_session()
            session.exec_command(ssh_cmd)
            session.settimeout(15)
            out = b""
            while True:
                try:
                    chunk = session.recv(4096)
                    if not chunk: break
                    out += chunk
                except: break
            print(f"  SSH output: {out.decode('utf-8', errors='replace')[:500]}")
        except Exception as e2:
            print(f"  Direct SSH also failed: {e2}")

    jump_transport.close()
    return results


if __name__ == "__main__":
    # Try paramiko jump approach
    results = ssh_paramiko_jump(
        proxy_host="100.85.169.63",
        proxy_port=1235,
        proxy_user="haowei",
        proxy_pass="aimslab@260612",
        dest_host="10.120.18.240",
        dest_port=6988,
        dest_user="aimslab-wqx-SDBS",
        dest_pass="zxdFSYnX6o",
        commands=[
            "whoami",
            "hostname",
            "pwd",
            "ls -la /",
            "find / -maxdepth 4 -name '*jhw*' -type d 2>/dev/null",
            "find / -maxdepth 4 -name '*景浩伟*' -type d 2>/dev/null",
            "df -h",
            "nvidia-smi 2>/dev/null | head -20 || echo 'No GPU'",
        ]
    )
