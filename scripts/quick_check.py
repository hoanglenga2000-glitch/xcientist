#!/usr/bin/env python3
"""Quick check: results storage + disk via S7"""
import paramiko, socks, socket, json, time

def create_proxy_socket():
    s = socks.socksocket()
    s.set_proxy(socks.SOCKS5, "127.0.0.1", 7890)
    s.settimeout(30)
    return s

def ssh_exec(cmd, timeout=30):
    sock = create_proxy_socket()
    sock.connect(("100.85.169.63", 1235))
    transport = paramiko.Transport(sock)
    transport.connect(username="aimslab-zoeXIdNC", password="n6oewebu0p")
    session = transport.open_session()
    session.exec_command(cmd)
    stdout = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session.recv_ready():
            stdout += session.recv(65536)
        if session.exit_status_ready():
            break
        time.sleep(0.1)
    while session.recv_ready():
        stdout += session.recv(65536)
    session.close(); transport.close(); sock.close()
    return stdout.decode("utf-8", errors="replace")

# Count results
cnt = ssh_exec("ls /hpc2hdd/home/aimslab/results/gpu_*.json 2>/dev/null | wc -l").strip()
print(f"RESULT COUNT: {cnt}")

# List recent
recent = ssh_exec("ls -lt /hpc2hdd/home/aimslab/results/gpu_*.json 2>/dev/null | head -10").strip()
print(f"\nRECENT FILES:\n{recent if recent else 'NONE'}")

# Latest result content
latest = ssh_exec("ls -t /hpc2hdd/home/aimslab/results/gpu_*.json 2>/dev/null | head -1").strip()
if latest:
    print(f"\nLATEST: {latest}")
    content = ssh_exec(f"cat {latest}").strip()
    print(content[:1500])
else:
    print("\nNO RESULTS FILES")

# Check /tmp for any results
tmp_files = ssh_exec("ls -la /tmp/gpu_*.json 2>/dev/null; ls -la /tmp/results*.json 2>/dev/null; echo '--'").strip()
print(f"\nTMP FILES:\n{tmp_files}")

# Disk
disk = ssh_exec("df -h /hpc2hdd/ 2>/dev/null; echo '---'; df -h /tmp/").strip()
print(f"\nDISK:\n{disk}")
