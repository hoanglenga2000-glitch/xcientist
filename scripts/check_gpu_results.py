#!/usr/bin/env python3
"""GPU Training Monitor - connects via SOCKS5 proxy to SSHPiper gateway"""
import paramiko
import socks
import socket
import json
import os
import time

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7890
GATEWAY_HOST = "100.85.169.63"
GATEWAY_PORT = 1235

CONTAINERS = {
    "S7": {"user": "aimslab-zoeXIdNC", "password_env": "HPC_S7_PASSWORD"},
    "S1": {"user": "aimslab-IwkteXqP", "password_env": "HPC_S1_PASSWORD"},
}

def create_proxy_socket():
    s = socks.socksocket()
    s.set_proxy(socks.SOCKS5, PROXY_HOST, PROXY_PORT)
    s.settimeout(30)
    return s

def ssh_exec(container_name, command, timeout=30):
    """Execute command on a container via SSHPiper gateway"""
    creds = CONTAINERS[container_name]
    password = os.environ.get(creds["password_env"]) or os.environ.get("GPU_SSH_PASSWORD")
    if not password:
        raise RuntimeError(f"Missing SSH password env for {container_name}")
    sock = create_proxy_socket()
    sock.connect((GATEWAY_HOST, GATEWAY_PORT))

    transport = paramiko.Transport(sock)
    transport.connect(username=creds["user"], password=password)

    session = transport.open_session()
    session.exec_command(command)

    stdout = b""
    stderr = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session.recv_ready():
            stdout += session.recv(65536)
        if session.recv_stderr_ready():
            stderr += session.recv_stderr(65536)
        if session.exit_status_ready():
            break
        time.sleep(0.1)

    # Drain any remaining data
    while session.recv_ready():
        stdout += session.recv(65536)
    while session.recv_stderr_ready():
        stderr += session.recv_stderr(65536)

    session.close()
    transport.close()
    sock.close()

    return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

def check_container(name):
    """Run all checks on a container"""
    print(f"\n{'='*60}")
    print(f"  CHECKING {name}")
    print(f"{'='*60}")

    results = {}

    # 1. Check training processes
    stdout, stderr = ssh_exec(name, "ps aux | grep gpu_train_competition | grep -v grep")
    results["processes"] = stdout.strip()
    print(f"[PROCESSES] {'RUNNING' if stdout.strip() else 'NONE'}")
    if stdout.strip():
        print(stdout.strip())

    # 2. GPU utilization
    stdout, stderr = ssh_exec(name, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader")
    results["gpu_util"] = stdout.strip()
    print(f"[GPU UTIL] {stdout.strip()}")

    # 3. Check memory
    stdout, stderr = ssh_exec(name, "nvidia-smi --query-gpu=utilization.memory --format=csv,noheader")
    results["gpu_mem"] = stdout.strip()
    print(f"[GPU MEM]  {stdout.strip()}")

    return results

def check_batch_logs():
    """Check batch log files on S7"""
    print(f"\n{'='*60}")
    print(f"  BATCH LOGS (via S7)")
    print(f"{'='*60}")

    for logfile in ["/tmp/batch_S7_v2.log", "/tmp/batch_pending.log"]:
        stdout, stderr = ssh_exec("S7", f"tail -20 {logfile} 2>/dev/null || echo 'FILE_NOT_FOUND'")
        print(f"\n--- {logfile} ---")
        if "FILE_NOT_FOUND" in stdout:
            print("  [NOT FOUND]")
        else:
            print(stdout.strip())

def check_results():
    """Check results in shared storage"""
    print(f"\n{'='*60}")
    print(f"  SHARED STORAGE RESULTS")
    print(f"{'='*60}")

    # Count results
    stdout, stderr = ssh_exec("S7", "ls /hpc2hdd/home/aimslab/results/gpu_*.json 2>/dev/null | wc -l || echo 0")
    count = stdout.strip()
    print(f"[COUNT] {count} result files found")

    # List recent results
    stdout, stderr = ssh_exec("S7", "ls -lt /hpc2hdd/home/aimslab/results/gpu_*.json 2>/dev/null | head -10 || echo 'NONE'")
    print(f"\n[RECENT FILES]")
    print(stdout.strip() if stdout.strip() else "NONE")

    # Check latest result for OOF scores
    stdout, stderr = ssh_exec("S7", "ls -t /hpc2hdd/home/aimslab/results/gpu_*.json 2>/dev/null | head -1")
    latest_file = stdout.strip()
    if latest_file:
        print(f"\n[LATEST RESULT] {latest_file}")
        stdout, stderr = ssh_exec("S7", f"cat {latest_file} 2>/dev/null | head -100")
        content = stdout.strip()
        if content:
            # Parse JSON and extract OOF scores
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    for key in data:
                        if "oof" in key.lower() or "score" in key.lower() or "result" in key.lower():
                            print(f"  {key}: {data[key]}")
                    # Print top-level keys
                    print(f"  Keys: {list(data.keys())[:20]}")
                elif isinstance(data, list) and len(data) > 0:
                    print(f"  Array of {len(data)} items, first item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'scalar'}")
            except:
                print(content[:500])
        else:
            print("  [EMPTY OR UNREADABLE]")
    else:
        print("[NO RESULT FILES]")

    return count

def check_disk():
    """Check disk space"""
    print(f"\n{'='*60}")
    print(f"  DISK SPACE (S7)")
    print(f"{'='*60}")
    stdout, stderr = ssh_exec("S7", "df -h /hpc2hdd/home/aimslab/results/ 2>/dev/null || df -h /tmp/")
    print(stdout.strip())

def main():
    print("=" * 60)
    print("  GPU TRAINING MONITOR -", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # Check both containers
    s7_results = check_container("S7")
    s1_results = check_container("S1")

    # Check batch logs
    check_batch_logs()

    # Check results storage
    count = check_results()

    # Disk space
    check_disk()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    s7_running = bool(s7_results.get("processes"))
    s1_running = bool(s1_results.get("processes"))

    if s7_running:
        print("[S7] Training RUNNING")
        print(f"     GPU: {s7_results.get('gpu_util', 'N/A')}")
    else:
        print("[S7] IDLE - no training process")

    if s1_running:
        print("[S1] Training RUNNING")
        print(f"     GPU: {s1_results.get('gpu_util', 'N/A')}")
    else:
        print("[S1] IDLE - no training process")

    print(f"[RESULTS] {count} json files in shared storage")

if __name__ == "__main__":
    main()
