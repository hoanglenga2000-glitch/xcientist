#!/usr/bin/env python3
"""
GPU Dual Monitor - monitors Optuna sweeps on GPU servers.
Connects to A800 login node via SOCKS5 proxy.
Server1 (A800) = login node itself. Server2 (A40) = inner Docker container.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime

import socks
import paramiko

# ── Configuration ──────────────────────────────────────────────────────────
LOGIN_HOST = "100.85.169.63"
LOGIN_PORT = 1235
LOGIN_USER = "aimslab-IwkteXqP"
SOCKS5_HOST = "127.0.0.1"
SOCKS5_PORT = 7890

# Server1 = login node (A800), commands run directly
# Server2 = A40 inner container (currently inaccessible)
SERVERS = [
    {
        "name": "server1-A800",
        "type": "direct",  # Run commands directly on login node
        "model": "XGBoost (running) + CatBoost/LightGBM (done)",
        "sweeps": [
            {
                "model": "CatBoost",
                "status": "completed",
                "result_file": "/hpc2hdd/home/aimslab/spaceship_titanic/result_5fold.json",
            },
            {
                "model": "LightGBM",
                "status": "completed",
                "result_file": "/hpc2hdd/home/aimslab/spaceship_titanic/result_lgb.json",
            },
            {
                "model": "XGBoost",
                "status": "running",
                "script_match": "_xgb_sweep",
                "log_file": "/hpc2hdd/home/aimslab/optuna_sweeps/server1-A800-xgboost/sweep.log",
                "out_dir": "/hpc2hdd/home/aimslab/optuna_sweeps/server1-A800-xgboost",
            },
        ],
    },
    {
        "name": "server2-A40",
        "type": "inner",  # SSH from login node to inner container
        "inner_host": "10.120.18.240",
        "inner_port": 6988,
        "inner_user": "aimslab-kdd-ai4s",
        "model": "CatBoost (inaccessible)",
        "sweeps": [],
        "inaccessible": True,
    },
]


def get_password() -> str:
    ps_script = r'$cred = Import-Clixml -Path "$env:APPDATA\ResearchAgentWorkstation\hpc_ssh_credential.xml"; Write-Output $cred.GetNetworkCredential().Password'
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DPAPI failed: {result.stderr}")
    return result.stdout.strip()


def ssh_to_login(password: str) -> paramiko.SSHClient:
    sock = socks.socksocket()
    sock.set_proxy(socks.SOCKS5, SOCKS5_HOST, SOCKS5_PORT)
    sock.settimeout(30)
    sock.connect((LOGIN_HOST, LOGIN_PORT))
    transport = paramiko.Transport(sock)
    transport.connect(username=LOGIN_USER, password=password)
    ssh = paramiko.SSHClient()
    ssh._transport = transport
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return ssh


def exec_remote(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> str:
    """Execute command on remote, return stdout."""
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8", errors="replace")


def check_server1(ssh: paramiko.SSHClient) -> dict:
    """Check A800 directly (login node)."""
    status = {
        "name": "server1-A800",
        "gpu": "NVIDIA A800-SXM4-80GB",
        "timestamp": datetime.now().isoformat(),
        "sweeps": [],
    }

    # GPU
    gpu = exec_remote(ssh, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null")
    status["gpu_info"] = gpu.strip()

    # Running processes
    procs = exec_remote(ssh, "ps aux | grep -E '_xgb_sweep|sweep_lgb|sweep_cb|sweep_5fold' | grep python | grep -v grep")
    status["running_processes"] = procs.strip()

    # Check each sweep
    for sweep_cfg in SERVERS[0]["sweeps"]:
        sweep_status = {"model": sweep_cfg["model"], "status": sweep_cfg["status"]}

        if sweep_cfg["status"] == "completed":
            result = exec_remote(ssh, f"cat {sweep_cfg['result_file']} 2>/dev/null")
            if result.strip():
                try:
                    r = json.loads(result.strip())
                    sweep_status["best_score"] = r.get("best_score") or r.get("best_5fold_cv") or r.get("best")
                    sweep_status["params"] = r.get("params") or r.get("best_params")
                except json.JSONDecodeError:
                    sweep_status["raw"] = result.strip()[:200]

        elif sweep_cfg["status"] == "running":
            # Check log
            log = exec_remote(ssh, f"tail -10 {sweep_cfg['log_file']} 2>/dev/null")
            if log.strip():
                # Extract best value from last lines
                for line in log.strip().split("\n"):
                    if "Best value:" in line:
                        sweep_status["latest_best"] = line.strip()[-80:]
                    if "Best trial:" in line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            sweep_status["progress"] = f"{parts[1]} trials, best={parts[4]}"

            # Check for completed results
            results = exec_remote(ssh, f"cat {sweep_cfg['out_dir']}/best_params.json 2>/dev/null")
            if results.strip():
                sweep_status["status"] = "completed"
                try:
                    sweep_status["best_params"] = json.loads(results.strip())
                except json.JSONDecodeError:
                    pass

            # Check process
            proc_check = exec_remote(ssh, f"ps aux | grep {sweep_cfg['script_match']} | grep python | grep -v grep")
            if not proc_check.strip():
                # Maybe finished
                results_file = exec_remote(ssh, f"ls -t {sweep_cfg['out_dir']}/results_*.json 2>/dev/null | head -1")
                if results_file.strip():
                    sweep_status["status"] = "completed"
                    sweep_status["result_file"] = results_file.strip()

        status["sweeps"].append(sweep_status)

    return status


def check_server2(ssh: paramiko.SSHClient) -> dict:
    """Check A40 status (inaccessible)."""
    return {
        "name": "server2-A40",
        "gpu": "NVIDIA A40 48GB (expected)",
        "timestamp": datetime.now().isoformat(),
        "status": "INACCESSIBLE",
        "reason": "aimslab-kdd-ai4s authentication incomplete - container may be stopped",
        "sweeps": [
            {"model": "CatBoost", "status": "not deployed - server unreachable"}
        ],
    }


def print_status(ssh: paramiko.SSHClient, statuses: list[dict]):
    """Pretty print status of all servers."""
    print("\n" + "=" * 70)
    print(f"  GPU DUAL MONITOR  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for s in statuses:
        print(f"\n  [{s['name']}]  GPU: {s.get('gpu', 'N/A')}")
        print(f"  {'-' * 50}")

        if s.get("inaccessible"):
            print(f"    STATUS: INACCESSIBLE - {s.get('reason', 'unknown')}")
            continue

        if s.get("gpu_info"):
            parts = s["gpu_info"].split(",")
            if len(parts) >= 3:
                print(f"    GPU: {parts[0].strip()}% util, {parts[1].strip()}/{parts[2].strip()} MiB")

        if s.get("running_processes"):
            print(f"    Running Python jobs:")
            for line in s["running_processes"].split("\n"):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 11:
                        print(f"      PID:{parts[1]} CPU:{parts[2]}% MEM:{parts[3]}%")

        for sweep in s.get("sweeps", []):
            print(f"\n    [{sweep['model']}] Status: {sweep['status'].upper()}")
            if sweep.get("best_score"):
                print(f"      Best CV Score: {sweep['best_score']:.6f}")
            if sweep.get("params") and isinstance(sweep["params"], dict):
                print(f"      Best Params: {json.dumps(sweep['params'], indent=2)[:200]}")
            if sweep.get("progress"):
                print(f"      Progress: {sweep['progress']}")
            if sweep.get("latest_best"):
                print(f"      Latest: {sweep['latest_best']}")
            if sweep.get("raw"):
                print(f"      Raw: {sweep['raw']}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--loop", "-l"):
        loop_mode = True
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    else:
        loop_mode = False
        interval = 60

    if loop_mode:
        print(f"Continuous monitor (interval={interval}s). Ctrl+C to stop.")
        try:
            password = get_password()
            while True:
                ssh = ssh_to_login(password)
                try:
                    statuses = [check_server1(ssh), check_server2(ssh)]
                    print_status(ssh, statuses)
                finally:
                    ssh.close()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        password = get_password()
        ssh = ssh_to_login(password)
        try:
            statuses = [check_server1(ssh), check_server2(ssh)]
            print_status(ssh, statuses)
        finally:
            ssh.close()


if __name__ == "__main__":
    main()
