#!/usr/bin/env python3
"""
GPU Process Monitor for A800 Optuna Sweep
Connects via SSH (SOCKS5 proxy) to check the status of the Optuna sweep process.
"""

import argparse
import datetime
import json
import os
import posixpath
import re


def _required_env(name, *, preserve=False):
    value = os.environ.get(name)
    if value is None or (value == "" if preserve else value.strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value if preserve else value.strip()


def _env_port(name, *, required):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        if required:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return None
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid port in environment variable: {name}") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"Invalid port in environment variable: {name}")
    return port


def _remote_workspace():
    value = _required_env("EVOMIND_HPC_REMOTE_WORKSPACE").rstrip("/")
    segments = [segment for segment in value.split("/") if segment]
    if (
        not re.fullmatch(r"/[A-Za-z0-9._/-]+", value)
        or not segments
        or any(segment in {".", ".."} for segment in segments)
    ):
        raise RuntimeError("Invalid remote workspace environment variable")
    return value



# ── Configuration ──────────────────────────────────────────────────────────────

SSH_HOST = _required_env("EVOMIND_HPC_HOST")
SSH_PORT = _env_port("EVOMIND_HPC_PORT", required=True)
SSH_USER = _required_env("EVOMIND_HPC_USER")
SSH_PASSWORD = _required_env("EVOMIND_HPC_PASSWORD", preserve=True)
REMOTE_WORKSPACE = _remote_workspace()
SOCKS5_HOST = os.environ.get("EVOMIND_HPC_SOCKS_HOST", "").strip()
SOCKS5_PORT = _env_port("EVOMIND_HPC_SOCKS_PORT", required=False)
if bool(SOCKS5_HOST) != (SOCKS5_PORT is not None):
    raise RuntimeError(
        "EVOMIND_HPC_SOCKS_HOST and EVOMIND_HPC_SOCKS_PORT must be set together"
    )

REMOTE_PARENT = posixpath.dirname(REMOTE_WORKSPACE)
WORK_DIR = posixpath.join(REMOTE_PARENT, "spaceship_titanic")
SWEEP_SCRIPT = posixpath.join(WORK_DIR, "sweep_5fold.py")
LOG_FILE = f"{WORK_DIR}/sweep_output.log"

# Rerun command (used if process crashed)
RERUN_CMD = (
    f"{posixpath.join(REMOTE_WORKSPACE, 'pyenvs', 's6e6_boosting', 'bin', 'python')} "
    f"{SWEEP_SCRIPT} > {LOG_FILE} 2>&1 &"
)


def get_password():
    """Return the credential injected by the verified workstation launcher."""
    return SSH_PASSWORD


def ssh_connect(host, port, user, password, socks_host, socks_port):
    """Establish SSH directly or through the configured SOCKS5 proxy."""
    from hpc_connect import secure_ssh_client

    connection_socket = None
    if socks_host:
        import socks

        connection_socket = socks.socksocket()
        connection_socket.set_proxy(socks.SOCKS5, socks_host, socks_port)
        connection_socket.settimeout(30)
        connection_socket.connect((host, port))

    ssh = secure_ssh_client()
    try:
        ssh.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            sock=connection_socket,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )
    except BaseException:
        ssh.close()
        if connection_socket is not None:
            connection_socket.close()
        raise

    return ssh


def run_remote(ssh, command, timeout=30):
    """Execute a command on the remote host and return stdout."""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err


def check_process(ssh):
    """Check if the sweep_5fold process is running. Returns (is_running, pid_list)."""
    out, _ = run_remote(ssh, "ps aux | grep sweep_5fold | grep -v grep")
    if not out:
        return False, []
    pids = []
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            pids.append(parts[1])  # PID is 2nd column
    return len(pids) > 0, pids


def get_tail_log(ssh, lines=20):
    """Get the last N lines of the log file."""
    out, _ = run_remote(ssh, f"tail -{lines} {LOG_FILE} 2>/dev/null || echo '[LOG_NOT_FOUND]'")
    return out


def list_work_dir(ssh):
    """List files in the work directory sorted by modification time."""
    out, _ = run_remote(ssh, f"ls -lt {WORK_DIR}/ 2>/dev/null || echo '[DIR_NOT_FOUND]'")
    return out


def read_results_json(ssh):
    """Look for results JSON files and read the best score."""
    # Find JSON files
    out, _ = run_remote(ssh, f"find {WORK_DIR}/ -maxdepth 1 -name '*.json' -type f 2>/dev/null")
    results = {}
    if out:
        json_files = [f.strip() for f in out.split("\n") if f.strip()]
        for jf in json_files:
            filename = os.path.basename(jf)
            content, _ = run_remote(ssh, f"cat {jf} 2>/dev/null")
            if content:
                try:
                    data = json.loads(content)
                    results[filename] = data
                except json.JSONDecodeError:
                    results[filename] = {"error": "invalid_json", "preview": content[:200]}
    return results


def extract_best_score(results):
    """Extract the best 5-fold CV score from results JSON data."""
    best_score = None
    best_source = None
    for filename, data in results.items():
        # Try common Optuna output patterns
        if isinstance(data, dict):
            # Pattern 1: data["best_trial"]["value"] or data["best_value"]
            if "best_value" in data:
                score = data["best_value"]
                if isinstance(score, (int, float)):
                    if best_score is None or score > best_score:
                        best_score = score
                        best_source = filename
            # Pattern 2: data["best_score"]
            if "best_score" in data:
                score = data["best_score"]
                if isinstance(score, (int, float)):
                    if best_score is None or score > best_score:
                        best_score = score
                        best_source = filename
            # Pattern 3: data["best_params"] + data["best_accuracy"]
            for key in ["best_accuracy", "best_result", "score", "accuracy", "cv_score"]:
                if key in data:
                    score = data[key]
                    if isinstance(score, (int, float)):
                        if best_score is None or score > best_score:
                            best_score = score
                            best_source = filename
            # Pattern 4: nested in a "study" or "optuna" key
            for key in ["study", "optuna", "result"]:
                if key in data and isinstance(data[key], dict):
                    sub = data[key]
                    for skey in ["best_value", "best_score", "best_accuracy"]:
                        if skey in sub:
                            score = sub[skey]
                            if isinstance(score, (int, float)):
                                if best_score is None or score > best_score:
                                    best_score = score
                                    best_source = f"{filename}.{key}.{skey}"
        elif isinstance(data, list) and len(data) > 0:
            # Pattern: list of trial results
            if isinstance(data[0], dict):
                values = [d.get("value") for d in data if isinstance(d.get("value"), (int, float))]
                if values:
                    max_val = max(values)
                    if best_score is None or max_val > best_score:
                        best_score = max_val
                        best_source = f"{filename}[best_trial]"

    return best_score, best_source


def monitor():
    """Main monitoring routine."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'='*70}")
    print(f"GPU MONITOR REPORT — {timestamp}")
    print(f"{'='*70}")
    route = (
        f"SOCKS5:{SOCKS5_HOST}:{SOCKS5_PORT}" if SOCKS5_HOST else "direct"
    )
    print(f"Target: {SSH_USER}@{SSH_HOST}:{SSH_PORT} ({route})")
    print(f"Script: {SWEEP_SCRIPT}")
    print()

    # 1. Connect
    print("[1/6] Loading verified HPC credentials from the environment...")
    password = get_password()
    print("      Credentials loaded successfully.")

    print(f"[2/6] Connecting via {route}...")
    ssh = ssh_connect(SSH_HOST, SSH_PORT, SSH_USER, password, SOCKS5_HOST, SOCKS5_PORT)
    print("      Connected.")

    # 3. Check process
    print("[3/6] Checking sweep_5fold process...")
    is_running, pids = check_process(ssh)
    if is_running:
        print(f"      RUNNING  (PIDs: {', '.join(pids)})")
    else:
        print("      STOPPED  (no matching process found)")

    # 4. Tail log
    print(f"[4/6] Reading log tail: {LOG_FILE}")
    log_tail = get_tail_log(ssh)
    print("      Log content (last 20 lines):")
    for line in log_tail.split("\n"):
        print(f"      | {line}")
    if not log_tail or log_tail == "[LOG_NOT_FOUND]":
        print("      WARNING: Log file not found or empty!")

    # 5. List work directory
    print(f"[5/6] Listing work directory: {WORK_DIR}")
    dir_listing = list_work_dir(ssh)
    print("      Files (sorted by modification time):")
    for line in dir_listing.split("\n")[:30]:  # limit to 30 entries
        print(f"      {line}")

    # 6. Read results
    print("[6/6] Searching for results JSON files...")
    results = read_results_json(ssh)
    if results:
        print(f"      Found {len(results)} JSON file(s):")
        for filename, data in results.items():
            if isinstance(data, dict) and "error" not in data:
                # Print a summary of keys
                keys = list(data.keys()) if isinstance(data, dict) else "list"
                print(f"        - {filename}: keys={keys[:10]}")
            else:
                print(f"        - {filename}: {data.get('error', 'unknown')}")
        best_score, best_source = extract_best_score(results)
        if best_score is not None:
            print(f"\n      *** BEST 5-FOLD CV SCORE: {best_score:.6f}  (from {best_source}) ***")
        else:
            print("\n      No numeric score found in results files.")
    else:
        print("      No JSON result files found.")

    ssh.close()

    # Final status
    print()
    print(f"{'='*70}")
    if is_running:
        print("FINAL STATUS: RUNNING")
    elif results:
        print("FINAL STATUS: COMPLETED  (results found, process exited)")
    elif log_tail and log_tail != "[LOG_NOT_FOUND]":
        last_line = log_tail.strip().split("\n")[-1].lower()
        if "error" in last_line or "traceback" in last_line or "exception" in last_line:
            print("FINAL STATUS: CRASHED  (error detected in log)")
        else:
            print("FINAL STATUS: CRASHED/STOPPED  (no process, no results, check log)")
    else:
        print("FINAL STATUS: CRASHED/UNKNOWN  (no process, no results, no log)")

    print(f"{'='*70}")

    return is_running, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Process Monitor for A800 Optuna Sweep")
    parser.add_argument("--interval", type=int, default=0,
                        help="Repeat every N seconds (0 = run once)")
    parser.add_argument("--rerun-if-crashed", action="store_true",
                        help="Automatically re-run the sweep script if crashed")
    args = parser.parse_args()

    import time
    while True:
        try:
            is_running, results = monitor()
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] MONITOR ERROR: {e}")
            import traceback
            traceback.print_exc()
            is_running = False
            results = {}

        if not is_running and not results and args.rerun_if_crashed:
            print()
            print("CRASHED — attempting automatic re-run...")
            try:
                password = get_password()
                ssh = ssh_connect(SSH_HOST, SSH_PORT, SSH_USER, password, SOCKS5_HOST, SOCKS5_PORT)
                run_remote(ssh, RERUN_CMD)
                print(f"Re-run command sent: {RERUN_CMD}")
                ssh.close()
            except Exception as e:
                print(f"FAILED to re-run: {e}")

        if args.interval <= 0:
            break
        print(f"\nWaiting {args.interval}s until next check...\n")
        time.sleep(args.interval)
