"""Read-only GPU probe for an alternate DPAPI-backed HPC profile.

The decrypted password is retained only in process memory and is never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    ALLOWED_GPU_REMOTE_ROOT,
    GpuSshConfig,
    SocksConfig,
    connect_ssh,
)


def _load_profile(profile_dir: Path) -> GpuSshConfig:
    credential_path = profile_dir / "hpc_ssh_credential.xml"
    metadata_path = profile_dir / "hpc_ssh_metadata.json"
    if not credential_path.is_file():
        credential_path = profile_dir / "credential.xml"
    if not metadata_path.is_file():
        metadata_path = profile_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    remote_root = str(metadata.get("remote_workspace") or "").rstrip("/")
    if remote_root != ALLOWED_GPU_REMOTE_ROOT:
        raise RuntimeError("Profile remote workspace is outside the allowed root")

    env = dict(os.environ)
    env["EVOMIND_HPC_CREDENTIAL_PATH"] = str(credential_path)
    script = (
        "$ErrorActionPreference='Stop';"
        "$c=Import-Clixml -LiteralPath $env:EVOMIND_HPC_CREDENTIAL_PATH;"
        "@{user=$c.UserName;password=$c.GetNetworkCredential().Password}"
        "|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=15,
    )
    secret = json.loads(completed.stdout.lstrip("\ufeff").strip())
    socks_host = str(metadata.get("socks_host") or "")
    socks = (
        SocksConfig(host=socks_host, port=int(metadata.get("socks_port") or 7890))
        if socks_host
        else None
    )
    return GpuSshConfig(
        host=str(metadata.get("host") or ""),
        port=int(metadata.get("port") or 0),
        username=str(secret.get("user") or ""),
        password=str(secret.get("password") or ""),
        socks=socks,
        jump_host=str(metadata.get("jump_host") or "") or None,
        jump_port=int(metadata.get("jump_port") or 22),
        jump_username=str(metadata.get("jump_user") or "") or None,
    )


def _run(client, command: str) -> dict[str, object]:
    _stdin, stdout, stderr = client.exec_command(command, timeout=30)
    out = stdout.read().decode("utf-8", "replace").strip()
    err = stderr.read().decode("utf-8", "replace").strip()
    return {
        "exit_code": stdout.channel.recv_exit_status(),
        "stdout": out,
        "stderr_present": bool(err),
        "stderr_type": err.splitlines()[-1] if err else "",
    }


def _parse_gpu(line: str) -> dict[str, object]:
    parts = [item.strip() for item in line.split(",")]
    return {
        "gpu_index": int(parts[0]),
        "gpu_name": parts[1],
        "memory_total_mib": int(float(parts[2])),
        "memory_used_mib": int(float(parts[3])),
        "utilization_percent": int(float(parts[4])),
        "temperature_c": int(float(parts[5])),
        "power_draw_w": float(parts[6]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    config = _load_profile(args.profile_dir.resolve())
    client = connect_ssh(config)
    root = ALLOWED_GPU_REMOTE_ROOT
    try:
        preflight = _run(
            client,
            f"printf 'USER=%s\\nHOST=%s\\n' \"$(id -un)\" \"$(hostname)\"; "
            f"test -d {root}; printf 'ROOT_EXISTS=%s\\n' $?; "
            f"test -w {root}; printf 'ROOT_WRITABLE=%s\\n' $?",
        )
        samples: list[dict[str, object]] = []
        for index in range(args.samples):
            gpu = _run(
                client,
                "nvidia-smi --query-gpu=index,name,memory.total,memory.used,"
                "utilization.gpu,temperature.gpu,power.draw "
                "--format=csv,noheader,nounits",
            )
            apps = _run(
                client,
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                "--format=csv,noheader,nounits",
            )
            app_lines = [line for line in str(apps["stdout"]).splitlines() if line.strip()]
            sample = _parse_gpu(str(gpu["stdout"]).splitlines()[0])
            sample.update(
                {
                    "sample": index + 1,
                    "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "exit_code": gpu["exit_code"],
                    "compute_apps": app_lines,
                    "compute_process_count": len(app_lines),
                    "stderr_present": bool(gpu["stderr_present"] or apps["stderr_present"]),
                }
            )
            samples.append(sample)
            if index + 1 < args.samples:
                time.sleep(args.interval_seconds)
        process_inspection = _run(
            client,
            "echo '---IDENTITY---'; id -un; hostname; "
            "echo '---DEDICATED_ROOT_PROCESSES---'; "
            f"ps -eo user:32,pid,ppid,etimes,pcpu,pmem,rss,args --sort=-pcpu | grep -F '{root}' | grep -v grep || true; "
            "echo '---VISIBLE_COMPUTE_PIDS---'; "
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits",
        )
    finally:
        client.close()

    now = datetime.now(timezone.utc).astimezone()
    policy = {
        "samples_required": 3,
        "max_memory_used_mib": 1024,
        "max_utilization_percent": 5,
        "compute_apps_required_empty": True,
    }
    result = {
        "schema": "evomind.hpc.alternate_gpu_live_probe.v1",
        "created_at": now.isoformat(),
        "remote_root": root,
        "preflight": preflight,
        "samples": samples,
        "process_inspection": process_inspection,
        "idle_policy": policy,
    }
    root_ready = bool(
        preflight["exit_code"] == 0
        and "ROOT_EXISTS=0" in str(preflight["stdout"])
        and "ROOT_WRITABLE=0" in str(preflight["stdout"])
    )
    result["dedicated_root_ready"] = root_ready
    result["idle_gate_passed"] = (
        root_ready
        and len(samples) >= policy["samples_required"]
        and all(
            int(sample["memory_used_mib"]) <= policy["max_memory_used_mib"]
            and int(sample["utilization_percent"]) <= policy["max_utilization_percent"]
            and int(sample["compute_process_count"]) == 0
            for sample in samples
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence": str(args.output.resolve()),
                "idle_gate_passed": result["idle_gate_passed"],
                "samples": samples,
                "process_inspection": process_inspection["stdout"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
