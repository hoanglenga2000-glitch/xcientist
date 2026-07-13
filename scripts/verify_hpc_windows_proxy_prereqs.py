from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def command_info(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"found": False, "path": None, "version": None}
    version = None
    try:
        completed = subprocess.run([path, "--version"], text=True, capture_output=True, timeout=10)
        version = (completed.stdout or completed.stderr).strip().splitlines()[0] if (completed.stdout or completed.stderr).strip() else None
    except Exception:
        version = None
    return {"found": True, "path": path, "version": version}


def run_bridge_status() -> dict[str, Any]:
    manager = ROOT / "scripts" / "manage_hpc_proxy_bridge.ps1"
    completed = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(manager), "status"],
        text=True,
        capture_output=True,
        timeout=20,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"raw_stdout": completed.stdout.strip(), "raw_stderr": completed.stderr.strip()}
    payload["returncode"] = completed.returncode
    return payload


def main() -> None:
    report = {
        "status": "passed",
        "winget": command_info("winget.exe") if not shutil.which("winget") else command_info("winget"),
        "ncat": command_info("ncat.exe") if not shutil.which("ncat") else command_info("ncat"),
        "hpc_7890_bridge": run_bridge_status(),
        "ncat_command_template": 'ssh -o ProxyCommand="ncat --proxy 127.0.0.1:7890 --proxy-type socks5 %h %p" ${EVOMIND_HPC_USER}@${EVOMIND_HPC_HOST} -p ${EVOMIND_HPC_PORT}',
        "python_proxy_command_template": 'ssh -o ProxyCommand="python scripts/hpc_socks_proxy.py 127.0.0.1 7890 %h %p" ${EVOMIND_HPC_USER}@${EVOMIND_HPC_HOST} -p ${EVOMIND_HPC_PORT}',
    }
    if not report["ncat"]["found"]:
        report["status"] = "ncat_missing_bridge_ready" if report["hpc_7890_bridge"].get("status") == "running" else "ncat_missing_bridge_not_running"
        report["next_action"] = "Install Nmap/Ncat in an elevated Windows terminal if strict PDF ncat verification is required."
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
