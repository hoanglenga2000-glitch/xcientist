"""Bridge xsci to the existing Next.js workstation dashboard."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANAGER = ROOT / "scripts" / "manage_workstation_dashboard.py"


def run_dashboard(
    command: str,
    *,
    port: int = 8088,
    timeout: float = 45.0,
    build: bool = False,
    force: bool = False,
) -> int:
    """Start/stop/status the real dashboard without creating a second UI stack."""
    if command not in {"start", "stop", "restart", "status"}:
        print("usage: xsci dashboard {start|stop|restart|status}")
        return 2
    if not MANAGER.exists():
        print(f"dashboard manager not found: {MANAGER}")
        return 1

    args = [
        sys.executable,
        str(MANAGER),
        command,
        "--port",
        str(port),
        "--timeout",
        str(timeout),
    ]
    if build:
        args.append("--build")
    if force:
        args.append("--force")

    proc = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace")
    return int(proc.returncode)
