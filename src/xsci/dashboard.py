"""Bridge xsci to the existing Next.js workstation dashboard."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT_POINTER = Path.home() / ".xsci" / "workstation-root.txt"


def _is_workstation_root(candidate: Path) -> bool:
    return (
        (candidate / "scripts" / "manage_workstation_dashboard.py").is_file()
        and (candidate / "web" / "research-agent-workstation" / "package.json").is_file()
    )


def resolve_workstation_root() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("EVOMIND_WORKSTATION_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    try:
        pointer = ROOT_POINTER.read_text(encoding="utf-8-sig").strip()
    except OSError:
        pointer = ""
    if pointer:
        candidates.append(Path(pointer).expanduser())
    candidates.append(SOURCE_ROOT)

    for candidate in candidates:
        resolved = candidate.resolve()
        if _is_workstation_root(resolved):
            return resolved
    return None


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
    root = resolve_workstation_root()
    if root is None:
        print(
            "dashboard requires the full EvoMind workstation source bundle; "
            "run install.ps1 from that bundle or set EVOMIND_WORKSTATION_ROOT"
        )
        return 1
    manager = root / "scripts" / "manage_workstation_dashboard.py"

    args = [
        sys.executable,
        str(manager),
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

    proc = subprocess.run(args, cwd=root, text=True, encoding="utf-8", errors="replace")
    return int(proc.returncode)
