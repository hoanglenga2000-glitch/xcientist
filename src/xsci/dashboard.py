"""Bridge xsci to the existing Next.js workstation dashboard."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT_POINTER = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "EvoMind" / "workstation-root.txt"


def _is_full_workstation_root(candidate: Path) -> bool:
    return (
        (candidate / "scripts" / "manage_workstation_dashboard.py").is_file()
        and (candidate / "web" / "research-agent-workstation" / "package.json").is_file()
    )


def resolve_workstation_root() -> Path:
    candidates: list[Path] = []
    explicit = os.environ.get("EVOMIND_WORKSTATION_ROOT")
    if explicit:
        candidates.append(Path(explicit))
    if ROOT_POINTER.is_file():
        try:
            candidates.append(Path(ROOT_POINTER.read_text(encoding="utf-8-sig").strip()))
        except OSError:
            pass
    candidates.append(SOURCE_ROOT)
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if _is_full_workstation_root(resolved):
            return resolved
    raise RuntimeError("full EvoMind workstation source bundle was not found")


ROOT = resolve_workstation_root()


def bootstrap_url_path(port: int = 8088) -> Path:
    suffix = "" if port == 8088 else f".{port}"
    return ROOT / "web" / "research-agent-workstation" / ".runtime-logs" / f"dashboard{suffix}.bootstrap.once"


def _validated_bootstrap_url(target: Path, port: int) -> str:
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("dashboard bootstrap file is missing or unsafe")
    size = target.stat().st_size
    if not 32 <= size <= 4096:
        raise RuntimeError("dashboard bootstrap file has an invalid size")
    value = target.read_text(encoding="utf-8").strip()
    parsed = urllib.parse.urlsplit(value)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise RuntimeError("dashboard bootstrap URL has an invalid port") from exc
    query = urllib.parse.parse_qs(parsed.query, strict_parsing=False)
    fragment = urllib.parse.parse_qs(parsed.fragment, strict_parsing=False)
    token = fragment.get("bootstrap", [""])[0]
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed_port != port
        or parsed.path != "/"
        or query.get("page") != ["assistant"]
        or not 24 <= len(token) <= 256
    ):
        raise RuntimeError("dashboard bootstrap URL failed the loopback session contract")
    return value


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
    try:
        root = resolve_workstation_root()
    except RuntimeError as exc:
        print(str(exc))
        return 1
    manager = root / "scripts" / "manage_workstation_dashboard.py"
    if not manager.exists():
        print(f"dashboard manager not found: {manager}")
        return 1

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


def open_dashboard(
    *,
    port: int = 8088,
    timeout: float = 90.0,
    build: bool = False,
) -> int:
    """Open a fresh authenticated workstation session in the default browser.

    Every invocation restarts through the verified lifecycle manager so the
    browser always receives a newly minted one-time bootstrap token. Merely
    finding a bootstrap file is insufficient: another browser or verifier may
    already have consumed its server-side token while leaving the file behind.
    The token stays in the URL fragment and is never printed.
    """
    rc = run_dashboard("restart", port=port, timeout=timeout, build=build, force=True)
    if rc != 0:
        return rc

    target = bootstrap_url_path(port)
    deadline = time.monotonic() + min(max(timeout, 1.0), 15.0)
    while not target.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        bootstrap_url = _validated_bootstrap_url(target, port)
    except (OSError, RuntimeError) as exc:
        print(json.dumps({
            "status": "failed",
            "stage": "bootstrap_url",
            "message": str(exc),
            "bootstrap_token_exposed": False,
        }, ensure_ascii=False, indent=2))
        return 1

    try:
        opened = webbrowser.open(bootstrap_url, new=2, autoraise=True)
    except webbrowser.Error as exc:
        print(json.dumps({
            "status": "failed",
            "stage": "browser_open",
            "message": str(exc),
            "bootstrap_token_exposed": False,
        }, ensure_ascii=False, indent=2))
        return 1
    if opened is False:
        print(json.dumps({
            "status": "failed",
            "stage": "browser_open",
            "message": "the default browser did not accept the local workstation URL",
            "bootstrap_token_exposed": False,
        }, ensure_ascii=False, indent=2))
        return 1

    target.unlink(missing_ok=True)
    print(json.dumps({
        "status": "opened",
        "url": f"http://127.0.0.1:{port}/?page=assistant",
        "session": "one_time_bootstrap_dispatched",
        "bootstrap_file_removed": not target.exists(),
        "bootstrap_token_exposed": False,
    }, ensure_ascii=False, indent=2))
    return 0
