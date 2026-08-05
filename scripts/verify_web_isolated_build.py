#!/usr/bin/env python3
"""Build the Next.js workstation in an isolated staging directory.

This verifier covers the Windows production workflow where the live 8088
dashboard keeps the current .next/standalone Prisma query-engine DLL loaded.
Running npm run build in-place can fail at the cleanup step with EPERM even
though the source compiles. The verifier copies only build inputs to a temp
directory, installs dependencies from package-lock, builds there, and records
evidence without stopping the live service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "web-isolated-build-20260804.json"
PROTECTED_PORTS = (8088, 8765, 17897, 65068)

COPY_ENTRIES = (
    "src",
    "public",
    "prisma",
    "scripts",
    "package.json",
    "package-lock.json",
    "next.config.mjs",
    "postcss.config.mjs",
    "tailwind.config.ts",
    "tsconfig.json",
    "next-env.d.ts",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1"},
        )
        output = completed.stdout or ""
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "output_sha256": hashlib.sha256(output.encode("utf-8", "replace")).hexdigest(),
            "output_tail": output[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return {
            "ok": False,
            "returncode": None,
            "seconds": round(time.perf_counter() - started, 3),
            "timeout": True,
            "output_sha256": hashlib.sha256(output.encode("utf-8", "replace")).hexdigest(),
            "output_tail": output[-12000:],
        }


def npm_command() -> list[str]:
    """Return a subprocess-safe npm command.

    On this Windows workstation npm.cmd lives next to a portable node.exe.  The
    cmd shim can fail with "The system cannot find the path specified" when
    launched from Python in a temp cwd.  Prefer node + npm-cli.js when it can be
    resolved; fall back to the shim only when needed.
    """

    node = shutil.which("node.exe") or shutil.which("node")
    for npm_shim in (shutil.which("npm.cmd"), shutil.which("npm")):
        if not npm_shim:
            continue
        npm_dir = Path(npm_shim).resolve().parent
        for candidate in (
            npm_dir / "node_modules" / "npm" / "bin" / "npm-cli.js",
            npm_dir / "node_modules" / "npm" / "bin" / "npm-cli.js",
            npm_dir.parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
        ):
            if node and candidate.is_file():
                return [node, str(candidate)]
    if shutil.which("npm.cmd"):
        return [shutil.which("npm.cmd") or "npm.cmd"]
    if shutil.which("npm"):
        return [shutil.which("npm") or "npm"]
    raise FileNotFoundError("npm executable was not found on PATH")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def port_snapshot() -> dict[str, bool]:
    return {str(port): port_open(port) for port in PROTECTED_PORTS}


def copy_inputs(stage: Path) -> list[str]:
    copied: list[str] = []
    for entry in COPY_ENTRIES:
        source = WEB / entry
        if not source.exists():
            continue
        target = stage / entry
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(".next", "node_modules", ".runtime-logs", "*.log", "__pycache__"),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied.append(entry)
    return copied


def build(*, keep_stage: bool, npm_ci_timeout: int, build_timeout: int) -> dict[str, Any]:
    before_ports = port_snapshot()
    stage_parent = Path(tempfile.mkdtemp(prefix="evomind-web-build-parent-")).resolve()
    stage = stage_parent / "web"
    stage.mkdir()
    copied = copy_inputs(stage)
    report: dict[str, Any] = {
        "schema": "evomind.web_isolated_build.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": str(WEB.resolve()),
        "stage": str(stage),
        "copied_entries": copied,
        "keep_stage": keep_stage,
        "protected_ports_before": before_ports,
    }
    try:
        npm = npm_command()
        report["npm_command"] = npm
        npm_ci = run_command([*npm, "ci", "--prefer-offline", "--no-audit", "--fund=false"], cwd=stage, timeout=npm_ci_timeout)
        report["npm_ci"] = npm_ci
        if not npm_ci["ok"]:
            report["status"] = "failed"
            return report

        npm_audit = run_command(
            [*npm, "audit", "--audit-level=low", "--registry=https://registry.npmjs.org"],
            cwd=stage,
            timeout=120,
        )
        report["npm_audit"] = npm_audit
        if not npm_audit["ok"]:
            report["status"] = "failed"
            return report

        prisma_generate = run_command([*npm, "run", "db:generate"], cwd=stage, timeout=120)
        report["prisma_generate"] = prisma_generate
        if not prisma_generate["ok"]:
            report["status"] = "failed"
            return report

        build_result = run_command([*npm, "run", "build"], cwd=stage, timeout=build_timeout)
        report["build"] = build_result
        build_id_path = stage / ".next" / "BUILD_ID"
        standalone_server = stage / ".next" / "standalone" / "server.js"
        package_lock = stage / "package-lock.json"
        report.update({
            "build_id": build_id_path.read_text(encoding="utf-8").strip() if build_id_path.is_file() else None,
            "standalone_server_exists": standalone_server.is_file(),
            "package_lock_sha256": sha256_file(package_lock) if package_lock.is_file() else None,
            "next_build_dir_exists": (stage / ".next").is_dir(),
            "status": "passed" if build_result["ok"] and standalone_server.is_file() and build_id_path.is_file() else "failed",
        })
        return report
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["protected_ports_after"] = port_snapshot()
        report["protected_ports_unchanged"] = report["protected_ports_before"] == report["protected_ports_after"]
        if keep_stage:
            report["stage_removed"] = False
        else:
            shutil.rmtree(stage_parent, ignore_errors=True)
            report["stage_removed"] = not stage_parent.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--npm-ci-timeout", type=int, default=300)
    parser.add_argument("--build-timeout", type=int, default=420)
    args = parser.parse_args()

    report = build(keep_stage=args.keep_stage, npm_ci_timeout=args.npm_ci_timeout, build_timeout=args.build_timeout)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({
        "status": report.get("status"),
        "output": str(target),
        "stage": report.get("stage"),
        "stage_removed": report.get("stage_removed"),
        "protected_ports_unchanged": report.get("protected_ports_unchanged"),
        "build_id": report.get("build_id"),
        "npm_ci_ok": (report.get("npm_ci") or {}).get("ok"),
        "npm_audit_ok": (report.get("npm_audit") or {}).get("ok"),
        "build_ok": (report.get("build") or {}).get("ok"),
        "standalone_server_exists": report.get("standalone_server_exists"),
    }, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" and report.get("protected_ports_unchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
