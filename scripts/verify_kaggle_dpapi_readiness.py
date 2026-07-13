from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "scripts" / "manage_kaggle_secret.ps1"
REPORT_DIR = ROOT / "workspace" / "verification"
JSON_REPORT = REPORT_DIR / "kaggle_dpapi_readiness.json"
MD_REPORT = REPORT_DIR / "kaggle_dpapi_readiness.md"


def fail(message: str, evidence: dict[str, Any] | None = None) -> None:
    raise SystemExit(
        json.dumps(
            {"status": "failed", "message": message, "evidence": evidence or {}},
            ensure_ascii=False,
            indent=2,
        )
    )


def decode_process_bytes(data: bytes) -> str:
    encodings = (
        ("utf-16", "utf-16-le", "utf-8-sig", "utf-8", "cp936", "gb18030")
        if data.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in data
        else ("utf-8-sig", "utf-8", "cp936", "gb18030")
    )
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _manager_environment() -> dict[str, str]:
    env = os.environ.copy()
    windows_root = Path(env.get("WINDIR") or r"C:\Windows")
    security_module_root = str(
        windows_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
    )
    module_paths = [
        value
        for value in env.get("PSModulePath", "").split(os.pathsep)
        if value
    ]
    if not any(
        value.casefold() == security_module_root.casefold()
        for value in module_paths
    ):
        module_paths.insert(0, security_module_root)
    env["PSModulePath"] = os.pathsep.join(module_paths)
    return env


def run_manager(command: str, *, allow_real_external: bool = False) -> dict[str, Any]:
    args = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(MANAGER),
        command,
    ]
    if allow_real_external:
        args.append("-AllowRealExternal")
    completed = subprocess.run(
        args,
        cwd=ROOT,
        env=_manager_environment(),
        capture_output=True,
        timeout=60,
    )
    stdout = decode_process_bytes(completed.stdout)
    stderr = decode_process_bytes(completed.stderr)
    if completed.returncode != 0:
        manager_error: dict[str, Any] = {}
        try:
            raw_error = json.loads(stdout)
            if isinstance(raw_error, dict):
                manager_error = {
                    key: raw_error.get(key)
                    for key in ("status", "error_code", "error_type", "credential_installed")
                }
        except json.JSONDecodeError:
            manager_error = {"status": "unparseable_error_payload"}
        fail(
            f"Kaggle secret manager {command} failed",
            {
                "returncode": completed.returncode,
                "manager_error": manager_error,
                "stderr": stderr[-1000:],
            },
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        fail(
            "Kaggle secret manager did not return clean JSON",
            {"error": str(error), "stderr": stderr[-1000:]},
        )
    if not isinstance(payload, dict):
        fail("Kaggle secret manager returned a non-object payload")
    return payload


def require(condition: bool, message: str, evidence: dict[str, Any] | None = None) -> None:
    if not condition:
        fail(message, evidence)


def _sanitized_tool_status(status: dict[str, Any]) -> dict[str, Any]:
    raw = status.get("tool_status")
    tool = raw if isinstance(raw, dict) else {}
    cli_path = str(tool.get("cli_path") or "")
    return {
        "python_package_installed": tool.get("python_package_installed") is True,
        "python_package_version": str(tool.get("python_package_version") or "unknown"),
        "cli_available": bool(cli_path),
        "cli_name": Path(cli_path).name if cli_path else None,
    }


def build_report(
    status: dict[str, Any],
    *,
    real_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manager_status = status.get("status")
    require(
        manager_status in {"configured", "not_configured"},
        "unexpected Kaggle credential status",
        {"status": manager_status},
    )
    credential_installed = bool(status.get("credential_installed"))
    require(
        credential_installed == (manager_status == "configured"),
        "Kaggle manager status and credential flag disagree",
    )
    token_type = str(status.get("token_type") or ("unknown" if credential_installed else "none"))
    require(
        token_type in {"none", "access_token", "legacy_username_key", "unknown"},
        "unexpected Kaggle token type",
        {"token_type": token_type},
    )
    credential_path = str(status.get("credential_path") or "")
    if credential_installed:
        require(bool(credential_path), "configured Kaggle credential has no protected-store path")
        try:
            in_repo = Path(credential_path).resolve().is_relative_to(ROOT.resolve())
        except (OSError, ValueError):
            in_repo = True
        require(not in_repo, "Kaggle credential store must be outside the repository")

    tool_status = _sanitized_tool_status(status)
    require(tool_status["python_package_installed"], "Kaggle Python package is not installed")
    require(
        tool_status["python_package_version"] != "unknown",
        "Kaggle Python package version is missing",
    )

    authenticated = False
    if real_smoke is not None:
        require(credential_installed, "real Kaggle smoke requires an installed protected credential")
        require(real_smoke.get("status") == "passed", "Kaggle real API smoke did not pass")
        require(real_smoke.get("real_external_called") is True, "Kaggle smoke did not call the real API")
        authenticated = True

    credential_status = (
        "authenticated_real_api"
        if authenticated
        else "configured_dpapi_unverified"
        if credential_installed
        else "not_configured"
    )
    overall_status = "passed" if authenticated else "auth_pending"
    conclusion = (
        "The protected Kaggle credential passed an explicit real API smoke. Official submission remains human-gated."
        if authenticated
        else "Kaggle authentication is pending. Install a protected token and run an explicit real API smoke before claiming readiness."
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": overall_status,
        "verification_method": "dpapi_status_and_real_api_smoke" if authenticated else "dpapi_status_only",
        "credential_status": credential_status,
        "credential_installed": credential_installed,
        "credential_store": "protected_user_dpapi" if credential_installed else "not_configured",
        "token_type": token_type,
        "authenticated": authenticated,
        "tool_status": tool_status,
        "safe_install_command": (
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            "scripts\\manage_kaggle_secret.ps1 install-token"
        ),
        "real_api_smoke_command": (
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            "scripts\\manage_kaggle_secret.ps1 smoke -AllowRealExternal"
        ),
        "human_gate_required_for_submission": True,
        "conclusion": conclusion,
    }


def write_markdown(report: dict[str, Any]) -> None:
    tool = report["tool_status"]
    lines = [
        "# Kaggle credential readiness",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Credential: `{report['credential_status']}`",
        f"- Authenticated: `{report['authenticated']}`",
        f"- Kaggle package: `{tool.get('python_package_version')}`",
        f"- Kaggle CLI available: `{tool.get('cli_available')}`",
        "",
        "## Boundary",
        "",
        report["conclusion"],
        "No credential value or user-specific credential path is written to this report.",
        "Official leaderboard submission always requires a Human Gate.",
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify protected Kaggle credential state without exposing secrets or implying authentication."
    )
    parser.add_argument(
        "--allow-real-external",
        action="store_true",
        help="explicitly run the Kaggle manager's real API smoke",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="write a sanitized runtime report under ignored workspace/verification",
    )
    args = parser.parse_args()

    require(MANAGER.is_file(), "missing Kaggle secret manager script")
    status = run_manager("status")
    real_smoke = run_manager("smoke", allow_real_external=True) if args.allow_real_external else None
    report = build_report(status, real_smoke=real_smoke)

    if args.write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report)
        report["report_paths"] = {
            "json": str(JSON_REPORT.relative_to(ROOT)).replace("\\", "/"),
            "markdown": str(MD_REPORT.relative_to(ROOT)).replace("\\", "/"),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
