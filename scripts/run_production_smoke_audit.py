from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"


def windows_path_from_wsl(path: Path) -> str | None:
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:\\" + "\\".join(parts[3:])
    return None


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def windows_python_from_wsl() -> str | None:
    candidates = [
        Path("/mnt/c/codex-python/python.exe"),
        Path("/mnt/c/Users/景浩伟/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return windows_path_from_wsl(candidate)
    return None


def powershell_project_command(cwd: Path, executable: str, args: list[str]) -> list[str]:
    windows_cwd = windows_path_from_wsl(cwd)
    if not windows_cwd:
        raise ValueError(f"Cannot convert WSL path to Windows path: {cwd}")
    command = f"Set-Location -LiteralPath {ps_quote(windows_cwd)}; & {ps_quote(executable)} " + " ".join(ps_quote(arg) for arg in args)
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]


def npm_command(script: str) -> list[str]:
    windows_web = windows_path_from_wsl(WEB)
    if os.name != "nt" and windows_web:
        return powershell_project_command(WEB, "npm.cmd", ["run", script])
    executable = "npm.cmd" if os.name == "nt" else "npm"
    return [executable, "run", script]


def python_command(*args: str) -> list[str]:
    windows_python = windows_python_from_wsl()
    if os.name != "nt" and windows_python and windows_path_from_wsl(ROOT):
        return powershell_project_command(ROOT, windows_python, list(args))
    return [sys.executable, *args]


def parse_json(text: str | None) -> Any:
    try:
        return json.loads(text or "")
    except json.JSONDecodeError:
        return None


def run(command: list[str], cwd: Path = ROOT, timeout: int = 240) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=env,
            timeout=timeout,
        )
        return {
            "command": " ".join(command),
            "cwd": str(cwd.relative_to(ROOT)) if cwd != ROOT else ".",
            "timeout_seconds": timeout,
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
            "json": parse_json(completed.stdout),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": " ".join(command),
            "cwd": str(cwd.relative_to(ROOT)) if cwd != ROOT else ".",
            "timeout_seconds": timeout,
            "returncode": 124,
            "timed_out": True,
            "stdout": (error.stdout or "").strip() if isinstance(error.stdout, str) else "",
            "stderr": (error.stderr or "").strip() if isinstance(error.stderr, str) else "",
            "json": None,
        }


def dashboard_port(base_url: str) -> int:
    parsed = urllib.parse.urlparse(base_url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def manage_dashboard(action: str, port: int) -> dict[str, Any]:
    return run(python_command("scripts/manage_workstation_dashboard.py", action, "--port", str(port)), ROOT, timeout=90)


def restart_dashboard_after_build(base_url: str) -> dict[str, Any]:
    port = dashboard_port(base_url)
    stop = manage_dashboard("stop", port)
    start = manage_dashboard("start", port)
    return {
        "command": f"restart dashboard on port {port}",
        "cwd": ".",
        "timeout_seconds": 180,
        "returncode": 0 if start["returncode"] == 0 else start["returncode"],
        "stdout": json.dumps({"stop": stop.get("json") or stop.get("stdout"), "start": start.get("json") or start.get("stdout")}, ensure_ascii=False),
        "stderr": start.get("stderr", ""),
        "json": {"stop": stop.get("json"), "start": start.get("json")},
    }


def check_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        body = response.read().decode("utf-8", errors="replace")
        return {
            "url": url,
            "http_status": response.status,
            "content_type": response.headers.get("content-type"),
            "excerpt": body[:500],
        }


def require_success(result: dict[str, Any]) -> None:
    if result["returncode"] != 0:
        raise SystemExit(json.dumps({"status": "failed", "failed_command": result}, ensure_ascii=False, indent=2))


def write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        "# 科研 Agent 工作站正式上线冒烟测试报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 总体状态：{report['overall_status']}",
        f"- Dashboard：{report['dashboard_url']}",
        f"- 检查数量：{len(report['checks'])}",
        "",
        "## 结论",
        "",
        report["conclusion"],
        "",
        "## 资源状态",
        "",
    ]
    resource_status = report.get("resource_status") or {}
    for name, status in resource_status.items():
        ready = status.get("ready")
        missing = ", ".join(status.get("missing") or []) or "无"
        lines.append(f"- {name}: {'ready' if ready else 'not ready'}；缺少：{missing}")

    lines.extend(["", "## 检查明细", ""])
    for check in report["checks"]:
        timeout = check.get("timeout_seconds")
        lines.append(f"- `{check['command']}`：returncode={check['returncode']}，timeout={timeout}")

    lines.extend(["", "## URL 检查", ""])
    for item in report["url_checks"]:
        lines.append(f"- {item['url']}：HTTP {item['http_status']}，{item.get('content_type')}")

    lines.extend(
        [
            "",
            "## 接入后复测命令",
            "",
            "- 无凭证安全降级：`python scripts\\run_production_smoke_audit.py --dashboard-url http://127.0.0.1:8088`",
            "- 配置 Claude/GPU 后强制真实接入：`python scripts\\run_production_smoke_audit.py --dashboard-url http://127.0.0.1:8088 --require-configured --allow-real-external`",
            "- 单独真实资源 smoke：`python scripts\\run_real_resource_smoke.py --dashboard-url http://127.0.0.1:8088 --require-configured`",
            "",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run a bounded production-style smoke audit for the research workstation.")
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8088")
    parser.add_argument("--container-name", default="research-agent-workstation")
    parser.add_argument("--allow-real-external", action="store_true", help="Allow real Claude/GPU invocation when credentials are configured.")
    parser.add_argument("--require-configured", action="store_true", help="Fail if Claude API or GPU SSH resources are missing.")
    parser.add_argument("--write-report", action="store_true", help="Write JSON and Markdown reports under docs/.")
    args = parser.parse_args()

    base = args.dashboard_url.rstrip("/")
    commands = [
        npm_command("typecheck"),
        npm_command("build"),
        python_command("scripts/verify_launch_integration_hardening.py"),
        python_command("scripts/verify_launch_resource_readiness.py", "--write-report"),
        python_command("scripts/verify_backend_resource_status.py", "--url", base),
        python_command("scripts/verify_no_plaintext_secrets.py"),
        python_command("scripts/run_full_acceptance.py", "--dashboard-url", base, "--container-name", args.container_name),
        python_command("scripts/run_real_resource_smoke.py", "--dashboard-url", base, "--container-name", args.container_name, "--skip-full-acceptance"),
    ]
    command_cwds = [WEB, WEB, ROOT, ROOT, ROOT, ROOT, ROOT, ROOT]
    command_timeouts = [120, 180, 90, 90, 60, 60, 240, 240]

    if args.require_configured:
        commands[-1].append("--require-configured")

    checks = []
    for index, (command, cwd, timeout) in enumerate(zip(commands, command_cwds, command_timeouts)):
        result = run(command, cwd, timeout=timeout)
        require_success(result)
        checks.append(result)
        if index == 1:
            restart = restart_dashboard_after_build(base)
            require_success(restart)
            checks.append(restart)

    url_checks = [check_url(f"{base}/api/workstation-summary"), check_url(base)]
    smoke_json = checks[-1].get("json") or {}
    resource_status = smoke_json.get("resource_status") or {}
    local_ready_json = next((check.get("json") for check in checks if "verify_launch_resource_readiness.py" in check["command"]), {}) or {}
    fully_ready = local_ready_json.get("overall_status") == "fully_ready"
    ready_for_external = local_ready_json.get("overall_status") == "ready_for_external_resources"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dashboard_url": base,
        "overall_status": "fully_ready" if fully_ready else "ready_for_external_resources" if ready_for_external else "not_ready",
        "resource_status": resource_status,
        "checks": checks,
        "url_checks": url_checks,
        "conclusion": (
            "本地系统、前端构建、完整验收、密钥安全、Claude/GPU 网关均已通过冒烟测试；外部资源配置后可用 --require-configured 做真实接入验收。"
            if ready_for_external or fully_ready
            else "冒烟测试未证明上线就绪，请查看 failed_command 或资源就绪审计。"
        ),
    }

    if args.write_report:
        json_path = ROOT / "docs" / "production_smoke_audit.json"
        md_path = ROOT / "docs" / "正式上线全面冒烟测试报告-20260611.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(report, md_path)
        report["report_paths"] = {
            "json": str(json_path.relative_to(ROOT)),
            "markdown": str(md_path.relative_to(ROOT)),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall_status"] == "not_ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
