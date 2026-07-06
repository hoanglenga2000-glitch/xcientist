from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
NEXT_DIR = WEB / ".next"
OUT_JSON = ROOT / "workspace" / "workstation_server_health_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_SERVER_HEALTH_20260630.md"

LEGACY_STDOUT = ROOT / "next-dev-8088.out.log"
LEGACY_STDERR = ROOT / "next-dev-8088.err.log"
RUNTIME_STDOUT = WEB / ".runtime-logs" / "dashboard.out.log"
RUNTIME_STDERR = WEB / ".runtime-logs" / "dashboard.err.log"

CHUNK_ERROR_PATTERNS = [
    "Cannot find module './",
    "MODULE_NOT_FOUND",
    "webpack-runtime.js",
    "ENOENT",
]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def fetch_text(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        request = Request(url, headers={"Accept": "text/html,application/json,text/css"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read(2_000_000)
            return {
                "target": path,
                "url": url,
                "status": response.status,
                "ok": response.status == 200 and bool(body),
                "content_type": response.headers.get("content-type", ""),
                "bytes": len(body),
                "text_sample": body[:4096].decode("utf-8", errors="replace"),
            }
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return {
            "target": path,
            "url": url,
            "status": exc.code,
            "ok": False,
            "error": body or str(exc.reason),
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "target": path,
            "url": url,
            "status": "error",
            "ok": False,
            "error": str(exc),
        }


def pids_on_port(port: int) -> list[int]:
    if sys.platform == "win32":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        pids: list[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return sorted(set(pids))
    result = subprocess.run(["sh", "-c", f"ss -ltnp 'sport = :{port}' 2>/dev/null || true"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    pids = []
    for match in re.finditer(r"pid=(\d+)", result.stdout):
        pids.append(int(match.group(1)))
    return sorted(set(pids))


def read_tail(path: Path, max_chars: int = 16000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def inspect_logs() -> dict[str, Any]:
    logs = [LEGACY_STDERR, LEGACY_STDOUT, RUNTIME_STDERR, RUNTIME_STDOUT]
    findings = []
    for path in logs:
        tail = read_tail(path)
        if not tail:
            findings.append({
                "path": safe_relative(path),
                "exists": path.exists(),
                "has_chunk_error": False,
                "matched_patterns": [],
            })
            continue
        matched = [pattern for pattern in CHUNK_ERROR_PATTERNS if pattern in tail]
        findings.append({
            "path": safe_relative(path),
            "exists": path.exists(),
            "has_chunk_error": bool(matched),
            "matched_patterns": matched,
            "tail_excerpt": tail[-1200:] if matched else "",
        })
    return {
        "logs": findings,
        "has_chunk_or_module_error": any(item["has_chunk_error"] for item in findings),
    }


def css_health(base_url: str, home: dict[str, Any], timeout: float) -> dict[str, Any]:
    html = home.get("text_sample") if home.get("ok") else ""
    match = re.search(r'href="([^"]*\.css[^"]*)"', html or "")
    if not match:
        return {"ok": False, "css_href": None, "error": "No CSS bundle href found in home page sample."}
    href = match.group(1)
    css_path = href if href.startswith("/") else f"/{href}"
    css = fetch_text(base_url, css_path, timeout)
    content = css.get("text_sample") or ""
    return {
        "ok": bool(css.get("ok")) and ("--tw-border-spacing-x" in content or "tailwind" in content.lower() or len(content) > 1000),
        "css_href": css_path,
        "status": css.get("status"),
        "bytes": css.get("bytes"),
        "content_type": css.get("content_type"),
        "error": css.get("error"),
    }


def run_repair(port: int) -> dict[str, Any]:
    script = ROOT / "scripts" / "restart_workstation_frontend.ps1"
    if not script.exists():
        return {"ok": False, "error": "restart_workstation_frontend.ps1 is missing."}
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Port",
        str(port),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    return {
        "ok": completed.returncode == 0,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }


def build_report(base_url: str, port: int, timeout: float, repair: bool) -> dict[str, Any]:
    home = fetch_text(base_url, "/", timeout)
    summary = fetch_text(base_url, "/api/workstation-summary", timeout)
    tasks = fetch_text(base_url, "/api/tasks", timeout)
    css = css_health(base_url, home, timeout)
    logs = inspect_logs()
    pids = pids_on_port(port)

    failure_reasons = []
    warnings = []
    if not pids:
        failure_reasons.append("port_not_listening")
    if not home.get("ok"):
        failure_reasons.append("home_not_ok")
    if not summary.get("ok"):
        failure_reasons.append("workstation_summary_not_ok")
    if not tasks.get("ok"):
        failure_reasons.append("tasks_api_not_ok")
    if not css.get("ok"):
        failure_reasons.append("css_bundle_not_ok")
    endpoint_failed = any(reason in failure_reasons for reason in ["home_not_ok", "workstation_summary_not_ok", "tasks_api_not_ok", "css_bundle_not_ok"])
    if logs.get("has_chunk_or_module_error") and endpoint_failed:
        failure_reasons.append("next_chunk_or_module_error_in_logs")
    elif logs.get("has_chunk_or_module_error"):
        warnings.append("historical_next_chunk_or_module_error_in_logs")

    repair_result = None
    if repair and failure_reasons:
        repair_result = run_repair(port)
        if repair_result.get("ok"):
            home = fetch_text(base_url, "/", timeout)
            summary = fetch_text(base_url, "/api/workstation-summary", timeout)
            tasks = fetch_text(base_url, "/api/tasks", timeout)
            css = css_health(base_url, home, timeout)
            logs = inspect_logs()
            pids = pids_on_port(port)
            failure_reasons = []
            warnings = []
            if not pids:
                failure_reasons.append("port_not_listening")
            if not home.get("ok"):
                failure_reasons.append("home_not_ok")
            if not summary.get("ok"):
                failure_reasons.append("workstation_summary_not_ok")
            if not tasks.get("ok"):
                failure_reasons.append("tasks_api_not_ok")
            if not css.get("ok"):
                failure_reasons.append("css_bundle_not_ok")
            endpoint_failed = any(reason in failure_reasons for reason in ["home_not_ok", "workstation_summary_not_ok", "tasks_api_not_ok", "css_bundle_not_ok"])
            if logs.get("has_chunk_or_module_error") and endpoint_failed:
                failure_reasons.append("next_chunk_or_module_error_in_logs")
            elif logs.get("has_chunk_or_module_error"):
                warnings.append("historical_next_chunk_or_module_error_in_logs")

    status = "passed" if not failure_reasons else "failed"
    return {
        "schema": "academic_research_os.workstation_server_health.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "port": port,
        "status": status,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "port_pids": pids,
        "next_dir_exists": NEXT_DIR.exists(),
        "next_build_id_exists": (NEXT_DIR / "BUILD_ID").exists(),
        "home": {key: value for key, value in home.items() if key != "text_sample"},
        "workstation_summary": {key: value for key, value in summary.items() if key != "text_sample"},
        "tasks_api": {key: value for key, value in tasks.items() if key != "text_sample"},
        "css": css,
        "log_diagnostics": logs,
        "repair_requested": repair,
        "repair_result": repair_result,
        "repair_command": "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restart_workstation_frontend.ps1 -Port 8088",
        "claim_boundary": (
            "This check proves local workstation server health only: port listener, page HTML, CSS bundle, "
            "summary API, tasks API, and recent Next.js chunk/module errors. It does not prove training, "
            "Kaggle submission, GPU availability, or Figma parity."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# 工作站服务健康检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 地址：`{report['base_url']}`",
        f"- 端口：`{report['port']}`",
        f"- 状态：`{report['status']}`",
        f"- 失败原因：`{', '.join(report['failure_reasons']) or 'none'}`",
        f"- 警告：`{', '.join(report['warnings']) or 'none'}`",
        f"- 监听进程：`{', '.join(str(pid) for pid in report['port_pids']) or 'none'}`",
        f"- .next 存在：`{report['next_dir_exists']}`",
        f"- BUILD_ID 存在：`{report['next_build_id_exists']}`",
        "",
        "## Endpoint Health",
        "",
        "| endpoint | status | ok | bytes |",
        "| --- | --- | --- | ---: |",
        f"| `/` | `{report['home'].get('status')}` | `{report['home'].get('ok')}` | {report['home'].get('bytes')} |",
        f"| `/api/workstation-summary` | `{report['workstation_summary'].get('status')}` | `{report['workstation_summary'].get('ok')}` | {report['workstation_summary'].get('bytes')} |",
        f"| `/api/tasks` | `{report['tasks_api'].get('status')}` | `{report['tasks_api'].get('ok')}` | {report['tasks_api'].get('bytes')} |",
        "",
        "## CSS",
        "",
        f"- CSS href：`{report['css'].get('css_href')}`",
        f"- CSS 状态：`{report['css'].get('status')}`",
        f"- CSS ok：`{report['css'].get('ok')}`",
        f"- CSS bytes：`{report['css'].get('bytes')}`",
        "",
        "## Log Diagnostics",
        "",
        f"- chunk/module error：`{report['log_diagnostics'].get('has_chunk_or_module_error')}`",
        "",
        "| log | exists | chunk/module error | patterns |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["log_diagnostics"]["logs"]:
        lines.append(
            f"| `{item['path']}` | `{item['exists']}` | `{item['has_chunk_error']}` | `{', '.join(item.get('matched_patterns') or []) or 'none'}` |"
        )
    lines.extend([
        "",
        "## Repair",
        "",
        f"- repair requested：`{report['repair_requested']}`",
        f"- repair command：`{report['repair_command']}`",
        f"- repair ok：`{(report.get('repair_result') or {}).get('ok') if report.get('repair_result') else 'not_run'}`",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local workstation server health and optionally repair the dev server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--timeout", type=float, default=12)
    parser.add_argument("--repair", action="store_true", help="Run restart_workstation_frontend.ps1 only if health checks fail.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report(args.base_url, args.port, args.timeout, args.repair)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown(report)

    print(json.dumps({
        "status": report["status"],
        "failure_reasons": report["failure_reasons"],
        "warnings": report["warnings"],
        "port_pids": report["port_pids"],
        "css_ok": report["css"].get("ok"),
        "summary_ok": report["workstation_summary"].get("ok"),
        "chunk_error": report["log_diagnostics"].get("has_chunk_or_module_error"),
        "repair_requested": report["repair_requested"],
        "repair_ok": (report.get("repair_result") or {}).get("ok") if report.get("repair_result") else None,
        "json": safe_relative(OUT_JSON) if args.write_report else None,
        "md": safe_relative(OUT_MD) if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
