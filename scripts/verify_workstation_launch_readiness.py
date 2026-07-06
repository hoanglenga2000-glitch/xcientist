from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from shutil import which


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
OUT_JSON = ROOT / "workspace" / "workstation_launch_readiness_20260630.json"
OUT_MD = ROOT / "reports" / "WORKSTATION_LAUNCH_READINESS_20260630.md"
FIGMA_GATE_JSON = ROOT / "workspace" / "workstation_figma_design_gate_20260630.json"


COMMANDS: list[dict[str, Any]] = [
    {
        "id": "server_health",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_server_health.py", "--write-report"],
        "claim": "The local workstation server responds with healthy HTML, CSS, summary API, tasks API, and no current Next.js chunk failure.",
    },
    {
        "id": "ui_component_wiring",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_ui_component_wiring.py"],
        "claim": "All workstation pages and clickable UI actions are wired to routes/actions.",
    },
    {
        "id": "ui_action_contract",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_ui_action_contract.py", "--write-report", "--live-post-safe"],
        "claim": "UI actions have global click audit coverage and direct workstation actions are accepted by backend handlers.",
    },
    {
        "id": "runtime_navigation_matrix",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_runtime_navigation.py", "--write-report"],
        "claim": "Runtime page navigation, URL aliases, fallback page handling, and read-only API endpoints respond on the running workstation.",
    },
    {
        "id": "browser_render_smoke",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_browser_render_smoke.py", "--write-report"],
        "claim": "Critical workstation pages render in a real headless Chromium-compatible browser with shell, sidebar, page markers, and interactive controls.",
    },
    {
        "id": "frontend_api_contract",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_frontend_api_contract.py", "--write-report"],
        "claim": "Frontend API client fetch calls are bound to existing Next.js route files and exported HTTP methods.",
    },
    {
        "id": "task_api_matrix",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_task_api_matrix.py", "--write-report"],
        "claim": "Task-level read-only APIs expose runs/gates/evidence/workflow/report/figures without triggering training.",
    },
    {
        "id": "evidence_backfill_plan",
        "kind": "script",
        "critical": False,
        "cmd": [sys.executable, "scripts/build_workstation_evidence_backfill_plan.py", "--write-report"],
        "claim": "Evidence/report gaps are classified into a workstation-safe remediation plan.",
    },
    {
        "id": "next_run_queue",
        "kind": "script",
        "critical": False,
        "cmd": [sys.executable, "scripts/build_workstation_next_run_queue.py", "--write-report"],
        "claim": "Registry-only tasks are converted into a blocked-but-auditable next-run queue.",
    },
    {
        "id": "plaintext_secret_scan",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_no_plaintext_secrets.py"],
        "claim": "Launch-critical source/config files do not contain plaintext secrets.",
    },
    {
        "id": "deepseek_cache_policy",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_deepseek_cache_policy.py"],
        "claim": "DeepSeek cache policy and stable prompt-prefix helpers are present.",
    },
    {
        "id": "deepseek_cache_hit_rate_target",
        "kind": "script",
        "critical": False,
        "cmd": [sys.executable, "scripts/verify_deepseek_cache_hit_rate_target.py", "--write-report"],
        "claim": "DeepSeek cache implementation can support the >=80% repeated-prompt target.",
    },
    {
        "id": "deepseek_cache_probe_api",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_deepseek_cache_probe_api.py", "--write-report"],
        "claim": "DeepSeek cache probe API can check local cache hit state without creating sessions or calling the external model.",
    },
    {
        "id": "external_resources_manifest",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_external_resources_manifest.py"],
        "claim": "External resource manifest keeps HPC/Kaggle/DeepSeek boundaries explicit.",
    },
    {
        "id": "gpu_job_templates",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_gpu_job_templates.py"],
        "claim": "GPU jobs are restricted to approved workstation templates.",
    },
    {
        "id": "kaggle_dpapi_readiness",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_kaggle_dpapi_readiness.py", "--write-report"],
        "claim": "Kaggle credential readiness is DPAPI/file based and submission remains human-gated.",
    },
    {
        "id": "kaggle_experiment_inventory",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/build_kaggle_experiment_inventory.py"],
        "claim": "Experiment inventory can be rebuilt from artifacts.",
    },
    {
        "id": "mlebench_style_leaderboard",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/build_mlebench_style_leaderboard_report.py"],
        "claim": "MLE-Bench style leaderboard report can be rebuilt without overclaiming medals.",
    },
    {
        "id": "workstation_training_progress",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/build_workstation_training_progress_report.py", "--write-report"],
        "claim": "Training progress can be summarized from workstation artifacts without triggering training or Kaggle submission.",
    },
    {
        "id": "learning_loop_readiness",
        "kind": "script",
        "critical": True,
        "cmd": [sys.executable, "scripts/verify_workstation_learning_loop_readiness.py", "--write-report"],
        "claim": "Automated learning loop evidence exists: retrospective memory, MLEvolve-style search orders, training inventory, benchmark gates, and next-run queue.",
    },
]


API_PATHS = [
    "/api/workstation-summary",
    "/api/tasks",
    "/api/settings",
    "/api/gpu/jobs",
    "/api/paper-evidence-bundle",
]

PAGES = [
    "overview",
    "control",
    "experiments",
    "data",
    "report",
    "code",
    "gpu",
    "evidence",
    "gates",
    "literature",
    "tasks",
    "runtime",
    "workflow",
    "settings",
    "design",
    "mission",
    "evidence-detail",
]


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def run_command(item: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    try:
        completed = subprocess.run(
            item["cmd"],
            cwd=ROOT,
            capture_output=True,
            timeout=timeout,
        )
        stdout = decode(completed.stdout)
        stderr = decode(completed.stderr)
        return {
            "id": item["id"],
            "kind": item["kind"],
            "critical": item["critical"],
            "claim": item["claim"],
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at": started,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(str(part) for part in item["cmd"]),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "id": item["id"],
            "kind": item["kind"],
            "critical": item["critical"],
            "claim": item["claim"],
            "status": "timeout",
            "returncode": None,
            "started_at": started,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(str(part) for part in item["cmd"]),
            "stdout_tail": decode(exc.stdout or b"")[-2000:] if isinstance(exc.stdout, bytes) else str(exc.stdout or "")[-2000:],
            "stderr_tail": decode(exc.stderr or b"")[-2000:] if isinstance(exc.stderr, bytes) else str(exc.stderr or "")[-2000:],
        }


def run_frontend(command: str, timeout: int = 240) -> dict[str, Any]:
    started = datetime.now().isoformat(timespec="seconds")
    npm = which("npm.cmd") or which("npm")
    if not npm:
        return {
            "id": f"frontend_{command}",
            "kind": "frontend",
            "critical": True,
            "claim": f"Frontend npm run {command} passes.",
            "status": "failed",
            "returncode": None,
            "started_at": started,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "command": f"npm run {command}",
            "stdout_tail": "",
            "stderr_tail": "npm executable was not found on PATH.",
        }
    completed = subprocess.run(
        [npm, "run", command],
        cwd=WEB,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "NO_COLOR": "1"},
    )
    stdout = decode(completed.stdout)
    stderr = decode(completed.stderr)
    return {
        "id": f"frontend_{command}",
        "kind": "frontend",
        "critical": True,
        "claim": f"Frontend npm run {command} passes.",
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_at": started,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "command": f"npm run {command}",
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def http_check(base_url: str, path: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        with urlopen(url, timeout=12) as response:
            body = response.read(512)
            return {
                "target": path,
                "url": url,
                "status": response.status,
                "ok": response.status == 200 and bool(body),
            }
    except Exception as exc:  # pragma: no cover - smoke utility
        return {
            "target": path,
            "url": url,
            "status": "error",
            "ok": False,
            "error": str(exc),
        }


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def summarize_inventory() -> dict[str, Any]:
    inventory = read_json(ROOT / "workspace" / "kaggle_experiment_inventory_20260624.json")
    leaderboard = read_json(ROOT / "workspace" / "mlebench_style_current_leaderboard_20260625.json")
    task_matrix = read_json(ROOT / "workspace" / "workstation_task_api_matrix_20260630.json")
    runtime_navigation = read_json(ROOT / "workspace" / "workstation_runtime_navigation_20260630.json")
    ui_action_contract = read_json(ROOT / "workspace" / "workstation_ui_action_contract_20260630.json")
    learning_loop = read_json(ROOT / "workspace" / "workstation_learning_loop_readiness_20260630.json")
    server_health = read_json(ROOT / "workspace" / "workstation_server_health_20260630.json")
    inv_summary = {
        "task_count_with_experiments": inventory.get("task_count_with_experiments"),
        "total_runs_observed": inventory.get("total_runs_observed"),
        "total_scored_runs": inventory.get("total_scored_runs"),
        "total_promoted_runs": inventory.get("total_promoted_runs"),
        "total_held_runs": inventory.get("total_held_runs"),
        "total_timeout_or_failed_runs": inventory.get("total_timeout_or_failed_runs"),
        "kaggle10_completion_status": inventory.get("kaggle10_completion_status"),
        "official_submission_records": len(inventory.get("official_submission_records") or []),
    }
    lb_summary = leaderboard.get("summary") or {}
    return {
        "inventory": inv_summary,
        "leaderboard": {
            "tasks_with_experiments": lb_summary.get("tasks_with_experiments"),
            "official_submission_tasks": lb_summary.get("official_submission_tasks"),
            "official_top30_count": lb_summary.get("official_top30_count"),
            "official_top30_rate_among_all_observed_tasks": lb_summary.get("official_top30_rate_among_all_observed_tasks"),
            "medal_count": lb_summary.get("medal_count"),
            "medal_rate": lb_summary.get("medal_rate"),
            "benchmark_claim_status": lb_summary.get("benchmark_claim_status"),
        },
        "task_api_matrix": {
            "status": task_matrix.get("status"),
            "task_count": task_matrix.get("task_count"),
            "minimum_closed_loop_visible_count": task_matrix.get("minimum_closed_loop_visible_count"),
            "full_reportable_loop_visible_count": task_matrix.get("full_reportable_loop_visible_count"),
            "evidence_missing_tasks": task_matrix.get("evidence_missing_tasks", []),
            "report_missing_tasks": task_matrix.get("report_missing_tasks", []),
        },
        "runtime_navigation": {
            "status": runtime_navigation.get("status"),
            "nav_page_count": runtime_navigation.get("nav_page_count"),
            "parser_page_count": runtime_navigation.get("parser_page_count"),
            "rendered_page_count": runtime_navigation.get("rendered_page_count"),
            "api_path_count": runtime_navigation.get("api_path_count"),
            "failed_pages": runtime_navigation.get("failed_pages", []),
            "failed_apis": runtime_navigation.get("failed_apis", []),
        },
        "ui_action_contract": {
            "status": ui_action_contract.get("status"),
            "global_click_delegate": ui_action_contract.get("global_click_delegate"),
            "backend_default_handler": ui_action_contract.get("backend_default_handler"),
            "backend_case_count": ui_action_contract.get("backend_case_count"),
            "data_ui_action_count": ui_action_contract.get("data_ui_action_count"),
            "direct_frontend_action_count": ui_action_contract.get("direct_frontend_action_count"),
            "telemetry_only_data_ui_action_count": ui_action_contract.get("telemetry_only_data_ui_action_count"),
            "failed_direct_actions": ui_action_contract.get("failed_direct_actions", []),
            "failed_data_ui_actions": ui_action_contract.get("failed_data_ui_actions", []),
            "failed_live_posts": ui_action_contract.get("failed_live_posts", []),
        },
        "learning_loop": {
            "status": learning_loop.get("status"),
            "failures": learning_loop.get("failures", []),
            "resource_blockers": learning_loop.get("resource_blockers", []),
            "memory_records": (learning_loop.get("memory") or {}).get("record_count"),
            "search_order_records": (learning_loop.get("search_orders") or {}).get("record_count"),
            "observed_runs": (learning_loop.get("training_progress") or {}).get("observed_runs"),
            "next_run_ready": (learning_loop.get("next_run_queue") or {}).get("ready_to_start_now"),
        },
        "server_health": {
            "status": server_health.get("status"),
            "failure_reasons": server_health.get("failure_reasons", []),
            "warnings": server_health.get("warnings", []),
            "port_pids": server_health.get("port_pids", []),
            "css_ok": (server_health.get("css") or {}).get("ok"),
            "summary_ok": (server_health.get("workstation_summary") or {}).get("ok"),
            "repair_command": server_health.get("repair_command"),
        },
    }


def resource_snapshot(base_url: str) -> dict[str, Any]:
    try:
        with urlopen(f"{base_url.rstrip()}/api/workstation-summary", timeout=15) as response:
            summary = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"status": "unavailable", "error": str(exc)}
    connector_status = summary.get("connector_status") or {}
    return {
        "status": "loaded",
        "connectors": {
            key: {
                "configured": value.get("configured"),
                "state": value.get("state"),
                "detail": value.get("detail"),
            }
            for key, value in connector_status.items()
            if isinstance(value, dict)
        },
        "kaggle_dpapi_readiness": summary.get("kaggle_dpapi_readiness"),
        "kaggle_new_competition_readiness": summary.get("kaggle_new_competition_readiness"),
    }


def normalize_figma_status(raw: str | None) -> dict[str, Any]:
    gate = read_json(FIGMA_GATE_JSON)
    if not raw and gate:
        return {
            "status": gate.get("status", "figma_gate_reported"),
            "note": gate.get("reason") or gate.get("claim_boundary") or "Figma gate artifact was loaded.",
            "blocked": bool(gate.get("blocked")),
            "verification_level": gate.get("verification_level"),
            "gate_report": str(FIGMA_GATE_JSON.relative_to(ROOT)).replace("\\", "/"),
        }
    if not raw:
        return {
            "status": "not_checked_by_script",
            "note": "Figma MCP requires plugin auth and is checked by the assistant/tool layer.",
            "blocked": True,
        }
    if "token_revoked" in raw or "401" in raw:
        return {
            "status": "blocked_figma_auth",
            "note": "Figma MCP returned HTTP 401 token_revoked; design-node verification cannot be claimed.",
            "evidence": raw[-1000:],
            "blocked": True,
        }
    lowered = raw.lower()
    if (
        "don't have edit access" in lowered
        or "do not have edit access" in lowered
        or "no edit access" in lowered
        or "permission" in lowered
        or "invalid_argument" in lowered
    ):
        return {
            "status": "blocked_figma_access",
            "note": "Figma MCP could not read the target file/node because the connector account lacks sufficient edit access.",
            "evidence": raw[-1000:],
            "blocked": True,
        }
    return {"status": "reported", "note": raw[-1000:], "blocked": False}


def write_figma_gate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.figma_status and FIGMA_GATE_JSON.exists():
        gate = read_json(FIGMA_GATE_JSON)
        return {
            "id": "figma_design_gate",
            "kind": "script",
            "critical": False,
            "claim": "Figma design verification gate is recorded as verified or explicitly blocked.",
            "status": "passed",
            "returncode": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "command": "read existing workspace/workstation_figma_design_gate_20260630.json",
            "stdout_tail": json.dumps({
                "status": gate.get("status"),
                "verification_level": gate.get("verification_level"),
                "blocked": gate.get("blocked"),
                "blocker": gate.get("blocker"),
            }, ensure_ascii=False),
            "stderr_tail": "",
        }
    cmd = [
        sys.executable,
        "scripts/verify_figma_design_gate.py",
        "--write-report",
    ]
    if args.figma_status:
        cmd.extend(["--probe-status", args.figma_status])
    result = run_command(
        {
            "id": "figma_design_gate",
            "kind": "script",
            "critical": False,
            "cmd": cmd,
            "claim": "Figma design verification gate is recorded as verified or explicitly blocked.",
        },
        timeout=60,
    )
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if args.include_frontend:
        checks.append(run_frontend("typecheck", timeout=240))
    if args.include_build:
        checks.append(run_frontend("build", timeout=360))
    checks.append(write_figma_gate(args))
    for item in COMMANDS:
        checks.append(run_command(item, timeout=args.command_timeout))

    page_smoke = [http_check(args.base_url, f"/?page={page}") for page in PAGES]
    api_smoke = [http_check(args.base_url, path) for path in API_PATHS]
    failed_http = [item for item in [*page_smoke, *api_smoke] if not item["ok"]]

    critical_failures = [
        item for item in checks
        if item.get("critical") and item.get("status") != "passed"
    ]
    soft_failures = [
        item for item in checks
        if not item.get("critical") and item.get("status") != "passed"
    ]

    figma = normalize_figma_status(args.figma_status)
    resources = resource_snapshot(args.base_url)
    inventory = summarize_inventory()
    learning_loop = (inventory.get("learning_loop") or {}) if isinstance(inventory, dict) else {}

    blockers: list[str] = []
    if critical_failures:
        blockers.append("critical_check_failed")
    if failed_http:
        blockers.append("page_or_api_smoke_failed")
    if figma.get("blocked") or figma["status"] == "blocked_figma_auth":
        blockers.append("figma_auth_blocked")
    connectors = resources.get("connectors") if isinstance(resources, dict) else {}
    gpu = connectors.get("gpu") if isinstance(connectors, dict) else None
    if isinstance(gpu, dict) and "blocked" in str(gpu.get("state", "")).lower():
        blockers.append("gpu_resource_blocked")
    if "deepseek_cache_below_80_for_batch_generation" in (learning_loop.get("resource_blockers") or []):
        blockers.append("deepseek_cache_below_80_for_batch_generation")

    status = "passed" if not critical_failures and not failed_http else "failed"
    launch_state = "ready_for_local_demo"
    if status != "passed":
        launch_state = "not_ready"
    elif "gpu_resource_blocked" in blockers:
        launch_state = "demo_ready_training_blocked_by_gpu"
    elif "deepseek_cache_below_80_for_batch_generation" in blockers:
        launch_state = "demo_ready_batch_code_generation_blocked_by_cache"
    elif "figma_auth_blocked" in blockers:
        launch_state = "demo_ready_figma_blocked"

    return {
        "schema": "academic_research_os.workstation_launch_readiness.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "status": status,
        "launch_state": launch_state,
        "blockers": blockers,
        "critical_failures": [item["id"] for item in critical_failures],
        "soft_failures": [item["id"] for item in soft_failures],
        "checks": checks,
        "page_smoke": page_smoke,
        "api_smoke": api_smoke,
        "figma": figma,
        "resources": resources,
        "experiment_summary": inventory,
        "claim_boundary": (
            "This report proves workstation launch/readiness signals only. "
            "It does not prove new Kaggle training, official submission, rank, medal, or MLE-Bench parity."
        ),
    }


def write_markdown_legacy(report: dict[str, Any]) -> None:
    summary = report["experiment_summary"]
    inv = summary.get("inventory") or {}
    lb = summary.get("leaderboard") or {}
    resources = report.get("resources") or {}
    connectors = resources.get("connectors") if isinstance(resources, dict) else {}

    lines = [
        "# AI 科研工作站上线 Readiness 总检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 总状态：`{report['status']}`",
        f"- 上线状态：`{report['launch_state']}`",
        f"- 阻断项：`{', '.join(report['blockers']) if report['blockers'] else 'none'}`",
        "",
        "## 结论",
        "",
    ]
    if report["status"] == "passed":
        lines.append("本地工作站页面、只读 API、组件 action 接线、安全扫描、Kaggle/DeepSeek/GPU 门禁脚本和实验统计链路均可复验。训练是否可继续取决于当前 GPU/HPC 资源门禁；Figma 高保真核验取决于 Figma OAuth 状态。")
    else:
        lines.append("上线总检查仍有关键失败项，不能声明系统完全打通。请先处理下方关键失败。")
    lines.extend([
        "",
        "## 关键检查",
        "",
        "| 检查 | 状态 | 关键 | 说明 |",
        "| --- | --- | --- | --- |",
    ])
    for item in report["checks"]:
        lines.append(f"| `{item['id']}` | `{item['status']}` | `{item['critical']}` | {item['claim']} |")

    lines.extend([
        "",
        "## 页面与 API Smoke",
        "",
        f"- 页面入口：`{sum(1 for item in report['page_smoke'] if item['ok'])}/{len(report['page_smoke'])}` 通过",
        f"- API 入口：`{sum(1 for item in report['api_smoke'] if item['ok'])}/{len(report['api_smoke'])}` 通过",
        "",
        "## 外部资源状态",
        "",
    ])
    if isinstance(connectors, dict) and connectors:
        for key, value in connectors.items():
            lines.append(f"- `{key}`：configured=`{value.get('configured')}`，state=`{value.get('state')}`")
    else:
        lines.append("- 未能从 `/api/workstation-summary` 读取 connector_status。")

    lines.extend([
        "",
        "## Figma 状态",
        "",
        f"- 状态：`{report['figma']['status']}`",
        f"- 说明：{report['figma']['note']}",
        "",
        "## 实验与榜单统计",
        "",
        f"- 任务数：`{inv.get('task_count_with_experiments')}`",
        f"- 观测 run：`{inv.get('total_runs_observed')}`",
        f"- 有分数 run：`{inv.get('total_scored_runs')}`",
        f"- promoted / held / timeout-or-failed：`{inv.get('total_promoted_runs')}` / `{inv.get('total_held_runs')}` / `{inv.get('total_timeout_or_failed_runs')}`",
        f"- Kaggle10 状态：`{inv.get('kaggle10_completion_status')}`",
        f"- 官方提交任务：`{lb.get('official_submission_tasks')}`",
        f"- 官方 top30 任务：`{lb.get('official_top30_count')}`",
        f"- medal count / medal rate：`{lb.get('medal_count')}` / `{lb.get('medal_rate')}`",
        f"- benchmark claim：`{lb.get('benchmark_claim_status')}`",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## 下一步",
        "",
        "1. 若要继续训练，先让 GPU/HPC gate 通过 `/api/gpu/connections/test`，并保持远端专用目录策略。",
        "2. 若要继续 Figma 高保真还原，先重新授权 Figma，再读取真实 node metadata 和 screenshot。",
        "3. 若要向老师汇报奖牌率，只能使用官方 Kaggle response 和 medal evidence；当前不能把 proxy/CV 当作奖牌。",
        "4. DeepSeek 批量 Code Agent 前继续复验缓存命中率，低于 80% 时先修缓存再批量生成。",
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(report: dict[str, Any]) -> None:
    summary = report["experiment_summary"]
    inv = summary.get("inventory") or {}
    lb = summary.get("leaderboard") or {}
    matrix = summary.get("task_api_matrix") or {}
    runtime = summary.get("runtime_navigation") or {}
    ui_actions = summary.get("ui_action_contract") or {}
    learning = summary.get("learning_loop") or {}
    server = summary.get("server_health") or {}
    resources = report.get("resources") or {}
    connectors = resources.get("connectors") if isinstance(resources, dict) else {}

    lines = [
        "# AI 科研工作站上线 Readiness 总检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 总状态：`{report['status']}`",
        f"- 上线状态：`{report['launch_state']}`",
        f"- 阻断项：`{', '.join(report['blockers']) if report['blockers'] else 'none'}`",
        "",
        "## 结论",
        "",
    ]
    if report["status"] == "passed":
        lines.append("本地工作站页面、只读 API、任务级闭环 API、组件 action 接线、安全扫描、Kaggle/DeepSeek/GPU 门禁脚本和实验统计链路均可复验。训练是否可继续取决于当前 GPU/HPC 资源门禁；Figma 高保真核验取决于 Figma OAuth 状态。")
    else:
        lines.append("上线总检查仍有关键失败项，不能声明系统完全打通。请先处理下方关键失败。")

    lines.extend([
        "",
        "## 关键检查",
        "",
        "| 检查 | 状态 | 关键 | 说明 |",
        "| --- | --- | --- | --- |",
    ])
    for item in report["checks"]:
        lines.append(f"| `{item['id']}` | `{item['status']}` | `{item['critical']}` | {item['claim']} |")

    lines.extend([
        "",
        "## 页面与 API Smoke",
        "",
        f"- 页面入口：`{sum(1 for item in report['page_smoke'] if item['ok'])}/{len(report['page_smoke'])}` 通过",
        f"- API 入口：`{sum(1 for item in report['api_smoke'] if item['ok'])}/{len(report['api_smoke'])}` 通过",
        f"- 服务健康：`{server.get('status')}`",
        f"- 服务监听进程：`{', '.join(str(pid) for pid in (server.get('port_pids') or [])) or 'none'}`",
        f"- 服务健康失败项：`{', '.join(server.get('failure_reasons') or []) or 'none'}`",
        f"- 服务健康警告：`{', '.join(server.get('warnings') or []) or 'none'}`",
        f"- CSS / summary API：`{server.get('css_ok')}` / `{server.get('summary_ok')}`",
        f"- 运行时导航矩阵：`{runtime.get('status')}`",
        f"- 运行时页面 / parser / render：`{runtime.get('nav_page_count')}` / `{runtime.get('parser_page_count')}` / `{runtime.get('rendered_page_count')}`",
        f"- 运行时 API 检查数：`{runtime.get('api_path_count')}`",
        f"- 运行时失败页面：`{len(runtime.get('failed_pages') or [])}`",
        f"- 运行时失败 API：`{len(runtime.get('failed_apis') or [])}`",
        f"- UI action 合约：`{ui_actions.get('status')}`",
        f"- data-ui-action / 直接业务 action：`{ui_actions.get('data_ui_action_count')}` / `{ui_actions.get('direct_frontend_action_count')}`",
        f"- 后端显式 action case：`{ui_actions.get('backend_case_count')}`",
        f"- 仅审计记录的 UI action：`{ui_actions.get('telemetry_only_data_ui_action_count')}`",
        f"- UI action 失败项：`{len(ui_actions.get('failed_direct_actions') or []) + len(ui_actions.get('failed_data_ui_actions') or []) + len(ui_actions.get('failed_live_posts') or [])}`",
        "",
        "## 任务闭环 API",
        "",
        f"- 任务 API 矩阵状态：`{matrix.get('status')}`",
        f"- 任务数：`{matrix.get('task_count')}`",
        f"- 最小闭环可见任务：`{matrix.get('minimum_closed_loop_visible_count')}`",
        f"- 完整可汇报闭环任务：`{matrix.get('full_reportable_loop_visible_count')}`",
        f"- evidence 缺失任务：`{', '.join(matrix.get('evidence_missing_tasks') or []) or 'none'}`",
        f"- report 缺失任务：`{', '.join(matrix.get('report_missing_tasks') or []) or 'none'}`",
        "",
        "## 自动化学习闭环",
        "",
        f"- 学习闭环状态：`{learning.get('status')}`",
        f"- retrospective memory 记录：`{learning.get('memory_records')}`",
        f"- search/evolution order 记录：`{learning.get('search_order_records')}`",
        f"- observed runs：`{learning.get('observed_runs')}`",
        f"- 下一轮队列可启动：`{learning.get('next_run_ready')}`",
        f"- 学习闭环失败项：`{', '.join(learning.get('failures') or []) or 'none'}`",
        f"- 学习闭环资源阻断：`{', '.join(learning.get('resource_blockers') or []) or 'none'}`",
        "",
        "## 外部资源状态",
        "",
    ])
    if isinstance(connectors, dict) and connectors:
        for key, value in connectors.items():
            lines.append(f"- `{key}`：configured=`{value.get('configured')}`，state=`{value.get('state')}`")
    else:
        lines.append("- 未能从 `/api/workstation-summary` 读取 connector_status。")

    lines.extend([
        "",
        "## Figma 状态",
        "",
        f"- 状态：`{report['figma']['status']}`",
        f"- 说明：{report['figma']['note']}",
        "",
        "## 实验与榜单统计",
        "",
        f"- 有实验任务数：`{inv.get('task_count_with_experiments')}`",
        f"- 观测 run：`{inv.get('total_runs_observed')}`",
        f"- 有分数 run：`{inv.get('total_scored_runs')}`",
        f"- promoted / held / timeout-or-failed：`{inv.get('total_promoted_runs')}` / `{inv.get('total_held_runs')}` / `{inv.get('total_timeout_or_failed_runs')}`",
        f"- Kaggle10 状态：`{inv.get('kaggle10_completion_status')}`",
        f"- 官方提交任务：`{lb.get('official_submission_tasks')}`",
        f"- 官方 top30 任务：`{lb.get('official_top30_count')}`",
        f"- medal count / medal rate：`{lb.get('medal_count')}` / `{lb.get('medal_rate')}`",
        f"- benchmark claim：`{lb.get('benchmark_claim_status')}`",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## 下一步",
        "",
        "1. 若要继续训练，先让 GPU/HPC gate 通过 `/api/gpu/connections/test`，并保持远端专用目录策略。",
        "2. 若要继续 Figma 高保真还原，先重新授权 Figma，再读取真实 node metadata 和 screenshot。",
        "3. 若要向老师汇报奖牌率，只能使用官方 Kaggle response 和 medal evidence；当前不能把 proxy/CV 当作奖牌。",
        "4. DeepSeek 批量 Code Agent 前继续复验缓存命中率，低于 80% 时先修缓存再批量生成。",
        "5. 优先补齐 evidence/report 缺失任务，把更多任务从最小闭环升级到完整可汇报闭环。",
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_clean(report: dict[str, Any]) -> None:
    summary = report["experiment_summary"]
    inv = summary.get("inventory") or {}
    lb = summary.get("leaderboard") or {}
    matrix = summary.get("task_api_matrix") or {}
    resources = report.get("resources") or {}
    connectors = resources.get("connectors") if isinstance(resources, dict) else {}

    lines = [
        "# AI 科研工作站上线 Readiness 总检查",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 工作站地址：`{report['base_url']}`",
        f"- 总状态：`{report['status']}`",
        f"- 上线状态：`{report['launch_state']}`",
        f"- 阻断项：`{', '.join(report['blockers']) if report['blockers'] else 'none'}`",
        "",
        "## 结论",
        "",
    ]
    if report["status"] == "passed":
        lines.append(
            "本地工作站页面、只读 API、任务级闭环 API、组件 action 接线、安全扫描、"
            "Kaggle/DeepSeek/GPU 门禁脚本和实验统计链路均可复验。"
            "是否可以继续训练取决于当前 GPU/HPC resource gate；Figma 高保真核验取决于 Figma OAuth 状态。"
        )
    else:
        lines.append("上线总检查仍有关键失败项，不能声明系统已经完全打通。")

    lines.extend([
        "",
        "## 关键检查",
        "",
        "| 检查 | 状态 | 关键 | 说明 |",
        "| --- | --- | --- | --- |",
    ])
    for item in report["checks"]:
        lines.append(f"| `{item['id']}` | `{item['status']}` | `{item['critical']}` | {item['claim']} |")

    lines.extend([
        "",
        "## 页面与 API Smoke",
        "",
        f"- 页面入口：`{sum(1 for item in report['page_smoke'] if item['ok'])}/{len(report['page_smoke'])}` 通过",
        f"- API 入口：`{sum(1 for item in report['api_smoke'] if item['ok'])}/{len(report['api_smoke'])}` 通过",
        "",
        "## 任务闭环 API",
        "",
        f"- 任务 API 矩阵状态：`{matrix.get('status')}`",
        f"- 任务数：`{matrix.get('task_count')}`",
        f"- 最小闭环可见任务：`{matrix.get('minimum_closed_loop_visible_count')}`",
        f"- 完整可汇报闭环任务：`{matrix.get('full_reportable_loop_visible_count')}`",
        f"- evidence 缺失任务：`{', '.join(matrix.get('evidence_missing_tasks') or []) or 'none'}`",
        f"- report 缺失任务：`{', '.join(matrix.get('report_missing_tasks') or []) or 'none'}`",
        "",
        "## 外部资源状态",
        "",
    ])
    if isinstance(connectors, dict) and connectors:
        for key, value in connectors.items():
            lines.append(f"- `{key}`：configured=`{value.get('configured')}`，state=`{value.get('state')}`")
    else:
        lines.append("- 未能从 `/api/workstation-summary` 读取 connector_status。")

    lines.extend([
        "",
        "## Figma 状态",
        "",
        f"- 状态：`{report['figma']['status']}`",
        f"- 说明：{report['figma']['note']}",
        "",
        "## 实验与榜单统计",
        "",
        f"- 有实验任务数：`{inv.get('task_count_with_experiments')}`",
        f"- 观测 run：`{inv.get('total_runs_observed')}`",
        f"- 有分数 run：`{inv.get('total_scored_runs')}`",
        f"- promoted / held / timeout-or-failed：`{inv.get('total_promoted_runs')}` / `{inv.get('total_held_runs')}` / `{inv.get('total_timeout_or_failed_runs')}`",
        f"- Kaggle10 状态：`{inv.get('kaggle10_completion_status')}`",
        f"- 官方提交任务：`{lb.get('official_submission_tasks')}`",
        f"- 官方 top30 任务：`{lb.get('official_top30_count')}`",
        f"- medal count / medal rate：`{lb.get('medal_count')}` / `{lb.get('medal_rate')}`",
        f"- benchmark claim：`{lb.get('benchmark_claim_status')}`",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## 下一步",
        "",
        "1. 继续训练前，先让 GPU/HPC gate 通过 `/api/gpu/connections/test`，并保持远程专用目录策略。",
        "2. 继续 Figma 高保真还原前，先重新授权 Figma，再读取真实 node metadata 和 screenshot。",
        "3. 向老师汇报奖牌率时，只能使用官方 Kaggle response 和 medal evidence；不能把 proxy/CV 当作奖牌。",
        "4. DeepSeek 批量 Code Agent 前继续复验缓存命中率，低于 80% 时先修缓存再批量生成。",
        "5. 优先补齐 evidence/report 缺失任务，把更多任务从最小闭环升级到完整可汇报闭环。",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate launch readiness checks for the AI research workstation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--include-frontend", action="store_true", help="Run npm run typecheck before the live smoke checks.")
    parser.add_argument("--include-build", action="store_true", help="Run npm run build. Restart a running Next server before live smoke if build rewrites .next.")
    parser.add_argument("--command-timeout", type=int, default=240)
    parser.add_argument("--figma-status", default=None, help="Optional external Figma MCP auth probe result.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report(args)
    if args.write_report:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown_clean(report)

    print(json.dumps({
        "status": report["status"],
        "launch_state": report["launch_state"],
        "blockers": report["blockers"],
        "critical_failures": report["critical_failures"],
        "soft_failures": report["soft_failures"],
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/") if args.write_report else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
