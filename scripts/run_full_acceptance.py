from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path(os.environ.get("TEMP", "/tmp")) / "research_agent_pycache"))
    env.setdefault("RESEARCH_INTEGRITY_GATE_PATH", str(Path(os.environ.get("TEMP", "/tmp")) / "research_integrity_gate.json"))
    env.setdefault("RESEARCH_AGENT_READ_ONLY_ACCEPTANCE", "1")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env)
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def require_success(result: dict) -> None:
    if result["returncode"] != 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "failed_command": result["command"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def check_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        content = response.read().decode("utf-8")
    return {"url": url, "status": "reachable", "content_excerpt": content[:500]}


def check_first_reachable_url(urls: list[str]) -> dict:
    failures = []
    for url in urls:
        try:
            return check_url(url)
        except urllib.error.HTTPError as error:
            failures.append({"url": url, "status": error.code})
    raise SystemExit(
        json.dumps(
            {
                "status": "failed",
                "failed_command": "url reachability",
                "stdout": "",
                "stderr": json.dumps({"failures": failures}, ensure_ascii=False),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def check_visual_acceptance_artifacts() -> dict:
    doc_path = ROOT / "docs" / "可视化验收记录.md"
    desktop_path = ROOT / "docs" / "visual_acceptance_desktop.png"
    mobile_path = ROOT / "docs" / "visual_acceptance_mobile.png"
    required_paths = [doc_path, desktop_path, mobile_path]
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    if missing:
        return {
            "command": "visual acceptance artifacts",
            "returncode": 1,
            "stdout": "",
            "stderr": f"missing visual acceptance artifacts: {missing}",
        }

    small_images = [
        str(path.relative_to(ROOT))
        for path in [desktop_path, mobile_path]
        if path.stat().st_size < 10_000
    ]
    doc_text = doc_path.read_text(encoding="utf-8")
    required_terms = ["科研 Agent 工作站", "titanic", "house_prices", "研究依据", "完整性检查", "长期路线图"]
    missing_terms = [term for term in required_terms if term not in doc_text]
    if small_images or missing_terms:
        return {
            "command": "visual acceptance artifacts",
            "returncode": 1,
            "stdout": "",
            "stderr": json.dumps(
                {"small_images": small_images, "missing_terms": missing_terms},
                ensure_ascii=False,
            ),
        }

    return {
        "command": "visual acceptance artifacts",
        "returncode": 0,
        "stdout": json.dumps(
            {
                "doc": str(doc_path.relative_to(ROOT)),
                "desktop_png_bytes": desktop_path.stat().st_size,
                "mobile_png_bytes": mobile_path.stat().st_size,
            },
            ensure_ascii=False,
        ),
        "stderr": "",
    }


TITANIC_REQUIRED_FILES = [
    "experiment_log.json",
    "data_quality.json",
    "model_results.json",
    "titanic_local_report.md",
    "titanic_local_report.docx",
    "task_scaffold.json",
    "task_scaffold.md",
    "workflow_stage_audit.json",
    "workflow_stage_audit.md",
]

TABULAR_REQUIRED_FILES = [
    "experiment_log.json",
    "data_quality.json",
    "model_results.json",
    "submission.csv",
    "task_scaffold.json",
    "task_scaffold.md",
    "post_scaffold_improvement.json",
    "post_scaffold_improvement.md",
    "workflow_stage_audit.json",
    "workflow_stage_audit.md",
    "local_report.md",
    "local_report.docx",
]


def _has_required_files(run_dir: Path, required_files: list[str]) -> bool:
    return all((run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in required_files)


def latest_experiment(task_id: str, required_files: list[str] | None = None) -> str:
    task_root = ROOT / "experiments" / task_id
    runs = sorted(path for path in task_root.iterdir() if path.is_dir())
    if not runs:
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "failed_command": f"latest experiment lookup for {task_id}",
                    "stdout": "",
                    "stderr": f"no experiment runs found under {task_root}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if required_files:
        for run_dir in reversed(runs):
            if _has_required_files(run_dir, required_files):
                return str(run_dir.relative_to(ROOT))
        raise SystemExit(
            json.dumps(
                {
                    "status": "failed",
                    "failed_command": f"latest complete experiment lookup for {task_id}",
                    "stdout": "",
                    "stderr": (
                        f"no complete experiment run found under {task_root}; "
                        f"required_files={required_files}"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return str(runs[-1].relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full local acceptance suite for the research agent workstation.")
    parser.add_argument("--dashboard-url", default=None, help="Optional running dashboard URL, for example http://127.0.0.1:8088")
    parser.add_argument("--container-name", default=None, help="Optional container name used to verify Docker-written artifacts.")
    parser.add_argument(
        "--skip-verified-launch-audit",
        action="store_true",
        help="Skip the prior launch-audit check when the verified launcher is producing that audit in this run.",
    )
    args = parser.parse_args()

    checks = []
    commands = [
        [sys.executable, "scripts/validate_titanic_experiment.py", "--experiment-dir", latest_experiment("titanic", TITANIC_REQUIRED_FILES), "--config", "configs/titanic.yaml"],
        [sys.executable, "scripts/validate_tabular_experiment.py", "--experiment-dir", latest_experiment("house_prices", TABULAR_REQUIRED_FILES), "--config", "configs/house_prices.yaml"],
        [sys.executable, "scripts/validate_tabular_experiment.py", "--experiment-dir", latest_experiment("telco_churn", TABULAR_REQUIRED_FILES), "--config", "configs/telco_churn.yaml"],
        [sys.executable, "scripts/verify_research_sources.py"],
        [sys.executable, "scripts/verify_research_integrity.py"],
        [sys.executable, "scripts/verify_dashboard.py"],
        [sys.executable, "scripts/verify_runtime_completeness.py", "--tasks", "titanic", "house_prices", "telco_churn"],
        [sys.executable, "scripts/verify_code_agent_patch_lifecycle.py", "--task-id", "house_prices"],
        [sys.executable, "scripts/verify_launch_resource_readiness.py"],
        [sys.executable, "scripts/verify_launch_integration_hardening.py"],
        [sys.executable, "scripts/verify_external_resources_manifest.py"],
        [sys.executable, "scripts/verify_hpc_windows_proxy_prereqs.py"],
        [sys.executable, "scripts/verify_training_optimization_readiness.py"],
        [sys.executable, "scripts/verify_kaggle_new_competition_readiness.py"],
        [sys.executable, "scripts/verify_kaggle_dpapi_readiness.py"],
        [sys.executable, "scripts/verify_gpu_job_templates.py"],
        [sys.executable, "scripts/verify_ui_localization_contract.py"],
        [sys.executable, "scripts/verify_ui_layout_quality.py"],
        [sys.executable, "scripts/verify_scientific_ui_polish.py"],
        [sys.executable, "scripts/verify_dashboard_manager.py"],
        [sys.executable, "scripts/verify_docker_service_naming.py"],
        [sys.executable, "scripts/verify_local_startup_contract.py"],
        [sys.executable, "scripts/verify_chrome_acceptance_record.py"],
        [sys.executable, "scripts/verify_resource_activation_runbook.py"],
        [sys.executable, "scripts/verify_no_plaintext_secrets.py"],
        [
            sys.executable, "-m", "compileall",
            "-x", r"scripts[\\/]_quarantine[\\/].*",
            "src", "scripts",
        ],
    ]
    if not args.skip_verified_launch_audit:
        commands.append([sys.executable, "scripts/verify_verified_workstation_launch_audit.py"])
    if args.dashboard_url:
        commands.append([sys.executable, "scripts/verify_dashboard.py", "--url", args.dashboard_url])
        commands.append([sys.executable, "scripts/verify_ui_localization_contract.py", "--url", args.dashboard_url])
        commands.append([sys.executable, "scripts/verify_gpu_job_templates.py", "--url", args.dashboard_url])
        commands.append([sys.executable, "scripts/verify_backend_resource_status.py", "--url", args.dashboard_url])
        commands.append([sys.executable, "scripts/verify_launch_go_no_go.py", "--dashboard-url", args.dashboard_url])
        commands.append([sys.executable, "scripts/verify_deepseek_provider.py", "--url", args.dashboard_url])
        external_gateway_command = [sys.executable, "scripts/verify_external_resource_gateways.py", "--url", args.dashboard_url]
        if args.container_name:
            external_gateway_command.extend(["--container-name", args.container_name])
        commands.append(external_gateway_command)
        final_blocker_command = [sys.executable, "scripts/verify_final_two_resource_blockers.py", "--dashboard-url", args.dashboard_url]
        if args.container_name:
            final_blocker_command.extend(["--container-name", args.container_name])
        commands.append(final_blocker_command)
        action_contract_command = [sys.executable, "scripts/verify_workstation_action_contract.py", "--url", args.dashboard_url]
        if args.container_name:
            action_contract_command.extend(["--container-name", args.container_name])
        commands.append(action_contract_command)
        commands.append([sys.executable, "scripts/verify_report_figure_workflow.py", "--url", args.dashboard_url, "--task-id", "house_prices"])
        commands.append([sys.executable, "scripts/verify_final_delivery_status.py", "--url", args.dashboard_url, "--require-json"])
        commands.append([sys.executable, "scripts/verify_academic_os_page_deeplinks.py", "--url", args.dashboard_url])

    for command in commands:
        result = run_command(command)
        require_success(result)
        checks.append(result)

    visual_check = check_visual_acceptance_artifacts()
    require_success(visual_check)
    checks.append(visual_check)

    url_checks = []
    if args.dashboard_url:
        base = args.dashboard_url.rstrip("/")
        url_checks.append(check_first_reachable_url([f"{base}/health", f"{base}/api/workstation-summary"]))
        url_checks.append(check_url(args.dashboard_url))

    summary = {
        "status": "passed",
        "checks_run": len(checks),
        "dashboard_url": args.dashboard_url,
        "url_checks": url_checks,
        "commands": [check["command"] for check in checks],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
