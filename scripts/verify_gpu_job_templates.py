from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TEMPLATES = {
    "house_prices_baseline",
    "titanic_baseline",
    "telco_churn_baseline",
    "all_tasks_baseline",
    "house_prices_seed_sweep",
    "titanic_seed_sweep",
    "telco_churn_seed_sweep",
    "all_tasks_seed_sweep",
    "playground_s6e6_lgbm_optuna",
}


def fail(message: str) -> None:
    raise SystemExit(f"GPU_TEMPLATE_CHECK_FAILED: {message}")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run_dry_plan() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_gpu_training_batch.py",
            "--tasks",
            "house_prices",
            "titanic",
            "telco_churn",
            "--seeds",
            "42",
            "2026",
            "777",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        fail(f"dry-run batch failed: {completed.stderr}")
    return json.loads(completed.stdout)


def static_template_check() -> dict[str, Any]:
    source = (ROOT / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "gpu-ssh-gateway.ts").read_text(encoding="utf-8")
    missing = sorted(template for template in REQUIRED_TEMPLATES if template not in source)
    if missing:
        fail(f"missing GPU templates in gateway source: {missing}")
    if 'taskId === "telco_churn"' not in source or '"telco_churn_baseline"' not in source:
        fail("telco_churn default GPU template is not wired")
    if "gpu_job_template_rejected" not in source:
        fail("invalid GPU template rejection is not auditable")
    return {"required_templates": sorted(REQUIRED_TEMPLATES), "missing": missing}


def live_api_check(base_url: str | None) -> dict[str, Any] | None:
    if not base_url:
        return None
    base = base_url.rstrip("/")
    telco = post_json(f"{base}/api/gpu/jobs", {"task_id": "telco_churn"})
    if telco.get("status") == "not_configured" and telco.get("missing_env"):
        pass
    elif telco.get("status") == "rejected" and (
        "hpc_execution_approval" in str(telco.get("error", ""))
        or "Missing required GPU job fields" in str(telco.get("error", ""))
    ):
        pass
    elif telco.get("status") not in {"submitted", "failed"}:
        fail(f"unexpected telco GPU job response: {telco}")
    invalid = post_json(f"{base}/api/gpu/jobs", {"task_id": "house_prices", "template": "not_allowed_shell"})
    if invalid.get("status") != "rejected":
        fail(f"invalid GPU template was not rejected: {invalid}")
    allowed = set(invalid.get("allowed_templates") or [])
    if not REQUIRED_TEMPLATES.issubset(allowed):
        fail(f"invalid template response did not include full allowed template list: {invalid}")
    return {
        "telco_default_status": telco.get("status"),
        "telco_artifact": telco.get("artifact_path"),
        "invalid_status": invalid.get("status"),
        "invalid_artifact": invalid.get("artifact_path"),
        "allowed_templates": sorted(allowed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GPU SSH whitelist templates cover all tasks and optimization sweep paths.")
    parser.add_argument("--url", default=None, help="Optional dashboard URL for live API checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static = static_template_check()
    dry = run_dry_plan()
    if dry.get("command_count") != 9:
        fail(f"expected 9 dry-run commands for 3 tasks x 3 seeds, got {dry.get('command_count')}")
    if dry.get("status") != "planned":
        fail(f"dry-run manifest was not planned: {dry}")
    live = live_api_check(args.url)
    report = {
        "status": "passed",
        "static": static,
        "dry_run_manifest": dry.get("manifest_path"),
        "dry_run_command_count": dry.get("command_count"),
        "tasks": dry.get("tasks"),
        "seeds": dry.get("seeds"),
        "live_api": live,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
