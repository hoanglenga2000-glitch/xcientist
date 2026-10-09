"""Execute the five Production Recovery smoke cases against the live workstation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workstation_local_auth import authenticated_headers


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workspace" / "verification" / "production_recovery" / "production_smoke"
TERMINAL = {"WAIT_RESULT_GATE", "FAILED", "COMPLETED"}


def request_json(base: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = authenticated_headers(base, {"Accept": "application/json", "Origin": base.rstrip("/")})
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base.rstrip('/')}{path}", data=body, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"error": raw}


def action(base: str, name: str, task_id: str, metadata: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    return request_json(base, "/api/workstation-actions", {"action": name, "task_id": task_id, "metadata": metadata or {}})


def find_run(summary: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    return next((run for run in summary.get("runs", []) if run.get("id") == run_id), None)


def poll_run(base: str, run_id: str, timeout_seconds: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    observations: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        status, summary = request_json(base, "/api/workstation-summary")
        run = find_run(summary, run_id) if status == 200 else None
        observation = {"at": datetime.now(timezone.utc).isoformat(), "http_status": status, "run": run}
        if not observations or observations[-1].get("run", {}).get("status") != (run or {}).get("status"):
            observations.append(observation)
        if run and str(run.get("status")) in TERMINAL:
            return run, observations
        time.sleep(3)
    return None, observations


def poll_run_until_status(
    base: str,
    run_id: str,
    expected_status: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    observations: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        status, summary = request_json(base, "/api/workstation-summary")
        run = find_run(summary, run_id) if status == 200 else None
        observation = {
            "at": datetime.now(timezone.utc).isoformat(),
            "http_status": status,
            "run": run,
        }
        if not observations or observations[-1].get("run", {}).get("status") != (run or {}).get("status"):
            observations.append(observation)
        if run and str(run.get("status")) == expected_status:
            return run, observations
        time.sleep(1)
    return None, observations


def run_verifier(command: list[str], timeout: int) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    stdout = (completed.stdout or "").strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        last = stdout.splitlines()[-1:] or [""]
        try:
            payload = json.loads(last[0])
        except json.JSONDecodeError:
            payload = None
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "payload": payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--hpc-job-id", type=int, required=True)
    parser.add_argument("--hpc-credential-profile", required=True)
    parser.add_argument("--hpc-resource-profile", required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    created_at = datetime.now(timezone.utc)
    smoke_id = f"production_smoke_{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    out = OUT / smoke_id
    out.mkdir(parents=True, exist_ok=False)
    task_id = "titanic"
    hpc_contract = {
        "job_id": args.hpc_job_id,
        "credential_profile": args.hpc_credential_profile,
        "resource_profile": args.hpc_resource_profile,
        "execution_backend": "hpc",
    }

    create_status, created = action(base, "create_workstation_run", task_id, {
        "trigger": smoke_id,
        "objective": "请用本地已有的 Titanic 数据完成一个小型二分类模型开发任务：自动检查数据、提出并比较方案、在 HPC 上训练、独立审核、生成候选 submission 和研究报告；不要使用本地 GPU，也不要提交 Kaggle。",
        "hpc_execution_contract": hpc_contract,
    })
    run_id = str(created.get("run_id") or "")
    if create_status != 200 or not run_id:
        raise RuntimeError(f"fresh task creation failed: {create_status} {created}")

    blocked_status, blocked = action(base, "tasks_dispatch_agents", task_id, {"run_id": run_id})
    python_dir = ROOT / "workspace" / "evomind_runs" / run_id
    governance = {
        "status": "passed" if blocked_status == 200 and blocked.get("execution_started") is False and not python_dir.exists() else "failed",
        "http_status": blocked_status,
        "response": blocked,
        "python_run_dir_created_before_gate": python_dir.exists(),
    }

    gate_id = blocked.get("gate_id")
    approve_status, approved = action(base, "approve_gate", task_id, {
        "run_id": run_id,
        "gate_id": gate_id,
        "gate_type": "plan_approval",
    })
    dispatch_status, dispatched = action(base, "tasks_dispatch_agents", task_id, {"run_id": run_id})
    terminal_run, observations = poll_run(base, run_id, args.timeout_seconds)
    fresh_status = str((terminal_run or {}).get("status") or "timeout")
    fresh_passed = fresh_status == "WAIT_RESULT_GATE"

    result_gate: dict[str, Any] | None = None
    final_gate: dict[str, Any] | None = None
    if fresh_passed:
        result_http, result_gate = action(base, "approve_gate", task_id, {"run_id": run_id, "gate_type": "result_approval"})
        final_http, final_gate = action(base, "approve_gate", task_id, {"run_id": run_id, "gate_type": "final_report_approval"})
        terminal_run, final_observations = poll_run_until_status(
            base,
            run_id,
            "COMPLETED",
            timeout_seconds=60,
        )
        observations.extend(final_observations)
        fresh_passed = result_http == 200 and final_http == 200 and (terminal_run or {}).get("status") == "COMPLETED"

    failure = run_verifier(["uv", "run", "python", "scripts/verify_failure_recovery_contract.py"], 180)
    memory_evolution = run_verifier(["uv", "run", "--with", "pandas", "--with", "scikit-learn", "python", "scripts/verify_memory_evolution_contract.py"], 240)
    version_http, version = request_json(base, "/api/system/version")
    connector_http, connectors = request_json(base, "/api/connectors/health")
    lineage_http, lineage = request_json(base, "/api/hpc/job-lineage")

    tests = {
        "001_fresh_research_task": {
            "status": "passed" if fresh_passed else "failed",
            "run_id": run_id,
            "create_http": create_status,
            "approve_http": approve_status,
            "dispatch_http": dispatch_status,
            "dispatch": dispatched,
            "terminal_run": terminal_run,
            "observations": observations,
            "result_gate": result_gate,
            "final_gate": final_gate,
        },
        "002_failure_recovery": {
            "status": "passed" if failure["returncode"] == 0 and (failure.get("payload") or {}).get("mock_used") is False else "failed",
            "verifier": failure,
        },
        "003_memory": {
            "status": "passed" if memory_evolution["returncode"] == 0 and (memory_evolution.get("payload") or {}).get("mock_used") is False else "failed",
            "verifier": memory_evolution,
        },
        "004_evolution": {
            "status": "passed" if memory_evolution["returncode"] == 0 and bool((memory_evolution.get("payload") or {}).get("selected_strategy")) else "failed",
            "verifier_artifact": (memory_evolution.get("payload") or {}).get("artifact"),
        },
        "005_governance": governance,
    }
    report = {
        "schema": "evomind.production_recovery.smoke.v1",
        "smoke_id": smoke_id,
        "created_at": created_at.isoformat(),
        "base_url": base,
        "mock_used": False,
        "tests": tests,
        "runtime_version": {"http_status": version_http, "payload": version},
        "connector_registry": {"http_status": connector_http, "payload": connectors},
        "hpc_job_lineage": {"http_status": lineage_http, "payload": lineage},
    }
    passed = all(test.get("status") == "passed" for test in tests.values()) and version_http == connector_http == lineage_http == 200
    report["status"] = "passed" if passed else "failed"
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "mock_used": False, "artifact": str(report_path.relative_to(ROOT)).replace("\\", "/"), "run_id": run_id, "tests": {key: value["status"] for key, value in tests.items()}}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
