#!/usr/bin/env python3
"""Verify the live workstation rejects a generic HPC task without a contract."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from workstation_local_auth import authenticated_headers  # noqa: E402


def request_json(
    base: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = authenticated_headers(
        base,
        {"Accept": "application/json", "Origin": base.rstrip("/")},
    )
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"error": raw}


def action(
    base: str,
    name: str,
    task_id: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    return request_json(
        base,
        "/api/workstation-actions",
        {"action": name, "task_id": task_id, "metadata": metadata or {}},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    create_status, created = action(base, "create_workstation_run", "titanic", {
        "trigger": f"live_hpc_contract_guard_{stamp}",
        "objective": (
            "请使用本地 Titanic 数据，在 HPC 上训练候选并生成证据；"
            "不要使用本地 GPU，不提交 Kaggle。"
        ),
    })
    run_id = str(created.get("run_id") or "")
    if create_status != 200 or not run_id:
        raise RuntimeError(f"guard run creation failed: {create_status}")

    dispatch_status, rejected = action(
        base,
        "tasks_dispatch_agents",
        "titanic",
        {"run_id": run_id},
    )
    _, summary = request_json(base, "/api/workstation-summary")
    run = next(
        (item for item in summary.get("runs", []) if item.get("id") == run_id),
        None,
    )
    python_run_dir = ROOT / "workspace" / "evomind_runs" / run_id
    expected_fields = {
        "job_id",
        "credential_profile",
        "resource_profile",
        "execution_backend",
    }
    checks = {
        "create_http_200": create_status == 200,
        "dispatch_http_422": dispatch_status == 422,
        "error_code": rejected.get("code") == "missing_hpc_execution_contract",
        "error_message": rejected.get("error") == "Missing HPC execution contract",
        "missing_fields_complete": set(rejected.get("missing_fields") or []) == expected_fields,
        "run_remains_waiting": (run or {}).get("status") == "WAIT_PLAN_GATE",
        "execution_directory_absent": not python_run_dir.exists(),
    }
    passed = all(checks.values())
    report = {
        "schema": "evomind.live_hpc_contract_guard.v1",
        "status": "passed" if passed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "checks": checks,
        "http": {"create": create_status, "dispatch": dispatch_status},
        "response": rejected,
        "run_status": (run or {}).get("status"),
        "execution_started": python_run_dir.exists(),
        "mock_used": False,
    }
    output = (
        ROOT
        / "workspace"
        / "verification"
        / "production_recovery"
        / "hpc_contract_guard"
        / f"guard_{stamp}"
        / "report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "run_id": run_id,
        "artifact": str(output.relative_to(ROOT)).replace("\\", "/"),
        "checks": checks,
        "mock_used": False,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
