#!/usr/bin/env python3
"""Verify a running production candidate without approving or dispatching work."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any], float]:
    base = base_url.rstrip("/")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = authenticated_headers(
        base,
        {
            "Accept": "application/json",
            "Origin": base,
            **({"Content-Type": "application/json"} if payload is not None else {}),
        },
    )
    request = urllib.request.Request(
        f"{base}{path}",
        data=payload,
        headers=headers,
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{method} {path} did not return JSON (HTTP {status})") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{method} {path} did not return a JSON object")
    return status, decoded, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18088")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default="production_candidate_runtime_smoke")
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    for name, path in (
        ("healthz", "/api/healthz"),
        ("system_version", "/api/system/version"),
        ("runtime_health", "/api/runtime/health"),
        ("connectors_health", "/api/connectors/health"),
        ("hpc_job_lineage", "/api/hpc/job-lineage"),
        ("workstation_summary_before", "/api/workstation-summary"),
    ):
        status, payload, elapsed_ms = request_json(args.base_url, "GET", path)
        checks[f"{name}_http_200"] = status == 200
        evidence[name] = {
            "http_status": status,
            "elapsed_ms": elapsed_ms,
            "payload": payload,
        }

    health = evidence["healthz"]["payload"]
    runtime_health = evidence["runtime_health"]["payload"]
    checks["dashboard_ready"] = health.get("ok") is True and health.get("status") == "ready"
    checks["runtime_ready"] = (
        runtime_health.get("status") == "ready"
        or runtime_health.get("ok") is True
        or runtime_health.get("reachable") is True
    )

    create_status, created, create_ms = request_json(
        args.base_url,
        "POST",
        "/api/workstation-actions",
        {
            "action": "create_workstation_run",
            "task_id": args.task_id,
            "metadata": {
                "trigger": "production_candidate_runtime_smoke",
                "objective": "Verify the production candidate stops at plan_approval without external execution.",
            },
        },
    )
    run_id = str(created.get("run_id") or "")
    evidence["create_workstation_run"] = {
        "http_status": create_status,
        "elapsed_ms": create_ms,
        "payload": created,
    }
    checks["run_created"] = (
        create_status == 200
        and created.get("ok") is True
        and bool(run_id)
        and created.get("status") == "WAIT_PLAN_GATE"
    )

    dispatch_status, dispatch, dispatch_ms = request_json(
        args.base_url,
        "POST",
        "/api/workstation-actions",
        {
            "action": "dispatch_task_agents",
            "task_id": args.task_id,
            "metadata": {
                "run_id": run_id,
                "objective": "Remain blocked at plan_approval during the production candidate smoke.",
            },
        },
    )
    evidence["dispatch_before_plan_gate"] = {
        "http_status": dispatch_status,
        "elapsed_ms": dispatch_ms,
        "payload": dispatch,
    }
    checks["plan_gate_blocks_execution"] = (
        dispatch_status == 200
        and dispatch.get("ok") is True
        and dispatch.get("run_id") == run_id
        and dispatch.get("execution_started") is False
        and dispatch.get("required_gate") == "plan_approval"
        and dispatch.get("status") == "WAIT_PLAN_GATE"
    )

    summary_deadline = time.monotonic() + 5
    summary_attempts = 0
    summary_status = 0
    summary_ms = 0.0
    runs: list[Any] = []
    projected_run: dict[str, Any] | None = None
    while time.monotonic() < summary_deadline:
        summary_attempts += 1
        summary_status, summary, elapsed_ms = request_json(
            args.base_url,
            "GET",
            "/api/workstation-summary",
        )
        summary_ms += elapsed_ms
        runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
        projected_run = next(
            (item for item in runs if isinstance(item, dict) and item.get("id") == run_id),
            None,
        )
        if projected_run is not None:
            break
        time.sleep(0.25)
    evidence["workstation_summary_after"] = {
        "http_status": summary_status,
        "request_elapsed_ms": round(summary_ms, 3),
        "attempts": summary_attempts,
        "run_projection": projected_run,
        "run_count": len(runs),
    }
    checks["summary_projects_waiting_run"] = (
        summary_status == 200
        and isinstance(projected_run, dict)
        and projected_run.get("status") == "WAIT_PLAN_GATE"
    )

    passed = all(checks.values())
    report = {
        "schema": "evomind.production_candidate.runtime_smoke.v1",
        "status": "passed" if passed else "failed",
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": args.base_url.rstrip("/"),
        "task_id": args.task_id,
        "run_id": run_id,
        "external_execution_requested": False,
        "gate_approval_performed": False,
        "checks": checks,
        "evidence": evidence,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "run_id": run_id,
        "checks": checks,
        "artifact": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
