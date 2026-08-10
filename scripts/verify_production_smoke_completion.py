#!/usr/bin/env python3
"""Reconcile one completed Production Smoke run against canonical evidence."""

from __future__ import annotations

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def request_json(base: str, path: str) -> tuple[int, dict[str, Any]]:
    headers = authenticated_headers(
        base,
        {"Accept": "application/json", "Origin": base.rstrip("/")},
    )
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode("utf-8", errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    report_path = Path(args.report).expanduser().resolve()
    report_path.relative_to(ROOT.resolve())
    original = read_json(report_path)
    test_001 = original.get("tests", {}).get("001_fresh_research_task", {})
    run_id = str(test_001.get("run_id") or "")
    if not run_id:
        raise ValueError("Production Smoke report has no Fresh Task run_id")

    python_run_dir = ROOT / "workspace" / "evomind_runs" / run_id
    workstation_run_dir = ROOT / "workspace" / "workstation_runs" / "titanic" / run_id
    ledger_path = ROOT / "workspace" / "run_ledger" / "runs" / f"{run_id}.json"
    lineage_path = ROOT / "workspace" / "hpc_job_lineage" / "jobs" / "aimslab__91051.json"
    run = read_json(python_run_dir / "run.json")
    ledger = read_json(ledger_path)
    receipt = read_json(python_run_dir / "hpc_job_receipt.json")
    identity_path = python_run_dir / str(receipt.get("source_artifact") or "")
    lineage = read_json(lineage_path)
    dispatch = read_json(workstation_run_dir / "hpc_dispatch_contract.json")
    api_status, summary = request_json(args.base_url.rstrip("/"), "/api/workstation-summary")
    api_run = next((item for item in summary.get("runs", []) if item.get("id") == run_id), None)

    task_values = list((run.get("tasks") or {}).values())
    gates = {
        "result": test_001.get("result_gate") or {},
        "final": test_001.get("final_gate") or {},
    }
    checks = {
        "original_mock_free": original.get("mock_used") is False,
        "fresh_run_directory": python_run_dir.is_dir(),
        "python_supervisor_completed": run.get("status") == "completed",
        "all_14_nodes_completed": len(task_values) == 14 and all(
            isinstance(item, dict) and item.get("status") == "completed"
            for item in task_values
        ),
        "no_open_requirements": run.get("open_requirements") == [],
        "canonical_ledger_completed": (
            ledger.get("status") == "COMPLETED"
            and ledger.get("lifecycle_state") == "COMPLETED"
            and ledger.get("run_id") == run_id
        ),
        "api_projection_completed": (
            api_status == 200 and isinstance(api_run, dict) and api_run.get("status") == "COMPLETED"
        ),
        "result_gate_approved": (
            gates["result"].get("decision") == "approved"
            and gates["result"].get("run_status") == "REPORTING"
        ),
        "final_gate_approved": (
            gates["final"].get("decision") == "approved"
            and gates["final"].get("run_status") == "COMPLETED"
        ),
        "hpc_dispatch_contract_bound": (
            dispatch.get("run_id") == run_id
            and int(dispatch.get("job_id") or 0) == 91051
            and dispatch.get("credential_profile") == "job91051"
            and dispatch.get("execution_backend") == "hpc"
        ),
        "remote_receipt_verified": (
            receipt.get("schema") == "evomind.hpc_job_receipt.v1"
            and receipt.get("run_id") == run_id
            and int(receipt.get("job_id") or 0) == 91051
            and receipt.get("credential_profile") == "job91051"
            and receipt.get("verified") is True
        ),
        "receipt_identity_hash": (
            identity_path.is_file()
            and str(receipt.get("source_sha256") or "").lower() == sha256(identity_path).lower()
        ),
        "lineage_completed": (
            lineage.get("run_id") == run_id
            and lineage.get("job_id") == "91051"
            and lineage.get("status") == "COMPLETED"
            and lineage.get("remote_receipt_path")
            == f"workspace/evomind_runs/{run_id}/hpc_job_receipt.json"
        ),
        "research_report_present": (python_run_dir / "research_report.md").is_file(),
        "candidate_submission_present": (python_run_dir / "submission.csv").is_file(),
        "artifact_manifest_present": (python_run_dir / "artifact_manifest.json").is_file(),
        "other_smoke_tests_passed": all(
            original.get("tests", {}).get(name, {}).get("status") == "passed"
            for name in (
                "002_failure_recovery",
                "003_memory",
                "004_evolution",
                "005_governance",
            )
        ),
    }
    passed = all(checks.values())
    tests = {
        "001_fresh_research_task": "passed" if all(
            checks[name]
            for name in (
                "python_supervisor_completed",
                "all_14_nodes_completed",
                "canonical_ledger_completed",
                "api_projection_completed",
                "result_gate_approved",
                "final_gate_approved",
                "hpc_dispatch_contract_bound",
                "remote_receipt_verified",
                "receipt_identity_hash",
                "lineage_completed",
                "research_report_present",
                "candidate_submission_present",
                "artifact_manifest_present",
            )
        ) else "failed",
        "002_failure_recovery": original.get("tests", {}).get("002_failure_recovery", {}).get("status"),
        "003_memory": original.get("tests", {}).get("003_memory", {}).get("status"),
        "004_evolution": original.get("tests", {}).get("004_evolution", {}).get("status"),
        "005_governance": original.get("tests", {}).get("005_governance", {}).get("status"),
    }
    verified = {
        "schema": "evomind.production_recovery.smoke_completion_verification.v1",
        "status": "passed" if passed else "failed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "source_report_sha256": sha256(report_path),
        "run_id": run_id,
        "tests": tests,
        "checks": checks,
        "mock_used": False,
    }
    output = report_path.with_name("completion_verification.json")
    output.write_text(
        json.dumps(verified, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": verified["status"],
        "run_id": run_id,
        "tests": tests,
        "artifact": str(output.relative_to(ROOT)).replace("\\", "/"),
        "mock_used": False,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
