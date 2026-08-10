#!/usr/bin/env python3
"""Fail closed until the same SIIM Run is ready for evidence-bound recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

VIDEO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = VIDEO_ROOT / "preproduction" / "production-contract.json"
OUTPUT_PATH = VIDEO_ROOT / "preproduction" / "pre-recording-gate.json"
EXPECTED_REPORT_PAGES = 9
EVOLUTION_MARKERS = ("EvoMind 四轮进化", "R1", "R2", "R3", "R4")
DELIVERABLE_NAMES = (
    "evomind-siim-isic-report.pdf",
    "evomind-siim-isic-results.csv",
    "evomind-siim-isic-code.zip",
    "evomind-siim-isic-evidence.zip",
)
REVIEW_PASSED_STATUSES = frozenset({"passed", "review_passed"})
GRADER_SCORED_STATUSES = frozenset({"completed", "passed", "scored", "succeeded"})


class PreRecordingGateError(RuntimeError):
    """The recording release evidence is absent or inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PreRecordingGateError(f"missing or unsafe {label}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreRecordingGateError(f"invalid {label}") from exc
    if not isinstance(payload, dict):
        raise PreRecordingGateError(f"{label} is not a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreRecordingGateError(message)


def verify_multiround_report(pdf_path: Path, html_path: Path) -> dict[str, Any]:
    """Bind recording permission to the final nine-page, four-round report."""

    require(
        pdf_path.is_file() and not pdf_path.is_symlink() and pdf_path.stat().st_size > 0,
        "professional PDF report is missing or unsafe",
    )
    require(
        html_path.is_file() and not html_path.is_symlink() and html_path.stat().st_size > 0,
        "professional HTML report is missing or unsafe",
    )
    try:
        with fitz.open(pdf_path) as document:
            page_count = document.page_count
            pdf_text = "\n".join(page.get_text("text") for page in document)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreRecordingGateError("professional PDF report is unreadable") from exc
    try:
        html_text = html_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise PreRecordingGateError("professional HTML report is unreadable") from exc
    require(page_count == EXPECTED_REPORT_PAGES, "professional PDF report is not the nine-page edition")
    for marker in EVOLUTION_MARKERS:
        require(marker in pdf_text, f"professional PDF report omits evolution marker: {marker}")
        require(marker in html_text, f"professional HTML report omits evolution marker: {marker}")
    return {
        "pdf_pages": page_count,
        "pdf_sha256": sha256_file(pdf_path),
        "html_sha256": sha256_file(html_path),
        "evolution_markers": list(EVOLUTION_MARKERS),
        "multiround_evolution_verified": True,
    }


def same_run(payload: dict[str, Any], run_id: str, label: str) -> None:
    require(payload.get("run_id") == run_id, f"{label} belongs to another Run")


def validate_gate(project_root: Path, contract_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    contract = read_json(contract_path.resolve(), "production contract")
    run_id = str(contract.get("run_id") or "")
    require(bool(run_id) and all(character.isalnum() or character in "_-" for character in run_id), "invalid Run ID")
    run_dir = (project_root / "workspace" / "evomind_runs" / run_id).resolve()
    require(run_dir.parent == (project_root / "workspace" / "evomind_runs").resolve(), "Run path escaped workspace")
    require(run_dir.is_dir() and not run_dir.is_symlink(), "same-Run directory is unavailable")

    run = read_json(run_dir / "run.json", "run.json")
    request = read_json(run_dir / "request.json", "request.json")
    task_graph = read_json(run_dir / "task_graph.json", "task_graph.json")
    workflow_contract = read_json(run_dir / "workflow_contract.json", "workflow_contract.json")
    review = read_json(run_dir / "review.json", "review.json")
    claim = read_json(run_dir / "claim_audit.json", "claim_audit.json")
    ledger = read_json(run_dir / "private_grader_ledger.json", "private_grader_ledger.json")
    grader = read_json(run_dir / "private_grader.json", "private_grader.json")
    deliverables = read_json(run_dir / "deliverables.json", "deliverables.json")
    manifest = read_json(run_dir / "artifact_manifest.json", "artifact_manifest.json")
    for label, payload in (
        ("run.json", run),
        ("workflow_contract.json", workflow_contract),
        ("review.json", review),
        ("claim_audit.json", claim),
        ("private_grader_ledger.json", ledger),
        ("private_grader.json", grader),
        ("deliverables.json", deliverables),
        ("artifact_manifest.json", manifest),
    ):
        same_run(payload, run_id, label)

    require(run.get("status") == "completed", "Run is not completed")
    require(task_graph.get("run_id") == run_id, "task graph belongs to another Run")
    task_records = task_graph.get("nodes") if isinstance(task_graph.get("nodes"), list) else []
    require(len(task_records) == 9, "task graph does not contain nine governed nodes")
    require(
        all(isinstance(task, dict) and task.get("status") == "completed" for task in task_records),
        "task graph contains incomplete nodes",
    )
    require(workflow_contract.get("task_type") == "image_classification", "workflow contract task type changed")
    require(workflow_contract.get("dataset") == "siim-isic-melanoma-classification", "workflow contract dataset changed")
    review_status = str(review.get("status") or "")
    require(review_status in REVIEW_PASSED_STATUSES, "Independent Review is not passed")
    require(claim.get("status") == "passed", "Claim Audit is not passed")
    require(int(ledger.get("execution_count") or 0) == 1, "private grader execution count is not exactly one")
    require(manifest.get("status") == "verified", "artifact manifest is not verified")
    require(int(manifest.get("private_grader_execution_count") or 0) == 1, "manifest grader count changed")
    require(manifest.get("official_submission") == "forbidden", "official submission boundary changed")
    require(deliverables.get("status") == "ready", "deliverables are not ready")
    compute = request.get("compute_policy") if isinstance(request.get("compute_policy"), dict) else {}
    submission = request.get("submission_policy") if isinstance(request.get("submission_policy"), dict) else {}
    require(compute.get("backend") == "hpc", "Run was not bound to HPC")
    require(compute.get("local_gpu_allowed") is False, "local GPU boundary changed")
    require(submission.get("official_submission") == "forbidden", "request submission boundary changed")
    require(grader.get("official_submission_executed") is False, "grader executed an official submission")
    grader_status = str(grader.get("status") or "")
    grader_score = grader.get("mle_private_grader_score", grader.get("score"))
    if grader_status == "failed_closed":
        require(grader_score is None, "failed-closed grader unexpectedly contains a score")
        require(
            bool(str(grader.get("failure_reason") or grader.get("error") or "").strip()),
            "failed-closed grader is missing its recorded reason",
        )
    else:
        require(grader_status in GRADER_SCORED_STATUSES, "private grader is not in a terminal state")
        require(
            not isinstance(grader_score, bool)
            and isinstance(grader_score, (int, float))
            and math.isfinite(float(grader_score))
            and 0 <= float(grader_score) <= 1,
            "terminal private grader score is invalid",
        )

    artifact_records = {
        str(record.get("path") or ""): record
        for record in manifest.get("artifacts") or []
        if isinstance(record, dict)
    }
    html_name = "research_report.html"
    html_path = run_dir / html_name
    html_artifact = artifact_records.get(html_name) or {}
    require(html_path.is_file() and not html_path.is_symlink(), "missing professional HTML report")
    html_digest = sha256_file(html_path)
    require(str(html_artifact.get("sha256") or "").lower() == html_digest, "manifest hash mismatch: research_report.html")
    require(int(html_artifact.get("bytes") or -1) == html_path.stat().st_size, "manifest byte count mismatch: research_report.html")
    delivery_records = {
        str(record.get("name") or ""): record
        for record in deliverables.get("files") or []
        if isinstance(record, dict)
    }
    verified_files = []
    for name in DELIVERABLE_NAMES:
        path = run_dir / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, f"missing deliverable: {name}")
        digest = sha256_file(path)
        artifact = artifact_records.get(name) or {}
        delivery = delivery_records.get(name) or {}
        require(str(artifact.get("sha256") or "").lower() == digest, f"manifest hash mismatch: {name}")
        require(int(artifact.get("bytes") or -1) == path.stat().st_size, f"manifest byte count mismatch: {name}")
        require(str(delivery.get("sha256") or "").lower() == digest, f"delivery hash mismatch: {name}")
        require(int(delivery.get("bytes") or -1) == path.stat().st_size, f"delivery byte count mismatch: {name}")
        require(str(delivery.get("download_url") or "").endswith(f"/{name}"), f"download URL mismatch: {name}")
        verified_files.append({"name": name, "bytes": path.stat().st_size, "sha256": digest})

    report_evidence = verify_multiround_report(
        run_dir / DELIVERABLE_NAMES[0],
        html_path,
    )

    checks = claim.get("checks") if isinstance(claim.get("checks"), dict) else {}
    for name in (
        "no_public_leaderboard_claim",
        "no_official_medal_claim",
        "no_clinical_diagnosis_claim",
        "private_grader_not_used_for_tuning",
        "candidate_hashes_unchanged",
        "official_submission_not_executed",
    ):
        require(checks.get(name) is True, f"Claim Audit check failed: {name}")

    return {
        "schema": "evomind.siim.video_pre_recording_gate.v1",
        "captured_at": utc_now(),
        "run_id": run_id,
        "status": "passed",
        "recording_allowed": True,
        "verified_download_count": len(verified_files),
        "verified_files": verified_files,
        "private_grader_execution_count": 1,
        "private_grader_outcome": grader_status,
        "private_grader_score": grader_score,
        "independent_review": review_status,
        "claim_audit": "passed",
        "official_submission_executed": False,
        "same_run_evidence": True,
        "report_evidence": report_evidence,
        "multiround_evolution_verified": True,
        "secrets_emitted": False,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_gate(args.project_root, args.contract)
    except (OSError, ValueError, PreRecordingGateError) as exc:
        failure = {
            "schema": "evomind.siim.video_pre_recording_gate.v1",
            "captured_at": utc_now(),
            "status": "blocked",
            "recording_allowed": False,
            "error_type": type(exc).__name__,
            "secrets_emitted": False,
        }
        atomic_json(args.output, failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 4
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
