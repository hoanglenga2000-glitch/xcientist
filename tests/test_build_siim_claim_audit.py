from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_siim_claim_audit as audit


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture_run(tmp_path: Path) -> Path:
    run_id = "evomind_siim_isic_claim_fixture"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    submission = run_dir / "submission.csv"
    submission.write_text("image_name,target\na,0.1\n", encoding="utf-8")
    submission_hash = audit.sha256_file(submission)
    write_json(run_dir / "run.json", {
        "run_id": run_id,
        "task_type": "image_classification",
        "dataset": "siim-isic-melanoma-classification",
        "gates": {"official_submission": "forbidden"},
    })
    write_json(run_dir / "metrics.json", {
        "run_id": run_id,
        "private_grader_execution_count": 0,
        "kaggle_submission_executed": False,
        "clinical_diagnosis_claimed": False,
    })
    write_json(run_dir / "hpc_runtime.json", {
        "run_id": run_id,
        "other_processes_modified": False,
        "signals_sent": 0,
    })
    write_json(run_dir / "candidate_freeze.json", {
        "run_id": run_id,
        "status": "frozen_before_private_grader",
        "tuning_closed": True,
        "official_submission": "forbidden",
        "private_grader_execution_count_before_freeze": 0,
        "frozen_at": "2026-07-29T00:00:00+00:00",
        "artifacts": [{"path": "submission.csv", "sha256": submission_hash}],
    })
    write_json(run_dir / "review.json", {
        "run_id": run_id,
        "status": "passed",
        "checks": {name: True for name in audit.REQUIRED_REVIEW_CHECKS},
        "artifact_hashes": {"submission.csv": submission_hash},
    })
    freeze_hash = audit.sha256_file(run_dir / "candidate_freeze.json")
    write_json(run_dir / "private_grader.json", {
        "run_id": run_id,
        "status": "passed",
        "execution_index": 1,
        "candidate_freeze_sha256": freeze_hash,
        "executed_after_freeze": True,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
        "mle_private_grader_score": 0.93,
        "completed_at": "2026-07-29T01:00:00+00:00",
    })
    write_json(run_dir / "private_grader_ledger.json", {
        "run_id": run_id,
        "status": "terminal_execution_recorded",
        "execution_count": 1,
        "candidate_freeze_sha256": freeze_hash,
        "result_sha256": audit.sha256_file(run_dir / "private_grader.json"),
        "feedback_used_for_tuning": False,
        "score": 0.93,
        "recorded_at": "2026-07-29T01:00:00+00:00",
    })
    return run_dir


def test_claim_audit_is_hash_bound_and_idempotent(tmp_path: Path) -> None:
    run_dir = fixture_run(tmp_path)
    first = audit.build_claim_audit(run_dir)
    second = audit.build_claim_audit(run_dir)
    payload = json.loads(Path(first["claim_audit"]).read_text(encoding="utf-8"))

    assert first == second
    assert payload["status"] == "passed"
    assert all(payload["checks"].values())
    assert payload["terminal_private_grader_score"] == pytest.approx(0.93)
    assert payload["signals_sent"] == 0


def test_immutable_claim_audit_source_survives_normalized_workflow_copy(tmp_path: Path) -> None:
    run_dir = fixture_run(tmp_path)
    source = run_dir / "claim_audit_source.json"
    first = audit.build_claim_audit(run_dir, output=source)
    source_bytes = source.read_bytes()
    normalized = json.loads(source_bytes)
    normalized["schema"] = "evomind.siim.claim_audit_ingress.v1"
    normalized["source_claim_audit_sha256"] = audit.sha256_file(source)
    write_json(run_dir / "claim_audit.json", normalized)

    second = audit.build_claim_audit(run_dir, output=source)

    assert first == second
    assert source.read_bytes() == source_bytes
    assert json.loads((run_dir / "claim_audit.json").read_text(encoding="utf-8"))["schema"] == (
        "evomind.siim.claim_audit_ingress.v1"
    )


def test_claim_audit_rejects_frozen_artifact_drift(tmp_path: Path) -> None:
    run_dir = fixture_run(tmp_path)
    (run_dir / "submission.csv").write_text("image_name,target\na,0.9\n", encoding="utf-8")

    with pytest.raises(audit.ClaimAuditError, match="frozen artifact changed"):
        audit.build_claim_audit(run_dir)


def test_claim_audit_rejects_grader_feedback_reuse(tmp_path: Path) -> None:
    run_dir = fixture_run(tmp_path)
    grader_path = run_dir / "private_grader.json"
    grader = json.loads(grader_path.read_text(encoding="utf-8"))
    grader["feedback_used_for_tuning"] = True
    write_json(grader_path, grader)
    ledger_path = run_dir / "private_grader_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["result_sha256"] = audit.sha256_file(grader_path)
    write_json(ledger_path, ledger)

    with pytest.raises(audit.ClaimAuditError, match="feedback entered tuning"):
        audit.build_claim_audit(run_dir)
