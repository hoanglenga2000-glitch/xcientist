from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.reconcile_siim_post_completion_state import (
    FROZEN_FILES,
    ReconciliationError,
    apply,
    resolve_layout,
    verify,
)


RUN_ID = "evomind_siim_isic_a800_job90353_20260730_095826"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "workspace" / "evomind_runs" / RUN_ID
    campaign = tmp_path / "workspace" / "hpc" / "job90353_siim_campaign" / RUN_ID
    run.mkdir(parents=True)
    campaign.mkdir(parents=True)
    tasks = {
        name: {"status": "completed"}
        for name in (
            "request_setup", "hpc_data_preflight", "data_audit", "research_design",
            "preprocessing_ablation", "full_training", "independent_review_freeze",
            "terminal_private_grader", "claim_audit_delivery",
        )
    }
    _write_json(run / "run.json", {"run_id": RUN_ID, "status": "completed", "tasks": tasks})
    _write_json(run / "research_design.json", {"run_id": RUN_ID, "status": "frozen_before_experiment", "budget_hours": {"ablation": 4, "formal_training": 18, "review_grade_delivery": 2, "total": 24}})
    _write_json(run / "workflow_contract.json", {"run_id": RUN_ID, "budget_hours": 24})
    _write_json(run / "candidate_freeze.json", {"run_id": RUN_ID, "status": "frozen_before_private_grader"})
    _write_json(run / "review.json", {"run_id": RUN_ID, "status": "review_passed"})
    _write_json(run / "private_grader.json", {"run_id": RUN_ID, "status": "failed_closed", "mle_private_grader_score": None, "official_submission_executed": False, "signals_sent": 0, "other_processes_modified": False})
    _write_json(run / "private_grader_ledger.json", {"run_id": RUN_ID, "execution_count": 1, "feedback_used_for_tuning": False, "official_submission_executed": False, "signals_sent": 0, "other_processes_modified": False})
    _write_json(run / "claim_audit.json", {"run_id": RUN_ID, "status": "passed", "signals_sent": 0, "other_processes_modified": False})
    _write_json(run / "deliverables.json", {"run_id": RUN_ID, "status": "ready"})
    for name in FROZEN_FILES[-4:]:
        (run / name).write_bytes((name + "\n").encode())
    artifacts = [
        {"path": name, "sha256": _sha(run / name)}
        for name in FROZEN_FILES
        if name != "artifact_manifest.json"
    ]
    _write_json(run / "artifact_manifest.json", {"run_id": RUN_ID, "status": "verified", "private_grader_execution_count": 1, "artifacts": artifacts})
    plan = {
        "run_id": RUN_ID,
        "budget_hours": {"ablation": 4, "formal_training": 72, "delivery": 2, "total": 78},
        "original_budget_hours": {"ablation": 4, "formal_training": 18, "delivery": 2, "total": 24},
        "budget_policy": "user_selected_A_corrected_full_closure",
    }
    _write_json(campaign / "campaign_plan.json", plan)
    _write_json(campaign / "campaign_plan_supersession.json", {"run_id": RUN_ID, "new_plan_sha256": _sha(campaign / "campaign_plan.json")})
    _write_json(campaign / "remote_evidence" / "campaign_state.json", {
        "run_id": RUN_ID, "status": "awaiting_collection_and_freeze",
        "corrected_formal_budget_started_at_epoch": 1000.0,
        "corrected_formal_budget_seconds": 259200,
        "updated_at": "1970-01-01T01:00:00+00:00",
        "completed_seeds": [43, 44, 45],
    })
    _write_json(campaign / "status.json", {"run_id": RUN_ID, "state": {"status": "awaiting_collection_and_freeze"}})
    _write_json(campaign / "end_to_end_supervisor.json", {"run_id": RUN_ID, "status": "failed_closed", "phase": "evidence_validation", "started_at": "2026-01-01T00:00:00+00:00", "iteration": 1})
    _write_json(campaign / "post_completion_delivery_watch.json", {"run_id": RUN_ID, "status": "waiting"})
    _write_json(campaign / "narration_audio_watch.json", {"run_id": RUN_ID, "status": "waiting"})
    _write_json(campaign / "published_status.json", {"run_id": RUN_ID, "status": "published"})
    video = tmp_path / "video.mp4"
    video.write_bytes(b"real-video-fixture")
    qa = tmp_path / "video-qa.json"
    _write_json(qa, {"status": "pass", "measured": {"sha256": _sha(video)}})
    return run, campaign, video


def test_reconciliation_is_append_only_and_idempotent(tmp_path: Path) -> None:
    run, campaign, video = _fixture(tmp_path)
    layout = resolve_layout(tmp_path, RUN_ID, video_path=video, video_qa_path=tmp_path / "video-qa.json")
    frozen_before = {name: _sha(run / name) for name in FROZEN_FILES}
    first = apply(layout)
    second = apply(layout)
    assert first["status"] == second["status"] == "passed"
    assert {name: _sha(run / name) for name in FROZEN_FILES} == frozen_before
    assert json.loads((run / "budget_supersession.json").read_text(encoding="utf-8"))["effective_budget_hours"]["total"] == 78
    assert json.loads((campaign / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((campaign / "end_to_end_supervisor.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((campaign / "post_completion_delivery_watch.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((campaign / "narration_audio_watch.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((campaign / "published_status.json").read_text(encoding="utf-8"))["authoritative_run_status"] == "completed"
    assert len(list((campaign / ".reconciliation_backups").glob("*.json"))) == 5
    assert verify(layout)["private_grader_execution_count"] == 1


def test_reconciliation_fails_closed_if_grader_count_changed(tmp_path: Path) -> None:
    run, _campaign, video = _fixture(tmp_path)
    ledger = json.loads((run / "private_grader_ledger.json").read_text())
    ledger["execution_count"] = 2
    _write_json(run / "private_grader_ledger.json", ledger)
    layout = resolve_layout(tmp_path, RUN_ID, video_path=video, video_qa_path=tmp_path / "video-qa.json")
    with pytest.raises(ReconciliationError, match="exactly one"):
        apply(layout)


def test_verify_only_requires_reconciliation_artifacts(tmp_path: Path) -> None:
    _run, _campaign, video = _fixture(tmp_path)
    layout = resolve_layout(tmp_path, RUN_ID, video_path=video, video_qa_path=tmp_path / "video-qa.json")
    with pytest.raises(ReconciliationError, match="budget supersession"):
        verify(layout)


def test_verify_only_does_not_rewrite_the_qa_report(tmp_path: Path) -> None:
    run, _campaign, video = _fixture(tmp_path)
    layout = resolve_layout(tmp_path, RUN_ID, video_path=video, video_qa_path=tmp_path / "video-qa.json")
    apply(layout)
    qa_path = run / "post_completion_reconciliation_qa.json"
    before = (qa_path.stat().st_mtime_ns, _sha(qa_path))
    assert verify(layout)["status"] == "passed"
    assert (qa_path.stat().st_mtime_ns, _sha(qa_path)) == before


def test_verify_fails_closed_on_frozen_delivery_mutation(tmp_path: Path) -> None:
    run, _campaign, video = _fixture(tmp_path)
    layout = resolve_layout(tmp_path, RUN_ID, video_path=video, video_qa_path=tmp_path / "video-qa.json")
    apply(layout)
    with (run / "evomind-siim-isic-code.zip").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ReconciliationError, match="artifact manifest hash mismatch"):
        verify(layout)


def test_verify_fails_closed_on_legacy_snapshot_archive_mutation(tmp_path: Path) -> None:
    _run, campaign, video = _fixture(tmp_path)
    layout = resolve_layout(tmp_path, RUN_ID, video_path=video, video_qa_path=tmp_path / "video-qa.json")
    apply(layout)
    archive = next((campaign / ".reconciliation_backups").glob("status.pre_*.json"))
    archive.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReconciliationError, match="archive hash mismatch"):
        verify(layout)
