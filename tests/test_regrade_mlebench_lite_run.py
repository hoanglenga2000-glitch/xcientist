from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import regrade_mlebench_lite_run as regrade


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_candidate(
    tmp_path: Path,
    *,
    gate_passed: bool = True,
    approval: bool = True,
    candidate_only: bool = True,
    confirmation_pending: bool = True,
) -> dict[str, Path]:
    run_id = "run"
    competition_id = "competition"
    run_dir = tmp_path / "runs" / run_id
    task_root = run_dir / competition_id
    attempt = task_root / "attempts" / "attempt_001"
    attempt.mkdir(parents=True)
    submission = attempt / "submission.csv"
    submission.write_bytes(b"id,target\n1,0.8\n")
    submission_sha256 = _sha256(submission)
    source_result = {
        "competition_id": competition_id,
        "status": regrade.CONFIRMATION_STATUS,
        "promotion_gate": {"passed": gate_passed},
        "candidate_only": candidate_only,
        "official_grader_confirmation_pending": confirmation_pending,
        "official_grader_withheld": True,
        "official_grader_executed": False,
        "valid_submission": True,
        "private_grader": {"reason": regrade.CONFIRMATION_REASON},
        "submission_path": str(submission),
        "submission_sha256": submission_sha256,
        "budget": {"seed": 42},
        "cv_score": 0.03,
    }
    result_path = task_root / "result.json"
    _write_json(result_path, source_result)
    manifest = {
        "schema": regrade.CANDIDATE_MANIFEST_SCHEMA,
        "frozen": True,
        "run_id": run_id,
        "competition_id": competition_id,
        "source_result_sha256": _sha256(result_path),
        "submission": {
            "path": str(submission),
            "sha256": submission_sha256,
            "size_bytes": submission.stat().st_size,
        },
    }
    manifest_path = task_root / "frozen_candidate_manifest.json"
    _write_json(manifest_path, manifest)
    approval_payload = {
        "schema": regrade.APPROVAL_SCHEMA,
        "approval_id": "human-approval-001",
        "approved": approval,
        "run_id": run_id,
        "competition_id": competition_id,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "submission_sha256": submission_sha256,
    }
    approval_path = run_dir / "approvals" / "competition.json"
    _write_json(approval_path, approval_payload)
    _write_json(
        run_dir / "checkpoint.json",
        {"run_id": run_id, "requested": [competition_id]},
    )
    return {
        "run_dir": run_dir,
        "task_root": task_root,
        "result": result_path,
        "submission": submission,
        "manifest": manifest_path,
        "approval": approval_path,
    }


def _valid_submission(*_args: object, **_kwargs: object) -> dict[str, Any]:
    return {"valid": True, "errors": []}


def _install_fake_grader(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []

    def fake_grade(submission_path: Path, *_args: object, **kwargs: object) -> dict[str, Any]:
        path = Path(submission_path)
        calls.append(path)
        report = {
            "status": "passed",
            "score": 0.81,
            "official_mlebench_grader_executed": True,
            "provenance": {"submission_sha256": _sha256(path)},
        }
        regrade.wave0.write_json(Path(kwargs["output_path"]), report)
        return report

    monkeypatch.setattr(regrade, "grade_private_submission", fake_grade)
    return calls


def _regrade(paths: dict[str, Path]) -> dict[str, Any]:
    return regrade.regrade_one(
        competition_id="competition",
        run_id="run",
        run_dir=paths["run_dir"],
        data_root=paths["run_dir"].parents[1],
        official_source_root=paths["run_dir"].parents[1],
        allowed_root=paths["run_dir"].parents[1],
        approval_path=paths["approval"],
    )


def test_next_regrade_dir_is_monotonic(tmp_path: Path):
    assert regrade.next_regrade_dir(tmp_path).name == "regrade_001"
    assert regrade.next_regrade_dir(tmp_path).name == "regrade_002"


def test_happy_path_grades_frozen_copy_and_preserves_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path)
    result_before = paths["result"].read_bytes()
    submission_before = paths["submission"].read_bytes()
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    grade_result = _regrade(paths)

    assert grade_result["status"] == "passed"
    assert grade_result["mle_private_grader_score"] == 0.81
    assert grade_result["official_grader_executed"] is True
    assert grade_result["frozen_candidate_artifacts_mutated"] is False
    assert grade_result["candidate_task_directory_appended"] is True
    assert len(calls) == 1
    assert calls[0].name == "frozen_submission.csv"
    assert calls[0].read_bytes() == submission_before
    assert paths["result"].read_bytes() == result_before
    assert paths["submission"].read_bytes() == submission_before
    receipt_path = paths["task_root"] / "regrades" / "official_private_grader.receipt.json"
    claim_path = paths["task_root"] / "regrades" / "official_private_grader.claim.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["one_shot"] is True
    assert receipt["submission_sha256"] == _sha256(paths["submission"])
    assert claim_path.is_file()
    assert Path(grade_result["regrade_dir"], "source_result.json").is_file()
    assert Path(grade_result["regrade_dir"], "candidate_manifest.json").is_file()
    assert Path(grade_result["regrade_dir"], "human_approval.json").is_file()


def test_failed_promotion_gate_fails_before_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path, gate_passed=False)
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(RuntimeError, match="promotion gate did not pass"):
        _regrade(paths)

    assert calls == []
    assert not (paths["task_root"] / "regrades").exists()


@pytest.mark.parametrize(
    ("candidate_only", "confirmation_pending", "message"),
    [
        (False, True, "not candidate-only"),
        (True, False, "no pending human grader confirmation"),
    ],
)
def test_source_result_requires_explicit_candidate_only_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_only: bool,
    confirmation_pending: bool,
    message: str,
):
    paths = _build_candidate(
        tmp_path,
        candidate_only=candidate_only,
        confirmation_pending=confirmation_pending,
    )
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(RuntimeError, match=message):
        _regrade(paths)

    assert calls == []


def test_missing_approval_fails_before_grader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = _build_candidate(tmp_path)
    paths["approval"].unlink()
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(FileNotFoundError, match="Missing human approval"):
        _regrade(paths)

    assert calls == []


def test_unapproved_record_fails_before_grader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = _build_candidate(tmp_path, approval=False)
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(RuntimeError, match="not approved"):
        _regrade(paths)

    assert calls == []


def test_submission_hash_drift_fails_before_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path)
    paths["submission"].write_bytes(b"id,target\n1,0.1\n")
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(RuntimeError, match="Submission SHA256 drift"):
        _regrade(paths)

    assert calls == []


def test_candidate_manifest_drift_fails_before_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["unexpected_mutation"] = True
    _write_json(paths["manifest"], manifest)
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(RuntimeError, match="manifest SHA256 does not match"):
        _regrade(paths)

    assert calls == []


def test_source_result_drift_fails_before_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path)
    source = json.loads(paths["result"].read_text(encoding="utf-8"))
    source["cv_score"] = 0.031
    _write_json(paths["result"], source)
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    with pytest.raises(RuntimeError, match="Source result SHA256"):
        _regrade(paths)

    assert calls == []


def test_one_shot_receipt_blocks_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = _build_candidate(tmp_path)
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    first = _regrade(paths)
    with pytest.raises(RuntimeError, match="receipt already exists"):
        _regrade(paths)

    assert first["status"] == "passed"
    assert len(calls) == 1


def test_grader_failure_is_receipted_and_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path)
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = 0

    def failing_grade(*_args: object, **_kwargs: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("grader exploded")

    monkeypatch.setattr(regrade, "grade_private_submission", failing_grade)
    with pytest.raises(RuntimeError, match="grader exploded"):
        _regrade(paths)
    with pytest.raises(RuntimeError, match="receipt already exists"):
        _regrade(paths)

    receipt_path = paths["task_root"] / "regrades" / "official_private_grader.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "grader_failed"
    assert receipt["grader_execution_state"] == "unknown_after_exception"
    assert calls == 1


def test_main_writes_separate_report_without_rewriting_candidate_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _build_candidate(tmp_path)
    summary_path = paths["run_dir"] / "summary.json"
    results_current_path = paths["run_dir"] / "results_current.json"
    _write_json(summary_path, {"sentinel": "frozen-summary"})
    _write_json(results_current_path, {"sentinel": "frozen-results"})
    frozen_paths = [
        paths["result"],
        paths["submission"],
        paths["run_dir"] / "checkpoint.json",
        summary_path,
        results_current_path,
    ]
    before = {path: path.read_bytes() for path in frozen_paths}
    monkeypatch.setattr(regrade, "validate_submission_file", _valid_submission)
    calls = _install_fake_grader(monkeypatch)

    exit_code = regrade.main(
        [
            "--run-id",
            "run",
            "--output-root",
            str(tmp_path / "runs"),
            "--data-root",
            str(tmp_path),
            "--official-source-root",
            str(tmp_path),
            "--allowed-root",
            str(tmp_path),
            "--approval-file",
            str(paths["approval"]),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert all(path.read_bytes() == payload for path, payload in before.items())
    report = json.loads(
        (paths["run_dir"] / "regrade_reports" / "regrade_001" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "passed"
    assert report["frozen_candidate_artifacts_mutated"] is False
    assert report["candidate_task_directory_appended"] is True


def test_cli_requires_approval_file_and_rejects_force():
    base = [
        "--run-id",
        "run",
        "--official-source-root",
        ".",
    ]
    with pytest.raises(SystemExit):
        regrade.parse_args(base)
    with pytest.raises(SystemExit):
        regrade.parse_args([*base, "--approval-file", "approval.json", "--force"])
