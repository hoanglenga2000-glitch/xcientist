"""Local-only tests for the fail-closed G21 study scaffold generator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import create_g21_human_baseline_study_scaffold as generator
from scripts import validate_g21_human_baseline_study as validator


COMPETITION = "cure_bench"
STUDY_ID = "study_g21_cure_human_v2"
METRIC = "accuracy"
DIRECTION = "higher_is_better"


def _create(path: Path) -> dict[str, Any]:
    return generator.create_scaffold(
        path,
        competition=COMPETITION,
        study_id=STUDY_ID,
        metric=METRIC,
        direction=DIRECTION,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _reseal_manifest(root: Path) -> None:
    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        target = root / entry["path"]
        entry["bytes"] = target.stat().st_size
        entry["sha256"] = _sha(target)
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"])
    _write_json(manifest_path, manifest)


def test_scaffold_is_closed_incomplete_and_contains_no_scores(tmp_path: Path) -> None:
    root = tmp_path / "cure-study-template"
    result = _create(root)

    assert result["status"] == generator.INCOMPLETE_STATUS
    assert result["training_authorized"] is False
    assert result["synthetic_scores_present"] is False
    assert result["side_effects"] == generator.NO_SIDE_EFFECTS

    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == generator.INCOMPLETE_STATUS
    assert manifest["training_authorized"] is False
    assert manifest["synthetic_fixture"] is True
    assert manifest["synthetic_scores_present"] is False
    assert manifest["manifest_self_excluded"] is True
    assert manifest["run_id"] == validator.competition_goal.FIXED_RUN_ID
    assert manifest["allocation"] == validator.competition_goal.FIXED_ALLOCATION
    assert {item["role"] for item in manifest["files"]} == set(validator.REQUIRED_ROLES)

    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert set(actual) == {item["path"] for item in manifest["files"]} | {validator.MANIFEST_NAME}
    for entry in manifest["files"]:
        target = actual[entry["path"]]
        assert entry["regular_file"] is True
        assert entry["symlink"] is False
        assert entry["bytes"] == target.stat().st_size
        assert entry["sha256"] == _sha(target)

    metrics = json.loads((root / generator.ROLE_PATHS["participant_metrics"]).read_text(encoding="utf-8"))
    baseline = json.loads((root / generator.ROLE_PATHS["human_baseline"]).read_text(encoding="utf-8"))
    assert metrics["status"] == generator.INCOMPLETE_STATUS
    assert metrics["participant_count"] == 0
    assert metrics["records"] == []
    assert metrics["contains_synthetic_scores"] is False
    assert baseline["status"] == generator.INCOMPLETE_STATUS
    assert baseline["mean"] is None
    assert baseline["sample_size"] == 0
    assert baseline["training_authorized"] is False


def test_validator_rejects_generated_scaffold_before_any_receipt(tmp_path: Path) -> None:
    root = tmp_path / "rejected-template"
    _create(root)
    receipt_path = tmp_path / "must-not-exist.json"

    with pytest.raises(validator.StudyValidationError) as caught:
        validator.validate_study_package(
            root,
            competition=COMPETITION,
            expected_metric=METRIC,
            expected_direction=DIRECTION,
        )
    assert caught.value.code == "SYNTHETIC_FIXTURE_FORBIDDEN"
    assert not receipt_path.exists()


def test_removing_template_marker_still_rejects_fewer_than_30_records(tmp_path: Path) -> None:
    root = tmp_path / "underpowered-template"
    _create(root)

    # Test-only mutation: prove the n>=30 gate independently of the deliberate
    # template marker.  These values never leave pytest's temporary directory.
    claim_path = root / generator.ROLE_PATHS["holdout_claim"]
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim.update(
        {
            "unconsumed_at_claim": True,
            "consumed_once": True,
            "selection_data_scope": "development_only",
            "prior_holdout_overlap": 0,
            "test_labels_exposed": False,
        }
    )
    _write_json(claim_path, claim)

    protocol_path = root / generator.ROLE_PATHS["protocol"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["holdout_claim_sha256"] = _sha(claim_path)
    _write_json(protocol_path, protocol)

    baseline_path = root / generator.ROLE_PATHS["human_baseline"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["holdout_claim_sha256"] = _sha(claim_path)
    baseline["protocol_file_sha256"] = _sha(protocol_path)
    _write_json(baseline_path, baseline)

    metrics_path = root / generator.ROLE_PATHS["participant_metrics"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["records"] = [
        {
            "participant_id": "p_" + hashlib.sha256(f"test-only-{index}".encode()).hexdigest(),
            "score": 0.5,
        }
        for index in range(validator.MIN_PARTICIPANTS - 1)
    ]
    metrics["participant_count"] = len(metrics["records"])
    _write_json(metrics_path, metrics)

    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["synthetic_fixture"] = False
    manifest["status"] = "EVIDENCE_COLLECTION_PENDING"
    _write_json(manifest_path, manifest)
    _reseal_manifest(root)

    with pytest.raises(validator.StudyValidationError) as caught:
        validator.validate_study_package(
            root,
            competition=COMPETITION,
            expected_metric=METRIC,
            expected_direction=DIRECTION,
        )
    assert validator.MIN_PARTICIPANTS == 30
    assert caught.value.code == "PARTICIPANT_COUNT_INVALID"


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_refuses_overwrite_and_preserves_existing_path(tmp_path: Path, existing_kind: str) -> None:
    target = tmp_path / "existing"
    if existing_kind == "directory":
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("preserve\n", encoding="utf-8")
    else:
        target.write_text("preserve\n", encoding="utf-8")
        sentinel = target
    before = sentinel.read_bytes()

    with pytest.raises(generator.ScaffoldError) as caught:
        _create(target)
    assert caught.value.code == "OUTPUT_ALREADY_EXISTS"
    assert sentinel.read_bytes() == before
    if existing_kind == "directory":
        assert {path.name for path in target.iterdir()} == {"sentinel.txt"}


def test_rejects_traversal_nonlocal_and_production_like_paths(tmp_path: Path) -> None:
    with pytest.raises(generator.ScaffoldError) as caught:
        _create(tmp_path / "safe" / ".." / "escape")
    assert caught.value.code == "OUTPUT_PATH_UNSAFE"

    with pytest.raises(generator.ScaffoldError) as caught:
        _create(Path(r"\\server\share\study"))
    assert caught.value.code == "NONLOCAL_PATH_FORBIDDEN"

    production_parent = tmp_path / "ProgramData" / "EvoMind"
    production_parent.mkdir(parents=True)
    with pytest.raises(generator.ScaffoldError) as caught:
        _create(production_parent / "study")
    assert caught.value.code == "PRODUCTION_PATH_FORBIDDEN"


def test_cli_reports_incomplete_and_second_call_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "cli-template"
    args = [
        "--output-dir",
        str(target),
        "--competition",
        COMPETITION,
        "--study-id",
        STUDY_ID,
        "--metric",
        METRIC,
        "--direction",
        DIRECTION,
    ]

    assert generator.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == generator.INCOMPLETE_STATUS
    assert output["training_authorized"] is False
    assert output["synthetic_scores_present"] is False

    assert generator.main(args) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "status": "REJECTED",
        "error": {"code": "OUTPUT_ALREADY_EXISTS", "detail": ""},
    }
    assert os.path.lexists(target)
