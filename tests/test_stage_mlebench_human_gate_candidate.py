from __future__ import annotations

from pathlib import Path

import pytest

from scripts import stage_mlebench_human_gate_candidate as stage


def _csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_public_submission_validation_requires_exact_schema_ids_and_probabilities(
    tmp_path: Path,
):
    sample = _csv(tmp_path / "sample.csv", "id,A,B\nrow-1,0,0\nrow-2,0,0\n")
    candidate = _csv(
        tmp_path / "candidate.csv",
        "id,A,B\nrow-1,0.25,0.75\nrow-2,0.5,0.5\n",
    )

    report = stage.validate_public_submission_csv(candidate, sample)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["rows"] == 2
    assert report["columns"] == ["id", "A", "B"]
    assert report["minimum_probability"] == 0.25
    assert report["maximum_probability"] == 0.75
    assert report["private_labels_used"] is False


@pytest.mark.parametrize(
    ("candidate_text", "expected_error"),
    [
        ("id,B,A\nrow-1,0.5,0.5\n", "header_or_column_order_mismatch"),
        ("id,A,B\nwrong,0.5,0.5\n", "id_order_mismatch_at_row_2"),
        ("id,A,B\nrow-1,nan,0.5\n", "invalid_probability_at_row_2"),
        ("id,A,B\nrow-1,1.1,-0.1\n", "invalid_probability_at_row_2"),
    ],
)
def test_public_submission_validation_fails_closed(
    tmp_path: Path,
    candidate_text: str,
    expected_error: str,
):
    sample = _csv(tmp_path / "sample.csv", "id,A,B\nrow-1,0,0\n")
    candidate = _csv(tmp_path / "candidate.csv", candidate_text)

    report = stage.validate_public_submission_csv(candidate, sample)

    assert report["valid"] is False
    assert expected_error in report["errors"]


def test_public_submission_validation_supports_finite_regression_values(tmp_path: Path):
    sample = _csv(tmp_path / "sample.csv", "key,fare_amount\na,11.35\nb,11.35\n")
    candidate = _csv(
        tmp_path / "candidate.csv",
        "key,fare_amount\na,8.25\nb,42.75\n",
    )

    report = stage.validate_public_submission_csv(
        candidate,
        sample,
        prediction_bounds=None,
    )

    assert report["valid"] is True
    assert report["minimum_probability"] == 8.25
    assert report["maximum_probability"] == 42.75


def test_extract_and_recompute_metric_supports_rmse():
    source = {
        "confirmation_gate": {"passed": True, "metric": "rmse", "direction": "minimize"},
        "metrics": {"ensemble_oof_rmse": 2.71},
    }
    independent = {"recomputed_ensemble_oof_rmse": 2.71}

    assert stage._extract_metric(source) == ("rmse", "minimize", 2.71)
    stage._verify_recomputed_metric(independent, source, 2.71)


@pytest.mark.parametrize("value", ["", ".hidden", "../escape", "slash/name", "x" * 129])
def test_validate_run_id_rejects_nonportable_or_escaping_values(value: str):
    with pytest.raises(stage.StagingError):
        stage.validate_run_id(value)


def test_ensure_within_rejects_root_and_escape_by_default(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    assert stage.ensure_within(root / "child", root) == (root / "child").resolve()
    with pytest.raises(stage.StagingError):
        stage.ensure_within(root, root)
    with pytest.raises(stage.StagingError):
        stage.ensure_within(root.parent / "outside", root)


def test_cli_has_no_replace_existing_escape_hatch():
    with pytest.raises(SystemExit):
        stage.parse_args(
            [
                "--package-dir",
                "package",
                "stage",
                "--run-id",
                "candidate-v1",
                "--replace-existing",
            ]
        )


def test_existing_staged_run_is_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_root = tmp_path / "human_gate_staged_runs"
    output_root.mkdir()
    target = output_root / "candidate-v1"
    target.mkdir()
    monkeypatch.setattr(stage, "ALLOWED_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        stage,
        "verify_human_gate_package",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(stage.StagingError, match="Immutable staged run already exists"):
        stage.stage_candidate_run(
            tmp_path / "package",
            run_id="candidate-v1",
            output_root=output_root,
        )


def test_staged_live_candidates_match_regrade_source_contract():
    root = stage.ALLOWED_OUTPUT_ROOT
    candidates = {
        "hg_jigsaw_multiseed_v3_20260727": "jigsaw-toxic-comment-classification-challenge",
        "hg_spooky_crossrun_xgb_v1_20260727": "spooky-author-identification",
    }
    if not all((root / run_id).is_dir() for run_id in candidates):
        pytest.skip("Live staged Human Gate fixtures are not present")

    for run_id, competition_id in candidates.items():
        run_root = root / run_id
        verified = stage.verify_staged_run(run_root)
        result = stage.read_json(
            run_root / competition_id / "result.json", label="staged result"
        )
        stage.regrade._validate_source_result(result, competition_id)
        assert verified["status"] == "verified_human_approval_pending"
        assert verified["approved"] is False
        assert verified["official_grader_executed"] is False
        assert not (run_root / competition_id / "regrades").exists()
