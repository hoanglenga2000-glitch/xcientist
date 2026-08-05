from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from research_os.mlebench_phase_a import (
    LITE22_BY_ID,
    LITE22_SPECS,
    METRIC_REGISTRY,
    MLEBenchContractError,
    audit_lite22_specs,
    compute_metric,
    export_specs,
    get_competition_spec,
    grade_private_submission,
    resolve_competition,
    to_proxy_score,
    validate_submission_file,
    validate_submission_frame,
)


def _materialize_competition(
    root: Path,
    competition_id: str,
    sample: pd.DataFrame,
    answers: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    spec = get_competition_spec(competition_id)
    competition_root = root / competition_id
    public = competition_root / "prepared" / "public"
    private = competition_root / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    sample_path = competition_root / spec.submission.sample_submission_relative
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)
    answers_path = competition_root / spec.submission.answers_relative
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    (answers if answers is not None else sample).to_csv(answers_path, index=False)

    # The resolver requires at least one declared train and test candidate.
    train_path = competition_root / spec.train_candidates[0]
    test_path = competition_root / spec.test_candidates[0]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("id,target\n1,0\n", encoding="utf-8")
    if test_path != sample_path and test_path != answers_path:
        test_path.write_text("id\n1\n", encoding="utf-8")
    return sample_path, answers_path


def test_lite_registry_contains_exactly_22_unique_specs():
    assert len(LITE22_SPECS) == 22
    assert len(LITE22_BY_ID) == 22
    assert {spec.competition_id for spec in LITE22_SPECS} == set(LITE22_BY_ID)


def test_every_spec_has_registered_metric_and_consistent_direction():
    assert len(METRIC_REGISTRY) == 9
    for spec in LITE22_SPECS:
        assert spec.metric in METRIC_REGISTRY
        assert spec.direction == METRIC_REGISTRY[spec.metric].direction
        assert spec.submission.id_columns
        assert spec.submission.sample_submission_relative.startswith("prepared/")
        assert spec.submission.answers_relative.startswith("prepared/private/")
        assert spec.train_candidates
        assert spec.test_candidates


def test_resolve_may2022_official_paths_and_schema(tmp_path: Path):
    sample = pd.DataFrame({"id": [10, 11], "target": [0.5, 0.5]})
    _materialize_competition(tmp_path, "tabular-playground-series-may-2022", sample)
    resolved = resolve_competition("tabular-playground-series-may-2022", tmp_path)
    assert resolved.sample_columns == ("id", "target")
    assert resolved.prediction_columns == ("target",)
    assert resolved.public_dir == (tmp_path / "tabular-playground-series-may-2022/prepared/public").resolve()
    assert resolved.answers_path.name == "test.csv"


def test_public_only_resolution_does_not_require_private_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sample = pd.DataFrame({"id": [10, 11], "target": [0.5, 0.5]})
    _, answers = _materialize_competition(
        tmp_path, "tabular-playground-series-may-2022", sample
    )
    shutil.rmtree(answers.parent)

    with pytest.raises(MLEBenchContractError, match="Missing official prepared paths"):
        resolve_competition("tabular-playground-series-may-2022", tmp_path)

    monkeypatch.setenv("EVOMIND_MLEBENCH_PUBLIC_ONLY", "1")
    resolved = resolve_competition(
        "tabular-playground-series-may-2022", tmp_path
    )
    assert resolved.public_dir.is_dir()
    assert not resolved.private_dir.exists()
    assert not resolved.answers_path.exists()


def test_resolver_rejects_unknown_competition(tmp_path: Path):
    with pytest.raises(MLEBenchContractError, match="not in MLE-Bench Lite 22"):
        resolve_competition("not-a-lite-task", tmp_path)


def test_valid_binary_probability_submission(tmp_path: Path):
    sample = pd.DataFrame({"id": [10, 11, 12], "target": [0.5, 0.5, 0.5]})
    _materialize_competition(tmp_path, "tabular-playground-series-may-2022", sample)
    submission = pd.DataFrame({"id": [10, 11, 12], "target": [0.1, 0.8, 0.4]})
    path = tmp_path / "submission.csv"
    submission.to_csv(path, index=False)
    result = validate_submission_file(path, "tabular-playground-series-may-2022", tmp_path)
    assert result["valid"] is True
    assert result["errors"] == []
    json.dumps(result)


def test_submission_reordered_ids_warns_but_remains_valid(tmp_path: Path):
    sample = pd.DataFrame({"id": [10, 11], "target": [0.5, 0.5]})
    _materialize_competition(tmp_path, "tabular-playground-series-may-2022", sample)
    resolved = resolve_competition("tabular-playground-series-may-2022", tmp_path)
    submission = pd.DataFrame({"id": [11, 10], "target": [0.8, 0.1]})
    result = validate_submission_frame(submission, sample, resolved)
    assert result["valid"] is True
    assert result["id_set_match"] is True
    assert result["id_order_match"] is False
    assert "id_order_differs_but_id_set_matches" in result["warnings"]


def test_official_sample_duplicate_ids_are_valid_when_multiplicity_and_order_match(tmp_path: Path):
    sample = pd.DataFrame({"id": ["same", "same", "other"], "target": [0.5, 0.5, 0.5]})
    _materialize_competition(tmp_path, "tabular-playground-series-may-2022", sample)
    resolved = resolve_competition("tabular-playground-series-may-2022", tmp_path)
    submission = pd.DataFrame({"id": ["same", "same", "other"], "target": [0.1, 0.8, 0.4]})
    result = validate_submission_frame(submission, sample, resolved)
    assert result["valid"] is True
    assert result["duplicate_id_count"] == 1
    assert result["sample_duplicate_id_count"] == 1
    assert result["id_multiset_match"] is True
    assert "duplicate_ids" not in result["errors"]


def test_duplicate_id_multiplicity_must_match_official_sample(tmp_path: Path):
    sample = pd.DataFrame({"id": ["same", "same", "other"], "target": [0.5, 0.5, 0.5]})
    _materialize_competition(tmp_path, "tabular-playground-series-may-2022", sample)
    resolved = resolve_competition("tabular-playground-series-may-2022", tmp_path)
    submission = pd.DataFrame({"id": ["same", "other", "other"], "target": [0.1, 0.8, 0.4]})
    result = validate_submission_frame(submission, sample, resolved)
    assert result["valid"] is False
    assert result["id_set_match"] is True
    assert result["id_multiset_match"] is False
    assert "id_multiplicity_mismatch" in result["errors"]


@pytest.mark.parametrize(
    ("submission", "error"),
    [
        (pd.DataFrame({"id": [10, 10], "target": [0.1, 0.2]}), "duplicate_ids"),
        (pd.DataFrame({"id": [10, 11], "target": [0.1, 1.2]}), "probability_out_of_range"),
        (pd.DataFrame({"id": [10, 11], "target": [0.1, np.inf]}), "nonfinite_predictions"),
        (pd.DataFrame({"id": [10, 11], "target": [0.1, np.nan]}), "missing_predictions"),
    ],
)
def test_invalid_binary_submissions_are_classified(tmp_path: Path, submission: pd.DataFrame, error: str):
    sample = pd.DataFrame({"id": [10, 11], "target": [0.5, 0.5]})
    _materialize_competition(tmp_path, "tabular-playground-series-may-2022", sample)
    resolved = resolve_competition("tabular-playground-series-may-2022", tmp_path)
    result = validate_submission_frame(submission, sample, resolved)
    assert result["valid"] is False
    assert error in result["errors"]


def test_multiclass_submission_requires_probability_row_sum(tmp_path: Path):
    sample = pd.DataFrame({"id": ["a", "b"], "EAP": [1 / 3, 1 / 3], "HPL": [1 / 3, 1 / 3], "MWS": [1 / 3, 1 / 3]})
    _materialize_competition(tmp_path, "spooky-author-identification", sample)
    resolved = resolve_competition("spooky-author-identification", tmp_path)
    invalid = pd.DataFrame({"id": ["a", "b"], "EAP": [0.6, 0.2], "HPL": [0.3, 0.2], "MWS": [0.2, 0.2]})
    result = validate_submission_frame(invalid, sample, resolved)
    assert result["valid"] is False
    assert result["row_sum_violation_count"] == 2
    assert "probability_rows_do_not_sum_to_one" in result["errors"]


def test_metric_registry_scores_all_core_shapes():
    assert compute_metric("roc_auc", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(1.0)
    assert compute_metric("accuracy", [0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert compute_metric("rmse", [0.0, 2.0], [0.0, 0.0]) == pytest.approx(np.sqrt(2.0))
    assert compute_metric("token_exact_accuracy", ["one", "two"], ["one", "bad"]) == 0.5
    y_multi = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    p_multi = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    assert compute_metric("mean_columnwise_roc_auc", y_multi, p_multi) == pytest.approx(1.0)
    assert 0.0 <= to_proxy_score("rmse", 1.0) <= 1.0


def test_export_specs_is_json_serializable(tmp_path: Path):
    path = export_specs(tmp_path / "specs.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["competition_count"] == 22
    assert len(payload["competitions"]) == 22


def test_official_mlebench_private_grader_adapter_executes(tmp_path: Path):
    source_root = Path(__file__).resolve().parents[1] / "external-projects" / "mle-bench"
    if not source_root.is_dir():
        pytest.skip("upstream MLE-Bench source tree is not available")
    sample = pd.DataFrame({"id": [10, 11, 12, 13], "target": [0.5, 0.5, 0.5, 0.5]})
    answers = pd.DataFrame({"id": [10, 11, 12, 13], "target": [0, 1, 0, 1]})
    _materialize_competition(
        tmp_path,
        "tabular-playground-series-may-2022",
        sample,
        answers,
    )
    submission = pd.DataFrame({"id": [10, 11, 12, 13], "target": [0.1, 0.9, 0.2, 0.8]})
    submission_path = tmp_path / "official_submission.csv"
    submission.to_csv(submission_path, index=False)
    report = grade_private_submission(
        submission_path,
        "tabular-playground-series-may-2022",
        tmp_path,
        official_source_root=source_root,
        seed=42,
        budget={"kind": "unit_test"},
    )
    assert report["status"] == "passed"
    assert report["official_mlebench_grader_executed"] is True
    assert report["score"] == pytest.approx(1.0)
    assert report["provenance"]["submission_sha256"]


def test_private_score_survives_unrankable_leaderboard_without_second_grader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sample = pd.DataFrame({"id": [10, 11, 12, 13], "target": [0.5, 0.5, 0.5, 0.5]})
    answers = pd.DataFrame({"id": [10, 11, 12, 13], "target": [0, 1, 0, 1]})
    _materialize_competition(
        tmp_path,
        "tabular-playground-series-may-2022",
        sample,
        answers,
    )
    submission_path = tmp_path / "score_only_submission.csv"
    pd.DataFrame({"id": [10, 11, 12, 13], "target": [0.1, 0.9, 0.2, 0.8]}).to_csv(
        submission_path,
        index=False,
    )
    leaderboard_path = tmp_path / "leaderboard_without_score.csv"
    pd.DataFrame({"scoreNullable": [0.95, 0.94]}).to_csv(leaderboard_path, index=False)
    answers_path = (
        tmp_path
        / "tabular-playground-series-may-2022"
        / "prepared"
        / "private"
        / "test.csv"
    )
    calls = {"grader": 0, "grade_csv": 0}

    class FakeBaseGrader:
        pass

    class FakeGrader(FakeBaseGrader):
        name = "fake-official-grader"
        grade_fn = staticmethod(lambda _submission, _answers: 1.0)

        def __call__(self, submission: pd.DataFrame, private_answers: pd.DataFrame) -> float:
            calls["grader"] += 1
            assert list(submission["id"]) == list(private_answers["id"])
            return 1.0

        def rank_score(self, _score, _leaderboard):
            raise AssertionError("Leaderboard must have a `score` column.")

        def is_lower_better(self, _leaderboard):
            raise AssertionError("Leaderboard must have a `score` column.")

    @dataclass(frozen=True)
    class FakeCompetition:
        id: str
        leaderboard: Path
        answers: Path
        grader: FakeBaseGrader

    competition = FakeCompetition(
        id="tabular-playground-series-may-2022",
        leaderboard=leaderboard_path,
        answers=answers_path,
        grader=FakeGrader(),
    )

    class FakeRegistry:
        def __init__(self, _data_root: Path) -> None:
            pass

        def get_competition(self, competition_id: str):
            assert competition_id == competition.id
            return competition

    def failing_after_score_grade_csv(submission: Path, selected_competition):
        calls["grade_csv"] += 1
        score = selected_competition.grader(
            pd.read_csv(submission),
            pd.read_csv(selected_competition.answers),
        )
        assert score == pytest.approx(1.0)
        raise AssertionError("Leaderboard must have a `score` column.")

    package = ModuleType("mlebench")
    package.__path__ = []  # type: ignore[attr-defined]
    grade_module = ModuleType("mlebench.grade")
    grade_module.grade_csv = failing_after_score_grade_csv  # type: ignore[attr-defined]
    grade_helpers_module = ModuleType("mlebench.grade_helpers")
    grade_helpers_module.Grader = FakeBaseGrader  # type: ignore[attr-defined]
    registry_module = ModuleType("mlebench.registry")
    registry_module.Registry = FakeRegistry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlebench", package)
    monkeypatch.setitem(sys.modules, "mlebench.grade", grade_module)
    monkeypatch.setitem(sys.modules, "mlebench.grade_helpers", grade_helpers_module)
    monkeypatch.setitem(sys.modules, "mlebench.registry", registry_module)

    report = grade_private_submission(
        submission_path,
        "tabular-playground-series-may-2022",
        tmp_path,
    )

    assert report["status"] == "passed"
    assert report["score"] == pytest.approx(1.0)
    assert report["official_mlebench_grader_executed"] is True
    assert report["provenance"]["official_grader_invocation_count"] == 1
    assert report["provenance"]["upstream_grade_csv_invocation_count"] == 1
    assert calls == {"grader": 1, "grade_csv": 1}
    assert report["ranking_error"]["reason"] == "Leaderboard must have a `score` column."
    assert report["rank_and_medal_evidence_available"] is False
    assert report["medal_claim_allowed"] is False
    assert report["official_rank_or_medal_claim_allowed"] is False
    assert report["exactly_once_contract_satisfied"] is True
    assert report["upstream_report"]["any_medal"] is False


def test_private_grader_proxy_blocks_an_upstream_second_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    sample = pd.DataFrame({"id": [10, 11], "target": [0.5, 0.5]})
    answers = pd.DataFrame({"id": [10, 11], "target": [0, 1]})
    _materialize_competition(
        tmp_path,
        "tabular-playground-series-may-2022",
        sample,
        answers,
    )
    submission_path = tmp_path / "double_call_submission.csv"
    pd.DataFrame({"id": [10, 11], "target": [0.1, 0.9]}).to_csv(submission_path, index=False)
    answers_path = (
        tmp_path
        / "tabular-playground-series-may-2022"
        / "prepared"
        / "private"
        / "test.csv"
    )
    leaderboard_path = tmp_path / "leaderboard.csv"
    pd.DataFrame({"score": [0.95, 0.90]}).to_csv(leaderboard_path, index=False)
    underlying_calls = 0

    class FakeBaseGrader:
        pass

    class FakeGrader(FakeBaseGrader):
        name = "fake-official-grader"
        grade_fn = staticmethod(lambda _submission, _answers: 1.0)

        def __call__(self, _submission, _answers):
            nonlocal underlying_calls
            underlying_calls += 1
            return 1.0

        def rank_score(self, _score, _leaderboard):
            return {}

        def is_lower_better(self, _leaderboard):
            return False

    @dataclass(frozen=True)
    class FakeCompetition:
        id: str
        leaderboard: Path
        answers: Path
        grader: FakeBaseGrader

    competition = FakeCompetition(
        id="tabular-playground-series-may-2022",
        leaderboard=leaderboard_path,
        answers=answers_path,
        grader=FakeGrader(),
    )

    class FakeRegistry:
        def __init__(self, _data_root: Path) -> None:
            pass

        def get_competition(self, _competition_id: str):
            return competition

    def double_call_grade_csv(submission: Path, selected_competition):
        submission_frame = pd.read_csv(submission)
        private_answers = pd.read_csv(selected_competition.answers)
        selected_competition.grader(submission_frame, private_answers)
        selected_competition.grader(submission_frame, private_answers)
        raise AssertionError("unreachable")

    package = ModuleType("mlebench")
    package.__path__ = []  # type: ignore[attr-defined]
    grade_module = ModuleType("mlebench.grade")
    grade_module.grade_csv = double_call_grade_csv  # type: ignore[attr-defined]
    grade_helpers_module = ModuleType("mlebench.grade_helpers")
    grade_helpers_module.Grader = FakeBaseGrader  # type: ignore[attr-defined]
    registry_module = ModuleType("mlebench.registry")
    registry_module.Registry = FakeRegistry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlebench", package)
    monkeypatch.setitem(sys.modules, "mlebench.grade", grade_module)
    monkeypatch.setitem(sys.modules, "mlebench.grade_helpers", grade_helpers_module)
    monkeypatch.setitem(sys.modules, "mlebench.registry", registry_module)

    report = grade_private_submission(
        submission_path,
        "tabular-playground-series-may-2022",
        tmp_path,
    )

    assert underlying_calls == 1
    assert report["status"] == "grader_failed"
    assert report["score"] is None
    assert report["provenance"]["official_grader_invocation_count"] == 1
    assert report["provenance"]["upstream_grade_csv_invocation_count"] == 1
    assert report["exactly_once_contract_satisfied"] is False
    assert report["medal_claim_allowed"] is False


def test_audit_reports_failures_instead_of_hiding_them(tmp_path: Path):
    report = audit_lite22_specs(tmp_path)
    assert report["competition_count"] == 22
    assert report["passed"] == 0
    assert report["failed"] == 22
    assert report["status"] == "failed"
