"""Unified MLE-Bench Lite contracts used by the Phase A and Wave 0 runners.

The module deliberately separates four concerns that were previously mixed in
``gpu_batch_trainer_v1.py``:

* immutable competition specifications;
* read-only resolution of the official MLE-Bench prepared data layout;
* metric and submission-format contracts; and
* an adapter around the upstream MLE-Bench private grader.

It contains no training side effects.  Importing it never creates directories,
downloads data, launches a GPU process, or submits to Kaggle.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    log_loss,
    mean_squared_error,
    mean_squared_log_error,
    roc_auc_score,
)

DEFAULT_REMOTE_ROOT = Path("/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra")
DEFAULT_OFFICIAL_DATA_ROOT = DEFAULT_REMOTE_ROOT / "mlebench_official_data"


class MLEBenchContractError(ValueError):
    """Raised when a path, metric, or submission violates the Phase A contract."""


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: str
    prediction_kind: str
    scorer: Callable[[Any, Any], float]


@dataclass(frozen=True)
class SubmissionContract:
    sample_submission_relative: str
    answers_relative: str
    id_columns: tuple[str, ...]
    prediction_columns: tuple[str, ...] = ()
    passthrough_columns: tuple[str, ...] = ()
    numeric_predictions: bool = True
    probability_bounds: bool = False
    row_sum_to_one: bool = False
    require_exact_columns: bool = True
    require_unique_ids: bool = True


@dataclass(frozen=True)
class CompetitionSpec:
    competition_id: str
    modality: str
    task_type: str
    metric: str
    direction: str
    adapter_family: str
    compute_band: str
    split_strategy: str
    target_columns: tuple[str, ...]
    train_candidates: tuple[str, ...]
    test_candidates: tuple[str, ...]
    submission: SubmissionContract
    planned_wave: str


@dataclass(frozen=True)
class ResolvedCompetition:
    spec: CompetitionSpec
    competition_root: Path
    public_dir: Path
    private_dir: Path
    sample_submission_path: Path
    answers_path: Path
    sample_columns: tuple[str, ...]
    prediction_columns: tuple[str, ...]
    train_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition_id": self.spec.competition_id,
            "modality": self.spec.modality,
            "task_type": self.spec.task_type,
            "metric": self.spec.metric,
            "direction": self.spec.direction,
            "adapter_family": self.spec.adapter_family,
            "compute_band": self.spec.compute_band,
            "split_strategy": self.spec.split_strategy,
            "planned_wave": self.spec.planned_wave,
            "competition_root": str(self.competition_root),
            "public_dir": str(self.public_dir),
            "private_dir": str(self.private_dir),
            "sample_submission_path": str(self.sample_submission_path),
            "answers_path": str(self.answers_path),
            "sample_columns": list(self.sample_columns),
            "id_columns": list(self.spec.submission.id_columns),
            "prediction_columns": list(self.prediction_columns),
            "train_paths": [str(path) for path in self.train_paths],
            "test_paths": [str(path) for path in self.test_paths],
        }


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _mean_columnwise_roc_auc(y_true: Any, y_pred: Any) -> float:
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 2 or pred.ndim != 2 or truth.shape != pred.shape:
        raise MLEBenchContractError("mean-columnwise ROC AUC requires equally shaped 2-D arrays")
    return float(np.mean([roc_auc_score(truth[:, i], pred[:, i]) for i in range(truth.shape[1])]))


def _mean_columnwise_rmsle(y_true: Any, y_pred: Any) -> float:
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 2 or pred.ndim != 2 or truth.shape != pred.shape:
        raise MLEBenchContractError("mean-columnwise RMSLE requires equally shaped 2-D arrays")
    if np.any(truth < 0) or np.any(pred < 0):
        raise MLEBenchContractError("RMSLE inputs must be non-negative")
    return float(
        np.mean(
            [math.sqrt(mean_squared_log_error(truth[:, i], pred[:, i])) for i in range(truth.shape[1])]
        )
    )


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "roc_auc": MetricDefinition("roc_auc", "maximize", "binary_probability", lambda y, p: float(roc_auc_score(y, p))),
    "mean_columnwise_roc_auc": MetricDefinition(
        "mean_columnwise_roc_auc", "maximize", "multilabel_probability", _mean_columnwise_roc_auc
    ),
    "log_loss": MetricDefinition("log_loss", "minimize", "binary_probability", lambda y, p: float(log_loss(y, p))),
    "multiclass_log_loss": MetricDefinition(
        "multiclass_log_loss", "minimize", "multiclass_probability", lambda y, p: float(log_loss(y, p))
    ),
    "accuracy": MetricDefinition("accuracy", "maximize", "label", lambda y, p: float(accuracy_score(y, p))),
    "quadratic_weighted_kappa": MetricDefinition(
        "quadratic_weighted_kappa",
        "maximize",
        "ordinal_label",
        lambda y, p: float(cohen_kappa_score(y, p, weights="quadratic")),
    ),
    "rmse": MetricDefinition("rmse", "minimize", "continuous", _rmse),
    "mean_columnwise_rmsle": MetricDefinition(
        "mean_columnwise_rmsle", "minimize", "multioutput_continuous", _mean_columnwise_rmsle
    ),
    "token_exact_accuracy": MetricDefinition(
        "token_exact_accuracy", "maximize", "token", lambda y, p: float(accuracy_score(y, p))
    ),
}


def compute_metric(metric_name: str, y_true: Any, y_pred: Any) -> float:
    """Compute a registered metric and reject NaN/Inf outputs."""

    try:
        definition = METRIC_REGISTRY[metric_name]
    except KeyError as exc:
        raise MLEBenchContractError(f"Unknown metric: {metric_name}") from exc
    score = float(definition.scorer(y_true, y_pred))
    if not math.isfinite(score):
        raise MLEBenchContractError(f"Metric {metric_name} produced a non-finite score")
    return score


def to_proxy_score(metric_name: str, score: float) -> float:
    """Map a metric to a bounded internal comparison value without claiming rank equivalence."""

    definition = METRIC_REGISTRY[metric_name]
    value = float(score)
    if definition.direction == "maximize":
        return max(0.0, min(1.0, value))
    return 1.0 / (1.0 + max(0.0, value))


def _submission(
    sample: str,
    answers: str,
    ids: str | Sequence[str],
    *,
    predictions: Sequence[str] = (),
    passthrough: Sequence[str] = (),
    numeric: bool = True,
    probabilities: bool = False,
    row_sum: bool = False,
) -> SubmissionContract:
    id_columns = (ids,) if isinstance(ids, str) else tuple(ids)
    return SubmissionContract(
        sample_submission_relative=sample,
        answers_relative=answers,
        id_columns=id_columns,
        prediction_columns=tuple(predictions),
        passthrough_columns=tuple(passthrough),
        numeric_predictions=numeric,
        probability_bounds=probabilities,
        row_sum_to_one=row_sum,
    )


def _spec(
    competition_id: str,
    modality: str,
    task_type: str,
    metric: str,
    adapter_family: str,
    compute_band: str,
    target_columns: Sequence[str],
    train_candidates: Sequence[str],
    test_candidates: Sequence[str],
    submission: SubmissionContract,
    planned_wave: str,
    split_strategy: str = "fixed_holdout_seeded",
) -> CompetitionSpec:
    return CompetitionSpec(
        competition_id=competition_id,
        modality=modality,
        task_type=task_type,
        metric=metric,
        direction=METRIC_REGISTRY[metric].direction,
        adapter_family=adapter_family,
        compute_band=compute_band,
        split_strategy=split_strategy,
        target_columns=tuple(target_columns),
        train_candidates=tuple(train_candidates),
        test_candidates=tuple(test_candidates),
        submission=submission,
        planned_wave=planned_wave,
    )


def _build_lite22_specs() -> tuple[CompetitionSpec, ...]:
    public = "prepared/public/"
    private = "prepared/private/"
    return (
        _spec(
            "aerial-cactus-identification", "vision", "binary_classification", "roc_auc",
            "vision_binary_auc", "small", ["has_cactus"], [public + "train.csv", public + "train.zip"],
            [public + "test.zip"],
            _submission(public + "sample_submission.csv", private + "test.csv", "id", predictions=["has_cactus"], probabilities=True),
            "Wave0 smoke",
        ),
        _spec(
            "aptos2019-blindness-detection", "vision", "ordinal_classification", "quadratic_weighted_kappa",
            "vision_ordinal_qwk", "large", ["diagnosis"], [public + "train.csv", public + "train_images"],
            [public + "test.csv", public + "test_images"],
            _submission(public + "sample_submission.csv", private + "test.csv", "id_code", predictions=["diagnosis"]),
            "Wave2",
        ),
        _spec(
            "denoising-dirty-documents", "vision", "image_restoration", "rmse", "vision_restoration_rmse",
            "small", ["value"], [public + "train", public + "train_cleaned"], [public + "test"],
            _submission(public + "sampleSubmission.csv", private + "answers.csv", "id", predictions=["value"]),
            "Wave1",
        ),
        _spec(
            "detecting-insults-in-social-commentary", "text", "binary_classification", "roc_auc", "text_binary_auc",
            "small", ["Insult"], [public + "train.csv"], [public + "test.csv"],
            _submission(
                public + "sample_submission_null.csv", private + "test.csv", "Comment", predictions=["Insult"],
                passthrough=["Date"], probabilities=True,
            ),
            "Wave1",
        ),
        _spec(
            "dog-breed-identification", "vision", "multiclass_classification", "multiclass_log_loss",
            "vision_multiclass_logloss", "medium", ["breed"], [public + "labels.csv", public + "train"],
            [public + "test"], _submission(public + "sample_submission.csv", private + "test.csv", "id", probabilities=True, row_sum=True),
            "Wave2",
        ),
        _spec(
            "dogs-vs-cats-redux-kernels-edition", "vision", "binary_classification", "log_loss", "vision_binary_logloss",
            "medium", ["label"], [public + "train", public + "train.zip"], [public + "test", public + "test.zip"],
            _submission(public + "sample_submission.csv", private + "answers.csv", "id", predictions=["label"], probabilities=True),
            "Wave1",
        ),
        _spec(
            "histopathologic-cancer-detection", "vision", "binary_classification", "roc_auc", "vision_binary_auc",
            "large", ["label"], [public + "train_labels.csv", public + "train"], [public + "test"],
            _submission(public + "sample_submission.csv", private + "answers.csv", "id", predictions=["label"], probabilities=True),
            "Wave2",
        ),
        _spec(
            "jigsaw-toxic-comment-classification-challenge", "text", "multilabel_classification", "mean_columnwise_roc_auc",
            "text_multilabel_mean_auc", "medium", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
            [public + "train.csv"], [public + "test.csv"],
            _submission(public + "sample_submission.csv", private + "test.csv", "id", probabilities=True), "Wave2",
        ),
        _spec(
            "leaf-classification", "tabular_plus_image", "multiclass_classification", "multiclass_log_loss",
            "tabular_multiclass_logloss", "small", ["species"], [public + "train.csv", public + "images"],
            [public + "test.csv"], _submission(public + "sample_submission.csv", private + "test.csv", "id", probabilities=True, row_sum=True),
            "Wave1",
        ),
        _spec(
            "mlsp-2013-birds", "audio", "binary_classification", "roc_auc", "audio_binary_auc", "small",
            ["Probability"], [public + "essential_data", public + "supplemental_data"], [public + "essential_data"],
            _submission(public + "sample_submission.csv", private + "answers.csv", "Id", predictions=["Probability"], probabilities=True),
            "Wave2",
        ),
        _spec(
            "new-york-city-taxi-fare-prediction", "tabular", "regression", "rmse", "tabular_regression_rmse", "medium",
            ["fare_amount"], [public + "labels.csv"], [public + "test.csv"],
            _submission(public + "sample_submission.csv", private + "test.csv", "key", predictions=["fare_amount"]), "Wave1",
        ),
        _spec(
            "nomad2018-predict-transparent-conductors", "tabular_materials", "multioutput_regression",
            "mean_columnwise_rmsle", "tabular_multioutput_rmsle", "small",
            ["formation_energy_ev_natom", "bandgap_energy_ev"], [public + "train.csv", public + "train"],
            [public + "test.csv", public + "test"],
            _submission(
                public + "sample_submission.csv", private + "test.csv", "id",
                predictions=["formation_energy_ev_natom", "bandgap_energy_ev"],
            ),
            "Wave2",
        ),
        _spec(
            "plant-pathology-2020-fgvc7", "vision", "multiclass_classification", "mean_columnwise_roc_auc",
            "vision_multiclass_mean_auc", "medium", ["healthy", "multiple_diseases", "rust", "scab"],
            [public + "train.csv", public + "images"], [public + "test.csv", public + "images"],
            _submission(public + "sample_submission.csv", private + "test.csv", "image_id", probabilities=True, row_sum=True), "Wave2",
        ),
        _spec(
            "random-acts-of-pizza", "text_plus_tabular", "binary_classification", "roc_auc", "text_tabular_binary_auc",
            "small", ["requester_received_pizza"], [public + "train.json"], [public + "test.json"],
            _submission(
                public + "sampleSubmission.csv", private + "test.csv", "request_id",
                predictions=["requester_received_pizza"], probabilities=True,
            ),
            "Wave1",
        ),
        _spec(
            "ranzcr-clip-catheter-line-classification", "vision", "multilabel_classification", "mean_columnwise_roc_auc",
            "vision_multilabel_mean_auc", "large", [], [public + "train.csv", public + "train"], [public + "test"],
            _submission(public + "sample_submission.csv", private + "test.csv", "StudyInstanceUID", probabilities=True), "Wave2",
        ),
        _spec(
            "siim-isic-melanoma-classification", "vision_plus_tabular", "binary_classification", "roc_auc",
            "vision_tabular_binary_auc", "very_large", ["target"], [public + "train.csv", public + "jpeg", public + "train"],
            [public + "test.csv", public + "jpeg", public + "test"],
            _submission(public + "sample_submission.csv", private + "test.csv", "image_name", predictions=["target"], probabilities=True),
            "Wave0 precheck only",
        ),
        _spec(
            "spooky-author-identification", "text", "multiclass_classification", "multiclass_log_loss",
            "text_multiclass_logloss", "small", ["author"], [public + "train.csv"], [public + "test.csv"],
            _submission(public + "sample_submission.csv", private + "test.csv", "id", probabilities=True, row_sum=True), "Wave0 smoke",
        ),
        _spec(
            "tabular-playground-series-dec-2021", "tabular", "multiclass_classification", "accuracy",
            "tabular_multiclass_accuracy", "medium", ["Cover_Type"], [public + "train.csv"], [public + "test.csv"],
            _submission(public + "sample_submission.csv", private + "test.csv", "Id", predictions=["Cover_Type"]), "Wave1",
        ),
        _spec(
            "tabular-playground-series-may-2022", "tabular", "binary_classification", "roc_auc", "tabular_binary_auc",
            "medium", ["target"], [public + "train.csv"], [public + "test.csv"],
            _submission(public + "sample_submission.csv", private + "test.csv", "id", predictions=["target"], probabilities=True),
            "Wave0 smoke",
        ),
        _spec(
            "text-normalization-challenge-english-language", "text_sequence", "token_normalization", "token_exact_accuracy",
            "text_normalization_rules_seq2seq", "small", ["after"], [public + "en_train.csv.zip"], [public + "en_test_2.csv.zip"],
            _submission(private + "sample_submission.csv", private + "answers.csv", "id", predictions=["after"], numeric=False), "Wave2",
        ),
        _spec(
            "text-normalization-challenge-russian-language", "text_sequence", "token_normalization", "token_exact_accuracy",
            "text_normalization_rules_seq2seq", "small", ["after"], [public + "ru_train.csv.zip"], [public + "ru_test_2.csv.zip"],
            _submission(private + "sample_submission.csv", private + "answers.csv", "id", predictions=["after"], numeric=False), "Wave2",
        ),
        _spec(
            "the-icml-2013-whale-challenge-right-whale-redux", "audio", "binary_classification", "roc_auc",
            "audio_binary_auc", "small", ["probability"], [public + "train2.zip"], [public + "test2.zip"],
            _submission(public + "sampleSubmission.csv", private + "test.csv", "clip", predictions=["probability"], probabilities=True),
            "Wave2",
        ),
    )


LITE22_SPECS: tuple[CompetitionSpec, ...] = _build_lite22_specs()
LITE22_BY_ID: dict[str, CompetitionSpec] = {spec.competition_id: spec for spec in LITE22_SPECS}

if len(LITE22_SPECS) != 22 or len(LITE22_BY_ID) != 22:  # import-time invariant, no side effect
    raise RuntimeError("The MLE-Bench Lite registry must contain exactly 22 unique competitions")


def get_competition_spec(competition_id: str) -> CompetitionSpec:
    try:
        return LITE22_BY_ID[competition_id]
    except KeyError as exc:
        raise MLEBenchContractError(f"Competition is not in MLE-Bench Lite 22: {competition_id}") from exc


def _safe_child(root: Path, *parts: str) -> Path:
    base = Path(root).expanduser().resolve()
    target = base.joinpath(*parts).resolve()
    if target != base and base not in target.parents:
        raise MLEBenchContractError(f"Resolved path escapes the configured root: {target}")
    return target


def _read_csv_header(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return tuple(next(csv.reader(handle), []))


def resolve_competition(
    competition: str | CompetitionSpec,
    data_root: str | Path = DEFAULT_OFFICIAL_DATA_ROOT,
    *,
    require_exists: bool = True,
    require_private: bool | None = None,
) -> ResolvedCompetition:
    if require_private is None:
        public_only = os.environ.get("EVOMIND_MLEBENCH_PUBLIC_ONLY", "").strip().lower()
        require_private = public_only not in {"1", "true", "yes"}
    spec = get_competition_spec(competition) if isinstance(competition, str) else competition
    root = Path(data_root).expanduser().resolve()
    competition_root = _safe_child(root, spec.competition_id)
    public_dir = _safe_child(competition_root, "prepared", "public")
    private_dir = _safe_child(competition_root, "prepared", "private")
    sample_path = _safe_child(competition_root, spec.submission.sample_submission_relative)
    answers_path = _safe_child(competition_root, spec.submission.answers_relative)
    sample_columns = _read_csv_header(sample_path)

    excluded = set(spec.submission.id_columns) | set(spec.submission.passthrough_columns)
    prediction_columns = spec.submission.prediction_columns or tuple(
        column for column in sample_columns if column not in excluded
    )
    train_paths = tuple(
        path for path in (_safe_child(competition_root, candidate) for candidate in spec.train_candidates) if path.exists()
    )
    test_paths = tuple(
        path for path in (_safe_child(competition_root, candidate) for candidate in spec.test_candidates) if path.exists()
    )

    if require_exists:
        missing = []
        required_paths = [
            ("competition_root", competition_root),
            ("public_dir", public_dir),
            ("sample_submission", sample_path),
        ]
        if require_private:
            required_paths.extend(
                [("private_dir", private_dir), ("answers", answers_path)]
            )
        for label, path in required_paths:
            if not path.exists():
                missing.append(f"{label}={path}")
        if missing:
            raise MLEBenchContractError("Missing official prepared paths: " + "; ".join(missing))
        if not sample_columns:
            raise MLEBenchContractError(f"Sample submission has no CSV header: {sample_path}")
        required_columns = set(spec.submission.id_columns) | set(spec.submission.passthrough_columns)
        required_columns |= set(spec.submission.prediction_columns)
        absent = sorted(required_columns - set(sample_columns))
        if absent:
            raise MLEBenchContractError(f"Sample submission is missing contract columns: {absent}")
        if not prediction_columns:
            raise MLEBenchContractError(f"No prediction columns resolved for {spec.competition_id}")
        if not train_paths or not test_paths:
            raise MLEBenchContractError(
                f"Input candidates did not resolve for {spec.competition_id}: "
                f"train={list(spec.train_candidates)}, test={list(spec.test_candidates)}"
            )

    return ResolvedCompetition(
        spec=spec,
        competition_root=competition_root,
        public_dir=public_dir,
        private_dir=private_dir,
        sample_submission_path=sample_path,
        answers_path=answers_path,
        sample_columns=sample_columns,
        prediction_columns=tuple(prediction_columns),
        train_paths=train_paths,
        test_paths=test_paths,
    )


def audit_lite22_specs(data_root: str | Path = DEFAULT_OFFICIAL_DATA_ROOT) -> dict[str, Any]:
    """Run the low-I/O Phase A path/schema audit for all 22 competitions."""

    rows: list[dict[str, Any]] = []
    for spec in LITE22_SPECS:
        try:
            resolved = resolve_competition(spec, data_root, require_exists=True)
            rows.append({"competition_id": spec.competition_id, "status": "passed", **resolved.to_dict()})
        except Exception as exc:
            rows.append(
                {
                    "competition_id": spec.competition_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    passed = sum(row["status"] == "passed" for row in rows)
    return {
        "schema": "evomind.mlebench_lite22.phase_a_audit.v1",
        "data_root": str(Path(data_root)),
        "competition_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "status": "passed" if passed == len(rows) else "failed",
        "rows": rows,
    }


def _id_key_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame[list(columns)].astype(str)
    if len(columns) == 1:
        return values.iloc[:, 0]
    return values.agg("\x1f".join, axis=1)


def validate_submission_frame(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    resolved: ResolvedCompetition,
) -> dict[str, Any]:
    """Validate rows, schema, IDs, numerical safety, and probability semantics."""

    contract = resolved.spec.submission
    errors: list[str] = []
    warnings: list[str] = []
    submission_columns = list(submission.columns)
    sample_columns = list(sample.columns)
    columns_match = submission_columns == sample_columns
    if contract.require_exact_columns and not columns_match:
        errors.append("columns_or_order_mismatch")
    rows_match = len(submission) == len(sample)
    if not rows_match:
        errors.append("row_count_mismatch")

    missing_required = [
        column
        for column in (*contract.id_columns, *contract.passthrough_columns, *resolved.prediction_columns)
        if column not in submission.columns
    ]
    if missing_required:
        errors.append("missing_required_columns")

    duplicate_id_count = 0
    sample_duplicate_id_count = 0
    id_set_match = False
    id_multiset_match = False
    id_order_match = False
    if not missing_required and all(column in sample.columns for column in contract.id_columns):
        submission_ids = _id_key_frame(submission, contract.id_columns)
        sample_ids = _id_key_frame(sample, contract.id_columns)
        duplicate_id_count = int(submission_ids.duplicated().sum())
        sample_duplicate_id_count = int(sample_ids.duplicated().sum())
        submission_id_list = submission_ids.tolist()
        sample_id_list = sample_ids.tolist()
        if contract.require_unique_ids and not sample_duplicate_id_count and duplicate_id_count:
            errors.append("duplicate_ids")
        id_set_match = set(submission_id_list) == set(sample_id_list)
        id_multiset_match = Counter(submission_id_list) == Counter(sample_id_list)
        id_order_match = submission_id_list == sample_id_list
        if not id_set_match:
            errors.append("id_set_mismatch")
        elif not id_multiset_match:
            errors.append("id_multiplicity_mismatch")
        elif not id_order_match:
            warnings.append("id_order_differs_but_id_set_matches")

    missing_prediction_count = 0
    nonfinite_prediction_count = 0
    nonnumeric_prediction_count = 0
    probability_out_of_range_count = 0
    row_sum_violation_count = 0
    available_predictions = [column for column in resolved.prediction_columns if column in submission.columns]
    if available_predictions:
        raw_predictions = submission[available_predictions]
        missing_prediction_count = int(raw_predictions.isna().sum().sum())
        if missing_prediction_count:
            errors.append("missing_predictions")
        if contract.numeric_predictions:
            numeric_predictions = raw_predictions.apply(pd.to_numeric, errors="coerce")
            nonnumeric_prediction_count = int((numeric_predictions.isna() & ~raw_predictions.isna()).sum().sum())
            if nonnumeric_prediction_count:
                errors.append("nonnumeric_predictions")
            values = numeric_predictions.to_numpy(dtype=float, na_value=np.nan)
            nonfinite_prediction_count = int(np.size(values) - np.isfinite(values).sum())
            if nonfinite_prediction_count:
                errors.append("nonfinite_predictions")
            if contract.probability_bounds:
                finite = np.isfinite(values)
                probability_out_of_range_count = int(((values < 0) | (values > 1))[finite].sum())
                if probability_out_of_range_count:
                    errors.append("probability_out_of_range")
            if contract.row_sum_to_one and values.ndim == 2:
                sums = np.sum(values, axis=1)
                row_sum_violation_count = int((~np.isclose(sums, 1.0, atol=1e-6)).sum())
                if row_sum_violation_count:
                    errors.append("probability_rows_do_not_sum_to_one")
    else:
        errors.append("prediction_columns_unavailable")

    errors = list(dict.fromkeys(errors))
    return {
        "schema": "evomind.mlebench_submission_validation.v1",
        "competition_id": resolved.spec.competition_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "row_count": int(len(submission)),
        "expected_row_count": int(len(sample)),
        "rows_match": bool(rows_match),
        "columns_match": bool(columns_match),
        "submission_columns": submission_columns,
        "expected_columns": sample_columns,
        "id_columns": list(contract.id_columns),
        "id_set_match": bool(id_set_match),
        "id_multiset_match": bool(id_multiset_match),
        "id_order_match": bool(id_order_match),
        "duplicate_id_count": duplicate_id_count,
        "sample_duplicate_id_count": sample_duplicate_id_count,
        "prediction_columns": available_predictions,
        "missing_prediction_count": missing_prediction_count,
        "nonnumeric_prediction_count": nonnumeric_prediction_count,
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "probability_out_of_range_count": probability_out_of_range_count,
        "row_sum_violation_count": row_sum_violation_count,
    }


def validate_submission_file(
    submission_path: str | Path,
    competition: str | CompetitionSpec,
    data_root: str | Path = DEFAULT_OFFICIAL_DATA_ROOT,
) -> dict[str, Any]:
    resolved = resolve_competition(competition, data_root, require_exists=True)
    path = Path(submission_path)
    if not path.is_file() or path.suffix.lower() != ".csv":
        return {
            "schema": "evomind.mlebench_submission_validation.v1",
            "competition_id": resolved.spec.competition_id,
            "valid": False,
            "errors": ["submission_file_missing_or_not_csv"],
            "submission_path": str(path),
        }
    submission = pd.read_csv(path)
    sample = pd.read_csv(resolved.sample_submission_path)
    result = validate_submission_frame(submission, sample, resolved)
    result["submission_path"] = str(path)
    result["sample_submission_path"] = str(resolved.sample_submission_path)
    return result


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    marker = path / "REVISION"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return value
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def grade_private_submission(
    submission_path: str | Path,
    competition: str | CompetitionSpec,
    data_root: str | Path = DEFAULT_OFFICIAL_DATA_ROOT,
    *,
    official_source_root: str | Path | None = None,
    seed: int | None = None,
    budget: Mapping[str, Any] | None = None,
    code_paths: Iterable[str | Path] = (),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and grade through the upstream MLE-Bench implementation.

    Failures are returned as evidence instead of being hidden.  ``official`` is
    true only after importing and executing the upstream grader successfully.
    """

    spec = get_competition_spec(competition) if isinstance(competition, str) else competition
    resolved = resolve_competition(spec, data_root, require_exists=True)
    validation = validate_submission_file(submission_path, spec, data_root)
    report: dict[str, Any] = {
        "schema": "evomind.mlebench_private_grader.v1",
        "competition_id": spec.competition_id,
        "status": "invalid_submission" if not validation.get("valid") else "pending",
        "official_mlebench_grader_executed": False,
        "score": None,
        "metric": spec.metric,
        "direction": spec.direction,
        "validation": validation,
        "provenance": {
            "seed": seed,
            "budget": dict(budget or {}),
            "submission_sha256": sha256_file(submission_path) if Path(submission_path).is_file() else None,
            "answers_sha256": sha256_file(resolved.answers_path) if resolved.answers_path.is_file() else None,
            "code_sha256": {
                str(Path(path)): sha256_file(path) for path in code_paths if Path(path).is_file()
            },
            "data_root": str(Path(data_root)),
            "answers_path": str(resolved.answers_path),
        },
    }
    if validation.get("valid"):
        source_root = Path(official_source_root).resolve() if official_source_root else None
        mlebench_module_snapshot: dict[str, Any] | None = None
        if source_root is not None:
            # A prior in-process readiness probe may have imported a different
            # pinned/fake ``mlebench`` tree.  Python otherwise reuses those
            # cached submodules even after this source root is put first on
            # sys.path, producing order-dependent grading.  Isolate the whole
            # namespace and restore it byte-for-byte after this single grade.
            mlebench_module_snapshot = {
                name: module
                for name, module in sys.modules.items()
                if name == "mlebench" or name.startswith("mlebench.")
            }
            for name in list(mlebench_module_snapshot):
                sys.modules.pop(name, None)
            sys.path.insert(0, str(source_root))
            report["provenance"]["mlebench_revision"] = _git_revision(source_root)
        official_grader_invocation_count = 0
        grade_csv_invocation_count = 0
        try:
            from mlebench.grade import grade_csv  # type: ignore
            from mlebench.grade_helpers import Grader  # type: ignore
            from mlebench.registry import Registry  # type: ignore

            registry = Registry(Path(data_root))
            official_competition = registry.get_competition(spec.competition_id)

            missing = object()

            class SingleInvocationGrader(Grader):
                """Capture one official score and reject any second attempt."""

                def __init__(self, delegate: Grader) -> None:
                    self._delegate = delegate
                    self.name = delegate.name
                    self.grade_fn = delegate.grade_fn
                    self.call_attempts = 0
                    self.delegate_invocations = 0
                    self.returned = False
                    self.captured_score: Any = missing

                def __call__(self, submission: pd.DataFrame, answers: Any) -> Any:
                    self.call_attempts += 1
                    if self.call_attempts != 1:
                        raise MLEBenchContractError("official competition grader attempted more than once")
                    self.delegate_invocations += 1
                    result = self._delegate(submission, answers)
                    self.captured_score = result
                    self.returned = True
                    return result

                def rank_score(self, score: Any, leaderboard: pd.DataFrame) -> dict[str, Any]:
                    return self._delegate.rank_score(score, leaderboard)

                def is_lower_better(self, leaderboard: pd.DataFrame) -> bool:
                    return self._delegate.is_lower_better(leaderboard)

            capture = SingleInvocationGrader(official_competition.grader)
            guarded_competition = replace(official_competition, grader=capture)

            def finite_score(value: Any) -> bool:
                if value is missing or value is None or isinstance(value, (bool, np.bool_)):
                    return False
                try:
                    return math.isfinite(float(value))
                except (TypeError, ValueError):
                    return False

            def score_only_report(value: Any) -> dict[str, Any]:
                return {
                    "competition_id": official_competition.id,
                    "score": float(value),
                    "valid_submission": True,
                    "submission_exists": True,
                    "submission_path": str(Path(submission_path)),
                    "rank_status": "unavailable",
                    "ranking_evidence_available": False,
                    "gold_threshold": None,
                    "silver_threshold": None,
                    "bronze_threshold": None,
                    "median_threshold": None,
                    "any_medal": False,
                    "gold_medal": False,
                    "silver_medal": False,
                    "bronze_medal": False,
                    "above_median": False,
                    "is_lower_better": None,
                }

            upstream_dict: dict[str, Any] | None = None
            ranking_error: Exception | None = None
            try:
                grade_csv_invocation_count = 1
                upstream = grade_csv(Path(submission_path), guarded_competition)
                upstream_dict = upstream.to_dict()
            except Exception as exc:
                ranking_error = exc

            official_grader_invocation_count = capture.delegate_invocations
            exactly_once = (
                capture.call_attempts == 1
                and capture.delegate_invocations == 1
                and capture.returned
                and finite_score(capture.captured_score)
            )
            if not exactly_once:
                if capture.call_attempts > 1:
                    raise MLEBenchContractError("official competition grader attempted more than once")
                if capture.delegate_invocations == 0:
                    raise ranking_error or RuntimeError("official MLE-Bench grader did not execute")
                if not capture.returned:
                    raise ranking_error or RuntimeError("official MLE-Bench grader did not return")
                raise RuntimeError("official MLE-Bench grader returned a non-finite score")

            score = float(capture.captured_score)
            if ranking_error is not None:
                upstream_dict = score_only_report(score)
                report.update(
                    {
                        "status": "passed",
                        "score_status": "passed",
                        "official_mlebench_grader_executed": True,
                        "score": score,
                        "upstream_report": upstream_dict,
                        "upstream_grade_csv_completed": False,
                        "ranking_status": "unavailable",
                        "ranking_error": {
                            "error_type": type(ranking_error).__name__,
                            "reason": str(ranking_error),
                        },
                        "rank_and_medal_evidence_available": False,
                        "historical_mlebench_rank_evidence_available": False,
                        "rank_claim_allowed": False,
                        "medal_claim_allowed": False,
                        "official_rank_or_medal_claim_allowed": False,
                        "exactly_once_contract_satisfied": True,
                    }
                )
            else:
                if upstream_dict is None or not upstream_dict.get("valid_submission"):
                    raise RuntimeError("official MLE-Bench grader returned an invalid report")
                upstream_score = upstream_dict.get("score")
                if not finite_score(upstream_score) or not math.isclose(
                    score,
                    float(upstream_score),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise MLEBenchContractError("captured score differs from the upstream report")
                report.update(
                    {
                        "status": "passed",
                        "score_status": "passed",
                        "official_mlebench_grader_executed": True,
                        "score": score,
                        "upstream_report": upstream_dict,
                        "upstream_grade_csv_completed": True,
                        "ranking_status": "historical_mlebench_available",
                        "rank_and_medal_evidence_available": True,
                        "historical_mlebench_rank_evidence_available": True,
                        "rank_claim_allowed": False,
                        "medal_claim_allowed": False,
                        "official_rank_or_medal_claim_allowed": False,
                        "exactly_once_contract_satisfied": True,
                    }
                )
            report["provenance"].update(
                {
                    "upstream_grade_csv_invocation_count": grade_csv_invocation_count,
                    "official_grader_invocation_count": official_grader_invocation_count,
                }
            )
            try:
                report["provenance"]["mlebench_version"] = importlib.metadata.version("mlebench")
            except importlib.metadata.PackageNotFoundError:
                report["provenance"]["mlebench_version"] = "source-tree"
        except Exception as exc:
            report.update(
                {
                    "status": "grader_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "score": None,
                    "medal_claim_allowed": False,
                    "official_rank_or_medal_claim_allowed": False,
                    "exactly_once_contract_satisfied": False,
                }
            )
            report["provenance"].update(
                {
                    "upstream_grade_csv_invocation_count": grade_csv_invocation_count,
                    "official_grader_invocation_count": official_grader_invocation_count,
                }
            )
        finally:
            if source_root is not None:
                try:
                    sys.path.remove(str(source_root))
                except ValueError:
                    pass
                for name in list(sys.modules):
                    if name == "mlebench" or name.startswith("mlebench."):
                        sys.modules.pop(name, None)
                if mlebench_module_snapshot is not None:
                    sys.modules.update(mlebench_module_snapshot)

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def export_specs(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "evomind.mlebench_lite22.specs.v1",
        "competition_count": len(LITE22_SPECS),
        "metrics": {
            name: {
                "direction": definition.direction,
                "prediction_kind": definition.prediction_kind,
            }
            for name, definition in METRIC_REGISTRY.items()
        },
        "competitions": [asdict(spec) for spec in LITE22_SPECS],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
