#!/usr/bin/env python3
"""Bounded public-OOF CPU diagnostics for TPS May 2022 feature/model upgrades."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts import mlebench_medal_recovery_adapters as recovery  # noqa: E402

COMPETITION_ID = "tabular-playground-series-may-2022"
NUMERIC_COLUMNS = tuple(f"f_{index:02d}" for index in range(31) if index != 27)
INTEGER_IDENTITY_COLUMNS = tuple(
    [f"f_{index:02d}" for index in range(7, 19)] + ["f_29", "f_30"]
)
CONTINUOUS_COLUMNS = tuple(
    column for column in NUMERIC_COLUMNS if column not in INTEGER_IDENTITY_COLUMNS
)
F27_ALPHABET = recovery.MAY2022_F27_ALPHABET
F27_WIDTH = recovery.MAY2022_F27_WIDTH


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_public_frame(frame: pd.DataFrame, *, require_target: bool) -> None:
    expected = {"id", "f_27", *NUMERIC_COLUMNS}
    if require_target:
        expected.add("target")
    missing = sorted(expected - set(frame.columns))
    unexpected_features = sorted(
        column
        for column in frame.columns
        if column.startswith("f_") and column not in {*NUMERIC_COLUMNS, "f_27"}
    )
    if missing or unexpected_features:
        raise RuntimeError(
            f"May 2022 raw schema differs: missing={missing} unexpected_features={unexpected_features}"
        )
    if frame["id"].duplicated().any():
        raise RuntimeError("May 2022 IDs are duplicated")
    numeric = frame[list(NUMERIC_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("May 2022 numeric input contains non-finite values")
    recovery._encode_may2022_f27(frame["f_27"])
    if require_target:
        target = frame["target"].to_numpy()
        if pd.isna(target).any() or set(np.unique(target).tolist()) != {0, 1}:
            raise RuntimeError("May 2022 target must contain exactly binary labels {0,1}")


def _fit_category_codes(
    fit_values: pd.Series,
    validation_values: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    fit_strings = fit_values.astype(str)
    validation_strings = validation_values.astype(str)
    categories = sorted(fit_strings.unique().tolist())
    mapping = {value: index for index, value in enumerate(categories)}
    fit_code = fit_strings.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
    validation_code = validation_strings.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
    counts = pd.Series(fit_code).value_counts(dropna=False)
    fit_frequency = pd.Series(fit_code).map(counts).fillna(0).to_numpy(dtype=np.float32) / len(fit_code)
    validation_frequency = (
        pd.Series(validation_code).map(counts).fillna(0).to_numpy(dtype=np.float32)
        / len(fit_code)
    )
    return fit_code, validation_code, fit_frequency, validation_frequency, mapping


def build_fold_categorical_features(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Build fold-local categorical/frequency features without target access."""

    validate_public_frame(fit, require_target=True)
    validate_public_frame(validation, require_target=True)
    fit_features = fit[list(NUMERIC_COLUMNS)].astype(np.float32).copy()
    validation_features = validation[list(NUMERIC_COLUMNS)].astype(np.float32).copy()
    categorical_names: list[str] = []
    mapping_sizes: dict[str, int] = {}

    for column in INTEGER_IDENTITY_COLUMNS:
        (
            fit_code,
            validation_code,
            fit_frequency,
            validation_frequency,
            mapping,
        ) = _fit_category_codes(fit[column], validation[column])
        categorical_name = f"cat__{column}"
        fit_features[categorical_name] = fit_code
        validation_features[categorical_name] = validation_code
        fit_features[f"freq__{column}"] = fit_frequency
        validation_features[f"freq__{column}"] = validation_frequency
        categorical_names.append(categorical_name)
        mapping_sizes[column] = len(mapping)

    fit_encoded = recovery._encode_may2022_f27(fit["f_27"])
    validation_encoded = recovery._encode_may2022_f27(validation["f_27"])
    fit_strings = fit["f_27"].astype(str)
    validation_strings = validation["f_27"].astype(str)
    for position in range(F27_WIDTH):
        fit_position = fit_strings.str[position]
        validation_position = validation_strings.str[position]
        (
            fit_code,
            validation_code,
            fit_frequency,
            validation_frequency,
            mapping,
        ) = _fit_category_codes(fit_position, validation_position)
        name = f"cat__f27_pos_{position}"
        fit_features[name] = fit_code
        validation_features[name] = validation_code
        fit_features[f"freq__f27_pos_{position}"] = fit_frequency
        validation_features[f"freq__f27_pos_{position}"] = validation_frequency
        categorical_names.append(name)
        mapping_sizes[f"f27_pos_{position}"] = len(mapping)

    for letter_index, letter in enumerate(F27_ALPHABET):
        fit_features[f"f27_count_{letter}"] = (fit_encoded == letter_index).sum(axis=1)
        validation_features[f"f27_count_{letter}"] = (
            validation_encoded == letter_index
        ).sum(axis=1)
    for output, encoded in (
        (fit_features, fit_encoded),
        (validation_features, validation_encoded),
    ):
        output["f27_unique_count"] = np.apply_along_axis(
            lambda row: len(np.unique(row)), 1, encoded
        )
        output["f27_repeat_count"] = F27_WIDTH - output["f27_unique_count"]
        output["f27_adjacent_equal_count"] = (encoded[:, 1:] == encoded[:, :-1]).sum(axis=1)
        output["f27_code_mean"] = encoded.mean(axis=1)
        output["f27_code_std"] = encoded.std(axis=1)

        continuous = output[list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float32)
        output["continuous_row_mean"] = continuous.mean(axis=1)
        output["continuous_row_std"] = continuous.std(axis=1)
        output["continuous_row_min"] = continuous.min(axis=1)
        output["continuous_row_max"] = continuous.max(axis=1)
        output["continuous_row_l2"] = np.sqrt(np.square(continuous).sum(axis=1))
        output["continuous_positive_count"] = (continuous > 0).sum(axis=1)

        sum_02_21 = output["f_02"] + output["f_21"]
        sum_05_22 = output["f_05"] + output["f_22"]
        sum_00_01_26 = output["f_00"] + output["f_01"] + output["f_26"]
        output["sum_f02_f21"] = sum_02_21
        output["sum_f05_f22"] = sum_05_22
        output["sum_f00_f01_f26"] = sum_00_01_26
        output["interaction_f02_f21"] = (sum_02_21 > 5.2).astype(np.int8) - (
            sum_02_21 < -5.3
        ).astype(np.int8)
        output["interaction_f05_f22"] = (sum_05_22 > 5.1).astype(np.int8) - (
            sum_05_22 < -5.4
        ).astype(np.int8)
        output["interaction_f00_f01_f26"] = (sum_00_01_26 > 5.0).astype(np.int8) - (
            sum_00_01_26 < -5.0
        ).astype(np.int8)

    if fit_features.columns.tolist() != validation_features.columns.tolist():
        raise RuntimeError("May 2022 CPU diagnostic schemas differ")
    if not np.isfinite(fit_features.to_numpy(dtype=np.float64)).all() or not np.isfinite(
        validation_features.to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("May 2022 CPU diagnostic features are non-finite")
    schema = {
        "columns": fit_features.columns.tolist(),
        "dtypes": [str(dtype) for dtype in fit_features.dtypes],
        "categorical_names": categorical_names,
    }
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return fit_features, validation_features, categorical_names, {
        "feature_count": fit_features.shape[1],
        "categorical_feature_count": len(categorical_names),
        "category_mapping_sizes": mapping_sizes,
        "feature_schema": schema,
        "feature_schema_sha256": schema_sha256,
        "target_derived_features": 0,
        "frequency_fit_scope": "outer_fit_only",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import roc_auc_score

    public = args.data_root / COMPETITION_ID / "prepared" / "public"
    train_path = public / "train.csv"
    train = pd.read_csv(train_path)
    validate_public_frame(train, require_target=True)
    with np.load(args.oof_bundle, allow_pickle=False) as archive:
        required = {"id", "target", "fold_assignment", "oof_probability"}
        if not required.issubset(archive.files):
            raise RuntimeError(f"May 2022 OOF bundle misses arrays: {sorted(required - set(archive.files))}")
        ids = np.asarray(archive["id"])
        target = np.asarray(archive["target"], dtype=np.int8)
        folds = np.asarray(archive["fold_assignment"], dtype=np.int16)
        incumbent = np.asarray(archive["oof_probability"], dtype=np.float64)
    if not np.array_equal(ids, train["id"].to_numpy()) or not np.array_equal(
        target, train["target"].to_numpy(dtype=np.int8)
    ):
        raise RuntimeError("May 2022 OOF bundle does not align with public train.csv")
    validation_mask = folds == int(args.fold)
    fit_mask = ~validation_mask
    if not validation_mask.any() or set(target[validation_mask].tolist()) != {0, 1}:
        raise RuntimeError("May 2022 requested validation fold is invalid")

    fit_features, validation_features, categorical_names, diagnostics = (
        build_fold_categorical_features(
            train.loc[fit_mask].reset_index(drop=True),
            train.loc[validation_mask].reset_index(drop=True),
        )
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(args.max_rounds),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=-1,
        min_child_samples=80,
        subsample=0.92,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.02,
        reg_lambda=3.0,
        max_bin=255,
        random_state=int(args.seed),
        n_jobs=int(args.threads),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        fit_features,
        target[fit_mask],
        eval_set=[(validation_features, target[validation_mask])],
        eval_metric="auc",
        categorical_feature=categorical_names,
        callbacks=[
            early_stopping(int(args.early_stopping_rounds), first_metric_only=True, verbose=False),
            log_evaluation(int(args.log_period)),
        ],
    )
    probability = model.predict_proba(
        validation_features,
        num_iteration=model.best_iteration_,
    )[:, 1]
    model_path = args.output.with_suffix(".lightgbm.txt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    incumbent_auc = float(roc_auc_score(target[validation_mask], incumbent[validation_mask]))
    diagnostic_auc = float(roc_auc_score(target[validation_mask], probability))
    payload = {
        "schema": "evomind.mlebench.may2022_cpu_feature_diagnostic.v1",
        "created_at": now_iso(),
        "status": "completed",
        "competition_id": COMPETITION_ID,
        "visibility_mode": "PUBLIC_ONLY",
        "data": {
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "rows": len(train),
            "fit_rows": int(fit_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "fold": int(args.fold),
        },
        "incumbent_same_fold_auc": incumbent_auc,
        "diagnostic_same_fold_auc": diagnostic_auc,
        "gain_over_incumbent": diagnostic_auc - incumbent_auc,
        "bronze_reference_auc": recovery.MAY2022_FOLD_PROMOTION_AUC,
        "bronze_reference_cleared": diagnostic_auc >= recovery.MAY2022_FOLD_PROMOTION_AUC,
        "model": {
            "family": "LightGBM_CPU_explicit_categories_fold_local_frequency",
            "best_iteration": int(model.best_iteration_),
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "features": diagnostics,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "threads": int(args.threads),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "input_oof_bundle": {
            "path": str(args.oof_bundle),
            "sha256": sha256_file(args.oof_bundle),
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": "One public OOF fold diagnostic is not an official medal or a promotion candidate.",
    }
    write_json_atomic(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--oof-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--max-rounds", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--early-stopping-rounds", type=int, default=180)
    parser.add_argument("--log-period", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
