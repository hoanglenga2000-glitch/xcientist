"""Evaluate a leakage-free cross-run Spooky meta candidate on public OOF data.

The evaluator combines numeric OOF/test probabilities from two completed,
duplicate-safe public-data runs.  For every held outer fold, the second-level
meta-model is fitted only on OOF rows from the other folds.  The script has no
private-label, official-grader, or Kaggle-submission integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.spooky_leakage_free_meta import (
        assemble_meta_features,
        cross_fit_logistic_meta,
        normalize_probability,
    )
except ImportError:  # Direct ``python scripts/...`` execution.
    from spooky_leakage_free_meta import (  # type: ignore[no-redef]
        assemble_meta_features,
        cross_fit_logistic_meta,
        normalize_probability,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "spooky_cross_run_meta_s42_frozen_plan_20260727.json"
)
EXPECTED_SCHEMA = "evomind.spooky.cross_run_meta_frozen_plan.v1"
TERMINAL_CURRENT_STATUSES = {
    "single_seed_gate_failed",
    "single_seed_gate_passed_confirmation_pending",
}
CLASS_COLUMNS = ("EAP", "HPL", "MWS")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_unicode(values: list[str]) -> np.ndarray:
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def validate_probability(name: str, values: np.ndarray, rows: int) -> np.ndarray:
    matrix = normalize_probability(np.asarray(values, dtype=np.float64))
    if matrix.shape != (rows, len(CLASS_COLUMNS)):
        raise ValueError(f"{name} has shape {matrix.shape}, expected {(rows, 3)}")
    return matrix


def load_numeric_bundle(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load only explicitly allowed numeric keys from an NPZ archive.

    Legacy bundles may also contain object-typed ID arrays.  Those arrays are
    intentionally ignored; row identity is verified from the public CSV files
    and the independently hashed truth/fold arrays instead.
    """

    result: dict[str, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive.files]
        if missing:
            raise ValueError(f"Bundle {path} is missing keys: {missing}")
        for key in keys:
            result[key] = np.asarray(archive[key])
    return result


def build_channels(
    prior: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Validate alignment and return named OOF/test probability channels."""

    truth = np.asarray(current["truth"], dtype=np.int64).reshape(-1)
    folds = np.asarray(current["fold"], dtype=np.int16).reshape(-1)
    if not np.array_equal(truth, np.asarray(prior["truth"], dtype=np.int64)):
        raise ValueError("Prior and current truth arrays are not identical")
    if not np.array_equal(folds, np.asarray(prior["fold"], dtype=np.int16)):
        raise ValueError("Prior and current fold assignments are not identical")
    if sorted(int(value) for value in np.unique(folds)) != [0, 1, 2, 3, 4]:
        raise ValueError("Cross-run meta requires the frozen five outer folds")

    train_rows = len(truth)
    prior_test = np.asarray(prior["transformer_test_by_fold"])
    current_test = np.asarray(current["transformer_test_by_fold"])
    if prior_test.ndim != 3 or current_test.ndim != 3:
        raise ValueError("Component test predictions must be foldwise tensors")
    if prior_test.shape[:2] != current_test.shape[:2] or prior_test.shape[0] != 5:
        raise ValueError("Prior and current foldwise test predictions are not aligned")
    test_rows = prior_test.shape[1]

    oof_channels = {
        "prior_sparse": validate_probability(
            "prior_sparse", prior["sparse_oof"], train_rows
        ),
        "prior_byte": validate_probability("prior_byte", prior["byte_oof"], train_rows),
        "prior_transformer": validate_probability(
            "prior_transformer", prior["transformer_oof"], train_rows
        ),
        "prior_candidate": validate_probability(
            "prior_candidate", prior["candidate_oof"], train_rows
        ),
        "current_transformer": validate_probability(
            "current_transformer", current["transformer_oof"], train_rows
        ),
        "current_sparse": validate_probability(
            "current_sparse", current["sparse_oof"], train_rows
        ),
        "current_candidate": validate_probability(
            "current_candidate", current["candidate_oof"], train_rows
        ),
    }
    test_channels_by_fold = {
        "prior_sparse": np.stack(
            [
                validate_probability(
                    f"prior_sparse_test_fold_{fold}",
                    prior["sparse_test_by_fold"][fold],
                    test_rows,
                )
                for fold in range(5)
            ]
        ),
        "prior_byte": np.stack(
            [
                validate_probability(
                    f"prior_byte_test_fold_{fold}",
                    prior["byte_test_by_fold"][fold],
                    test_rows,
                )
                for fold in range(5)
            ]
        ),
        "prior_transformer": np.stack(
            [
                validate_probability(
                    f"prior_transformer_test_fold_{fold}",
                    prior["transformer_test_by_fold"][fold],
                    test_rows,
                )
                for fold in range(5)
            ]
        ),
        "prior_candidate": np.repeat(
            validate_probability("prior_candidate_test", prior["candidate_test"], test_rows)[
                None, :, :
            ],
            5,
            axis=0,
        ),
        "current_transformer": np.stack(
            [
                validate_probability(
                    f"current_transformer_test_fold_{fold}",
                    current["transformer_test_by_fold"][fold],
                    test_rows,
                )
                for fold in range(5)
            ]
        ),
        "current_sparse": np.stack(
            [
                validate_probability(
                    f"current_sparse_test_fold_{fold}",
                    current["sparse_test_by_fold"][fold],
                    test_rows,
                )
                for fold in range(5)
            ]
        ),
        "current_candidate": np.repeat(
            validate_probability(
                "current_candidate_test", current["candidate_test"], test_rows
            )[None, :, :],
            5,
            axis=0,
        ),
    }
    return truth, folds, np.arange(test_rows, dtype=np.int64), oof_channels, test_channels_by_fold


def assemble_cross_run_features(
    oof_channels: dict[str, np.ndarray],
    test_channels_by_fold: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    oof_features, contract = assemble_meta_features(oof_channels)
    fold_features = []
    for fold in range(5):
        values, fold_contract = assemble_meta_features(
            {name: matrix[fold] for name, matrix in test_channels_by_fold.items()}
        )
        if fold_contract["blocks"] != contract["blocks"]:
            raise ValueError("Cross-run train/test feature contracts differ")
        fold_features.append(values)
    return oof_features, np.stack(fold_features), contract


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected cross-run frozen-plan schema")
    if plan.get("status") != "frozen_waiting_current_run":
        raise ValueError("Cross-run plan is not frozen")
    if plan.get("competition_id") != "spooky-author-identification":
        raise ValueError("Cross-run plan targets another competition")
    if plan.get("public_data_only") is not True:
        raise ValueError("Cross-run plan is not public-data-only")
    boundaries = plan.get("boundaries", {})
    expected_false = (
        "private_labels_used",
        "official_grader_executed",
        "kaggle_submission_executed",
    )
    if any(boundaries.get(name) is not False for name in expected_false):
        raise ValueError("Cross-run plan enables a forbidden boundary")
    if boundaries.get("process_signals_sent") != 0:
        raise ValueError("Cross-run plan permits process signals")

    for section, fields in (
        ("source", ("evaluator", "meta_helper")),
        ("prior", ("summary", "bundle")),
        ("current", ("plan",)),
        ("inputs", ("train", "test", "sample")),
    ):
        for field in fields:
            record = plan[section][field]
            path = Path(record["path"]).resolve()
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen artifact drift: {section}.{field}")
    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    return plan


def evaluate(plan_path: Path) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    current_summary_path = Path(plan["current"]["summary_path"]).resolve()
    if not current_summary_path.is_file():
        raise FileNotFoundError(f"Current summary is not ready: {current_summary_path}")
    current_summary = read_json(current_summary_path)
    if current_summary.get("status") not in TERMINAL_CURRENT_STATUSES:
        raise RuntimeError("Current Spooky run is not terminal")
    if current_summary.get("run_id") != plan["current"]["run_id"]:
        raise RuntimeError("Current summary run ID drifted")
    if current_summary.get("plan_sha256") != plan["current"]["plan"]["sha256"]:
        raise RuntimeError("Current summary plan hash drifted")
    prediction = current_summary.get("prediction_bundle") or {}
    current_bundle_path = Path(prediction.get("path", "")).resolve()
    if (
        not current_bundle_path.is_file()
        or prediction.get("sha256") != sha256_file(current_bundle_path)
    ):
        raise RuntimeError("Current prediction bundle is missing or hash-invalid")

    prior_path = Path(plan["prior"]["bundle"]["path"])
    prior = load_numeric_bundle(
        prior_path,
        (
            "truth",
            "fold",
            "sparse_oof",
            "byte_oof",
            "transformer_oof",
            "candidate_oof",
            "sparse_test_by_fold",
            "byte_test_by_fold",
            "transformer_test_by_fold",
            "candidate_test",
        ),
    )
    current = load_numeric_bundle(
        current_bundle_path,
        (
            "truth",
            "fold",
            "transformer_oof",
            "sparse_oof",
            "candidate_oof",
            "transformer_test_by_fold",
            "sparse_test_by_fold",
            "candidate_test",
        ),
    )
    truth, folds, _, oof_channels, test_channels = build_channels(prior, current)
    oof_features, test_features, feature_contract = assemble_cross_run_features(
        oof_channels, test_channels
    )
    meta = plan["meta"]
    candidate_oof, candidate_test, write_counts, meta_contract = cross_fit_logistic_meta(
        oof_features,
        test_features,
        truth,
        folds,
        c_value=float(meta["c_value"]),
        max_iter=int(meta["max_iter"]),
        random_state=int(meta["random_state"]),
    )

    train = pd.read_csv(Path(plan["inputs"]["train"]["path"]))
    test = pd.read_csv(Path(plan["inputs"]["test"]["path"]))
    sample = pd.read_csv(Path(plan["inputs"]["sample"]["path"]))
    class_to_index = {name: index for index, name in enumerate(CLASS_COLUMNS)}
    csv_truth = train["author"].map(class_to_index).to_numpy(dtype=np.int64)
    if not np.array_equal(csv_truth, truth):
        raise RuntimeError("Bundle truth does not match the public train CSV")
    if len(test) != len(candidate_test) or len(sample) != len(candidate_test):
        raise RuntimeError("Cross-run test rows do not match public inputs")
    if not np.array_equal(test["id"].astype(str), sample["id"].astype(str)):
        raise RuntimeError("Public test and sample IDs are not aligned")

    score = float(
        -np.log(np.clip(candidate_oof[np.arange(len(truth)), truth], 1e-15, 1.0)).mean()
    )
    threshold = float(plan["promotion_gate"]["single_seed_log_loss_threshold"])
    passed = score <= threshold
    output_dir = Path(plan["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "spooky_cross_run_meta_oof_and_test.npz"
    temporary_bundle = bundle_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_bundle,
        truth=truth,
        fold=folds,
        candidate_oof=candidate_oof,
        candidate_test=candidate_test,
        candidate_write_counts=write_counts,
        train_id=safe_unicode(train["id"].astype(str).tolist()),
        test_id=safe_unicode(test["id"].astype(str).tolist()),
    )
    temporary_bundle.replace(bundle_path)
    submission = sample.copy()
    submission.loc[:, list(CLASS_COLUMNS)] = candidate_test
    submission_path = output_dir / "candidate_submission_withheld.csv"
    submission.to_csv(submission_path, index=False)

    component_scores = {
        name: float(
            -np.log(np.clip(values[np.arange(len(truth)), truth], 1e-15, 1.0)).mean()
        )
        for name, values in oof_channels.items()
    }
    result = {
        "schema": "evomind.spooky.cross_run_meta_result.v1",
        "status": (
            "single_seed_gate_passed_confirmation_pending"
            if passed
            else "single_seed_gate_failed"
        ),
        "competition_id": "spooky-author-identification",
        "run_id": plan["output"]["run_id"],
        "plan_path": plan["_plan_path"],
        "plan_sha256": plan["_plan_sha256"],
        "prior_run_id": plan["prior"]["run_id"],
        "current_run_id": plan["current"]["run_id"],
        "component_oof_log_loss": component_scores,
        "candidate_oof_log_loss": score,
        "single_seed_threshold": threshold,
        "single_seed_passed": passed,
        "confirmation_required": True,
        "promotion_allowed": False,
        "feature_contract": feature_contract,
        "meta_contract": meta_contract,
        "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
        "submission_withheld": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
        },
        "checks": {
            "truth_and_folds_identical": True,
            "public_csv_truth_identical": True,
            "exact_once_oof": bool(np.all(write_counts == 1)),
            "finite_normalized_oof": bool(
                np.isfinite(candidate_oof).all()
                and np.allclose(candidate_oof.sum(axis=1), 1.0, atol=1e-7)
            ),
            "finite_normalized_test": bool(
                np.isfinite(candidate_test).all()
                and np.allclose(candidate_test.sum(axis=1), 1.0, atol=1e-7)
            ),
            "private_labels_unused": True,
            "official_grader_not_executed": True,
            "kaggle_submission_not_executed": True,
            "process_signals_sent": 0,
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "process_signals_sent": 0,
        "claim_boundary": "Public cross-fitted OOF evidence is not an official score or medal.",
    }
    write_json_atomic(output_dir / "cross_run_meta_result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-plan", type=Path, default=DEFAULT_PLAN)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate(args.frozen_plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
