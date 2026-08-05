"""Independently reconstruct and verify a completed Spooky DeBERTa OOF v2 run."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.run_spooky_deberta_oof_v2 import (
        build_meta_inputs,
        index_sha256,
        safe_unicode_ids,
        verify_source_contract,
    )
    from scripts.run_spooky_transformer_oof import (
        CLASS_COLUMNS,
        build_duplicate_safe_folds,
        multiclass_log_loss,
        normalize_probability,
        now_iso,
        sha256_file,
    )
    from scripts.spooky_leakage_free_meta import cross_fit_logistic_meta
    from scripts.spooky_style_features import extract_style_features
except ImportError:  # Direct ``python scripts/...`` execution.
    from run_spooky_deberta_oof_v2 import (  # type: ignore[no-redef]
        build_meta_inputs,
        index_sha256,
        safe_unicode_ids,
        verify_source_contract,
    )
    from run_spooky_transformer_oof import (  # type: ignore[no-redef]
        CLASS_COLUMNS,
        build_duplicate_safe_folds,
        multiclass_log_loss,
        normalize_probability,
        now_iso,
        sha256_file,
    )
    from spooky_leakage_free_meta import cross_fit_logistic_meta  # type: ignore[no-redef]
    from spooky_style_features import extract_style_features  # type: ignore[no-redef]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _array_check(values: np.ndarray, *, rows: int, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(values)
    normalized = normalize_probability(raw)
    expected = (rows, len(CLASS_COLUMNS))
    report = {
        "name": name,
        "shape": list(raw.shape),
        "expected_shape": list(expected),
        "finite": bool(np.isfinite(raw).all()),
        "normalized": bool(np.allclose(raw.sum(axis=1), 1.0, atol=1e-7)) if raw.ndim == 2 else False,
        "passed": bool(
            raw.shape == expected
            and np.isfinite(raw).all()
            and np.allclose(raw, normalized, atol=1e-12, rtol=0.0)
        ),
    }
    return normalized, report


def _load_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    numeric_names = (
        "truth",
        "fold",
        "transformer_oof",
        "sparse_oof",
        "transformer_test_by_fold",
        "sparse_test_by_fold",
        "candidate_oof",
        "candidate_test",
        "component_write_counts",
        "candidate_write_counts",
    )
    arrays: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"allow_pickle": False, "path": str(path), "id_arrays": {}}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set((*numeric_names, "train_id", "test_id")) - set(archive.files))
        if missing:
            raise RuntimeError(f"Spooky v2 bundle is missing arrays: {missing}")
        for name in numeric_names:
            arrays[name] = np.asarray(archive[name])
        for name in ("train_id", "test_id"):
            try:
                values = np.asarray(archive[name])
            except ValueError as exc:
                report["id_arrays"][name] = {
                    "status": "object_dtype_rejected",
                    "error": str(exc),
                    "passed": False,
                }
                continue
            arrays[name] = values
            report["id_arrays"][name] = {
                "status": "unicode" if values.dtype.kind == "U" else "unsafe_dtype",
                "dtype": str(values.dtype),
                "rows": len(values),
                "passed": values.ndim == 1 and values.dtype.kind == "U",
            }
    report["id_arrays_safe"] = all(
        report["id_arrays"].get(name, {}).get("passed", False)
        for name in ("train_id", "test_id")
    )
    return arrays, report


def reconstruct_candidate(
    component_oof: dict[str, np.ndarray],
    component_test: dict[str, np.ndarray],
    train_style: np.ndarray,
    test_style: np.ndarray,
    truth: np.ndarray,
    folds: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    oof_features, test_features, feature_contract = build_meta_inputs(
        component_oof,
        component_test,
        train_style,
        test_style,
    )
    candidate_oof, candidate_test, counts, meta_contract = cross_fit_logistic_meta(
        oof_features,
        test_features,
        truth,
        folds,
        c_value=c_value,
        max_iter=max_iter,
        random_state=random_state,
    )
    return candidate_oof, candidate_test, counts, feature_contract, meta_contract


def _verify_fold_artifacts(
    run_dir: Path,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    test_rows: int,
    plan_sha256: str,
    runner_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    train_rows = sum(len(valid) for _, valid in splits)
    components = {
        name: np.full((train_rows, len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
        for name in ("transformer", "sparse")
    }
    test_by_fold = {
        name: np.zeros((len(splits), test_rows, len(CLASS_COLUMNS)), dtype=np.float64)
        for name in components
    }
    counts = np.zeros(train_rows, dtype=np.uint8)
    records: list[dict[str, Any]] = []
    for fold, (fit_indices, valid_indices) in enumerate(splits):
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        metadata_path = run_dir / f"fold_{fold}_result.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        with np.load(prediction_path, allow_pickle=False) as archive:
            stored_valid = np.asarray(archive["valid_indices"], dtype=np.int64)
            stored_test = np.asarray(archive["test_indices"], dtype=np.int64)
            record_checks = {
                "prediction_hash": metadata.get("prediction_sha256") == sha256_file(prediction_path),
                "plan_hash": metadata.get("plan_sha256") == plan_sha256,
                "runner_hash": metadata.get("runner_sha256") == runner_sha256,
                "valid_indices": np.array_equal(stored_valid, valid_indices),
                "test_indices": np.array_equal(stored_test, np.arange(test_rows, dtype=np.int64)),
                "fit_index_hash": metadata.get("fit_index_sha256") == index_sha256(fit_indices),
                "valid_index_hash": metadata.get("valid_index_sha256") == index_sha256(valid_indices),
                "fit_valid_disjoint": bool(metadata.get("fit_valid_disjoint"))
                and not bool(np.intersect1d(fit_indices, valid_indices).size),
                "fixed_checkpoint": metadata.get("checkpoint_selection_used_outer_fold") is False,
                "private_labels_unused": metadata.get("private_labels_used") is False,
                "official_grader_not_executed": metadata.get("official_grader_executed") is False,
                "kaggle_submission_not_executed": metadata.get("kaggle_submission_executed") is False,
                "no_process_signals": metadata.get("process_signals_sent") == 0,
            }
            for name in components:
                valid_probability, valid_report = _array_check(
                    archive[f"{name}_valid"], rows=len(valid_indices), name=f"fold_{fold}_{name}_valid"
                )
                test_probability, test_report = _array_check(
                    archive[f"{name}_test"], rows=test_rows, name=f"fold_{fold}_{name}_test"
                )
                components[name][valid_indices] = valid_probability
                test_by_fold[name][fold] = test_probability
                record_checks[f"{name}_valid_probability"] = valid_report["passed"]
                record_checks[f"{name}_test_probability"] = test_report["passed"]
            counts[valid_indices] += 1
        records.append(
            {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "valid_rows": len(valid_indices),
                "checks": record_checks,
                "passed": all(record_checks.values()),
            }
        )
    return components, test_by_fold, counts, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    public_dir = args.public_dir.resolve()
    plan_path = args.plan.resolve()
    output = args.output.resolve() if args.output else run_dir / "independent_verification.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    bundle_path = run_dir / "spooky_deberta_oof_v2.npz"
    submission_path = run_dir / "candidate_submission_withheld.csv"
    required_files = [plan_path, summary_path, manifest_path, bundle_path, submission_path]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    plan_sha256 = sha256_file(plan_path)
    source_records = verify_source_contract(plan)
    runner_sha256 = source_records["runner"]["actual_sha256"]
    input_paths = {
        "public_train_sha256": public_dir / "train.csv",
        "public_test_sha256": public_dir / "test.csv",
        "public_sample_submission_sha256": public_dir / "sample_submission.csv",
    }
    input_checks = {
        name: path.is_file() and sha256_file(path) == plan["inputs"].get(name)
        for name, path in input_paths.items()
    }
    train = pd.read_csv(input_paths["public_train_sha256"])
    test = pd.read_csv(input_paths["public_test_sha256"])
    sample = pd.read_csv(input_paths["public_sample_submission_sha256"])
    labels = train["author"].map({value: index for index, value in enumerate(CLASS_COLUMNS)})
    if labels.isna().any():
        raise ValueError("Public train labels do not match the frozen class order")
    truth = labels.to_numpy(dtype=np.int64)
    train_text = train["text"].fillna("").astype(str).tolist()
    test_text = test["text"].fillna("").astype(str).tolist()
    splits, expected_folds, duplicate_report = build_duplicate_safe_folds(
        train_text,
        truth,
        folds=int(plan["training"]["folds"]),
        seed=int(plan["training"]["seed"]),
    )
    train_style, style_contract = extract_style_features(train_text)
    test_style, test_style_contract = extract_style_features(test_text)
    arrays, bundle_load = _load_bundle(bundle_path)
    fold_components, fold_test, fold_counts, fold_records = _verify_fold_artifacts(
        run_dir,
        splits,
        test_rows=len(test),
        plan_sha256=plan_sha256,
        runner_sha256=runner_sha256,
    )

    bundle_components: dict[str, np.ndarray] = {}
    bundle_test: dict[str, np.ndarray] = {}
    probability_checks: dict[str, bool] = {}
    for name in ("transformer", "sparse"):
        oof, oof_report = _array_check(arrays[f"{name}_oof"], rows=len(train), name=f"{name}_oof")
        test_matrix = np.asarray(arrays[f"{name}_test_by_fold"], dtype=np.float64)
        test_passed = bool(
            test_matrix.shape == (len(splits), len(test), len(CLASS_COLUMNS))
            and np.isfinite(test_matrix).all()
            and np.allclose(test_matrix.sum(axis=2), 1.0, atol=1e-7)
        )
        bundle_components[name] = oof
        bundle_test[name] = test_matrix
        probability_checks[f"{name}_oof"] = oof_report["passed"]
        probability_checks[f"{name}_test"] = test_passed
        probability_checks[f"{name}_fold_reconstruction"] = bool(
            np.allclose(oof, fold_components[name], atol=1e-12, rtol=0.0)
            and np.allclose(test_matrix, fold_test[name], atol=1e-12, rtol=0.0)
        )

    candidate_oof, candidate_test, candidate_counts, feature_contract, meta_contract = (
        reconstruct_candidate(
            bundle_components,
            bundle_test,
            train_style,
            test_style,
            truth,
            expected_folds,
            c_value=float(plan["meta"]["c_value"]),
            max_iter=int(plan["meta"]["max_iter"]),
            random_state=int(plan["training"]["seed"]) + 9000,
        )
    )
    stored_candidate_oof, candidate_oof_report = _array_check(
        arrays["candidate_oof"], rows=len(train), name="candidate_oof"
    )
    stored_candidate_test, candidate_test_report = _array_check(
        arrays["candidate_test"], rows=len(test), name="candidate_test"
    )
    probability_checks.update(
        {
            "candidate_oof": candidate_oof_report["passed"],
            "candidate_test": candidate_test_report["passed"],
            "candidate_oof_reconstruction": bool(
                np.allclose(candidate_oof, stored_candidate_oof, atol=1e-12, rtol=0.0)
            ),
            "candidate_test_reconstruction": bool(
                np.allclose(candidate_test, stored_candidate_test, atol=1e-12, rtol=0.0)
            ),
        }
    )
    recomputed_score = multiclass_log_loss(truth, candidate_oof)
    submission = pd.read_csv(submission_path)
    identity_checks = {
        "train_id": bool(
            "train_id" in arrays
            and np.array_equal(arrays["train_id"], safe_unicode_ids(train["id"].astype(str).tolist()))
        ),
        "test_id": bool(
            "test_id" in arrays
            and np.array_equal(arrays["test_id"], safe_unicode_ids(test["id"].astype(str).tolist()))
        ),
        "submission_id": submission["id"].astype(str).tolist() == test["id"].astype(str).tolist(),
        "sample_id": sample["id"].astype(str).tolist() == test["id"].astype(str).tolist(),
    }
    submission_probability = submission.loc[:, list(CLASS_COLUMNS)].to_numpy(dtype=np.float64)
    submission_checks = {
        "columns": list(submission.columns) == ["id", *CLASS_COLUMNS],
        "finite": bool(np.isfinite(submission_probability).all()),
        "normalized": bool(np.allclose(submission_probability.sum(axis=1), 1.0, atol=1e-7)),
        "matches_candidate": bool(
            np.allclose(submission_probability, candidate_test, atol=1e-12, rtol=0.0)
        ),
        "withheld": submission_path.name == "candidate_submission_withheld.csv",
    }
    boundary_checks = {
        "public_data_only": plan.get("public_data_only") is True,
        "private_labels_unused": all(
            value.get("private_labels_used") is False for value in (plan, manifest, summary)
        ),
        "official_grader_not_executed": all(
            value.get("official_grader_executed") is False for value in (manifest, summary)
        ),
        "kaggle_submission_not_executed": all(
            value.get("kaggle_submission_executed") is False for value in (manifest, summary)
        ),
        "human_gate_preserved": manifest.get("human_gate_preserved") is True,
        "no_process_signals": manifest.get("process_signals_sent") == 0
        and summary.get("process_signals_sent") == 0,
    }
    contract_checks = {
        "plan_schema": plan.get("schema") == "evomind.spooky.deberta_oof_frozen_plan.v2",
        "plan_frozen": plan.get("status") == "frozen_before_training",
        "plan_hash": summary.get("plan_sha256") == plan_sha256
        and manifest.get("plan_sha256") == plan_sha256,
        "source_contract": all(record["passed"] for record in source_records.values()),
        "input_hashes": all(input_checks.values()),
        "duplicate_report": duplicate_report == plan["inputs"]["duplicate_group_report"],
        "fold_assignment": np.array_equal(np.asarray(arrays["fold"], dtype=np.int16), expected_folds),
        "truth": np.array_equal(np.asarray(arrays["truth"], dtype=np.int64), truth),
        "fold_exact_once": bool(np.all(fold_counts == 1)),
        "bundle_component_counts": np.array_equal(
            np.asarray(arrays["component_write_counts"], dtype=np.uint8), fold_counts
        ),
        "candidate_counts": np.array_equal(
            np.asarray(arrays["candidate_write_counts"], dtype=np.uint8), candidate_counts
        )
        and bool(np.all(candidate_counts == 1)),
        "fold_artifacts": all(record["passed"] for record in fold_records),
        "bundle_ids_safe": bundle_load["id_arrays_safe"],
        "identities": all(identity_checks.values()),
        "probabilities": all(probability_checks.values()),
        "submission": all(submission_checks.values()),
        "summary_score": math.isclose(
            recomputed_score,
            float(summary.get("candidate_oof_log_loss", math.inf)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "summary_bundle_hash": summary.get("prediction_bundle", {}).get("sha256")
        == sha256_file(bundle_path),
        "summary_submission_hash": summary.get("submission_withheld", {}).get("sha256")
        == sha256_file(submission_path),
        "style_contract": style_contract["feature_names"] == test_style_contract["feature_names"],
        "meta_contract": meta_contract["exact_once_oof"] is True,
        "boundaries": all(boundary_checks.values()),
    }
    report = {
        "schema": "evomind.spooky.deberta_oof_independent_verification.v2",
        "created_at": now_iso(),
        "run_id": summary.get("run_id"),
        "status": "passed" if all(contract_checks.values()) else "failed",
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "source_contract": source_records,
        "prediction_bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
        "submission_withheld": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "input_checks": input_checks,
        "bundle_load": bundle_load,
        "identity_checks": identity_checks,
        "probability_checks": probability_checks,
        "submission_checks": submission_checks,
        "boundary_checks": boundary_checks,
        "fold_records": fold_records,
        "duplicate_group_report": duplicate_report,
        "meta_feature_contract": feature_contract,
        "meta_contract": meta_contract,
        "recomputed_candidate_oof_log_loss": recomputed_score,
        "contract_checks": contract_checks,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
