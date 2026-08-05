"""Evaluate a nested, leakage-free XGBoost Spooky cross-run candidate.

The candidate combines probability channels from two public-data OOF runs with
deterministic public-text style features.  For every held outer fold, model and
temperature selection happens on a different inner fold.  The selected fixed
configuration is then retrained on all remaining outer-fold rows before it
scores the held fold and the corresponding foldwise test features.

No private labels, official grader, Kaggle submission, GPU, or process-control
path is present in this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_spooky_cross_run_meta import (
        CLASS_COLUMNS,
        build_channels,
        load_numeric_bundle,
        read_json,
        sha256_file,
        validate_probability,
    )
    from scripts.spooky_style_features import extract_style_features
except ImportError:  # Direct ``python scripts/...`` execution.
    from evaluate_spooky_cross_run_meta import (  # type: ignore[no-redef]
        CLASS_COLUMNS,
        build_channels,
        load_numeric_bundle,
        read_json,
        sha256_file,
        validate_probability,
    )
    from spooky_style_features import extract_style_features  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "spooky_cross_run_xgb_s42_frozen_plan_20260727.json"
)
EXPECTED_SCHEMA = "evomind.spooky.cross_run_xgb_frozen_plan.v1"
ISOLATED_RUNTIME_CACHE_TAG = "cpython-312"
TERMINAL_CURRENT_STATUSES = {
    "single_seed_gate_failed",
    "single_seed_gate_passed_confirmation_pending",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def safe_unicode(values: Sequence[str]) -> np.ndarray:
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def index_sha256(indices: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def tree_sha256(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(value for value in Path(root).rglob("*") if value.is_file()):
        if (
            "__pycache__" in path.parts
            and path.suffix.lower() in {".pyc", ".pyo"}
            and f".{ISOLATED_RUNTIME_CACHE_TAG}." not in path.name
        ):
            continue
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += path.stat().st_size
    return digest.hexdigest(), file_count, total_bytes


def multiclass_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(truth, dtype=np.int64).reshape(-1)
    matrix = validate_probability("metric_probability", probability, len(target))
    return float(-np.log(np.clip(matrix[np.arange(len(target)), target], 1e-15, 1.0)).mean())


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be finite and positive")
    matrix = validate_probability("temperature_input", probability, len(probability))
    logits = np.log(np.clip(matrix, 1e-15, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def assemble_xgb_features(
    probability_channels: Mapping[str, np.ndarray],
    style_features: np.ndarray,
    *,
    channel_order: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if tuple(probability_channels) != tuple(channel_order):
        raise ValueError("Probability channel order diverged from the frozen plan")
    style = np.asarray(style_features, dtype=np.float64)
    if style.ndim != 2 or not np.isfinite(style).all():
        raise ValueError("Style features must be one finite matrix")
    rows = style.shape[0]
    blocks: list[np.ndarray] = []
    contract: list[dict[str, Any]] = []
    offset = 0
    for name in channel_order:
        probability = validate_probability(name, probability_channels[name], rows)
        ordered = np.sort(probability, axis=1)
        values = np.concatenate(
            (
                probability,
                np.log(np.clip(probability, 1e-7, 1.0)),
                (-np.sum(probability * np.log(np.clip(probability, 1e-7, 1.0)), axis=1))[
                    :, None
                ],
                (ordered[:, -1] - ordered[:, -2])[:, None],
            ),
            axis=1,
        )
        blocks.append(values)
        contract.append(
            {
                "name": name,
                "kind": "probability_log_entropy_margin",
                "start": offset,
                "stop": offset + values.shape[1],
            }
        )
        offset += values.shape[1]
    blocks.append(style)
    contract.append(
        {"name": "style", "kind": "deterministic_dense", "start": offset, "stop": offset + style.shape[1]}
    )
    features = np.concatenate(blocks, axis=1)
    return features, {
        "schema": "evomind.spooky.cross_run_xgb_feature_contract.v1",
        "rows": rows,
        "columns": features.shape[1],
        "blocks": contract,
        "channel_order": list(channel_order),
        "private_labels_used": False,
    }


def assemble_foldwise_xgb_features(
    test_channels_by_fold: Mapping[str, np.ndarray],
    test_style: np.ndarray,
    *,
    channel_order: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = np.asarray(test_channels_by_fold[channel_order[0]]).shape[0]
    matrices: list[np.ndarray] = []
    contracts: list[dict[str, Any]] = []
    for fold in range(folds):
        matrix, contract = assemble_xgb_features(
            {name: np.asarray(test_channels_by_fold[name])[fold] for name in channel_order},
            test_style,
            channel_order=channel_order,
        )
        matrices.append(matrix)
        contracts.append(contract)
    if any(record["blocks"] != contracts[0]["blocks"] for record in contracts[1:]):
        raise RuntimeError("Foldwise test feature contracts diverged")
    return np.stack(matrices), contracts


def _make_xgb_model(xgb_classifier: Any, config: Mapping[str, Any], *, seed: int) -> Any:
    params = dict(config["params"])
    return xgb_classifier(
        objective="multi:softprob",
        num_class=len(CLASS_COLUMNS),
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        n_jobs=int(params.pop("n_jobs")),
        random_state=seed,
        verbosity=0,
        **params,
    )


def cross_fit_nested_xgb(
    features: np.ndarray,
    test_features_by_fold: np.ndarray,
    truth: np.ndarray,
    fold_assignment: np.ndarray,
    *,
    configurations: Sequence[Mapping[str, Any]],
    temperatures: Sequence[float],
    inner_validation_fold_offset: int,
    random_state: int,
    xgb_classifier: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64).reshape(-1)
    folds = np.asarray(fold_assignment, dtype=np.int16).reshape(-1)
    foldwise_test = np.asarray(test_features_by_fold, dtype=np.float64)
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("Outer folds must be contiguous and include at least three folds")
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or len(labels) != len(folds):
        raise ValueError("Features, truth, and folds are not aligned")
    if not np.isfinite(matrix).all() or not np.isfinite(foldwise_test).all():
        raise ValueError("Meta features must be finite")
    if foldwise_test.shape != (
        len(unique_folds),
        foldwise_test.shape[1],
        matrix.shape[1],
    ):
        raise ValueError("Foldwise test features are not aligned")
    if not configurations or not temperatures:
        raise ValueError("Nested selection requires frozen configurations and temperatures")

    candidate_oof = np.full((len(labels), len(CLASS_COLUMNS)), np.nan, dtype=np.float64)
    write_counts = np.zeros(len(labels), dtype=np.uint8)
    test_predictions = np.zeros(
        (len(unique_folds), foldwise_test.shape[1], len(CLASS_COLUMNS)), dtype=np.float64
    )
    records: list[dict[str, Any]] = []
    for outer_fold in unique_folds:
        outer_fit = np.flatnonzero(folds != outer_fold)
        outer_score = np.flatnonzero(folds == outer_fold)
        inner_valid_fold = (outer_fold + inner_validation_fold_offset) % len(unique_folds)
        if inner_valid_fold == outer_fold:
            raise ValueError("Inner validation fold overlaps the held outer fold")
        inner_train = np.flatnonzero((folds != outer_fold) & (folds != inner_valid_fold))
        inner_valid = np.flatnonzero(folds == inner_valid_fold)
        selection_records: list[dict[str, Any]] = []
        for config_index, config in enumerate(configurations):
            model = _make_xgb_model(
                xgb_classifier,
                config,
                seed=random_state + outer_fold * 100 + config_index,
            )
            model.fit(matrix[inner_train], labels[inner_train])
            if not np.array_equal(np.asarray(model.classes_), np.arange(len(CLASS_COLUMNS))):
                raise RuntimeError("XGBoost class order changed")
            inner_probability = validate_probability(
                "inner_probability", model.predict_proba(matrix[inner_valid]), len(inner_valid)
            )
            temperature_records = []
            for temperature in temperatures:
                calibrated = temperature_scale(inner_probability, float(temperature))
                temperature_records.append(
                    {
                        "temperature": float(temperature),
                        "log_loss": multiclass_log_loss(labels[inner_valid], calibrated),
                    }
                )
            selected_temperature = min(
                temperature_records, key=lambda value: (value["log_loss"], value["temperature"])
            )
            selection_records.append(
                {
                    "name": str(config["name"]),
                    "config_index": config_index,
                    "temperature": selected_temperature["temperature"],
                    "inner_log_loss": selected_temperature["log_loss"],
                    "temperature_records": temperature_records,
                }
            )
        selected = min(
            selection_records,
            key=lambda value: (value["inner_log_loss"], value["name"], value["temperature"]),
        )
        selected_config = configurations[int(selected["config_index"])]
        final_model = _make_xgb_model(
            xgb_classifier,
            selected_config,
            seed=random_state + outer_fold * 1000 + 99,
        )
        final_model.fit(matrix[outer_fit], labels[outer_fit])
        held_probability = temperature_scale(
            final_model.predict_proba(matrix[outer_score]), selected["temperature"]
        )
        test_probability = temperature_scale(
            final_model.predict_proba(foldwise_test[outer_fold]), selected["temperature"]
        )
        candidate_oof[outer_score] = held_probability
        write_counts[outer_score] += 1
        test_predictions[outer_fold] = test_probability
        records.append(
            {
                "outer_score_fold": outer_fold,
                "outer_fit_folds": [value for value in unique_folds if value != outer_fold],
                "inner_validation_fold": inner_valid_fold,
                "inner_train_folds": [
                    value for value in unique_folds if value not in (outer_fold, inner_valid_fold)
                ],
                "selected_configuration": selected["name"],
                "selected_temperature": selected["temperature"],
                "inner_selected_log_loss": selected["inner_log_loss"],
                "held_outer_log_loss": multiclass_log_loss(labels[outer_score], held_probability),
                "outer_fit_index_sha256": index_sha256(outer_fit),
                "outer_score_index_sha256": index_sha256(outer_score),
                "inner_train_index_sha256": index_sha256(inner_train),
                "inner_valid_index_sha256": index_sha256(inner_valid),
                "fit_score_disjoint": not bool(np.intersect1d(outer_fit, outer_score).size),
                "selection_records": selection_records,
            }
        )
    if not np.all(write_counts == 1) or not np.isfinite(candidate_oof).all():
        raise RuntimeError("Nested XGBoost OOF predictions were not written exactly once")
    candidate_test = validate_probability(
        "candidate_test", test_predictions.mean(axis=0), test_predictions.shape[1]
    )
    return candidate_oof, candidate_test, write_counts, {
        "schema": "evomind.spooky.nested_cross_fit_xgboost_meta.v1",
        "outer_folds": unique_folds,
        "selection": "select_on_inner_fold_retrain_outer_fit_score_held_outer_fold",
        "inner_validation_fold_offset": inner_validation_fold_offset,
        "configurations": [dict(value) for value in configurations],
        "temperatures": [float(value) for value in temperatures],
        "random_state": random_state,
        "feature_count": matrix.shape[1],
        "records": records,
        "exact_once_oof": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
    }


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected cross-run XGBoost frozen-plan schema")
    if plan.get("status") != "frozen_waiting_current_run":
        raise ValueError("Cross-run XGBoost plan is not frozen")
    if plan.get("competition_id") != "spooky-author-identification":
        raise ValueError("Cross-run XGBoost plan targets another competition")
    if plan.get("public_data_only") is not True:
        raise ValueError("Cross-run XGBoost plan is not public-data-only")
    boundaries = plan.get("boundaries", {})
    for name in (
        "private_labels_used",
        "official_grader_executed",
        "kaggle_submission_executed",
    ):
        if boundaries.get(name) is not False:
            raise ValueError(f"Cross-run XGBoost plan boundary is invalid: {name}")
    if boundaries.get("process_signals_sent") != 0:
        raise ValueError("Cross-run XGBoost plan permits process signals")
    for section, fields in (
        ("source", ("evaluator", "cross_run_helper", "style_features")),
        ("runtime", ("manifest",)),
        ("prior", ("summary", "bundle")),
        ("current", ("plan",)),
        ("inputs", ("train", "test", "sample")),
        ("research_evidence", ("readme", "experiments")),
    ):
        for field in fields:
            record = plan[section][field]
            path = Path(record["path"]).resolve()
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen artifact drift: {section}.{field}")
    plan["_plan_path"] = str(plan_path)
    plan["_plan_sha256"] = sha256_file(plan_path)
    return plan


def load_xgboost_runtime(plan: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    runtime = plan["runtime"]
    manifest = read_json(Path(runtime["manifest"]["path"]))
    package_root = Path(manifest["package_root"]).resolve()
    digest, file_count, total_bytes = tree_sha256(package_root)
    if digest != manifest["tree_sha256"] or digest != runtime["tree_sha256"]:
        raise RuntimeError("Isolated XGBoost runtime tree hash drifted")
    if file_count != int(manifest["file_count"]) or total_bytes != int(
        manifest["total_bytes"]
    ):
        raise RuntimeError("Isolated XGBoost runtime size contract drifted")
    sys.path.insert(0, str(package_root))
    module = importlib.import_module("xgboost")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError("XGBoost imported outside the isolated package root") from exc
    if module.__version__ != runtime["version"]:
        raise RuntimeError("XGBoost runtime version drifted")
    return module.XGBClassifier, {
        "version": module.__version__,
        "module": str(module_path),
        "package_root": str(package_root),
        "tree_sha256": digest,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "device": "cpu",
    }


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
    if not current_bundle_path.is_file() or prediction.get("sha256") != sha256_file(
        current_bundle_path
    ):
        raise RuntimeError("Current prediction bundle is missing or hash-invalid")
    if any(
        current_summary.get(name) not in (False, 0)
        for name in (
            "private_labels_used",
            "official_grader_executed",
            "kaggle_submission_executed",
            "process_signals_sent",
        )
    ):
        raise RuntimeError("Current summary boundary contract failed")

    prior = load_numeric_bundle(
        Path(plan["prior"]["bundle"]["path"]),
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
    channel_order = tuple(plan["features"]["channel_order"])
    if tuple(oof_channels) != channel_order:
        raise RuntimeError("Cross-run channel order diverged from the frozen plan")

    train_path = Path(plan["inputs"]["train"]["path"])
    test_path = Path(plan["inputs"]["test"]["path"])
    sample_path = Path(plan["inputs"]["sample"]["path"])
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    public_truth = train["author"].map(
        {name: index for index, name in enumerate(CLASS_COLUMNS)}
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(public_truth, truth):
        raise RuntimeError("Bundle truth does not match the public train CSV")
    if not np.array_equal(test["id"].astype(str), sample["id"].astype(str)):
        raise RuntimeError("Public test and sample IDs are not aligned")
    train_style, train_style_contract = extract_style_features(
        train["text"].fillna("").astype(str).tolist()
    )
    test_style, test_style_contract = extract_style_features(
        test["text"].fillna("").astype(str).tolist()
    )
    if train_style_contract["feature_names"] != test_style_contract["feature_names"]:
        raise RuntimeError("Train and test style feature contracts diverged")
    features, feature_contract = assemble_xgb_features(
        oof_channels, train_style, channel_order=channel_order
    )
    test_features, test_feature_contracts = assemble_foldwise_xgb_features(
        test_channels, test_style, channel_order=channel_order
    )

    xgb_classifier, runtime_report = load_xgboost_runtime(plan)
    meta = plan["meta"]
    candidate_oof, candidate_test, counts, meta_contract = cross_fit_nested_xgb(
        features,
        test_features,
        truth,
        folds,
        configurations=meta["configurations"],
        temperatures=meta["temperatures"],
        inner_validation_fold_offset=int(meta["inner_validation_fold_offset"]),
        random_state=int(meta["random_state"]),
        xgb_classifier=xgb_classifier,
    )
    score = multiclass_log_loss(truth, candidate_oof)
    threshold = float(plan["promotion_gate"]["single_seed_log_loss_threshold"])
    single_seed_passed = score <= threshold
    output_dir = Path(plan["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "spooky_cross_run_xgb_oof_and_test.npz"
    np.savez_compressed(
        bundle_path,
        truth=truth,
        fold=folds,
        candidate_oof=candidate_oof,
        candidate_test=candidate_test,
        candidate_write_counts=counts,
        train_id=safe_unicode(train["id"].astype(str).tolist()),
        test_id=safe_unicode(test["id"].astype(str).tolist()),
    )
    submission = sample.copy()
    submission.loc[:, list(CLASS_COLUMNS)] = candidate_test
    submission_path = output_dir / "candidate_submission_withheld.csv"
    submission.to_csv(submission_path, index=False)
    component_scores = {
        name: multiclass_log_loss(truth, values) for name, values in oof_channels.items()
    }
    result: dict[str, Any] = {
        "schema": "evomind.spooky.cross_run_xgb_result.v1",
        "created_at": now_iso(),
        "run_id": plan["output"]["run_id"],
        "status": (
            "single_seed_gate_passed_confirmation_pending"
            if single_seed_passed
            else "single_seed_gate_failed"
        ),
        "plan": {"path": plan["_plan_path"], "sha256": plan["_plan_sha256"]},
        "prior_run_id": plan["prior"]["run_id"],
        "current_run_id": plan["current"]["run_id"],
        "current_summary_status": current_summary["status"],
        "runtime": runtime_report,
        "feature_contract": feature_contract,
        "test_feature_contracts": test_feature_contracts,
        "style_feature_contract": train_style_contract,
        "meta_contract": meta_contract,
        "component_oof_log_loss": component_scores,
        "candidate_oof_log_loss": score,
        "single_seed_threshold": threshold,
        "single_seed_passed": single_seed_passed,
        "confirmation_required": True,
        "promotion_allowed": False,
        "prediction_bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
        "submission_withheld": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
        },
        "checks": {
            "truth_matches_public_csv": True,
            "fold_alignment": True,
            "exact_once_oof": bool(np.all(counts == 1)),
            "finite_normalized_oof": bool(
                np.isfinite(candidate_oof).all()
                and np.allclose(candidate_oof.sum(axis=1), 1.0, atol=1e-7)
            ),
            "finite_normalized_test": bool(
                np.isfinite(candidate_test).all()
                and np.allclose(candidate_test.sum(axis=1), 1.0, atol=1e-7)
            ),
            "nested_selection": meta_contract["exact_once_oof"] is True,
            "isolated_xgboost_runtime": runtime_report["tree_sha256"]
            == plan["runtime"]["tree_sha256"],
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": (
            "Leakage-free public OOF candidate only. A passing single-seed gate requires "
            "confirmation and remains neither an official score nor a medal claim."
        ),
    }
    result_path = output_dir / "cross_run_xgb_result.json"
    write_json_atomic(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args.plan.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
