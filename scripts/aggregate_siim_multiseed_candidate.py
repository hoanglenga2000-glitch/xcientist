#!/usr/bin/env python3
"""Freeze and independently verify a three-seed SIIM public-only ensemble."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402

COMPETITION_ID = "siim-isic-melanoma-classification"
EXPECTED_FORMAL_SEEDS = (43, 44, 45)
R2_FORMAL_SEEDS = (46, 47, 48)
ALLOWED_FORMAL_SEED_SETS = {EXPECTED_FORMAL_SEEDS, R2_FORMAL_SEEDS}
EXPECTED_ABLATION_SEEDS = (40, 41, 42)
DEFAULT_COLLECTED_ROOT = common.DEFAULT_COLLECTED_ROOT
COMPONENT_OOF_KEYS = {
    "pure_image": "pure_image_oof",
    "lesion_focus": "lesion_focus_oof",
    "image_metadata_fusion": "image_metadata_fusion_oof",
    "metadata_catboost": "metadata_catboost_oof",
}
EVOLUTION_BUNDLE_KEYS = {
    "seed_final_foldrank_oof": "seed_final_fold_percentile_rank_oof",
    "seed_final_foldrank_test": "seed_final_fold_percentile_rank_test",
    "pure_image_foldrank_oof": "pure_image_percentile_rank_oof",
    "pure_image_foldrank_test": "pure_image_percentile_rank_test",
    "image_metadata_fusion_foldrank_oof": "image_metadata_fusion_percentile_rank_oof",
    "image_metadata_fusion_foldrank_test": "image_metadata_fusion_percentile_rank_test",
}


class SiimAggregationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SiimAggregationError(message)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def staged_file_record(path: Path, *, staging_root: Path, final_root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(staging_root.resolve())
    except ValueError as exc:
        raise SiimAggregationError("staged artifact escaped the transaction directory") from exc
    record = common.file_record(resolved)
    record["path"] = str((final_root / relative).resolve())
    return record


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def _unique_indexed_file(
    root: Path,
    name: str,
    indexed: dict[str, dict[str, Any]],
    *,
    prefer_top_level: bool = False,
) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and not path.is_symlink()]
    if prefer_top_level:
        top = [path for path in matches if "attempts" not in path.parts]
        if len(top) == 1:
            matches = top
    require(len(matches) == 1, f"Expected one {name} under {root}, found {len(matches)}")
    path = matches[0].resolve()
    require(str(path) in indexed, f"{name} was not hash-collected")
    return path


def _seed_artifacts(run_root: Path, seed: int) -> dict[str, Any]:
    indexed = common._validate_collection(run_root)
    competition_root = run_root / COMPETITION_ID
    result_path = competition_root / "result.json"
    require(str(result_path.resolve()) in indexed, "SIIM top-level result was not hash-collected")
    result = _read_json(result_path)
    require(result.get("competition_id") == COMPETITION_ID, "SIIM competition changed")
    require(result.get("metric") == "roc_auc", "SIIM metric changed")
    require(result.get("direction") == "maximize", "SIIM metric direction changed")
    require(result.get("valid_submission") is True, f"SIIM seed {seed} submission invalid")
    require(result.get("official_grader_executed") is False, "SIIM grader ran before freeze")
    require(result.get("mle_private_grader_score") is None, "SIIM private score entered training")
    require(result.get("kaggle_public_score") is None, "SIIM Kaggle public score is present")
    require(result.get("kaggle_private_score") is None, "SIIM Kaggle private score is present")
    require(int((result.get("budget") or {}).get("seed", -1)) == seed, "SIIM model seed changed")
    bundle_path = _unique_indexed_file(
        competition_root, "siim_fold_ensemble.npz", indexed
    )
    oof_path = _unique_indexed_file(
        competition_root, "siim_oof_predictions.csv", indexed
    )
    submissions = [
        path.resolve()
        for path in competition_root.rglob("submission.csv")
        if path.is_file() and common.sha256_file(path) == result.get("submission_sha256")
    ]
    submissions = [path for path in submissions if str(path) in indexed]
    require(bool(submissions), "SIIM withheld submission was not hash-collected")
    return {
        "seed": seed,
        "run_id": run_root.name,
        "result": result,
        "result_path": result_path,
        "bundle_path": bundle_path,
        "oof_path": oof_path,
        "submission_path": sorted(submissions, key=lambda value: len(value.parts))[0],
    }


def _load_bundle(item: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(item["bundle_path"], allow_pickle=False) as bundle:
        required = {
            "train_id",
            "patient_id",
            "leakage_group",
            "target",
            "fold",
            "crossfit_probability",
            "test_id",
            "test_probability",
        }
        require(required <= set(bundle.files), "SIIM prediction bundle is incomplete")
        values = {
            "train_id": np.asarray(bundle["train_id"], dtype=np.str_),
            "patient_id": np.asarray(bundle["patient_id"], dtype=np.str_),
            "leakage_group": np.asarray(bundle["leakage_group"], dtype=np.str_),
            "target": np.asarray(bundle["target"], dtype=np.int8),
            "fold": np.asarray(bundle["fold"], dtype=np.int16),
            "oof": np.asarray(bundle["crossfit_probability"], dtype=np.float64),
            "test_id": np.asarray(bundle["test_id"], dtype=np.str_),
            "test": np.asarray(bundle["test_probability"], dtype=np.float64),
        }
        for component, bundle_key in COMPONENT_OOF_KEYS.items():
            if bundle_key in bundle.files:
                values[f"component_{component}"] = np.asarray(
                    bundle[bundle_key], dtype=np.float64
                )
        for name, bundle_key in EVOLUTION_BUNDLE_KEYS.items():
            if bundle_key in bundle.files:
                values[name] = np.asarray(bundle[bundle_key], dtype=np.float64)
    require(values["train_id"].shape == (28_984,), "SIIM train row count changed")
    require(values["test_id"].shape == (4_142,), "SIIM test row count changed")
    require(len(np.unique(values["train_id"])) == 28_984, "SIIM train IDs are duplicated")
    require(len(np.unique(values["test_id"])) == 4_142, "SIIM test IDs are duplicated")
    require(set(np.unique(values["target"]).tolist()) == {0, 1}, "SIIM labels are invalid")
    require(set(np.unique(values["fold"]).tolist()) == {0, 1, 2, 3, 4}, "SIIM fold coverage changed")
    require(np.isfinite(values["oof"]).all(), "SIIM OOF contains non-finite values")
    require(np.isfinite(values["test"]).all(), "SIIM test prediction contains non-finite values")
    require(np.logical_and(values["oof"] >= 0, values["oof"] <= 1).all(), "SIIM OOF escaped [0,1]")
    require(np.logical_and(values["test"] >= 0, values["test"] <= 1).all(), "SIIM test escaped [0,1]")
    require(float(np.std(values["oof"])) > 1e-8, "SIIM OOF is constant")
    for component in COMPONENT_OOF_KEYS:
        key = f"component_{component}"
        if key not in values:
            continue
        require(values[key].shape == (28_984,), f"SIIM {component} OOF row count changed")
        require(np.isfinite(values[key]).all(), f"SIIM {component} OOF contains non-finite values")
        require(
            np.logical_and(values[key] >= 0, values[key] <= 1).all(),
            f"SIIM {component} OOF escaped [0,1]",
        )
    for name in EVOLUTION_BUNDLE_KEYS:
        if name not in values:
            continue
        expected_rows = 4_142 if name.endswith("_test") else 28_984
        require(values[name].shape == (expected_rows,), f"SIIM {name} row count changed")
        require(np.isfinite(values[name]).all(), f"SIIM {name} contains non-finite values")
        require(
            np.logical_and(values[name] >= 0, values[name] <= 1).all(),
            f"SIIM {name} escaped [0,1]",
        )
    return values


def select_oof_threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, probability)
    finite = np.isfinite(thresholds)
    require(bool(finite.any()), "SIIM threshold search produced no finite candidate")
    indices = np.flatnonzero(finite)
    best = indices[int(np.argmax((true_positive_rate - false_positive_rate)[finite]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def threshold_metrics(
    truth: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = probability >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "precision": divide(tp, tp + fp),
        "negative_predictive_value": divide(tn, tn + fn),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def grouped_auc_bootstrap(
    truth: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(int(samples)):
        drawn = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in drawn])
        if len(np.unique(truth[indices])) != 2:
            continue
        scores.append(float(roc_auc_score(truth[indices], probability[indices])))
    require(len(scores) >= max(100, int(samples * 0.8)), "SIIM grouped bootstrap was unstable")
    lower, upper = np.quantile(np.asarray(scores), [0.025, 0.975])
    return {
        "method": "leakage_group_cluster_bootstrap",
        "requested_samples": int(samples),
        "valid_samples": len(scores),
        "seed": int(seed),
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def paired_grouped_auc_delta_bootstrap(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(int(samples)):
        drawn = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in drawn])
        if len(np.unique(truth[indices])) != 2:
            continue
        deltas.append(
            float(
                roc_auc_score(truth[indices], candidate[indices])
                - roc_auc_score(truth[indices], baseline[indices])
            )
        )
    require(
        len(deltas) >= max(100, int(samples * 0.8)),
        "SIIM paired grouped bootstrap was unstable",
    )
    array = np.asarray(deltas, dtype=np.float64)
    lower, upper = np.quantile(array, [0.025, 0.975])
    return {
        "method": "paired_leakage_group_cluster_multinomial_bootstrap",
        "requested_samples": int(samples),
        "valid_samples": len(deltas),
        "seed": int(seed),
        "confidence_level": 0.95,
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "probability_delta_gt_zero": float(np.mean(array > 0.0)),
    }


def _downsample_curve(x: np.ndarray, y: np.ndarray, *, max_points: int = 101) -> dict[str, list[float]]:
    require(x.ndim == 1 and y.ndim == 1 and len(x) == len(y), "invalid metric curve")
    require(len(x) >= 2, "metric curve is empty")
    if len(x) <= max_points:
        indices = np.arange(len(x))
    else:
        indices = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return {
        "x": np.asarray(x[indices], dtype=np.float64).tolist(),
        "y": np.asarray(y[indices], dtype=np.float64).tolist(),
    }


def metric_curves(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    false_positive_rate, true_positive_rate, _ = roc_curve(truth, probability)
    precision, recall, _ = precision_recall_curve(truth, probability)
    roc_points = _downsample_curve(false_positive_rate, true_positive_rate)
    # sklearn returns PR recall in descending order; reports use a left-to-right axis.
    pr_points = _downsample_curve(recall[::-1], precision[::-1])
    return {
        "roc_curve": {
            "false_positive_rate": roc_points["x"],
            "true_positive_rate": roc_points["y"],
        },
        "precision_recall_curve": {
            "recall": pr_points["x"],
            "precision": pr_points["y"],
        },
    }


def aggregate_predictions(
    items: list[dict[str, Any]],
    *,
    evolution_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference: dict[str, np.ndarray] | None = None
    oof_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    seed_auc: list[float] = []
    component_predictions: dict[str, list[np.ndarray]] = {}
    evolution_predictions: dict[str, list[np.ndarray]] = {}
    expected_component_keys: set[str] | None = None
    for item in items:
        values = _load_bundle(item)
        component_keys = {
            key for key in values if key.startswith("component_")
        }
        if expected_component_keys is None:
            expected_component_keys = component_keys
        else:
            require(
                component_keys == expected_component_keys,
                "SIIM component inventory changed between formal seeds",
            )
        if reference is None:
            reference = values
        else:
            for name in ("train_id", "patient_id", "leakage_group", "target", "test_id"):
                require(np.array_equal(values[name], reference[name]), f"SIIM seed alignment changed: {name}")
        auc = float(roc_auc_score(values["target"], values["oof"]))
        require(abs(auc - float(item["result"]["cv_score"])) <= 1e-9, "SIIM result/OOF AUC changed")
        seed_auc.append(auc)
        oof_predictions.append(values["oof"])
        test_predictions.append(values["test"])
        for key in sorted(component_keys):
            component_predictions.setdefault(key, []).append(values[key])
        for key in EVOLUTION_BUNDLE_KEYS:
            if key in values:
                evolution_predictions.setdefault(key, []).append(values[key])
    require(reference is not None, "SIIM formal seed inventory is empty")
    baseline_oof = np.mean(np.stack(oof_predictions, axis=0), axis=0)
    baseline_test = np.mean(np.stack(test_predictions, axis=0), axis=0)
    selected_oof = baseline_oof
    selected_test = baseline_test
    evolution: dict[str, Any] | None = None
    if evolution_protocol is not None:
        require(
            evolution_protocol.get("candidate_id")
            == "r2_foldwise_rank_channel_consensus_v1",
            "unsupported SIIM evolution candidate protocol",
        )
        require(
            set(evolution_predictions) == set(EVOLUTION_BUNDLE_KEYS),
            "SIIM evolution prediction streams are incomplete",
        )
        require(
            all(len(values) == len(items) for values in evolution_predictions.values()),
            "SIIM evolution prediction stream seed coverage changed",
        )
        configured = (
            (evolution_protocol.get("score_definition") or {}).get("weights") or {}
        )
        weights = {
            "seed_final_foldrank": float(configured.get("seed_final_foldrank", -1)),
            "pure_image": float(configured.get("pure_image", -1)),
            "image_metadata_fusion": float(
                configured.get("image_metadata_fusion", -1)
            ),
        }
        require(weights == {
            "seed_final_foldrank": 0.5,
            "pure_image": 0.3,
            "image_metadata_fusion": 0.2,
        }, "SIIM R2 fixed weights changed")
        candidate_oof = (
            weights["seed_final_foldrank"]
            * np.mean(np.stack(evolution_predictions["seed_final_foldrank_oof"]), axis=0)
            + weights["pure_image"]
            * np.mean(np.stack(evolution_predictions["pure_image_foldrank_oof"]), axis=0)
            + weights["image_metadata_fusion"]
            * np.mean(
                np.stack(evolution_predictions["image_metadata_fusion_foldrank_oof"]),
                axis=0,
            )
        )
        candidate_test = (
            weights["seed_final_foldrank"]
            * np.mean(np.stack(evolution_predictions["seed_final_foldrank_test"]), axis=0)
            + weights["pure_image"]
            * np.mean(np.stack(evolution_predictions["pure_image_foldrank_test"]), axis=0)
            + weights["image_metadata_fusion"]
            * np.mean(
                np.stack(evolution_predictions["image_metadata_fusion_foldrank_test"]),
                axis=0,
            )
        )
        require(np.isfinite(candidate_oof).all(), "SIIM R2 OOF contains non-finite values")
        require(np.isfinite(candidate_test).all(), "SIIM R2 test score contains non-finite values")
        selected_oof = np.clip(candidate_oof, 0.0, 1.0)
        selected_test = np.clip(candidate_test, 0.0, 1.0)
        evolution = {
            "candidate_id": evolution_protocol["candidate_id"],
            "weights": weights,
            "baseline_oof": baseline_oof,
            "baseline_test": baseline_test,
            "candidate_oof": selected_oof,
            "candidate_test": selected_test,
        }
    return {
        **reference,
        "oof": selected_oof,
        "test": selected_test,
        "baseline_oof": baseline_oof,
        "baseline_test": baseline_test,
        "evolution": evolution,
        "seed_auc": seed_auc,
        "component_oof": {
            key.removeprefix("component_"): np.mean(np.stack(predictions, axis=0), axis=0)
            for key, predictions in sorted(component_predictions.items())
        },
    }


def aggregate(
    *,
    plan_path: Path,
    collected_root: Path,
    sample_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    collected_root = Path(collected_root).resolve()
    sample_path = Path(sample_path).resolve()
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "SIIM frozen candidate directory already exists")
    plan = _read_json(plan_path)
    require(plan.get("schema") == "evomind.siim.hpc_campaign_plan.v1", "SIIM plan schema changed")
    formal_seeds = tuple(int(value) for value in plan.get("formal_seeds") or [])
    ablation_seeds = tuple(int(value) for value in plan.get("ablation_seeds") or [])
    require(formal_seeds in ALLOWED_FORMAL_SEED_SETS, "SIIM formal seeds changed")
    require(ablation_seeds == EXPECTED_ABLATION_SEEDS, "SIIM ablation seeds changed")
    require(set(formal_seeds).isdisjoint(ablation_seeds), "SIIM formal seed entered ablation")
    formal_runs = list(plan.get("formal_runs") or [])
    require([int(item["seed"]) for item in formal_runs] == list(formal_seeds), "SIIM run/seed order changed")
    items = [
        _seed_artifacts(collected_root / str(run["run_id"]), int(run["seed"]))
        for run in formal_runs
    ]
    evolution_protocol = plan.get("evolution_protocol")
    if evolution_protocol is not None:
        require(isinstance(evolution_protocol, dict), "SIIM evolution protocol is invalid")
        require(formal_seeds == (46, 47, 48), "SIIM R2 requires fresh formal seeds 46/47/48")
    values = aggregate_predictions(items, evolution_protocol=evolution_protocol)

    sample = pd.read_csv(sample_path)
    require(list(sample.columns) == ["image_name", "target"], "SIIM sample schema changed")
    require(len(sample) == 4_142, "SIIM sample row count changed")
    require(
        np.array_equal(sample["image_name"].astype(str).to_numpy(), values["test_id"]),
        "SIIM sample/test ID order changed",
    )

    bootstrap_samples = int((plan.get("evaluation") or {}).get("bootstrap_samples", 2_000))
    evolution_decision: dict[str, Any] | None = None
    if values["evolution"] is not None:
        baseline_auc = float(roc_auc_score(values["target"], values["baseline_oof"]))
        candidate_auc = float(roc_auc_score(values["target"], values["oof"]))
        baseline_pr = float(average_precision_score(values["target"], values["baseline_oof"]))
        candidate_pr = float(average_precision_score(values["target"], values["oof"]))
        fold_deltas: list[float] = []
        for fold in range(5):
            mask = values["fold"] == fold
            fold_deltas.append(
                float(
                    roc_auc_score(values["target"][mask], values["oof"][mask])
                    - roc_auc_score(
                        values["target"][mask], values["baseline_oof"][mask]
                    )
                )
            )
        paired = paired_grouped_auc_delta_bootstrap(
            values["target"],
            values["baseline_oof"],
            values["oof"],
            values["leakage_group"],
            samples=bootstrap_samples,
            seed=20260802,
        )
        gate = dict((evolution_protocol or {}).get("adoption_gate") or {})
        checks = {
            "overall_roc_auc_delta": candidate_auc - baseline_auc
            >= float(gate.get("minimum_overall_roc_auc_delta", 0.0005)),
            "pr_auc_non_regression": candidate_pr - baseline_pr
            >= float(gate.get("minimum_pr_auc_delta", 0.0)),
            "positive_fold_count": sum(delta > 0 for delta in fold_deltas)
            >= int(gate.get("minimum_positive_fold_count", 4)),
            "worst_fold_regression": min(fold_deltas)
            >= -float(gate.get("maximum_worst_fold_auc_regression", 0.001)),
            "paired_group_bootstrap": paired["probability_delta_gt_zero"]
            >= float(
                gate.get(
                    "minimum_paired_group_bootstrap_probability_delta_gt_zero",
                    0.9,
                )
            ),
        }
        adopted = all(checks.values())
        evolution_decision = {
            "candidate_id": evolution_protocol["candidate_id"],
            "adopted": adopted,
            "checks": checks,
            "baseline_roc_auc": baseline_auc,
            "candidate_roc_auc": candidate_auc,
            "roc_auc_delta": candidate_auc - baseline_auc,
            "baseline_pr_auc": baseline_pr,
            "candidate_pr_auc": candidate_pr,
            "pr_auc_delta": candidate_pr - baseline_pr,
            "fold_auc_delta": fold_deltas,
            "paired_group_bootstrap": paired,
            "private_labels_used": False,
        }
        if not adopted:
            values["oof"] = values["baseline_oof"]
            values["test"] = values["baseline_test"]
    threshold = select_oof_threshold(values["target"], values["oof"])
    fold_rows = []
    for fold in range(5):
        mask = values["fold"] == fold
        fold_rows.append({
            "fold": fold,
            "rows": int(mask.sum()),
            "positives": int(values["target"][mask].sum()),
            "roc_auc": float(roc_auc_score(values["target"][mask], values["oof"][mask])),
            "pr_auc": float(average_precision_score(values["target"][mask], values["oof"][mask])),
            "brier": float(brier_score_loss(values["target"][mask], values["oof"][mask])),
        })
    confidence_interval = grouped_auc_bootstrap(
        values["target"],
        values["oof"],
        values["leakage_group"],
        samples=bootstrap_samples,
        seed=20260729,
    )
    fraction_positive, mean_predicted = calibration_curve(
        values["target"], values["oof"], n_bins=10, strategy="quantile"
    )
    curves = metric_curves(values["target"], values["oof"])
    component_metrics = {
        name: {
            "roc_auc": float(roc_auc_score(values["target"], probability)),
            "pr_auc": float(average_precision_score(values["target"], probability)),
        }
        for name, probability in values["component_oof"].items()
    }
    component_metrics["final_ensemble"] = {
        "roc_auc": float(roc_auc_score(values["target"], values["oof"])),
        "pr_auc": float(average_precision_score(values["target"], values["oof"])),
    }
    metrics = {
        "schema": "evomind.siim.multiseed_metrics.v1",
        "run_id": str(plan["run_id"]),
        "metric_scope": "independent_offline_patient_content_grouped_oof",
        "roc_auc": float(roc_auc_score(values["target"], values["oof"])),
        "pr_auc": float(average_precision_score(values["target"], values["oof"])),
        "brier": float(brier_score_loss(values["target"], values["oof"])),
        "seed_roc_auc": values["seed_auc"],
        "fold_roc_auc_mean": float(np.mean([row["roc_auc"] for row in fold_rows])),
        "fold_roc_auc_std": float(np.std([row["roc_auc"] for row in fold_rows], ddof=1)),
        "fold_metrics": fold_rows,
        "model_components": component_metrics,
        "evolution_decision": evolution_decision,
        **curves,
        "patient_grouped_bootstrap_roc_auc_95ci": confidence_interval,
        "fixed_oof_threshold_metrics": threshold_metrics(
            values["target"], values["oof"], threshold
        ),
        "calibration_curve": {
            "mean_predicted_probability": mean_predicted.tolist(),
            "fraction_positive": fraction_positive.tolist(),
        },
        "historical_thresholds": dict(plan.get("historical_thresholds") or {}),
        "private_grader_execution_count": 0,
        "kaggle_submission_executed": False,
        "clinical_diagnosis_claimed": False,
    }

    final_output_dir = output_dir
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging.",
            dir=final_output_dir.parent,
        )
    ).resolve()
    try:
        candidate_path = staging_dir / "candidate_submission_withheld.csv"
        pd.DataFrame({"image_name": values["test_id"], "target": values["test"]}).to_csv(
            candidate_path, index=False
        )
        oof_path = staging_dir / "ensemble_oof_predictions.csv"
        pd.DataFrame({
            "image_name": values["train_id"],
            "patient_id": values["patient_id"],
            "leakage_group": values["leakage_group"],
            "target": values["target"],
            "fold": values["fold"],
            "baseline_probability": values["baseline_oof"],
            "candidate_probability": (
                values["evolution"]["candidate_oof"]
                if values["evolution"] is not None
                else values["baseline_oof"]
            ),
            "probability": values["oof"],
        }).to_csv(oof_path, index=False)
        fold_path = staging_dir / "fold_metrics.csv"
        pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
        metrics_path = staging_dir / "metrics.json"
        common.write_json_atomic(metrics_path, metrics)
        frozen_plan_path = staging_dir / "frozen_plan.json"
        shutil.copyfile(plan_path, frozen_plan_path)

        seed_records = [
            {
                "model_seed": item["seed"],
                "run_id": item["run_id"],
                "oof_roc_auc": auc,
                "result": common.file_record(item["result_path"]),
                "prediction_bundle": common.file_record(item["bundle_path"]),
                "oof_predictions": common.file_record(item["oof_path"]),
                "submission": common.file_record(item["submission_path"]),
            }
            for item, auc in zip(items, values["seed_auc"], strict=True)
        ]
        freeze = {
            "schema": "evomind.siim.candidate_freeze.v1",
            "created_at": now_iso(),
            "run_id": str(plan["run_id"]),
            "status": "frozen_before_private_grader",
            "formal_seeds": list(formal_seeds),
            "ablation_seeds": list(ablation_seeds),
            "seed_sets_disjoint": True,
            "seed_records": seed_records,
            "candidate_submission": staged_file_record(
                candidate_path, staging_root=staging_dir, final_root=final_output_dir
            ),
            "ensemble_oof_predictions": staged_file_record(
                oof_path, staging_root=staging_dir, final_root=final_output_dir
            ),
            "metrics": staged_file_record(
                metrics_path, staging_root=staging_dir, final_root=final_output_dir
            ),
            "private_grader_execution_count_before_freeze": 0,
            "official_submission_executed": False,
            "candidate_hash_bound": True,
            "evolution_protocol": evolution_protocol,
            "evolution_decision": evolution_decision,
        }
        freeze_path = staging_dir / "candidate_freeze.json"
        common.write_json_atomic(freeze_path, freeze)
        independent = {
            "schema": "evomind.siim.multiseed_independent_verification.v1",
            "created_at": now_iso(),
            "run_id": str(plan["run_id"]),
            "status": "passed",
            "passed": True,
            "train_rows": 28_984,
            "test_rows": 4_142,
            "positive_rows": int(values["target"].sum()),
            "oof_coverage_exactly_once": True,
            "patient_content_grouping_bound": True,
            "train_id_order_verified": True,
            "test_id_order_verified": True,
            "submission_schema_verified": True,
            "formal_ablation_seed_separation_verified": True,
            "recomputed_metrics": metrics,
            "candidate_freeze_sha256": common.sha256_file(freeze_path),
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "other_processes_modified": False,
            "signals_sent": 0,
            "claim_boundary": "Medical imaging research benchmark; not a clinical diagnosis.",
        }
        independent_path = staging_dir / "independent_verification.json"
        common.write_json_atomic(independent_path, independent)
        material = [
            candidate_path,
            oof_path,
            fold_path,
            metrics_path,
            frozen_plan_path,
            freeze_path,
            independent_path,
        ]
        artifact_manifest = {
            "schema": "evomind.siim.frozen_artifact_manifest.v1",
            "created_at": now_iso(),
            "run_id": str(plan["run_id"]),
            "artifacts": [
                staged_file_record(path, staging_root=staging_dir, final_root=final_output_dir)
                for path in material
            ],
            "artifact_count": len(material),
            "all_sha256_bound": True,
        }
        manifest_path = staging_dir / "artifact_manifest.json"
        common.write_json_atomic(manifest_path, artifact_manifest)
        candidate_sha256 = common.sha256_file(candidate_path)
        require(not final_output_dir.exists(), "SIIM frozen candidate directory appeared during build")
        os.replace(staging_dir, final_output_dir)
        return {
            "status": "frozen_before_private_grader",
            "run_id": str(plan["run_id"]),
            "output_dir": str(final_output_dir),
            "metrics": metrics,
            "candidate_sha256": candidate_sha256,
            "artifact_manifest": str(final_output_dir / "artifact_manifest.json"),
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--collected-root", type=Path, default=DEFAULT_COLLECTED_ROOT)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            aggregate(
                plan_path=args.plan,
                collected_root=args.collected_root,
                sample_path=args.sample_submission,
                output_dir=args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
