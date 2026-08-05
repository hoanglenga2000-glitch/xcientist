#!/usr/bin/env python3
"""Aggregate three public-OOF Spooky XGBoost meta seeds for Human Gate review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EXPECTED_PLAN_SCHEMA = "evomind.spooky.cross_run_xgb_confirmation_plan.v1"
EXPECTED_RESULT_SCHEMA = "evomind.spooky.cross_run_xgb_result.v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_probability(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"{name} must be an N x 3 probability matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError(f"{name} contains invalid probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError(f"{name} probability rows are not normalized")
    return matrix


def multiclass_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(truth, dtype=np.int64).reshape(-1)
    matrix = validate_probability("metric_probability", probability)
    if len(target) != len(matrix):
        raise ValueError("Truth and probability row counts differ")
    return float(
        -np.log(np.clip(matrix[np.arange(len(target)), target], 1e-15, 1.0)).mean()
    )


def aggregate_probability_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("At least one probability matrix is required")
    validated = [validate_probability(f"seed_{index}", value) for index, value in enumerate(matrices)]
    expected_shape = validated[0].shape
    if any(value.shape != expected_shape for value in validated[1:]):
        raise ValueError("Seed probability shapes differ")
    return validate_probability("mean_probability", np.mean(validated, axis=0))


def confirmation_gate(
    *,
    model_seeds: Sequence[int],
    seed_scores: Sequence[float],
    ensemble_score: float,
    threshold: float,
) -> dict[str, Any]:
    scores = [float(value) for value in seed_scores]
    checks = {
        "exact_model_seeds_40_41_42": list(model_seeds) == [40, 41, 42],
        "all_seed_scores_at_or_below_threshold": bool(scores)
        and max(scores) <= threshold,
        "ensemble_score_at_or_below_threshold": ensemble_score <= threshold,
        "finite_scores": bool(
            np.isfinite(np.asarray(scores + [ensemble_score], dtype=np.float64)).all()
        ),
        "public_data_only": True,
        "human_gate_preserved": True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def validate_plan(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    plan = read_json(path)
    if plan.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise ValueError("Unexpected Spooky XGBoost confirmation-plan schema")
    if plan.get("status") != "frozen_before_aggregation":
        raise ValueError("Spooky XGBoost confirmation plan is not frozen")
    if plan.get("competition_id") != "spooky-author-identification":
        raise ValueError("Spooky XGBoost confirmation plan targets another competition")
    boundaries = plan.get("boundaries") or {}
    if any(
        boundaries.get(name) is not False
        for name in (
            "private_labels_used",
            "official_grader_executed",
            "kaggle_submission_executed",
        )
    ) or boundaries.get("process_signals_sent") != 0:
        raise ValueError("Spooky XGBoost confirmation boundary is invalid")
    for record in [plan["source"], plan["sample_submission"], *plan["seed_results"]]:
        artifact = Path(record["path"]).resolve()
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"Frozen confirmation artifact drift: {artifact}")
    plan["_path"] = str(path)
    plan["_sha256"] = sha256_file(path)
    return plan


def aggregate(plan_path: Path) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    records: list[dict[str, Any]] = []
    truths: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    oof_values: list[np.ndarray] = []
    test_values: list[np.ndarray] = []
    train_ids: list[np.ndarray] = []
    test_ids: list[np.ndarray] = []

    for frozen in sorted(plan["seed_results"], key=lambda item: int(item["model_seed"])):
        result_path = Path(frozen["path"]).resolve()
        result = read_json(result_path)
        if result.get("schema") != EXPECTED_RESULT_SCHEMA:
            raise ValueError(f"Unexpected seed result schema: {result_path}")
        if any(
            result.get(name) not in (False, 0)
            for name in (
                "private_labels_used",
                "official_grader_executed",
                "kaggle_submission_executed",
                "process_signals_sent",
            )
        ):
            raise ValueError(f"Seed result boundary failed: {result_path}")
        bundle_record = result.get("prediction_bundle") or {}
        bundle_path = Path(bundle_record.get("path", "")).resolve()
        if not bundle_path.is_file() or sha256_file(bundle_path) != bundle_record.get(
            "sha256"
        ):
            raise ValueError(f"Seed prediction bundle drift: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            required = {
                "truth",
                "fold",
                "candidate_oof",
                "candidate_test",
                "train_id",
                "test_id",
            }
            if not required <= set(bundle.files):
                raise ValueError(f"Seed bundle is incomplete: {bundle_path}")
            truths.append(np.asarray(bundle["truth"], dtype=np.int64))
            folds.append(np.asarray(bundle["fold"], dtype=np.int16))
            oof_values.append(validate_probability("candidate_oof", bundle["candidate_oof"]))
            test_values.append(validate_probability("candidate_test", bundle["candidate_test"]))
            train_ids.append(np.asarray(bundle["train_id"]).astype(str))
            test_ids.append(np.asarray(bundle["test_id"]).astype(str))
        records.append(
            {
                "model_seed": int(frozen["model_seed"]),
                "random_state": int(result["meta_contract"]["random_state"]),
                "run_id": result["run_id"],
                "status": result["status"],
                "candidate_oof_log_loss": float(result["candidate_oof_log_loss"]),
                "single_seed_passed": result.get("single_seed_passed") is True,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "bundle_path": str(bundle_path),
                "bundle_sha256": sha256_file(bundle_path),
            }
        )

    for name, arrays in (
        ("truth", truths),
        ("fold", folds),
        ("train_id", train_ids),
        ("test_id", test_ids),
    ):
        if any(not np.array_equal(arrays[0], value) for value in arrays[1:]):
            raise ValueError(f"Seed {name} arrays differ")

    ensemble_oof = aggregate_probability_matrices(oof_values)
    ensemble_test = aggregate_probability_matrices(test_values)
    seed_scores = [multiclass_log_loss(truths[0], value) for value in oof_values]
    ensemble_score = multiclass_log_loss(truths[0], ensemble_oof)
    threshold = float(plan["confirmation_gate"]["maximum_log_loss"])
    gate = confirmation_gate(
        model_seeds=[record["model_seed"] for record in records],
        seed_scores=seed_scores,
        ensemble_score=ensemble_score,
        threshold=threshold,
    )
    gate["checks"]["all_source_seed_gates_passed"] = all(
        record["single_seed_passed"] for record in records
    )
    gate["passed"] = all(gate["checks"].values())

    output_dir = Path(plan["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "spooky_cross_run_xgb_multiseed_oof_and_test.npz"
    np.savez_compressed(
        bundle_path,
        truth=truths[0],
        fold=folds[0],
        candidate_oof=ensemble_oof,
        candidate_test=ensemble_test,
        train_id=train_ids[0],
        test_id=test_ids[0],
    )
    sample = pd.read_csv(Path(plan["sample_submission"]["path"]))
    if not np.array_equal(sample["id"].astype(str).to_numpy(), test_ids[0]):
        raise ValueError("Sample submission IDs differ from seed bundles")
    class_columns = [column for column in sample.columns if column != "id"]
    if len(class_columns) != 3:
        raise ValueError("Spooky sample submission must contain three class columns")
    sample.loc[:, class_columns] = ensemble_test
    submission_path = output_dir / "candidate_submission_withheld.csv"
    sample.to_csv(submission_path, index=False)

    result = {
        "schema": "evomind.spooky.cross_run_xgb_confirmation_result.v1",
        "created_at": now_iso(),
        "status": (
            "confirmation_passed_human_gate_pending"
            if gate["passed"]
            else "confirmation_failed"
        ),
        "competition_id": "spooky-author-identification",
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "seed_records": records,
        "metrics": {
            "seed_log_loss": seed_scores,
            "minimum_seed_log_loss": min(seed_scores),
            "mean_seed_log_loss": statistics.fmean(seed_scores),
            "maximum_seed_log_loss": max(seed_scores),
            "seed_log_loss_population_std": statistics.pstdev(seed_scores),
            "ensemble_oof_log_loss": ensemble_score,
        },
        "confirmation_gate": {**plan["confirmation_gate"], **gate},
        "candidate_ready_for_human_gate": gate["passed"],
        "prediction_bundle": {
            "path": str(bundle_path),
            "sha256": sha256_file(bundle_path),
        },
        "submission_withheld": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": (
            "Three-seed public OOF confirmation is ready for Human Gate review; "
            "it is not an official score or medal."
        ),
    }
    result_path = Path(plan["output"]["result"]).resolve()
    write_json_atomic(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = aggregate(args.plan.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["candidate_ready_for_human_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
