#!/usr/bin/env python3
"""Build a three-seed public-OOF Jigsaw candidate for Human Gate review.

The source prediction archives contain legacy object ID arrays.  This module
deliberately loads numeric arrays only with ``allow_pickle=False`` and rebuilds
the ID order from the hash-frozen public CSV files.
"""

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
from sklearn.metrics import roc_auc_score

EXPECTED_PLAN_SCHEMA = "evomind.jigsaw.multiseed_confirmation_plan.v1"
EXPECTED_RUN_SCHEMA = "evomind.jigsaw.transformer_run.v1"
EXPECTED_VERIFICATION_SCHEMA = "evomind.jigsaw.transformer_independent_verification.v1"
TARGET_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


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
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def safe_unicode_ids(values: Sequence[Any]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def close_metric(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return bool(np.isfinite([left, right]).all() and abs(left - right) <= tolerance)


def validate_probability(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"{name} must be an N x {len(TARGET_COLUMNS)} matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(f"{name} contains values outside [0, 1]")
    return matrix


def validate_truth(values: np.ndarray) -> np.ndarray:
    truth = np.asarray(values, dtype=np.int8)
    if truth.ndim != 2 or truth.shape[1] != len(TARGET_COLUMNS):
        raise ValueError("truth has an unexpected shape")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("truth must be binary")
    return truth


def mean_columnwise_auc(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, list[float]]:
    labels = validate_truth(truth)
    scores = validate_probability("prediction", prediction)
    if labels.shape != scores.shape:
        raise ValueError("Truth and prediction shapes differ")
    per_label = [
        float(roc_auc_score(labels[:, index], scores[:, index]))
        for index in range(labels.shape[1])
    ]
    return float(statistics.fmean(per_label)), per_label


def aggregate_probability_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("At least one probability matrix is required")
    validated = [
        validate_probability(f"seed_{index}", matrix)
        for index, matrix in enumerate(matrices)
    ]
    expected_shape = validated[0].shape
    if any(matrix.shape != expected_shape for matrix in validated[1:]):
        raise ValueError("Seed probability shapes differ")
    return validate_probability("probability_mean", np.mean(validated, axis=0))


def load_numeric_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load numeric arrays only; never deserialize legacy object ID arrays."""

    required = {
        "truth",
        "fold",
        "candidate_oof",
        "candidate_test",
        "candidate_write_counts",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        if not required <= set(archive.files):
            raise ValueError(f"Jigsaw bundle is incomplete: {path}")
        values = {
            "truth": validate_truth(archive["truth"]),
            "fold": np.asarray(archive["fold"], dtype=np.int16),
            "candidate_oof": validate_probability(
                "candidate_oof", archive["candidate_oof"]
            ),
            "candidate_test": validate_probability(
                "candidate_test", archive["candidate_test"]
            ),
            "candidate_write_counts": np.asarray(
                archive["candidate_write_counts"], dtype=np.uint8
            ),
        }
    if values["fold"].shape != (len(values["truth"]),):
        raise ValueError("Fold assignment shape differs from truth")
    if values["candidate_oof"].shape != values["truth"].shape:
        raise ValueError("Candidate OOF shape differs from truth")
    if values["candidate_write_counts"].shape != values["truth"].shape:
        raise ValueError("Candidate write-count shape differs from truth")
    if not np.all(values["candidate_write_counts"] == 1):
        raise ValueError("Candidate OOF was not written exactly once")
    return values


def validate_file_record(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Frozen {label} hash drifted: {path}")
    return path


def relocated_artifact_record_matches(
    observed: Mapping[str, Any], frozen: Mapping[str, Any]
) -> bool:
    """Match a materialized artifact by content when its path has changed."""

    observed_path = str(observed.get("path") or "").strip()
    observed_sha = str(observed.get("sha256") or "").strip().lower()
    frozen_sha = str(frozen.get("sha256") or "").strip().lower()
    return bool(
        observed_path
        and len(observed_sha) == 64
        and len(frozen_sha) == 64
        and observed_sha == frozen_sha
    )


def model_seed_from_run_plan(run_plan: Mapping[str, Any]) -> int:
    training = run_plan.get("training") or {}
    values = [
        int(training[name])
        for name in ("model_seed", "seed")
        if training.get(name) is not None
    ]
    if not values:
        raise ValueError("Jigsaw run plan does not freeze a model seed")
    if len(set(values)) != 1:
        raise ValueError("Jigsaw run plan seed fields disagree")
    return values[0]


def validate_plan(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    plan = read_json(path)
    if plan.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise ValueError("Unexpected Jigsaw multiseed plan schema")
    if plan.get("status") != "frozen_before_aggregation":
        raise ValueError("Jigsaw multiseed plan is not frozen")
    if plan.get("competition_id") != "jigsaw-toxic-comment-classification-challenge":
        raise ValueError("Jigsaw multiseed plan targets another competition")
    boundaries = plan.get("boundaries") or {}
    if any(
        boundaries.get(name) is not False
        for name in (
            "private_labels_used",
            "official_grader_executed",
            "kaggle_submission_executed",
        )
    ) or boundaries.get("process_signals_sent") != 0:
        raise ValueError("Jigsaw multiseed plan boundary is invalid")
    for label, record in (plan.get("implementation") or {}).items():
        validate_file_record(record, f"implementation {label}")
    for label, record in (plan.get("public_inputs") or {}).items():
        validate_file_record(record, f"public input {label}")
    if [int(record["model_seed"]) for record in plan.get("seed_runs", [])] != [
        40,
        41,
        42,
    ]:
        raise ValueError("Jigsaw multiseed plan must freeze seeds 40, 41, and 42")
    for seed in plan["seed_runs"]:
        for label in ("run_plan", "summary", "independent_verification", "prediction_bundle"):
            validate_file_record(seed[label], f"seed {seed['model_seed']} {label}")
        if "boundary_audit" in seed:
            validate_file_record(
                seed["boundary_audit"], f"seed {seed['model_seed']} boundary audit"
            )
    validate_file_record(plan["queue_status"], "confirmation queue status")
    plan["_path"] = str(path)
    plan["_sha256"] = sha256_file(path)
    return plan


def validate_seed_run(
    frozen: Mapping[str, Any], numeric: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    seed = int(frozen["model_seed"])
    run_plan = read_json(Path(frozen["run_plan"]["path"]))
    summary_path = Path(frozen["summary"]["path"]).resolve()
    verification_path = Path(frozen["independent_verification"]["path"]).resolve()
    summary = read_json(summary_path)
    verification = read_json(verification_path)
    if run_plan.get("status") != "frozen_before_training":
        raise ValueError(f"Seed {seed} run plan is not frozen")
    if model_seed_from_run_plan(run_plan) != seed:
        raise ValueError(f"Seed {seed} run plan seed mismatch")
    if summary.get("schema") != EXPECTED_RUN_SCHEMA:
        raise ValueError(f"Seed {seed} summary schema mismatch")
    if summary.get("run_id") != frozen.get("run_id"):
        raise ValueError(f"Seed {seed} run ID mismatch")
    if summary.get("plan_sha256") != frozen["run_plan"]["sha256"]:
        raise ValueError(f"Seed {seed} summary/plan hash mismatch")
    if summary.get("status") != "promotion_gate_passed":
        raise ValueError(f"Seed {seed} did not pass its promotion gate")
    if (summary.get("promotion_gate") or {}).get("passed") is not True:
        raise ValueError(f"Seed {seed} summary promotion gate is false")
    for name in (
        "private_labels_used",
        "official_grader_executed",
        "kaggle_submission_executed",
    ):
        if summary.get(name) is not False:
            raise ValueError(f"Seed {seed} summary boundary failed: {name}")
    if summary.get("human_gate_preserved") is not True:
        raise ValueError(f"Seed {seed} summary did not preserve Human Gate")
    prediction = summary.get("prediction_bundle") or {}
    if not relocated_artifact_record_matches(prediction, frozen["prediction_bundle"]):
        raise ValueError(f"Seed {seed} summary bundle record mismatch")
    summary_bundle_path = str(Path(prediction["path"]).resolve())
    frozen_bundle_path = str(Path(frozen["prediction_bundle"]["path"]).resolve())

    if verification.get("schema") != EXPECTED_VERIFICATION_SCHEMA:
        raise ValueError(f"Seed {seed} verification schema mismatch")
    if verification.get("run_id") != frozen.get("run_id"):
        raise ValueError(f"Seed {seed} verification run ID mismatch")
    if verification.get("plan_sha256") != frozen["run_plan"]["sha256"]:
        raise ValueError(f"Seed {seed} verification/plan hash mismatch")
    if verification.get("status") != "promotion_gate_passed":
        raise ValueError(f"Seed {seed} independent verification did not pass")
    if verification.get("full_contract_valid") is not True:
        raise ValueError(f"Seed {seed} independent contract is invalid")
    for name in (
        "private_labels_used",
        "official_grader_executed",
        "kaggle_submission_executed",
    ):
        if verification.get(name) is not False:
            raise ValueError(f"Seed {seed} verification boundary failed: {name}")

    candidate_auc, per_label = mean_columnwise_auc(
        numeric["truth"], numeric["candidate_oof"]
    )
    if not close_metric(candidate_auc, float(summary["candidate_oof_auc"])):
        raise ValueError(f"Seed {seed} summary AUC mismatch")
    metrics = verification.get("metrics") or {}
    if not close_metric(candidate_auc, float(metrics.get("candidate_auc", float("nan")))):
        raise ValueError(f"Seed {seed} verification AUC mismatch")
    gain = float((summary.get("promotion_gate") or {}).get("gain_over_strongest_base"))
    return {
        "model_seed": seed,
        "run_id": summary["run_id"],
        "status": summary["status"],
        "candidate_oof_auc": candidate_auc,
        "candidate_per_label_auc": dict(zip(TARGET_COLUMNS, per_label, strict=True)),
        "gain_over_strongest_base": gain,
        "promotion_passed": True,
        "run_plan": dict(frozen["run_plan"]),
        "summary": dict(frozen["summary"]),
        "independent_verification": dict(frozen["independent_verification"]),
        "prediction_bundle": dict(frozen["prediction_bundle"]),
        "prediction_bundle_provenance": {
            "summary_path": prediction["path"],
            "materialized_path": frozen["prediction_bundle"]["path"],
            "sha256": frozen["prediction_bundle"]["sha256"],
            "relocated": summary_bundle_path != frozen_bundle_path,
            "identity": "sha256",
        },
        **(
            {"boundary_audit": dict(frozen["boundary_audit"])}
            if "boundary_audit" in frozen
            else {}
        ),
    }


def confirmation_gate(
    *,
    plan_gate: Mapping[str, Any],
    seed_records: Sequence[Mapping[str, Any]],
    ensemble_auc: float,
    arrays_consistent: bool,
) -> dict[str, Any]:
    aucs = [float(record["candidate_oof_auc"]) for record in seed_records]
    gains = [float(record["gain_over_strongest_base"]) for record in seed_records]
    checks = {
        "exact_model_seeds_40_41_42": [
            int(record["model_seed"]) for record in seed_records
        ]
        == [40, 41, 42],
        "all_seed_promotion_gates_passed": all(
            record.get("promotion_passed") is True for record in seed_records
        ),
        "all_seed_auc_at_or_above_minimum": min(aucs)
        >= float(plan_gate["minimum_seed_auc"]),
        "mean_seed_auc_at_or_above_minimum": statistics.fmean(aucs)
        >= float(plan_gate["minimum_mean_auc"]),
        "ensemble_auc_at_or_above_minimum": ensemble_auc
        >= float(plan_gate["minimum_mean_auc"]),
        "all_seed_gain_at_or_above_minimum": min(gains)
        >= float(plan_gate["minimum_seed_gain"]),
        "population_std_at_or_below_maximum": statistics.pstdev(aucs)
        <= float(plan_gate["maximum_population_std"]),
        "truth_fold_arrays_identical": arrays_consistent,
        "numeric_bundles_loaded_without_pickle": True,
        "public_id_order_reconstructed": True,
        "private_labels_unused": True,
        "human_gate_preserved": True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def aggregate(plan_path: Path) -> dict[str, Any]:
    plan = validate_plan(plan_path)
    public_train = pd.read_csv(Path(plan["public_inputs"]["train"]["path"]))
    public_test = pd.read_csv(Path(plan["public_inputs"]["test"]["path"]))
    sample = pd.read_csv(Path(plan["public_inputs"]["sample_submission"]["path"]))
    if list(sample.columns) != ["id", *TARGET_COLUMNS]:
        raise ValueError("Unexpected Jigsaw sample-submission columns")
    if "id" not in public_train or "id" not in public_test:
        raise ValueError("Public Jigsaw CSV files do not contain IDs")
    train_ids = public_train["id"].astype(str).to_numpy()
    test_ids = public_test["id"].astype(str).to_numpy()
    if not np.array_equal(sample["id"].astype(str).to_numpy(), test_ids):
        raise ValueError("Public test and sample-submission ID orders differ")
    public_truth = validate_truth(
        public_train.loc[:, list(TARGET_COLUMNS)].to_numpy()
    )

    numeric_bundles: list[dict[str, np.ndarray]] = []
    seed_records: list[dict[str, Any]] = []
    for frozen in plan["seed_runs"]:
        numeric = load_numeric_bundle(Path(frozen["prediction_bundle"]["path"]))
        if len(numeric["truth"]) != len(train_ids):
            raise ValueError(f"Seed {frozen['model_seed']} train-row count mismatch")
        if len(numeric["candidate_test"]) != len(test_ids):
            raise ValueError(f"Seed {frozen['model_seed']} test-row count mismatch")
        if not np.array_equal(numeric["truth"], public_truth):
            raise ValueError(f"Seed {frozen['model_seed']} truth differs from public train.csv")
        numeric_bundles.append(numeric)
        seed_records.append(validate_seed_run(frozen, numeric))

    arrays_consistent = all(
        np.array_equal(numeric_bundles[0][name], bundle[name])
        for bundle in numeric_bundles[1:]
        for name in ("truth", "fold")
    )
    if not arrays_consistent:
        raise ValueError("Seed truth or fold arrays differ")
    ensemble_oof = aggregate_probability_matrices(
        [bundle["candidate_oof"] for bundle in numeric_bundles]
    )
    ensemble_test = aggregate_probability_matrices(
        [bundle["candidate_test"] for bundle in numeric_bundles]
    )
    ensemble_auc, ensemble_per_label = mean_columnwise_auc(public_truth, ensemble_oof)
    gate = confirmation_gate(
        plan_gate=plan["confirmation_gate"],
        seed_records=seed_records,
        ensemble_auc=ensemble_auc,
        arrays_consistent=arrays_consistent,
    )

    output_dir = Path(plan["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "jigsaw_multiseed_probability_mean_oof_and_test.npz"
    np.savez_compressed(
        bundle_path,
        truth=public_truth,
        fold=numeric_bundles[0]["fold"],
        candidate_oof=ensemble_oof,
        candidate_test=ensemble_test,
        candidate_write_counts=np.ones(public_truth.shape, dtype=np.uint8),
        train_id=safe_unicode_ids(train_ids),
        test_id=safe_unicode_ids(test_ids),
    )
    submission = sample.copy()
    submission.loc[:, list(TARGET_COLUMNS)] = ensemble_test
    submission_path = output_dir / "candidate_submission_withheld.csv"
    submission.to_csv(submission_path, index=False)

    aucs = [float(record["candidate_oof_auc"]) for record in seed_records]
    gains = [float(record["gain_over_strongest_base"]) for record in seed_records]
    result = {
        "schema": "evomind.jigsaw.multiseed_probability_confirmation_result.v1",
        "created_at": now_iso(),
        "status": (
            "confirmation_passed_human_gate_pending"
            if gate["passed"]
            else "confirmation_failed"
        ),
        "competition_id": "jigsaw-toxic-comment-classification-challenge",
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "aggregation_method": "arithmetic_mean_probability",
        "seed_records": seed_records,
        "metrics": {
            "seed_candidate_oof_auc": aucs,
            "minimum_seed_candidate_oof_auc": min(aucs),
            "mean_seed_candidate_oof_auc": statistics.fmean(aucs),
            "maximum_seed_candidate_oof_auc": max(aucs),
            "seed_candidate_oof_auc_population_std": statistics.pstdev(aucs),
            "minimum_seed_gain_over_strongest_base": min(gains),
            "ensemble_oof_auc": ensemble_auc,
            "ensemble_per_label_auc": dict(
                zip(TARGET_COLUMNS, ensemble_per_label, strict=True)
            ),
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
        "public_id_source": {
            "train": dict(plan["public_inputs"]["train"]),
            "test": dict(plan["public_inputs"]["test"]),
            "sample_submission": dict(plan["public_inputs"]["sample_submission"]),
        },
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": (
            "Three-seed public OOF probability confirmation is ready for Human Gate "
            "review; it is not an official score or medal."
        ),
    }
    result_path = Path(plan["output"]["result"]).resolve()
    write_json_atomic(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate(args.plan.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["candidate_ready_for_human_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
