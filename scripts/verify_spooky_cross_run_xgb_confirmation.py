#!/usr/bin/env python3
"""Independently verify the public-OOF Spooky XGBoost confirmation artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_spooky_cross_run_xgb_confirmation import (  # noqa: E402
    multiclass_log_loss,
    read_json,
    sha256_file,
    validate_probability,
    write_json_atomic,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def close_metric(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return bool(np.isfinite([left, right]).all() and abs(left - right) <= tolerance)


def verify(plan_path: Path, result_path: Path, output_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    result_path = Path(result_path).resolve()
    plan = read_json(plan_path)
    result = read_json(result_path)
    errors: list[str] = []

    if plan.get("schema") != "evomind.spooky.cross_run_xgb_confirmation_plan.v1":
        errors.append("plan_schema")
    if result.get("schema") != "evomind.spooky.cross_run_xgb_confirmation_result.v1":
        errors.append("result_schema")
    if result.get("plan_path") != str(plan_path):
        errors.append("plan_path")
    if result.get("plan_sha256") != sha256_file(plan_path):
        errors.append("plan_sha256")
    if result.get("status") != "confirmation_passed_human_gate_pending":
        errors.append("result_status")

    for name in (
        "private_labels_used",
        "official_grader_executed",
        "kaggle_submission_executed",
    ):
        if result.get(name) is not False:
            errors.append(name)
    if result.get("process_signals_sent") != 0:
        errors.append("process_signals_sent")

    seed_records = result.get("seed_records") or []
    if [record.get("model_seed") for record in seed_records] != [40, 41, 42]:
        errors.append("model_seeds")
    frozen_by_seed = {
        int(record["model_seed"]): record for record in plan.get("seed_results", [])
    }
    recomputed_seed_scores: list[float] = []
    for record in seed_records:
        seed = int(record.get("model_seed", -1))
        frozen = frozen_by_seed.get(seed)
        source = Path(record.get("result_path", "")).resolve()
        if frozen is None or source != Path(frozen["path"]).resolve():
            errors.append(f"seed_{seed}_path")
            continue
        if not source.is_file() or sha256_file(source) != frozen["sha256"]:
            errors.append(f"seed_{seed}_hash")
            continue
        seed_result = read_json(source)
        if seed_result.get("single_seed_passed") is not True:
            errors.append(f"seed_{seed}_gate")
        recomputed_seed_scores.append(float(seed_result["candidate_oof_log_loss"]))

    bundle_record = result.get("prediction_bundle") or {}
    bundle_path = Path(bundle_record.get("path", "")).resolve()
    truth = np.empty(0, dtype=np.int64)
    test_id = np.empty(0, dtype=str)
    candidate_oof = np.empty((0, 3), dtype=np.float64)
    candidate_test = np.empty((0, 3), dtype=np.float64)
    if not bundle_path.is_file() or sha256_file(bundle_path) != bundle_record.get("sha256"):
        errors.append("prediction_bundle")
    else:
        with np.load(bundle_path, allow_pickle=False) as bundle:
            required = {"truth", "fold", "candidate_oof", "candidate_test", "test_id"}
            if not required <= set(bundle.files):
                errors.append("prediction_bundle_arrays")
            else:
                truth = np.asarray(bundle["truth"], dtype=np.int64)
                test_id = np.asarray(bundle["test_id"]).astype(str)
                candidate_oof = validate_probability(
                    "candidate_oof", bundle["candidate_oof"]
                )
                candidate_test = validate_probability(
                    "candidate_test", bundle["candidate_test"]
                )

    ensemble_score = (
        multiclass_log_loss(truth, candidate_oof) if len(truth) == len(candidate_oof) else float("nan")
    )
    reported_metrics = result.get("metrics") or {}
    if not close_metric(
        ensemble_score, float(reported_metrics.get("ensemble_oof_log_loss", float("nan")))
    ):
        errors.append("ensemble_oof_log_loss")
    reported_seed_scores = [float(value) for value in reported_metrics.get("seed_log_loss", [])]
    if len(recomputed_seed_scores) != 3 or not np.allclose(
        recomputed_seed_scores, reported_seed_scores, rtol=0.0, atol=1e-12
    ):
        errors.append("seed_log_loss")

    submission_record = result.get("submission_withheld") or {}
    submission_path = Path(submission_record.get("path", "")).resolve()
    if not submission_path.is_file() or sha256_file(submission_path) != submission_record.get(
        "sha256"
    ):
        errors.append("submission_withheld")
    elif len(candidate_test):
        submission = pd.read_csv(submission_path)
        columns = [column for column in submission.columns if column != "id"]
        if not np.array_equal(submission["id"].astype(str).to_numpy(), test_id):
            errors.append("submission_ids")
        if not np.allclose(
            submission[columns].to_numpy(dtype=np.float64),
            candidate_test,
            rtol=0.0,
            atol=1e-12,
        ):
            errors.append("submission_probabilities")

    threshold = float(plan.get("confirmation_gate", {}).get("maximum_log_loss", float("nan")))
    if not (
        np.isfinite(threshold)
        and len(recomputed_seed_scores) == 3
        and max(recomputed_seed_scores) <= threshold
        and ensemble_score <= threshold
    ):
        errors.append("confirmation_threshold")
    if result.get("confirmation_gate", {}).get("passed") is not True:
        errors.append("confirmation_gate")

    passed = not errors
    report = {
        "schema": "evomind.spooky.cross_run_xgb_confirmation_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed" if passed else "verification_failed",
        "ok": passed,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "recomputed_seed_log_loss": recomputed_seed_scores,
        "recomputed_ensemble_oof_log_loss": ensemble_score,
        "maximum_log_loss": threshold,
        "candidate_ready_for_human_gate": passed,
        "errors": errors,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": (
            "Independent public OOF verification is not an official score or medal."
        ),
    }
    write_json_atomic(output_path.resolve(), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify(args.plan, args.result, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
