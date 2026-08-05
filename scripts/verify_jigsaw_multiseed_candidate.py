#!/usr/bin/env python3
"""Independently verify a three-seed public-OOF Jigsaw candidate."""

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

from scripts.aggregate_jigsaw_multiseed_candidate import (  # noqa: E402
    TARGET_COLUMNS,
    aggregate_probability_matrices,
    close_metric,
    load_numeric_bundle,
    mean_columnwise_auc,
    read_json,
    safe_unicode_ids,
    sha256_file,
    validate_probability,
    validate_truth,
    write_json_atomic,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def verify(plan_path: Path, result_path: Path, output_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    result_path = Path(result_path).resolve()
    output_path = Path(output_path).resolve()
    plan = read_json(plan_path)
    result = read_json(result_path)
    errors: list[str] = []

    if plan.get("schema") != "evomind.jigsaw.multiseed_confirmation_plan.v1":
        errors.append("plan_schema")
    if result.get("schema") != "evomind.jigsaw.multiseed_probability_confirmation_result.v1":
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
    if result.get("human_gate_preserved") is not True:
        errors.append("human_gate_preserved")

    public_train = pd.read_csv(Path(plan["public_inputs"]["train"]["path"]))
    public_test = pd.read_csv(Path(plan["public_inputs"]["test"]["path"]))
    sample = pd.read_csv(Path(plan["public_inputs"]["sample_submission"]["path"]))
    public_truth = validate_truth(
        public_train.loc[:, list(TARGET_COLUMNS)].to_numpy()
    )
    train_ids = safe_unicode_ids(public_train["id"].astype(str).tolist())
    test_ids = safe_unicode_ids(public_test["id"].astype(str).tolist())
    if list(sample.columns) != ["id", *TARGET_COLUMNS]:
        errors.append("sample_columns")
    if not np.array_equal(sample["id"].astype(str).to_numpy(), test_ids.astype(str)):
        errors.append("public_test_sample_id_order")

    frozen_by_seed = {
        int(record["model_seed"]): record for record in plan.get("seed_runs", [])
    }
    reported_records = result.get("seed_records") or []
    if [int(record.get("model_seed", -1)) for record in reported_records] != [40, 41, 42]:
        errors.append("model_seeds")

    source_bundles: list[dict[str, np.ndarray]] = []
    recomputed_seed_auc: list[float] = []
    recomputed_seed_gain: list[float] = []
    for reported in reported_records:
        seed = int(reported.get("model_seed", -1))
        frozen = frozen_by_seed.get(seed)
        if frozen is None:
            errors.append(f"seed_{seed}_frozen_record")
            continue
        for label in (
            "run_plan",
            "summary",
            "independent_verification",
            "prediction_bundle",
        ):
            source = Path(frozen[label]["path"]).resolve()
            if not source.is_file() or sha256_file(source) != frozen[label]["sha256"]:
                errors.append(f"seed_{seed}_{label}_hash")
        bundle_path = Path(frozen["prediction_bundle"]["path"]).resolve()
        if not bundle_path.is_file():
            continue
        try:
            numeric = load_numeric_bundle(bundle_path)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"seed_{seed}_bundle:{type(exc).__name__}")
            continue
        if not np.array_equal(numeric["truth"], public_truth):
            errors.append(f"seed_{seed}_truth")
        if len(numeric["candidate_test"]) != len(test_ids):
            errors.append(f"seed_{seed}_test_rows")
        score, _per_label = mean_columnwise_auc(
            numeric["truth"], numeric["candidate_oof"]
        )
        source_summary = read_json(Path(frozen["summary"]["path"]))
        source_verification = read_json(
            Path(frozen["independent_verification"]["path"])
        )
        if not close_metric(score, float(source_summary["candidate_oof_auc"])):
            errors.append(f"seed_{seed}_summary_auc")
        if not close_metric(
            score,
            float(
                (source_verification.get("metrics") or {}).get(
                    "candidate_auc", float("nan")
                )
            ),
        ):
            errors.append(f"seed_{seed}_verification_auc")
        reported_score = float(reported.get("candidate_oof_auc", float("nan")))
        if not close_metric(score, reported_score):
            errors.append(f"seed_{seed}_reported_auc")
        gain = float(
            (source_summary.get("promotion_gate") or {}).get(
                "gain_over_strongest_base", float("nan")
            )
        )
        if not close_metric(
            gain, float(reported.get("gain_over_strongest_base", float("nan")))
        ):
            errors.append(f"seed_{seed}_reported_gain")
        recomputed_seed_auc.append(score)
        recomputed_seed_gain.append(gain)
        source_bundles.append(numeric)

    ensemble_oof = np.empty((0, len(TARGET_COLUMNS)), dtype=np.float64)
    ensemble_test = np.empty((0, len(TARGET_COLUMNS)), dtype=np.float64)
    ensemble_auc = float("nan")
    ensemble_per_label: list[float] = []
    if len(source_bundles) == 3:
        for name in ("truth", "fold"):
            if any(
                not np.array_equal(source_bundles[0][name], bundle[name])
                for bundle in source_bundles[1:]
            ):
                errors.append(f"source_{name}_arrays")
        ensemble_oof = aggregate_probability_matrices(
            [bundle["candidate_oof"] for bundle in source_bundles]
        )
        ensemble_test = aggregate_probability_matrices(
            [bundle["candidate_test"] for bundle in source_bundles]
        )
        ensemble_auc, ensemble_per_label = mean_columnwise_auc(
            public_truth, ensemble_oof
        )

    output_bundle_record = result.get("prediction_bundle") or {}
    output_bundle_path = Path(output_bundle_record.get("path", "")).resolve()
    if (
        not output_bundle_path.is_file()
        or sha256_file(output_bundle_path) != output_bundle_record.get("sha256")
    ):
        errors.append("prediction_bundle")
    elif len(ensemble_oof):
        try:
            with np.load(output_bundle_path, allow_pickle=False) as archive:
                required = {
                    "truth",
                    "fold",
                    "candidate_oof",
                    "candidate_test",
                    "candidate_write_counts",
                    "train_id",
                    "test_id",
                }
                if not required <= set(archive.files):
                    errors.append("prediction_bundle_arrays")
                else:
                    checks = {
                        "truth": np.array_equal(archive["truth"], public_truth),
                        "fold": np.array_equal(
                            archive["fold"], source_bundles[0]["fold"]
                        ),
                        "candidate_oof": np.allclose(
                            validate_probability(
                                "candidate_oof", archive["candidate_oof"]
                            ),
                            ensemble_oof,
                            rtol=0.0,
                            atol=1e-15,
                        ),
                        "candidate_test": np.allclose(
                            validate_probability(
                                "candidate_test", archive["candidate_test"]
                            ),
                            ensemble_test,
                            rtol=0.0,
                            atol=1e-15,
                        ),
                        "candidate_write_counts": np.all(
                            archive["candidate_write_counts"] == 1
                        ),
                        "train_id": np.array_equal(
                            archive["train_id"].astype(str), train_ids.astype(str)
                        ),
                        "test_id": np.array_equal(
                            archive["test_id"].astype(str), test_ids.astype(str)
                        ),
                    }
                    errors.extend(
                        f"prediction_bundle_{name}"
                        for name, passed in checks.items()
                        if not bool(passed)
                    )
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"prediction_bundle_load:{type(exc).__name__}")

    submission_record = result.get("submission_withheld") or {}
    submission_path = Path(submission_record.get("path", "")).resolve()
    if (
        not submission_path.is_file()
        or sha256_file(submission_path) != submission_record.get("sha256")
    ):
        errors.append("submission_withheld")
    elif len(ensemble_test):
        submission = pd.read_csv(submission_path)
        if list(submission.columns) != list(sample.columns):
            errors.append("submission_columns")
        if not np.array_equal(
            submission["id"].astype(str).to_numpy(), test_ids.astype(str)
        ):
            errors.append("submission_ids")
        if not np.allclose(
            submission.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.float64),
            ensemble_test,
            rtol=0.0,
            atol=5e-16,
        ):
            errors.append("submission_probabilities")

    reported_metrics = result.get("metrics") or {}
    if len(recomputed_seed_auc) != 3 or not np.allclose(
        recomputed_seed_auc,
        reported_metrics.get("seed_candidate_oof_auc", []),
        rtol=0.0,
        atol=1e-12,
    ):
        errors.append("seed_candidate_oof_auc")
    if not close_metric(
        ensemble_auc, float(reported_metrics.get("ensemble_oof_auc", float("nan")))
    ):
        errors.append("ensemble_oof_auc")

    gate = plan.get("confirmation_gate") or {}
    if not (
        len(recomputed_seed_auc) == 3
        and min(recomputed_seed_auc) >= float(gate["minimum_seed_auc"])
        and float(np.mean(recomputed_seed_auc)) >= float(gate["minimum_mean_auc"])
        and ensemble_auc >= float(gate["minimum_mean_auc"])
        and min(recomputed_seed_gain) >= float(gate["minimum_seed_gain"])
        and float(np.std(recomputed_seed_auc))
        <= float(gate["maximum_population_std"])
    ):
        errors.append("confirmation_thresholds")
    if (result.get("confirmation_gate") or {}).get("passed") is not True:
        errors.append("confirmation_gate")
    if result.get("candidate_ready_for_human_gate") is not True:
        errors.append("candidate_ready_for_human_gate")

    passed = not errors
    report = {
        "schema": "evomind.jigsaw.multiseed_probability_confirmation_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed" if passed else "verification_failed",
        "ok": passed,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "recomputed_seed_candidate_oof_auc": recomputed_seed_auc,
        "recomputed_seed_gain_over_strongest_base": recomputed_seed_gain,
        "recomputed_ensemble_oof_auc": ensemble_auc,
        "recomputed_ensemble_per_label_auc": dict(
            zip(TARGET_COLUMNS, ensemble_per_label, strict=True)
        ),
        "candidate_ready_for_human_gate": passed,
        "errors": errors,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": (
            "Independent three-seed public OOF verification is not an official score "
            "or medal."
        ),
    }
    write_json_atomic(output_path, report)
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
