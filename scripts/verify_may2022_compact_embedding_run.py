#!/usr/bin/env python3
"""Independently verify a completed May-2022 compact-embedding OOF run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_compact_embedding_s42_frozen_plan.json"
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
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_fold_record(record: dict[str, Any], fold: int) -> None:
    require(int(record.get("fold", -1)) == fold, "May fold record index mismatch")
    require(
        record.get("outer_validation_used_for_epoch_selection") is False,
        "May outer validation was used for epoch selection",
    )
    require(record.get("outer_refit_fixed_budget") is True, "May outer refit was not fixed-budget")
    selected_epoch = int(record.get("selected_epoch", 0))
    require(selected_epoch >= 1, "May selected epoch is invalid")
    selection_history = record.get("selection_history") or []
    refit_history = record.get("refit_history") or []
    require(selection_history, "May inner selection history is missing")
    require(
        all("inner_validation_auc" in item for item in selection_history),
        "May inner selection history lacks validation AUC",
    )
    require(
        len(refit_history) == selected_epoch,
        "May outer refit epoch count differs from the selected fixed budget",
    )
    require(
        all("inner_validation_auc" not in item for item in refit_history),
        "May outer refit unexpectedly contains outer validation metrics",
    )


def assemble_fold_predictions(
    run_dir: Path,
    fold_assignment: np.ndarray,
    *,
    fold_count: int,
    test_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    oof = np.full(len(fold_assignment), np.nan, dtype=np.float64)
    test_sum = np.zeros(test_rows, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in range(fold_count):
        fold_dir = run_dir / f"fold_{fold:02d}"
        result_path = fold_dir / "result.npz"
        metadata_path = fold_dir / "metadata.json"
        checkpoint_path = fold_dir / "model.pt"
        require(result_path.is_file(), f"May fold {fold} result is missing")
        require(metadata_path.is_file(), f"May fold {fold} metadata is missing")
        require(checkpoint_path.is_file(), f"May fold {fold} checkpoint is missing")
        metadata = read_json(metadata_path)
        require(
            metadata.get("result_sha256") == sha256_file(result_path),
            f"May fold {fold} result hash mismatch",
        )
        require(
            metadata.get("checkpoint_sha256") == sha256_file(checkpoint_path),
            f"May fold {fold} checkpoint hash mismatch",
        )
        require(metadata.get("private_labels_used") is False, "May fold used private labels")
        require(
            metadata.get("official_grader_executed") is False,
            "May fold executed the official grader",
        )
        require(metadata.get("process_signals_sent") == 0, "May fold sent process signals")
        record = metadata.get("fold_record") or {}
        validate_fold_record(record, fold)
        expected_validation = np.flatnonzero(fold_assignment == fold)
        with np.load(result_path, allow_pickle=False) as result:
            validation = np.asarray(result["validation_indices"], dtype=np.int64)
            valid_probability = np.asarray(result["validation_probability"], dtype=np.float64)
            test_probability = np.asarray(result["test_probability"], dtype=np.float64)
        require(
            np.array_equal(validation, expected_validation),
            f"May fold {fold} validation indices differ from the manifest",
        )
        require(
            len(valid_probability) == len(validation) and np.isfinite(valid_probability).all(),
            f"May fold {fold} validation probabilities are invalid",
        )
        require(
            len(test_probability) == test_rows and np.isfinite(test_probability).all(),
            f"May fold {fold} test probabilities are invalid",
        )
        oof[validation] = valid_probability
        test_sum += test_probability / fold_count
        records.append(record)
    require(np.isfinite(oof).all(), "May independently assembled OOF is incomplete")
    return oof, test_sum, records


def evaluate_gate(
    target: np.ndarray,
    fold_assignment: np.ndarray,
    probability: np.ndarray,
    *,
    aggregate_threshold: float,
    every_fold_threshold: float,
) -> dict[str, Any]:
    aggregate = float(roc_auc_score(target, probability))
    fold_auc = {
        str(int(fold)): float(
            roc_auc_score(target[fold_assignment == fold], probability[fold_assignment == fold])
        )
        for fold in sorted(np.unique(fold_assignment).tolist())
    }
    aggregate_passed = aggregate >= aggregate_threshold
    folds_passed = min(fold_auc.values()) >= every_fold_threshold
    return {
        "aggregate_oof_auc": aggregate,
        "fold_auc": fold_auc,
        "minimum_fold_auc": min(fold_auc.values()),
        "aggregate_passed": aggregate_passed,
        "all_folds_passed": folds_passed,
        "passed": bool(aggregate_passed and folds_passed),
    }


def verify_run(run_dir: Path, plan_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    plan_path = plan_path.resolve()
    plan = read_json(plan_path)
    plan_sha256 = sha256_file(plan_path)
    runner_path = Path(plan["runner_path"])
    require(plan.get("status") == "frozen_before_training", "May plan is not frozen")
    require(sha256_file(runner_path) == plan["runner_sha256"], "May runner hash mismatch")
    summary_path = run_dir / "summary.json"
    contract_path = run_dir / "run_contract.json"
    manifest_path = run_dir / "fold_manifest.npz"
    bundle_path = run_dir / "may2022_compact_oof_bundle.npz"
    candidate_path = run_dir / "candidate_submission_withheld.csv"
    for path in (summary_path, contract_path, manifest_path, bundle_path, candidate_path):
        require(path.is_file(), f"May required run artifact is missing: {path.name}")
    summary = read_json(summary_path)
    contract = read_json(contract_path)
    require(summary.get("plan_sha256") == plan_sha256, "May summary plan hash mismatch")
    require(contract.get("plan_sha256") == plan_sha256, "May run contract plan hash mismatch")
    require(summary.get("runner_sha256") == plan["runner_sha256"], "May summary runner mismatch")
    require(summary.get("seed") == plan["seed"], "May summary seed mismatch")
    require(summary.get("folds") == plan["folds"], "May summary fold count mismatch")
    require(summary.get("private_labels_used") is False, "May summary used private labels")
    require(summary.get("official_grader_executed") is False, "May summary ran grader")
    require(summary.get("kaggle_submission_executed") is False, "May summary submitted")
    require(summary.get("submission_withheld") is True, "May candidate was not withheld")
    require(summary.get("process_signals_sent") == 0, "May summary sent process signals")

    public_dir = Path(plan["execution"]["public_dir"])
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_path = public_dir / "sample_submission.csv"
    hashes = plan["public_input_sha256"]
    require(sha256_file(train_path) == hashes["train"], "May train hash mismatch")
    require(sha256_file(test_path) == hashes["test"], "May test hash mismatch")
    require(sha256_file(sample_path) == hashes["sample"], "May sample hash mismatch")
    train = pd.read_csv(train_path, usecols=["id", "target"])
    test = pd.read_csv(test_path, usecols=["id"])
    sample = pd.read_csv(sample_path, usecols=["id"])
    require(len(train) == 800_000 and len(test) == 100_000, "May public row count mismatch")
    require(np.array_equal(test["id"].to_numpy(), sample["id"].to_numpy()), "May test IDs mismatch")
    with np.load(manifest_path, allow_pickle=False) as manifest:
        manifest_id = np.asarray(manifest["id"])
        manifest_target = np.asarray(manifest["target"], dtype=np.int8)
        fold_assignment = np.asarray(manifest["fold_assignment"], dtype=np.int16)
        manifest_seed = int(np.asarray(manifest["seed"]).ravel()[0])
    require(np.array_equal(manifest_id, train["id"].to_numpy()), "May fold manifest IDs mismatch")
    require(
        np.array_equal(manifest_target, train["target"].to_numpy(dtype=np.int8)),
        "May fold manifest target mismatch",
    )
    require(manifest_seed == plan["seed"], "May fold manifest seed mismatch")
    require(
        set(np.unique(fold_assignment).tolist()) == set(range(plan["folds"])),
        "May fold manifest coverage mismatch",
    )
    assembled_oof, assembled_test, fold_records = assemble_fold_predictions(
        run_dir, fold_assignment, fold_count=plan["folds"], test_rows=len(test)
    )
    with np.load(bundle_path, allow_pickle=False) as bundle:
        bundle_id = np.asarray(bundle["id"])
        bundle_target = np.asarray(bundle["target"], dtype=np.int8)
        bundle_fold = np.asarray(bundle["fold_assignment"], dtype=np.int16)
        bundle_oof = np.asarray(bundle["oof_probability"], dtype=np.float64)
        bundle_test_id = np.asarray(bundle["test_id"])
        bundle_test = np.asarray(bundle["test_probability"], dtype=np.float64)
    require(np.array_equal(bundle_id, manifest_id), "May bundle train IDs mismatch")
    require(np.array_equal(bundle_target, manifest_target), "May bundle target mismatch")
    require(np.array_equal(bundle_fold, fold_assignment), "May bundle fold mismatch")
    require(np.array_equal(bundle_test_id, test["id"].to_numpy()), "May bundle test IDs mismatch")
    require(np.allclose(bundle_oof, assembled_oof, atol=1e-7), "May bundle OOF mismatch")
    require(np.allclose(bundle_test, assembled_test, atol=1e-7), "May bundle test mismatch")
    gate = evaluate_gate(
        manifest_target,
        fold_assignment,
        bundle_oof,
        aggregate_threshold=float(plan["gates"]["aggregate_oof_auc"]),
        every_fold_threshold=float(plan["gates"]["every_fold_oof_auc"]),
    )
    require(
        abs(gate["aggregate_oof_auc"] - float(summary["exact_deployment_oof_auc"])) < 1e-9,
        "May summary aggregate AUC differs from independent recomputation",
    )
    require(
        abs(gate["minimum_fold_auc"] - float(summary["minimum_fold_auc"])) < 1e-9,
        "May summary minimum fold AUC differs from independent recomputation",
    )
    require(summary["single_seed_gate"]["passed"] == gate["passed"], "May gate mismatch")
    require(summary["multi_seed_confirmation"]["pending"] is True, "May confirmation not pending")
    require(summary["multi_seed_confirmation"]["passed"] is False, "May confirmation overclaimed")
    require(summary.get("fold_records") == fold_records, "May summary fold records mismatch")
    candidate = pd.read_csv(candidate_path)
    require(np.array_equal(candidate["id"].to_numpy(), test["id"].to_numpy()), "May candidate IDs mismatch")
    require(
        np.allclose(candidate["target"].to_numpy(), bundle_test, atol=1e-7),
        "May candidate predictions mismatch",
    )
    return {
        "schema": "evomind.mlebench.may2022_compact_independent_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed",
        "ok": True,
        "run_dir": str(run_dir),
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "runner_sha256": plan["runner_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "bundle_sha256": sha256_file(bundle_path),
        "candidate_sha256": sha256_file(candidate_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "fold_count": plan["folds"],
        "gate": gate,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "submission_withheld": True,
        "process_signals_sent": 0,
        "claim_boundary": "Independent public OOF verification is not an official medal.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.run_dir / "independent_verification.json"
    try:
        report = verify_run(args.run_dir, args.plan)
    except Exception as exc:
        report = {
            "schema": "evomind.mlebench.may2022_compact_independent_verification.v1",
            "created_at": now_iso(),
            "status": "verification_failed",
            "ok": False,
            "error_type": type(exc).__name__,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        }
    write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
