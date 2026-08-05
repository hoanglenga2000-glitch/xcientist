#!/usr/bin/env python3
"""Aggregate May-2022 seeds 42/43/44 into a verified Human Gate package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402

COMPETITION_ID = "tabular-playground-series-may-2022"
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
)
DEFAULT_COLLECTED_ROOT = common.DEFAULT_COLLECTED_ROOT
DEFAULT_SAMPLE = (
    PROJECT_ROOT
    / "workspace"
    / "local_gpu"
    / "mlebench_official_data"
    / COMPETITION_ID
    / "prepared"
    / "public"
    / "sample_submission.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "human_gate" / "may2022_nested_multiseed_s42_s43_s44_20260728"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise common.TaxiAggregationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed_artifacts(run_root: Path, seed: int) -> dict[str, Any]:
    indexed = common._validate_collection(run_root)
    competition = run_root / COMPETITION_ID
    result_path = competition / "result.json"
    require(str(result_path.resolve()) in indexed, "May top-level result was not collected")
    result = _read_json(result_path)
    require(result.get("competition_id") == COMPETITION_ID, "May competition mismatch")
    require(result.get("metric") == "roc_auc" and result.get("direction") == "maximize", "May metric drift")
    require(result.get("valid_submission") is True, "May submission validation failed")
    require(result.get("official_grader_executed") is False, "May run executed grader")
    gate = result.get("promotion_gate") or {}
    require(gate.get("passed") is True, f"May seed {seed} promotion gate failed")
    bundle_candidates = [
        path for path in competition.rglob("may2022_oof_ensemble.npz") if path.is_file()
    ]
    require(len(bundle_candidates) == 1, "May OOF bundle inventory changed")
    bundle_path = bundle_candidates[0]
    require(str(bundle_path.resolve()) in indexed, "May OOF bundle was not hash-collected")
    submissions = [
        path
        for path in competition.rglob("submission.csv")
        if path.is_file() and common.sha256_file(path) == result.get("submission_sha256")
    ]
    require(bool(submissions), "May withheld submission was not hash-collected")
    return {
        "seed": seed,
        "run_id": run_root.name,
        "result": result,
        "result_path": result_path,
        "bundle_path": bundle_path,
        "submission_path": sorted(submissions, key=lambda path: len(path.parts))[0],
    }


def _load_and_verify(seed_items: list[dict[str, Any]]) -> dict[str, Any]:
    reference: dict[str, np.ndarray] | None = None
    predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    seed_auc: list[float] = []
    seed_min_fold_auc: list[float] = []
    for item in seed_items:
        with np.load(item["bundle_path"], allow_pickle=False) as bundle:
            current = {
                "id": np.asarray(bundle["id"]),
                "target": np.asarray(bundle["target"], dtype=np.int8),
                "fold": np.asarray(bundle["fold_assignment"], dtype=np.int16),
                "oof": np.asarray(bundle["oof_probability"], dtype=np.float64),
                "test_id": np.asarray(bundle["test_id"]),
                "test": np.asarray(bundle["test_probability"], dtype=np.float64),
            }
        require(np.isfinite(current["oof"]).all() and np.isfinite(current["test"]).all(), "May predictions are non-finite")
        if reference is None:
            reference = current
        else:
            for key in ("id", "target", "fold", "test_id"):
                require(np.array_equal(current[key], reference[key]), f"May multiseed alignment drifted: {key}")
        auc = float(roc_auc_score(current["target"], current["oof"]))
        folds = [
            float(roc_auc_score(current["target"][current["fold"] == fold], current["oof"][current["fold"] == fold]))
            for fold in sorted(np.unique(current["fold"]).tolist())
        ]
        require(abs(auc - float(item["result"]["cv_score"])) <= 1e-9, "May result/OOF AUC drifted")
        predictions.append(current["oof"])
        test_predictions.append(current["test"])
        seed_auc.append(auc)
        seed_min_fold_auc.append(min(folds))
    assert reference is not None
    ensemble_oof = np.mean(np.vstack(predictions), axis=0)
    ensemble_test = np.mean(np.vstack(test_predictions), axis=0)
    return {
        "id": reference["id"],
        "target": reference["target"],
        "fold": reference["fold"],
        "test_id": reference["test_id"],
        "ensemble_test": ensemble_test,
        "seed_auc": seed_auc,
        "seed_min_fold_auc": seed_min_fold_auc,
        "ensemble_auc": float(roc_auc_score(reference["target"], ensemble_oof)),
    }


def aggregate(*, plan_path: Path, collected_root: Path, sample_path: Path, output_dir: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    collected_root = Path(collected_root).resolve()
    sample_path = Path(sample_path).resolve()
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "May Human Gate package is immutable and already exists")
    plan = _read_json(plan_path)
    seeds = [int(value) for value in plan.get("seeds") or []]
    run_ids = [str(value) for value in plan.get("run_ids") or []]
    require(seeds == [42, 43, 44] and len(run_ids) == 3, "May multiseed plan drifted")
    sample_record = plan.get("public_sample_submission") or {}
    require(
        sample_path.is_file()
        and sample_path.stat().st_size == int(sample_record.get("bytes", -1))
        and common.sha256_file(sample_path) == sample_record.get("sha256"),
        "May sample-submission identity drifted",
    )
    items = [_seed_artifacts(collected_root / run_id, seed) for seed, run_id in zip(seeds, run_ids, strict=True)]
    values = _load_and_verify(items)
    promotion = plan.get("promotion_contract") or {}
    ready = bool(
        min(values["seed_auc"]) >= float(promotion["confirmation_seed_auc_minimum"])
        and min(values["seed_min_fold_auc"]) >= float(promotion["every_fold_oof_auc_minimum"])
        and values["ensemble_auc"] >= float(promotion["aggregate_oof_auc_minimum"])
    )
    require(ready, "May multiseed confirmation gate did not pass")
    sample = pd.read_csv(sample_path)
    require(list(sample.columns) == ["id", "target"], "May sample schema drifted")
    require(np.array_equal(sample["id"].to_numpy(), values["test_id"]), "May sample/test IDs drifted")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_submission_withheld.csv"
    pd.DataFrame({"id": values["test_id"], "target": values["ensemble_test"]}).to_csv(candidate_path, index=False)
    frozen_plan_path = output_dir / "frozen_plan.json"
    shutil.copyfile(plan_path, frozen_plan_path)
    seed_records = [
        {
            "model_seed": item["seed"], "run_id": item["run_id"],
            "oof_auc": auc, "minimum_fold_auc": minimum,
            "result": common.file_record(item["result_path"]),
            "prediction_bundle": common.file_record(item["bundle_path"]),
            "submission": common.file_record(item["submission_path"]),
        }
        for item, auc, minimum in zip(items, values["seed_auc"], values["seed_min_fold_auc"], strict=True)
    ]
    candidate_record = common.file_record(candidate_path)
    result = {
        "schema": "evomind.mlebench.may2022_multiseed_confirmation.v1", "created_at": now_iso(),
        "competition_id": COMPETITION_ID, "status": "confirmation_passed_human_gate_pending",
        "candidate_ready_for_human_gate": True,
        "metrics": {"seed_oof_auc": values["seed_auc"], "minimum_seed_oof_auc": min(values["seed_auc"]), "ensemble_oof_auc": values["ensemble_auc"]},
        "confirmation_gate": {"passed": True, "metric": "roc_auc", "direction": "maximize", "thresholds": promotion},
        "seed_records": seed_records, "submission_withheld": candidate_record,
        "sample_submission": common.file_record(sample_path), "private_labels_used": False,
        "official_grader_executed": False, "kaggle_submission_executed": False,
        "process_signals_sent": 0, "human_gate_preserved": True,
        "claim_boundary": "Public OOF confirmation only; no official score or medal is claimed.",
    }
    result_path = output_dir / "may2022_multiseed_confirmation_result.json"
    common.write_json_atomic(result_path, result)
    independent = {
        "schema": "evomind.mlebench.may2022_multiseed_independent_verification.v1", "created_at": now_iso(),
        "status": "verification_passed", "ok": True, "candidate_ready_for_human_gate": True, "errors": [],
        "plan_sha256": common.sha256_file(frozen_plan_path), "result_sha256": common.sha256_file(result_path),
        "recomputed_seed_oof_auc": values["seed_auc"], "recomputed_ensemble_oof_auc": values["ensemble_auc"],
        "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False,
        "process_signals_sent": 0, "human_gate_preserved": True,
        "claim_boundary": "Independent public OOF verification is not an official score or medal.",
    }
    independent_path = output_dir / "independent_verification.json"
    common.write_json_atomic(independent_path, independent)
    readme = output_dir / "README.md"
    readme.write_text("# May-2022 multiseed Human Gate package\n\nCandidate-only; grading remains disabled.\n", encoding="utf-8")
    files = [common.file_record(path) for path in (candidate_path, frozen_plan_path, independent_path, result_path, readme)]
    manifest = {
        "schema": "evomind.human_gate.candidate_package.v1", "created_at": now_iso(),
        "status": "ready_for_human_review_not_submitted", "competition_id": COMPETITION_ID,
        "public_oof_metrics": result["metrics"], "candidate_ready_for_human_gate": True, "files": files,
        "automatic_submission": False, "private_labels_used": False, "official_grader_executed": False,
        "kaggle_submission_executed": False, "process_signals_sent": 0,
        "claim_boundary": "Ready for Human Gate review only; official medal count remains unchanged.",
    }
    manifest_path = output_dir / "manifest.json"
    common.write_json_atomic(manifest_path, manifest)
    common.write_json_atomic(output_dir / "package_verification.json", {
        "schema": "evomind.human_gate.package_verification.v1", "created_at": now_iso(), "status": "verified",
        "files": files + [common.file_record(manifest_path)], "candidate_csv_present": True,
        "independent_verification_passed": True, "automatic_submission": False,
    })
    return {"status": "ready_for_human_review_not_submitted", "package": str(output_dir), "metrics": result["metrics"], "candidate_sha256": candidate_record["sha256"], "official_grader_executed": False}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--collected-root", type=Path, default=DEFAULT_COLLECTED_ROOT)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(aggregate(plan_path=args.plan, collected_root=args.collected_root, sample_path=args.sample_submission, output_dir=args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
