#!/usr/bin/env python3
"""Aggregate Dog Breed frozen-head seeds into an immutable Human Gate package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402
from scripts import queue_hpc88240_dog_frozen_head_after_taxi as queue  # noqa: E402

COMPETITION_ID = queue.COMPETITION_ID
DEFAULT_PLAN = queue.DEFAULT_PLAN
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
    PROJECT_ROOT
    / "workspace"
    / "human_gate"
    / "dog_breed_frozen_head_multiseed_s46_s47_20260728"
)


class DogAggregationError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DogAggregationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def multiclass_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(truth, dtype=np.int64).reshape(-1)
    values = np.asarray(probability, dtype=np.float64)
    require(values.ndim == 2 and len(labels) == len(values), "Dog probability shape drifted")
    require(np.isfinite(values).all(), "Dog probabilities contain non-finite values")
    require(np.logical_and(values >= 0.0, values <= 1.0).all(), "Dog probabilities escaped [0, 1]")
    row_sum = values.sum(axis=1)
    require(np.allclose(row_sum, 1.0, rtol=0.0, atol=1e-6), "Dog probability rows do not sum to one")
    require(
        np.logical_and(labels >= 0, labels < values.shape[1]).all(),
        "Dog labels escaped the class range",
    )
    clipped = np.clip(values[np.arange(len(labels)), labels], 1e-15, 1.0)
    return float(-np.mean(np.log(clipped)))


def _seed_artifacts(run_root: Path, seed: int) -> dict[str, Any]:
    indexed = common._validate_collection(run_root)
    competition = run_root / COMPETITION_ID
    result_path = competition / "result.json"
    require(str(result_path.resolve()) in indexed, "Dog top-level result was not hash-collected")
    result = _read_json(result_path)
    require(result.get("competition_id") == COMPETITION_ID, "Dog competition mismatch")
    require(
        result.get("metric") == "multiclass_log_loss"
        and result.get("direction") == "minimize",
        "Dog metric drifted",
    )
    require(result.get("valid_submission") is True, "Dog submission validation failed")
    require(result.get("official_grader_executed") is False, "Dog run executed a grader")
    require(result.get("private_labels_used") is False, "Dog run used private labels")
    require(int((result.get("budget") or {}).get("seed", -1)) == seed, "Dog result seed drifted")
    gate = result.get("promotion_gate") or {}
    require(gate.get("passed") is True, f"Dog seed {seed} promotion gate failed")
    bundles = [
        path for path in competition.rglob("vision_oof_and_test.npz") if path.is_file()
    ]
    require(len(bundles) == 1, "Dog prediction bundle inventory drifted")
    bundle_path = bundles[0]
    require(str(bundle_path.resolve()) in indexed, "Dog prediction bundle was not hash-collected")
    submissions = [
        path
        for path in competition.rglob("submission.csv")
        if path.is_file() and common.sha256_file(path) == result.get("submission_sha256")
    ]
    require(bool(submissions), "Dog withheld submission was not hash-collected")
    return {
        "seed": seed,
        "run_id": run_root.name,
        "result": result,
        "result_path": result_path,
        "bundle_path": bundle_path,
        "submission_path": sorted(submissions, key=lambda path: len(path.parts))[0],
    }


def _load_bundle(item: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(item["bundle_path"], allow_pickle=False) as bundle:
        required = {
            "train_id",
            "test_id",
            "class_names",
            "truth",
            "selected_oof_probability",
            "selected_test_probability",
            "fold",
        }
        require(required <= set(bundle.files), "Dog prediction bundle lacks immutable alignment metadata")
        current = {
            "train_id": np.asarray(bundle["train_id"], dtype=np.str_),
            "test_id": np.asarray(bundle["test_id"], dtype=np.str_),
            "class_names": np.asarray(bundle["class_names"], dtype=np.str_),
            "truth": np.asarray(bundle["truth"], dtype=np.int64),
            "oof": np.asarray(bundle["selected_oof_probability"], dtype=np.float64),
            "test": np.asarray(bundle["selected_test_probability"], dtype=np.float64),
            "fold": np.asarray(bundle["fold"], dtype=np.int16),
        }
    require(current["train_id"].shape == (9199,), "Dog train ID count drifted")
    require(current["test_id"].shape == (1023,), "Dog test ID count drifted")
    require(current["class_names"].shape == (120,), "Dog class count drifted")
    require(current["truth"].shape == (9199,), "Dog truth count drifted")
    require(current["oof"].shape == (9199, 120), "Dog OOF matrix shape drifted")
    require(current["test"].shape == (1023, 120), "Dog test matrix shape drifted")
    require(current["fold"].shape == (9199,), "Dog fold vector shape drifted")
    require(len(np.unique(current["train_id"])) == 9199, "Dog train IDs are not unique")
    require(len(np.unique(current["test_id"])) == 1023, "Dog test IDs are not unique")
    require(len(np.unique(current["class_names"])) == 120, "Dog class names are not unique")
    require(set(np.unique(current["fold"]).tolist()) == {0, 1, 2, 3, 4}, "Dog fold coverage drifted")
    multiclass_log_loss(current["truth"], current["oof"])
    require(
        np.isfinite(current["test"]).all()
        and np.logical_and(current["test"] >= 0.0, current["test"] <= 1.0).all()
        and np.allclose(current["test"].sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
        "Dog test probability contract failed",
    )
    return current


def _load_and_verify(seed_items: list[dict[str, Any]]) -> dict[str, Any]:
    reference: dict[str, np.ndarray] | None = None
    oof_probabilities: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []
    seed_log_loss: list[float] = []
    for item in seed_items:
        current = _load_bundle(item)
        if reference is None:
            reference = current
        else:
            for key in ("train_id", "test_id", "class_names", "truth"):
                require(np.array_equal(current[key], reference[key]), f"Dog multiseed alignment drifted: {key}")
        score = multiclass_log_loss(current["truth"], current["oof"])
        require(
            abs(score - float(item["result"]["cv_score"])) <= 1e-9,
            "Dog result/OOF log loss drifted",
        )
        oof_probabilities.append(current["oof"])
        test_probabilities.append(current["test"])
        seed_log_loss.append(score)
    require(reference is not None, "Dog seed inventory is empty")
    ensemble_oof = np.mean(np.stack(oof_probabilities, axis=0), axis=0)
    ensemble_test = np.mean(np.stack(test_probabilities, axis=0), axis=0)
    ensemble_test /= np.maximum(ensemble_test.sum(axis=1, keepdims=True), 1e-15)
    return {
        "train_id": reference["train_id"],
        "test_id": reference["test_id"],
        "class_names": reference["class_names"],
        "truth": reference["truth"],
        "ensemble_test": ensemble_test,
        "seed_log_loss": seed_log_loss,
        "ensemble_log_loss": multiclass_log_loss(reference["truth"], ensemble_oof),
    }


def aggregate(
    *, plan_path: Path, collected_root: Path, sample_path: Path, output_dir: Path
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    collected_root = Path(collected_root).resolve()
    sample_path = Path(sample_path).resolve()
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Dog Human Gate package is immutable and already exists")
    plan = queue.validate_plan(plan_path)
    confirmation = plan["confirmation"]
    seeds = [int(value) for value in confirmation["seeds"]]
    run_ids = [str(value) for value in confirmation["run_ids"]]
    require(seeds == [46, 47] and len(run_ids) == 2, "Dog multiseed plan drifted")
    sample_record = plan["public_sample_submission"]
    require(
        sample_path.is_file()
        and sample_path.stat().st_size == int(sample_record["bytes"])
        and common.sha256_file(sample_path) == sample_record["sha256"],
        "Dog sample-submission identity drifted",
    )
    items = [
        _seed_artifacts(collected_root / run_id, seed)
        for seed, run_id in zip(seeds, run_ids, strict=True)
    ]
    values = _load_and_verify(items)
    maximum_seed = float(confirmation["every_seed_oof_log_loss_maximum"])
    maximum_ensemble = float(confirmation["aggregate_oof_log_loss_maximum"])
    require(max(values["seed_log_loss"]) <= maximum_seed, "Dog worst-seed confirmation gate failed")
    require(values["ensemble_log_loss"] <= maximum_ensemble, "Dog ensemble confirmation gate failed")

    sample = pd.read_csv(sample_path)
    require(len(sample) == 1023 and len(sample.columns) == 121, "Dog sample dimensions drifted")
    require(str(sample.columns[0]) == "id", "Dog sample ID column drifted")
    require(
        np.array_equal(sample["id"].astype(str).to_numpy(), values["test_id"]),
        "Dog sample/test ID order drifted",
    )
    require(
        np.array_equal(np.asarray(sample.columns[1:], dtype=np.str_), values["class_names"]),
        "Dog class order drifted",
    )

    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_submission_withheld.csv"
    candidate = pd.DataFrame(values["ensemble_test"], columns=values["class_names"])
    candidate.insert(0, "id", values["test_id"])
    candidate.to_csv(candidate_path, index=False)
    frozen_plan_path = output_dir / "frozen_plan.json"
    shutil.copyfile(plan_path, frozen_plan_path)
    seed_records = [
        {
            "model_seed": item["seed"],
            "run_id": item["run_id"],
            "oof_log_loss": score,
            "result": common.file_record(item["result_path"]),
            "prediction_bundle": common.file_record(item["bundle_path"]),
            "submission": common.file_record(item["submission_path"]),
        }
        for item, score in zip(items, values["seed_log_loss"], strict=True)
    ]
    candidate_record = common.file_record(candidate_path)
    result = {
        "schema": "evomind.mlebench.dog_breed_frozen_head_multiseed_confirmation.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "status": "confirmation_passed_human_gate_pending",
        "candidate_ready_for_human_gate": True,
        "metrics": {
            "seed_oof_log_loss": values["seed_log_loss"],
            "maximum_seed_oof_log_loss": max(values["seed_log_loss"]),
            "ensemble_oof_log_loss": values["ensemble_log_loss"],
        },
        "confirmation_gate": {
            "passed": True,
            "metric": "multiclass_log_loss",
            "direction": "minimize",
            "thresholds": {
                "every_seed_oof_log_loss_maximum": maximum_seed,
                "aggregate_oof_log_loss_maximum": maximum_ensemble,
            },
        },
        "seed_records": seed_records,
        "submission_withheld": candidate_record,
        "sample_submission": common.file_record(sample_path),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Public OOF confirmation only; no official score or medal is claimed.",
    }
    result_path = output_dir / "dog_breed_multiseed_confirmation_result.json"
    common.write_json_atomic(result_path, result)
    independent = {
        "schema": "evomind.mlebench.dog_breed_multiseed_independent_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed",
        "ok": True,
        "candidate_ready_for_human_gate": True,
        "errors": [],
        "plan_sha256": common.sha256_file(frozen_plan_path),
        "result_sha256": common.sha256_file(result_path),
        "recomputed_seed_oof_log_loss": values["seed_log_loss"],
        "recomputed_ensemble_oof_log_loss": values["ensemble_log_loss"],
        "class_order_verified": True,
        "train_id_order_verified": True,
        "test_id_order_verified": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Independent public OOF verification is not an official score or medal.",
    }
    independent_path = output_dir / "independent_verification.json"
    common.write_json_atomic(independent_path, independent)
    readme = output_dir / "README.md"
    readme.write_text(
        "# Dog Breed frozen-head multiseed Human Gate package\n\n"
        "Candidate-only; official grading and Kaggle submission remain disabled.\n",
        encoding="utf-8",
    )
    files = [
        common.file_record(path)
        for path in (candidate_path, frozen_plan_path, independent_path, result_path, readme)
    ]
    manifest = {
        "schema": "evomind.human_gate.candidate_package.v1",
        "created_at": now_iso(),
        "status": "ready_for_human_review_not_submitted",
        "competition_id": COMPETITION_ID,
        "public_oof_metrics": result["metrics"],
        "candidate_ready_for_human_gate": True,
        "files": files,
        "automatic_submission": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Ready for Human Gate review only; official medal count remains unchanged.",
    }
    manifest_path = output_dir / "manifest.json"
    common.write_json_atomic(manifest_path, manifest)
    common.write_json_atomic(
        output_dir / "package_verification.json",
        {
            "schema": "evomind.human_gate.package_verification.v1",
            "created_at": now_iso(),
            "status": "verified",
            "files": files + [common.file_record(manifest_path)],
            "candidate_csv_present": True,
            "independent_verification_passed": True,
            "automatic_submission": False,
        },
    )
    return {
        "status": "ready_for_human_review_not_submitted",
        "package": str(output_dir),
        "metrics": result["metrics"],
        "candidate_sha256": candidate_record["sha256"],
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--collected-root", type=Path, default=DEFAULT_COLLECTED_ROOT)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
