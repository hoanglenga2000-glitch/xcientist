#!/usr/bin/env python3
"""Independently verify a completed Leaf multi-seed, dual-backbone OOF run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "leaf_multibackbone_frozen_plan.json"
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
        raise TypeError(f"Expected JSON object: {path}")
    return payload


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


def normalize_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    require(probability.ndim == 2, "Leaf probability is not a matrix")
    require(np.isfinite(probability).all(), "Leaf probability contains non-finite values")
    require((probability >= 0.0).all(), "Leaf probability contains negative values")
    row_sum = probability.sum(axis=1, keepdims=True)
    require((row_sum > 0.0).all(), "Leaf probability contains empty rows")
    return probability / row_sum


def apply_blend(
    components: Sequence[np.ndarray], weights: Sequence[float], temperature: float
) -> np.ndarray:
    matrices = [normalize_probability(value) for value in components]
    weight_array = np.asarray(weights, dtype=np.float64)
    require(bool(matrices), "Leaf independent blend has no components")
    require(len(matrices) == len(weight_array), "Leaf blend component count differs")
    require(np.isfinite(weight_array).all(), "Leaf blend weights are non-finite")
    require((weight_array >= 0.0).all(), "Leaf blend weights are negative")
    require(float(weight_array.sum()) > 0.0, "Leaf blend weights sum to zero")
    require(float(temperature) > 0.0, "Leaf blend temperature is invalid")
    weight_array /= weight_array.sum()
    pooled = np.zeros_like(matrices[0], dtype=np.float64)
    for weight, matrix in zip(weight_array, matrices, strict=True):
        require(matrix.shape == pooled.shape, "Leaf blend matrices differ in shape")
        pooled += float(weight) * np.clip(matrix, 1e-12, 1.0)
    logits = np.log(np.clip(pooled, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


def reconstruct_crossfit(
    components: dict[str, np.ndarray],
    folds: np.ndarray,
    records: Sequence[dict[str, Any]],
) -> np.ndarray:
    names = tuple(components)
    matrices = [normalize_probability(components[name]) for name in names]
    assignment = np.asarray(folds, dtype=np.int16).reshape(-1)
    require(
        sorted(np.unique(assignment).tolist()) == list(range(len(records))),
        "Leaf fold assignment is not contiguous",
    )
    rebuilt = np.full_like(matrices[0], np.nan, dtype=np.float64)
    write_counts = np.zeros(len(assignment), dtype=np.uint8)
    for record in records:
        fold = int(record["fold"])
        validation = assignment == fold
        rebuilt[validation] = apply_blend(
            [matrix[validation] for matrix in matrices],
            record["weights"],
            float(record["temperature"]),
        )
        write_counts[validation] += 1
    require(np.all(write_counts == 1), "Leaf independent OOF was not written exactly once")
    require(np.isfinite(rebuilt).all(), "Leaf independent OOF is incomplete")
    return rebuilt


def verify_artifact_manifest(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    records = manifest.get("artifacts") or []
    require(
        manifest.get("schema") == "evomind.leaf.multibackbone_artifacts.v1",
        "Leaf artifact manifest schema differs",
    )
    require(
        manifest.get("artifact_count") == len(records) and bool(records),
        "Leaf artifact manifest count differs",
    )
    seen: set[str] = set()
    verified = []
    for record in records:
        relative = str(record.get("path", ""))
        require(relative and relative not in seen, "Leaf artifact path is empty or duplicated")
        seen.add(relative)
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("Leaf artifact escapes the run directory") from exc
        require(path.is_file(), f"Leaf artifact is missing: {relative}")
        require(path.stat().st_size == record.get("bytes"), f"Leaf size differs: {relative}")
        digest = sha256_file(path)
        require(digest == record.get("sha256"), f"Leaf hash differs: {relative}")
        verified.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(verified),
    }


def verify_seed_bundle(
    path: Path,
    seed_record: dict[str, Any],
    *,
    train_id: np.ndarray,
    target: np.ndarray,
    test_id: np.ndarray,
    classes: Sequence[str],
    expected_folds: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    require(path.is_file(), f"Leaf seed artifact is missing: {path.name}")
    require(sha256_file(path) == seed_record.get("artifact_sha256"), "Leaf seed hash differs")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "train_id",
            "target",
            "classes",
            "fold_assignment",
            "component_names",
            "crossfit_probability",
            "test_id",
            "test_probability",
        }
        require(required.issubset(archive.files), "Leaf seed bundle misses required arrays")
        bundle_train_id = np.asarray(archive["train_id"])
        bundle_target = np.asarray(archive["target"]).astype(str)
        bundle_classes = np.asarray(archive["classes"]).astype(str).tolist()
        folds = np.asarray(archive["fold_assignment"], dtype=np.int16)
        component_names = np.asarray(archive["component_names"]).astype(str).tolist()
        reported_oof = normalize_probability(archive["crossfit_probability"])
        bundle_test_id = np.asarray(archive["test_id"])
        test_probability = normalize_probability(archive["test_probability"])
        components = {
            name: np.asarray(archive[f"oof__{name}"], dtype=np.float64)
            for name in component_names
        }
    require(np.array_equal(bundle_train_id, train_id), "Leaf seed train IDs differ")
    require(np.array_equal(bundle_target, target.astype(str)), "Leaf seed targets differ")
    require(bundle_classes == [str(value) for value in classes], "Leaf seed classes differ")
    require(np.array_equal(bundle_test_id, test_id), "Leaf seed test IDs differ")
    require(
        sorted(np.unique(folds).tolist()) == list(range(expected_folds)),
        "Leaf seed fold coverage differs",
    )
    require(len(reported_oof) == len(train_id), "Leaf seed OOF row count differs")
    require(len(test_probability) == len(test_id), "Leaf seed test row count differs")
    records = seed_record.get("crossfit_blend_records") or []
    require(len(records) == expected_folds, "Leaf cross-fit blend record count differs")
    rebuilt = reconstruct_crossfit(components, folds, records)
    require(
        np.allclose(rebuilt, reported_oof, rtol=0.0, atol=1e-12),
        "Leaf reported cross-fit probabilities differ from reconstruction",
    )
    score = float(log_loss(target.astype(str), rebuilt, labels=list(classes)))
    require(
        math.isclose(
            score,
            float(seed_record["cross_fitted_log_loss"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Leaf seed log loss differs from independent recomputation",
    )
    return {
        "seed": int(seed_record["seed"]),
        "recomputed_cross_fitted_log_loss": score,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path),
        "folds": sorted(np.unique(folds).tolist()),
    }, test_probability, folds


def verify_run(run_dir: Path, plan_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    plan_path = plan_path.resolve()
    plan = read_json(plan_path)
    require(
        plan.get("schema") == "evomind.leaf.multibackbone_frozen_plan.v1",
        "Leaf frozen plan schema differs",
    )
    require(plan.get("status") == "frozen_waiting_after_siim", "Leaf plan is not frozen")
    plan_sha256 = sha256_file(plan_path)
    runner_path = Path(plan["implementation"]["runner"]["path"])
    require(
        sha256_file(runner_path) == plan["implementation"]["runner"]["sha256"],
        "Leaf runner hash differs from the frozen plan",
    )

    report_path = run_dir / "leaf_multibackbone_oof.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    image_manifest_path = run_dir / "leaf_image_manifest.csv"
    candidate_path = run_dir / "submission_withheld.csv"
    for path in (report_path, artifact_manifest_path, image_manifest_path, candidate_path):
        require(path.is_file(), f"Leaf required artifact is missing: {path.name}")
    report = read_json(report_path)
    training = plan["training"]
    require(report.get("run_id") == training["run_id"], "Leaf run ID differs")
    require(report.get("competition_id") == plan["competition_id"], "Leaf competition differs")
    require(report.get("external_seeds") == training["seeds"], "Leaf seeds differ")
    require(report.get("folds") == training["folds"], "Leaf folds differ")
    require(report.get("backbones") == training["backbones"], "Leaf backbones differ")
    require(report.get("private_labels_used") is False, "Leaf used private labels")
    require(report.get("private_scores_used_for_tuning") is False, "Leaf used private scores")
    require(report.get("official_grader_executed") is False, "Leaf ran official grader")
    require(report.get("kaggle_submission_executed") is False, "Leaf submitted to Kaggle")
    require(report.get("official_score_claimed") is False, "Leaf claimed an official score")

    public_dir = (
        Path(training["data_root"]) / plan["competition_id"] / "prepared" / "public"
    )
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    require(len(train) == report["train_rows"], "Leaf train row count differs")
    require(len(test) == report["test_rows"], "Leaf test row count differs")
    classes = sample.columns[1:].astype(str).tolist()
    require(sorted(train["species"].astype(str).unique().tolist()) == sorted(classes), "Leaf classes differ")

    manifest = pd.read_csv(image_manifest_path)
    require(len(manifest) == len(train) + len(test), "Leaf image manifest row count differs")
    require(sha256_file(image_manifest_path) == report["manifest_sha256"], "Leaf image manifest hash differs")
    require(manifest.iloc[: len(train)]["id"].to_numpy().tolist() == train["id"].to_numpy().tolist(), "Leaf manifest train IDs differ")

    seed_results = []
    seed_test = []
    fold_assignments = []
    for seed_record in report.get("per_seed") or []:
        artifact = Path(seed_record["artifact"])
        result, probability, folds = verify_seed_bundle(
            artifact,
            seed_record,
            train_id=train["id"].to_numpy(),
            target=train["species"].astype(str).to_numpy(),
            test_id=test["id"].to_numpy(),
            classes=classes,
            expected_folds=int(training["folds"]),
        )
        groups = manifest.iloc[: len(train)]["sha256"].astype(str).to_numpy()
        frame = pd.DataFrame({"group": groups, "fold": folds})
        require(int(frame.groupby("group")["fold"].nunique().max()) == 1, "Leaf image group crossed a fold")
        seed_results.append(result)
        seed_test.append(probability)
        fold_assignments.append(folds)
    require([item["seed"] for item in seed_results] == training["seeds"], "Leaf verified seed order differs")

    scores = [item["recomputed_cross_fitted_log_loss"] for item in seed_results]
    mean_score = float(np.mean(scores))
    maximum_score = float(np.max(scores))
    gate_passed = (
        mean_score <= float(plan["objective"]["promotion_mean_log_loss"])
        and maximum_score <= float(plan["objective"]["promotion_max_seed_log_loss"])
    )
    require(math.isclose(mean_score, float(report["mean_cross_fitted_log_loss"]), abs_tol=1e-12), "Leaf mean score differs")
    require(math.isclose(maximum_score, float(report["maximum_seed_cross_fitted_log_loss"]), abs_tol=1e-12), "Leaf max score differs")
    require(report["promotion_gate"]["passed"] is gate_passed, "Leaf gate differs")
    require(report["status"] == ("promotion_gate_passed" if gate_passed else "promotion_gate_failed"), "Leaf terminal status differs")

    rebuilt_test = normalize_probability(np.mean(seed_test, axis=0))
    candidate = pd.read_csv(candidate_path)
    require(candidate.columns.tolist() == sample.columns.tolist(), "Leaf candidate schema differs")
    require(candidate.iloc[:, 0].tolist() == sample.iloc[:, 0].tolist(), "Leaf candidate ID order differs")
    require(
        np.allclose(candidate[classes].to_numpy(dtype=np.float64), rebuilt_test, rtol=0.0, atol=1e-12),
        "Leaf candidate differs from the multi-seed mean",
    )
    require(sha256_file(candidate_path) == report["submission_sha256"], "Leaf candidate hash differs")
    artifacts = verify_artifact_manifest(run_dir, artifact_manifest_path)
    return {
        "schema": "evomind.leaf.multibackbone_independent_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed",
        "ok": True,
        "candidate_ready": gate_passed,
        "run_dir": str(run_dir),
        "run_id": training["run_id"],
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "runner_sha256": sha256_file(runner_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "seed_results": seed_results,
        "mean_cross_fitted_log_loss": mean_score,
        "maximum_seed_cross_fitted_log_loss": maximum_score,
        "promotion_gate_passed": gate_passed,
        "artifact_manifest": artifacts,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
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
            "schema": "evomind.leaf.multibackbone_independent_verification.v1",
            "created_at": now_iso(),
            "status": "verification_failed",
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        }
    write_json_atomic(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
