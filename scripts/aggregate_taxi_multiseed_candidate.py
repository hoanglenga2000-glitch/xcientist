#!/usr/bin/env python3
"""Independently aggregate verified Taxi seed runs into a Human Gate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMPETITION_ID = "new-york-city-taxi-fare-prediction"
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "taxi_source_audited_multiseed_plan_current.json"
)
DEFAULT_COLLECTED_ROOT = (
    PROJECT_ROOT / "workspace" / "hpc" / "mlebench_remote_ops" / "collected"
)
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
    / "taxi_route_cache_multiseed_s43_s44_s45_20260728"
)


class TaxiAggregationError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxiAggregationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"Unsafe artifact: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unique_file(root: Path, name: str, *, prefer_top_level: bool = False) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and not path.is_symlink()]
    if prefer_top_level:
        top = [path for path in matches if "attempts" not in path.parts]
        if len(top) == 1:
            return top[0]
    require(len(matches) == 1, f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _validate_collection(run_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = run_root / "collection_manifest.json"
    require(manifest_path.is_file(), f"Collection manifest missing: {run_root.name}")
    manifest = _read_json(manifest_path)
    require(manifest.get("passed") is True, "Remote collection did not pass")
    require(manifest.get("run_id") == run_root.name, "Collection run ID mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files") or []:
        local = Path(str(item.get("local") or "")).resolve()
        try:
            local.relative_to(run_root.resolve())
        except ValueError as exc:
            raise TaxiAggregationError("Collection artifact escaped the run root") from exc
        require(local.is_file() and not local.is_symlink(), f"Collected artifact missing: {local}")
        require(
            local.stat().st_size == int(item.get("bytes", -1))
            and sha256_file(local) == item.get("sha256"),
            f"Collected artifact drifted: {local.name}",
        )
        indexed[str(local)] = dict(item)
    require(bool(indexed), "Collection manifest contains no files")
    return indexed


def _seed_artifacts(run_root: Path, seed: int) -> dict[str, Any]:
    indexed = _validate_collection(run_root)
    competition_root = run_root / COMPETITION_ID
    result_path = competition_root / "result.json"
    require(str(result_path.resolve()) in indexed, "Top-level Taxi result was not hash-collected")
    result = _read_json(result_path)
    require(result.get("competition_id") == COMPETITION_ID, "Taxi competition mismatch")
    require(result.get("metric") == "rmse" and result.get("direction") == "minimize", "Taxi metric drift")
    require(result.get("valid_submission") is True, "Taxi submission validation failed")
    require(result.get("official_grader_executed") is False, "Taxi run executed a grader")
    require((result.get("evidence_contract") or {}).get("private_labels_used") is False, "Taxi run used private labels")
    budget = result.get("budget") or {}
    require(int(budget.get("seed", -1)) == seed, "Taxi result seed mismatch")
    require(budget.get("precomputed_public_cache") is not None, "Taxi run rebuilt base features")
    require(budget.get("precomputed_route_stat_cache") is not None, "Taxi run rebuilt route statistics")
    require(budget.get("feature_build_on_gpu_run") is False, "Taxi feature rebuild occurred on GPU")
    require(budget.get("route_stat_build_on_gpu_run") is False, "Taxi route-stat rebuild occurred on GPU")
    gate = result.get("promotion_gate") or {}
    evidence = gate.get("evidence") or {}
    require(gate.get("passed") is True, f"Taxi seed {seed} promotion gate failed")
    oof_path = competition_root / "taxi_oof_manifest.csv"
    fold_records_path = competition_root / "taxi_oof_fold_records.json"
    require(str(oof_path.resolve()) in indexed, "Taxi OOF manifest was not hash-collected")
    require(str(fold_records_path.resolve()) in indexed, "Taxi fold records were not hash-collected")
    candidates = [
        path
        for path in competition_root.rglob("submission.csv")
        if path.is_file() and sha256_file(path) == result.get("submission_sha256")
    ]
    require(bool(candidates), "Taxi withheld submission hash was not collected")
    submission_path = sorted(candidates, key=lambda path: len(path.parts))[0]
    return {
        "seed": seed,
        "run_id": run_root.name,
        "result": result,
        "result_path": result_path,
        "oof_path": oof_path,
        "fold_records_path": fold_records_path,
        "submission_path": submission_path,
        "random_oof_rmse": float(result["cv_score"]),
        "temporal_stress_rmse": float(evidence["temporal_stress_rmse"]),
        "geographic_stress_rmse": float(evidence["geographic_stress_rmse"]),
    }


def recompute_oof_metrics(seed_items: list[dict[str, Any]], *, chunksize: int) -> dict[str, Any]:
    readers = [pd.read_csv(item["oof_path"], chunksize=chunksize) for item in seed_items]
    squared_error = np.zeros(len(seed_items), dtype=np.float64)
    ensemble_squared_error = 0.0
    rows = 0
    while True:
        chunks: list[pd.DataFrame] = []
        ended = []
        for reader in readers:
            try:
                chunks.append(next(reader))
                ended.append(False)
            except StopIteration:
                ended.append(True)
        if all(ended):
            break
        require(not any(ended), "Taxi OOF manifests have different row counts")
        reference = chunks[0]
        required = ["key", "source_row", "fare_amount", "duplicate_group", "fold", "oof_prediction"]
        require(list(reference.columns) == required, "Taxi OOF manifest schema drifted")
        target = reference["fare_amount"].to_numpy(dtype=np.float64)
        predictions = []
        for index, chunk in enumerate(chunks):
            require(list(chunk.columns) == required, "Taxi OOF manifest schema drifted")
            for column in ("key", "source_row", "fare_amount", "duplicate_group", "fold"):
                require(chunk[column].equals(reference[column]), f"Taxi OOF alignment drifted: {column}")
            prediction = chunk["oof_prediction"].to_numpy(dtype=np.float64)
            require(np.isfinite(prediction).all(), "Taxi OOF contains non-finite predictions")
            predictions.append(prediction)
            squared_error[index] += float(np.square(prediction - target).sum())
        ensemble = np.mean(np.vstack(predictions), axis=0)
        ensemble_squared_error += float(np.square(ensemble - target).sum())
        rows += len(reference)
    require(rows > 0, "Taxi OOF manifests are empty")
    return {
        "rows": rows,
        "seed_rmse": [math.sqrt(value / rows) for value in squared_error],
        "ensemble_rmse": math.sqrt(ensemble_squared_error / rows),
    }


def build_submission(seed_items: list[dict[str, Any]], sample_path: Path, output: Path) -> dict[str, Any]:
    sample = pd.read_csv(sample_path)
    require(list(sample.columns) == ["key", "fare_amount"], "Taxi sample schema drifted")
    predictions = []
    for item in seed_items:
        submission = pd.read_csv(item["submission_path"])
        require(list(submission.columns) == list(sample.columns), "Taxi submission schema drifted")
        require(submission["key"].astype(str).equals(sample["key"].astype(str)), "Taxi submission IDs drifted")
        values = submission["fare_amount"].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), "Taxi submission contains non-finite predictions")
        predictions.append(values)
    candidate = sample.copy()
    candidate["fare_amount"] = np.mean(np.vstack(predictions), axis=0)
    candidate.to_csv(output, index=False)
    return file_record(output)


def aggregate(
    *,
    plan_path: Path,
    collected_root: Path,
    sample_path: Path,
    output_dir: Path,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    collected_root = Path(collected_root).resolve()
    sample_path = Path(sample_path).resolve()
    output_dir = Path(output_dir).resolve()
    require(chunksize > 0, "Taxi aggregation chunksize must be positive")
    require(not output_dir.exists(), "Taxi Human Gate package is immutable and already exists")
    plan = _read_json(plan_path)
    require(plan.get("competition_id") == COMPETITION_ID, "Taxi plan competition mismatch")
    seeds = [int(value) for value in plan.get("seeds") or []]
    run_ids = [str(value) for value in plan.get("run_ids") or []]
    require(seeds == [43, 44, 45] and len(run_ids) == 3, "Taxi multiseed contract drifted")
    sample_record = plan.get("public_sample_submission") or {}
    require(
        sample_path.is_file()
        and sample_path.stat().st_size == int(sample_record.get("bytes", -1))
        and sha256_file(sample_path) == sample_record.get("sha256"),
        "Taxi public sample-submission identity drifted",
    )
    seed_items = [
        _seed_artifacts(collected_root / run_id, seed)
        for seed, run_id in zip(seeds, run_ids, strict=True)
    ]
    metrics = recompute_oof_metrics(seed_items, chunksize=chunksize)
    for item, recomputed in zip(seed_items, metrics["seed_rmse"], strict=True):
        require(abs(recomputed - item["random_oof_rmse"]) <= 1e-9, "Taxi result/OOF RMSE drifted")
    promotion = plan.get("promotion_contract") or {}
    candidate_ready = bool(
        max(metrics["seed_rmse"]) <= float(promotion["every_seed_random_oof_rmse_maximum"])
        and max(item["temporal_stress_rmse"] for item in seed_items)
        <= float(promotion["every_seed_temporal_stress_rmse_maximum"])
        and max(item["geographic_stress_rmse"] for item in seed_items)
        <= float(promotion["every_seed_geographic_stress_rmse_maximum"])
        and float(np.mean(metrics["seed_rmse"]))
        <= float(promotion["mean_random_oof_rmse_maximum"])
        and metrics["ensemble_rmse"] <= float(promotion["mean_random_oof_rmse_maximum"])
    )
    require(candidate_ready, "Taxi multiseed confirmation gate did not pass")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_submission_withheld.csv"
    candidate_record = build_submission(seed_items, sample_path, candidate_path)
    frozen_plan_path = output_dir / "frozen_plan.json"
    shutil.copyfile(plan_path, frozen_plan_path)
    seed_records = [
        {
            "model_seed": item["seed"],
            "run_id": item["run_id"],
            "random_oof_rmse": item["random_oof_rmse"],
            "recomputed_random_oof_rmse": recomputed,
            "temporal_stress_rmse": item["temporal_stress_rmse"],
            "geographic_stress_rmse": item["geographic_stress_rmse"],
            "result": file_record(item["result_path"]),
            "oof_manifest": file_record(item["oof_path"]),
            "fold_records": file_record(item["fold_records_path"]),
            "submission": file_record(item["submission_path"]),
        }
        for item, recomputed in zip(seed_items, metrics["seed_rmse"], strict=True)
    ]
    result = {
        "schema": "evomind.mlebench.taxi_multiseed_confirmation.v1",
        "created_at": now_iso(),
        "competition_id": COMPETITION_ID,
        "status": "confirmation_passed_human_gate_pending",
        "candidate_ready_for_human_gate": True,
        "metrics": {
            "seed_oof_rmse": metrics["seed_rmse"],
            "mean_seed_oof_rmse": float(np.mean(metrics["seed_rmse"])),
            "maximum_seed_oof_rmse": max(metrics["seed_rmse"]),
            "ensemble_oof_rmse": metrics["ensemble_rmse"],
        },
        "confirmation_gate": {
            "passed": True,
            "metric": "rmse",
            "direction": "minimize",
            "thresholds": promotion,
        },
        "seed_records": seed_records,
        "submission_withheld": candidate_record,
        "sample_submission": file_record(sample_path),
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Public OOF confirmation only; no official score or medal is claimed.",
    }
    result_path = output_dir / "taxi_multiseed_confirmation_result.json"
    write_json_atomic(result_path, result)
    independent = {
        "schema": "evomind.mlebench.taxi_multiseed_independent_verification.v1",
        "created_at": now_iso(),
        "status": "verification_passed",
        "ok": True,
        "candidate_ready_for_human_gate": True,
        "errors": [],
        "plan_sha256": sha256_file(frozen_plan_path),
        "result_sha256": sha256_file(result_path),
        "recomputed_seed_oof_rmse": metrics["seed_rmse"],
        "recomputed_ensemble_oof_rmse": metrics["ensemble_rmse"],
        "oof_rows": metrics["rows"],
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "human_gate_preserved": True,
        "claim_boundary": "Independent public OOF verification is not an official score or medal.",
    }
    independent_path = output_dir / "independent_verification.json"
    write_json_atomic(independent_path, independent)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Taxi multiseed Human Gate package\n\nCandidate-only; official grader and Kaggle submission remain disabled.\n",
        encoding="utf-8",
    )
    package_files = [
        file_record(path)
        for path in (
            candidate_path,
            frozen_plan_path,
            independent_path,
            result_path,
            readme_path,
        )
    ]
    manifest = {
        "schema": "evomind.human_gate.candidate_package.v1",
        "created_at": now_iso(),
        "status": "ready_for_human_review_not_submitted",
        "competition_id": COMPETITION_ID,
        "public_oof_metrics": result["metrics"],
        "candidate_ready_for_human_gate": True,
        "files": package_files,
        "automatic_submission": False,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": "Ready for Human Gate review only; official medal count remains unchanged.",
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    verification = {
        "schema": "evomind.human_gate.package_verification.v1",
        "created_at": now_iso(),
        "status": "verified",
        "files": package_files + [file_record(manifest_path)],
        "candidate_csv_present": True,
        "independent_verification_passed": True,
        "automatic_submission": False,
    }
    write_json_atomic(output_dir / "package_verification.json", verification)
    return {
        "status": "ready_for_human_review_not_submitted",
        "package": str(output_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_sha256": candidate_record["sha256"],
        "metrics": result["metrics"],
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--collected-root", type=Path, default=DEFAULT_COLLECTED_ROOT)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = aggregate(
        plan_path=args.plan,
        collected_root=args.collected_root,
        sample_path=args.sample_submission,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
