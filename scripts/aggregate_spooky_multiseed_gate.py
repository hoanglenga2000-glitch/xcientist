#!/usr/bin/env python3
"""Build the withheld Spooky three-seed promotion gate from verified OOF runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASS_COLUMNS = ("EAP", "HPL", "MWS")
DEFAULT_REQUIRED_SEEDS = (40, 41, 42)
DEFAULT_MEAN_THRESHOLD = 0.285
DEFAULT_MAX_SEED_THRESHOLD = 0.315


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def normalized_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(CLASS_COLUMNS):
        raise ValueError("Spooky multi-seed probability shape is invalid")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("Spooky multi-seed probability values are invalid")
    row_sum = probability.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("Spooky multi-seed probability row has no mass")
    return probability / row_sum


def multiclass_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(
        log_loss(
            np.asarray(truth, dtype=np.int64),
            normalized_probability(probability),
            labels=np.arange(len(CLASS_COLUMNS)),
        )
    )


@dataclass(frozen=True)
class SeedRun:
    seed: int
    run_dir: Path
    run_id: str
    score: float
    truth: np.ndarray
    oof: np.ndarray
    test: np.ndarray
    train_id: np.ndarray
    test_id: np.ndarray
    summary_path: Path
    verification_path: Path
    bundle_path: Path
    submission_path: Path
    checks: dict[str, bool]


def _all_true(values: Any) -> bool:
    return isinstance(values, dict) and bool(values) and all(value is True for value in values.values())


def load_seed_run(seed: int, run_dir: Path) -> SeedRun:
    resolved = Path(run_dir).resolve()
    summary_path = resolved / "summary.json"
    verification_path = resolved / "independent_verification.json"
    bundle_path = resolved / "spooky_transformer_oof_and_test.npz"
    submission_path = resolved / "candidate_submission_withheld.csv"
    for path in (summary_path, verification_path, bundle_path, submission_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    verification = json.loads(verification_path.read_text(encoding="utf-8-sig"))
    with np.load(bundle_path, allow_pickle=False) as archive:
        required = {"truth", "candidate_oof", "candidate_test", "train_id", "test_id"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"Spooky seed {seed} bundle is missing arrays: {missing}")
        truth = np.asarray(archive["truth"], dtype=np.int64).reshape(-1)
        oof = normalized_probability(archive["candidate_oof"])
        test = normalized_probability(archive["candidate_test"])
        train_id = np.asarray(archive["train_id"]).astype(str).reshape(-1)
        test_id = np.asarray(archive["test_id"]).astype(str).reshape(-1)

    if len(truth) != len(oof) or len(train_id) != len(oof) or len(test_id) != len(test):
        raise ValueError(f"Spooky seed {seed} bundle row counts differ")
    score = multiclass_log_loss(truth, oof)
    submission = pd.read_csv(submission_path)
    submission_columns = list(submission.columns)
    expected_columns = ["id", *CLASS_COLUMNS]
    reported_seed = summary.get("promotion_gate", {}).get("seed")
    summary_score = summary.get("candidate_oof_log_loss")
    verification_score = verification.get("recomputed_candidate_oof_log_loss")
    checks = {
        "requested_seed_matches_summary": reported_seed == seed,
        "terminal_single_seed_status": summary.get("status")
        == "single_seed_gate_passed_confirmation_pending",
        "single_seed_gate_passed": summary.get("promotion_gate", {}).get("single_seed_passed")
        is True,
        "summary_score_recomputed": isinstance(summary_score, (int, float))
        and math.isclose(score, float(summary_score), rel_tol=0.0, abs_tol=1e-12),
        "independent_verification_passed": verification.get("status") == "passed",
        "independent_contract_checks_passed": _all_true(verification.get("contract_checks")),
        "independent_score_recomputed": isinstance(verification_score, (int, float))
        and math.isclose(score, float(verification_score), rel_tol=0.0, abs_tol=1e-12),
        "verification_summary_hash": verification.get("summary", {}).get("sha256")
        == sha256_file(summary_path),
        "verification_bundle_hash": verification.get("prediction_bundle", {}).get("sha256")
        == sha256_file(bundle_path),
        "verification_submission_hash": verification.get("submission_withheld", {}).get("sha256")
        == sha256_file(submission_path),
        "summary_bundle_hash": summary.get("prediction_bundle", {}).get("sha256")
        == sha256_file(bundle_path),
        "summary_submission_hash": summary.get("submission_withheld", {}).get("sha256")
        == sha256_file(submission_path),
        "submission_columns": submission_columns == expected_columns,
        "submission_ids": "id" in submission
        and submission["id"].astype(str).tolist() == test_id.tolist(),
        "submission_probability": set(CLASS_COLUMNS).issubset(submission)
        and np.allclose(
            submission[list(CLASS_COLUMNS)].to_numpy(dtype=np.float64),
            test,
            rtol=0.0,
            atol=1e-12,
        ),
        "private_labels_unused": summary.get("private_labels_used") is False
        and verification.get("private_labels_used") is False,
        "official_grader_not_executed": summary.get("official_grader_executed") is False
        and verification.get("official_grader_executed") is False,
        "kaggle_submission_not_executed": summary.get("kaggle_submission_executed") is False
        and verification.get("kaggle_submission_executed") is False,
    }
    return SeedRun(
        seed=seed,
        run_dir=resolved,
        run_id=str(summary.get("run_id", "")),
        score=score,
        truth=truth,
        oof=oof,
        test=test,
        train_id=train_id,
        test_id=test_id,
        summary_path=summary_path,
        verification_path=verification_path,
        bundle_path=bundle_path,
        submission_path=submission_path,
        checks=checks,
    )


def aggregate_runs(
    seed_runs: dict[int, Path],
    output_dir: Path,
    *,
    required_seeds: Sequence[int] = DEFAULT_REQUIRED_SEEDS,
    mean_threshold: float = DEFAULT_MEAN_THRESHOLD,
    max_seed_threshold: float = DEFAULT_MAX_SEED_THRESHOLD,
) -> dict[str, Any]:
    required = tuple(int(value) for value in required_seeds)
    supplied = tuple(sorted(int(value) for value in seed_runs))
    if len(set(required)) != len(required) or not required:
        raise ValueError("Required Spooky seeds must be unique and non-empty")
    if set(supplied) != set(required):
        raise ValueError(f"Expected exact Spooky seeds {sorted(required)}, received {list(supplied)}")
    runs = [load_seed_run(seed, seed_runs[seed]) for seed in required]
    reference = runs[0]
    identity_checks = {
        "truth_aligned": all(np.array_equal(reference.truth, run.truth) for run in runs[1:]),
        "train_id_aligned": all(
            np.array_equal(reference.train_id, run.train_id) for run in runs[1:]
        ),
        "test_id_aligned": all(np.array_equal(reference.test_id, run.test_id) for run in runs[1:]),
        "oof_shape_aligned": all(reference.oof.shape == run.oof.shape for run in runs[1:]),
        "test_shape_aligned": all(reference.test.shape == run.test.shape for run in runs[1:]),
    }
    all_seed_checks = all(all(run.checks.values()) for run in runs)
    identities_aligned = all(identity_checks.values())
    scores = [run.score for run in runs]
    mean_score = float(np.mean(scores))
    maximum_score = float(np.max(scores))
    gate_checks = {
        "required_seeds_complete": supplied == tuple(sorted(required)),
        "every_seed_independently_verified": all_seed_checks,
        "averaged_oof_test_ids_aligned": identities_aligned,
        "mean_seed_log_loss_at_or_below_threshold": mean_score <= mean_threshold,
        "maximum_seed_log_loss_at_or_below_threshold": maximum_score <= max_seed_threshold,
        "private_labels_unused": all(run.checks["private_labels_unused"] for run in runs),
        "official_grader_not_executed": all(
            run.checks["official_grader_not_executed"] for run in runs
        ),
        "kaggle_submission_not_executed": all(
            run.checks["kaggle_submission_not_executed"] for run in runs
        ),
        "submission_withheld": True,
    }
    promotion_allowed = all(gate_checks.values())
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact_records: dict[str, dict[str, Any]] = {}
    aggregate_score: float | None = None
    if identities_aligned:
        averaged_oof = normalized_probability(np.mean([run.oof for run in runs], axis=0))
        averaged_test = normalized_probability(np.mean([run.test for run in runs], axis=0))
        aggregate_score = multiclass_log_loss(reference.truth, averaged_oof)
        bundle_path = out / "spooky_multiseed_oof_and_test.npz"
        write_npz_atomic(
            bundle_path,
            required_seeds=np.asarray(required, dtype=np.int64),
            truth=reference.truth,
            averaged_oof=averaged_oof,
            averaged_test=averaged_test,
            train_id=reference.train_id,
            test_id=reference.test_id,
            seed_scores=np.asarray(scores, dtype=np.float64),
        )
        submission_path = out / "candidate_submission_multiseed_withheld.csv"
        submission = pd.DataFrame({"id": reference.test_id})
        for index, column in enumerate(CLASS_COLUMNS):
            submission[column] = averaged_test[:, index]
        write_csv_atomic(submission_path, submission)
        artifact_records = {
            "averaged_prediction_bundle": {
                "path": str(bundle_path),
                "sha256": sha256_file(bundle_path),
            },
            "candidate_submission_withheld": {
                "path": str(submission_path),
                "sha256": sha256_file(submission_path),
            },
        }

    report = {
        "schema": "evomind.spooky.multiseed_promotion_gate.v1",
        "created_at": now_iso(),
        "status": "promotion_gate_passed" if promotion_allowed else "promotion_gate_failed",
        "promotion_allowed": promotion_allowed,
        "required_seeds": list(required),
        "thresholds": {
            "mean_seed_log_loss": mean_threshold,
            "maximum_individual_seed_log_loss": max_seed_threshold,
        },
        "seed_scores": {str(run.seed): run.score for run in runs},
        "mean_seed_log_loss": mean_score,
        "maximum_individual_seed_log_loss": maximum_score,
        "averaged_oof_log_loss": aggregate_score,
        "identity_checks": identity_checks,
        "gate_checks": gate_checks,
        "seed_runs": [
            {
                "seed": run.seed,
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "candidate_oof_log_loss": run.score,
                "checks": run.checks,
                "summary": {
                    "path": str(run.summary_path),
                    "sha256": sha256_file(run.summary_path),
                },
                "independent_verification": {
                    "path": str(run.verification_path),
                    "sha256": sha256_file(run.verification_path),
                },
                "prediction_bundle": {
                    "path": str(run.bundle_path),
                    "sha256": sha256_file(run.bundle_path),
                },
                "submission_withheld": {
                    "path": str(run.submission_path),
                    "sha256": sha256_file(run.submission_path),
                },
            }
            for run in runs
        ],
        "artifacts": artifact_records,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": (
            "Public-data OOF promotion gate only. An official medal claim requires a separate "
            "human-approved MLE-Bench private grading step."
        ),
    }
    write_json_atomic(out / "spooky_multiseed_gate.json", report)
    return report


def parse_seed_run(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("--seed-run must use SEED=RUN_DIR")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seed-run SEED must be an integer") from exc
    return seed, Path(path_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-run", action="append", type=parse_seed_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-seeds", default="40,41,42")
    parser.add_argument("--mean-threshold", type=float, default=DEFAULT_MEAN_THRESHOLD)
    parser.add_argument("--max-seed-threshold", type=float, default=DEFAULT_MAX_SEED_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required = tuple(int(value.strip()) for value in args.required_seeds.split(",") if value.strip())
    seed_runs: dict[int, Path] = {}
    for seed, path in args.seed_run:
        if seed in seed_runs:
            raise ValueError(f"Duplicate --seed-run for seed {seed}")
        seed_runs[seed] = path
    report = aggregate_runs(
        seed_runs,
        args.output_dir,
        required_seeds=required,
        mean_threshold=args.mean_threshold,
        max_seed_threshold=args.max_seed_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["promotion_allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
