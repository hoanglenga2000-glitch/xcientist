#!/usr/bin/env python3
"""Independently verify a RANZCR cross-run candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scripts import aggregate_ranzcr_crossrun_candidate as aggregate
except ModuleNotFoundError:
    import aggregate_ranzcr_crossrun_candidate as aggregate


def verify(candidate_dir: Path, public_dir: Path) -> dict:
    report = json.loads((candidate_dir / "aggregation_result.json").read_text(encoding="utf-8"))
    bundle_path = candidate_dir / "ranzcr_crossrun_oof_and_test.npz"
    submission_path = candidate_dir / "submission_withheld.csv"
    aggregate.require(report.get("schema") == "evomind.ranzcr.crossrun_candidate.v1", "RANZCR report schema differs")
    aggregate.require(aggregate.sha256_file(bundle_path) == report["prediction_bundle"]["sha256"], "RANZCR bundle hash differs")
    aggregate.require(aggregate.sha256_file(submission_path) == report["submission"]["sha256"], "RANZCR submission hash differs")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        truth = np.asarray(bundle["truth"], dtype=np.float64)
        folds = np.asarray(bundle["fold"], dtype=np.int16)
        baseline_oof = aggregate.normalize_probability(bundle["baseline_oof"])
        highres_oof = aggregate.normalize_probability(bundle["highres_oof"])
        candidate_oof = aggregate.normalize_probability(bundle["candidate_oof"])
        baseline_test = aggregate.normalize_probability(bundle["baseline_test"])
        highres_test = aggregate.normalize_probability(bundle["highres_test"])
        candidate_test = aggregate.normalize_probability(bundle["candidate_test"])
        weights = np.asarray(bundle["deployment_highres_weights"], dtype=np.float64)
    rebuilt_oof, rebuilt_weights, records = aggregate.crossfit_blend(
        truth, baseline_oof, highres_oof, folds, grid_step=0.025
    )
    rebuilt_test = (1.0 - rebuilt_weights[None, :]) * baseline_test + rebuilt_weights[None, :] * highres_test
    auc, labels = aggregate.mean_column_auc(truth, rebuilt_oof)
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    submission = pd.read_csv(submission_path)
    checks = {
        "oof_exact": bool(np.allclose(candidate_oof, rebuilt_oof, rtol=0.0, atol=1e-12)),
        "weights_exact": bool(np.allclose(weights, rebuilt_weights, rtol=0.0, atol=1e-12)),
        "test_exact": bool(np.allclose(candidate_test, rebuilt_test, rtol=0.0, atol=1e-12)),
        "auc_exact": bool(abs(auc - float(report["candidate"]["oof_auc"])) <= 1e-12),
        "label_auc_exact": bool(
            np.allclose(
                labels,
                [report["candidate"]["label_auc"][name] for name in aggregate.TARGET_COLUMNS],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "records_exact": records == report["candidate"]["crossfit_records"],
        "submission_columns": submission.columns.tolist() == sample.columns.tolist(),
        "submission_ids": submission.iloc[:, 0].tolist() == sample.iloc[:, 0].tolist(),
        "submission_probability": bool(
            np.allclose(
                submission.loc[:, aggregate.TARGET_COLUMNS].to_numpy(dtype=np.float64),
                rebuilt_test,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "no_private_labels": report.get("private_labels_used") is False,
        "no_official_grader": report.get("official_grader_executed") is False,
        "no_kaggle_submission": report.get("kaggle_submission_executed") is False,
        "no_process_signals": report.get("process_signals_sent") == 0,
    }
    passed = all(checks.values())
    return {
        "schema": "evomind.ranzcr.crossrun_independent_verification.v1",
        "created_at": aggregate.now_iso(),
        "status": "verification_passed" if passed else "verification_failed",
        "passed": passed,
        "candidate_ready": passed and report["candidate"]["passed"],
        "candidate_oof_auc": auc,
        "checks": checks,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "process_signals_sent": 0,
        "claim_boundary": "Verified public OOF candidate; official grading remains Human-Gated.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.candidate_dir / "independent_verification.json"
    report = verify(args.candidate_dir.resolve(), args.public_dir.resolve())
    aggregate.write_json_atomic(output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
