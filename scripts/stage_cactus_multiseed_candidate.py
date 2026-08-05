#!/usr/bin/env python3
"""Verify a collected Cactus multiseed bundle and build its Human Gate package."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402

COMPETITION_ID = "aerial-cactus-identification"
DEFAULT_SOURCE_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "cactus_multiseed_highres_hpc88240_frozen_plan_v2_20260727.json"
)
DEFAULT_COLLECTED = PROJECT_ROOT / "workspace" / "hpc" / "job88240_cactus_human_gate_postrun" / "collected"
DEFAULT_OUTPUT = PROJECT_ROOT / "workspace" / "human_gate" / "cactus_convnext384_multiseed_s40_s41_s42_20260728"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise common.TaxiAggregationError(message)


def build_package(*, collected_dir: Path, source_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    collected_dir = Path(collected_dir).resolve()
    source_plan_path = Path(source_plan_path).resolve()
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "Cactus Human Gate package is immutable and already exists")
    report_path = collected_dir / "candidate_report.json"
    verification_path = collected_dir / "independent_verification.json"
    bundle_path = collected_dir / "cactus_multiseed_oof_and_test.npz"
    sample_path = collected_dir / "sample_submission.csv"
    for path in (report_path, verification_path, bundle_path, sample_path, source_plan_path):
        require(path.is_file() and not path.is_symlink(), f"Cactus artifact missing: {path.name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in report.get("seeds") or []]
    require(seeds == [40, 41, 42], "Cactus seed contract drifted")
    require(report.get("candidate_ready") is True and float(report.get("public_oof_auc")) >= 0.9997, "Cactus promotion gate failed")
    require(report.get("candidate_only") is True and report.get("valid_submission") is True, "Cactus candidate boundary failed")
    require(report.get("official_grader_executed") is False and report.get("private_labels_used") is False, "Cactus data boundary failed")
    require(verification.get("passed") is True and verification.get("candidate_ready") is True, "Cactus remote independent verification failed")
    require(common.sha256_file(bundle_path) == report["prediction_bundle"]["sha256"], "Cactus bundle hash drifted")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        truth = np.asarray(bundle["truth"]).reshape(-1)
        oof = np.asarray(bundle["selected_oof_probability"], dtype=np.float64).reshape(-1)
        test = np.asarray(bundle["selected_test_probability"], dtype=np.float64).reshape(-1)
        bundle_seeds = np.asarray(bundle["seeds"], dtype=np.int64).tolist()
    require(bundle_seeds == seeds, "Cactus bundle seeds drifted")
    require(len(truth) == len(oof) and len(test) > 0, "Cactus bundle cardinality failed")
    require(np.isfinite(oof).all() and np.isfinite(test).all(), "Cactus probabilities are non-finite")
    require(np.all((oof >= 0) & (oof <= 1)) and np.all((test >= 0) & (test <= 1)), "Cactus probability range failed")
    auc = float(roc_auc_score(truth, oof))
    require(abs(auc - float(report["public_oof_auc"])) <= 1e-12, "Cactus AUC recomputation drifted")
    require(abs(auc - float(verification["public_oof_auc"])) <= 1e-12, "Cactus verifier AUC drifted")
    sample = pd.read_csv(sample_path)
    require(list(sample.columns) == ["id", "has_cactus"], "Cactus sample schema drifted")
    require(len(sample) == len(test), "Cactus sample/test cardinality drifted")
    expected_sample = source_plan["data_contract"]["core"]["sample_submission.csv"]
    require(sample_path.stat().st_size == int(expected_sample["bytes"]) and common.sha256_file(sample_path) == expected_sample["sha256"], "Cactus public sample identity drifted")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_submission_withheld.csv"
    candidate = sample.copy(); candidate["has_cactus"] = test; candidate.to_csv(candidate_path, index=False)
    candidate_record = common.file_record(candidate_path)
    frozen_plan = {
        "schema": "evomind.cactus.human_gate_postrun_plan.v1", "created_at": now_iso(),
        "competition_id": COMPETITION_ID, "seeds": seeds,
        "confirmation_gate": {"metric": "roc_auc", "direction": "maximize", "confirmation_seeds": seeds, "minimum_ensemble_oof_auc": 0.9997},
        "source_plan": common.file_record(source_plan_path), "sample_submission": common.file_record(sample_path),
        "boundaries": {"private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True},
        "claim_boundary": "Frozen post-run public OOF package; not an official medal.",
    }
    frozen_plan_path = output_dir / "frozen_plan.json"; common.write_json_atomic(frozen_plan_path, frozen_plan)
    result = {
        "schema": "evomind.cactus.multiseed_confirmation_result.v1", "created_at": now_iso(),
        "competition_id": COMPETITION_ID, "status": "confirmation_passed_human_gate_pending",
        "candidate_ready_for_human_gate": True, "metrics": {"ensemble_oof_auc": auc},
        "confirmation_gate": {"passed": True, "metric": "roc_auc", "direction": "maximize", "threshold": 0.9997},
        "seed_records": [{"model_seed": seed} for seed in seeds], "submission_withheld": candidate_record,
        "sample_submission": common.file_record(sample_path), "source_candidate_report": common.file_record(report_path),
        "source_independent_verification": common.file_record(verification_path), "source_prediction_bundle": common.file_record(bundle_path),
        "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False,
        "process_signals_sent": 0, "human_gate_preserved": True,
        "claim_boundary": "Public OOF confirmation only; no official score or medal is claimed.",
    }
    result_path = output_dir / "cactus_multiseed_confirmation_result.json"; common.write_json_atomic(result_path, result)
    independent = {
        "schema": "evomind.cactus.local_independent_verification.v1", "created_at": now_iso(),
        "status": "verification_passed", "ok": True, "candidate_ready_for_human_gate": True, "errors": [],
        "plan_sha256": common.sha256_file(frozen_plan_path), "result_sha256": common.sha256_file(result_path),
        "recomputed_ensemble_oof_auc": auc, "private_labels_used": False, "official_grader_executed": False,
        "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True,
        "claim_boundary": "Independent public OOF verification is not an official score or medal.",
    }
    independent_path = output_dir / "independent_verification.json"; common.write_json_atomic(independent_path, independent)
    readme = output_dir / "README.md"; readme.write_text("# Cactus multiseed Human Gate package\n\nCandidate-only; official grading remains disabled.\n", encoding="utf-8")
    files = [common.file_record(path) for path in (candidate_path, frozen_plan_path, independent_path, result_path, readme)]
    manifest = {
        "schema": "evomind.human_gate.candidate_package.v1", "created_at": now_iso(),
        "status": "ready_for_human_review_not_submitted", "competition_id": COMPETITION_ID,
        "public_oof_metrics": result["metrics"], "candidate_ready_for_human_gate": True, "files": files,
        "automatic_submission": False, "private_labels_used": False, "official_grader_executed": False,
        "kaggle_submission_executed": False, "process_signals_sent": 0,
        "claim_boundary": "Ready for Human Gate review only; official medal count remains unchanged.",
    }
    manifest_path = output_dir / "manifest.json"; common.write_json_atomic(manifest_path, manifest)
    common.write_json_atomic(output_dir / "package_verification.json", {
        "schema": "evomind.human_gate.package_verification.v1", "created_at": now_iso(), "status": "verified",
        "files": files + [common.file_record(manifest_path)], "candidate_csv_present": True,
        "independent_verification_passed": True, "automatic_submission": False,
    })
    return {"status": "ready_for_human_review_not_submitted", "package": str(output_dir), "ensemble_oof_auc": auc, "candidate_sha256": candidate_record["sha256"], "official_grader_executed": False}


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=DEFAULT_COLLECTED)
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv); print(json.dumps(build_package(collected_dir=args.collected_dir, source_plan_path=args.source_plan, output_dir=args.output_dir), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
