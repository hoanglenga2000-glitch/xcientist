#!/usr/bin/env python3
"""Verify collected SIIM nested-OOF artifacts and build a Human Gate package."""

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
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402

COMPETITION_ID = "siim-isic-melanoma-classification"
DEFAULT_SOURCE_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "siim_final_candidate_hpc88240_frozen_plan_20260727.json"
DEFAULT_COLLECTED = PROJECT_ROOT / "workspace" / "hpc" / "job88240_siim_human_gate_postrun" / "collected"
DEFAULT_OUTPUT = PROJECT_ROOT / "workspace" / "human_gate" / "siim_nested_multimodal_s42_20260728"


def now_iso(): return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition: raise common.TaxiAggregationError(message)


def build_package(*, collected_dir: Path, source_plan_path: Path, output_dir: Path):
    collected_dir = Path(collected_dir).resolve(); source_plan_path = Path(source_plan_path).resolve(); output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "SIIM Human Gate package already exists")
    names = ("independent_verification.json", "result.json", "submission.csv", "siim_oof_predictions.csv", "siim_fold_ensemble.npz", "sample_submission.csv")
    for name in names: require((collected_dir / name).is_file(), f"SIIM artifact missing: {name}")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8")); verification = json.loads((collected_dir / "independent_verification.json").read_text(encoding="utf-8")); result = json.loads((collected_dir / "result.json").read_text(encoding="utf-8"))
    require(verification.get("passed") is True and verification.get("candidate_ready") is True, "SIIM remote verification failed")
    require(result.get("status") == "promotion_gate_passed_confirmation_pending", "SIIM promotion gate failed")
    require(result.get("candidate_only") is True and result.get("official_grader_executed") is False, "SIIM candidate boundary failed")
    oof = pd.read_csv(collected_dir / "siim_oof_predictions.csv")
    required = {"image_name", "target", "fold", "blended_probability"}; require(required <= set(oof.columns), "SIIM OOF schema drifted")
    truth = oof["target"].to_numpy(dtype=np.float64); probability = oof["blended_probability"].to_numpy(dtype=np.float64)
    require(set(np.unique(truth)) == {0.0, 1.0} and np.isfinite(probability).all(), "SIIM OOF values invalid")
    auc = float(roc_auc_score(truth, probability)); require(abs(auc - float(result["cv_score"])) <= 1e-12, "SIIM result AUC drifted")
    require(abs(auc - float(verification["oof"]["recomputed_auc"])) <= 1e-12, "SIIM verifier AUC drifted")
    sample = pd.read_csv(collected_dir / "sample_submission.csv"); submission = pd.read_csv(collected_dir / "submission.csv")
    with np.load(collected_dir / "siim_fold_ensemble.npz", allow_pickle=False) as bundle:
        test_id = np.asarray(bundle["test_id"]).astype(str); test_probability = np.asarray(bundle["test_probability"], dtype=np.float64)
    rebuilt = sample[["image_name"]].merge(pd.DataFrame({"image_name": test_id, "target": test_probability}), on="image_name", how="left", validate="one_to_one")
    require(submission.columns.tolist() == sample.columns.tolist() == ["image_name", "target"], "SIIM submission schema drifted")
    require(submission["image_name"].astype(str).tolist() == sample["image_name"].astype(str).tolist(), "SIIM submission ID order drifted")
    require(np.allclose(submission["target"].to_numpy(dtype=float), rebuilt["target"].to_numpy(dtype=float), atol=1e-12, rtol=0), "SIIM submission rebuild drifted")
    output_dir.mkdir(parents=True); candidate = output_dir / "candidate_submission_withheld.csv"; shutil.copyfile(collected_dir / "submission.csv", candidate); candidate_record = common.file_record(candidate)
    frozen_plan = {"schema": "evomind.siim.human_gate_postrun_plan.v1", "created_at": now_iso(), "competition_id": COMPETITION_ID, "seeds": [42], "single_seed_nested_oof_confirmation": True, "confirmation_gate": {"metric": "roc_auc", "direction": "maximize", "confirmation_seeds": [42], "minimum_oof_auc": float(source_plan["objective"]["promotion_auc"]), "outer_folds": 5, "inner_folds": 3}, "source_plan": common.file_record(source_plan_path), "sample_submission": common.file_record(collected_dir / "sample_submission.csv"), "boundaries": {"private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True}, "claim_boundary": "Frozen nested-OOF public package; not an official medal."}
    frozen = output_dir / "frozen_plan.json"; common.write_json_atomic(frozen, frozen_plan)
    confirmation = {"schema": "evomind.siim.hpc_confirmation_result.v1", "created_at": now_iso(), "competition_id": COMPETITION_ID, "status": "confirmation_passed_human_gate_pending", "candidate_ready_for_human_gate": True, "metrics": {"ensemble_oof_auc": auc}, "confirmation_gate": {"passed": True, "metric": "roc_auc", "direction": "maximize", "threshold": source_plan["objective"]["promotion_auc"]}, "seed_records": [{"model_seed": 42, "run_id": source_plan["training"]["run_id"]}], "submission_withheld": candidate_record, "sample_submission": common.file_record(collected_dir / "sample_submission.csv"), "source_remote_verification": common.file_record(collected_dir / "independent_verification.json"), "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True, "claim_boundary": "Nested public OOF confirmation only; no official medal is claimed."}
    result_path = output_dir / "siim_confirmation_result.json"; common.write_json_atomic(result_path, confirmation)
    independent = {"schema": "evomind.siim.local_human_gate_verification.v1", "created_at": now_iso(), "status": "verification_passed", "ok": True, "candidate_ready_for_human_gate": True, "errors": [], "plan_sha256": common.sha256_file(frozen), "result_sha256": common.sha256_file(result_path), "recomputed_ensemble_oof_auc": auc, "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True, "claim_boundary": "Independent public OOF verification is not an official score or medal."}
    independent_path = output_dir / "independent_verification.json"; common.write_json_atomic(independent_path, independent)
    readme = output_dir / "README.md"; readme.write_text("# SIIM nested multimodal Human Gate package\n\nCandidate-only; grading remains disabled.\n", encoding="utf-8")
    files = [common.file_record(path) for path in (candidate, frozen, independent_path, result_path, readme)]
    manifest = {"schema": "evomind.human_gate.candidate_package.v1", "created_at": now_iso(), "status": "ready_for_human_review_not_submitted", "competition_id": COMPETITION_ID, "public_oof_metrics": confirmation["metrics"], "candidate_ready_for_human_gate": True, "files": files, "automatic_submission": False, "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "claim_boundary": "Ready for Human Gate review only; official medal count remains unchanged."}
    manifest_path = output_dir / "manifest.json"; common.write_json_atomic(manifest_path, manifest); common.write_json_atomic(output_dir / "package_verification.json", {"schema": "evomind.human_gate.package_verification.v1", "created_at": now_iso(), "status": "verified", "files": files + [common.file_record(manifest_path)], "candidate_csv_present": True, "independent_verification_passed": True, "automatic_submission": False})
    return {"status": "ready_for_human_review_not_submitted", "package": str(output_dir), "ensemble_oof_auc": auc, "candidate_sha256": candidate_record["sha256"], "official_grader_executed": False}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--collected-dir", type=Path, default=DEFAULT_COLLECTED); parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv); print(json.dumps(build_package(collected_dir=args.collected_dir, source_plan_path=args.source_plan, output_dir=args.output_dir), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
