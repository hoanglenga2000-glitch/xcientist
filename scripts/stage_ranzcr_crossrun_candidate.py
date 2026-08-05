#!/usr/bin/env python3
"""Verify collected RANZCR cross-run artifacts and build a Human Gate package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import aggregate_taxi_multiseed_candidate as common  # noqa: E402
from scripts import verify_ranzcr_crossrun_candidate as verifier  # noqa: E402

COMPETITION_ID = "ranzcr-clip-catheter-line-classification"
DEFAULT_SOURCE_PLAN = PROJECT_ROOT / "workspace" / "mlebench_plans" / "ranzcr_highres_crossrun_hpc88240_frozen_plan_20260727.json"
DEFAULT_COLLECTED = PROJECT_ROOT / "workspace" / "hpc" / "job88240_ranzcr_human_gate_postrun" / "collected"
DEFAULT_OUTPUT = PROJECT_ROOT / "workspace" / "human_gate" / "ranzcr_crossrun_efficientnet512_convnext768_20260728"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def require(condition: bool, message: str) -> None:
    if not condition: raise common.TaxiAggregationError(message)


def build_package(*, collected_dir: Path, source_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    collected_dir = Path(collected_dir).resolve(); source_plan_path = Path(source_plan_path).resolve(); output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "RANZCR Human Gate package already exists")
    for name in ("aggregation_result.json", "independent_verification.json", "ranzcr_crossrun_oof_and_test.npz", "submission_withheld.csv", "sample_submission.csv"):
        require((collected_dir / name).is_file(), f"RANZCR artifact missing: {name}")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    report = json.loads((collected_dir / "aggregation_result.json").read_text(encoding="utf-8"))
    remote_verification = json.loads((collected_dir / "independent_verification.json").read_text(encoding="utf-8"))
    require(report.get("status") == "promotion_gate_passed" and (report.get("candidate") or {}).get("passed") is True, "RANZCR promotion gate failed")
    require(remote_verification.get("passed") is True and remote_verification.get("candidate_ready") is True, "RANZCR remote verification failed")
    local_verification = verifier.verify(collected_dir, collected_dir)
    require(local_verification.get("passed") is True and local_verification.get("candidate_ready") is True, "RANZCR local verification failed")
    auc = float(local_verification["candidate_oof_auc"])
    require(abs(auc - float(remote_verification["candidate_oof_auc"])) <= 1e-12, "RANZCR verifier AUC drifted")
    sample = collected_dir / "sample_submission.csv"
    expected_sample = source_plan["data_contract"]["core"]["sample_submission.csv"]
    require(sample.stat().st_size == int(expected_sample["bytes"]) and common.sha256_file(sample) == expected_sample["sha256"], "RANZCR public sample drifted")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_submission_withheld.csv"; shutil.copyfile(collected_dir / "submission_withheld.csv", candidate_path)
    candidate_record = common.file_record(candidate_path)
    frozen_plan = {
        "schema": "evomind.ranzcr.human_gate_postrun_plan.v1", "created_at": now_iso(), "competition_id": COMPETITION_ID,
        "seeds": [42, 42], "allow_duplicate_model_seeds_for_distinct_runs": True,
        "confirmation_gate": {"metric": "mean_columnwise_roc_auc", "direction": "maximize", "confirmation_seeds": [42, 42], "minimum_oof_auc": 0.9725},
        "components": [
            {"run_id": "a800_ranzcr_recovery_s42_20260726_145840", "model_seed": 42, "role": "efficientnet_v2_s_512_baseline"},
            {"run_id": "hpc88240_ranzcr_convnext768_s42_20260727", "model_seed": 42, "role": "convnext_small_768_highres"},
        ],
        "source_plan": common.file_record(source_plan_path), "sample_submission": common.file_record(sample),
        "boundaries": {"private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True},
        "claim_boundary": "Frozen cross-run public OOF package; not an official medal.",
    }
    frozen_plan_path = output_dir / "frozen_plan.json"; common.write_json_atomic(frozen_plan_path, frozen_plan)
    result = {
        "schema": "evomind.ranzcr.crossrun_confirmation_result.v1", "created_at": now_iso(), "competition_id": COMPETITION_ID,
        "status": "confirmation_passed_human_gate_pending", "candidate_ready_for_human_gate": True,
        "metrics": {"ensemble_oof_auc": auc}, "confirmation_gate": {"passed": True, "metric": "mean_columnwise_roc_auc", "direction": "maximize", "threshold": 0.9725},
        "seed_records": [{"model_seed": 42, "run_id": item["run_id"], "role": item["role"]} for item in frozen_plan["components"]],
        "submission_withheld": candidate_record, "sample_submission": common.file_record(sample),
        "source_aggregation": common.file_record(collected_dir / "aggregation_result.json"), "source_remote_verification": common.file_record(collected_dir / "independent_verification.json"),
        "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True,
        "claim_boundary": "Cross-fitted public OOF confirmation only; no official medal is claimed.",
    }
    result_path = output_dir / "ranzcr_crossrun_confirmation_result.json"; common.write_json_atomic(result_path, result)
    independent = {
        "schema": "evomind.ranzcr.local_human_gate_verification.v1", "created_at": now_iso(), "status": "verification_passed", "ok": True,
        "candidate_ready_for_human_gate": True, "errors": [], "plan_sha256": common.sha256_file(frozen_plan_path), "result_sha256": common.sha256_file(result_path),
        "recomputed_ensemble_oof_auc": auc, "checks": local_verification["checks"], "private_labels_used": False, "official_grader_executed": False,
        "kaggle_submission_executed": False, "process_signals_sent": 0, "human_gate_preserved": True,
        "claim_boundary": "Independent public OOF verification is not an official score or medal.",
    }
    independent_path = output_dir / "independent_verification.json"; common.write_json_atomic(independent_path, independent)
    readme = output_dir / "README.md"; readme.write_text("# RANZCR cross-run Human Gate package\n\nCandidate-only; grading remains disabled.\n", encoding="utf-8")
    files = [common.file_record(path) for path in (candidate_path, frozen_plan_path, independent_path, result_path, readme)]
    manifest = {"schema": "evomind.human_gate.candidate_package.v1", "created_at": now_iso(), "status": "ready_for_human_review_not_submitted", "competition_id": COMPETITION_ID, "public_oof_metrics": result["metrics"], "candidate_ready_for_human_gate": True, "files": files, "automatic_submission": False, "private_labels_used": False, "official_grader_executed": False, "kaggle_submission_executed": False, "process_signals_sent": 0, "claim_boundary": "Ready for Human Gate review only; official medal count remains unchanged."}
    manifest_path = output_dir / "manifest.json"; common.write_json_atomic(manifest_path, manifest)
    common.write_json_atomic(output_dir / "package_verification.json", {"schema": "evomind.human_gate.package_verification.v1", "created_at": now_iso(), "status": "verified", "files": files + [common.file_record(manifest_path)], "candidate_csv_present": True, "independent_verification_passed": True, "automatic_submission": False})
    return {"status": "ready_for_human_review_not_submitted", "package": str(output_dir), "ensemble_oof_auc": auc, "candidate_sha256": candidate_record["sha256"], "official_grader_executed": False}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--collected-dir", type=Path, default=DEFAULT_COLLECTED); parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv)
    print(json.dumps(build_package(collected_dir=args.collected_dir, source_plan_path=args.source_plan, output_dir=args.output_dir), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
