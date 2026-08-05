from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import stage_mlebench_human_gate_candidate as stage
from scripts import stage_siim_hpc_candidate as siim


def test_siim_single_seed_nested_oof_package_is_stageable(tmp_path: Path):
    collected = tmp_path / "collected"; collected.mkdir()
    truth = np.array([0, 1, 0, 1, 0, 1]); prob = np.array([0.01, 0.99, 0.02, 0.98, 0.03, 0.97])
    pd.DataFrame({"image_name": [f"tr-{i}" for i in range(6)], "target": truth, "fold": [0, 0, 1, 1, 2, 2], "blended_probability": prob}).to_csv(collected / "siim_oof_predictions.csv", index=False)
    sample = pd.DataFrame({"image_name": ["te-1", "te-2"], "target": [0.5, 0.5]}); sample.to_csv(collected / "sample_submission.csv", index=False)
    submission = pd.DataFrame({"image_name": ["te-1", "te-2"], "target": [0.2, 0.8]}); submission.to_csv(collected / "submission.csv", index=False)
    np.savez(collected / "siim_fold_ensemble.npz", test_id=np.array(["te-1", "te-2"]), test_probability=np.array([0.2, 0.8]))
    (collected / "result.json").write_text(json.dumps({"status": "promotion_gate_passed_confirmation_pending", "candidate_only": True, "official_grader_executed": False, "cv_score": 1.0}), encoding="utf-8")
    (collected / "independent_verification.json").write_text(json.dumps({"passed": True, "candidate_ready": True, "oof": {"recomputed_auc": 1.0}}), encoding="utf-8")
    plan = tmp_path / "plan.json"; plan.write_text(json.dumps({"objective": {"promotion_auc": 0.942}, "training": {"run_id": "siim-s42"}}), encoding="utf-8")
    package = tmp_path / "human_gate" / "siim"
    report = siim.build_package(collected_dir=collected, source_plan_path=plan, output_dir=package)
    assert report["ensemble_oof_auc"] == 1.0
    public_root = tmp_path / "public"; staged_sample = public_root / siim.COMPETITION_ID / "prepared" / "public" / "sample_submission.csv"
    staged_sample.parent.mkdir(parents=True); staged_sample.write_bytes((collected / "sample_submission.csv").read_bytes())
    verified = stage.verify_human_gate_package(package, public_data_root=public_root, allowed_package_root=tmp_path / "human_gate")
    assert verified.model_seeds == (42,)
    assert verified.cv_score == 1.0
