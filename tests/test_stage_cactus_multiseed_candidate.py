from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import stage_cactus_multiseed_candidate as cactus
from scripts import stage_mlebench_human_gate_candidate as stage


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cactus_candidate_builds_stageable_human_gate_package(tmp_path: Path):
    collected = tmp_path / "collected"; collected.mkdir()
    truth = np.array([0, 1, 0, 1, 0, 1])
    oof = np.array([0.001, 0.999, 0.002, 0.998, 0.003, 0.997], dtype=np.float64)
    test = np.array([0.2, 0.8], dtype=np.float64)
    bundle = collected / "cactus_multiseed_oof_and_test.npz"
    np.savez_compressed(bundle, truth=truth, selected_oof_probability=oof, selected_test_probability=test, seeds=np.array([40, 41, 42]))
    sample = collected / "sample_submission.csv"
    pd.DataFrame({"id": ["x", "y"], "has_cactus": [0.5, 0.5]}).to_csv(sample, index=False)
    report = {"seeds": [40, 41, 42], "public_oof_auc": 1.0, "promotion_auc": 0.9997, "candidate_ready": True, "candidate_only": True, "valid_submission": True, "official_grader_executed": False, "private_labels_used": False, "prediction_bundle": {"sha256": cactus.common.sha256_file(bundle)}}
    _json(collected / "candidate_report.json", report)
    _json(collected / "independent_verification.json", {"passed": True, "candidate_ready": True, "public_oof_auc": 1.0})
    source_plan = tmp_path / "source_plan.json"
    _json(source_plan, {"data_contract": {"core": {"sample_submission.csv": {"bytes": sample.stat().st_size, "sha256": cactus.common.sha256_file(sample)}}}})
    package = tmp_path / "human_gate" / "cactus-package"
    result = cactus.build_package(collected_dir=collected, source_plan_path=source_plan, output_dir=package)
    assert result["ensemble_oof_auc"] == 1.0
    public_root = tmp_path / "public"
    staged_sample = public_root / cactus.COMPETITION_ID / "prepared" / "public" / "sample_submission.csv"
    staged_sample.parent.mkdir(parents=True); staged_sample.write_bytes(sample.read_bytes())
    verified = stage.verify_human_gate_package(package, public_data_root=public_root, allowed_package_root=tmp_path / "human_gate")
    assert verified.model_seeds == (40, 41, 42)
    assert verified.cv_score == 1.0
