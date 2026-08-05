from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import aggregate_ranzcr_crossrun_candidate as aggregate
from scripts import stage_mlebench_human_gate_candidate as stage
from scripts import stage_ranzcr_crossrun_candidate as ranzcr
from scripts import verify_ranzcr_crossrun_candidate as verify


def test_ranzcr_crossrun_package_supports_distinct_runs_with_same_seed(tmp_path: Path):
    rows = 30; tests = 4; labels = len(aggregate.TARGET_COLUMNS)
    truth = np.tile((np.arange(rows) % 2)[:, None], (1, labels)).astype(np.float32)
    fold = np.arange(rows) % 3
    baseline_oof = np.where(truth == 1, 0.96, 0.04); highres_oof = np.where(truth == 1, 0.99, 0.01)
    baseline_test = np.full((tests, labels), 0.4); highres_test = np.full((tests, labels), 0.6)
    baseline = tmp_path / "baseline.npz"; highres = tmp_path / "highres.npz"
    np.savez(baseline, truth=truth, fold=fold, selected_oof_probability=baseline_oof, selected_test_probability=baseline_test)
    np.savez(highres, truth=truth, fold=fold, selected_oof_probability=highres_oof, selected_test_probability=highres_test)
    public = tmp_path / "public"; public.mkdir()
    pd.DataFrame({"StudyInstanceUID": [f"tr-{i}" for i in range(rows)]}).to_csv(public / "train.csv", index=False)
    sample = pd.DataFrame({"StudyInstanceUID": [f"te-{i}" for i in range(tests)], **{name: 0.5 for name in aggregate.TARGET_COLUMNS}})
    sample.to_csv(public / "sample_submission.csv", index=False)
    collected = tmp_path / "collected"
    aggregate.aggregate(baseline_path=baseline, highres_path=highres, public_dir=public, output_dir=collected, grid_step=0.025, promotion_auc=0.9725)
    remote_verification = verify.verify(collected, public)
    aggregate.write_json_atomic(collected / "independent_verification.json", remote_verification)
    (collected / "sample_submission.csv").write_bytes((public / "sample_submission.csv").read_bytes())
    source_plan = tmp_path / "plan.json"
    source_plan.write_text(json.dumps({"data_contract": {"core": {"sample_submission.csv": {"bytes": (public / "sample_submission.csv").stat().st_size, "sha256": aggregate.sha256_file(public / "sample_submission.csv")}}}}), encoding="utf-8")
    package = tmp_path / "human_gate" / "ranzcr"
    report = ranzcr.build_package(collected_dir=collected, source_plan_path=source_plan, output_dir=package)
    assert report["ensemble_oof_auc"] == 1.0
    public_root = tmp_path / "public_root"; staged_sample = public_root / ranzcr.COMPETITION_ID / "prepared" / "public" / "sample_submission.csv"
    staged_sample.parent.mkdir(parents=True); staged_sample.write_bytes((public / "sample_submission.csv").read_bytes())
    verified = stage.verify_human_gate_package(package, public_data_root=public_root, allowed_package_root=tmp_path / "human_gate")
    assert verified.model_seeds == (42, 42)
    assert verified.cv_score == 1.0
