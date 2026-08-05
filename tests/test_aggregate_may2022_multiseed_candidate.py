from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts import aggregate_may2022_multiseed_candidate as aggregate
from scripts import stage_mlebench_human_gate_candidate as stage


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(root: Path, run_id: str, seed: int, shift: float) -> None:
    competition = root / run_id / aggregate.COMPETITION_ID
    attempt = competition / "attempts" / "attempt_001"
    attempt.mkdir(parents=True)
    target = np.tile(np.array([0, 1], dtype=np.int8), 6)
    fold = np.repeat(np.arange(3, dtype=np.int16), 4)
    oof = np.clip(0.02 + 0.96 * target + shift, 0.001, 0.999)
    test_id = np.array([101, 102, 103])
    test = np.clip(np.array([0.2, 0.5, 0.8]) + shift, 0.001, 0.999)
    bundle = attempt / "may2022_oof_ensemble.npz"
    np.savez(bundle, id=np.arange(len(target)), target=target, fold_assignment=fold, oof_probability=oof, test_id=test_id, test_probability=test)
    submission = attempt / "submission.csv"
    pd.DataFrame({"id": test_id, "target": test}).to_csv(submission, index=False)
    auc = float(roc_auc_score(target, oof))
    result = {"competition_id": aggregate.COMPETITION_ID, "status": "candidate_ready", "metric": "roc_auc", "direction": "maximize", "cv_score": auc, "valid_submission": True, "submission_sha256": aggregate.common.sha256_file(submission), "official_grader_executed": False, "promotion_gate": {"passed": True}}
    result_path = competition / "result.json"
    _write_json(result_path, result)
    files = [{"local": str(path.resolve()), "bytes": path.stat().st_size, "sha256": aggregate.common.sha256_file(path)} for path in (bundle, submission, result_path)]
    _write_json(root / run_id / "collection_manifest.json", {"passed": True, "run_id": run_id, "files": files})


def test_may_multiseed_package_is_stageable(tmp_path: Path):
    plan = json.loads(aggregate.DEFAULT_PLAN.read_text(encoding="utf-8"))
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": [101, 102, 103], "target": [0.5, 0.5, 0.5]}).to_csv(sample, index=False)
    plan["public_sample_submission"] = {"remote_path": "/tmp/sample_submission.csv", "bytes": sample.stat().st_size, "sha256": aggregate.common.sha256_file(sample)}
    plan_path = tmp_path / "plan.json"; _write_json(plan_path, plan)
    collected = tmp_path / "collected"
    for seed, run_id, shift in zip(plan["seeds"], plan["run_ids"], [0.0, -0.005, 0.005], strict=True):
        _run(collected, run_id, seed, shift)
    package = tmp_path / "human_gate" / "may-package"
    report = aggregate.aggregate(plan_path=plan_path, collected_root=collected, sample_path=sample, output_dir=package)
    assert report["metrics"]["ensemble_oof_auc"] == 1.0
    public_root = tmp_path / "public"
    staged_sample = public_root / aggregate.COMPETITION_ID / "prepared" / "public" / "sample_submission.csv"
    staged_sample.parent.mkdir(parents=True); staged_sample.write_bytes(sample.read_bytes())
    verified = stage.verify_human_gate_package(package, public_data_root=public_root, allowed_package_root=tmp_path / "human_gate")
    assert verified.model_seeds == (42, 43, 44)
    assert verified.metric == "roc_auc"
