from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.deploy_job89941_taxi_cpu_candidate import (
    REMOTE_UNIFIED_SITE_PACKAGES,
    TaxiCpuDeploymentError,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/test/taxi"


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def make_plan(tmp_path: Path) -> Path:
    base = tmp_path / "base.json"
    route = tmp_path / "route.json"
    base.write_text("{}", encoding="utf-8")
    route.write_text("{}", encoding="utf-8")
    candidate = ROOT / "scripts" / "diagnose_taxi_cpu_candidate.py"
    deployment = ROOT / "scripts" / "deploy_job89941_taxi_cpu_candidate.py"
    plan = {
        "schema": "evomind.hpc.job89941_taxi_cpu_candidate_plan.v1",
        "status": "approved_candidate_only",
        "job_id": 89941,
        "profile": "job89941_cpu",
        "candidate_source": record(candidate),
        "deployment_source": record(deployment),
        "base_cache": {
            "remote_path": REMOTE + "/base/cache_manifest.json",
            "local_evidence_path": str(base),
            "bytes": base.stat().st_size,
            "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        },
        "route_stat_cache": {
            "remote_path": REMOTE + "/route/route_stat_manifest.json",
            "local_evidence_path": str(route),
            "bytes": route.stat().st_size,
            "sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
        },
        "remote_run_dir": REMOTE,
        "remote_source": REMOTE + "/source.py",
        "remote_plan": REMOTE + "/plan.json",
        "remote_state": REMOTE + "/state.json",
        "remote_log": REMOTE + "/train.log",
        "remote_output_dir": REMOTE + "/candidate",
        "runtime": {
            "threads": 60,
            "iterations": 1400,
            "seed": 43,
            "cuda_visible_devices": "",
            "pythonpath": REMOTE_UNIFIED_SITE_PACKAGES,
        },
        "boundary": {
            "visibility_mode": "PUBLIC_ONLY",
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "gpu_used": False,
            "process_signals_allowed": False,
            "other_processes_may_be_modified": False,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_validate_plan_accepts_frozen_cpu_candidate(tmp_path: Path) -> None:
    plan = validate_plan(make_plan(tmp_path))
    assert plan["runtime"]["threads"] == 60
    assert len(plan["_sha256"]) == 64


def test_validate_plan_rejects_gpu_or_escaped_path(tmp_path: Path) -> None:
    path = make_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan["boundary"]["gpu_used"] = True
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TaxiCpuDeploymentError, match="boundary"):
        validate_plan(path)

    path = make_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan["remote_output_dir"] = "/tmp/escape"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TaxiCpuDeploymentError, match="escaped"):
        validate_plan(path)
