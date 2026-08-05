from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_siim_final_candidate_hpc.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_siim_hpc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_path_guard() -> None:
    module = load_module()
    module.require_remote_path(
        "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/evomind_mle22/job88240_siim"
    )
    with pytest.raises(RuntimeError, match="escaped"):
        module.require_remote_path("/tmp/not-allowed")


def test_plan_validator_rejects_signals_before_sources(tmp_path: Path) -> None:
    module = load_module()
    plan = {
        "schema": "evomind.siim.final_candidate_frozen_plan.v1",
        "status": "frozen_waiting_prerequisites",
        "execution_target": {
            "job_id": 88240,
            "gpu": "NVIDIA A40",
            "process_signals_allowed": True,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="permits signals"):
        module.validate_hpc_plan(path)
