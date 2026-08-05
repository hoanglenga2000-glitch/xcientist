from __future__ import annotations

import json
import posixpath
from pathlib import Path

import pytest

from scripts import deploy_job89941_may2022_public_cache as deploy


def _record(relative: str) -> dict[str, object]:
    path = deploy.PROJECT_ROOT / Path(*relative.split("/"))
    return {
        "local_path": str(path.resolve()),
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": deploy.sha256_file(path),
    }


def _plan_payload() -> dict[str, object]:
    data_root = f"{deploy.ALLOWED_GPU_REMOTE_ROOT}/mlebench_official_data"
    base = f"{data_root}/{deploy.EXPECTED_COMPETITION}/prepared/public"
    hashes = {
        "train.csv": (283_303_880, "f0940d6a4c3536752bdc3ba99251fc3020662ea43b28dfad520d810b3cde5514"),
        "test.csv": (35_229_612, "6e758d247d1212a0b31733f751b9ce3f97e4704687294e11aa4290bbab0131ec"),
        "sample_submission.csv": (1_100_010, "9f557b7177f5983dedabfc538aea050561bb4e4a362bfe4a35040959c26a6f6a"),
    }
    return {
        "schema": deploy.PLAN_SCHEMA,
        "status": "frozen",
        "job_id": 89941,
        "competition_id": deploy.EXPECTED_COMPETITION,
        "resource_mode": "cpu_only",
        "visibility_mode": "PUBLIC_ONLY",
        "run_id": "job89941_may2022_public_cache_s42_v1_test",
        "remote_root": deploy.ALLOWED_GPU_REMOTE_ROOT,
        "data_root": data_root,
        "seed": 42,
        "folds": 5,
        "threads": 48,
        "inputs": [
            {
                "relative_path": f"prepared/public/{name}",
                "path": f"{base}/{name}",
                "bytes": size,
                "sha256": digest,
            }
            for name, (size, digest) in hashes.items()
        ],
        "sources": [_record(path) for path in sorted(deploy.EXPECTED_SOURCE_PATHS)],
        "contracts": {
            "full_public_training_rows": True,
            "private_files_read": [],
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_allowed": False,
            "other_processes_may_be_modified": False,
            "gpu_used": False,
            "cache_manifest_written_last": True,
        },
    }


def _write_plan(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload or _plan_payload()), encoding="utf-8")
    return path


def test_plan_validation_binds_exact_sources_and_public_inputs(tmp_path: Path):
    plan = deploy.validate_plan(_write_plan(tmp_path))
    assert {record.relative_path for record in plan.sources} == deploy.EXPECTED_SOURCE_PATHS
    assert {item["relative_path"] for item in plan.payload["inputs"]} == deploy.EXPECTED_INPUT_PATHS
    layout = deploy.build_layout(plan)
    assert layout.base.startswith(deploy.ALLOWED_GPU_REMOTE_ROOT + "/")
    assert "/job89941_may2022_public_cache/" in layout.cache
    argv = deploy.build_runner_argv(plan, layout)
    assert "--verify-only" not in argv
    assert "private" not in " ".join(argv).lower()
    assert deploy.build_runner_argv(plan, layout, verify_only=True)[-1] == "--verify-only"


def test_plan_validation_rejects_private_input(tmp_path: Path):
    payload = _plan_payload()
    payload["inputs"][0]["relative_path"] = "prepared/private/train.csv"
    payload["inputs"][0]["path"] = (
        f"{payload['data_root']}/{deploy.EXPECTED_COMPETITION}/prepared/private/train.csv"
    )
    with pytest.raises(deploy.DeploymentError, match="input inventory"):
        deploy.validate_plan(_write_plan(tmp_path, payload))


def test_plan_validation_rejects_source_drift(tmp_path: Path):
    payload = _plan_payload()
    payload["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(deploy.DeploymentError, match="source drifted"):
        deploy.validate_plan(_write_plan(tmp_path, payload))


def test_runtime_environment_disables_cuda_and_limits_threads(tmp_path: Path):
    plan = deploy.validate_plan(_write_plan(tmp_path))
    env = deploy._runtime_env(deploy.build_layout(plan), 48)
    assert env["PYTHONPATH"].split(":")[0] == posixpath.join(
        deploy.ALLOWED_GPU_REMOTE_ROOT, ".deps"
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["EVOMIND_MLEBENCH_PUBLIC_ONLY"] == "1"
    assert env["OMP_NUM_THREADS"] == "48"
    assert env["MKL_NUM_THREADS"] == "48"
