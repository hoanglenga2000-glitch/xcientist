from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest

from scripts import mlebench_compute_policy as policy
from scripts import (
    queue_jigsaw_confirmation_after_leaf,
    queue_leaf_after_siim,
    queue_siim_after_jigsaw,
    queue_siim_final_candidate,
    run_jigsaw_transformer_oof,
    run_leaf_multibackbone_oof,
    run_mlebench_lite_full,
    run_siim_preprocessing_ablation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_policy(root: Path, *, backend: str = "hpc_only", allow: bool = False) -> Path:
    path = root / policy.POLICY_RELATIVE_PATH
    policy.atomic_write_json(
        path,
        {
            "schema": policy.POLICY_SCHEMA,
            "compute_backend": backend,
            "allow_local_gpu_training": allow,
            "preferred_hpc_job_id": "88240",
            "hpc_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        },
    )
    return path


def test_checked_in_policy_enforces_hpc_only() -> None:
    record = policy.load_compute_policy(PROJECT_ROOT)

    assert record["valid"] is True
    assert record["active"] is True
    assert record["compute_backend"] == "hpc_only"
    assert record["allow_local_gpu_training"] is False
    assert record["preferred_hpc_job_id"] == "88240"


def test_missing_policy_preserves_legacy_behavior(tmp_path: Path) -> None:
    record = policy.load_compute_policy(tmp_path)

    assert record["present"] is False
    assert record["active"] is False
    assert record["fail_closed"] is False


def test_invalid_existing_policy_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / policy.POLICY_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text('{"compute_backend":', encoding="utf-8")

    record = policy.load_compute_policy(tmp_path, read_attempts=1)

    assert record["present"] is True
    assert record["valid"] is False
    assert record["active"] is True
    assert record["fail_closed"] is True
    assert record["allow_local_gpu_training"] is False
    assert "JSONDecodeError" in record["error"]


def test_policy_read_retries_transient_sharing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_policy(tmp_path)
    original = Path.read_bytes
    attempts = 0

    def flaky_read_bytes(candidate: Path) -> bytes:
        nonlocal attempts
        if candidate.resolve() == path.resolve() and attempts == 0:
            attempts += 1
            raise PermissionError("transient sharing violation")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)
    record = policy.load_compute_policy(
        tmp_path,
        read_attempts=2,
        retry_seconds=0,
    )

    assert attempts == 1
    assert record["valid"] is True
    assert record["active"] is True


def test_atomic_json_writer_leaves_complete_artifact_only(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "status.json"

    policy.atomic_write_json(destination, {"status": "complete", "value": 7})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "complete",
        "value": 7,
    }
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


QUEUE_MODULES = (
    (queue_siim_after_jigsaw, "evomind.siim.training_queue.v1"),
    (queue_siim_final_candidate, "evomind.siim.final_candidate_queue.v1"),
    (queue_leaf_after_siim, "evomind.leaf.training_queue.v1"),
    (
        queue_jigsaw_confirmation_after_leaf,
        "evomind.jigsaw.confirmation_queue.v1",
    ),
)


@pytest.mark.parametrize(("module", "schema"), QUEUE_MODULES)
def test_queue_exits_before_plan_or_popen_when_hpc_only(
    module: ModuleType,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_policy(tmp_path)
    status_path = tmp_path / f"{module.__name__}.status.json"
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("queue progressed beyond the HPC-only policy guard")

    monkeypatch.setattr(module, "validate_frozen_plan", unexpected)
    monkeypatch.setattr(module.subprocess, "Popen", unexpected)

    result = module.main(
        [
            "--plan",
            str(tmp_path / "not-needed-plan.json"),
            "--status",
            str(status_path),
        ]
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["schema"] == schema
    assert payload["status"] == policy.QUEUE_SUPERSEDED_STATUS
    assert payload["training_started"] is False
    assert payload["training_pid"] is None
    assert payload["process_signals_sent"] == 0
    assert payload["compute_policy"]["active"] is True


RUNNER_MODULES = (
    run_jigsaw_transformer_oof,
    run_siim_preprocessing_ablation,
    run_leaf_multibackbone_oof,
    run_mlebench_lite_full,
)


@pytest.mark.parametrize("module", RUNNER_MODULES)
def test_runner_exits_with_evidence_before_training_setup(
    module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_policy(tmp_path)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    output_root = tmp_path / "outputs"
    run_id = "blocked-run"

    if module is run_jigsaw_transformer_oof:
        argv = [
            "--public-dir",
            str(tmp_path / "public"),
            "--sparse-bundle",
            str(tmp_path / "sparse.npz"),
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
            "--frozen-plan",
            str(tmp_path / "plan.json"),
            "--hf-cache",
            str(tmp_path / "hf-cache"),
        ]
        evidence_path = output_root / run_id / "hpc_only_policy_block.json"
    elif module is run_siim_preprocessing_ablation:
        argv = [
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
        ]
        evidence_path = output_root / "runs" / run_id / "hpc_only_policy_block.json"
    elif module is run_leaf_multibackbone_oof:
        argv = [
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(output_root),
            "--run-id",
            run_id,
        ]
        evidence_path = output_root / "runs" / run_id / "hpc_only_policy_block.json"
    else:
        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()
        output_root = allowed_root / "outputs"
        argv = [
            "--allowed-root",
            str(allowed_root),
            "--data-root",
            str(allowed_root / "data"),
            "--output-root",
            str(output_root),
            "--official-source-root",
            str(allowed_root / "official-source"),
            "--run-id",
            run_id,
        ]
        evidence_path = output_root / run_id / "hpc_only_policy_block.json"

    result = module.main(argv)

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["status"] == policy.RUNNER_BLOCKED_STATUS
    assert payload["training_started"] is False
    assert payload["cuda_initialization_started"] is False
    assert payload["process_signals_sent"] == 0
    assert payload["compute_policy"]["active"] is True

