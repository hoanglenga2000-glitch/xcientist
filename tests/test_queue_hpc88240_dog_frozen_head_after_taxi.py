from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import queue_hpc88240_dog_frozen_head_after_taxi as queue


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plan(tmp_path: Path) -> dict:
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"frozen-dog-bundle")
    taxi_plan = tmp_path / "taxi-plan.json"
    taxi_run_ids = ["taxi-s43", "taxi-s44", "taxi-s45"]
    _json(taxi_plan, {"run_ids": taxi_run_ids})
    path = tmp_path / "dog-plan.json"
    _json(
        path,
        {
            "schema": queue.PLAN_SCHEMA,
            "status": "frozen_waiting_for_taxi_terminal",
            "competition_id": queue.COMPETITION_ID,
            "visibility_mode": "PUBLIC_ONLY",
            "remote_root": queue.ops.ALLOWED_GPU_REMOTE_ROOT,
            "automatic_official_grader": False,
            "kaggle_submission_enabled": False,
            "bundle": {
                "path": str(bundle.resolve()),
                "bytes": bundle.stat().st_size,
                "sha256": queue.sha256_file(bundle),
            },
            "serial_dependency": {
                "taxi_plan_path": str(taxi_plan.resolve()),
                "taxi_plan_sha256": queue.sha256_file(taxi_plan),
                "taxi_run_ids": taxi_run_ids,
                "all_taxi_seeds_terminal_before_dog": True,
            },
            "diagnostic": {
                "run_id": "dog-diagnostic-s46",
                "seed": 46,
                "backbone": "convnext_small",
                "training_mode": "frozen_backbone_head",
                "head_learning_rate": 0.001,
                "fold_limit": 1,
                "folds": 5,
                "epochs": 4,
                "batch_size": 32,
                "parent_fold0_epoch1_log_loss": 0.156225621700287,
                "parent_fold0_top1_accuracy": 0.946195652173913,
            },
            "confirmation": {
                "seeds": [46, 47],
                "run_ids": ["dog-full-s46", "dog-full-s47"],
                "epochs": 8,
                "folds": 5,
                "batch_size": 32,
                "aggregate_oof_log_loss_maximum": 0.04,
                "every_seed_oof_log_loss_maximum": 0.04,
            },
            "confirmation_gate": {
                "confirmation_seeds": [46, 47],
                "every_seed_oof_log_loss_maximum": 0.04,
                "aggregate_oof_log_loss_maximum": 0.04,
            },
            "public_inputs": {
                "root": queue.DOG_PUBLIC_ROOT,
                "train_count": 9199,
                "test_count": 1023,
                "private_paths_read": [],
                **{
                    name: {"path": f"{queue.DOG_PUBLIC_ROOT}/{name}", **record}
                    for name, record in queue.DOG_PUBLIC_INPUT_CONTRACT.items()
                },
            },
            "public_sample_submission": {
                "remote_path": f"{queue.DOG_PUBLIC_ROOT}/sample_submission.csv",
                **queue.DOG_PUBLIC_INPUT_CONTRACT["sample_submission.csv"],
            },
            "boundaries": {
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
                "process_signals_sent": 0,
                "human_gate_preserved": True,
            },
        },
    )
    return queue.validate_plan(path)


def _taxi_status(tmp_path: Path, plan: dict, *, status: str) -> Path:
    path = tmp_path / "taxi-status.json"
    _json(
        path,
        {
            "status": status,
            "plan_sha256": plan["serial_dependency"]["taxi_plan_sha256"],
            "completed_seeds": [
                {"run_id": run_id}
                for run_id in plan["serial_dependency"]["taxi_run_ids"]
            ],
        },
    )
    return path


def _terminal() -> dict:
    return {"process": "stopped", "summary": {"competition_count": 1}}


def test_plan_validation_binds_bundle_taxi_inventory_and_human_gate(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan["bundle"]["sha256"] == queue.sha256_file(Path(plan["_bundle_path"]))
    assert plan["serial_dependency"]["taxi_run_ids"] == [
        "taxi-s43",
        "taxi-s44",
        "taxi-s45",
    ]
    assert plan["diagnostic"]["training_mode"] == "frozen_backbone_head"
    assert plan["confirmation"]["seeds"] == [46, 47]
    assert plan["automatic_official_grader"] is False
    assert plan["kaggle_submission_enabled"] is False


def test_taxi_not_terminal_never_reads_gpu_or_launches(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    taxi = _taxi_status(tmp_path, plan, status="seed_active")
    monkeypatch.setattr(
        queue.ops,
        "sample_gpu_idle_gate",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("GPU gate called")),
    )
    monkeypatch.setattr(
        queue.ops,
        "read_remote_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote status called")),
    )
    monkeypatch.setattr(
        queue.ops,
        "start_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("launch called")),
    )

    result = queue.run_once(plan, taxi_status_path=taxi, evidence_dir=tmp_path)

    assert result["status"] == "waiting_for_taxi_terminal"
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_diagnostic_gate_failure_never_launches_full(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    taxi = _taxi_status(tmp_path, plan, status="all_seeds_terminal")
    monkeypatch.setattr(queue.ops, "read_remote_status", lambda _run_id: _terminal())
    monkeypatch.setattr(queue.taxi_queue, "remote_run_terminal", lambda _remote: True)
    monkeypatch.setattr(
        queue,
        "_collected_result",
        lambda run_id: {
            "competition_id": queue.COMPETITION_ID,
            "continuation_allowed": False,
            "run_id": run_id,
        },
    )
    monkeypatch.setattr(
        queue,
        "_launch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full launched")),
    )

    result = queue.run_once(plan, taxi_status_path=taxi, evidence_dir=tmp_path)

    assert result["status"] == "diagnostic_gate_failed"
    assert result["full_runs_launched"] == []
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_diagnostic_pass_launches_only_first_full_seed(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    taxi = _taxi_status(tmp_path, plan, status="all_seeds_terminal")

    def remote(run_id: str):
        if run_id == plan["diagnostic"]["run_id"]:
            return _terminal()
        raise queue.ops.RemoteOpsError("missing")

    launches = []
    monkeypatch.setattr(queue.ops, "read_remote_status", remote)
    monkeypatch.setattr(queue.taxi_queue, "remote_run_terminal", lambda _remote: True)
    monkeypatch.setattr(
        queue,
        "_collected_result",
        lambda _run_id: {
            "competition_id": queue.COMPETITION_ID,
            "continuation_allowed": True,
        },
    )

    def launch(_plan, _evidence_dir, *, run_id: str, seed: int, diagnostic: bool):
        launches.append((run_id, seed, diagnostic))
        return {"status": "full_seed_active", "start": {"run_id": run_id}}

    monkeypatch.setattr(queue, "_launch", launch)

    result = queue.run_once(plan, taxi_status_path=taxi, evidence_dir=tmp_path)

    assert result["status"] == "full_seed_active"
    assert launches == [(plan["confirmation"]["run_ids"][0], 46, False)]
    assert result["active_seed"] == {
        "seed": 46,
        "run_id": plan["confirmation"]["run_ids"][0],
    }


@pytest.mark.parametrize("failed_seed", [46, 47])
def test_any_full_seed_gate_failure_stops_successors(
    tmp_path: Path, monkeypatch, failed_seed: int
) -> None:
    plan = _plan(tmp_path)
    taxi = _taxi_status(tmp_path, plan, status="all_seeds_terminal")
    monkeypatch.setattr(queue.ops, "read_remote_status", lambda _run_id: _terminal())
    monkeypatch.setattr(queue.taxi_queue, "remote_run_terminal", lambda _remote: True)

    def collected(run_id: str) -> dict:
        if run_id == plan["diagnostic"]["run_id"]:
            return {
                "competition_id": queue.COMPETITION_ID,
                "continuation_allowed": True,
            }
        seed = 46 if run_id.endswith("s46") else 47
        passed = seed != failed_seed
        return {
            "competition_id": queue.COMPETITION_ID,
            "promotion_gate": {"passed": passed},
            "cv_score": 0.03 if passed else 0.05,
        }

    monkeypatch.setattr(queue, "_collected_result", collected)
    monkeypatch.setattr(
        queue,
        "_launch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("successor launched")),
    )

    result = queue.run_once(plan, taxi_status_path=taxi, evidence_dir=tmp_path)

    assert result["status"] == "full_seed_gate_failed"
    assert result["failed_seed"] == failed_seed
    assert [item["seed"] for item in result["completed_seeds"]] == (
        [] if failed_seed == 46 else [46]
    )
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_launch_contract_remains_candidate_only_without_grader_or_kaggle(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path)
    gates = iter([{"passed": True}, {"passed": True}])
    calls = {}
    monkeypatch.setattr(queue.ops, "sample_gpu_idle_gate", lambda **_kwargs: next(gates))
    monkeypatch.setattr(queue.ops, "deploy_bundle", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(queue.ops, "cuda_smoke", lambda *_args, **_kwargs: {"passed": True})

    def start(*_args, **kwargs):
        calls.update(kwargs)
        return {"argv": ["--candidate-only"], "process": "running"}

    monkeypatch.setattr(queue.ops, "start_run", start)

    result = queue._launch(
        plan,
        tmp_path,
        run_id=plan["diagnostic"]["run_id"],
        seed=46,
        diagnostic=True,
    )

    assert result["status"] == "diagnostic_active"
    assert calls["competitions"] == [queue.COMPETITION_ID]
    assert calls["allow_concurrent_with_cpu_light"] is False
    assert calls["runner_contract_args"] == queue._runner_args(plan, diagnostic=True)
    assert result["start"]["argv"] == ["--candidate-only"]
