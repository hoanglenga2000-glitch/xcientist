from __future__ import annotations

import json
from pathlib import Path

from scripts import queue_hpc88240_taxi_after_may as queue


def test_current_taxi_plan_is_frozen_and_candidate_only() -> None:
    plan = queue.validate_plan()

    assert plan["seeds"] == [43, 44, 45]
    assert plan["execution_contract"]["candidate_only"] is True
    assert plan["execution_contract"]["official_grader_executed"] is False
    assert plan["execution_contract"]["verified_precomputed_cache_required"] is True
    assert plan["execution_contract"]["verified_route_stat_cache_required"] is True
    assert plan["execution_contract"]["gpu_route_stat_rebuild_forbidden"] is True
    assert plan["public_precomputed_cache"]["cache_seed"] == 42
    assert plan["verified_route_stat_sidecar"]["cache_seed"] == 42
    assert "--taxi-require-precomputed-cache" in plan["runner_contract_args"]
    assert "--taxi-route-stat-cache-dir" in plan["runner_contract_args"]
    assert "--taxi-require-route-stat-cache" in plan["runner_contract_args"]
    assert plan["_bundle_verification"]["passed"] is True


def test_waiting_may_never_samples_or_launches(tmp_path: Path, monkeypatch) -> None:
    plan = queue.validate_plan()
    may = tmp_path / "may.json"
    may.write_text(
        json.dumps(
            {
                "status": "waiting_for_cactus_terminal",
                "plan_sha256": plan["serial_dependency"]["plan"]["sha256"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        queue.ops,
        "sample_gpu_idle_gate",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("gate called")),
    )

    result = queue.run_once(plan, may_status_path=may, evidence_dir=tmp_path)

    assert result["status"] == "waiting_for_may_multiseed_terminal"


def test_terminal_may_and_busy_gpu_waits_without_launch(tmp_path: Path, monkeypatch) -> None:
    plan = queue.validate_plan()
    may = tmp_path / "may.json"
    may.write_text(
        json.dumps(
            {
                "status": "all_seeds_terminal",
                "plan_sha256": plan["serial_dependency"]["plan"]["sha256"],
                "completed_seeds": [
                    {"seed": seed, "run_id": run_id}
                    for seed, run_id in zip(
                        plan["serial_dependency"]["seeds"],
                        plan["serial_dependency"]["run_ids"],
                        strict=True,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = {"status": 0}

    def status(run_id: str):
        calls["status"] += 1
        raise queue.ops.RemoteOpsError("not found")

    monkeypatch.setattr(queue.ops, "read_remote_status", status)
    monkeypatch.setattr(queue.ops, "sample_gpu_idle_gate", lambda **kwargs: {"passed": False})
    monkeypatch.setattr(
        queue.ops,
        "start_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("launch called")),
    )

    result = queue.run_once(plan, may_status_path=may, evidence_dir=tmp_path)

    assert result["status"] == "waiting_for_gpu_idle"
    assert result["next_seed"] == 43
    assert calls["status"] == 1
