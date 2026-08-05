from __future__ import annotations

import copy

import pytest

from scripts import generate_taxi_source_audited_plan as generate


def test_taxi_multiseed_plan_is_bounded_serial_and_candidate_only() -> None:
    plan = generate.build_plan()

    assert plan["seeds"] == [43, 44, 45]
    assert len(set(plan["run_ids"])) == 3
    assert plan["serial_dependency"]["required_terminal_before_taxi"] is True
    assert plan["compute_gate"]["launch_when_gate_false"] is False
    assert plan["training_contract"]["duplicate_safe_oof"] is True
    assert plan["training_contract"]["fold_local_route_statistics"] is True
    assert plan["training_contract"]["full_data_refit"] is True
    assert plan["promotion_contract"]["all_three_external_seeds_required"] is True
    assert plan["promotion_contract"]["official_private_grader_approved"] is False
    assert plan["execution_contract"]["local_gpu_allowed"] is False
    assert plan["execution_contract"]["process_signals_allowed"] is False


def test_taxi_plan_rejects_stale_persisted_readiness(monkeypatch, tmp_path) -> None:
    live = generate.readiness_verifier.verify()
    stale = copy.deepcopy(live)
    stale["source"]["sha256"] = "0" * 64
    stale_path = tmp_path / "taxi_source_audit_readiness.json"
    stale_path.write_text(generate.json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(generate, "READINESS", stale_path)

    with pytest.raises(RuntimeError, match="persisted source readiness is stale"):
        generate.build_plan()
