from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from research_os.agent.llm_refinement_workflow import (
    create_refinement_plan,
    decide_refinement_plan,
    parse_refinement_changes,
    read_refinement_plan,
    reserve_refinement_run,
    run_llm_refinement,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _parent_run(tmp_path):
    run_id = "qwen7b_parent_reviewed"
    root = tmp_path / "workspace" / "evomind_runs" / run_id
    _write_json(
        root / "run.json",
        {
            "status": "completed",
            "run_id": run_id,
            "gates": {
                "reviewer": "passed",
                "claim_audit": "passed",
                "adapter_reload": "passed",
                "model_publication": "blocked",
            },
        },
    )
    _write_json(
        root / "request.json",
        {
            "task_type": "llm_finetune",
            "compute_policy": {"local_gpu_allowed": False},
        },
    )
    _write_json(root / "review.json", {"status": "passed", "checks": {"evidence": True}})
    _write_json(root / "claim_audit.json", {"status": "passed"})
    _write_json(
        root / "qlora_config.json",
        {
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "learning_rate": 0.0002,
            "epochs": 2,
            "acceptance": {"domain_composite_improvement_pp": 5.0},
        },
    )
    adapter = root / "llm_output" / "adapter" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_bytes(b"reviewed-parent-adapter")
    _write_json(root / "llm_output" / "adapter" / "adapter_config.json", {"r": 16})
    _write_json(root / "llm_output" / "adapter_reload.json", {"passed": True})
    _write_json(root / "llm_output" / "metrics.json", {"after": {"domain_composite": 80.0}})
    _write_json(root / "data" / "dataset_manifest.json", {"data_hash": "fixed"})
    required = [
        "data/dataset_manifest.json",
        "llm_output/adapter/adapter_config.json",
        "llm_output/adapter/adapter_model.safetensors",
        "llm_output/adapter_reload.json",
        "llm_output/metrics.json",
        "review.json",
        "claim_audit.json",
    ]
    _write_json(
        root / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "path": relative,
                    "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                }
                for relative in required
            ],
        },
    )
    return run_id, root, adapter


def test_natural_language_refinement_resolves_only_explicit_changes():
    changes = parse_refinement_changes(
        "验证曲线后期仍有波动。请降低学习率，继续训练一个短周期，并与上一轮结果对比。",
        {"learning_rate": 0.0002, "epochs": 2},
    )
    assert {item["field"] for item in changes} == {"learning_rate", "epochs"}
    assert next(item for item in changes if item["field"] == "learning_rate")["value"] == 0.0001
    assert next(item for item in changes if item["field"] == "epochs")["value"] == 1.0


def test_refinement_requires_human_gate_before_creating_child_run(tmp_path):
    parent_run_id, _parent_dir, _adapter = _parent_run(tmp_path)
    plan = create_refinement_plan(
        tmp_path,
        parent_run_id=parent_run_id,
        prompt="请降低学习率，继续训练一个短周期。",
        refinement_id="refine_gate_contract",
    )
    assert plan["status"] == "awaiting_human_gate"
    assert plan["gate"]["decision"] == "pending"

    with pytest.raises(ValueError, match="Human Gate"):
        run_llm_refinement(tmp_path, plan["refinement_id"], run_id="qwen7b_refine_blocked")
    assert not (tmp_path / "workspace" / "evomind_runs" / "qwen7b_refine_blocked").exists()


def test_gate_decision_is_durable_and_parent_artifact_is_unchanged(tmp_path):
    parent_run_id, _parent_dir, adapter = _parent_run(tmp_path)
    before = hashlib.sha256(adapter.read_bytes()).hexdigest()
    plan = create_refinement_plan(
        tmp_path,
        parent_run_id=parent_run_id,
        prompt="请降低学习率，继续训练一个短周期。",
        refinement_id="refine_parent_preserved",
    )
    approved = decide_refinement_plan(tmp_path, plan["refinement_id"], "approve")
    assert approved["status"] == "approved"
    assert approved["gate"]["decision"] == "approved"
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == before


def test_approve_and_child_run_reservation_are_idempotent(tmp_path):
    parent_run_id, _parent_dir, _adapter = _parent_run(tmp_path)
    plan = create_refinement_plan(
        tmp_path,
        parent_run_id=parent_run_id,
        prompt="请降低学习率，继续训练一个短周期。",
        refinement_id="refine_idempotent_reservation",
    )

    first_approval = decide_refinement_plan(tmp_path, plan["refinement_id"], "approve")
    second_approval = decide_refinement_plan(tmp_path, plan["refinement_id"], "approve")
    assert second_approval == first_approval

    first, should_start = reserve_refinement_run(
        tmp_path,
        plan["refinement_id"],
        "qwen7b_refine_reserved_one",
    )
    second, should_start_again = reserve_refinement_run(
        tmp_path,
        plan["refinement_id"],
        "qwen7b_refine_reserved_two",
    )
    assert should_start is True
    assert should_start_again is False
    assert first["child_run_id"] == "qwen7b_refine_reserved_one"
    assert second["child_run_id"] == "qwen7b_refine_reserved_one"
    assert second["launch_attempts"] == 1


def test_concurrent_reservation_launches_exactly_one_child(tmp_path):
    parent_run_id, _parent_dir, _adapter = _parent_run(tmp_path)
    plan = create_refinement_plan(
        tmp_path,
        parent_run_id=parent_run_id,
        prompt="请降低学习率，继续训练一个短周期。",
        refinement_id="refine_concurrent_reservation",
    )
    decide_refinement_plan(tmp_path, plan["refinement_id"], "approve")

    def reserve(index: int):
        return reserve_refinement_run(
            tmp_path,
            plan["refinement_id"],
            f"qwen7b_refine_concurrent_{index}",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, (1, 2)))

    assert sum(1 for _payload, should_start in results if should_start) == 1
    assert len({payload["child_run_id"] for payload, _should_start in results}) == 1


def test_unknown_refinement_does_not_silently_change_training(tmp_path):
    _parent_run(tmp_path)
    with pytest.raises(ValueError, match="no supported"):
        parse_refinement_changes("请把结果做得更好看。", {"learning_rate": 0.0002, "epochs": 2})


def test_refinement_rejects_parent_with_incomplete_reviewer_checks(tmp_path):
    parent_run_id, parent_dir, _adapter = _parent_run(tmp_path)
    _write_json(parent_dir / "review.json", {"status": "passed", "checks": {"evidence": False}})
    with pytest.raises(ValueError, match="Reviewer checks"):
        create_refinement_plan(
            tmp_path,
            parent_run_id=parent_run_id,
            prompt="请降低学习率，继续训练一个短周期。",
            refinement_id="refine_bad_reviewer",
        )


def test_refinement_rejects_parent_with_tampered_adapter(tmp_path):
    parent_run_id, _parent_dir, adapter = _parent_run(tmp_path)
    adapter.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash verification"):
        create_refinement_plan(
            tmp_path,
            parent_run_id=parent_run_id,
            prompt="请降低学习率，继续训练一个短周期。",
            refinement_id="refine_tampered_parent",
        )


def test_refinement_plan_reconciles_terminal_child_status(tmp_path):
    parent_run_id, _parent_dir, _adapter = _parent_run(tmp_path)
    plan = create_refinement_plan(
        tmp_path,
        parent_run_id=parent_run_id,
        prompt="请降低学习率，继续训练一个短周期。",
        refinement_id="refine_reconcile_child",
    )
    decide_refinement_plan(tmp_path, plan["refinement_id"], "approve")
    reserved, _should_start = reserve_refinement_run(
        tmp_path,
        plan["refinement_id"],
        "qwen7b_refine_reconcile_child",
    )
    child = tmp_path / "workspace" / "evomind_runs" / reserved["child_run_id"]
    _write_json(child / "run.json", {"run_id": child.name, "status": "needs_continuation"})

    reconciled = read_refinement_plan(tmp_path, plan["refinement_id"])

    assert reconciled["status"] == "needs_continuation"
    persisted = json.loads(
        (tmp_path / "workspace" / "tasks" / "evomind-qwen7b-finetune" / "refinements" / f"{plan['refinement_id']}.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "needs_continuation"
    _write_json(child / "refinement.json", {**reconciled, "status": "running"})
    stale_refinement = child / "refinement.json"
    _write_json(
        child / "artifact_manifest.json",
        {
            "schema": "evomind.llm_artifact_manifest.v1",
            "artifacts": [
                {
                    "path": "refinement.json",
                    "sha256": hashlib.sha256(stale_refinement.read_bytes()).hexdigest(),
                    "bytes": stale_refinement.stat().st_size,
                    "kind": "refinement.json",
                }
            ],
        },
    )

    read_refinement_plan(tmp_path, plan["refinement_id"])

    child_copy = json.loads((child / "refinement.json").read_text(encoding="utf-8"))
    assert child_copy["status"] == "needs_continuation"
    manifest = json.loads((child / "artifact_manifest.json").read_text(encoding="utf-8"))
    refinement_artifact = manifest["artifacts"][0]
    assert refinement_artifact["bytes"] == (child / "refinement.json").stat().st_size
    assert refinement_artifact["sha256"] == hashlib.sha256((child / "refinement.json").read_bytes()).hexdigest()
