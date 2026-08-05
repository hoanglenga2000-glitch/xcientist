from __future__ import annotations

import json

from research_os.agent.llm_finetune_workflow import (
    CORRECTION_SPECS,
    HPC_TASK_OVERHEAD_SECONDS,
    LlmFinetuneExecutors,
    _correction_task_plan,
    _finalize_exhausted_rejected_review,
    _is_supported_hpc_gpu,
    _recover_legacy_rejected_review,
)
from research_os.agent.multi_agent import (
    AgentRoleSpec,
    AgentTask,
    MultiAgentStore,
    MultiAgentSupervisor,
    create_run,
)
from xsci.user_request import parse_user_request


def _role(name: str) -> AgentRoleSpec:
    return AgentRoleSpec(role=name, capabilities=(name,), output_contract=("evidence",))


def _request():
    return parse_user_request(
        "请用本地 EvoMind 文档微调 7B 大模型，在远程 A40 上训练，最多 20 分钟，不要使用本地 GPU，也不要发布模型。"
    )


def test_corrections_use_one_evidence_driven_variable_per_attempt(tmp_path):
    request = _request()
    run = create_run(
        objective=request.objective,
        tasks=[AgentTask("review", "review", "IndependentReviewer")],
        roles=[_role("IndependentReviewer")],
        run_id="run-correction-contract",
    )
    store = MultiAgentStore(tmp_path)
    executor = LlmFinetuneExecutors(
        workspace_root=tmp_path,
        run_dir=tmp_path,
        request=request,
        runtime=object(),
        store=store,
    )
    base = {
        "schema": "evomind.qlora_config.v1",
        "epochs": 2,
        "learning_rate": 0.0002,
        "seed": 20260722,
        "acceptance": {"domain_composite_improvement_pp": 5.0},
    }
    (tmp_path / "qlora_config.json").write_text(json.dumps(base), encoding="utf-8")

    executor.correction_design(
        AgentTask("correction_01_design", "design", "TrainingDesigner", payload={"correction_attempt": 1}),
        run,
        1,
    )
    correction_one = json.loads((tmp_path / "qlora_config.json").read_text(encoding="utf-8"))
    assert [key for key in correction_one if correction_one[key] != base.get(key)] == ["epochs"]
    assert correction_one["epochs"] == 3
    assert correction_one["learning_rate"] == 0.0002

    executor.correction_design(
        AgentTask("correction_02_design", "design", "TrainingDesigner", payload={"correction_attempt": 2}),
        run,
        2,
    )
    correction_two = json.loads((tmp_path / "qlora_config.json").read_text(encoding="utf-8"))
    assert [key for key in correction_two if correction_two[key] != correction_one.get(key)] == ["learning_rate"]
    assert correction_two["epochs"] == 3
    assert correction_two["learning_rate"] == 0.0001
    assert CORRECTION_SPECS[1]["field"] == "epochs"
    assert CORRECTION_SPECS[2]["field"] == "learning_rate"


def test_supported_hpc_gpu_contract_accepts_a40_and_a800_only():
    assert _is_supported_hpc_gpu("NVIDIA A40") is True
    assert _is_supported_hpc_gpu("NVIDIA A800-SXM4-80GB") is True
    assert _is_supported_hpc_gpu("NVIDIA RTX 4060 Laptop GPU") is False


def test_correction_plan_redirects_claim_audit_to_latest_reviewer():
    run = create_run(
        objective="correction plan",
        tasks=[
            AgentTask("independent_review", "review", "IndependentReviewer"),
            AgentTask("claim_audit", "audit", "ClaimAuditAgent", dependencies=("independent_review",)),
        ],
        roles=[
            _role("IndependentReviewer"),
            _role("ClaimAuditAgent"),
            _role("TrainingDesigner"),
            _role("HpcRuntimeAgent"),
            _role("EvaluatorAgent"),
        ],
        run_id="run-correction-plan",
    )

    tasks, overrides = _correction_task_plan(
        run=run,
        source_review_task_id="independent_review",
        correction_attempt=1,
        timeout_seconds=1200,
    )

    assert [task.task_id for task in tasks] == [
        "correction_01_design",
        "correction_01_hpc_train",
        "correction_01_evaluate",
        "correction_01_review",
    ]
    assert tasks[1].timeout_seconds == 1200
    assert overrides == {"claim_audit": ("correction_01_review",)}


def test_legacy_rejected_reviewer_is_migrated_once(tmp_path):
    request = _request()
    roles = [
        _role("IndependentReviewer"),
        _role("TrainingDesigner"),
        _role("HpcRuntimeAgent"),
        _role("EvaluatorAgent"),
        _role("ClaimAuditAgent"),
        _role("SynthesisAgent"),
    ]
    run = create_run(
        objective=request.objective,
        tasks=[
            AgentTask("independent_review", "review", "IndependentReviewer"),
            AgentTask("claim_audit", "audit", "ClaimAuditAgent", dependencies=("independent_review",)),
            AgentTask("synthesis", "synthesis", "SynthesisAgent", dependencies=("claim_audit",)),
        ],
        roles=roles,
        run_id="run-legacy-review",
    )
    run.tasks["independent_review"].status = "failed"
    run.tasks["independent_review"].error = "RuntimeError: review_rejected"
    run.tasks["independent_review"].idempotency_key = "legacy-review-key"
    run.tasks["claim_audit"].status = "skipped"
    run.tasks["synthesis"].status = "skipped"
    run.status = "needs_continuation"
    (tmp_path / "review.json").write_text(
        json.dumps(
            {"status": "rejected", "unresolved": ["domain_improvement"], "checks": {"domain_improvement": False}}
        ),
        encoding="utf-8",
    )
    store = MultiAgentStore(tmp_path)
    supervisor = MultiAgentSupervisor(run, store, {})

    assert (
        _recover_legacy_rejected_review(
            run=run,
            store=store,
            supervisor=supervisor,
            request=request,
            run_dir=tmp_path,
        )
        is True
    )

    assert run.tasks["independent_review"].status == "completed"
    assert run.tasks["independent_review"].error == ""
    assert run.tasks["claim_audit"].dependencies == ("correction_01_review",)
    assert run.tasks["correction_01_hpc_train"].timeout_seconds == 1200 + HPC_TASK_OVERHEAD_SECONDS
    assert (
        _recover_legacy_rejected_review(
            run=run,
            store=store,
            supervisor=supervisor,
            request=request,
            run_dir=tmp_path,
        )
        is False
    )


def test_exhausted_reviewer_rejection_becomes_stable_terminal_no_go(tmp_path):
    run = create_run(
        objective="bounded correction rejection",
        tasks=[
            AgentTask(
                "correction_02_review",
                "review",
                "IndependentReviewer",
                payload={"correction_attempt": 2},
            ),
            AgentTask(
                "claim_audit",
                "audit",
                "ClaimAuditAgent",
                dependencies=("correction_02_review",),
            ),
            AgentTask("synthesis", "synthesis", "SynthesisAgent", dependencies=("claim_audit",)),
        ],
        roles=[_role("IndependentReviewer"), _role("ClaimAuditAgent"), _role("SynthesisAgent")],
        run_id="run-exhausted-rejection",
    )
    run.tasks["correction_02_review"].status = "failed"
    run.tasks["correction_02_review"].error = "RuntimeError: review_rejected_after_max_corrections"
    run.tasks["correction_02_review"].idempotency_key = "review-key"
    run.tasks["claim_audit"].status = "skipped"
    run.tasks["synthesis"].status = "skipped"
    run.status = "needs_continuation"
    (tmp_path / "reviews").mkdir()
    (tmp_path / "reviews" / "correction_02_review.json").write_text(
        json.dumps(
            {
                "status": "rejected",
                "correction_attempt": 2,
                "next_action": "needs_continuation",
                "unresolved": ["domain_improvement"],
                "acceptance_contract": {"domain_composite_improvement_pp": 5.0},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "llm_output").mkdir()
    (tmp_path / "llm_output" / "metrics.json").write_text(
        json.dumps({"improvement_pp": 1.57, "local_gpu_used": False, "model_published": False}),
        encoding="utf-8",
    )
    (tmp_path / "llm_output" / "adapter_reload.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    store = MultiAgentStore(tmp_path)

    assert _finalize_exhausted_rejected_review(run=run, store=store, run_dir=tmp_path) is True
    assert run.status == "rejected"
    assert run.tasks["correction_02_review"].status == "completed"
    assert run.tasks["correction_02_review"].error == ""
    assert run.gates["claim_audit"] == "not_run_reviewer_rejected"
    assert run.next_action == "revise_data_or_training_strategy"
    assert run.open_requirements == ["domain_improvement: 1.570 pp < 5.000 pp after 2 corrections"]
    assert (tmp_path / "no_go_report.md").is_file()
    assert _finalize_exhausted_rejected_review(run=run, store=store, run_dir=tmp_path) is False
