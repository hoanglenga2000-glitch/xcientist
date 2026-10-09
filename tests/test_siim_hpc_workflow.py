from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_os.agent.multi_agent import AgentTask, validate_task_graph
from research_os.agent.siim_hpc_workflow import (
    ABLATION_SEEDS,
    DELIVERABLE_NAMES,
    FORMAL_SEEDS,
    FROZEN_FILES,
    OPTION_A_BUDGET_HOURS,
    OPTION_A_BUDGET_POLICY,
    ORIGINAL_BUDGET_HOURS,
    SiimEvidenceError,
    SiimHpcExecutors,
    build_siim_hpc_run,
    resume_siim_hpc_research,
    run_siim_hpc_research,
    stage_siim_ingress,
)
from xsci import multi_agent_cli
from xsci.user_request import parse_user_request

REQUEST = (
    "请使用 SIIM-ISIC Melanoma Classification 皮肤镜图像，在 HPC A800 上训练，"
    "按患者分组验证、独立复核，并生成报告和证据包；不提交公开榜单。"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "text",
    [
        "训练 SIIM 医疗影像模型",
        "分析 ISIC 皮肤镜图像并训练",
        "完成黑色素瘤影像分类实验",
        "Train a melanoma classifier from dermoscopy images.",
    ],
)
def test_siim_terms_select_the_locked_image_hpc_policy(text):
    request = parse_user_request(text)

    assert request.task_type == "image_classification"
    assert request.dataset == "siim-isic-melanoma-classification"
    assert request.compute_policy.backend == "hpc"
    assert request.compute_policy.remote_gpu_required is True
    assert request.compute_policy.local_gpu_allowed is False
    assert request.submission_policy.official_submission == "forbidden"
    assert request.submission_policy.create_candidate is True
    assert request.budget.max_minutes == 1440
    assert set(DELIVERABLE_NAMES).issubset(request.deliverables)


def test_siim_task_graph_is_the_fixed_nine_node_review_gated_chain():
    run = build_siim_hpc_run(parse_user_request(REQUEST), run_id="evomind_siim_graph_test")

    validate_task_graph(run.tasks, run.roles)
    assert list(run.tasks) == [
        "request_setup",
        "hpc_data_preflight",
        "data_audit",
        "research_design",
        "preprocessing_ablation",
        "full_training",
        "independent_review_freeze",
        "terminal_private_grader",
        "claim_audit_delivery",
    ]
    assert run.tasks["preprocessing_ablation"].dependencies == ("research_design",)
    assert run.tasks["full_training"].dependencies == ("preprocessing_ablation",)
    assert run.tasks["independent_review_freeze"].dependencies == ("full_training",)
    assert run.tasks["terminal_private_grader"].dependencies == ("independent_review_freeze",)
    assert run.tasks["claim_audit_delivery"].dependencies == ("terminal_private_grader",)
    assert run.tasks["preprocessing_ablation"].resource_type == "hpc_gpu"
    assert run.tasks["full_training"].resource_type == "hpc_gpu"
    assert run.gates["official_submission"] == "forbidden"
    assert run.gates["private_grader"] == "terminal_once"
    assert run.gates["post_grader_tuning"] == "forbidden"
    assert set(ABLATION_SEEDS).isdisjoint(FORMAL_SEEDS)


def test_missing_real_hpc_evidence_fails_closed_without_metrics_or_downloads(tmp_path):
    run_id = "evomind_siim_missing_evidence"

    result = run_siim_hpc_research(tmp_path, parse_user_request(REQUEST), run_id=run_id)

    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    assert result.status == "needs_continuation"
    assert result.tasks["request_setup"].status == "completed"
    assert result.tasks["hpc_data_preflight"].status == "failed"
    assert result.tasks["data_audit"].status == "skipped"
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "private_grader.json").exists()
    assert not (run_dir / "artifact_manifest.json").exists()
    snapshot = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert snapshot["dataset_profile"]["status"] == "awaiting_verified_evidence"
    assert snapshot["hpc_runtime"]["status"] == "awaiting_verified_evidence"
    assert snapshot["deliverables"]["status"] == "unavailable"
    assert all(item["sha256"] is None for item in snapshot["deliverables"]["files"])
    workflow_contract = json.loads((run_dir / "workflow_contract.json").read_text(encoding="utf-8"))
    assert workflow_contract["budget_hours"] == ORIGINAL_BUDGET_HOURS
    assert workflow_contract["budget_policy"] == "original_24_hour_template"


def test_corrected_campaign_budget_is_frozen_into_workflow_and_research_design(tmp_path):
    run_id = "evomind_siim_corrected_budget"
    request = parse_user_request(REQUEST)
    run = build_siim_hpc_run(request, run_id=run_id)
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    campaign_plan = _write_json(
        tmp_path / "workspace" / "hpc" / "job89508_siim_campaign" / run_id / "campaign_plan.json",
        {
            "run_id": run_id,
            "budget_hours": OPTION_A_BUDGET_HOURS,
            "original_budget_hours": ORIGINAL_BUDGET_HOURS,
            "budget_policy": OPTION_A_BUDGET_POLICY,
        },
    )
    executors = SiimHpcExecutors(workspace_root=tmp_path, run_dir=run_dir, request=request)
    executors.request_setup(AgentTask("request_setup", "setup", "RequestPlanner"), None, run)  # type: ignore[arg-type]
    _write_json(run_dir / "data_audit.json", {"run_id": run_id, "status": "passed"})
    result = executors.design(AgentTask("research_design", "design", "ResearchDesigner"), None, run)  # type: ignore[arg-type]

    workflow_contract = json.loads((run_dir / "workflow_contract.json").read_text(encoding="utf-8"))
    research_design = json.loads((run_dir / "research_design.json").read_text(encoding="utf-8"))
    assert workflow_contract["budget_hours"] == OPTION_A_BUDGET_HOURS
    assert workflow_contract["budget_policy"] == OPTION_A_BUDGET_POLICY
    assert workflow_contract["budget_source_sha256"] == _sha256(campaign_plan)
    assert research_design["budget_hours"] == {
        "ablation": 4,
        "formal_training": 72,
        "review_grade_delivery": 2,
        "total": 78,
    }
    assert research_design["budget_policy"] == OPTION_A_BUDGET_POLICY
    assert "78-hour" in result.conclusion


def test_corrected_campaign_budget_fails_closed_if_provenance_drifted(tmp_path):
    run_id = "evomind_siim_budget_provenance_drift"
    request = parse_user_request(REQUEST)
    run = build_siim_hpc_run(request, run_id=run_id)
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    _write_json(
        tmp_path / "workspace" / "hpc" / "job89508_siim_campaign" / run_id / "campaign_plan.json",
        {
            "run_id": run_id,
            "budget_hours": {**OPTION_A_BUDGET_HOURS, "formal_training": 71, "total": 77},
            "original_budget_hours": ORIGINAL_BUDGET_HOURS,
            "budget_policy": OPTION_A_BUDGET_POLICY,
        },
    )
    executors = SiimHpcExecutors(workspace_root=tmp_path, run_dir=run_dir, request=request)

    with pytest.raises(SiimEvidenceError, match="corrected option-A"):
        executors.request_setup(AgentTask("request_setup", "setup", "RequestPlanner"), None, run)  # type: ignore[arg-type]


def test_resume_ingests_same_run_preflight_then_stops_at_next_missing_gate(tmp_path):
    run_id = "evomind_siim_resume_preflight"
    run_siim_hpc_research(tmp_path, parse_user_request(REQUEST), run_id=run_id)
    evidence = tmp_path / "collected"
    runtime = _write_json(
        evidence / "hpc_preflight.json",
        {
            "schema": "evomind.siim.hpc_preflight.v1",
            "run_id": run_id,
            "status": "passed",
            "job_id": "89508",
            "credential_profile": "job89508",
            "remote_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
            "gpu": {"name": "NVIDIA A800-SXM4-80GB", "memory_total_mb": 81920},
            "samples": [
                {"free_memory_mb": 70_000, "other_process_memory_mb": 0, "gpu_utilization": 0} for _ in range(5)
            ],
            "other_processes_modified": False,
            "signals_sent": 0,
            "launch_decision": "go",
        },
    )
    dataset = _write_json(
        evidence / "dataset_profile.json",
        {
            "schema": "evomind.siim.dataset_profile.v1",
            "run_id": run_id,
            "status": "passed",
            "dataset": "siim-isic-melanoma-classification",
            "counts": {
                "files": 33129,
                "bytes": 25765345055,
                "train_images": 28984,
                "test_images": 4142,
                "positive_rows": 513,
                "patients": 2056,
            },
            "manifest_sha256": "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13",
            "complete": True,
        },
    )
    stage_siim_ingress(
        tmp_path,
        run_id,
        {
            "hpc_preflight.json": runtime,
            "dataset_profile.json": dataset,
        },
    )

    result = resume_siim_hpc_research(tmp_path, run_id)

    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    assert result.status == "needs_continuation"
    assert result.tasks["hpc_data_preflight"].status == "completed"
    assert result.tasks["data_audit"].status == "failed"
    snapshot = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert snapshot["dataset_profile"]["counts"]["train_images"] == 28984
    assert snapshot["hpc_runtime"]["gpu"]["name"] == "NVIDIA A800-SXM4-80GB"
    assert snapshot["historical_thresholds"]["historical_private_score"] == 0.92165


def test_terminal_private_grader_is_recorded_once_and_freeze_drift_is_rejected(tmp_path):
    run_id = "evomind_siim_grader_once"
    request = parse_user_request(REQUEST)
    run = build_siim_hpc_run(request, run_id=run_id)
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    frozen_file = _write_json(run_dir / "metrics.json", {"run_id": run_id, "roc_auc": 0.91})
    freeze = _write_json(
        run_dir / "candidate_freeze.json",
        {
            "schema": "evomind.siim.candidate_freeze.v1",
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "frozen_at": "2026-07-29T14:00:00+00:00",
            "configuration_sha256": "a" * 64,
            "artifacts": [
                {"path": "metrics.json", "sha256": _sha256(frozen_file), "bytes": frozen_file.stat().st_size}
            ],
            "private_grader_execution_count_before_freeze": 0,
            "tuning_closed": True,
        },
    )
    grader = _write_json(
        run_dir / "ingress" / "private_grader.json",
        {
            "schema": "evomind.siim.private_grader.v1",
            "run_id": run_id,
            "status": "passed",
            "execution_id": "terminal-once-fixture",
            "execution_index": 1,
            "candidate_freeze_sha256": _sha256(freeze),
            "executed_after_freeze": True,
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
            "mle_private_grader_score": 0.923,
        },
    )
    executors = SiimHpcExecutors(workspace_root=tmp_path, run_dir=run_dir, request=request)
    task = AgentTask("terminal_private_grader", "grade", "TerminalGrader")

    first = executors.terminal_grader(task, None, run)  # type: ignore[arg-type]
    second = executors.terminal_grader(task, None, run)  # type: ignore[arg-type]

    assert first.metrics["mle_private_grader_score"] == 0.923
    assert second.metrics["mle_private_grader_score"] == 0.923
    ledger = json.loads((run_dir / "private_grader_ledger.json").read_text(encoding="utf-8"))
    assert ledger["execution_count"] == 1
    assert ledger["result_sha256"] == _sha256(grader)
    frozen_file.write_text('{"run_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(SiimEvidenceError, match="frozen artifact changed"):
        executors.terminal_grader(task, None, run)  # type: ignore[arg-type]


def test_terminal_private_grader_reconciles_output_written_before_ledger(tmp_path):
    run_id = "evomind_siim_grader_reconcile"
    request = parse_user_request(REQUEST)
    run = build_siim_hpc_run(request, run_id=run_id)
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    frozen_file = _write_json(run_dir / "metrics.json", {"run_id": run_id, "roc_auc": 0.92})
    freeze = _write_json(
        run_dir / "candidate_freeze.json",
        {
            "schema": "evomind.siim.candidate_freeze.v1",
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "configuration_sha256": "b" * 64,
            "artifacts": [
                {"path": "metrics.json", "sha256": _sha256(frozen_file), "bytes": frozen_file.stat().st_size}
            ],
            "private_grader_execution_count_before_freeze": 0,
            "tuning_closed": True,
        },
    )
    ingress = _write_json(
        run_dir / "ingress" / "private_grader.json",
        {
            "schema": "evomind.siim.private_grader.v1",
            "run_id": run_id,
            "status": "passed",
            "execution_id": "terminal-reconcile-fixture",
            "execution_index": 1,
            "candidate_freeze_sha256": _sha256(freeze),
            "executed_after_freeze": True,
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
            "mle_private_grader_score": 0.924,
        },
    )
    (run_dir / "private_grader.json").write_bytes(ingress.read_bytes())
    (run_dir / "private_grader.lock").write_text(
        json.dumps({"pid": 999999, "run_id": run_id, "result_sha256": _sha256(ingress)}),
        encoding="utf-8",
    )
    executors = SiimHpcExecutors(workspace_root=tmp_path, run_dir=run_dir, request=request)
    task = AgentTask("terminal_private_grader", "grade", "TerminalGrader")

    result = executors.terminal_grader(task, None, run)  # type: ignore[arg-type]

    assert result.metrics["mle_private_grader_score"] == 0.924
    ledger = json.loads((run_dir / "private_grader_ledger.json").read_text(encoding="utf-8"))
    assert ledger["execution_count"] == 1
    assert ledger["result_sha256"] == _sha256(ingress)
    assert not (run_dir / "private_grader.lock").exists()


def test_terminal_private_grader_rejects_stale_lock_without_registered_output(tmp_path):
    run_id = "evomind_siim_grader_stale_lock"
    request = parse_user_request(REQUEST)
    run = build_siim_hpc_run(request, run_id=run_id)
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    frozen_file = _write_json(run_dir / "metrics.json", {"run_id": run_id})
    freeze = _write_json(
        run_dir / "candidate_freeze.json",
        {
            "schema": "evomind.siim.candidate_freeze.v1",
            "run_id": run_id,
            "status": "frozen_before_private_grader",
            "configuration_sha256": "c" * 64,
            "artifacts": [
                {"path": "metrics.json", "sha256": _sha256(frozen_file), "bytes": frozen_file.stat().st_size}
            ],
            "tuning_closed": True,
        },
    )
    _write_json(
        run_dir / "ingress" / "private_grader.json",
        {
            "run_id": run_id,
            "status": "passed",
            "execution_index": 1,
            "candidate_freeze_sha256": _sha256(freeze),
            "executed_after_freeze": True,
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
            "score": 0.9,
        },
    )
    (run_dir / "private_grader.lock").write_text("{}", encoding="utf-8")
    executors = SiimHpcExecutors(workspace_root=tmp_path, run_dir=run_dir, request=request)
    task = AgentTask("terminal_private_grader", "grade", "TerminalGrader")

    with pytest.raises(SiimEvidenceError, match="already in progress"):
        executors.terminal_grader(task, None, run)  # type: ignore[arg-type]

    assert not (run_dir / "private_grader_ledger.json").exists()


def test_ingress_target_cannot_escape_the_run(tmp_path):
    run_id = "evomind_siim_ingress_guard"
    run_siim_hpc_research(tmp_path, parse_user_request(REQUEST), run_id=run_id)
    source = _write_json(tmp_path / "evidence.json", {"run_id": run_id})

    with pytest.raises(ValueError, match="invalid SIIM ingress target"):
        stage_siim_ingress(tmp_path, run_id, {"../escape.json": source})


def _completed(run_id: str):
    return SimpleNamespace(status="completed", run_id=run_id, seq=9, open_requirements=[])


def test_cli_dispatches_and_resumes_the_siim_workflow(tmp_path, monkeypatch):
    request_path = tmp_path / "request.txt"
    request_path.write_text(REQUEST, encoding="utf-8")
    run_calls = []
    resume_calls = []
    monkeypatch.setattr(multi_agent_cli, "active_root", lambda: tmp_path)
    monkeypatch.setattr(
        multi_agent_cli,
        "run_siim_hpc_research",
        lambda root, request, *, run_id: run_calls.append((root, request, run_id)) or _completed(run_id),
    )
    exit_code = multi_agent_cli.main(
        [
            "run",
            "--request-file",
            str(request_path),
                "--run-id",
                "evomind_siim_cli_dispatch",
                "--hpc-job-id",
                "90353",
                "--hpc-credential-profile",
                "job90353",
                "--hpc-resource-profile",
                "aimslab_a800_80gb",
                "--execution-backend",
                "hpc",
            ]
        )
    assert exit_code == 0
    assert run_calls[0][1].task_type == "image_classification"
    assert run_calls[0][1].compute_policy.job_id == 90353
    assert run_calls[0][1].compute_policy.credential_profile == "job90353"
    assert run_calls[0][1].compute_policy.resource_profile == "aimslab_a800_80gb"
    assert run_calls[0][1].compute_policy.backend == "hpc"

    run_id = "evomind_siim_cli_resume"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "request.json", parse_user_request(REQUEST).to_dict())
    monkeypatch.setattr(
        multi_agent_cli,
        "resume_siim_hpc_research",
        lambda root, selected_run_id: resume_calls.append((root, selected_run_id)) or _completed(selected_run_id),
    )
    exit_code = multi_agent_cli.main(["resume", "--run-id", run_id])
    assert exit_code == 0
    assert resume_calls == [(tmp_path, run_id)]


def test_complete_ingress_contract_reaches_four_hash_bound_deliverables(tmp_path):
    run_id = "evomind_siim_complete_contract"
    request = parse_user_request(REQUEST)
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    source = tmp_path / "collected"
    result = run_siim_hpc_research(tmp_path, request, run_id=run_id)
    assert result.status == "needs_continuation"

    preflight = _write_json(
        source / "hpc_preflight.json",
        {
            "run_id": run_id,
            "status": "passed",
            "job_id": "89508",
            "credential_profile": "job89508",
            "remote_root": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
            "gpu": {"name": "NVIDIA A800-SXM4-80GB", "memory_total_mb": 81920},
            "samples": [{"free_memory_mb": 70_000, "other_process_memory_mb": 0} for _ in range(5)],
            "other_processes_modified": False,
            "signals_sent": 0,
            "launch_decision": "go",
        },
    )
    dataset = _write_json(
        source / "dataset_profile.json",
        {
            "run_id": run_id,
            "status": "passed",
            "dataset": "siim-isic-melanoma-classification",
            "counts": {
                "files": 33129,
                "bytes": 25765345055,
                "train_images": 28984,
                "test_images": 4142,
                "positive_rows": 513,
                "patients": 2056,
            },
            "manifest_sha256": "8fef392368fcc433f4532129312f3b2ef7118d76cad04dc70f694246c34eff13",
            "complete": True,
        },
    )
    audit = _write_json(
        source / "data_audit.json",
        {
            "run_id": run_id,
            "status": "passed",
            "train_rows": 28984,
            "test_rows": 4142,
            "positive_rows": 513,
            "patients": 2056,
            "patient_group_overlap": 0,
            "content_group_overlap": 0,
            "target_in_test_features": False,
            "private_label_access_count": 0,
            "split_policy": "patient_and_content_grouped",
        },
    )
    stage_siim_ingress(
        tmp_path,
        run_id,
        {
            "hpc_preflight.json": preflight,
            "dataset_profile.json": dataset,
            "data_audit.json": audit,
        },
    )
    result = resume_siim_hpc_research(tmp_path, run_id)
    assert result.tasks["research_design"].status == "completed"
    assert result.tasks["preprocessing_ablation"].status == "failed"

    ablation = _write_json(
        source / "preprocessing_ablation.json",
        {
            "run_id": run_id,
            "status": "passed",
            "seeds": [40, 41, 42],
            "profiles": [
                {"profile": name, "mean_roc_auc": 0.9}
                for name in (
                    "raw_multiview",
                    "border_removal",
                    "color_constancy",
                    "hair_suppression",
                    "robust_combined_pipeline",
                )
            ],
            "patient_group_overlap": 0,
            "content_group_overlap": 0,
            "selected_profile": "raw_multiview",
            "decision": {"mean_gain": 0.0, "worst_fold_delta": 0.0, "seed_passes": 3},
            "rejected_profiles": ["border_removal"],
        },
    )
    stage_siim_ingress(tmp_path, run_id, {"preprocessing_ablation.json": ablation})
    result = resume_siim_hpc_research(tmp_path, run_id)
    assert result.tasks["preprocessing_ablation"].status == "completed"
    assert result.tasks["full_training"].status == "failed"

    training = source / "training"
    metrics = _write_json(
        training / "metrics.json",
        {
            "run_id": run_id,
            "roc_auc": 0.91,
            "pr_auc": 0.42,
            "brier": 0.02,
            "mle_private_grader_score": None,
        },
    )
    fold_metrics = _write_csv(
        training / "fold_metrics.csv",
        ["fold", "roc_auc"],
        [{"fold": fold, "roc_auc": 0.9 + fold / 100} for fold in range(5)],
    )
    oof = _write_csv(
        training / "oof_predictions.csv",
        ["image_name", "probability", "fold", "coverage"],
        [
            {
                "image_name": f"train_{index:05d}",
                "probability": 0.01 + (index % 90) / 100,
                "fold": index % 5,
                "coverage": 1,
            }
            for index in range(28984)
        ],
    )
    test_rows = [{"image_name": f"test_{index:05d}", "target": 0.1 + (index % 80) / 100} for index in range(4142)]
    submission = _write_csv(training / "submission.csv", ["image_name", "target"], test_rows)
    sample = _write_csv(
        training / "sample_submission.csv",
        ["image_name", "target"],
        [{"image_name": row["image_name"], "target": 0} for row in test_rows],
    )
    history = _write_json(training / "training_history.json", {"run_id": run_id, "outer_folds": 5})
    telemetry = training / "hpc_telemetry.jsonl"
    telemetry.write_text(
        "".join(
            json.dumps(
                {
                    "run_id": run_id,
                    "job_id": 89508,
                    "credential_profile": "job89508",
                    "gpu": "A800",
                    "seq": seq,
                }
            )
            + "\n"
            for seq in range(3)
        ),
        encoding="utf-8",
    )
    training_result = _write_json(
        training / "training_result.json",
        {
            "run_id": run_id,
            "status": "completed",
            "formal_seeds": [43],
            "official_submission_executed": False,
            "private_grader_execution_count": 0,
            "private_label_access_count": 0,
            "selected_batch_size": 96,
        },
    )
    stage_siim_ingress(
        tmp_path,
        run_id,
        {
            "training/metrics.json": metrics,
            "training/fold_metrics.csv": fold_metrics,
            "training/oof_predictions.csv": oof,
            "training/submission.csv": submission,
            "training/sample_submission.csv": sample,
            "training/training_history.json": history,
            "training/hpc_telemetry.jsonl": telemetry,
            "training/training_result.json": training_result,
        },
    )
    result = resume_siim_hpc_research(tmp_path, run_id)
    assert result.tasks["full_training"].status == "completed"
    assert result.tasks["independent_review_freeze"].status == "failed"
    runtime = json.loads((run_dir / "hpc_runtime.json").read_text(encoding="utf-8"))
    assert runtime["training_progress"] == {
        "status": "completed",
        "completed_formal_seeds": 1,
        "total_formal_seeds": 3,
        "completed_outer_folds": 5,
        "total_outer_folds": 5,
        "selected_epoch_min": None,
        "selected_epoch_max": None,
        "peak_model_memory_allocated_mib": None,
    }
    assert runtime["telemetry_summary"]["sample_count"] == 3
    assert runtime["telemetry_summary"]["max_other_process_memory_mib"] == 0
    assert runtime["telemetry_summary"]["hold_sample_count"] == 0

    review = _write_json(
        source / "review.json",
        {
            "run_id": run_id,
            "status": "passed",
            "checks": {
                name: True
                for name in (
                    "patient_group_overlap_zero",
                    "content_group_overlap_zero",
                    "oof_coverage_exactly_once",
                    "submission_schema_and_order",
                    "private_labels_unavailable_during_training",
                    "private_grader_not_executed",
                    "official_submission_not_executed",
                )
            },
            "artifact_hashes": {name: _sha256(run_dir / name) for name in FROZEN_FILES},
        },
    )
    stage_siim_ingress(tmp_path, run_id, {"review.json": review})
    result = resume_siim_hpc_research(tmp_path, run_id)
    assert result.tasks["independent_review_freeze"].status == "completed"
    assert result.tasks["terminal_private_grader"].status == "failed"

    grader = _write_json(
        source / "private_grader.json",
        {
            "run_id": run_id,
            "status": "passed",
            "execution_id": "terminal-once-contract-test",
            "execution_index": 1,
            "candidate_freeze_sha256": _sha256(run_dir / "candidate_freeze.json"),
            "executed_after_freeze": True,
            "feedback_used_for_tuning": False,
            "official_submission_executed": False,
            "mle_private_grader_score": 0.921,
        },
    )
    stage_siim_ingress(tmp_path, run_id, {"private_grader.json": grader})
    result = resume_siim_hpc_research(tmp_path, run_id)
    assert result.tasks["terminal_private_grader"].status == "completed"
    assert result.tasks["claim_audit_delivery"].status == "failed"

    claim = _write_json(
        source / "claim_audit.json",
        {
            "run_id": run_id,
            "status": "passed",
            "checks": {
                name: True
                for name in (
                    "no_public_leaderboard_claim",
                    "no_official_medal_claim",
                    "no_clinical_diagnosis_claim",
                    "private_grader_not_used_for_tuning",
                    "candidate_hashes_unchanged",
                    "official_submission_not_executed",
                )
            },
        },
    )
    deliverables = source / "deliverables"
    html = deliverables / "research_report.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("<!doctype html><title>SIIM contract report</title>", encoding="utf-8")
    pdf = deliverables / "evomind-siim-isic-report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% contract fixture\n%%EOF\n")
    results_csv = _write_csv(deliverables / "evomind-siim-isic-results.csv", ["image_name", "target"], test_rows)
    code_zip = deliverables / "evomind-siim-isic-code.zip"
    with zipfile.ZipFile(code_zip, "w") as archive:
        archive.writestr("src/train.py", "print('fixture')\n")
    evidence_zip = deliverables / "evomind-siim-isic-evidence.zip"
    with zipfile.ZipFile(evidence_zip, "w") as archive:
        archive.writestr("metrics.json", "{}\n")
        archive.writestr("review.json", "{}\n")
        archive.writestr("claim_audit.json", "{}\n")
    stage_siim_ingress(
        tmp_path,
        run_id,
        {
            "claim_audit.json": claim,
            "deliverables/research_report.html": html,
            "deliverables/evomind-siim-isic-report.pdf": pdf,
            "deliverables/evomind-siim-isic-results.csv": results_csv,
            "deliverables/evomind-siim-isic-code.zip": code_zip,
            "deliverables/evomind-siim-isic-evidence.zip": evidence_zip,
        },
    )

    result = resume_siim_hpc_research(tmp_path, run_id)

    assert result.status == "completed"
    snapshot = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert snapshot["deliverables"]["status"] == "ready"
    assert len(snapshot["deliverables"]["files"]) == 4
    assert all(item["sha256"] and item["download_url"] for item in snapshot["deliverables"]["files"])
    assert json.loads((run_dir / "private_grader_ledger.json").read_text(encoding="utf-8"))["execution_count"] == 1
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert {item["path"] for item in manifest["artifacts"]}.issuperset(DELIVERABLE_NAMES)
