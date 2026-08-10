from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from xsci.assistant_context import (
    build_assistant_context,
    render_architecture_summary,
    render_current_run_summary,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    run_id = "qwen_refine_test"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    _write_json(tmp_path / "workspace" / "current_run.json", {
        "schema": "evomind.current_run.v1",
        "task_id": "evomind-qwen7b-finetune",
        "run_id": run_id,
        "run_dir": f"workspace/evomind_runs/{run_id}",
        "status": "completed",
        "last_seq": 12,
    })
    _write_json(run_dir / "run.json", {
        "schema": "evomind.multi_agent.run.v1",
        "run_id": run_id,
        "status": "completed",
        "objective": "lower learning rate; password=must-not-leak",
        "seq": 12,
        "tasks": {
            "train": {"status": "completed"},
            "review": {"status": "completed"},
        },
        "gates": {"reviewer": "passed", "claim_audit": "passed"},
        "open_requirements": [],
        "next_action": "review_final_report",
        "roles": {"IndependentReviewer": {}, "SynthesisAgent": {}},
    })
    _write_json(run_dir / "request.json", {
        "task_type": "llm_finetune",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "training_method": "qlora_4bit",
        "compute_policy": {"backend": "hpc"},
    })
    _write_json(run_dir / "version.json", {"version": "V2", "parent_preserved": True})
    _write_json(run_dir / "version_comparison.json", {
        "metric": "fixed_test_domain_composite",
        "outcome": "no_material_change",
        "v1": 88.40,
        "v2": 88.43,
        "delta_pp": 0.03,
    })
    _write_json(run_dir / "llm_output" / "metrics.json", {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "training_method": "4-bit QLoRA",
        "local_gpu_used": False,
        "before": {"domain_composite": 55.43},
        "after": {"domain_composite": 88.43},
        "improvement_pp": 33.0,
    })
    _write_json(run_dir / "review.json", {"status": "passed", "unresolved": []})
    _write_json(run_dir / "claim_audit.json", {"status": "passed"})
    _write_json(run_dir / "hpc_llm_probe.json", {
        "gpu_inventory": [{"name": "NVIDIA A40"}],
        "torch": {"cuda_available": True},
        "user": "secret-user",
        "pwd": "test-secret-path",
    })
    _write_json(run_dir / "artifact_manifest.json", {
        "review_status": "passed",
        "claim_audit_status": "passed",
        "model_publication": "blocked",
    })
    for relative in ("model_card.md", "research_report.md"):
        path = run_dir / relative
        path.write_text("evidence", encoding="utf-8")
    _write_json(run_dir / "llm_output" / "adapter" / "adapter_model.safetensors", {})
    return tmp_path, run_dir


def test_context_packet_reads_current_run_and_redacts_sensitive_values(tmp_path: Path) -> None:
    root, _run_dir = _workspace(tmp_path)
    packet = build_assistant_context(root)

    assert packet.current_run["status"] == "completed"
    assert packet.current_run["version"] == "V2"
    assert packet.current_run["compute"]["gpu_names"] == ["NVIDIA A40"]
    assert packet.current_run["metrics"]["version_outcome"] == "no_material_change"
    assert packet.public_status()["current_task"] is True
    prompt = packet.prompt_block()
    assert "must-not-leak" not in prompt
    assert "secret-user" not in prompt
    assert "/secret/path" not in prompt


def test_context_packet_rejects_pointer_that_escapes_workspace(tmp_path: Path) -> None:
    _write_json(tmp_path / "workspace" / "current_run.json", {
        "schema": "evomind.current_run.v1",
        "task_id": "evomind-qwen7b-finetune",
        "run_id": "outside",
        "run_dir": "../outside",
    })
    packet = build_assistant_context(tmp_path)
    assert packet.current_run == {"available": False, "status": "invalid_pointer"}


def test_context_renderers_describe_real_architecture_and_version_result(tmp_path: Path) -> None:
    root, _run_dir = _workspace(tmp_path)
    packet = build_assistant_context(root)

    architecture = render_architecture_summary(packet)
    status = render_current_run_summary(packet)
    assert "Executive Supervisor" in architecture
    assert "Independent Reviewer" in architecture
    assert "V1=88.40" in status
    assert "V2=88.43" in status
    assert "no_material_change" in status
    assert "本地 GPU 未使用" in status


def test_flat_retrospective_memory_is_counted(tmp_path: Path) -> None:
    memory_path = tmp_path / "experiments" / "evolution" / "retrospective_memory.json"
    _write_json(memory_path, [
        {"task_type": "tabular", "failure_pattern": "oom", "dataset_profile": {"evidence_level": "failure"}},
        {
            "task_type": "llm_finetune",
            "reusable_strategy": "reuse reviewed adapter",
            "metric_delta": 1.0,
            "dataset_profile": {"evidence_level": "validated", "run_success": True, "promoted": True},
        },
    ])
    packet = build_assistant_context(tmp_path)
    assert packet.memory["record_count"] == 2
    assert packet.memory["validated_count"] == 1
    assert packet.memory["failure_count"] == 1


def test_context_exposes_only_verified_current_run_deliverables(tmp_path: Path) -> None:
    root, run_dir = _workspace(tmp_path)
    delivery = run_dir / "delivery"
    delivery.mkdir()
    report = delivery / "report.pdf"
    report.write_bytes(b"verified-report")
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    _write_json(run_dir / "deliverables.json", {
        "files": [{
            "name": "report.pdf",
            "path": "report.pdf",
            "bytes": report.stat().st_size,
            "sha256": digest,
            "download_url": "/api/download/report.pdf",
        }],
    })
    _write_json(run_dir / "artifact_manifest.json", {
        "review_status": "passed",
        "claim_audit_status": "passed",
        "model_publication": "blocked",
        "artifacts": [{
            "path": "delivery/report.pdf",
            "bytes": report.stat().st_size,
            "sha256": digest,
        }],
    })
    packet = build_assistant_context(root)
    deliverable = packet.current_run["artifacts"]["deliverables"][0]
    assert deliverable["verified"] is True
    assert deliverable["absolute_path"] == str(report.resolve())
    assert deliverable["workspace_relative_path"].endswith("delivery/report.pdf")
    assert deliverable["sha256"] == digest


def test_context_exposes_reviewed_literature_and_grouped_siim_validation(tmp_path: Path) -> None:
    task_id = "siim-isic-melanoma-classification"
    run_id = "evomind_siim_fixture"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    _write_json(tmp_path / "workspace" / "current_run.json", {
        "schema": "evomind.current_run.v1",
        "task_id": task_id,
        "run_id": run_id,
        "run_dir": f"workspace/evomind_runs/{run_id}",
        "status": "completed",
    })
    _write_json(run_dir / "run.json", {
        "run_id": run_id,
        "status": "completed",
        "tasks": {"data_audit": {"status": "completed"}},
        "gates": {},
    })
    _write_json(run_dir / "request.json", {"task_type": "image_classification"})
    _write_json(run_dir / "metrics.json", {
        "roc_auc": 0.9225357247684676,
        "pr_auc": 0.23970933285011597,
        "brier": 0.29524735217259324,
        "metric_scope": "independent_offline_patient_content_grouped_oof",
        "fold_roc_auc_mean": 0.922480488386402,
        "fold_roc_auc_std": 0.008603472136512058,
        "patient_grouped_bootstrap_roc_auc_95ci": {
            "lower": 0.9110109632922689,
            "upper": 0.9327846750376818,
            "valid_samples": 2000,
        },
    })
    _write_json(run_dir / "review.json", {
        "status": "review_passed",
        "review_scope": "public_train_grouped_oof_and_withheld_test_predictions",
        "checks": {
            "patient_group_overlap_zero": True,
            "content_group_overlap_zero": True,
            "oof_coverage_exactly_once": True,
        },
    })
    manifest_path = (
        tmp_path / "workspace" / "tasks" / task_id / "rag" / "context_2026-08-03T09-54-21-128Z.json"
    )
    _write_json(manifest_path, {
        "task_id": task_id,
        "query": "SIIM ISIC melanoma patient level cross validation",
        "context_path": f"workspace/tasks/{task_id}/rag/context_2026-08-03T09-54-21-128Z.md",
        "manifest_path": f"workspace/tasks/{task_id}/rag/{manifest_path.name}",
        "integrity": {"external_verified": 1, "fabricated": 0},
        "papers": [{
            "title": "Effect of patient-contextual skin images in melanoma diagnosis",
            "year": "2024",
            "source": "crossref",
            "doi": "10.1111/jdv.20479",
            "url": "https://doi.org/10.1111/jdv.20479",
        }],
    })
    _write_json(
        manifest_path.parent / "citation_audits" / "citation_audit_fixture.json",
        {
            "task_id": task_id,
            "status": "passed",
            "gate": "citation_gate_passed",
            "claim": "The challenge evaluated patient-contextual images for human and AI melanoma diagnosis.",
            "paper_id": "crossref-10.1111_jdv.20479",
            "conclusion": "The claim is eligible for report use with this citation and its current wording.",
        },
    )

    packet = build_assistant_context(tmp_path)

    assert packet.current_run["metrics"]["roc_auc"] == 0.9225357247684676
    assert packet.current_run["review"]["checks"]["patient_group_overlap_zero"] is True
    assert packet.evidence["literature"]["paper_count"] == 1
    assert packet.evidence["literature"]["papers"][0]["doi"] == "10.1111/jdv.20479"
    assert packet.evidence["citation_audits"][0]["status"] == "passed"
    assert "10.1111/jdv.20479" in packet.prompt_block()

    from xsci.kaggle_conversation import ConversationAgent

    prompt = "请基于当前唯一 SIIM Run，用三条要点列出 grouped OOF 的 ROC-AUC、PR-AUC、Brier，并注明不是 Kaggle 官方成绩。"
    answer = ConversationAgent(client=None).chat(
        prompt,
        SimpleNamespace(workspace_root=str(tmp_path), selected_task=task_id),
        context=packet,
    )
    assert "ROC-AUC=0.9225" in answer
    assert "PR-AUC=0.2397" in answer
    assert "Brier=0.2952" in answer
    assert "不是 Kaggle 官方成绩" in answer
