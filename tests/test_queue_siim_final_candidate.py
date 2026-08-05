from __future__ import annotations

import inspect
import json

from scripts import queue_siim_final_candidate as queue


def test_training_command_keeps_candidate_and_single_gpu_gates(tmp_path):
    plan = {
        "competition_id": "siim-isic-melanoma-classification",
        "planner": {
            "evidence": {"execution_plan": {"path": str(tmp_path / "gpt56.json")}},
        },
        "implementation": {"full_runner": {"path": str(tmp_path / "full.py")}},
        "training": {
            "python": "python.exe",
            "data_root": str(tmp_path / "data"),
            "output_root": str(tmp_path / "runs"),
            "allowed_root": str(tmp_path),
            "official_source_root": str(tmp_path / "mle-bench"),
            "run_id": "siim_candidate",
            "seed": 42,
            "backbone": "convnext_small",
            "secondary_backbone": "efficientnet_v2_s",
            "epochs": 8,
            "outer_folds": 5,
            "inner_folds": 3,
            "image_size": 384,
            "workers": 0,
            "learning_rate": 3e-4,
            "metadata_iterations": 700,
            "catboost_task_type": "GPU",
        },
    }
    command = queue.build_training_command(
        plan,
        selected_profile="robust_multiview_v1",
        selected_batch_size=8,
        ablation_report=str(tmp_path / "ablation.json"),
    )
    assert "--candidate-only" in command
    assert "--hold-cuda-lease" in command
    assert command[command.index("--siim-workers") + 1] == "0"
    assert command[command.index("--siim-batch-size") + 1] == "8"
    assert command[command.index("--siim-preprocessing-profile") + 1] == (
        "robust_multiview_v1"
    )
    assert "private_grade" not in " ".join(command)


def test_ablation_snapshot_binds_selected_profile_and_frozen_hashes(tmp_path):
    ablation_plan_path = tmp_path / "ablation_plan.json"
    report_path = tmp_path / "ablation.json"
    ablation_plan = {
        "training": {
            "run_id": "ablation_run",
            "evaluation_seeds": [40, 41, 42],
            "folds": 3,
            "profiles": ["raw_multiview_v1", "robust_multiview_v1"],
        }
    }
    ablation_plan_path.write_text(json.dumps(ablation_plan), encoding="utf-8")
    report = {
        "schema": "evomind.siim_preprocessing_ablation.v1",
        "competition_id": "siim-isic-melanoma-classification",
        "run_id": "ablation_run",
        "passed": True,
        "full_public_train_scope": True,
        "patient_content_group_isolation": True,
        "validation_coverage_exactly_once": True,
        "evaluation_seeds": [40, 41, 42],
        "fold_count": 3,
        "profile_order": ["raw_multiview_v1", "robust_multiview_v1"],
        "selected_profile": "robust_multiview_v1",
        "adapter_source_sha256": "a" * 64,
        "wave2_source_sha256": "b" * 64,
        "private_labels_used_for_training": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "official_score_claimed": False,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan = {
        "competition_id": "siim-isic-melanoma-classification",
        "serial_dependency": {
            "plan": {"path": str(ablation_plan_path)},
            "report": str(report_path),
        },
        "implementation": {
            "adapter": {"sha256": "a" * 64},
            "wave2": {"sha256": "b" * 64},
        },
    }
    snapshot = queue.ablation_snapshot(plan)
    assert snapshot["ready"] is True
    assert snapshot["selected_profile"] == "robust_multiview_v1"
    report["adapter_source_sha256"] = "c" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert queue.ablation_snapshot(plan)["ready"] is False


def test_queue_source_never_signals_or_invokes_official_actions():
    source = inspect.getsource(queue)
    assert "terminate(" not in source
    assert ".kill(" not in source
    assert "private_grade(" not in source
    assert '"process_signals_sent": 0' in source
    assert '"official_grader_executed": False' in source
    assert '"kaggle_submission_executed": False' in source


def test_parse_compute_apps_filters_malformed_rows():
    output = "123, C:\\Python\\python.exe\nnot-a-pid, bad\n456, chrome.exe\n"
    assert queue.parse_compute_apps(output) == [
        {"pid": 123, "process_name": "C:\\Python\\python.exe"},
        {"pid": 456, "process_name": "chrome.exe"},
    ]


def test_current_plan_uses_calibrated_wddm_gate():
    plan = queue.validate_frozen_plan(queue.DEFAULT_PLAN)
    assert plan["launch_contract"]["gpu_idle_gate_mode"] == (
        "calibrated_wddm_multimetric_v1"
    )
    policy = plan["_idle_policy"]
    baseline = {
        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "utilization_percent": 26,
        "memory_utilization_percent": 26,
        "memory_used_mib": 1246,
        "memory_free_mib": 6712,
        "temperature_c": 51,
        "pstate": "P8",
        "power_draw_w": 7.05,
        "sm_clock_mhz": 285,
        "compute_applications": [],
        "python_compute_applications": [],
    }
    assert queue.calibrated_idle.evaluate_idle(policy, baseline)["idle"] is True
