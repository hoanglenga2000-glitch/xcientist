from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "queue_leaf_after_siim.py"


def load_module():
    spec = importlib.util.spec_from_file_location("queue_leaf_after_siim", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def complete_staging(plan: dict) -> dict:
    expected = plan["data_contract"]
    return {
        "schema": "evomind.leaf.public_staging.v1",
        "competition_id": "leaf-classification",
        "status": "size_verified_complete",
        "completed_files": expected["file_count"],
        "total_files": expected["file_count"],
        "completed_bytes": expected["total_bytes"],
        "total_bytes": expected["total_bytes"],
        "inventory_manifest_sha256": expected["manifest_sha256"],
        "errors": [],
        "private_paths_requested": False,
        "remote_writes_performed": False,
        "process_signals_sent": 0,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def complete_siim(plan: dict) -> dict:
    siim = plan["_siim_plan"]
    training = siim["training"]
    return {
        "schema": "evomind.siim_preprocessing_ablation.v1",
        "competition_id": "siim-isic-melanoma-classification",
        "run_id": plan["serial_dependency"]["siim_run_id"],
        "passed": True,
        "full_public_train_scope": True,
        "evaluation_seeds": training["evaluation_seeds"],
        "fold_count": training["folds"],
        "profile_order": training["profiles"],
        "selected_profile": training["profiles"][0],
        "adapter_source_sha256": siim["implementation"]["adapter"]["sha256"],
        "wave2_source_sha256": siim["implementation"]["wave2"]["sha256"],
        "private_labels_used_for_training": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "official_score_claimed": False,
    }


def complete_siim_final(plan: dict) -> dict:
    dependency = plan["serial_dependency"]["siim_final"]
    return {
        "schema": "evomind.siim.final_candidate_completion_watcher.v1",
        "run_id": dependency["run_id"],
        "status": "promotion_gate_failed",
        "plan_sha256": dependency["plan"]["sha256"],
        "candidate_ready": False,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def test_current_frozen_plan_and_every_declared_hash_are_verified() -> None:
    module = load_module()
    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)

    assert plan["planner"]["requested_model"] == "gpt-5.6-sol"
    assert plan["planner"]["served_model"] == "gpt-5.6-sol"
    assert plan["training"]["seeds"] == [40, 41, 42]
    assert plan["training"]["backbones"] == [
        "convnext_small",
        "efficientnet_v2_s",
    ]
    assert {value["name"] for value in plan["_validated_artifacts"]} == {
        "planner.source",
        "data_contract.inventory",
        "implementation.runner",
        "implementation.preflight",
        "implementation.stager",
        "implementation.queue",
        "implementation.idle_gate",
        "serial_dependency.siim_plan",
        "serial_dependency.siim_final_plan",
    }
    assert plan["launch_contract"]["gpu_idle_gate_mode"] == (
        "calibrated_wddm_multimetric_v1"
    )


def test_prerequisites_require_complete_leaf_data_and_siim_report(
    tmp_path: Path,
) -> None:
    module = load_module()
    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)
    staging = tmp_path / "leaf_staging.json"
    siim = tmp_path / "siim_report.json"
    siim_final = tmp_path / "siim_final_watcher.json"
    write_json(staging, complete_staging(plan))
    write_json(siim, complete_siim(plan))
    write_json(siim_final, complete_siim_final(plan))

    snapshot = module.prerequisite_snapshot(
        plan,
        staging_report_path=staging,
        siim_report_path=siim,
        siim_final_watcher_path=siim_final,
    )
    assert snapshot["ready"] is True
    assert snapshot["staging"]["checks"]["no_process_signals"] is True
    assert snapshot["siim"]["status"] == "validated_complete"

    payload = json.loads(staging.read_text(encoding="utf-8"))
    payload["completed_bytes"] -= 1
    write_json(staging, payload)
    assert (
        module.prerequisite_snapshot(
            plan,
            staging_report_path=staging,
            siim_report_path=siim,
            siim_final_watcher_path=siim_final,
        )["ready"]
        is False
    )


def test_siim_report_fails_closed_on_grader_or_missing_report(tmp_path: Path) -> None:
    module = load_module()
    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)
    missing = module.siim_report_snapshot(plan, tmp_path / "missing.json")
    assert missing["ready"] is False
    assert missing["status"] == "missing"

    siim_path = tmp_path / "siim.json"
    payload = complete_siim(plan)
    payload["official_grader_executed"] = True
    write_json(siim_path, payload)
    snapshot = module.siim_report_snapshot(plan, siim_path)
    assert snapshot["ready"] is False
    assert snapshot["checks"]["no_official_grader"] is False


def test_gpu_parser_and_command_preserve_serial_withheld_contract() -> None:
    module = load_module()
    applications = module.parse_compute_apps(
        "81104, C:\\Python\\python.exe\n6024, C:\\Chrome\\chrome.exe\n"
    )
    assert applications[0]["pid"] == 81104
    assert applications[0]["process_name"].endswith("python.exe")

    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)
    command = module.build_training_command(plan)
    assert command[0] == plan["training"]["python"]
    assert command[1] == plan["training"]["script"]
    assert command[command.index("--seeds") + 1] == "40,41,42"
    assert (
        command[command.index("--backbones") + 1]
        == "convnext_small,efficientnet_v2_s"
    )
    assert command[command.index("--promotion-mean-log-loss") + 1] == "0.0135"
    joined = " ".join(command).lower()
    assert "kaggle" not in joined
    assert "grader" not in joined


def test_leaf_waits_for_terminal_siim_final_verification(tmp_path: Path) -> None:
    module = load_module()
    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)
    watcher = tmp_path / "siim_final.json"
    write_json(watcher, complete_siim_final(plan))
    assert module.siim_final_snapshot(plan, watcher)["ready"] is True
    payload = json.loads(watcher.read_text(encoding="utf-8"))
    payload["process_signals_sent"] = 1
    write_json(watcher, payload)
    assert module.siim_final_snapshot(plan, watcher)["ready"] is False


def test_queue_source_contains_no_process_control_primitive() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Stop-Process" not in source
    assert "taskkill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert '"process_signals_sent": 0' in source
