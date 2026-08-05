from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "queue_siim_after_jigsaw.py"


def load_module():
    spec = importlib.util.spec_from_file_location("queue_siim_after_jigsaw", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_current_frozen_plan_is_complete_and_hash_verified() -> None:
    module = load_module()
    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)

    assert plan["planner"]["requested_model"] == "gpt-5.6-sol"
    assert plan["planner"]["served_model"] == "gpt-5.6-sol"
    assert plan["training"]["evaluation_seeds"] == [40, 41, 42]
    assert len(plan["training"]["profiles"]) == 5
    assert plan["launch_contract"]["gpu_idle_gate_mode"] == (
        "calibrated_wddm_multimetric_v1"
    )
    assert plan["_idle_policy"]["legacy_gate"]["classification"] == (
        "persistent_false_busy_on_WDDM"
    )


def test_prerequisite_snapshot_requires_exact_complete_contract(tmp_path: Path) -> None:
    module = load_module()
    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)
    jigsaw = tmp_path / "jigsaw.json"
    staging = tmp_path / "staging.json"
    may = tmp_path / "may_summary.json"
    expected = plan["data_contract"]
    write_json(jigsaw, {"status": "verification_passed"})
    write_json(
        staging,
        {
            "status": "size_verified_complete",
            "completed_files": expected["file_count"],
            "completed_bytes": expected["total_bytes"],
            "inventory_manifest_sha256": expected["manifest_sha256"],
            "errors": [],
            "private_paths_requested": False,
            "remote_writes_performed": False,
            "process_signals_sent": 0,
        },
    )
    may_dependency = plan["serial_dependency"]["may2022"]
    write_json(
        may,
        {
            "schema": "evomind.mlebench.may2022_compact_embedding_run.v1",
            "run_id": may_dependency["run_id"],
            "status": "single_seed_gate_failed",
            "plan_sha256": may_dependency["plan_sha256"],
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    )

    snapshot = module.prerequisite_snapshot(
        plan,
        jigsaw_watcher_path=jigsaw,
        staging_report_path=staging,
        may_summary_path=may,
    )
    assert snapshot["ready"] is True

    payload = json.loads(staging.read_text(encoding="utf-8"))
    payload["completed_bytes"] -= 1
    write_json(staging, payload)
    assert (
        module.prerequisite_snapshot(
            plan,
            jigsaw_watcher_path=jigsaw,
            staging_report_path=staging,
            may_summary_path=may,
        )["ready"]
        is False
    )


def test_calibrated_gate_rejects_active_python_and_accepts_wddm_baseline() -> None:
    module = load_module()
    policy = module.validate_frozen_plan(module.DEFAULT_PLAN)["_idle_policy"]
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
    assert module.calibrated_idle.evaluate_idle(policy, baseline)["idle"] is True
    baseline["python_compute_applications"] = [
        {"pid": 42, "process_name": "python.exe"}
    ]
    assert module.calibrated_idle.evaluate_idle(policy, baseline)["idle"] is False


def test_gpu_parser_and_command_preserve_serial_contract() -> None:
    module = load_module()
    applications = module.parse_compute_apps(
        "81104, C:\\\\Python\\\\python.exe\n6024, C:\\\\Chrome\\\\chrome.exe\n"
    )
    assert applications[0]["pid"] == 81104
    assert applications[0]["process_name"].endswith("python.exe")

    plan = module.validate_frozen_plan(module.DEFAULT_PLAN)
    command = module.build_training_command(plan)
    assert command[0] == plan["training"]["python"]
    assert command[1] == plan["training"]["script"]
    assert command[command.index("--evaluation-seeds") + 1] == "40,41,42"
    assert "--no-resume-cache" not in command
    assert "kaggle" not in " ".join(command).lower()
    assert "grader" not in " ".join(command).lower()
