from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "queue_may2022_after_spooky.py"
PLAN_PATH = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_compact_embedding_s42_frozen_plan_v2_20260728.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("queue_may2022_after_spooky", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_plan_and_command_preserve_serial_gpu_contract() -> None:
    module = load_module()
    plan = module.validate_frozen_plan(PLAN_PATH)
    command = module.build_training_command(plan)

    assert plan["planner"]["requested_model"] == "gpt-5.6-sol"
    assert plan["planner"]["served_model"] == "gpt-5.6-sol"
    assert plan["execution"]["single_gpu_strict_serial"] is True
    assert plan["execution"]["process_signals_allowed"] is False
    assert plan["private_labels_allowed"] is False
    assert plan["execution"]["gpu_idle_gate_mode"] == "calibrated_wddm_multimetric_v1"
    assert plan["_idle_policy"]["legacy_gate"]["classification"] == (
        "persistent_false_busy_on_WDDM"
    )
    assert plan["_idle_policy"]["requirements"]["consecutive_checks"] == 3
    assert plan["_idle_policy"]["requirements"][
        "python_compute_applications_must_be_empty"
    ] is True
    assert command[0] == plan["execution"]["python"]
    assert command[1] == plan["execution"]["script"]
    assert "--plan" in command
    assert "--run-id" in command


def test_compute_app_parser_identifies_python_without_signaling() -> None:
    module = load_module()
    rows = module.parse_compute_apps(
        "123, C:\\Python\\python.exe\n456, C:\\Windows\\explorer.exe\n"
    )
    assert rows == [
        {"pid": 123, "process_name": "C:\\Python\\python.exe"},
        {"pid": 456, "process_name": "C:\\Windows\\explorer.exe"},
    ]


def test_calibrated_idle_gate_rejects_active_python_and_accepts_wddm_baseline() -> None:
    module = load_module()
    policy = module.validate_frozen_plan(PLAN_PATH)["_idle_policy"]
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
    result = module.calibrated_idle.evaluate_idle(policy, baseline)
    assert result["idle"] is False
    assert result["checks"]["no_python_compute"] is False


def test_launch_claim_is_atomic_and_never_replaces_existing_claim(tmp_path: Path) -> None:
    module = load_module()
    claim = tmp_path / "launch_claim.json"
    first = {"run_id": "may-s42", "queue_pid": 100}
    second = {"run_id": "may-s42", "queue_pid": 200}
    assert module.acquire_launch_claim(claim, first) is True
    assert module.acquire_launch_claim(claim, second) is False
    assert json.loads(claim.read_text(encoding="utf-8")) == first


def test_spooky_snapshot_requires_terminal_full_fold_public_run(tmp_path: Path) -> None:
    module = load_module()
    report = tmp_path / "summary.json"
    report.write_text(
        json.dumps(
            {
                "schema": "evomind.spooky.transformer_run.v1",
                "run_id": "spooky-s42",
                "status": "single_seed_gate_failed",
                "requested_folds": [0, 1, 2, 3, 4],
                "private_labels_used": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "execution": {
            "serial_dependency": {
                "report": str(report),
                "run_id": "spooky-s42",
                "terminal_statuses": [
                    "single_seed_gate_passed_confirmation_pending",
                    "single_seed_gate_failed",
                ],
            }
        }
    }
    snapshot = module.spooky_snapshot(plan)
    assert snapshot["ready"] is True
    broken = json.loads(report.read_text(encoding="utf-8"))
    broken["requested_folds"] = [0, 1, 2, 3]
    report.write_text(json.dumps(broken), encoding="utf-8")
    assert module.spooky_snapshot(plan)["ready"] is False
