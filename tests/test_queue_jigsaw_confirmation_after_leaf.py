from __future__ import annotations

import json
from pathlib import Path

from scripts import queue_jigsaw_confirmation_after_leaf as queue


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_leaf_snapshot_requires_terminal_zero_signal_contract(tmp_path: Path):
    watcher = tmp_path / "leaf_watcher.json"
    leaf_plan = tmp_path / "leaf_plan.json"
    leaf_plan.write_text("{}", encoding="utf-8")
    plan_sha = queue.sha256_file(leaf_plan)
    plan = {
        "serial_dependency": {
            "watcher_path": str(watcher),
            "watcher_schema": "evomind.local_candidate_verification_watcher.v1",
            "run_id": "leaf-run",
            "terminal_statuses": ["verification_passed"],
            "plan": {"path": str(leaf_plan), "sha256": plan_sha},
        }
    }
    payload = {
        "schema": "evomind.local_candidate_verification_watcher.v1",
        "run_id": "leaf-run",
        "status": "verification_passed",
        "plan_sha256": plan_sha,
        "candidate_ready": False,
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    _write_json(watcher, payload)
    snapshot = queue.leaf_snapshot(plan)
    assert snapshot["ready"] is True
    assert snapshot["candidate_ready"] is False

    payload["process_signals_sent"] = 1
    _write_json(watcher, payload)
    assert queue.leaf_snapshot(plan)["ready"] is False


def _command_fixture(tmp_path: Path) -> tuple[dict, dict]:
    seed_plan = {
        "_run_id": "jigsaw-s40",
        "_path": str(tmp_path / "seed40.json"),
        "_sha256": "seed-plan-hash",
        "_model_seed": 40,
        "model": {"repo_id": "model", "revision": "revision"},
        "training": {
            "seed": 40,
            "fold_seed": 42,
            "model_seed": 40,
            "epochs_per_fold": 2,
            "max_length": 192,
            "train_batch_size": 8,
            "eval_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "num_workers": 2,
        },
        "execution": {"token_cache_dir": str(tmp_path / "token_cache")},
    }
    plan = {
        "source": {
            "runner": {"path": str(tmp_path / "runner.py")},
            "verifier": {"path": str(tmp_path / "verifier.py")},
        },
        "inputs": {"sparse_bundle": {"path": str(tmp_path / "sparse.npz")}},
        "execution": {
            "training_python": str(tmp_path / "train-python.exe"),
            "verification_python": str(tmp_path / "verify-python.exe"),
            "public_dir": str(tmp_path / "public"),
            "output_root": str(tmp_path / "runs"),
            "hf_cache": str(tmp_path / "hf"),
        },
    }
    return plan, seed_plan


def test_training_command_separates_fold_and_model_seeds(tmp_path: Path):
    plan, seed_plan = _command_fixture(tmp_path)
    command = queue.build_training_command(plan, seed_plan)
    joined = " ".join(command).lower()
    assert command[command.index("--fold-seed") + 1] == "42"
    assert command[command.index("--model-seed") + 1] == "40"
    assert command[command.index("--seed") + 1] == "40"
    assert "--token-cache-dir" in command
    assert "official_grader" not in joined
    assert "kaggle" not in joined


def test_seed_plan_validation_requires_fixed_seed42_folds(tmp_path: Path):
    source = {
        "runner": {"path": "runner.py", "sha256": "runner-hash"},
        "verifier": {"path": "verifier.py", "sha256": "verifier-hash"},
    }
    inputs = {
        "public_train_sha256": "train",
        "public_test_sha256": "test",
        "public_sample_submission_sha256": "sample",
        "sparse_bundle_sha256": "sparse",
        "sparse_bundle_numeric_arrays_loaded_without_pickle": True,
        "legacy_object_id_arrays_ignored": True,
        "order_contract": "fixed_seed42_outer_folds_plus_public_truth_exact_match",
    }
    seed_plan_path = tmp_path / "seed40.json"
    payload = {
        "schema": queue.EXPECTED_SEED_PLAN_SCHEMA,
        "status": "frozen_before_training",
        "competition_id": "jigsaw-toxic-comment-classification-challenge",
        "training": {"seed": 40, "fold_seed": 42, "model_seed": 40, "folds": 5},
        "inputs": inputs,
        "implementation": source,
        "execution": {
            "run_id": "jigsaw-s40",
            "automatic_kaggle_submission": False,
            "official_private_grader_before_gate": False,
            "process_signals_allowed": False,
            "human_gate_preserved": True,
        },
    }
    _write_json(seed_plan_path, payload)
    record = {
        "model_seed": 40,
        "run_id": "jigsaw-s40",
        "plan": {
            "path": str(seed_plan_path),
            "sha256": queue.sha256_file(seed_plan_path),
        },
    }
    validated = queue.validate_seed_plan(
        record, seed=40, source=source, inputs=inputs
    )
    assert validated["_model_seed"] == 40
    assert validated["training"]["fold_seed"] == 42

    payload["training"]["fold_seed"] = 40
    _write_json(seed_plan_path, payload)
    record["plan"]["sha256"] = queue.sha256_file(seed_plan_path)
    try:
        queue.validate_seed_plan(record, seed=40, source=source, inputs=inputs)
    except RuntimeError as exc:
        assert "confirmation split" in str(exc)
    else:
        raise AssertionError("Seed-specific outer folds were accepted")


def _report(seed: int, auc: float, gain: float, *, passed: bool = True) -> dict:
    return {
        "schema": queue.EXPECTED_REPORT_SCHEMA,
        "status": "promotion_gate_passed" if passed else "promotion_gate_failed",
        "run_id": f"jigsaw-s{seed}",
        "run_dir": f"run-{seed}",
        "plan_sha256": f"plan-{seed}",
        "seed_contract": {"fold_assignment_seed": 42, "model_seed": seed},
        "metrics": {
            "candidate_auc": auc,
            "gain_over_strongest_base": gain,
        },
        "full_contract_valid": True,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "process_signals_sent": 0,
    }


def test_report_validation_requires_exact_seed_and_human_gate():
    seed_plan = {
        "_run_id": "jigsaw-s40",
        "_sha256": "plan-40",
        "_model_seed": 40,
    }
    report = _report(40, 0.9905, 0.0010)
    assert queue.report_valid(report, seed_plan) is True
    report["seed_contract"]["fold_assignment_seed"] = 40
    assert queue.report_valid(report, seed_plan) is False


def test_multiseed_aggregate_requires_all_three_reproducible_seeds(tmp_path: Path):
    plan = {
        "_path": str(tmp_path / "plan.json"),
        "_sha256": "umbrella-plan",
        "_seed42_audit": {
            "run_dir": "run-42",
            "metrics": {
                "candidate_auc": 0.9909,
                "gain_over_strongest_base": 0.0011,
            },
        },
        "confirmation_gate": {
            "minimum_seed_auc": 0.987,
            "minimum_mean_auc": 0.987,
            "minimum_seed_gain": 0.0003,
            "maximum_population_std": 0.0025,
        },
    }
    result = queue.aggregate_confirmation(
        plan,
        [_report(40, 0.9901, 0.0008), _report(41, 0.9898, 0.0007)],
    )
    assert result["status"] == "confirmation_passed_human_gate_pending"
    assert result["candidate_ready_for_human_gate"] is True
    assert result["confirmation_gate"]["passed"] is True
    assert [item["model_seed"] for item in result["seed_records"]] == [40, 41, 42]

    failed = queue.aggregate_confirmation(
        plan,
        [_report(40, 0.9869, 0.0008, passed=False), _report(41, 0.9898, 0.0007)],
    )
    assert failed["status"] == "confirmation_failed"
    assert failed["candidate_ready_for_human_gate"] is False


def test_invariants_are_strict_serial_and_zero_signal():
    assert queue.invariant_fields() == {
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "human_gate_preserved": True,
        "strict_single_gpu_serial": True,
    }
