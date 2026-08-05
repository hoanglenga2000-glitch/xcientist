from __future__ import annotations

import json
from pathlib import Path

from scripts import watch_local_candidate_verification as watcher


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_commands_never_include_grader_or_kaggle(tmp_path: Path):
    for profile in ("leaf", "spooky", "may2022"):
        spec = {
            "profile": profile,
            "python": tmp_path / "python.exe",
            "verifier": tmp_path / f"verify_{profile}.py",
            "run_dir": tmp_path / profile,
            "plan_path": tmp_path / f"{profile}.json",
            "output": tmp_path / profile / "independent_verification.json",
            "public_dir": tmp_path / "public",
        }
        command = watcher.build_verifier_command(spec)
        joined = " ".join(map(str, command)).lower()
        assert "grader" not in joined
        assert "kaggle" not in joined
        assert "--run-dir" in command
        assert "--output" in command
        assert ("--public-dir" in command) is (profile == "spooky")


def test_terminal_snapshot_requires_zero_signal_public_boundary(tmp_path: Path):
    terminal = tmp_path / "summary.json"
    queue = tmp_path / "queue.json"
    write_json(
        terminal,
        {
            "run_id": "run-1",
            "status": "single_seed_gate_failed",
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
        },
    )
    write_json(
        queue,
        {
            "process_signals_sent": 0,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    spec = {
        "terminal_path": terminal,
        "queue": queue,
        "terminal_statuses": {"single_seed_gate_failed"},
        "run_id": "run-1",
    }
    assert watcher.terminal_snapshot(spec)["ready"] is True

    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["process_signals_sent"] = 1
    write_json(queue, payload)
    assert watcher.terminal_snapshot(spec)["ready"] is False


def test_report_passed_contracts_are_profile_specific():
    assert watcher.report_passed(
        "spooky", {"status": "passed", "ok": None}
    ) is True
    assert watcher.report_passed(
        "leaf", {"status": "verification_passed", "ok": True}
    ) is True
    assert watcher.report_passed(
        "may2022", {"status": "verification_passed", "ok": True}
    ) is True


def test_invariant_fields_preserve_human_gate():
    assert watcher.invariant_fields() == {
        "process_signals_sent": 0,
        "private_labels_used": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def test_frozen_provenance_fields_survive_terminal_state_transitions(tmp_path: Path):
    spec = {
        "plan_path": tmp_path / "plan.json",
        "plan_sha256": "a" * 64,
        "verifier": tmp_path / "verify.py",
        "verifier_sha256": "b" * 64,
    }

    assert watcher.frozen_provenance_fields(spec) == {
        "plan_path": str(spec["plan_path"]),
        "plan_sha256": "a" * 64,
        "verifier_path": str(spec["verifier"]),
        "verifier_sha256": "b" * 64,
    }


def test_may2022_profile_tracks_calibrated_queue() -> None:
    queue = Path(watcher.PROFILE_SPECS["may2022"]["queue"])
    assert queue.name == "may2022_training_queue_calibrated.json"
    assert "gate20" not in queue.name


def test_leaf_profile_tracks_calibrated_queue() -> None:
    queue = Path(watcher.PROFILE_SPECS["leaf"]["queue"])
    assert queue.name == "leaf_training_queue_calibrated.json"
