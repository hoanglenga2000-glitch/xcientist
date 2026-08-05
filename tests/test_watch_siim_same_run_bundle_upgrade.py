from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import watch_siim_same_run_bundle_upgrade as watcher

RUN_ID = "evomind_siim_bundle_upgrade_fixture"
OLD_SHA = "1" * 64
NEW_SHA = "2" * 64


def _plan() -> dict[str, Any]:
    return {
        "schema": "evomind.siim.same_run_bundle_upgrade_plan.v1",
        "run_id": RUN_ID,
        "job_id": watcher.campaign.HPC_JOB_ID,
        "credential_profile": watcher.campaign.CREDENTIAL_PROFILE,
        "old_bundle_sha256": OLD_SHA,
        "new_bundle_sha256": NEW_SHA,
        "activation_preconditions": {
            "remote_supervisor_process_exists": False,
            "allowed_remote_states": ["failed", "needs_continuation"],
            "same_run_required": True,
            "ablation_fold_count": 3,
            "ablation_evaluation_seeds": [40, 41, 42],
        },
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def _report() -> dict[str, Any]:
    return {
        "passed": True,
        "fold_count": 3,
        "evaluation_seed_count": 3,
        "evaluation_seeds": [40, 41, 42],
        "run_directory_action": "resumed",
    }


class FakeApi:
    def __init__(
        self,
        *,
        remote_status: str = "failed",
        remote_process: bool = False,
        local_process: bool = False,
        source_sha: str = NEW_SHA,
        report: Mapping[str, Any] | None = None,
        gate_passed: bool = True,
    ) -> None:
        self.remote_status = remote_status
        self.remote_process = remote_process
        self.local_process = local_process
        self.source_sha = source_sha
        self.report_payload = dict(report or _report())
        self.gate_passed = gate_passed
        self.calls: list[str] = []

    def status(self, run_id: str) -> Mapping[str, Any]:
        self.calls.append("status")
        return {"state": {"status": self.remote_status}, "process_exists": self.remote_process}

    def local_supervisor_active(self, run_id: str) -> bool:
        self.calls.append("local_supervisor_active")
        return self.local_process

    def source_bundle_sha256(self) -> str:
        self.calls.append("source_bundle_sha256")
        return self.source_sha

    def ablation_report(self, run_id: str) -> Mapping[str, Any]:
        self.calls.append("ablation_report")
        return self.report_payload

    def prepare(self, run_id: str) -> Mapping[str, Any]:
        self.calls.append("prepare")
        return {"bundle_sha256": NEW_SHA}

    def gate(self, run_id: str, *, interval_seconds: int, output_path: Path) -> Mapping[str, Any]:
        self.calls.append("gate")
        return {
            "passed": self.gate_passed,
            "hold_reasons": [] if self.gate_passed else ["gpu_not_idle"],
        }

    def launch(self, run_id: str, *, gate_path: Path, launch_path: Path) -> Mapping[str, Any]:
        self.calls.append("launch")
        return {
            "action": "new_supervisor_started",
            "supervisor_pid": 4321,
            "bundle_sha256": NEW_SHA,
        }

    def start_local_supervisor(self, run_id: str) -> int:
        self.calls.append("start_local_supervisor")
        return 9876


def test_active_remote_supervisor_is_only_observed() -> None:
    api = FakeApi(remote_status="ablation_training", remote_process=True)

    state = watcher.run_iteration(
        RUN_ID,
        _plan(),
        sample_interval_seconds=0,
        api=api,
    )

    assert state["status"] == "waiting_remote_supervisor_boundary"
    assert api.calls == ["status", "local_supervisor_active"]


def test_idle_boundary_waits_for_old_local_supervisor_to_exit() -> None:
    api = FakeApi(local_process=True)

    state = watcher.run_iteration(
        RUN_ID,
        _plan(),
        sample_interval_seconds=0,
        api=api,
    )

    assert state["status"] == "waiting_local_supervisor_exit"
    assert api.calls == ["status", "local_supervisor_active"]


def test_valid_idle_boundary_activates_new_bundle_and_both_supervisors(tmp_path, monkeypatch) -> None:
    api = FakeApi()
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})

    state = watcher.run_iteration(
        RUN_ID,
        _plan(),
        sample_interval_seconds=0,
        api=api,
    )

    assert state["status"] == "launched"
    assert state["bundle_sha256"] == NEW_SHA
    assert state["remote_supervisor_pid"] == 4321
    assert state["local_supervisor_pid"] == 9876
    assert api.calls == [
        "status",
        "local_supervisor_active",
        "source_bundle_sha256",
        "ablation_report",
        "prepare",
        "gate",
        "launch",
        "start_local_supervisor",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fold_count", 2, "fold count"),
        ("evaluation_seeds", [40, 41, 99], "evaluation seeds"),
        ("run_directory_action", "created", "resume run"),
    ],
)
def test_invalid_ablation_evidence_blocks_before_prepare(
    tmp_path,
    monkeypatch,
    field,
    value,
    message,
) -> None:
    report = _report()
    report[field] = value
    api = FakeApi(report=report)
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})

    with pytest.raises(watcher.SameRunBundleUpgradeError, match=message):
        watcher.run_iteration(
            RUN_ID,
            _plan(),
            sample_interval_seconds=0,
            api=api,
        )

    assert "prepare" not in api.calls
    assert "launch" not in api.calls


def test_source_bundle_drift_blocks_before_remote_write(tmp_path, monkeypatch) -> None:
    api = FakeApi(source_sha="3" * 64)
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})

    with pytest.raises(watcher.SameRunBundleUpgradeError, match="differs"):
        watcher.run_iteration(
            RUN_ID,
            _plan(),
            sample_interval_seconds=0,
            api=api,
        )

    assert "ablation_report" not in api.calls
    assert "prepare" not in api.calls


def test_gpu_hold_never_launches(tmp_path, monkeypatch) -> None:
    api = FakeApi(gate_passed=False)
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})

    state = watcher.run_iteration(
        RUN_ID,
        _plan(),
        sample_interval_seconds=0,
        api=api,
    )

    assert state["status"] == "activation_gate_hold"
    assert state["signals_sent"] == 0
    assert state["other_processes_modified"] is False
    assert "launch" not in api.calls
    assert "start_local_supervisor" not in api.calls


def test_idle_unknown_remote_state_fails_closed() -> None:
    api = FakeApi(remote_status="missing", remote_process=False)

    with pytest.raises(watcher.SameRunBundleUpgradeError, match="allowed upgrade boundary"):
        watcher.run_iteration(
            RUN_ID,
            _plan(),
            sample_interval_seconds=0,
            api=api,
        )

    assert api.calls == ["status", "local_supervisor_active"]


def test_reused_pid_for_unrelated_process_does_not_block_bundle_upgrade(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "end_to_end_supervisor.json").write_text(
        '{"supervisor_pid":23884}', encoding="utf-8"
    )
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})
    monkeypatch.setattr(watcher, "process_exists", lambda _pid: True)
    monkeypatch.setattr(watcher, "process_command_line", lambda _pid: "wps.exe")

    assert watcher.ProductionApi().local_supervisor_active(RUN_ID) is False


def test_matching_supervisor_pid_and_command_is_active(tmp_path, monkeypatch) -> None:
    (tmp_path / "end_to_end_supervisor.json").write_text(
        '{"supervisor_pid":1234}', encoding="utf-8"
    )
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})
    monkeypatch.setattr(watcher, "process_exists", lambda _pid: True)
    command = (
        f'python "{watcher.PROJECT_ROOT / "scripts" / "supervise_siim_job89508_end_to_end.py"}" '
        f"--run-id {RUN_ID}"
    )
    monkeypatch.setattr(watcher, "process_command_line", lambda _pid: command)

    assert watcher.ProductionApi().local_supervisor_active(RUN_ID) is True


def test_unreadable_live_process_identity_fails_closed(tmp_path, monkeypatch) -> None:
    (tmp_path / "end_to_end_supervisor.json").write_text(
        '{"supervisor_pid":1234}', encoding="utf-8"
    )
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})
    monkeypatch.setattr(watcher, "process_exists", lambda _pid: True)
    monkeypatch.setattr(watcher, "process_command_line", lambda _pid: None)

    with pytest.raises(watcher.SameRunBundleUpgradeError, match="identity is unreadable"):
        watcher.ProductionApi().local_supervisor_active(RUN_ID)
