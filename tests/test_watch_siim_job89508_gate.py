from __future__ import annotations

from pathlib import Path

import pytest

from research_agent_workstation.server.core.gpu_credentials import (
    CredentialError,
    RetryableTransportError,
)
from scripts import watch_siim_job89508_gate as watcher


class FakeCampaign:
    def __init__(self, *, passed: bool) -> None:
        self.passed = passed
        self.calls: list[tuple[str, object]] = []

    def gate(self, run_id: str, *, interval_seconds: int = 15) -> dict[str, object]:
        self.calls.append(("gate", (run_id, interval_seconds)))
        return {
            "created_at": "2026-07-29T00:00:00+00:00",
            "passed": self.passed,
            "hold_reasons": [] if self.passed else ["blocked_process_state"],
            "samples": [
                {
                    "gpus": [
                        {
                            "memory_free_mib": 81_917,
                            "utilization_percent": 0,
                        }
                    ]
                }
            ] * 5,
        }

    def publish_status(self, run_id: str) -> dict[str, object]:
        self.calls.append(("publish", run_id))
        return {"status": "published"}

    def launch(self, run_id: str, *, max_gate_age: int = 600) -> dict[str, object]:
        self.calls.append(("launch", (run_id, max_gate_age)))
        return {
            "action": "new_supervisor_started",
            "supervisor_pid": 1234,
        }


def test_hold_is_published_without_launch() -> None:
    api = FakeCampaign(passed=False)
    state = watcher.run_iteration(
        "evomind_siim_isic_a800_20260729_223613",
        iteration=1,
        sample_interval_seconds=0,
        max_gate_age=600,
        api=api,
    )

    assert state["status"] == "hold"
    assert state["samples"] == 5
    assert state["signals_sent"] == 0
    assert state["other_processes_modified"] is False
    assert [call[0] for call in api.calls] == ["gate", "publish"]


def test_fresh_go_launches_exactly_once() -> None:
    api = FakeCampaign(passed=True)
    state = watcher.run_iteration(
        "evomind_siim_isic_a800_20260729_223613",
        iteration=2,
        sample_interval_seconds=0,
        max_gate_age=600,
        api=api,
    )

    assert state["status"] == "launched"
    assert state["launch_action"] == "new_supervisor_started"
    assert state["supervisor_pid"] == 1234
    assert [call[0] for call in api.calls] == ["gate", "publish", "launch"]


def test_watch_requires_named_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)
    with pytest.raises(watcher.GateWatcherError, match="job89508"):
        watcher.watch(
            "evomind_siim_isic_a800_20260729_223613",
            poll_seconds=1,
            sample_interval_seconds=0,
            max_gate_age=600,
            max_wait_seconds=1,
            once=True,
            api=FakeCampaign(passed=False),
        )


def test_process_lock_rejects_second_holder(tmp_path: Path) -> None:
    lock = tmp_path / "watch.lock"
    with watcher.exclusive_process_lock(lock):
        with pytest.raises(watcher.GateWatcherError, match="already running"):
            with watcher.exclusive_process_lock(lock):
                pass


def test_only_transport_failures_are_retryable() -> None:
    class SSHException(Exception):
        pass

    assert watcher.is_transient_error(SSHException("fixture drop")) is True
    assert watcher.is_transient_error(TimeoutError("fixture timeout")) is True
    assert watcher.is_transient_error(RetryableTransportError("SOCKS fixture drop")) is True
    assert watcher.is_transient_error(CredentialError("fixture credential failure")) is False
    assert watcher.is_transient_error(ValueError("evidence conflict")) is False


def test_watch_records_retryable_socks_failure_without_exiting_as_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryableCampaign(FakeCampaign):
        def gate(self, run_id: str, *, interval_seconds: int = 15) -> dict[str, object]:
            raise RetryableTransportError("SOCKS5 connection closed during handshake")

    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", str(watcher.campaign.HPC_JOB_ID))
    monkeypatch.setenv(
        "EVOMIND_HPC_CREDENTIAL_PROFILE",
        watcher.campaign.CREDENTIAL_PROFILE,
    )
    monkeypatch.setattr(watcher.campaign, "local_paths", lambda _run_id: {"root": tmp_path})

    state = watcher.watch(
        "evomind_siim_isic_a800_20260729_223613",
        poll_seconds=1,
        sample_interval_seconds=0,
        max_gate_age=600,
        max_wait_seconds=0,
        once=True,
        api=RetryableCampaign(passed=False),
    )

    assert state["status"] == "retryable_error"
    assert state["error_type"] == "RetryableTransportError"
