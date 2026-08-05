from __future__ import annotations

from scripts.verify_backend_resource_status import gpu_current_gate_ready


def test_gpu_gate_uses_current_run_authoritative_state_over_stale_history() -> None:
    item = {
        "current_gate_ready": True,
        "evidence": {
            "latest_ssh_connection": {"present": True, "passed": False},
            "latest_s6e6_dependency_gate": {"status": "passed"},
        },
    }

    assert gpu_current_gate_ready(item) is True


def test_gpu_gate_preserves_explicit_backend_block() -> None:
    item = {
        "current_gate_ready": False,
        "evidence": {
            "latest_ssh_connection": {"present": True, "passed": True},
            "latest_s6e6_dependency_gate": {"status": "passed"},
        },
    }

    assert gpu_current_gate_ready(item) is False


def test_gpu_gate_falls_back_for_legacy_summary_without_authoritative_state() -> None:
    item = {
        "evidence": {
            "latest_ssh_connection": {"present": True, "passed": True},
            "latest_s6e6_dependency_gate": {"status": "passed"},
        },
    }

    assert gpu_current_gate_ready(item) is True
