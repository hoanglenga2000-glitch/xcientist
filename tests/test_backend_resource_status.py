from __future__ import annotations

from scripts.verify_backend_resource_status import gpu_current_gate_ready, local_connector_ready


def test_local_connector_requires_canonical_and_raw_state() -> None:
    item = {
        "configured": True,
        "state": "READY",
        "raw_state": "rule_based",
        "source": "connector_health_service",
    }

    assert local_connector_ready(item, "rule_based") is True
    assert local_connector_ready({**item, "state": "DEGRADED"}, "rule_based") is False
    assert local_connector_ready({**item, "raw_state": "unknown"}, "rule_based") is False
    assert local_connector_ready({**item, "source": "legacy_cache"}, "rule_based") is False


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
