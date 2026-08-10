"""Contracts for the WDDM-aware RTX 4060 idle gate."""

from __future__ import annotations

import inspect
import os
import shutil

import pytest

from scripts import local_rtx4060_idle_gate as gate


def idle_fixture(**overrides):
    value = {
        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "utilization_percent": 35,
        "memory_utilization_percent": 25,
        "memory_used_mib": 1230,
        "memory_free_mib": 6720,
        "temperature_c": 50,
        "pstate": "P8",
        "power_draw_w": 8.0,
        "sm_clock_mhz": 300,
        "compute_applications": [],
        "python_compute_applications": [],
    }
    value.update(overrides)
    return value


def test_frozen_policy_binds_baseline_and_execution_evidence():
    policy = gate.validate_policy()
    assert policy["device"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert policy["legacy_gate"]["classification"] == "persistent_false_busy_on_WDDM"
    assert policy["requirements"]["consecutive_checks"] == 3
    assert policy["process_signals_allowed"] is False


def test_calibrated_idle_requires_every_hardware_check():
    policy = gate.validate_policy()
    result = gate.evaluate_idle(policy, idle_fixture())
    assert result["idle"] is True
    assert all(result["checks"].values())
    assert gate.evaluate_idle(policy, idle_fixture(pstate="P2"))["idle"] is False
    assert gate.evaluate_idle(policy, idle_fixture(power_draw_w=30.0))["idle"] is False
    assert gate.evaluate_idle(policy, idle_fixture(memory_free_mib=4096))["idle"] is False
    assert gate.evaluate_idle(
        policy,
        idle_fixture(
            python_compute_applications=[{"pid": 123, "process_name": "python.exe"}]
        ),
    )["idle"] is False


def test_launch_contract_is_hash_bound_to_calibrated_policy():
    policy = gate.validate_policy()
    contract = {
        "gpu_idle_gate_mode": "calibrated_wddm_multimetric_v1",
        "gpu_idle_consecutive_checks": policy["requirements"]["consecutive_checks"],
        "gpu_idle_minimum_check_interval_seconds": policy["requirements"][
            "minimum_check_interval_seconds"
        ],
        "idle_gate_policy": {
            "path": policy["_path"],
            "sha256": policy["_sha256"],
        },
    }
    assert gate.validate_launch_contract(contract)["_sha256"] == policy["_sha256"]
    contract["idle_gate_policy"]["sha256"] = "0" * 64
    try:
        gate.validate_launch_contract(contract)
    except RuntimeError as exc:
        assert "binding changed" in str(exc)
    else:
        raise AssertionError("tampered idle policy binding was accepted")


def test_parser_ignores_non_pid_compute_rows():
    assert gate.parse_compute_apps("N/A, denied\n123, C:\\Python\\python.exe\n") == [
        {"pid": 123, "process_name": "C:\\Python\\python.exe"}
    ]


def test_parser_accepts_localized_bytes_and_empty_output(monkeypatch):
    monkeypatch.setattr(gate.locale, "getpreferredencoding", lambda _do_setlocale=False: "gb18030")
    encoded = "123, C:\\工具\\python.exe\n".encode("gb18030")
    assert gate.parse_compute_apps(encoded) == [
        {"pid": 123, "process_name": "C:\\工具\\python.exe"}
    ]
    assert gate.parse_compute_apps(None) == []


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("EVOMIND_RUN_LIVE_GPU_TESTS") != "1" or not shutil.which("nvidia-smi"),
    reason="set EVOMIND_RUN_LIVE_GPU_TESTS=1 to run the local hardware integration probe",
)
def test_live_query_has_every_required_measurement():
    policy = gate.validate_policy()
    gpu = gate.query_gpu()
    result = gate.evaluate_idle(policy, gpu)
    assert set(result["checks"]) == {
        "expected_device",
        "gpu_utilization",
        "memory_utilization",
        "memory_free",
        "temperature",
        "pstate",
        "power_draw",
        "sm_clock",
        "no_python_compute",
    }


def test_source_has_no_process_control_primitive():
    source = inspect.getsource(gate)
    assert "Stop-Process" not in source
    assert "taskkill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "os.kill(" not in source
