#!/usr/bin/env python3
"""Hash-bound WDDM-aware idle classification for the local RTX 4060."""

from __future__ import annotations

import csv
import hashlib
import json
import locale
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "local_rtx4060_wddm_idle_gate_frozen_v2_20260728.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy_path = Path(path).resolve()
    policy = read_json(policy_path)
    if policy.get("schema") != "evomind.local_gpu.calibrated_idle_gate.v1":
        raise RuntimeError("Unexpected local GPU idle-gate schema")
    if policy.get("status") != "frozen_before_use":
        raise RuntimeError("Local GPU idle gate is not frozen")
    if policy.get("device") != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError("Local GPU idle gate targets a different device")
    if policy.get("platform_mode") != "Windows_WDDM":
        raise RuntimeError("Local GPU idle gate targets a different platform mode")
    for name, artifact in {
        "baseline": policy.get("baseline"),
        **(policy.get("bound_execution_evidence") or {}),
    }.items():
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Idle gate is missing {name}")
        artifact_path = Path(artifact.get("path", "")).resolve()
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get(
            "sha256"
        ):
            raise RuntimeError(f"Idle-gate artifact changed: {name}")
    requirements = policy.get("requirements") or {}
    expected_keys = {
        "maximum_gpu_utilization_percent",
        "maximum_memory_utilization_percent",
        "minimum_memory_free_mib",
        "maximum_temperature_c",
        "allowed_pstates",
        "maximum_power_draw_w",
        "maximum_sm_clock_mhz",
        "python_compute_applications_must_be_empty",
        "consecutive_checks",
        "minimum_check_interval_seconds",
    }
    if set(requirements) != expected_keys:
        raise RuntimeError("Local GPU idle-gate requirements changed")
    if (
        requirements["python_compute_applications_must_be_empty"] is not True
        or policy.get("single_gpu_strict_serial") is not True
        or policy.get("preemption_allowed") is not False
        or policy.get("process_signals_allowed") is not False
        or policy.get("official_grader_executed") is not False
        or policy.get("kaggle_submission_executed") is not False
    ):
        raise RuntimeError("Local GPU idle-gate safety contract changed")
    baseline = read_json(Path(policy["baseline"]["path"]))
    if (
        baseline.get("device") != policy["device"]
        or baseline.get("summary", {}).get("maximum_python_compute_applications") != 0
        or baseline.get("process_signals_sent") != 0
    ):
        raise RuntimeError("Local GPU baseline does not prove a signal-free idle sample")
    policy["_path"] = str(policy_path)
    policy["_sha256"] = sha256_file(policy_path)
    return policy


def validate_launch_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Resolve and hash-check a queue's calibrated WDDM launch policy."""
    if contract.get("gpu_idle_gate_mode") != "calibrated_wddm_multimetric_v1":
        raise RuntimeError("Queue does not use the calibrated WDDM idle gate")
    binding = contract.get("idle_gate_policy") or {}
    policy_path = Path(str(binding.get("path", ""))).resolve()
    if not policy_path.is_file() or sha256_file(policy_path) != binding.get("sha256"):
        raise RuntimeError("Queue calibrated idle-gate binding changed")
    policy = validate_policy(policy_path)
    requirements = policy["requirements"]
    if (
        contract.get("gpu_idle_consecutive_checks")
        != requirements["consecutive_checks"]
        or contract.get("gpu_idle_minimum_check_interval_seconds")
        != requirements["minimum_check_interval_seconds"]
    ):
        raise RuntimeError("Queue idle timing differs from the calibrated policy")
    return policy


def decode_command_output(output: bytes | str | None) -> str:
    """Decode localized Windows CLI output without a subprocess reader thread."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    encodings = ["utf-8", locale.getpreferredencoding(False), "gb18030"]
    for encoding in dict.fromkeys(encodings):
        try:
            return output.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return output.decode("utf-8", errors="replace")


def parse_compute_apps(output: bytes | str | None) -> list[dict[str, Any]]:
    applications: list[dict[str, Any]] = []
    for row in csv.reader(decode_command_output(output).splitlines()):
        if len(row) < 2:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        applications.append({"pid": pid, "process_name": row[1].strip()})
    return applications


def query_gpu() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.free,"
            "temperature.gpu,pstate,power.draw,clocks.current.sm",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=False,
        timeout=30,
    )
    gpu_output = decode_command_output(completed.stdout)
    if not gpu_output.strip():
        raise RuntimeError("nvidia-smi returned no local GPU data")
    values = [value.strip() for value in gpu_output.splitlines()[0].split(",")]
    if len(values) < 9:
        raise RuntimeError("nvidia-smi returned an incomplete local GPU row")
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=False,
        timeout=30,
    )
    applications = parse_compute_apps(compute.stdout)
    python_compute = [
        value
        for value in applications
        if "python" in Path(value["process_name"]).name.lower()
    ]
    return {
        "name": values[0],
        "utilization_percent": int(values[1]),
        "memory_utilization_percent": int(values[2]),
        "memory_used_mib": int(values[3]),
        "memory_free_mib": int(values[4]),
        "temperature_c": int(values[5]),
        "pstate": values[6],
        "power_draw_w": float(values[7]),
        "sm_clock_mhz": int(values[8]),
        "compute_applications": applications,
        "python_compute_applications": python_compute,
    }


def evaluate_idle(policy: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    requirements = policy["requirements"]
    checks = {
        "expected_device": policy["device"].lower() in gpu["name"].lower(),
        "gpu_utilization": gpu["utilization_percent"]
        <= requirements["maximum_gpu_utilization_percent"],
        "memory_utilization": gpu["memory_utilization_percent"]
        <= requirements["maximum_memory_utilization_percent"],
        "memory_free": gpu["memory_free_mib"] >= requirements["minimum_memory_free_mib"],
        "temperature": gpu["temperature_c"] <= requirements["maximum_temperature_c"],
        "pstate": gpu["pstate"] in requirements["allowed_pstates"],
        "power_draw": gpu["power_draw_w"] <= requirements["maximum_power_draw_w"],
        "sm_clock": gpu["sm_clock_mhz"] <= requirements["maximum_sm_clock_mhz"],
        "no_python_compute": not gpu["python_compute_applications"],
    }
    return {"idle": all(checks.values()), "checks": checks}


def policy_record(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": policy["_path"],
        "sha256": policy["_sha256"],
        "device": policy["device"],
        "platform_mode": policy["platform_mode"],
        "requirements": policy["requirements"],
        "legacy_gate": policy["legacy_gate"],
    }
