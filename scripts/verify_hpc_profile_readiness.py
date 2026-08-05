#!/usr/bin/env python3
"""Read-only readiness audit for a named EvoMind HPC DPAPI profile.

The verifier does not decrypt credentials, open SSH sockets, start training,
copy data, run graders, or touch Kaggle.  It exists to turn profile lifecycle
state into a machine-readable gate before any remote/HPC action is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_agent_workstation.server.core import gpu_credentials  # noqa: E402


SCHEMA = "evomind.hpc.profile_readiness.v1"
EXPECTED_GATEWAY_HOST = gpu_credentials.HPC_SSH_GATEWAY_HOST
EXPECTED_GATEWAY_PORT = gpu_credentials.HPC_SSH_GATEWAY_PORT
EXPECTED_REMOTE_ROOT = gpu_credentials.ALLOWED_GPU_REMOTE_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_profile(profile: str) -> str:
    value = str(profile or "").strip()
    if value in {"", ".", ".."} or not gpu_credentials.SAFE_CREDENTIAL_PROFILE.fullmatch(value):
        raise ValueError("unsafe HPC credential profile")
    if not re.fullmatch(r"job[1-9][0-9]*", value):
        raise ValueError("HPC profile readiness requires job<job_id> naming")
    return value


def _profile_dir(profile: str, *, appdata: str | Path | None = None) -> Path:
    root = Path(appdata or os.environ.get("APPDATA", "")).expanduser()
    if not str(root):
        raise ValueError("APPDATA is required")
    if not root.is_absolute():
        raise ValueError("APPDATA must be absolute")
    return (root / "ResearchAgentWorkstation" / "profiles" / profile).resolve(strict=False)


def _safe_child(root: Path, name: str) -> Path:
    child = (root / name).resolve(strict=False)
    child.relative_to(root.resolve(strict=False))
    return child


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def evaluate_profile(profile: str = "job90353", *, appdata: str | Path | None = None) -> dict[str, Any]:
    selected = _safe_profile(profile)
    directory = _profile_dir(selected, appdata=appdata)
    metadata_path = _safe_child(directory, gpu_credentials.PROFILE_METADATA_FILENAME)
    credential_path = _safe_child(directory, gpu_credentials.PROFILE_CREDENTIAL_FILENAME)
    frozen_path = _safe_child(directory, gpu_credentials.PROFILE_FROZEN_TOMBSTONE_FILENAME)
    retired_path = _safe_child(directory, gpu_credentials.PROFILE_RETIRED_TOMBSTONE_FILENAME)

    metadata, metadata_error = _read_json(metadata_path)
    checks: dict[str, bool] = {
        "profile_directory_exists": directory.is_dir() and not directory.is_symlink(),
        "credential_file_exists": credential_path.is_file() and not credential_path.is_symlink(),
        "metadata_file_exists": metadata_path.is_file() and not metadata_path.is_symlink(),
        "no_frozen_tombstone": not frozen_path.exists() and not frozen_path.is_symlink(),
        "no_retired_tombstone": not retired_path.exists() and not retired_path.is_symlink(),
        "metadata_parseable": metadata is not None,
    }
    details: dict[str, Any] = {
        "profile": selected,
        "profile_dir": str(directory),
        "metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
        "credential_file_present": checks["credential_file_exists"],
        "metadata_error": metadata_error,
    }

    if metadata is None:
        for key in (
            "schema_v2",
            "credential_profile_matches",
            "job_id_matches",
            "profile_state_active",
            "allocation_binding_id_valid",
            "allocation_generation_positive",
            "profile_instance_id_present",
            "gateway_binding_matches",
            "allocation_private_ip_not_used",
            "socks_proxy_bound",
            "remote_root_allowed",
            "host_uuid_present",
            "gpu_uuid_present",
            "container_binding_sha256_valid",
            "known_hosts_present",
        ):
            checks[key] = False
    else:
        expected_job_id = int(selected.removeprefix("job"))
        schema_ok = metadata.get("schema") == gpu_credentials.PROFILE_METADATA_SCHEMA
        checks.update(
            {
                "schema_v2": schema_ok,
                "credential_profile_matches": metadata.get("credential_profile") == selected,
                "job_id_matches": metadata.get("job_id") == expected_job_id,
                "profile_state_active": metadata.get("profile_state") == gpu_credentials.PROFILE_STATE_ACTIVE,
                "allocation_binding_id_valid": bool(
                    gpu_credentials.SAFE_ALLOCATION_BINDING_ID.fullmatch(
                        str(metadata.get("allocation_binding_id") or "")
                    )
                ),
                "allocation_generation_positive": isinstance(metadata.get("allocation_generation"), int)
                and not isinstance(metadata.get("allocation_generation"), bool)
                and int(metadata.get("allocation_generation")) > 0,
                "profile_instance_id_present": bool(str(metadata.get("profile_instance_id") or "").strip()),
                "gateway_binding_matches": metadata.get("host") == EXPECTED_GATEWAY_HOST
                and int(metadata.get("port") or 0) == EXPECTED_GATEWAY_PORT,
                "allocation_private_ip_not_used": not str(metadata.get("host") or "").startswith("10.120."),
                "socks_proxy_bound": bool(str(metadata.get("socks_host") or "").strip())
                and int(metadata.get("socks_port") or 0) > 0,
                "remote_root_allowed": str(metadata.get("remote_workspace") or "").rstrip("/")
                == EXPECTED_REMOTE_ROOT,
                "host_uuid_present": bool(
                    str(metadata.get("expected_host_uuid") or metadata.get("host_uuid") or "").strip()
                ),
                "gpu_uuid_present": bool(
                    str(metadata.get("expected_gpu_uuid") or metadata.get("gpu_uuid") or "").strip()
                ),
            }
        )
        try:
            expected_binding = gpu_credentials.compute_container_binding_sha256(metadata)
            checks["container_binding_sha256_valid"] = (
                str(metadata.get("container_binding_sha256") or "") == expected_binding
            )
        except Exception:
            expected_binding = None
            checks["container_binding_sha256_valid"] = False
        known_hosts_name = str(metadata.get("known_hosts_path") or gpu_credentials.PROFILE_KNOWN_HOSTS_FILENAME)
        try:
            known_hosts_path = _safe_child(directory, known_hosts_name)
            known_hosts_present = known_hosts_path.is_file() and not known_hosts_path.is_symlink()
        except ValueError:
            known_hosts_path = directory / gpu_credentials.PROFILE_KNOWN_HOSTS_FILENAME
            known_hosts_present = False
        checks["known_hosts_present"] = known_hosts_present
        details.update(
            {
                "schema": metadata.get("schema"),
                "profile_state": metadata.get("profile_state"),
                "job_id": metadata.get("job_id"),
                "allocation_generation": metadata.get("allocation_generation"),
                "gateway_host": metadata.get("host"),
                "gateway_port": metadata.get("port"),
                "socks_host": metadata.get("socks_host"),
                "socks_port": metadata.get("socks_port"),
                "remote_workspace": metadata.get("remote_workspace"),
                "expected_host_uuid_present": checks["host_uuid_present"],
                "expected_gpu_uuid_present": checks["gpu_uuid_present"],
                "computed_container_binding_sha256": expected_binding,
                "known_hosts_present": known_hosts_present,
            }
        )

    failed = [name for name, ok in checks.items() if not ok]
    status = "ready" if not failed else "failed_closed"
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": status,
        "profile": selected,
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "boundaries": {
            "dpapi_decrypted": False,
            "ssh_connections": 0,
            "remote_commands": 0,
            "training_started": False,
            "grader_calls": 0,
            "kaggle_submissions": 0,
        },
        "claim_boundary": "local profile metadata readiness only; no remote/HPC connection proof",
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="job90353")
    parser.add_argument("--appdata", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_profile(args.profile, appdata=args.appdata)
    if args.output:
        output = write_json_atomic(args.output, report)
        report = {**report, "output": str(output), "output_sha256": sha256_file(output)}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
