#!/usr/bin/env python3
"""Build a no-secret re-enrollment plan for the governed job90353 HPC profile.

The script is read-only.  It consumes the local profile-readiness report and
emits copyable, operator-facing commands for secure DPAPI re-enrollment and
identity bootstrap.  It never asks for, receives, logs, decrypts, or writes a
password; it never opens a socket or starts any remote action.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "evomind.hpc.job90353_reenrollment_plan.v1"
PROFILE_READINESS_SCHEMA = "evomind.hpc.profile_readiness.v1"
PROFILE = "job90353"
RUN_ID = "evomind_siim_isic_a800_job90353_20260730_095826"
REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
DEFAULT_READINESS = ROOT / "workspace" / "hpc" / "job90353_profile_readiness_current.json"
DEFAULT_OUTPUT = ROOT / "workspace" / "hpc" / "job90353_reenrollment_plan_current.json"
SAFE_BINDING = re.compile(r"^[a-z][a-z0-9._-]{7,95}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _default_binding_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"job90353-reenroll-{stamp}"


def build_plan(
    readiness_path: str | Path = DEFAULT_READINESS,
    *,
    allocation_binding_id: str | None = None,
    allocation_generation: int | None = None,
    cluster_name: str | None = None,
    image_name: str | None = None,
    environment_type: str | None = None,
    allocation_created_at: str | None = None,
    role_account: str | None = None,
    allocation_private_host: str | None = None,
    allocation_private_port: int | None = None,
) -> dict[str, Any]:
    path = Path(readiness_path).expanduser().resolve()
    readiness = _read_json(path)
    if readiness.get("schema") != PROFILE_READINESS_SCHEMA or readiness.get("profile") != PROFILE:
        raise ValueError("profile readiness evidence is not for job90353")
    failed = [str(item) for item in readiness.get("failed_checks") or []]
    details = readiness.get("details") if isinstance(readiness.get("details"), dict) else {}
    current_generation = details.get("allocation_generation")
    if allocation_generation is None:
        try:
            allocation_generation = max(1, int(current_generation or 0) + 1)
        except (TypeError, ValueError):
            allocation_generation = 1
    binding = allocation_binding_id or _default_binding_id()
    if not SAFE_BINDING.fullmatch(binding):
        raise ValueError("allocation binding id must be lowercase safe slug, length 8-96")
    account = str(role_account or "").strip()
    if account and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", account):
        raise ValueError("role account must be a safe account name without secrets")
    private_host = str(allocation_private_host or "").strip()
    if private_host and not re.fullmatch(r"10[.]120[.][0-9]{1,3}[.][0-9]{1,3}", private_host):
        raise ValueError("allocation private host must be a 10.120.x.x metadata address")
    private_port = int(allocation_private_port or 0)
    if private_port and not (1 <= private_port <= 65535):
        raise ValueError("allocation private port is out of range")

    must_reenroll = readiness.get("status") != "ready"
    plan_status = "ready_for_operator_secret_entry" if must_reenroll else "already_ready_no_reenrollment_needed"
    install_command = (
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        "scripts\\install_hpc_ssh_credential_from_stdin.ps1 "
        "-User " + (account or "<HPC_ROLE_ACCOUNT>") + " "
        "-Profile job90353 "
        "-HostName 100.85.169.63 "
        "-Port 1235 "
        "-SocksHost " + str(details.get("socks_host") or "127.0.0.1") + " "
        "-SocksPort " + str(int(details.get("socks_port") or 17897)) + " "
        "-AllocationBindingId " + binding + " "
        "-AllocationGeneration " + str(allocation_generation)
    )
    bootstrap_commands = [
        "$env:EVOMIND_HPC_CREDENTIAL_PROFILE='job90353'",
        "$env:EVOMIND_SIIM_RUN_ID='" + RUN_ID + "'",
        "python scripts/bootstrap_hpc_profile_identity.py",
        "python scripts/verify_hpc_profile_readiness.py --profile job90353 "
        "--output workspace/hpc/job90353_profile_readiness_current.json",
    ]
    resume_commands = [
        "python scripts/stage_siim_public_from_hpc.py --inventory-only "
        "--destination workspace/mle_ab_data_prepare_r3/official-wsl-v2/siim-isic-melanoma-classification/prepared/public",
        "python scripts/verify_mle_ab_campaign_readiness.py "
        "--campaign workspace/mle_ab_campaigns/evomind-mle-lite-screen-experience-mcgs-v1-r4-20260802 "
        "--data-root workspace/mle_ab_data_prepare_r3/official-wsl-v2 "
        "--execution-environment workspace/mle_ab_campaigns/evomind-mle-lite-screen-experience-mcgs-v1-r4-20260802/execution-environment-20260804T050148.json",
    ]
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": plan_status,
        "profile": PROFILE,
        "run_id": RUN_ID,
        "readiness_evidence": {
            "path": str(path),
            "status": readiness.get("status"),
            "failed_checks": failed,
            "claim_boundary": readiness.get("claim_boundary"),
        },
        "operator_inputs_required": [
            {
                "name": "profile-bound HPC role account username/password",
                "method": "secure PowerShell prompt or stdin pipe outside Codex logs",
                "logged_by_plan": False,
                "role_account_logged": bool(account),
            },
            {
                "name": "fresh allocation binding id",
                "value": binding,
                "logged_by_plan": True,
            },
            {
                "name": "allocation generation",
                "value": allocation_generation,
                "logged_by_plan": True,
            },
        ],
        "allocation_page_metadata": {
            "cluster_name": str(cluster_name or "") if cluster_name else None,
            "image_name": str(image_name or "") if image_name else None,
            "environment_type": str(environment_type or "") if environment_type else None,
            "allocation_created_at": str(allocation_created_at or "") if allocation_created_at else None,
            "role_account": account or None,
            "allocation_private_endpoint": (
                {"host": private_host, "port": private_port}
                if private_host and private_port
                else None
            ),
            "allocation_private_endpoint_policy": (
                "metadata_only_forbidden_off_campus_direct_target"
                if private_host
                else None
            ),
            "password_recorded": False,
        },
        "commands": {
            "install_provisioning_profile": install_command,
            "bootstrap_identity": bootstrap_commands,
            "post_bootstrap_verification": bootstrap_commands[-1],
            "resume_siim_prepared_data_gate_after_ready": resume_commands,
        },
        "acceptance_gates": [
            "install output status=installed and profile_state=provisioning",
            "bootstrap output profile_state=active and exactly one visible A800 GPU identity",
            "verify_hpc_profile_readiness status=ready with no failed_checks",
            "stage_siim_public_from_hpc --inventory-only writes a verified public_staging_inventory.json",
            "r4 readiness remains 0 completion, 0 grader, 0 Kaggle and advances SIIM prepared-data contract",
        ],
        "safety_boundaries": {
            "this_plan_decrypts_dpapi": False,
            "this_plan_opens_ssh": False,
            "this_plan_runs_remote_commands": False,
            "this_plan_starts_training": False,
            "this_plan_calls_grader": False,
            "this_plan_submits_kaggle": False,
            "allowed_remote_root": REMOTE_ROOT,
            "forbidden_direct_allocation_ip": "10.120.0.0/16",
        },
        "human_warning": (
            "Do not paste the HPC password into Codex, shell history, logs, reports, or videos. "
            "Run the install command interactively so PowerShell collects the secret as SecureString."
        ),
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--allocation-binding-id")
    parser.add_argument("--allocation-generation", type=int)
    parser.add_argument("--cluster-name")
    parser.add_argument("--image-name")
    parser.add_argument("--environment-type")
    parser.add_argument("--allocation-created-at")
    parser.add_argument("--role-account")
    parser.add_argument("--allocation-private-host")
    parser.add_argument("--allocation-private-port", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(
        args.readiness,
        allocation_binding_id=args.allocation_binding_id,
        allocation_generation=args.allocation_generation,
        cluster_name=args.cluster_name,
        image_name=args.image_name,
        environment_type=args.environment_type,
        allocation_created_at=args.allocation_created_at,
        role_account=args.role_account,
        allocation_private_host=args.allocation_private_host,
        allocation_private_port=args.allocation_private_port,
    )
    output = write_json_atomic(args.output, plan)
    print(json.dumps({"status": plan["status"], "output": str(output), "schema": plan["schema"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
