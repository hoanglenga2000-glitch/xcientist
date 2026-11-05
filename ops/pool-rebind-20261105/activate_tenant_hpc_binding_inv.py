from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research_agent_workstation.server.core import gpu_credentials as gc

TENANT = re.compile(r"^tenant_[a-f0-9]{24}$")
PROFILE_INSTANCE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SHA256 = re.compile(r"^[a-f0-9]{64}$")
RECEIPT_SCHEMA = "evomind.tenant_hpc_activation_receipt.v1"
REMOTE_ROOT = gc.ALLOWED_GPU_REMOTE_ROOT


class ActivationError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationError(f"{label}_unreadable") from exc
    if not isinstance(value, dict):
        raise ActivationError(f"{label}_invalid")
    return value


def _atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = _json_bytes(value)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_prepared(prepared: Path, target: Path) -> None:
    try:
        os.link(prepared, target)
    except FileExistsError as exc:
        raise ActivationError("activation_receipt_already_exists") from exc
    prepared.unlink(missing_ok=True)


def _profile_name(tenant_id: str, job_id: int, generation: int) -> str:
    return f"{tenant_id}_job{job_id}_g{generation}"


def _validate_args(args: argparse.Namespace) -> str:
    if not TENANT.fullmatch(args.tenant_id):
        raise ActivationError("tenant_id_invalid")
    if args.job_id <= 0 or args.allocation_generation <= 0:
        raise ActivationError("allocation_identity_invalid")
    if not PROFILE_INSTANCE.fullmatch(args.profile_instance_id):
        raise ActivationError("profile_instance_id_invalid")
    if not args.state_root.is_absolute():
        raise ActivationError("state_root_must_be_absolute")
    profile = _profile_name(args.tenant_id, args.job_id, args.allocation_generation)
    if os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE") != profile:
        raise ActivationError("profile_environment_binding_mismatch")
    if os.environ.get("EVOMIND_SIIM_HPC_JOB_ID") != str(args.job_id):
        raise ActivationError("job_environment_binding_mismatch")
    return profile


def _validate_metadata(
    metadata: dict[str, Any], args: argparse.Namespace, profile: str
) -> str:
    if (
        metadata.get("schema") != gc.PROFILE_METADATA_SCHEMA
        or metadata.get("credential_profile") != profile
        or int(metadata.get("job_id") or 0) != args.job_id
        or int(metadata.get("allocation_generation") or 0) != args.allocation_generation
        or metadata.get("profile_instance_id") != args.profile_instance_id
        or metadata.get("tenant_id") != args.tenant_id
        or not str(metadata.get("owner_principal_id") or "")
        or not gc.SAFE_ALLOCATION_BINDING_ID.fullmatch(
            str(metadata.get("allocation_binding_id") or "")
        )
    ):
        raise ActivationError("tenant_profile_activation_binding_invalid")
    return str(metadata["owner_principal_id"])


def _validate_receipt(
    receipt: dict[str, Any],
    *,
    args: argparse.Namespace,
    profile: str,
    owner: str,
) -> None:
    gpu_uuids = receipt.get("gpu_uuids")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("tenant_id") != args.tenant_id
        or receipt.get("owner_principal_id") != owner
        or int(receipt.get("job_id") or 0) != args.job_id
        or receipt.get("credential_profile") != profile
        or int(receipt.get("allocation_generation") or 0) != args.allocation_generation
        or receipt.get("profile_instance_id") != args.profile_instance_id
        or not gc.SAFE_ALLOCATION_BINDING_ID.fullmatch(
            str(receipt.get("allocation_binding_id") or "")
        )
        or not SHA256.fullmatch(str(receipt.get("container_binding_sha256") or ""))
        or not str(receipt.get("host_uuid") or "").strip()
        or not isinstance(gpu_uuids, list)
        or not gpu_uuids
        or any(not isinstance(value, str) or not value.strip() for value in gpu_uuids)
        or len({value.strip().lower() for value in gpu_uuids}) != len(gpu_uuids)
        or receipt.get("remote_root") != REMOTE_ROOT
        or receipt.get("designated_proxy_path_verified") is not True
        or receipt.get("job_container_verified") is not True
        or receipt.get("five_of_five_identity_samples") is not True
        or receipt.get("connection_ready") is not True
    ):
        raise ActivationError("activation_receipt_invalid")


def _metadata_matches_receipt(metadata: dict[str, Any], receipt: dict[str, Any]) -> bool:
    return (
        metadata.get("profile_state") == gc.PROFILE_STATE_ACTIVE
        and metadata.get("allocation_binding_id") == receipt.get("allocation_binding_id")
        and metadata.get("expected_host_uuid") == receipt.get("host_uuid")
        and metadata.get("expected_gpu_uuid") == ",".join(receipt.get("gpu_uuids") or [])
        and metadata.get("remote_workspace") == receipt.get("remote_root")
        and metadata.get("container_binding_sha256") == receipt.get("container_binding_sha256")
        and gc.compute_container_binding_sha256(metadata)
        == receipt.get("container_binding_sha256")
    )


def _sample_identity(
    config: gc.GpuSshConfig,
    *,
    args: argparse.Namespace,
    connector: Callable[..., Any],
    verifier: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in range(5):
        client = connector(config, timeout=args.timeout)
        try:
            sample = verifier(client, config, expected_job_id=args.job_id)
        finally:
            client.close()
        if (
            sample.get("designated_proxy_path_verified") is not True
            or sample.get("job_container_verified") is not True
            or sample.get("signals_sent") != 0
            or sample.get("other_processes_modified") is not False
        ):
            raise ActivationError("current_five_of_five_activation_failed")
        sample = dict(sample)
        sample["sample_index"] = index + 1
        sample["sampled_at_utc"] = _utc_now().isoformat()
        samples.append(sample)
        if index < 4 and args.interval_seconds:
            time.sleep(args.interval_seconds)
    identity_keys = (
        "host_uuid",
        "gpu_uuids",
        "gpu_count",
        "gpu_names",
        "gpu_name",
        "gpu_memory_per_device_mib",
        "gpu_memory_total_mib",
        "remote_root",
    )
    reference = {key: samples[0].get(key) for key in identity_keys}
    if any({key: sample.get(key) for key in identity_keys} != reference for sample in samples[1:]):
        raise ActivationError("five_of_five_identity_drift")
    return samples


def activate_tenant_binding(
    args: argparse.Namespace,
    *,
    lock_factory: Callable[[Path], Any] = gc.profile_lifecycle_lock,
    config_loader: Callable[[], gc.GpuSshConfig] = gc.load_gpu_ssh_config_for_identity_bootstrap_locked,
    connector: Callable[..., Any] = gc.connect_ssh_for_identity_bootstrap,
    verifier: Callable[..., dict[str, Any]] = gc.verify_job_container_identity,
    publisher: Callable[[Path, Path], None] = _publish_prepared,
) -> dict[str, Any]:
    profile = _validate_args(args)
    profile_dir = gc._credential_state_dir(profile)
    metadata_path = profile_dir / gc.PROFILE_METADATA_FILENAME
    tenant_root = args.state_root / "tenants" / args.tenant_id
    evidence_path = tenant_root / "activation-evidence" / f"g{args.allocation_generation}.json"
    receipt_dir = tenant_root / "activation-receipts"
    target = receipt_dir / f"g{args.allocation_generation}.json"
    prepared = receipt_dir / f"g{args.allocation_generation}.prepared.json"

    with lock_factory(profile_dir):
        metadata = _read_json(metadata_path, "profile_metadata")
        owner = _validate_metadata(metadata, args, profile)
        if target.exists():
            receipt = _read_json(target, "activation_receipt")
            _validate_receipt(receipt, args=args, profile=profile, owner=owner)
            if not _metadata_matches_receipt(metadata, receipt):
                raise ActivationError("active_metadata_receipt_mismatch")
            prepared.unlink(missing_ok=True)
            return receipt

        if prepared.exists():
            receipt = _read_json(prepared, "prepared_activation_receipt")
            _validate_receipt(receipt, args=args, profile=profile, owner=owner)
            if metadata.get("profile_state") == gc.PROFILE_STATE_ACTIVE:
                if not _metadata_matches_receipt(metadata, receipt):
                    raise ActivationError("prepared_receipt_metadata_mismatch")
                publisher(prepared, target)
                return receipt
            if metadata.get("profile_state") != gc.PROFILE_STATE_PROVISIONING:
                raise ActivationError("profile_state_not_recoverable")
            prepared.unlink()

        if metadata.get("profile_state") != gc.PROFILE_STATE_PROVISIONING:
            raise ActivationError("profile_is_not_provisioning")

        config = config_loader()
        if (
            not config.strict_named_profile
            or config.profile_state != gc.PROFILE_STATE_PROVISIONING
            or config.credential_profile != profile
            or config.profile_instance_id != args.profile_instance_id
        ):
            raise ActivationError("provisioning_loader_binding_invalid")
        samples = _sample_identity(
            config,
            args=args,
            connector=connector,
            verifier=verifier,
        )
        sample = samples[-1]
        activated_at = _utc_now().isoformat()
        active_metadata = dict(metadata)
        active_metadata.update(
            {
                "profile_state": gc.PROFILE_STATE_ACTIVE,
                "profile_state_reason": "current_five_of_five_identity_activation_verified",
                "lifecycle_revision": int(metadata.get("lifecycle_revision") or 0) + 1,
                "state_changed_at": activated_at,
                "activated_at": activated_at,
                "expected_host_uuid": sample["host_uuid"],
                "expected_gpu_uuid": ",".join(sample["gpu_uuids"]),
                # 2026-11-05: freeze the observed GPU inventory so later sessions verify model/memory per UUID (non-A800 cards).
                "expected_gpu_inventory": gc.normalize_gpu_inventory(sample.get("gpu_inventory"), required=True),
            }
        )
        active_metadata["container_binding_sha256"] = gc.compute_container_binding_sha256(
            active_metadata
        )
        evidence = {
            "schema": "evomind.hpc.windows_cloud_readiness.v1",
            "status": "job_container_verified",
            "job_id": args.job_id,
            "credential_profile": profile,
            "required_samples": 5,
            "samples": samples,
            "designated_proxy_path_verified": True,
            "job_container_verified": True,
            "five_of_five_identity_samples": True,
            "connection_ready": True,
            "training_started": False,
            "signals_sent": 0,
            "other_processes_modified": False,
        }
        _atomic_json(evidence_path, evidence)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "tenant_id": args.tenant_id,
            "owner_principal_id": owner,
            "job_id": args.job_id,
            "credential_profile": profile,
            "allocation_binding_id": active_metadata["allocation_binding_id"],
            "allocation_generation": args.allocation_generation,
            "profile_instance_id": args.profile_instance_id,
            "container_binding_sha256": active_metadata["container_binding_sha256"],
            "host_uuid": sample["host_uuid"],
            "gpu_uuids": sample["gpu_uuids"],
            "gpu_count": sample["gpu_count"],
            "gpu_names": sample["gpu_names"],
            "gpu_name": sample["gpu_name"],
            "gpu_memory_per_device_mib": sample["gpu_memory_per_device_mib"],
            "gpu_memory_total_mib": sample["gpu_memory_total_mib"],
            "remote_root": sample["remote_root"],
            "designated_proxy_path_verified": True,
            "job_container_verified": True,
            "five_of_five_identity_samples": True,
            "connection_ready": True,
            "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "activated_at_utc": activated_at,
        }
        _atomic_json(prepared, receipt, exclusive=True)
        _atomic_json(metadata_path, active_metadata)
        publisher(prepared, target)
        return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--allocation-generation", required=True, type=int)
    parser.add_argument("--profile-instance-id", required=True)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = activate_tenant_binding(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "code": str(exc), "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
