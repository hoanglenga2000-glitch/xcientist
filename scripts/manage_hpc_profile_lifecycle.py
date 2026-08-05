#!/usr/bin/env python3
"""Freeze or retire a named EvoMind HPC DPAPI profile without decrypting it.

Lifecycle transitions are local-only.  A tombstone is created before metadata
is replaced so every runtime loader fails closed even if the process stops
halfway through the transition.  Tombstones are never removed or overwritten
by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_agent_workstation.server.core.gpu_credentials import (  # noqa: E402
    PROFILE_FROZEN_TOMBSTONE_FILENAME,
    PROFILE_METADATA_FILENAME,
    PROFILE_METADATA_SCHEMA,
    PROFILE_RETIRED_TOMBSTONE_FILENAME,
    PROFILE_STATE_ACTIVE,
    PROFILE_STATE_FROZEN,
    PROFILE_STATE_PROVISIONING,
    PROFILE_STATE_RETIRED,
    PROFILE_STATES,
    SAFE_ALLOCATION_BINDING_ID,
    SAFE_CREDENTIAL_PROFILE,
    CredentialError,
    compute_container_binding_sha256,
    profile_lifecycle_lock,
)

SAFE_REASON = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")


class LifecycleError(RuntimeError):
    """Raised when a lifecycle transition would weaken profile isolation."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _marker_present(path: Path) -> bool:
    try:
        return path.exists() or path.is_symlink()
    except OSError:
        return True


def _profile_dir(profile: str, *, appdata: Path | None = None) -> Path:
    if profile in {".", ".."} or not SAFE_CREDENTIAL_PROFILE.fullmatch(profile):
        raise LifecycleError("profile contains unsafe characters")
    if not re.fullmatch(r"job[1-9][0-9]*", profile):
        raise LifecycleError("lifecycle commands require job<job_id> profile naming")
    root = appdata
    if root is None:
        raw = str(os.environ.get("APPDATA") or "").strip()
        if not raw:
            raise LifecycleError("APPDATA is required")
        root = Path(raw).expanduser()
    if not root.is_absolute():
        raise LifecycleError("APPDATA must be absolute")
    return (root / "ResearchAgentWorkstation" / "profiles" / profile).resolve(
        strict=False
    )


def _managed_profile_path(profile_dir: Path, name: str) -> Path:
    try:
        candidate = (profile_dir / name).resolve(strict=False)
        candidate.relative_to(profile_dir.resolve(strict=False))
    except (OSError, ValueError) as exc:
        raise LifecycleError("profile artifact escaped the managed profile directory") from exc
    return candidate


def _load_metadata(profile: str, profile_dir: Path) -> tuple[dict[str, Any], bytes]:
    path = _managed_profile_path(profile_dir, PROFILE_METADATA_FILENAME)
    try:
        original = path.read_bytes()
        payload = json.loads(original.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError("profile metadata is unreadable") from exc
    if not isinstance(payload, dict):
        raise LifecycleError("profile metadata must be a JSON object")
    if payload.get("schema") != PROFILE_METADATA_SCHEMA:
        raise LifecycleError(
            "legacy profile metadata is quarantined; secure re-enrollment is required"
        )
    if payload.get("credential_profile") != profile:
        raise LifecycleError("profile metadata binding changed")
    expected_job_id = int(profile.removeprefix("job"))
    try:
        job_id = int(payload.get("job_id") or 0)
        generation = int(payload.get("allocation_generation") or 0)
        revision = int(payload.get("lifecycle_revision") or 0)
        instance_id = str(uuid.UUID(str(payload.get("profile_instance_id") or "")))
    except (TypeError, ValueError, AttributeError) as exc:
        raise LifecycleError("profile lifecycle identity is invalid") from exc
    binding_id = str(payload.get("allocation_binding_id") or "").strip()
    if (
        job_id != expected_job_id
        or generation <= 0
        or revision <= 0
        or instance_id != str(payload.get("profile_instance_id") or "").lower()
        or not SAFE_ALLOCATION_BINDING_ID.fullmatch(binding_id)
    ):
        raise LifecycleError("profile lifecycle identity is invalid")
    state = str(payload.get("profile_state") or "")
    if state not in PROFILE_STATES:
        raise LifecycleError("profile lifecycle state is invalid")
    if state == PROFILE_STATE_ACTIVE:
        digest = str(payload.get("container_binding_sha256") or "")
        try:
            expected_digest = compute_container_binding_sha256(payload)
        except (TypeError, ValueError) as exc:
            raise LifecycleError("active profile container binding fields are invalid") from exc
        if digest != expected_digest:
            raise LifecycleError("active profile container binding identity changed")
    return payload, original


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise LifecycleError("profile lifecycle tombstone already exists") from exc


def _stage_metadata(path: Path, payload: bytes) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _transition_profile_unlocked(
    *,
    profile: str,
    target_state: str,
    reason: str,
    now: datetime | None = None,
    appdata: Path | None = None,
) -> dict[str, Any]:
    """Apply one irreversible freeze/retire transition and return audit facts."""

    if target_state not in {PROFILE_STATE_FROZEN, PROFILE_STATE_RETIRED}:
        raise LifecycleError("target state must be frozen or retired")
    if not SAFE_REASON.fullmatch(reason):
        raise LifecycleError("reason must be a stable lowercase reason code")
    directory = _profile_dir(profile, appdata=appdata)
    metadata_path = _managed_profile_path(directory, PROFILE_METADATA_FILENAME)
    metadata, original = _load_metadata(profile, directory)
    previous_state = str(metadata["profile_state"])
    frozen_path = directory / PROFILE_FROZEN_TOMBSTONE_FILENAME
    retired_path = directory / PROFILE_RETIRED_TOMBSTONE_FILENAME
    if target_state == PROFILE_STATE_FROZEN:
        if previous_state not in {PROFILE_STATE_PROVISIONING, PROFILE_STATE_ACTIVE}:
            raise LifecycleError("profile cannot transition to frozen from its current state")
        if _marker_present(frozen_path) or _marker_present(retired_path):
            raise LifecycleError("profile lifecycle tombstone already exists")
        tombstone_path = frozen_path
    else:
        if previous_state not in {
            PROFILE_STATE_PROVISIONING,
            PROFILE_STATE_ACTIVE,
            PROFILE_STATE_FROZEN,
        }:
            raise LifecycleError("profile cannot transition to retired from its current state")
        if _marker_present(retired_path):
            raise LifecycleError("profile retirement tombstone already exists")
        if previous_state == PROFILE_STATE_FROZEN and (
            not frozen_path.is_file() or frozen_path.is_symlink()
        ):
            raise LifecycleError("frozen profile is missing its immutable tombstone")
        if previous_state != PROFILE_STATE_FROZEN and _marker_present(frozen_path):
            raise LifecycleError("profile metadata and freeze tombstone disagree")
        tombstone_path = retired_path

    changed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    changed_at_text = changed_at.isoformat(timespec="seconds").replace("+00:00", "Z")
    revision = int(metadata["lifecycle_revision"]) + 1
    tombstone = {
        "schema": "evomind.hpc.profile_tombstone.v1",
        "credential_profile": profile,
        "job_id": int(metadata["job_id"]),
        "profile_instance_id": metadata["profile_instance_id"],
        "allocation_binding_id": metadata["allocation_binding_id"],
        "allocation_generation": int(metadata["allocation_generation"]),
        "previous_state": previous_state,
        "profile_state": target_state,
        "reason": reason,
        "lifecycle_revision": revision,
        "created_at": changed_at_text,
        "metadata_sha256_before": _sha256(original),
        "credential_decrypted": False,
        "network_accessed": False,
    }
    tombstone["tombstone_id"] = _sha256(_canonical_json_bytes(tombstone))
    tombstone_payload = json.dumps(
        tombstone,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    updated = dict(metadata)
    updated.update(
        {
            "profile_state": target_state,
            "profile_state_reason": reason,
            "state_changed_at": changed_at_text,
            "updated_at": changed_at_text,
            "lifecycle_revision": revision,
        }
    )
    updated[f"{target_state}_at"] = changed_at_text
    tombstone_hashes = dict(updated.get("tombstone_sha256") or {})
    tombstone_hashes[tombstone_path.name] = _sha256(tombstone_payload)
    updated["tombstone_sha256"] = tombstone_hashes
    metadata_payload = json.dumps(
        updated,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    staged_metadata = _stage_metadata(metadata_path, metadata_payload)
    marker_created = False
    try:
        _write_exclusive(tombstone_path, tombstone_payload)
        marker_created = True
        os.replace(staged_metadata, metadata_path)
    except Exception:
        staged_metadata.unlink(missing_ok=True)
        if marker_created:
            tombstone_path.unlink(missing_ok=True)
        raise
    finally:
        staged_metadata.unlink(missing_ok=True)

    return {
        "status": target_state,
        "credential_profile": profile,
        "job_id": int(metadata["job_id"]),
        "profile_instance_id": metadata["profile_instance_id"],
        "allocation_binding_id": metadata["allocation_binding_id"],
        "allocation_generation": int(metadata["allocation_generation"]),
        "previous_state": previous_state,
        "profile_state": target_state,
        "reason": reason,
        "lifecycle_revision": revision,
        "tombstone": tombstone_path.name,
        "tombstone_sha256": _sha256(tombstone_payload),
        "credential_decrypted": False,
        "network_accessed": False,
    }


def transition_profile(
    *,
    profile: str,
    target_state: str,
    reason: str,
    now: datetime | None = None,
    appdata: Path | None = None,
) -> dict[str, Any]:
    """Serialize and apply one irreversible local lifecycle transition."""

    directory = _profile_dir(profile, appdata=appdata)
    try:
        with profile_lifecycle_lock(directory):
            return _transition_profile_unlocked(
                profile=profile,
                target_state=target_state,
                reason=reason,
                now=now,
                appdata=appdata,
            )
    except CredentialError as exc:
        raise LifecycleError("profile lifecycle lock failed") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, target_state in (
        ("freeze", PROFILE_STATE_FROZEN),
        ("retire", PROFILE_STATE_RETIRED),
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--profile", required=True)
        subparser.add_argument("--reason", required=True)
        subparser.set_defaults(target_state=target_state)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = transition_profile(
            profile=args.profile,
            target_state=args.target_state,
            reason=args.reason,
        )
    except LifecycleError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    except Exception:
        print(
            json.dumps(
                {"status": "error", "message": "unexpected lifecycle transition failure"},
                ensure_ascii=False,
            )
        )
        return 3
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
