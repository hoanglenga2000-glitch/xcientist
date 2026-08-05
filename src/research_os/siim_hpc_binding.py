"""Validated, durable allocation binding for the governed SIIM workflow."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_JOB_ID = 89508
SAFE_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
JOB_PROFILE = re.compile(r"^job([1-9][0-9]{0,8})$")


class SiimHpcBindingError(ValueError):
    pass


@dataclass(frozen=True)
class SiimHpcBinding:
    job_id: int
    credential_profile: str

    @property
    def job_tag(self) -> str:
        return f"job{self.job_id}"

    def schema(self, name: str) -> str:
        if not name or not re.fullmatch(r"[a-z0-9_]+", name):
            raise SiimHpcBindingError("invalid SIIM schema suffix")
        return f"evomind.siim.{self.job_tag}.{name}.v1"


def validate_binding(job_id: Any, credential_profile: Any = None) -> SiimHpcBinding:
    try:
        selected_job = int(job_id)
    except (TypeError, ValueError) as exc:
        raise SiimHpcBindingError("SIIM HPC job_id must be an integer") from exc
    if selected_job <= 0 or selected_job > 999_999_999:
        raise SiimHpcBindingError("SIIM HPC job_id is outside the supported range")
    expected_profile = f"job{selected_job}"
    selected_profile = expected_profile if credential_profile is None else str(credential_profile).strip()
    if (
        selected_profile in {".", ".."}
        or not SAFE_PROFILE.fullmatch(selected_profile)
        or selected_profile != expected_profile
    ):
        raise SiimHpcBindingError("SIIM credential profile must exactly match the bound job_id")
    return SiimHpcBinding(selected_job, selected_profile)


def binding_from_environment() -> SiimHpcBinding:
    raw_job = os.environ.get("EVOMIND_SIIM_HPC_JOB_ID", "").strip()
    raw_profile = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE", "").strip()
    if not raw_job and raw_profile:
        match = JOB_PROFILE.fullmatch(raw_profile)
        if match:
            raw_job = match.group(1)
    return validate_binding(raw_job or DEFAULT_JOB_ID, raw_profile or None)


def binding_from_compute_policy(policy: Any) -> SiimHpcBinding:
    job_id = getattr(policy, "job_id", None)
    profile = getattr(policy, "credential_profile", None)
    if job_id is not None:
        # An allocation persisted in the request is authoritative.  Derive its
        # canonical profile locally when the older/partial payload omitted it;
        # never combine that allocation with a profile from the current process.
        return validate_binding(job_id, profile)
    if profile is not None:
        selected_profile = str(profile).strip()
        match = JOB_PROFILE.fullmatch(selected_profile)
        if not match:
            raise SiimHpcBindingError("SIIM credential profile must encode a canonical job_id")
        return validate_binding(match.group(1), selected_profile)
    return binding_from_environment()


__all__ = [
    "DEFAULT_JOB_ID",
    "SiimHpcBinding",
    "SiimHpcBindingError",
    "binding_from_compute_policy",
    "binding_from_environment",
    "validate_binding",
]
