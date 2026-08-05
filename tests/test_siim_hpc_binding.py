from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_os.siim_hpc_binding import (
    SiimHpcBindingError,
    binding_from_compute_policy,
    binding_from_environment,
    validate_binding,
)


def test_default_binding_preserves_existing_job89508(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOMIND_SIIM_HPC_JOB_ID", raising=False)
    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)

    binding = binding_from_environment()

    assert (binding.job_id, binding.credential_profile, binding.job_tag) == (
        89508,
        "job89508",
        "job89508",
    )


def test_profile_can_supply_the_allocation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOMIND_SIIM_HPC_JOB_ID", raising=False)
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job90353")

    binding = binding_from_environment()

    assert binding.job_id == 90353
    assert binding.schema("launch") == "evomind.siim.job90353.launch.v1"


def test_compute_policy_binding_is_durable_and_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVOMIND_SIIM_HPC_JOB_ID", raising=False)
    monkeypatch.delenv("EVOMIND_HPC_CREDENTIAL_PROFILE", raising=False)

    binding = binding_from_compute_policy(SimpleNamespace(job_id=90353, credential_profile="job90353"))

    assert binding.job_id == 90353
    assert binding.credential_profile == "job90353"


def test_explicit_policy_job_does_not_inherit_environment_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "89508")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job89508")

    binding = binding_from_compute_policy(SimpleNamespace(job_id=90353, credential_profile=None))

    assert binding.job_id == 90353
    assert binding.credential_profile == "job90353"


def test_explicit_policy_profile_can_restore_its_job_without_environment_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "89508")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "job89508")

    binding = binding_from_compute_policy(SimpleNamespace(job_id=None, credential_profile="job90353"))

    assert binding.job_id == 90353
    assert binding.credential_profile == "job90353"


@pytest.mark.parametrize(
    ("job_id", "profile"),
    [
        (90353, "job89508"),
        (0, "job0"),
        (90353, "../job90353"),
        (90353, ""),
    ],
)
def test_mismatched_or_unsafe_binding_fails_closed(job_id: int, profile: str) -> None:
    with pytest.raises(SiimHpcBindingError):
        validate_binding(job_id, profile)
