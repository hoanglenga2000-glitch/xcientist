from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evomind_runtime.credential_leases import (
    CredentialLeaseManager,
    CredentialLeaseState,
    JsonLeaseBackend,
    PreflightEvidence,
)


def _expiry(minutes: int = 30) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _evidence(**changes):
    values = {
        "connector_version_sha256": "a" * 64,
        "adapter_sha256": "b" * 64,
        "destination_capability_id": "directory_demo",
        "destination_write_verified": True,
        "runtime_dependencies_verified": True,
        "parser_verified": True,
        "time_contract_verified": True,
        "evidence_sha256": "c" * 64,
    }
    values.update(changes)
    return PreflightEvidence(**values)


def test_lease_requires_preflight_before_decryption_and_never_projects_ref(tmp_path):
    manager = CredentialLeaseManager(JsonLeaseBackend(tmp_path))
    lease = manager.create(
        purpose="sftp",
        tenant_id="tenant_a",
        project_id="project_a",
        credential_ref="opaque-secret-reference",
        expires_at_utc=_expiry(),
    )
    assert "credential_ref_sha256" not in lease.public_dict()
    manager.transition(lease.lease_id, CredentialLeaseState.LEASED)
    with pytest.raises(ValueError, match="preflight"):
        manager.transition(lease.lease_id, CredentialLeaseState.CONNECTOR_PREFLIGHTED)
    manager.transition(lease.lease_id, CredentialLeaseState.CONNECTOR_PREFLIGHTED, preflight=_evidence())
    decrypted = manager.transition(lease.lease_id, CredentialLeaseState.DECRYPTED)
    assert decrypted.plaintext_exposed is True
    assert decrypted.safe_to_retry is False


def test_pre_decryption_failure_remains_retryable_but_post_decryption_revokes(tmp_path):
    manager = CredentialLeaseManager(JsonLeaseBackend(tmp_path))
    lease = manager.create(
        purpose="sftp",
        tenant_id="tenant_a",
        project_id="project_a",
        credential_ref="ref-a",
        expires_at_utc=_expiry(),
    )
    manager.transition(lease.lease_id, CredentialLeaseState.LEASED)
    retryable = manager.fail(lease.lease_id, failure_signature="parser_missing")
    assert retryable.state == "leased"
    assert retryable.safe_to_retry is True
    manager.transition(lease.lease_id, CredentialLeaseState.CONNECTOR_PREFLIGHTED, preflight=_evidence())
    manager.transition(lease.lease_id, CredentialLeaseState.DECRYPTED)
    revoked = manager.fail(lease.lease_id, failure_signature="unknown_after_decrypt")
    assert revoked.state == "revoked"
    assert revoked.safe_to_retry is False


def test_incomplete_preflight_and_terminal_replay_fail_closed(tmp_path):
    manager = CredentialLeaseManager(JsonLeaseBackend(tmp_path))
    lease = manager.create(
        purpose="api",
        tenant_id="tenant_a",
        project_id="project_a",
        credential_ref="ref-b",
        expires_at_utc=_expiry(),
    )
    manager.transition(lease.lease_id, CredentialLeaseState.LEASED)
    with pytest.raises(ValueError, match="incomplete"):
        manager.transition(
            lease.lease_id,
            CredentialLeaseState.CONNECTOR_PREFLIGHTED,
            preflight=_evidence(parser_verified=False),
        )
    manager.transition(lease.lease_id, CredentialLeaseState.REVOKED)
    with pytest.raises(ValueError, match="invalid"):
        manager.transition(lease.lease_id, CredentialLeaseState.LEASED)
