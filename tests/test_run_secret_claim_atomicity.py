from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evomind_runtime.run_secrets import RunSecretStore


def test_claim_refs_rolls_back_the_current_partial_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunSecretStore(tmp_path)
    run_id = "run_" + "a" * 32
    secret_ref = "rs_" + "A" * 43
    identity = {
        "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
        "owner_principal_id": "admin",
        "job_id": 92257,
        "allocation_generation": 21,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job92257-fixture",
        "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job92257_g21",
    }
    pending_meta, pending_envelope = store._pending_paths(secret_ref)
    claimed_meta, claimed_envelope = store._claimed_paths(run_id, secret_ref)
    pending_meta.parent.mkdir(parents=True, exist_ok=True)
    envelope = b'{"encrypted":"fixture"}\n'
    pending_envelope.write_bytes(envelope)
    pending_meta.write_text(json.dumps({
        "schema": "evomind.run_secret_pending.v1",
        "state": "pending",
        "purpose": "weather4cast_sftp",
        "secret_ref_sha256": hashlib.sha256(secret_ref.encode()).hexdigest(),
        "envelope_sha256": hashlib.sha256(envelope).hexdigest(),
        "expires_at_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        **identity,
    }) + "\n", encoding="utf-8")
    original_unlink = Path.unlink
    failed_once = False

    def fail_pending_metadata_unlink(path: Path, *args, **kwargs):
        nonlocal failed_once
        if path == pending_meta and not failed_once:
            failed_once = True
            raise OSError("fixture unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending_metadata_unlink)
    with pytest.raises(OSError, match="fixture unlink failure"):
        store.claim_refs(run_id, [secret_ref], managed_hpc_identity=identity)

    restored = json.loads(pending_meta.read_text(encoding="utf-8"))
    assert restored["schema"] == "evomind.run_secret_pending.v1"
    assert restored["state"] == "pending"
    assert "run_id" not in restored and "claimed_at_utc" not in restored
    assert pending_envelope.read_bytes() == envelope
    assert not claimed_meta.exists()
    assert not claimed_envelope.exists()
