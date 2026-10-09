from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evomind_runtime.models import ToolResult
from evomind_runtime.run_secrets import RunSecretStore
from evomind_runtime.runtime import AgentRuntime


IDENTITY = {
    "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
    "owner_principal_id": "admin",
    "job_id": 92257,
    "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job92257_g21",
    "allocation_generation": 21,
    "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
    "allocation_binding_id": "aimslab-job92257-fixture",
}


def _waiting_weather_run(runtime: AgentRuntime, suffix: str) -> tuple[dict, dict]:
    run = runtime.assistant.create_run(
        prompt="fixture Weather controlled binding",
        conversation_id=f"secret_binding_{suffix}",
        managed_hpc_identity=IDENTITY,
        start=False,
    )
    waiting = runtime.invoke_tool(
        run["id"],
        "competition_data_accelerate",
        {"competition": "weather4cast", "timeout_seconds": 1800},
        idempotency_key=f"secret-binding-{suffix}",
    )
    runtime.store.update_assistant_run(run["id"], status="waiting_approval")
    return run, waiting


def _write_pending(
    runtime: AgentRuntime,
    secret_ref: str,
    *,
    identity: dict | None = None,
    expires_at: datetime | None = None,
) -> None:
    bound_identity = dict(identity or IDENTITY)
    digest = hashlib.sha256(secret_ref.encode("utf-8")).hexdigest()
    envelope = b'{"fixture":true}\n'
    pending = runtime.runtime_root / "run-secrets" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "evomind.run_secret_pending.v1",
        "state": "pending",
        "secret_ref_sha256": digest,
        "secret_id": "00000000-0000-4000-8000-000000000001",
        "purpose": "weather4cast_sftp",
        **bound_identity,
        "key_id": "0" * 64,
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "expires_at_utc": (expires_at or datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "envelope_sha256": hashlib.sha256(envelope).hexdigest(),
        "secret_values_echoed": False,
    }
    (pending / f"{digest}.json").write_text(json.dumps(record), encoding="utf-8")
    (pending / f"{digest}.envelope").write_bytes(envelope)


def _fixture_consumer(_envelope_path: Path, metadata_path: Path) -> bytes:
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    aad = "|".join(
        str(record.get(name) or "")
        for name in (
            "tenant_id",
            "owner_principal_id",
            "secret_id",
            "purpose",
            "job_id",
            "allocation_generation",
            "profile_instance_id",
            "key_id",
            "expires_at_utc",
        )
    )
    return json.dumps(
        {
            "schema": "evomind.run_secret_payload.v1",
            "purpose": "weather4cast_sftp",
            "binding_sha256": hashlib.sha256(aad.encode("utf-8")).hexdigest(),
        }
    ).encode("utf-8")


def test_fixed_run_attach_exact_approval_and_single_consume_without_remote_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run, waiting = _waiting_weather_run(runtime, "consume")
    secret_ref = "rs_" + "A" * 43
    _write_pending(runtime, secret_ref)

    attached = runtime.assistant.action(
        run["id"], "attach_secret_refs", "", [secret_ref], IDENTITY,
    )
    assert attached["id"] == run["id"]
    assert attached["status"] == "waiting_approval"
    assert attached["controlled_secret_availability"] == {"weather4cast_sftp": 1}
    assert secret_ref not in json.dumps(attached, sort_keys=True)
    assert all(secret_ref not in json.dumps(item, sort_keys=True) for item in runtime.store.list_events(run["id"]))
    assert runtime.store.get_approval(waiting["approval"]["id"])["status"] == "pending"
    assert runtime.store.get_tool_call(waiting["tool_call"]["id"])["status"] == "waiting_approval"

    invocations: list[dict] = []

    def fixture_handler(args: dict, context) -> ToolResult:
        invocations.append(dict(args))
        payload = RunSecretStore(context.runtime_root, consumer=_fixture_consumer).consume_for_run(
            context.session_id,
            purpose="weather4cast_sftp",
            managed_hpc_identity=dict(context.metadata["managed_hpc_identity"]),
        )
        return ToolResult("", True, {"purpose_ok": payload["purpose"] == "weather4cast_sftp"}, "fixture consumed")

    monkeypatch.setitem(runtime.registry._handlers, "competition_data_accelerate", fixture_handler)
    decision = runtime.decide_approval(waiting["approval"]["id"], True, "fixture exact approval")

    assert decision["status"] == "completed"
    assert invocations == [{"competition": "weather4cast", "timeout_seconds": 1800}]
    assert RunSecretStore(runtime.runtime_root).available_purposes_for_run(run["id"]) == {}
    with pytest.raises(FileNotFoundError, match="controlled Weather4cast secret"):
        RunSecretStore(runtime.runtime_root, consumer=_fixture_consumer).consume_for_run(
            run["id"], purpose="weather4cast_sftp", managed_hpc_identity=IDENTITY,
        )
    tombstones = list((runtime.runtime_root / "run-secrets" / "consumed" / run["id"]).glob("*.json"))
    assert len(tombstones) == 1
    tombstone = json.loads(tombstones[0].read_text(encoding="utf-8"))
    assert tombstone["status"] == "consumed"
    assert tombstone["secret_values_logged"] is False
    runtime.close()


def test_fixed_run_attach_retry_is_idempotent_and_does_not_duplicate_approval(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run, waiting = _waiting_weather_run(runtime, "retry")
    secret_ref = "rs_" + "B" * 43
    _write_pending(runtime, secret_ref)

    first = runtime.assistant.action(run["id"], "attach_secret_refs", "", [secret_ref], IDENTITY)
    second = runtime.assistant.action(run["id"], "attach_secret_refs", "", [secret_ref], IDENTITY)

    assert first["controlled_secret_availability"] == second["controlled_secret_availability"] == {"weather4cast_sftp": 1}
    assert len([item for item in second["approvals"] if item["status"] == "pending"]) == 1
    assert runtime.store.get_approval(waiting["approval"]["id"])["status"] == "pending"
    added = [item for item in runtime.store.list_events(run["id"]) if item["event_type"] == "run_secret_refs_added"]
    assert len(added) == 1
    runtime.close()


def test_fixed_run_secret_rejects_identity_expiry_cross_run_and_settled_approval(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run, waiting = _waiting_weather_run(runtime, "reject")

    for marker, drift in (
        ("C", {"allocation_generation": 20}),
        ("G", {"tenant_id": "tenant_bbbbbbbbbbbbbbbbbbbbbbbb"}),
    ):
        wrong_identity_ref = "rs_" + marker * 43
        _write_pending(runtime, wrong_identity_ref, identity={**IDENTITY, **drift})
        with pytest.raises(ValueError, match="mismatched or expired"):
            runtime.assistant.action(run["id"], "attach_secret_refs", "", [wrong_identity_ref], IDENTITY)

    expired_ref = "rs_" + "D" * 43
    _write_pending(runtime, expired_ref, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(ValueError, match="mismatched or expired"):
        runtime.assistant.action(run["id"], "attach_secret_refs", "", [expired_ref], IDENTITY)

    valid_ref = "rs_" + "E" * 43
    _write_pending(runtime, valid_ref)
    runtime.assistant.action(run["id"], "attach_secret_refs", "", [valid_ref], IDENTITY)
    other_run, _ = _waiting_weather_run(runtime, "cross_run")
    assert not RunSecretStore(runtime.runtime_root).claimed_ref_is_live(
        other_run["id"], valid_ref, managed_hpc_identity=IDENTITY, purpose="weather4cast_sftp",
    )
    with pytest.raises(FileNotFoundError, match="controlled Weather4cast secret"):
        RunSecretStore(runtime.runtime_root, consumer=_fixture_consumer).consume_for_run(
            other_run["id"], purpose="weather4cast_sftp", managed_hpc_identity=IDENTITY,
        )

    runtime.store.decide_approval(waiting["approval"]["id"], False, "fixture settled")
    another_ref = "rs_" + "F" * 43
    _write_pending(runtime, another_ref)
    with pytest.raises(ValueError, match="requires_one_pending_approval"):
        runtime.assistant.action(run["id"], "attach_secret_refs", "", [another_ref], IDENTITY)
    runtime.close()
