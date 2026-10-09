from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import evomind_runtime.run_secrets as run_secrets_module
from evomind_runtime.run_secrets import RunSecretStore


IDENTITY = {
    "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
    "owner_principal_id": "fixture-owner",
    "job_id": 91805,
    "allocation_generation": 19,
    "profile_instance_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    "allocation_binding_id": "allocation-fixture-g19",
    "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job91805_g19",
}


def _seed_pending(root: Path, secret_ref: str, *, expires: datetime | None = None, identity: dict | None = None, purpose: str = "weather4cast_sftp") -> dict:
    runtime_root = root / "runtime"
    pending = runtime_root / "run-secrets" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(secret_ref.encode("utf-8")).hexdigest()
    envelope = b'{"schema":"fixture-envelope"}\n'
    envelope_path = pending / f"{digest}.envelope"
    envelope_path.write_bytes(envelope)
    values = identity or IDENTITY
    record = {
        "schema": "evomind.run_secret_pending.v1",
        "state": "pending",
        "secret_ref_sha256": digest,
        "secret_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "purpose": purpose,
        **values,
        "key_id": "c" * 64,
        "expires_at_utc": (expires or (datetime.now(timezone.utc) + timedelta(minutes=5))).isoformat(),
        "envelope_sha256": hashlib.sha256(envelope).hexdigest(),
    }
    (pending / f"{digest}.json").write_text(json.dumps(record), encoding="utf-8")
    return {"runtime_root": runtime_root, "digest": digest, "record": record}


def _payload(record: dict, **overrides) -> bytes:
    aad = "|".join(
        str(record[name])
        for name in (
            "tenant_id", "owner_principal_id", "secret_id", "purpose", "job_id",
            "allocation_generation", "profile_instance_id", "key_id", "expires_at_utc",
        )
    )
    value = {
        "schema": "evomind.run_secret_payload.v1",
        "purpose": record["purpose"],
        "binding_sha256": hashlib.sha256(aad.encode("utf-8")).hexdigest(),
        **({
            "host": "sftp.example.invalid",
            "port": 22,
            "username": "fixture-user",
            "password": "fixture-secret",
            "remote_path": "/official",
            "target_subdir": "official",
        } if record["purpose"] == "weather4cast_sftp" else {
            "username": "fixture-user",
            "token": "fixture-token",
        }),
        **overrides,
    }
    return json.dumps(value).encode("utf-8")


def test_run_secret_is_claimed_consumed_once_and_tombstoned_without_values(tmp_path: Path) -> None:
    # The fixture has no managed data root, so it retains the explicit runtime root.
    secret_ref = "rs_" + "A" * 43
    seeded = _seed_pending(tmp_path, secret_ref)
    store = RunSecretStore(seeded["runtime_root"], consumer=lambda _envelope, _metadata: _payload(seeded["record"]))

    assert store.claim_refs("run_secret_fixture", [secret_ref], managed_hpc_identity=IDENTITY) == [secret_ref]
    claimed_root = seeded["runtime_root"] / "run-secrets" / "claimed" / "run_secret_fixture"
    assert not (seeded["runtime_root"] / "run-secrets" / "pending" / f"{seeded['digest']}.envelope").exists()
    assert (claimed_root / f"{seeded['digest']}.envelope").is_file()
    assert store.available_purposes_for_run("run_secret_fixture") == {"weather4cast_sftp": 1}

    payload = store.consume_for_run("run_secret_fixture", purpose="weather4cast_sftp", managed_hpc_identity=IDENTITY)
    assert payload["username"] == "fixture-user" and payload["password"] == "fixture-secret"
    assert not list(claimed_root.glob("*"))
    tombstone_path = seeded["runtime_root"] / "run-secrets" / "consumed" / "run_secret_fixture" / f"{seeded['digest']}.json"
    tombstone_text = tombstone_path.read_text(encoding="utf-8")
    assert "fixture-user" not in tombstone_text and "fixture-secret" not in tombstone_text
    assert json.loads(tombstone_text)["status"] == "consumed"
    with pytest.raises(FileNotFoundError):
        store.consume_for_run("run_secret_fixture", purpose="weather4cast_sftp", managed_hpc_identity=IDENTITY)
    assert store.available_purposes_for_run("run_secret_fixture") == {}


def test_weather4cast_consumes_one_live_reference_while_ignoring_expired_claims(tmp_path: Path) -> None:
    expired_ref = "rs_" + "H" * 43
    live_ref = "rs_" + "I" * 43
    expired = _seed_pending(
        tmp_path,
        expired_ref,
        expires=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    live = _seed_pending(tmp_path, live_ref)
    store = RunSecretStore(
        live["runtime_root"],
        consumer=lambda _envelope, _metadata: _payload(live["record"]),
    )
    claimed_root = live["runtime_root"] / "run-secrets" / "claimed" / "run_weather_live"
    claimed_root.mkdir(parents=True)
    for seeded in (expired, live):
        pending = seeded["runtime_root"] / "run-secrets" / "pending"
        record = json.loads((pending / f"{seeded['digest']}.json").read_text(encoding="utf-8"))
        record.update(
            {
                "schema": "evomind.run_secret_claimed.v1",
                "state": "claimed",
                "run_id": "run_weather_live",
                "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        (claimed_root / f"{seeded['digest']}.json").write_text(json.dumps(record), encoding="utf-8")
        (pending / f"{seeded['digest']}.envelope").replace(claimed_root / f"{seeded['digest']}.envelope")
        (pending / f"{seeded['digest']}.json").unlink()

    assert store.available_purposes_for_run("run_weather_live") == {"weather4cast_sftp": 1}
    payload = store.consume_for_run(
        "run_weather_live",
        purpose="weather4cast_sftp",
        managed_hpc_identity=IDENTITY,
    )
    assert payload["purpose"] == "weather4cast_sftp"
    assert payload["username"] == "fixture-user"


def test_weather4cast_selects_newest_of_multiple_live_references(tmp_path: Path) -> None:
    first_ref = "rs_" + "J" * 43
    newest_ref = "rs_" + "K" * 43
    first = _seed_pending(tmp_path / "first", first_ref)
    newest = _seed_pending(tmp_path / "newest", newest_ref)
    runtime_root = newest["runtime_root"]
    claimed = runtime_root / "run-secrets" / "claimed" / "run_weather_multiple"
    claimed.mkdir(parents=True, exist_ok=True)
    for seeded, offset in ((first, 2), (newest, 1)):
        pending = seeded["runtime_root"] / "run-secrets" / "pending"
        record = json.loads((pending / f"{seeded['digest']}.json").read_text(encoding="utf-8"))
        record.update({
            "schema": "evomind.run_secret_claimed.v1",
            "state": "claimed",
            "run_id": "run_weather_multiple",
            "claimed_at_utc": (datetime.now(timezone.utc) - timedelta(minutes=offset)).isoformat(),
        })
        (claimed / f"{seeded['digest']}.json").write_text(json.dumps(record), encoding="utf-8")
        (pending / f"{seeded['digest']}.envelope").replace(claimed / f"{seeded['digest']}.envelope")
        (pending / f"{seeded['digest']}.json").unlink()
    store = RunSecretStore(runtime_root, consumer=lambda _envelope, _metadata: _payload(newest["record"]))

    payload = store.consume_for_run(
        "run_weather_multiple",
        purpose="weather4cast_sftp",
        managed_hpc_identity=IDENTITY,
    )
    assert payload["purpose"] == "weather4cast_sftp"
    assert len(list(claimed.glob("*.json"))) == 1


def test_run_secret_claim_rejects_expiry_and_identity_drift_without_moving_envelope(tmp_path: Path) -> None:
    secret_ref = "rs_" + "B" * 43
    seeded = _seed_pending(tmp_path, secret_ref, expires=datetime.now(timezone.utc) - timedelta(seconds=1))
    store = RunSecretStore(seeded["runtime_root"], consumer=lambda *_args: b"{}")
    with pytest.raises(ValueError, match="mismatched or expired"):
        store.claim_refs("run_expired", [secret_ref], managed_hpc_identity=IDENTITY)
    assert (seeded["runtime_root"] / "run-secrets" / "pending" / f"{seeded['digest']}.envelope").is_file()

    other_ref = "rs_" + "C" * 43
    other = _seed_pending(tmp_path / "other", other_ref)
    other_store = RunSecretStore(other["runtime_root"], consumer=lambda *_args: b"{}")
    with pytest.raises(ValueError, match="mismatched or expired"):
        other_store.claim_refs("run_drift", [other_ref], managed_hpc_identity={**IDENTITY, "job_id": 99999})


def test_run_secret_payload_binding_failure_consumes_and_tombstones_failed_reference(tmp_path: Path) -> None:
    secret_ref = "rs_" + "D" * 43
    seeded = _seed_pending(tmp_path, secret_ref)
    store = RunSecretStore(
        seeded["runtime_root"],
        consumer=lambda _envelope, _metadata: _payload(seeded["record"], binding_sha256="0" * 64),
    )
    store.claim_refs("run_bad_binding", [secret_ref], managed_hpc_identity=IDENTITY)
    with pytest.raises(ValueError, match="binding mismatch"):
        store.consume_for_run("run_bad_binding", purpose="weather4cast_sftp", managed_hpc_identity=IDENTITY)
    tombstone = json.loads((seeded["runtime_root"] / "run-secrets" / "consumed" / "run_bad_binding" / f"{seeded['digest']}.json").read_text(encoding="utf-8"))
    assert tombstone["status"] == "failed"
    assert not (seeded["runtime_root"] / "run-secrets" / "claimed" / "run_bad_binding" / f"{seeded['digest']}.envelope").exists()


def test_kaggle_references_are_consumed_one_per_competition(tmp_path: Path) -> None:
    refs = ["rs_" + letter * 43 for letter in ("E", "F", "G")]
    seeded = [_seed_pending(tmp_path, ref, purpose="kaggle_api") for ref in refs]
    store = RunSecretStore(seeded[0]["runtime_root"], consumer=lambda _envelope, metadata: _payload(next(item["record"] for item in seeded if item["record"]["secret_ref_sha256"] == json.loads(metadata.read_text())["secret_ref_sha256"])) )
    assert store.claim_refs("run_kaggle", refs, managed_hpc_identity=IDENTITY) == refs
    for _ in refs:
        payload = store.consume_for_run("run_kaggle", purpose="kaggle_api", managed_hpc_identity=IDENTITY)
        assert payload["purpose"] == "kaggle_api"
    assert not list((seeded[0]["runtime_root"] / "run-secrets" / "claimed" / "run_kaggle").glob("*"))


def test_secret_consumer_resolves_pwsh_without_windows_powershell_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pwsh = tmp_path / "pwsh.exe"
    powershell = tmp_path / "powershell.exe"
    pwsh.write_bytes(b"pwsh")
    powershell.write_bytes(b"powershell")

    monkeypatch.setattr(
        run_secrets_module.shutil,
        "which",
        lambda name: str(pwsh) if name in {"pwsh", "pwsh.exe"} else str(powershell),
    )

    assert run_secrets_module._resolve_powershell7() == str(pwsh)


def test_secret_consumer_fails_closed_when_only_windows_powershell_is_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    powershell = tmp_path / "powershell.exe"
    powershell.write_bytes(b"powershell")
    monkeypatch.setattr(run_secrets_module.shutil, "which", lambda name: str(powershell) if name == "powershell.exe" else None)
    for name in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.setenv(name, str(tmp_path / name))

    with pytest.raises(RuntimeError, match="requires PowerShell 7"):
        run_secrets_module._resolve_powershell7()


def test_secret_consumer_preserves_sanitized_process_failure_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunSecretStore(tmp_path / "runtime", consumer=lambda *_args: b"{}")
    monkeypatch.setattr(run_secrets_module, "_resolve_powershell7", lambda: "pwsh.exe")

    monkeypatch.setattr(
        run_secrets_module.subprocess,
        "run",
        lambda *_args, **_kwargs: run_secrets_module.subprocess.CompletedProcess(
            [], 86, stdout=b"", stderr=b"RUN_SECRET_CONSUMER_ERROR:ValueError"
        ),
    )
    with pytest.raises(RuntimeError, match="host_exit|process_exit"):
        store._consume_with_service_certificate(tmp_path / "envelope", tmp_path / "metadata")

    monkeypatch.setattr(
        run_secrets_module.subprocess,
        "run",
        lambda *_args, **_kwargs: run_secrets_module.subprocess.CompletedProcess(
            [], 0, stdout=b"not-base64!", stderr=b""
        ),
    )
    with pytest.raises(RuntimeError, match="invalid_base64"):
        store._consume_with_service_certificate(tmp_path / "envelope", tmp_path / "metadata")


def test_consumer_error_marker_is_whitelisted_without_exposing_stderr() -> None:
    assert run_secrets_module._consumer_error_type(
        b"RUN_SECRET_CONSUMER_ERROR:CryptographicException\nprivate details"
    ) == "CryptographicException"
    assert run_secrets_module._consumer_error_type(
        b"RUN_SECRET_CONSUMER_ERROR:password=leaked"
    ) == "Unknown"
    assert run_secrets_module._consumer_error_type(b"unrelated stderr") == "Unknown"


def test_consumer_host_error_marker_is_whitelisted() -> None:
    assert run_secrets_module._consumer_host_error_type(
        b"RUN_SECRET_CONSUMER_ERROR:ParameterBindingException"
    ) == "ParameterBindingException"
    assert run_secrets_module._consumer_host_error_type(
        b"RUN_SECRET_CONSUMER_ERROR:password=leaked"
    ) == "Unknown"


def test_consumer_stage_marker_is_whitelisted() -> None:
    assert run_secrets_module._consumer_error_type(
        b"RUN_SECRET_CONSUMER_ERROR:Stage_oaep"
    ) == "Stage_oaep"


def test_consumer_preserves_exact_utc_expiry_text_for_expiry_and_aad() -> None:
    source = run_secrets_module._CONSUMER_PS1
    assert "$metadataRaw=Get-Content" in source
    assert "$expiresMatch=[regex]::Match($metadataRaw" in source
    assert "$expiresText=$expiresMatch.Groups['value'].Value" in source
    assert "$expiresAt=[DateTimeOffset]::Parse($expiresText" in source
    assert ",$metadata.key_id,$expiresText" in source
    assert "$metadata.expires_at_utc" not in source
