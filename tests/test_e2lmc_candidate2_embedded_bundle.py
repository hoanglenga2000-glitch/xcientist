from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e2lmc_candidate2_embedded_bundle.py"
SPEC = importlib.util.spec_from_file_location("e2lmc_candidate2_embedded_bundle", SCRIPT)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def test_embedded_payloads_have_frozen_hashes_and_expected_sizes() -> None:
    source = bundle.embedded_source_bytes()
    ledger = bundle.embedded_ledger_bytes()
    logic = bundle.embedded_logic_bytes()

    assert len(source) == 135_783
    assert hashlib.sha256(source).hexdigest() == bundle.EMBEDDED_SOURCE_SHA256
    assert hashlib.sha256(ledger).hexdigest() == bundle.EMBEDDED_LEDGER_SHA256
    assert hashlib.sha256(logic).hexdigest() == bundle.EMBEDDED_LOGIC_SHA256
    assert hashlib.sha256(logic).hexdigest() == hashlib.sha256(
        (SCRIPT.parent / "e2lmc_candidate2_goal_v2.py").read_bytes()
    ).hexdigest()


def test_ledger_is_immutable_and_bound_to_embedded_source() -> None:
    payload = json.loads(bundle.embedded_ledger_bytes())
    assert payload["schema"] == bundle.EMBEDDED_LEDGER_SCHEMA
    assert payload["official_source_sha256"] == bundle.EMBEDDED_SOURCE_SHA256
    assert payload["read_only"] is True
    assert payload["immutable"] is True
    assert payload["provenance"] == bundle.LEDGER_PROVENANCE
    assert payload["hidden_test_labels_used"] is False
    assert payload["test_labels_used"] is False


def test_tampered_embedded_payload_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = bundle._EMBEDDED_SOURCE_GZIP_B64
    # Keep valid base64/gzip framing while changing the decompressed digest.
    replacement = original[:-1] + ("A" if original[-1] != "A" else "B")
    monkeypatch.setattr(bundle, "_EMBEDDED_SOURCE_GZIP_B64", replacement)
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle.embedded_source_bytes()
    assert caught.value.code in {"EMBEDDED_PAYLOAD_SHA_MISMATCH", "EMBEDDED_PAYLOAD_INVALID"}


def test_override_inputs_are_rejected_without_reading_them(tmp_path: Path) -> None:
    output = tmp_path / "out"
    source = tmp_path / "caller-source.json"
    source.write_text("not used", encoding="utf-8")
    assert bundle.main(
        [
            "--data-dir",
            str(tmp_path / "missing-data-root"),
            "--source-json",
            str(source),
            "--out-dir",
            str(output),
        ]
    ) == 2
    gate = json.loads((output / "exact-gate.json").read_text(encoding="utf-8"))
    assert gate["code"] == "EMBEDDED_INPUT_OVERRIDE"


def test_data_dir_is_compatibility_only_and_missing_root_does_not_gate_input(tmp_path: Path) -> None:
    output = tmp_path / "out"
    code = bundle.main(
        ["--data-dir", str(tmp_path / "does-not-exist"), "--out-dir", str(output)]
    )
    assert code == 0
    comparison = json.loads((output / "candidate-vs-baseline.json").read_text(encoding="utf-8"))
    assert comparison["gate_passed"] is False
    receipt = json.loads((output / "embedded-bundle-receipt.json").read_text(encoding="utf-8"))
    assert receipt["embedded_source_sha256"] == bundle.EMBEDDED_SOURCE_SHA256
    assert receipt["embedded_ledger_sha256"] == bundle.EMBEDDED_LEDGER_SHA256
    assert receipt["network_access"] is False
    assert receipt["remote_writes"] == 0


def test_bundle_success_augments_existing_manifest_and_preserves_gate_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"

    def fake_logic(_path: Path) -> object:
        def fake_main(argv: list[str]) -> int:
            destination = Path(argv[argv.index("--out-dir") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "official-source-receipt.json").write_text(
                json.dumps({"url": bundle.SOURCE_URL, "bytes": 135_783, "sha256": bundle.EMBEDDED_SOURCE_SHA256}),
                encoding="utf-8",
            )
            (destination / "environment-lock.json").write_text(
                json.dumps({"network_access": False}), encoding="utf-8"
            )
            (destination / "metrics.json").write_text(
                json.dumps({"gate_passed": False}), encoding="utf-8"
            )
            (destination / "artifact-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "evomind.e2lmc.candidate2.artifact_manifest.v2",
                        "source_sha256": "implementation-sha",
                        "official_source_sha256": bundle.EMBEDDED_SOURCE_SHA256,
                        "holdout_ledger_sha256": bundle.EMBEDDED_LEDGER_SHA256,
                        "gate_passed": False,
                        "files": [],
                    }
                ),
                encoding="utf-8",
            )
            return 0

        return SimpleNamespace(main=fake_main, __file__=str(SCRIPT))

    monkeypatch.setattr(bundle, "_load_offline_logic", fake_logic)
    assert bundle.main(["--out-dir", str(output)]) == 0

    receipt = json.loads((output / "embedded-bundle-receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["ledger_provenance"] == bundle.LEDGER_PROVENANCE
    assert receipt["network_access"] is False

    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "evomind.e2lmc.candidate2.artifact_manifest.v2"
    assert manifest["bundle_schema"] == bundle.BUNDLE_SCHEMA
    assert manifest["embedded_source_sha256"] == bundle.EMBEDDED_SOURCE_SHA256
    assert manifest["embedded_ledger_sha256"] == bundle.EMBEDDED_LEDGER_SHA256
    assert manifest["network_access"] is False
    assert {entry["name"] for entry in manifest["files"]} == {
        "embedded-bundle-receipt.json",
        "environment-lock.json",
        "metrics.json",
        "official-source-receipt.json",
    }
    manifest_receipt = json.loads((output / "artifact-manifest-receipt.json").read_text(encoding="utf-8"))
    assert manifest_receipt["sha256"] == hashlib.sha256(
        (output / "artifact-manifest.json").read_bytes()
    ).hexdigest()


def test_no_network_fallback_or_remote_command_in_wrapper() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("urllib", "urlopen", "requests", "subprocess", "socket", "os.system"):
        assert forbidden not in source


def test_fingerprint_is_non_sensitive_and_explicitly_offline() -> None:
    fingerprint = bundle.bundle_fingerprint()
    assert fingerprint["network_access"] is False
    assert fingerprint["ledger_provenance"] == bundle.LEDGER_PROVENANCE
    assert fingerprint["source_sha256"] == bundle.EMBEDDED_SOURCE_SHA256
    assert fingerprint["ledger_sha256"] == bundle.EMBEDDED_LEDGER_SHA256
    rendered = json.dumps(fingerprint, sort_keys=True)
    assert "password" not in rendered.casefold()
    assert "credential" not in rendered.casefold()
