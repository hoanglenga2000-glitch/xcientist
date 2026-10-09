from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e2lmc_candidate4_embedded_bundle.py"
SPEC = importlib.util.spec_from_file_location("e2lmc_candidate4_embedded_bundle", SCRIPT)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def _ledger() -> dict:
    return json.loads(bundle.embedded_ledger_bytes())


def _ledger_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _patch_ledger(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    monkeypatch.setattr(bundle, "embedded_ledger_bytes", lambda: _ledger_bytes(payload))


def _write_minimal_candidate4_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    schemas = {
        "task-contract-v2.json": "evomind.e2lmc.candidate4.task_contract.v2",
        "training-config.json": "evomind.e2lmc.candidate4.training_config.v2",
        "dataset-audit.json": "evomind.e2lmc.candidate4.dataset_audit.v2",
        "split-manifest.json": "evomind.e2lmc.candidate4.split_manifest.v2",
        "checkpoint-index.json": "evomind.e2lmc.candidate4.checkpoint_index.v1",
        "candidate-vs-human.json": "evomind.e2lmc.candidate4.candidate_vs_human.v1",
        "retrospective-memory.json": "evomind.retrospective_memory.v2",
    }
    (root / "official-source-receipt.json").write_text(
        json.dumps({"url": bundle.SOURCE_URL, "bytes": 135783, "sha256": bundle.EMBEDDED_SOURCE_SHA256}),
        encoding="utf-8",
    )
    for name, schema in schemas.items():
        (root / name).write_text(json.dumps({"schema": schema}), encoding="utf-8")
    (root / "environment-lock.json").write_text(json.dumps({"network_access": False}), encoding="utf-8")
    (root / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema": "evomind.e2lmc.candidate4.artifact_manifest.v2",
                "official_source_sha256": bundle.EMBEDDED_SOURCE_SHA256,
                "holdout_ledger_sha256": bundle.EMBEDDED_LEDGER_SHA256,
                "gate_passed": False,
                "files": [],
            }
        ),
        encoding="utf-8",
    )


def test_payload_hashes_and_candidate4_identity_are_frozen() -> None:
    source = bundle.embedded_source_bytes()
    ledger = bundle.embedded_ledger_bytes()
    logic = bundle.embedded_logic_bytes()
    assert len(source) == 135_783
    assert hashlib.sha256(source).hexdigest() == bundle.EMBEDDED_SOURCE_SHA256
    assert hashlib.sha256(ledger).hexdigest() == bundle.EMBEDDED_LEDGER_SHA256
    assert hashlib.sha256(logic).hexdigest() == bundle.EMBEDDED_LOGIC_SHA256
    assert all(len(value) == 64 and all(char in "0123456789abcdef" for char in value) for value in (
        bundle.EMBEDDED_SOURCE_SHA256,
        bundle.EMBEDDED_LEDGER_SHA256,
        bundle.EMBEDDED_LOGIC_SHA256,
    ))
    text = logic.decode("utf-8")
    assert "candidate-2" not in text and "candidate2" not in text
    assert "e2lmc-v2-fresh-iteration-candidate-4" in text
    candidate4_source = SCRIPT.parent / "e2lmc_candidate4_goal_v2.py"
    assert candidate4_source.is_file()
    assert hashlib.sha256(candidate4_source.read_bytes()).hexdigest() == bundle.EMBEDDED_LOGIC_SHA256


def test_ledger_contains_candidate1_and_candidate3_points() -> None:
    payload = _ledger()
    points = {(item["experiment"], float(item["iteration"])) for item in payload["consumed_iteration_points"]}
    expected_c3 = set(bundle.CANDIDATE3_HOLDOUT_POINTS)
    assert payload["consumed_point_count"] == 44
    assert payload["candidate1_verified_point_count"] == 32
    assert payload["candidate3_verified_point_count"] == 12
    assert expected_c3 <= points
    assert set(payload["available_experiments"]) == set(bundle.AVAILABLE_EXPERIMENTS)
    assert len(points) == 44


def test_ledger_metadata_is_bound_to_verified_parent_artifacts() -> None:
    payload = _ledger()
    metadata = bundle._ledger_metadata(bundle.EMBEDDED_SOURCE_SHA256)
    assert metadata["candidate_id"] == "e2lmc-v2-fresh-iteration-candidate-4"
    assert metadata["ledger_provenance"] == "derived_from_verified_candidate_artifacts"
    assert metadata["provenance_note"].endswith("no external signature")
    assert metadata["external_signature"] is False
    assert {item["candidate_id"] for item in metadata["parent_candidates"]} == {
        "e2lmc-v2-baseline-candidate-1",
        "e2lmc-v2-fresh-iteration-candidate-3",
    }
    assert payload["candidate3_result_sha256"] == bundle.CANDIDATE3_RESULT_SHA256
    assert payload["candidate3_holdout_point_rows_sha256"] == bundle.CANDIDATE3_HOLDOUT_POINT_ROWS_SHA256


def test_canonical_point_hash_is_reproducible() -> None:
    payload = _ledger()
    canonical = bundle._canonical_point_rows(payload["consumed_iteration_points"])
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert digest == payload["canonical_consumed_points_sha256"]
    assert digest == bundle.bundle_fingerprint()["ledger_lineage"]["canonical_consumed_points_sha256"]
    fingerprint = bundle.bundle_fingerprint()
    assert fingerprint["consumed_point_count"] == 44
    assert fingerprint["candidate3_result_sha256"] == bundle.CANDIDATE3_RESULT_SHA256


def test_ledger_rejects_missing_candidate3_point(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _ledger()
    payload["consumed_iteration_points"] = payload["consumed_iteration_points"][1:]
    payload["consumed_point_count"] = 43
    _patch_ledger(monkeypatch, payload)
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._ledger_metadata(bundle.EMBEDDED_SOURCE_SHA256)
    assert caught.value.code == "LEDGER_POINT_COUNT_MISMATCH"


def test_ledger_rejects_duplicate_or_altered_candidate3_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _ledger()
    payload["candidate3_verified_holdout_points"] = payload["candidate3_verified_holdout_points"][:-1]
    _patch_ledger(monkeypatch, payload)
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._ledger_metadata(bundle.EMBEDDED_SOURCE_SHA256)
    assert caught.value.code == "CANDIDATE3_POINTS_MISSING"


def test_ledger_rejects_available_allowlist_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _ledger()
    payload["available_experiments"] = ["dense-1b-arch1", "unexpected"]
    _patch_ledger(monkeypatch, payload)
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._ledger_metadata(bundle.EMBEDDED_SOURCE_SHA256)
    assert caught.value.code == "LEDGER_AVAILABLE_EXPERIMENTS_MISMATCH"


def test_ledger_rejects_source_or_mutability_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _ledger()
    payload["official_source_sha256"] = "0" * 64
    _patch_ledger(monkeypatch, payload)
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._ledger_metadata(bundle.EMBEDDED_SOURCE_SHA256)
    assert caught.value.code == "LEDGER_SOURCE_MISMATCH"

    payload = _ledger()
    payload["official_source_sha256"] = bundle.EMBEDDED_SOURCE_SHA256
    payload["read_only"] = False
    _patch_ledger(monkeypatch, payload)
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._ledger_metadata(bundle.EMBEDDED_SOURCE_SHA256)
    assert caught.value.code == "LEDGER_NOT_IMMUTABLE"


@pytest.mark.parametrize("payload_name", ["source", "ledger", "logic"])
def test_tampered_payload_fails_closed(monkeypatch: pytest.MonkeyPatch, payload_name: str) -> None:
    attr = {
        "source": "_EMBEDDED_SOURCE_GZIP_B64",
        "ledger": "_EMBEDDED_LEDGER_GZIP_B64",
        "logic": "_EMBEDDED_LOGIC_GZIP_B64",
    }[payload_name]
    raw = {
        "source": bundle.embedded_source_bytes,
        "ledger": bundle.embedded_ledger_bytes,
        "logic": bundle.embedded_logic_bytes,
    }[payload_name]() + b"tampered"
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    monkeypatch.setattr(bundle, attr, encoded)
    with pytest.raises(bundle.BundleGateError) as caught:
        {
            "source": bundle.embedded_source_bytes,
            "ledger": bundle.embedded_ledger_bytes,
            "logic": bundle.embedded_logic_bytes,
        }[payload_name]()
    assert caught.value.code == "EMBEDDED_PAYLOAD_SHA_MISMATCH"


def test_caller_source_and_ledger_overrides_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "out"
    source = tmp_path / "caller-source.json"
    source.write_text("must not be read", encoding="utf-8")
    assert bundle.main(["--source-json", str(source), "--out-dir", str(output)]) == 2
    gate = json.loads((output / "exact-gate.json").read_text(encoding="utf-8"))
    assert gate["code"] == "EMBEDDED_INPUT_OVERRIDE"
    assert gate["candidate_id"] == "e2lmc-v2-fresh-iteration-candidate-4"


def test_data_dir_is_compatibility_only_and_network_is_disabled(tmp_path: Path) -> None:
    output = tmp_path / "out"
    code = bundle.main(["--data-dir", str(tmp_path / "missing"), "--out-dir", str(output)])
    assert code == 0
    receipt = json.loads((output / "embedded-bundle-receipt.json").read_text(encoding="utf-8"))
    assert receipt["network_access"] is False
    assert receipt["remote_writes"] == 0
    assert receipt["consumed_point_count"] == 44
    assert receipt["candidate_id"] == "e2lmc-v2-fresh-iteration-candidate-4"


def test_success_annotation_requires_candidate4_manifest_and_writes_lineage(tmp_path: Path) -> None:
    output = tmp_path / "out"
    _write_minimal_candidate4_artifacts(output)
    fingerprint = bundle.bundle_fingerprint()
    bundle._annotate_success(output, fingerprint)
    receipt = json.loads((output / "embedded-bundle-receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert receipt["ledger_lineage"]["candidate_id"] == receipt["candidate_id"]
    assert manifest["schema"] == "evomind.e2lmc.candidate4.artifact_manifest.v2"
    assert manifest["candidate_id"] == receipt["candidate_id"]
    assert manifest["embedded_logic_sha256"] == bundle.EMBEDDED_LOGIC_SHA256
    assert json.loads((output / "artifact-manifest-receipt.json").read_text(encoding="utf-8"))["sha256"] == hashlib.sha256(
        (output / "artifact-manifest.json").read_bytes()
    ).hexdigest()


def test_success_annotation_fails_closed_before_completed_receipt_when_artifacts_missing(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._annotate_success(output, bundle.bundle_fingerprint())
    assert caught.value.code == "CANDIDATE4_ARTIFACTS_INCOMPLETE"
    assert not (output / "embedded-bundle-receipt.json").exists()


def test_old_manifest_schema_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "out"
    _write_minimal_candidate4_artifacts(output)
    (output / "artifact-manifest.json").write_text(
        json.dumps({"schema": "evomind.e2lmc.candidate2.artifact_manifest.v2"}), encoding="utf-8"
    )
    with pytest.raises(bundle.BundleGateError) as caught:
        bundle._refresh_artifact_manifest(output, bundle.bundle_fingerprint(), "embedded-bundle-receipt.json")
    assert caught.value.code == "ARTIFACT_MANIFEST_SCHEMA_MISMATCH"


def test_gate_annotation_preserves_exact_code_and_lineage(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "exact-gate.json").write_text(
        json.dumps({"status": "waiting_exact_gate", "code": "MODEL_GATE", "resume_point": "model"}),
        encoding="utf-8",
    )
    bundle._annotate_gate(output, bundle.bundle_fingerprint())
    gate = json.loads((output / "exact-gate.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "embedded-bundle-receipt.json").read_text(encoding="utf-8"))
    assert gate["code"] == "MODEL_GATE"
    assert gate["candidate_id"] == "e2lmc-v2-fresh-iteration-candidate-4"
    assert receipt["status"] == "waiting_exact_gate"
    assert receipt["gate"]["code"] == "MODEL_GATE"


def test_no_network_or_remote_execution_primitives() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("urllib", "urlopen", "requests", "subprocess", "socket", "os.system"):
        assert forbidden not in source


def test_fingerprint_is_deterministic_and_non_sensitive() -> None:
    first = bundle.bundle_fingerprint()
    second = bundle.bundle_fingerprint()
    assert first == second
    rendered = json.dumps(first, sort_keys=True)
    assert "password" not in rendered.casefold()
    assert "credential" not in rendered.casefold()
    assert first["network_access"] is False
    assert first["ledger_lineage"]["external_signature"] is False
