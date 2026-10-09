"""Focused tests for the local-only G21 organizer response validator.

Every positive package is an ephemeral, inert, ``test_only`` synthetic
contract fixture under pytest's temporary directory.  The production-facing
CLI must reject it, and even the explicit Python test-mode receipt must keep
training authorization false.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any

import pytest

from scripts import validate_g21_organizer_response_bundle as validator


RUN_ID = validator.FIXED_RUN_ID
ALLOCATION = validator.FIXED_ALLOCATION


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metric_contract(competition: str) -> tuple[str, str]:
    if competition == "e2lmc":
        return "SQ/RC/CS composite", "higher_is_better"
    if competition == "open_polymer":
        return "wMAE", "lower_is_better"
    return "official_ariel_private_score", "lower_is_better"


def _identity(
    schema: str,
    *,
    competition: str,
    bundle_id: str,
    request_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "run_id": RUN_ID,
        "allocation": ALLOCATION,
        "competition": competition,
        "bundle_id": bundle_id,
        "request_fingerprint": request_fingerprint,
        "test_only": True,
    }


def _manifest_entry(root: Path, path: str, role: str) -> dict[str, Any]:
    target = root / path
    return {
        "path": path,
        "role": role,
        "artifact_id": f"artifact_{role}",
        "bytes": target.stat().st_size,
        "sha256": _sha(target),
        "regular_file": True,
        "symlink": False,
    }


ROLE_PATHS = {
    "request_binding": "request-binding.json",
    "organizer_response": "organizer-response.json",
    "organizer_response_source": "organizer-response-source.json",
    "organizer_signature_or_publication_receipt": "organizer-signature-receipt.json",
    "protocol": "protocol.json",
    "official_scorer": "official-scorer.json",
    "private_evaluator_manifest": "private-evaluator-manifest.json",
    "holdout_ledger": "holdout-ledger.json",
    "holdout_claim": "holdout-claim.json",
    "participant_metrics": "participant-metrics.json",
    "leakage_audit": "leakage-audit.json",
}


def _reseal_manifest(root: Path) -> None:
    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        _manifest_entry(root, ROLE_PATHS[entry["role"]], entry["role"])
        for entry in manifest["files"]
    ]
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(entry["bytes"] for entry in manifest["files"])
    _write_json(manifest_path, manifest)


def _build_bundle(tmp_path: Path, competition: str) -> Path:
    """Build an inert synthetic package that can pass only explicit test mode."""

    root = tmp_path / f"{competition}-organizer-response-test-fixture"
    root.mkdir()
    bundle_id = f"test_only_{competition}_bundle_v1"
    request_fingerprint = validator.compute_request_fingerprint(competition)
    template, request_contract, template_sha, _ = validator._load_request_template(competition)
    del template
    metric, direction = _metric_contract(competition)
    common = {
        "competition": competition,
        "bundle_id": bundle_id,
        "request_fingerprint": request_fingerprint,
    }

    request_id = f"test_only_request_{competition}_v1"
    request = {
        **_identity(validator.REQUEST_SCHEMA, **common),
        "request_id": request_id,
        "request_kind": request_contract["request_kind"],
        "request_template_sha256": template_sha,
        "requested_contract_is_proposed_not_observed": True,
        "client_idempotency_key": f"test_only_idempotency_{competition}_v1",
    }
    _write_json(root / ROLE_PATHS["request_binding"], request)

    scorer = {
        **_identity(validator.SCORER_SCHEMA, **common),
        "scorer_id": f"test_only_{competition}_official_scorer_v1",
        "official_metric": metric,
        "direction": direction,
        "official_organizer_issued": True,
        "implementation_sha256": _fake_sha(f"{competition}-scorer-implementation"),
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["official_scorer"], scorer)

    protocol_id = f"test_only_{competition}_protocol_v1"
    protocol = {
        **_identity(validator.PROTOCOL_SCHEMA, **common),
        "protocol_id": protocol_id,
        "protocol_text": f"Synthetic test-only frozen organizer protocol for {competition}; no endpoint was contacted.",
        "official_metric": metric,
        "direction": direction,
        "metric_source": "organizer_signed_official_scorer",
        "protocol_comparable": True,
        "human_candidate_inputs_equal": True,
        "confidence_level": 0.95,
        "bootstrap_method": validator.BOOTSTRAP_METHOD,
        "official_scorer_sha256": _sha(root / ROLE_PATHS["official_scorer"]),
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["protocol"], protocol)

    evaluation_id = f"test_only_{competition}_evaluation_v1"
    expected_method, expected_path = validator._expected_receipt_endpoint(request_contract, evaluation_id)
    assert expected_method == "GET"
    endpoint_url = f"https://organizer.example.invalid{expected_path}"
    source = {
        **_identity("evomind.synthetic-organizer-response-source.test-only.v1", **common),
        "response_kind": "completed_organizer_evaluation",
        "evaluation_id": evaluation_id,
        "source_notice": "INERT SYNTHETIC TEST FIXTURE; NOT ORGANIZER EVIDENCE",
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["organizer_response_source"], source)

    records = [
        {
            "participant_id": "p_" + hashlib.sha256(f"test-only-{competition}-{index}".encode()).hexdigest(),
            "score": 0.40 + (index % 10) * 0.01,
        }
        for index in range(30)
    ]
    participant_metrics = {
        **_identity(validator.PARTICIPANT_METRICS_SCHEMA, **common),
        "metric": metric,
        "direction": direction,
        "participant_count": len(records),
        "records": records,
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["participant_metrics"], participant_metrics)

    evaluator = {
        **_identity(validator.EVALUATOR_SCHEMA, **common),
        "evaluator_id": f"test_only_{competition}_private_evaluator_v1",
        "contract_status": "OBSERVED_FROZEN",
        "endpoint_observed": True,
        "outputs_scores_and_hashes_only": True,
        "human_candidate_input_manifest_equal": True,
        "protocol_sha256": _sha(root / ROLE_PATHS["protocol"]),
        "official_scorer_sha256": _sha(root / ROLE_PATHS["official_scorer"]),
        "source_sha256": _sha(root / ROLE_PATHS["organizer_response_source"]),
        "participant_metrics_sha256": _sha(root / ROLE_PATHS["participant_metrics"]),
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["private_evaluator_manifest"], evaluator)

    ledger_id = f"test_only_{competition}_holdout_ledger_v1"
    ledger = {
        **_identity(validator.LEDGER_SCHEMA, **common),
        "ledger_id": ledger_id,
        "private_evaluator_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["holdout_ledger"], ledger)

    claim_id = f"test_only_{competition}_holdout_claim_v1"
    claim = {
        **_identity(validator.CLAIM_SCHEMA, **common),
        "claim_id": claim_id,
        "claim_transaction_id": f"test_only_{competition}_claim_transaction_v1",
        "ledger_id": ledger_id,
        "ledger_sha256": _sha(root / ROLE_PATHS["holdout_ledger"]),
        "private_evaluator_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
        "atomic_claim": True,
        "unconsumed_at_claim": True,
        "previously_consumed": False,
        "consumed_once": True,
        "overlap_count": 0,
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["holdout_claim"], claim)

    leakage = {
        **_identity(validator.LEAKAGE_SCHEMA, **common),
        "source_sha256": _sha(root / ROLE_PATHS["organizer_response_source"]),
        "protocol_sha256": _sha(root / ROLE_PATHS["protocol"]),
        "official_scorer_sha256": _sha(root / ROLE_PATHS["official_scorer"]),
        "private_evaluator_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
        "holdout_claim_sha256": _sha(root / ROLE_PATHS["holdout_claim"]),
        "participant_metrics_sha256": _sha(root / ROLE_PATHS["participant_metrics"]),
        "hidden_labels_present_in_bundle": False,
        "candidate_saw_human_responses": False,
        "human_saw_candidate_outputs": False,
        "prior_overlap_count": 0,
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["leakage_audit"], leakage)

    role_hashes = {
        role: _sha(root / path)
        for role, path in ROLE_PATHS.items()
        if (root / path).exists()
    }
    signed_payload = validator._signature_binding_payload(
        competition=competition,
        bundle_id=bundle_id,
        request_fingerprint=request_fingerprint,
        role_hashes=role_hashes,
    )
    proof = {
        **_identity(validator.PROOF_SCHEMA, **common),
        "proof_id": f"test_only_{competition}_organizer_proof_v1",
        "proof_kind": "organizer_signature",
        "verification_status": "VERIFIED",
        "signer_authority": "official_organizer",
        "source_url_or_doi": endpoint_url,
        "signed_binding_sha256": hashlib.sha256(validator._canonical_json_bytes(signed_payload)).hexdigest(),
        "organizer_signature": "TEST_ONLY_INERT_SIGNATURE_NOT_REAL",
        "public_key_fingerprint_sha256": _fake_sha(f"{competition}-test-only-public-key"),
        "test_labels_exposed": False,
    }
    _write_json(root / ROLE_PATHS["organizer_signature_or_publication_receipt"], proof)

    scores = [float(record["score"]) for record in records]
    human_mean = fmean(scores)
    ci_lower, ci_upper = validator.deterministic_bootstrap_ci(scores)
    response: dict[str, Any] = {
        **_identity(validator.RESPONSE_SCHEMA, **common),
        "request_id": request_id,
        "request_kind": request_contract["request_kind"],
        "requested_contract_is_proposed_not_observed": False,
        "endpoint_observed": True,
        "observed_endpoint": {
            "method": expected_method,
            "url": endpoint_url,
            "status_code": 200,
            "observed_at_utc": "2026-08-31T00:00:00Z",
            "response_source_sha256": _sha(root / ROLE_PATHS["organizer_response_source"]),
        },
        "evaluation_id": evaluation_id,
        "status": "COMPLETED",
        "protocol_id": protocol_id,
        "protocol_sha256": _sha(root / ROLE_PATHS["protocol"]),
        "metric": metric,
        "direction": direction,
        "official_scorer_sha256": _sha(root / ROLE_PATHS["official_scorer"]),
        "human_included_count": len(records),
        "confidence_level": 0.95,
        "source_url_or_doi": endpoint_url,
        "source_title": f"Synthetic test-only {competition} organizer receipt",
        "source_authority": "official_organizer",
        "source_filename": ROLE_PATHS["organizer_response_source"],
        "source_content_type": "application/json",
        "source_bytes": (root / ROLE_PATHS["organizer_response_source"]).stat().st_size,
        "source_sha256": _sha(root / ROLE_PATHS["organizer_response_source"]),
        "artifact_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
        "organizer_signature": _sha(root / ROLE_PATHS["organizer_signature_or_publication_receipt"]),
        "test_labels_exposed": False,
    }
    if competition == "e2lmc":
        response.update(
            {
                "private_experiment_claim_id": claim_id,
                "private_experiment_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
                "private_experiment_previously_consumed": False,
                "participant_level_sq_rc_cs_artifact_sha256": _sha(root / ROLE_PATHS["participant_metrics"]),
                "human_composite_mean": human_mean,
                "human_composite_ci95_lower": ci_lower,
                "human_composite_ci95_upper": ci_upper,
                "candidate_sq": 0.5,
                "candidate_rc": 0.6,
                "candidate_cs": 0.7,
                "candidate_composite": 0.6,
                "experiment_overlap_count": 0,
            }
        )
    elif competition == "ariel_2025":
        response.update(
            {
                "private_planet_claim_id": claim_id,
                "private_planet_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
                "private_planet_previously_consumed": False,
                "participant_level_official_metric_artifact_sha256": _sha(root / ROLE_PATHS["participant_metrics"]),
                "human_metric_mean": human_mean,
                "human_metric_ci95_lower": ci_lower,
                "human_metric_ci95_upper": ci_upper,
                "candidate_official_metric": 0.35,
                "input_manifest_equal_for_human_and_candidate": True,
                "planet_overlap_count": 0,
            }
        )
    else:
        response.update(
            {
                "private_molecule_claim_id": claim_id,
                "private_molecule_target_manifest_sha256": _sha(root / ROLE_PATHS["private_evaluator_manifest"]),
                "private_molecule_manifest_previously_consumed": False,
                "targets": validator.OPEN_POLYMER_TARGETS,
                "target_weights_sha256": _fake_sha("test-only-polymer-target-weights"),
                "valid_pair_manifest_sha256": _fake_sha("test-only-polymer-valid-pairs"),
                "missing_prediction_policy_sha256": _fake_sha("test-only-polymer-missing-policy"),
                "participant_level_wmae_artifact_sha256": _sha(root / ROLE_PATHS["participant_metrics"]),
                "human_wmae_mean": human_mean,
                "human_wmae_ci95_lower": ci_lower,
                "human_wmae_ci95_upper": ci_upper,
                "candidate_wmae": 0.35,
                "valid_pairs_equal_for_human_and_candidate": True,
                "polymer_overlap_count": 0,
            }
        )
    _write_json(root / ROLE_PATHS["organizer_response"], response)

    files = [_manifest_entry(root, path, role) for role, path in ROLE_PATHS.items()]
    manifest = {
        "schema": validator.MANIFEST_SCHEMA,
        "manifest_self_excluded": True,
        "synthetic_fixture": True,
        "test_only": True,
        "inert_no_external_actions": True,
        "run_id": RUN_ID,
        "allocation": ALLOCATION,
        "competition": competition,
        "bundle_id": bundle_id,
        "request_fingerprint": request_fingerprint,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }
    _write_json(root / validator.MANIFEST_NAME, manifest)
    return root


def _validate(root: Path, competition: str) -> dict[str, Any]:
    metric, direction = _metric_contract(competition)
    return validator.validate_organizer_response_bundle(
        root,
        competition=competition,
        expected_metric=metric,
        expected_direction=direction,
        allow_test_fixture=True,
    )


@pytest.mark.parametrize("competition", ["e2lmc", "ariel_2025", "open_polymer"])
def test_inert_test_only_bundle_closes_all_bindings_but_never_authorizes_training(
    tmp_path: Path,
    competition: str,
) -> None:
    root = _build_bundle(tmp_path, competition)
    receipt = _validate(root, competition)

    assert receipt["status"] == "VALIDATED_TEST_FIXTURE"
    assert receipt["synthetic_fixture"] is True
    assert receipt["real_organizer_evidence_accepted"] is False
    assert receipt["training_authorized_by_receipt"] is False
    assert receipt["training_authorized_by_this_receipt"] is False
    assert receipt["ready_for_gpu"] is False
    assert receipt["test_labels_exposed"] is False
    assert receipt["file_count"] == len(validator.REQUIRED_ROLES)
    assert receipt["side_effects"] == {
        "network_access": False,
        "production_writes": 0,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
        "signals_sent": 0,
        "training_tool_calls": 0,
    }


def test_default_api_and_cli_reject_synthetic_fixture_without_writing_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _build_bundle(tmp_path, "e2lmc")
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        validator.validate_organizer_response_bundle(root, competition="e2lmc")
    assert caught.value.code == "SYNTHETIC_FIXTURE_FORBIDDEN"

    receipt_path = tmp_path / "must-not-exist.json"
    assert validator.main(
        [
            "--bundle-dir",
            str(root),
            "--competition",
            "e2lmc",
            "--receipt",
            str(receipt_path),
        ]
    ) == 2
    assert not receipt_path.exists()
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "SYNTHETIC_FIXTURE_FORBIDDEN"


def test_explicit_test_receipt_is_outside_bundle_and_still_non_authorizing(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "open_polymer")
    receipt = _validate(root, "open_polymer")
    receipt_path = tmp_path / "test-only-validation-receipt.json"
    validator.write_validation_receipt(receipt, receipt_path, bundle_dir=root)
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["training_authorized_by_receipt"] is False
    assert written["ready_for_gpu"] is False

    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        validator.write_validation_receipt(receipt, root / "inside.json", bundle_dir=root)
    assert caught.value.code == "RECEIPT_INSIDE_BUNDLE_FORBIDDEN"


def test_missing_observed_endpoint_is_rejected(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "e2lmc")
    response_path = root / ROLE_PATHS["organizer_response"]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["endpoint_observed"] = False
    _write_json(response_path, response)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "e2lmc")
    assert caught.value.code == "ENDPOINT_NOT_OBSERVED"


def test_wrong_source_sha_is_rejected_before_receipt(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "ariel_2025")
    with (root / ROLE_PATHS["organizer_response_source"]).open("ab") as handle:
        handle.write(b"tamper\n")
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "ariel_2025")
    assert caught.value.code in {"FILE_BYTES_MISMATCH", "FILE_SHA256_MISMATCH"}


def test_unsafe_manifest_path_is_rejected(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "e2lmc")
    manifest_path = root / validator.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../outside.json"
    _write_json(manifest_path, manifest)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "e2lmc")
    assert caught.value.code == "MANIFEST_PATH_UNSAFE"


def test_symlinked_bundle_file_is_rejected_when_platform_allows_symlinks(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "open_polymer")
    source_path = root / ROLE_PATHS["organizer_response_source"]
    target = tmp_path / "outside-source.json"
    target.write_bytes(source_path.read_bytes())
    source_path.unlink()
    try:
        os.symlink(target, source_path)
    except OSError:
        pytest.skip("symlink creation unavailable on this Windows host")
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "open_polymer")
    assert caught.value.code == "SYMLINK_FORBIDDEN"


def test_metric_drift_between_signed_protocol_and_response_is_rejected(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "open_polymer")
    response_path = root / ROLE_PATHS["organizer_response"]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["metric"] = "normalized RMSE"
    _write_json(response_path, response)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "open_polymer")
    assert caught.value.code == "METRIC_DRIFT_DETECTED"


def test_any_test_label_exposure_is_rejected(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "ariel_2025")
    response_path = root / ROLE_PATHS["organizer_response"]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["test_labels_exposed"] = True
    _write_json(response_path, response)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "ariel_2025")
    assert caught.value.code == "TEST_LABEL_EXPOSURE_INVALID"


def test_non_atomic_claim_is_rejected(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "e2lmc")
    claim_path = root / ROLE_PATHS["holdout_claim"]
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["atomic_claim"] = False
    _write_json(claim_path, claim)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "e2lmc")
    assert caught.value.code == "HOLDOUT_ATOMIC_CLAIM_NOT_PROVEN"


def test_unsigned_response_is_rejected(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "open_polymer")
    proof_path = root / ROLE_PATHS["organizer_signature_or_publication_receipt"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["verification_status"] = "UNVERIFIED"
    proof["organizer_signature"] = ""
    _write_json(proof_path, proof)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "open_polymer")
    assert caught.value.code == "ORGANIZER_RESPONSE_UNSIGNED"


def test_request_template_replay_cannot_be_treated_as_observed_response(tmp_path: Path) -> None:
    root = _build_bundle(tmp_path, "e2lmc")
    response_path = root / ROLE_PATHS["organizer_response"]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["requested_contract_is_proposed_not_observed"] = True
    _write_json(response_path, response)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "e2lmc")
    assert caught.value.code == "REQUEST_TEMPLATE_REPLAY_FORBIDDEN"


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("run_id", "run_wrong", "RUN_BINDING_INVALID"),
        ("request_fingerprint", "0" * 64, "REQUEST_FINGERPRINT_MISMATCH"),
    ],
)
def test_fixed_run_and_request_fingerprint_binding_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
    expected_code: str,
) -> None:
    root = _build_bundle(tmp_path, "ariel_2025")
    response_path = root / ROLE_PATHS["organizer_response"]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response[field] = value
    _write_json(response_path, response)
    _reseal_manifest(root)
    with pytest.raises(validator.OrganizerBundleValidationError) as caught:
        _validate(root, "ariel_2025")
    assert caught.value.code == expected_code
