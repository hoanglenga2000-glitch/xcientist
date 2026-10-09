"""Validate a closed local organizer-evidence response bundle for G21.

This validator covers only the organizer-private evaluator gates for E2LMC,
Ariel Data Challenge 2025, and Open Polymer.  It deliberately has no network,
subprocess, production database, HPC, GPU, approval, or training capability.

The request template in ``configs/g21_external_evidence_request_templates.json``
is a proposal, never evidence.  A passing bundle must instead contain an
observed, completed response whose endpoint, source bytes, request fingerprint,
official protocol/scorer, private evaluator, atomic one-time holdout claim,
participant metrics, leakage audit, and signature/publication receipt all close
under one self-excluded artifact manifest.

Even a real passing bundle produces only a local structural-validation receipt.
That receipt always says ``training_authorized_by_receipt=false``.  A synthetic
fixture can exercise the positive path only through the Python API's explicit
``allow_test_fixture=True`` argument; the CLI never accepts synthetic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "configs" / "g21_external_evidence_request_templates.json"

FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
SUPPORTED_COMPETITIONS = frozenset({"e2lmc", "ariel_2025", "open_polymer"})

MANIFEST_NAME = "artifact-manifest.json"
MANIFEST_SCHEMA = "evomind.organizer-evidence-response-bundle-manifest.v1"
REQUEST_SCHEMA = "evomind.organizer-evidence-request-binding.v1"
RESPONSE_SCHEMA = "evomind.organizer-evidence-response.v1"
PROTOCOL_SCHEMA = "evomind.organizer-private-evaluator-protocol.v1"
SCORER_SCHEMA = "evomind.organizer-official-scorer-manifest.v1"
EVALUATOR_SCHEMA = "evomind.organizer-private-evaluator-manifest.v1"
LEDGER_SCHEMA = "evomind.private-evaluator-holdout-ledger.v1"
CLAIM_SCHEMA = "evomind.private-evaluator-holdout-claim.v1"
PARTICIPANT_METRICS_SCHEMA = "evomind.organizer-private-evaluator-participant-metrics.v1"
LEAKAGE_SCHEMA = "evomind.organizer-private-evaluator-leakage-audit.v1"
PROOF_SCHEMA = "evomind.organizer-signature-publication-receipt.v1"
RECEIPT_SCHEMA = "evomind.g21-organizer-response-validation-receipt.v1"

REQUIRED_ROLES = frozenset(
    {
        "request_binding",
        "organizer_response",
        "organizer_response_source",
        "organizer_signature_or_publication_receipt",
        "protocol",
        "official_scorer",
        "private_evaluator_manifest",
        "holdout_ledger",
        "holdout_claim",
        "participant_metrics",
        "leakage_audit",
    }
)
JSON_ROLES = REQUIRED_ROLES

KNOWN_METRIC_CONTRACTS: dict[str, tuple[str, str]] = {
    "e2lmc": ("SQ/RC/CS composite", "higher_is_better"),
    "open_polymer": ("wMAE", "lower_is_better"),
}
VALID_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})
SOURCE_AUTHORITIES = frozenset({"official_organizer", "peer_reviewed", "official_dataset"})
PROOF_KINDS = frozenset({"organizer_signature", "official_publication_receipt"})
OPEN_POLYMER_TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
PSEUDONYM_RE = re.compile(r"^(?:p|participant)_[0-9a-f]{16,64}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
MAX_FILES = 64
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
MIN_PARTICIPANTS = 30
MAX_PARTICIPANTS = 100_000
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_METHOD = "participant_cluster_percentile_bootstrap_v1"

PII_KEYS = frozenset(
    {
        "name",
        "first_name",
        "last_name",
        "full_name",
        "email",
        "email_address",
        "phone",
        "phone_number",
        "address",
        "ip",
        "ip_address",
        "date_of_birth",
        "dob",
        "passport",
        "passport_number",
        "national_id",
        "government_id",
        "username",
        "cookie",
        "session_cookie",
    }
)
LABEL_DECLARATION_KEYS = frozenset(
    {
        "test_labels_exposed",
        "hidden_labels_exposed",
        "hidden_labels_included",
        "labels_exposed",
    }
)
LABEL_CONTENT_KEYS = frozenset(
    {
        "test_labels",
        "hidden_test_labels",
        "private_labels",
        "ground_truth",
        "answer_key",
        "label_values",
    }
)


class OrganizerBundleValidationError(ValueError):
    """A stable fail-closed validation error."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise OrganizerBundleValidationError(code, detail)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.stat().st_size <= MAX_JSON_BYTES, "JSON_FILE_TOO_LARGE", label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrganizerBundleValidationError("JSON_INVALID", label) from exc
    _require(isinstance(value, dict), "JSON_OBJECT_REQUIRED", label)
    return value


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _scan_sensitive_keys(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            _require(normalized not in PII_KEYS, "PII_KEY_FORBIDDEN", f"{location}.{key}")
            if normalized in LABEL_DECLARATION_KEYS:
                _require(nested is False, "TEST_LABEL_EXPOSURE_INVALID", f"{location}.{key}")
            if normalized in LABEL_CONTENT_KEYS:
                _require(nested in (None, False, "", [], {}), "TEST_LABEL_CONTENT_FORBIDDEN", f"{location}.{key}")
            _scan_sensitive_keys(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_sensitive_keys(nested, location=f"{location}[{index}]")


def _safe_relative_path(value: Any) -> str:
    raw = str(value or "")
    _require(bool(raw) and "\\" not in raw and not raw.startswith("/"), "MANIFEST_PATH_UNSAFE", raw)
    path = PurePosixPath(raw)
    _require(
        not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts),
        "MANIFEST_PATH_UNSAFE",
        raw,
    )
    return path.as_posix()


def _is_descendant(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_nonlocal_or_production_path(path: Path, *, write: bool) -> None:
    text = str(path)
    _require(not text.startswith("\\\\"), "NONLOCAL_PATH_FORBIDDEN", text)
    if write:
        normalized = "/" + text.replace("\\", "/").casefold().strip("/") + "/"
        _require("/programdata/evomind/" not in normalized, "PRODUCTION_WRITE_FORBIDDEN")


def _collect_actual_files(root: Path) -> dict[str, Path]:
    actual: dict[str, Path] = {}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            _require(not candidate.is_symlink(), "SYMLINK_FORBIDDEN", candidate.relative_to(root).as_posix())
        for filename in filenames:
            candidate = current_path / filename
            relative = candidate.relative_to(root).as_posix()
            _require(not candidate.is_symlink(), "SYMLINK_FORBIDDEN", relative)
            _require(candidate.is_file(), "REGULAR_FILE_REQUIRED", relative)
            actual[relative] = candidate
    return actual


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def deterministic_bootstrap_ci(scores: Sequence[float]) -> tuple[float, float]:
    """Return the deterministic participant-cluster percentile 95% CI."""

    _require(len(scores) >= MIN_PARTICIPANTS, "PARTICIPANT_COUNT_INVALID")
    sample_count = len(scores)
    seed = BOOTSTRAP_SEED.to_bytes(8, "big", signed=False)
    replicates: list[float] = []
    for replicate in range(BOOTSTRAP_ROUNDS):
        selected: list[float] = []
        counter = 0
        while len(selected) < sample_count:
            block = hashlib.sha256(
                seed + replicate.to_bytes(8, "big", signed=False) + counter.to_bytes(4, "big", signed=False)
            ).digest()
            counter += 1
            for offset in range(0, len(block), 8):
                selected.append(float(scores[int.from_bytes(block[offset : offset + 8], "big") % sample_count]))
                if len(selected) == sample_count:
                    break
        replicates.append(fmean(selected))
    replicates.sort()
    return _percentile(replicates, 0.025), _percentile(replicates, 0.975)


def _load_request_template(competition: str) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    _require(competition in SUPPORTED_COMPETITIONS, "COMPETITION_INVALID", competition)
    _require(TEMPLATE_PATH.is_file() and not TEMPLATE_PATH.is_symlink(), "REQUEST_TEMPLATE_MISSING")
    template = _read_json(TEMPLATE_PATH, label="request_template")
    _require(template.get("schema") == "evomind.g21_external_evidence_request_templates.v1", "REQUEST_TEMPLATE_SCHEMA_INVALID")
    _require(template.get("status") == "LOCAL_REQUEST_TEMPLATES_ONLY", "REQUEST_TEMPLATE_STATUS_INVALID")
    binding = template.get("binding")
    _require(isinstance(binding, dict), "REQUEST_TEMPLATE_BINDING_MISSING")
    _require(binding.get("run_id") == FIXED_RUN_ID, "REQUEST_TEMPLATE_RUN_BINDING_INVALID")
    _require(binding.get("allocation") == FIXED_ALLOCATION, "REQUEST_TEMPLATE_ALLOCATION_BINDING_INVALID")
    requests = template.get("organizer_evidence_requests")
    _require(isinstance(requests, dict) and isinstance(requests.get(competition), dict), "REQUEST_TEMPLATE_COMPETITION_MISSING")
    request_contract = dict(requests[competition])
    _require(
        request_contract.get("requested_contract_is_proposed_not_observed") is True,
        "REQUEST_TEMPLATE_PROPOSAL_MARKER_MISSING",
    )
    fingerprint_payload = {
        "template_schema": template["schema"],
        "binding": binding,
        "competition": competition,
        "request_contract": request_contract,
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(fingerprint_payload))
    return template, request_contract, _sha256_file(TEMPLATE_PATH), fingerprint


def compute_request_fingerprint(competition: str) -> str:
    """Return the fixed request-contract fingerprint for one competition."""

    return _load_request_template(competition)[3]


def _load_closed_manifest(
    root: Path,
    *,
    competition: str,
    request_fingerprint: str,
    allow_test_fixture: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Path], bool]:
    manifest_path = root / MANIFEST_NAME
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "MANIFEST_MISSING")
    manifest = _read_json(manifest_path, label=MANIFEST_NAME)
    _scan_sensitive_keys(manifest)
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "MANIFEST_SCHEMA_INVALID")
    _require(manifest.get("manifest_self_excluded") is True, "MANIFEST_SELF_EXCLUSION_MISSING")
    synthetic = manifest.get("synthetic_fixture") is True
    if synthetic:
        _require(allow_test_fixture, "SYNTHETIC_FIXTURE_FORBIDDEN")
        _require(manifest.get("test_only") is True, "SYNTHETIC_FIXTURE_TEST_ONLY_MARKER_MISSING")
        _require(manifest.get("inert_no_external_actions") is True, "SYNTHETIC_FIXTURE_INERT_MARKER_MISSING")
    else:
        _require(manifest.get("test_only") is not True, "TEST_ONLY_BUNDLE_FORBIDDEN")

    _require(manifest.get("run_id") == FIXED_RUN_ID, "RUN_BINDING_INVALID", MANIFEST_NAME)
    _require(manifest.get("allocation") == FIXED_ALLOCATION, "ALLOCATION_BINDING_INVALID", MANIFEST_NAME)
    _require(manifest.get("competition") == competition, "COMPETITION_BINDING_INVALID", MANIFEST_NAME)
    _require(manifest.get("request_fingerprint") == request_fingerprint, "REQUEST_FINGERPRINT_MISMATCH", MANIFEST_NAME)
    bundle_id = str(manifest.get("bundle_id") or "")
    _require(SAFE_ID_RE.fullmatch(bundle_id) is not None, "BUNDLE_ID_INVALID")

    files = manifest.get("files")
    _require(isinstance(files, list) and 0 < len(files) <= MAX_FILES, "MANIFEST_FILES_INVALID")
    by_role: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for raw in files:
        _require(isinstance(raw, dict), "MANIFEST_ENTRY_INVALID")
        relative = _safe_relative_path(raw.get("path"))
        role = str(raw.get("role") or "")
        artifact_id = str(raw.get("artifact_id") or "")
        byte_count = raw.get("bytes")
        sha256 = str(raw.get("sha256") or "").casefold()
        _require(SAFE_ID_RE.fullmatch(role) is not None, "MANIFEST_ROLE_INVALID", relative)
        _require(SAFE_ID_RE.fullmatch(artifact_id) is not None, "MANIFEST_ARTIFACT_ID_INVALID", relative)
        _require(isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count >= 0, "MANIFEST_BYTES_INVALID", relative)
        _require(SHA256_RE.fullmatch(sha256) is not None, "MANIFEST_SHA256_INVALID", relative)
        _require(raw.get("regular_file") is True and raw.get("symlink") is False, "MANIFEST_FILE_TYPE_INVALID", relative)
        _require(relative != MANIFEST_NAME, "MANIFEST_SELF_ENTRY_FORBIDDEN")
        _require(relative not in by_path, "MANIFEST_DUPLICATE_PATH", relative)
        _require(role not in by_role, "MANIFEST_DUPLICATE_ROLE", role)
        normalized = dict(raw)
        normalized.update({"path": relative, "role": role, "sha256": sha256})
        by_path[relative] = normalized
        by_role[role] = normalized
        total_bytes += byte_count

    _require(set(by_role) == REQUIRED_ROLES, "MANIFEST_ROLE_CLOSURE_MISMATCH")
    _require(manifest.get("file_count") == len(files), "MANIFEST_FILE_COUNT_MISMATCH")
    _require(manifest.get("total_bytes") == total_bytes, "MANIFEST_TOTAL_BYTES_MISMATCH")
    _require(total_bytes <= MAX_TOTAL_BYTES, "BUNDLE_TOO_LARGE")

    actual = _collect_actual_files(root)
    _require(set(actual) == set(by_path) | {MANIFEST_NAME}, "MANIFEST_CLOSURE_MISMATCH")
    for relative, entry in by_path.items():
        path = actual[relative]
        _require(path.stat().st_size == entry["bytes"], "FILE_BYTES_MISMATCH", relative)
        _require(_sha256_file(path) == entry["sha256"], "FILE_SHA256_MISMATCH", relative)
    return manifest, by_role, actual, synthetic


def _load_role_json(root: Path, by_role: Mapping[str, Mapping[str, Any]], role: str) -> dict[str, Any]:
    value = _read_json(root / str(by_role[role]["path"]), label=role)
    _scan_sensitive_keys(value, location=role)
    return value


def _identity_fields(value: Mapping[str, Any], *, competition: str, bundle_id: str, label: str) -> None:
    _require(value.get("run_id") == FIXED_RUN_ID, "RUN_BINDING_INVALID", label)
    _require(value.get("allocation") == FIXED_ALLOCATION, "ALLOCATION_BINDING_INVALID", label)
    _require(value.get("competition") == competition, "COMPETITION_BINDING_INVALID", label)
    _require(value.get("bundle_id") == bundle_id, "BUNDLE_BINDING_INVALID", label)


def _require_sha(value: Mapping[str, Any], field: str, expected: str, *, label: str) -> None:
    actual = str(value.get(field) or "").casefold()
    _require(SHA256_RE.fullmatch(actual) is not None, "SHA256_FIELD_INVALID", f"{label}.{field}")
    _require(actual == expected, "SHA_BINDING_MISMATCH", f"{label}.{field}")


def _require_https_or_doi(value: Any, *, label: str, allow_test_fixture: bool) -> str:
    raw = str(value or "").strip()
    _require(bool(raw), "SOURCE_URL_OR_DOI_MISSING", label)
    if DOI_RE.fullmatch(raw):
        return raw
    parsed = urlparse(raw)
    _require(parsed.scheme == "https" and bool(parsed.hostname), "SOURCE_URL_OR_DOI_INVALID", label)
    _require(parsed.username is None and parsed.password is None, "SOURCE_URL_CREDENTIALS_FORBIDDEN", label)
    host = str(parsed.hostname).casefold()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if not allow_test_fixture:
        _require(host not in {"localhost"} and not host.endswith(".localhost") and not host.endswith(".invalid"), "NONAUTHORITATIVE_SOURCE_HOST", label)
        _require(address is None or not (address.is_private or address.is_loopback or address.is_link_local), "NONAUTHORITATIVE_SOURCE_HOST", label)
    return raw


def _expected_receipt_endpoint(request_contract: Mapping[str, Any], evaluation_id: str) -> tuple[str, str]:
    endpoints = request_contract.get("requested_endpoints")
    _require(isinstance(endpoints, list) and endpoints, "REQUEST_TEMPLATE_ENDPOINTS_INVALID")
    receipt_endpoints = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, dict)
        and str(endpoint.get("method") or "").upper() == "GET"
        and "{evaluation_id}" in str(endpoint.get("path") or "")
    ]
    _require(len(receipt_endpoints) == 1, "REQUEST_TEMPLATE_RECEIPT_ENDPOINT_INVALID")
    endpoint = receipt_endpoints[0]
    return "GET", str(endpoint["path"]).replace("{evaluation_id}", evaluation_id)


def _require_response_fields(response: Mapping[str, Any], request_contract: Mapping[str, Any]) -> None:
    fields = request_contract.get("required_response_fields")
    _require(isinstance(fields, list) and fields, "REQUEST_TEMPLATE_RESPONSE_FIELDS_INVALID")
    for field in fields:
        key = str(field)
        _require(key in response and response[key] is not None and response[key] != "", "RESPONSE_REQUIRED_FIELD_MISSING", key)


def _participant_field_names(competition: str) -> tuple[str, str, str, str]:
    if competition == "e2lmc":
        return (
            "participant_level_sq_rc_cs_artifact_sha256",
            "human_composite_mean",
            "human_composite_ci95_lower",
            "human_composite_ci95_upper",
        )
    if competition == "ariel_2025":
        return (
            "participant_level_official_metric_artifact_sha256",
            "human_metric_mean",
            "human_metric_ci95_lower",
            "human_metric_ci95_upper",
        )
    return (
        "participant_level_wmae_artifact_sha256",
        "human_wmae_mean",
        "human_wmae_ci95_lower",
        "human_wmae_ci95_upper",
    )


def _claim_response_fields(competition: str) -> tuple[str, str, str, str]:
    if competition == "e2lmc":
        return (
            "private_experiment_claim_id",
            "private_experiment_manifest_sha256",
            "private_experiment_previously_consumed",
            "experiment_overlap_count",
        )
    if competition == "ariel_2025":
        return (
            "private_planet_claim_id",
            "private_planet_manifest_sha256",
            "private_planet_previously_consumed",
            "planet_overlap_count",
        )
    return (
        "private_molecule_claim_id",
        "private_molecule_target_manifest_sha256",
        "private_molecule_manifest_previously_consumed",
        "polymer_overlap_count",
    )


def _signature_binding_payload(
    *,
    competition: str,
    bundle_id: str,
    request_fingerprint: str,
    role_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "competition": competition,
        "bundle_id": bundle_id,
        "request_fingerprint": request_fingerprint,
        "source_sha256": role_hashes["organizer_response_source"],
        "protocol_sha256": role_hashes["protocol"],
        "official_scorer_sha256": role_hashes["official_scorer"],
        "private_evaluator_manifest_sha256": role_hashes["private_evaluator_manifest"],
        "holdout_claim_sha256": role_hashes["holdout_claim"],
    }


def validate_organizer_response_bundle(
    bundle_dir: Path,
    *,
    competition: str,
    expected_metric: str | None = None,
    expected_direction: str | None = None,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    """Validate one bundle and return an in-memory, non-authorizing receipt."""

    _require(competition in SUPPORTED_COMPETITIONS, "COMPETITION_INVALID", competition)
    if expected_direction is not None:
        _require(expected_direction in VALID_DIRECTIONS, "EXPECTED_DIRECTION_INVALID")
    original = Path(bundle_dir)
    _require(not original.is_symlink(), "BUNDLE_DIRECTORY_SYMLINK_FORBIDDEN")
    bundle_dir = original.resolve(strict=True)
    _reject_nonlocal_or_production_path(bundle_dir, write=False)
    _require(bundle_dir.is_dir(), "BUNDLE_DIRECTORY_INVALID")

    template, request_contract, template_sha256, request_fingerprint = _load_request_template(competition)
    manifest, by_role, _actual, synthetic = _load_closed_manifest(
        bundle_dir,
        competition=competition,
        request_fingerprint=request_fingerprint,
        allow_test_fixture=allow_test_fixture,
    )
    bundle_id = str(manifest["bundle_id"])
    payloads = {role: _load_role_json(bundle_dir, by_role, role) for role in JSON_ROLES}

    response = payloads["organizer_response"]
    if response.get("schema") == template.get("schema") or response.get("requested_contract_is_proposed_not_observed") is True:
        raise OrganizerBundleValidationError("REQUEST_TEMPLATE_REPLAY_FORBIDDEN")

    for role, value in payloads.items():
        _identity_fields(value, competition=competition, bundle_id=bundle_id, label=role)
        _require(value.get("request_fingerprint") == request_fingerprint, "REQUEST_FINGERPRINT_MISMATCH", role)
        if synthetic:
            _require(value.get("test_only") is True, "SYNTHETIC_FIXTURE_TEST_ONLY_MARKER_MISSING", role)
        else:
            _require(value.get("test_only") is not True, "TEST_ONLY_BUNDLE_FORBIDDEN", role)

    schemas = {
        "request_binding": REQUEST_SCHEMA,
        "organizer_response": RESPONSE_SCHEMA,
        "protocol": PROTOCOL_SCHEMA,
        "official_scorer": SCORER_SCHEMA,
        "private_evaluator_manifest": EVALUATOR_SCHEMA,
        "holdout_ledger": LEDGER_SCHEMA,
        "holdout_claim": CLAIM_SCHEMA,
        "participant_metrics": PARTICIPANT_METRICS_SCHEMA,
        "leakage_audit": LEAKAGE_SCHEMA,
        "organizer_signature_or_publication_receipt": PROOF_SCHEMA,
    }
    for role, schema in schemas.items():
        _require(payloads[role].get("schema") == schema, f"{role.upper()}_SCHEMA_INVALID")

    source = payloads["organizer_response_source"]
    _require(source.get("schema") != template.get("schema"), "REQUEST_TEMPLATE_REPLAY_FORBIDDEN")
    _require(source.get("requested_contract_is_proposed_not_observed") is not True, "REQUEST_TEMPLATE_REPLAY_FORBIDDEN")
    _require(source.get("response_kind") == "completed_organizer_evaluation", "ORGANIZER_SOURCE_NOT_COMPLETED_RESPONSE")

    request = payloads["request_binding"]
    _require(request.get("request_kind") == request_contract.get("request_kind"), "REQUEST_KIND_MISMATCH")
    _require(request.get("request_template_sha256") == template_sha256, "REQUEST_TEMPLATE_SHA256_MISMATCH")
    _require(request.get("requested_contract_is_proposed_not_observed") is True, "REQUEST_PROPOSAL_MARKER_MISSING")
    request_id = str(request.get("request_id") or "")
    _require(SAFE_ID_RE.fullmatch(request_id) is not None, "REQUEST_ID_INVALID")
    _require(SAFE_ID_RE.fullmatch(str(request.get("client_idempotency_key") or "")) is not None, "REQUEST_IDEMPOTENCY_KEY_INVALID")

    _require(response.get("schema") == RESPONSE_SCHEMA, "ORGANIZER_RESPONSE_SCHEMA_INVALID")
    _require(response.get("status") == "COMPLETED", "ORGANIZER_RESPONSE_NOT_COMPLETED")
    _require(response.get("request_id") == request_id, "RESPONSE_REQUEST_ID_MISMATCH")
    _require(response.get("request_kind") == request_contract.get("request_kind"), "RESPONSE_REQUEST_KIND_MISMATCH")
    _require(response.get("requested_contract_is_proposed_not_observed") is False, "RESPONSE_OBSERVED_MARKER_INVALID")
    _require(response.get("endpoint_observed") is True, "ENDPOINT_NOT_OBSERVED")
    evaluation_id = str(response.get("evaluation_id") or "")
    _require(SAFE_ID_RE.fullmatch(evaluation_id) is not None, "EVALUATION_ID_INVALID")
    _require_response_fields(response, request_contract)

    role_hashes = {role: str(entry["sha256"]) for role, entry in by_role.items()}
    source_entry = by_role["organizer_response_source"]
    source_url = _require_https_or_doi(
        response.get("source_url_or_doi"),
        label="organizer_response.source_url_or_doi",
        allow_test_fixture=synthetic and allow_test_fixture,
    )
    _require(response.get("source_authority") in SOURCE_AUTHORITIES, "SOURCE_AUTHORITY_INVALID")
    _require(len(str(response.get("source_title") or "").strip()) >= 8, "SOURCE_TITLE_INVALID")
    _require(response.get("source_filename") == Path(str(source_entry["path"])).name, "SOURCE_FILENAME_MISMATCH")
    _require(response.get("source_content_type") in {"application/json", "application/pdf", "text/html", "text/plain"}, "SOURCE_CONTENT_TYPE_INVALID")
    _require(response.get("source_bytes") == source_entry["bytes"], "SOURCE_BYTES_MISMATCH")
    _require_sha(response, "source_sha256", role_hashes["organizer_response_source"], label="organizer_response")
    _require(role_hashes["organizer_response_source"] != template_sha256, "REQUEST_TEMPLATE_REPLAY_FORBIDDEN")

    observed = response.get("observed_endpoint")
    _require(isinstance(observed, dict), "OBSERVED_ENDPOINT_MISSING")
    expected_method, expected_path = _expected_receipt_endpoint(request_contract, evaluation_id)
    _require(str(observed.get("method") or "").upper() == expected_method, "OBSERVED_ENDPOINT_METHOD_MISMATCH")
    endpoint_url = _require_https_or_doi(
        observed.get("url"),
        label="organizer_response.observed_endpoint.url",
        allow_test_fixture=synthetic and allow_test_fixture,
    )
    endpoint_parsed = urlparse(endpoint_url)
    _require(endpoint_parsed.scheme == "https", "OBSERVED_ENDPOINT_URL_INVALID")
    _require(endpoint_parsed.path == expected_path, "OBSERVED_ENDPOINT_PATH_MISMATCH")
    _require(observed.get("status_code") == 200, "OBSERVED_ENDPOINT_STATUS_INVALID")
    _require(bool(str(observed.get("observed_at_utc") or "")), "OBSERVED_ENDPOINT_TIME_MISSING")
    _require_sha(observed, "response_source_sha256", role_hashes["organizer_response_source"], label="observed_endpoint")

    protocol = payloads["protocol"]
    scorer = payloads["official_scorer"]
    evaluator = payloads["private_evaluator_manifest"]
    ledger = payloads["holdout_ledger"]
    claim = payloads["holdout_claim"]
    metrics = payloads["participant_metrics"]
    leakage = payloads["leakage_audit"]
    proof = payloads["organizer_signature_or_publication_receipt"]

    metric = str(protocol.get("official_metric") or "")
    direction = str(protocol.get("direction") or "")
    _require(bool(metric), "OFFICIAL_METRIC_MISSING")
    _require(direction in VALID_DIRECTIONS, "OFFICIAL_DIRECTION_INVALID")
    if competition in KNOWN_METRIC_CONTRACTS:
        known_metric, known_direction = KNOWN_METRIC_CONTRACTS[competition]
        _require(metric == known_metric, "OFFICIAL_METRIC_MISMATCH")
        _require(direction == known_direction, "OFFICIAL_DIRECTION_MISMATCH")
    else:
        normalized_metric = metric.casefold()
        _require(
            all(token not in normalized_metric for token in ("normalized rmse", "normalized_rmse", "internal proxy", "proxy metric")),
            "ARIEL_INTERNAL_PROXY_FORBIDDEN",
        )
        _require(protocol.get("metric_source") == "organizer_signed_official_scorer", "ARIEL_METRIC_SOURCE_INVALID")
    if expected_metric is not None:
        _require(metric == expected_metric, "EXPECTED_METRIC_MISMATCH")
    if expected_direction is not None:
        _require(direction == expected_direction, "EXPECTED_DIRECTION_MISMATCH")

    for label, value in (("official_scorer", scorer), ("organizer_response", response), ("participant_metrics", metrics)):
        _require(value.get("metric") == metric or value.get("official_metric") == metric, "METRIC_DRIFT_DETECTED", label)
        _require(value.get("direction") == direction, "DIRECTION_DRIFT_DETECTED", label)
    _require(protocol.get("protocol_comparable") is True, "PROTOCOL_NOT_COMPARABLE")
    _require(protocol.get("human_candidate_inputs_equal") is True, "PROTOCOL_INPUT_PARITY_MISSING")
    _require(protocol.get("confidence_level") == 0.95, "PROTOCOL_CONFIDENCE_INVALID")
    _require(protocol.get("bootstrap_method") == BOOTSTRAP_METHOD, "PROTOCOL_BOOTSTRAP_METHOD_INVALID")
    protocol_id = str(protocol.get("protocol_id") or "")
    _require(SAFE_ID_RE.fullmatch(protocol_id) is not None, "PROTOCOL_ID_INVALID")
    _require(len(str(protocol.get("protocol_text") or "").strip()) >= 20, "PROTOCOL_TEXT_INCOMPLETE")
    _require(response.get("protocol_id") == protocol_id, "RESPONSE_PROTOCOL_ID_MISMATCH")
    _require_sha(response, "protocol_sha256", role_hashes["protocol"], label="organizer_response")
    _require_sha(protocol, "official_scorer_sha256", role_hashes["official_scorer"], label="protocol")

    _require(scorer.get("official_organizer_issued") is True, "SCORER_NOT_ORGANIZER_ISSUED")
    _require(SAFE_ID_RE.fullmatch(str(scorer.get("scorer_id") or "")) is not None, "SCORER_ID_INVALID")
    implementation_sha = str(scorer.get("implementation_sha256") or "").casefold()
    _require(SHA256_RE.fullmatch(implementation_sha) is not None, "SCORER_IMPLEMENTATION_SHA256_INVALID")
    _require(response.get("official_scorer_sha256") == role_hashes["official_scorer"], "RESPONSE_SCORER_SHA256_MISMATCH")

    _require(evaluator.get("contract_status") == "OBSERVED_FROZEN", "PRIVATE_EVALUATOR_NOT_OBSERVED")
    _require(evaluator.get("endpoint_observed") is True, "PRIVATE_EVALUATOR_ENDPOINT_NOT_OBSERVED")
    _require(evaluator.get("outputs_scores_and_hashes_only") is True, "PRIVATE_EVALUATOR_OUTPUT_POLICY_INVALID")
    _require(evaluator.get("human_candidate_input_manifest_equal") is True, "PRIVATE_EVALUATOR_INPUT_PARITY_MISSING")
    _require_sha(evaluator, "protocol_sha256", role_hashes["protocol"], label="private_evaluator_manifest")
    _require_sha(evaluator, "official_scorer_sha256", role_hashes["official_scorer"], label="private_evaluator_manifest")
    _require_sha(evaluator, "source_sha256", role_hashes["organizer_response_source"], label="private_evaluator_manifest")
    _require_sha(evaluator, "participant_metrics_sha256", role_hashes["participant_metrics"], label="private_evaluator_manifest")
    _require(response.get("artifact_manifest_sha256") == role_hashes["private_evaluator_manifest"], "RESPONSE_ARTIFACT_MANIFEST_SHA256_MISMATCH")

    ledger_id = str(ledger.get("ledger_id") or "")
    claim_id = str(claim.get("claim_id") or "")
    _require(SAFE_ID_RE.fullmatch(ledger_id) is not None, "HOLDOUT_LEDGER_ID_INVALID")
    _require(SAFE_ID_RE.fullmatch(claim_id) is not None, "HOLDOUT_CLAIM_ID_INVALID")
    _require_sha(ledger, "private_evaluator_manifest_sha256", role_hashes["private_evaluator_manifest"], label="holdout_ledger")
    _require(claim.get("ledger_id") == ledger_id, "HOLDOUT_LEDGER_BINDING_INVALID")
    _require_sha(claim, "ledger_sha256", role_hashes["holdout_ledger"], label="holdout_claim")
    _require_sha(claim, "private_evaluator_manifest_sha256", role_hashes["private_evaluator_manifest"], label="holdout_claim")
    _require(claim.get("atomic_claim") is True, "HOLDOUT_ATOMIC_CLAIM_NOT_PROVEN")
    _require(SAFE_ID_RE.fullmatch(str(claim.get("claim_transaction_id") or "")) is not None, "HOLDOUT_CLAIM_TRANSACTION_INVALID")
    _require(claim.get("unconsumed_at_claim") is True, "HOLDOUT_NOT_UNCONSUMED_AT_CLAIM")
    _require(claim.get("previously_consumed") is False, "HOLDOUT_PREVIOUSLY_CONSUMED")
    _require(claim.get("consumed_once") is True, "HOLDOUT_SINGLE_CONSUMPTION_NOT_PROVEN")
    _require(claim.get("overlap_count") == 0, "HOLDOUT_OVERLAP_NONZERO")

    claim_field, evaluator_field, prior_field, overlap_field = _claim_response_fields(competition)
    _require(response.get(claim_field) == claim_id, "RESPONSE_HOLDOUT_CLAIM_ID_MISMATCH")
    _require(response.get(evaluator_field) == role_hashes["private_evaluator_manifest"], "RESPONSE_PRIVATE_EVALUATOR_SHA256_MISMATCH")
    _require(response.get(prior_field) is False, "RESPONSE_PRIVATE_EVALUATOR_PREVIOUSLY_CONSUMED")
    _require(response.get(overlap_field) == 0, "RESPONSE_PRIVATE_EVALUATOR_OVERLAP_NONZERO")

    records = metrics.get("records")
    _require(isinstance(records, list) and MIN_PARTICIPANTS <= len(records) <= MAX_PARTICIPANTS, "PARTICIPANT_COUNT_INVALID")
    _require(metrics.get("participant_count") == len(records), "PARTICIPANT_COUNT_MISMATCH")
    participant_ids: set[str] = set()
    scores: list[float] = []
    for index, record in enumerate(records):
        _require(isinstance(record, dict), "PARTICIPANT_RECORD_INVALID", str(index))
        participant_id = str(record.get("participant_id") or "")
        _require(PSEUDONYM_RE.fullmatch(participant_id) is not None, "PARTICIPANT_ID_INVALID", str(index))
        _require(participant_id not in participant_ids, "PARTICIPANT_ID_DUPLICATE", participant_id)
        participant_ids.add(participant_id)
        _require(_finite_number(record.get("score")), "PARTICIPANT_SCORE_INVALID", participant_id)
        scores.append(float(record["score"]))
    _require(response.get("human_included_count") == len(records), "RESPONSE_PARTICIPANT_COUNT_MISMATCH")
    participant_sha_field, mean_field, lower_field, upper_field = _participant_field_names(competition)
    _require(response.get(participant_sha_field) == role_hashes["participant_metrics"], "RESPONSE_PARTICIPANT_METRICS_SHA256_MISMATCH")
    mean = fmean(scores)
    ci_lower, ci_upper = deterministic_bootstrap_ci(scores)
    for field, expected in ((mean_field, mean), (lower_field, ci_lower), (upper_field, ci_upper)):
        _require(
            _finite_number(response.get(field))
            and math.isclose(float(response[field]), expected, rel_tol=0.0, abs_tol=1e-12),
            "RESPONSE_HUMAN_STATISTIC_MISMATCH",
            field,
        )

    if competition == "e2lmc":
        for field in ("candidate_sq", "candidate_rc", "candidate_cs", "candidate_composite"):
            _require(_finite_number(response.get(field)), "CANDIDATE_METRIC_INVALID", field)
    elif competition == "ariel_2025":
        _require(_finite_number(response.get("candidate_official_metric")), "CANDIDATE_METRIC_INVALID")
        _require(response.get("input_manifest_equal_for_human_and_candidate") is True, "ARIEL_INPUT_MANIFEST_PARITY_MISSING")
    else:
        _require(_finite_number(response.get("candidate_wmae")), "CANDIDATE_METRIC_INVALID")
        _require(response.get("targets") == OPEN_POLYMER_TARGETS, "OPEN_POLYMER_TARGETS_INVALID")
        _require(response.get("valid_pairs_equal_for_human_and_candidate") is True, "OPEN_POLYMER_VALID_PAIR_PARITY_MISSING")
        for field in ("target_weights_sha256", "valid_pair_manifest_sha256", "missing_prediction_policy_sha256"):
            _require(SHA256_RE.fullmatch(str(response.get(field) or "").casefold()) is not None, "OPEN_POLYMER_SHA256_FIELD_INVALID", field)

    for field, role in (
        ("source_sha256", "organizer_response_source"),
        ("protocol_sha256", "protocol"),
        ("official_scorer_sha256", "official_scorer"),
        ("private_evaluator_manifest_sha256", "private_evaluator_manifest"),
        ("holdout_claim_sha256", "holdout_claim"),
        ("participant_metrics_sha256", "participant_metrics"),
    ):
        _require_sha(leakage, field, role_hashes[role], label="leakage_audit")
    _require(leakage.get("hidden_labels_present_in_bundle") is False, "HIDDEN_LABELS_PRESENT_IN_BUNDLE")
    _require(leakage.get("candidate_saw_human_responses") is False, "CANDIDATE_HUMAN_RESPONSE_LEAK")
    _require(leakage.get("human_saw_candidate_outputs") is False, "HUMAN_CANDIDATE_OUTPUT_LEAK")
    _require(leakage.get("prior_overlap_count") == 0, "LEAKAGE_OVERLAP_NONZERO")

    _require(proof.get("proof_kind") in PROOF_KINDS, "ORGANIZER_PROOF_KIND_INVALID")
    _require(proof.get("verification_status") == "VERIFIED", "ORGANIZER_RESPONSE_UNSIGNED")
    _require(proof.get("signer_authority") == "official_organizer", "ORGANIZER_SIGNER_AUTHORITY_INVALID")
    _require(proof.get("source_url_or_doi") == source_url, "ORGANIZER_PROOF_SOURCE_MISMATCH")
    expected_signed_binding = _sha256_bytes(
        _canonical_json_bytes(
            _signature_binding_payload(
                competition=competition,
                bundle_id=bundle_id,
                request_fingerprint=request_fingerprint,
                role_hashes=role_hashes,
            )
        )
    )
    _require(proof.get("signed_binding_sha256") == expected_signed_binding, "ORGANIZER_PROOF_BINDING_MISMATCH")
    if proof.get("proof_kind") == "organizer_signature":
        _require(bool(str(proof.get("organizer_signature") or "").strip()), "ORGANIZER_RESPONSE_UNSIGNED")
        _require(SHA256_RE.fullmatch(str(proof.get("public_key_fingerprint_sha256") or "").casefold()) is not None, "ORGANIZER_PUBLIC_KEY_FINGERPRINT_INVALID")
    else:
        _require(SAFE_ID_RE.fullmatch(str(proof.get("publication_receipt_id") or "")) is not None, "PUBLICATION_RECEIPT_ID_INVALID")
        _require_https_or_doi(
            proof.get("publication_url_or_doi"),
            label="organizer_proof.publication_url_or_doi",
            allow_test_fixture=synthetic and allow_test_fixture,
        )
    _require(response.get("organizer_signature") == role_hashes["organizer_signature_or_publication_receipt"], "RESPONSE_ORGANIZER_PROOF_SHA256_MISMATCH")

    manifest_sha256 = _sha256_file(bundle_dir / MANIFEST_NAME)
    tree_descriptor = [
        {"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
        for entry in sorted(by_role.values(), key=lambda item: str(item["path"]))
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "VALIDATED_TEST_FIXTURE" if synthetic else "VALIDATED",
        "validated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "competition": competition,
        "bundle_id": bundle_id,
        "evaluation_id": evaluation_id,
        "request_template_sha256": template_sha256,
        "request_fingerprint": request_fingerprint,
        "bundle_manifest_sha256": manifest_sha256,
        "bundle_tree_sha256": _sha256_bytes(_canonical_json_bytes(tree_descriptor)),
        "file_count": len(by_role),
        "total_bytes": manifest["total_bytes"],
        "observed_endpoint": {"method": expected_method, "path": expected_path, "status_code": 200},
        "source": {
            "url_or_doi": source_url,
            "authority": response["source_authority"],
            "bytes": source_entry["bytes"],
            "sha256": role_hashes["organizer_response_source"],
        },
        "protocol_id": protocol_id,
        "protocol_sha256": role_hashes["protocol"],
        "official_metric": metric,
        "direction": direction,
        "official_scorer_sha256": role_hashes["official_scorer"],
        "private_evaluator_manifest_sha256": role_hashes["private_evaluator_manifest"],
        "holdout_claim_id": claim_id,
        "holdout_claim_sha256": role_hashes["holdout_claim"],
        "participant_count": len(records),
        "recomputed_human_mean": mean,
        "recomputed_human_ci95": {"lower": ci_lower, "upper": ci_upper},
        "test_labels_exposed": False,
        "synthetic_fixture": synthetic,
        "real_organizer_evidence_accepted": not synthetic,
        "training_authorized_by_receipt": False,
        "training_authorized_by_this_receipt": False,
        "ready_for_gpu": False,
        "claim_boundary": "This local receipt validates bundle structure and bindings only; it never authorizes training.",
        "side_effects": {
            "network_access": False,
            "production_writes": 0,
            "hpc_accessed": False,
            "gpu_touched": False,
            "remote_writes": 0,
            "signals_sent": 0,
            "training_tool_calls": 0,
        },
    }


def write_validation_receipt(receipt: Mapping[str, Any], receipt_path: Path, *, bundle_dir: Path) -> None:
    """Atomically create a receipt outside the immutable bundle directory."""

    bundle_root = Path(bundle_dir).resolve(strict=True)
    parent = receipt_path.parent.resolve(strict=True)
    target = parent / receipt_path.name
    _reject_nonlocal_or_production_path(target, write=True)
    _require(not _is_descendant(target, bundle_root), "RECEIPT_INSIDE_BUNDLE_FORBIDDEN")
    _require(not target.exists(), "RECEIPT_ALREADY_EXISTS", str(target))
    payload = _canonical_json_bytes(dict(receipt)) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--competition", choices=sorted(SUPPORTED_COMPETITIONS), required=True)
    parser.add_argument("--expected-metric")
    parser.add_argument("--expected-direction", choices=sorted(VALID_DIRECTIONS))
    parser.add_argument("--receipt", type=Path, required=True, help="New local receipt path outside the bundle")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = validate_organizer_response_bundle(
            args.bundle_dir,
            competition=args.competition,
            expected_metric=args.expected_metric,
            expected_direction=args.expected_direction,
            allow_test_fixture=False,
        )
        write_validation_receipt(receipt, args.receipt, bundle_dir=args.bundle_dir)
    except (OSError, OrganizerBundleValidationError) as exc:
        error = exc.to_dict() if isinstance(exc, OrganizerBundleValidationError) else {"code": "LOCAL_IO_ERROR", "detail": type(exc).__name__}
        print(json.dumps({"status": "REJECTED", "error": error}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.receipt.resolve()),
                "competition": receipt["competition"],
                "bundle_manifest_sha256": receipt["bundle_manifest_sha256"],
                "training_authorized_by_receipt": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
