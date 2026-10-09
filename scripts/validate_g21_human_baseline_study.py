"""Validate one local, source-bound G21 human-baseline study package.

This command is deliberately narrower than the production Goal API.  It reads
one closed local directory, recomputes participant-level summary statistics,
and (only after every check passes) writes a validation receipt outside that
directory.  It has no network, subprocess, production-database, HPC, GPU, or
training capability.

The package root must contain ``artifact-manifest.json``.  The manifest is
self-excluded (a file cannot contain its own hash) and must close every other
regular file in the directory.  Required roles are: ``protocol``,
``task_manifest``, ``scorer``, ``human_baseline_source``, ``holdout_ledger``,
``holdout_claim``, ``participant_metrics``, and ``human_baseline``.
"""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evomind_runtime import competition_goal


MANIFEST_NAME = "artifact-manifest.json"
MANIFEST_SCHEMA = "evomind.human-participant-baseline-study-manifest.v1"
RECEIPT_SCHEMA = "evomind.g21-human-baseline-study-validation-receipt.v1"
PROTOCOL_SCHEMA = "evomind.human-participant-baseline-study.v1"
TASK_SCHEMA = "evomind.human-baseline-task-manifest.v1"
SCORER_SCHEMA = "evomind.human-baseline-scorer.v1"
PARTICIPANT_METRICS_SCHEMA = "evomind.human-participant-metrics.v1"
HOLDOUT_LEDGER_SCHEMA = "evomind.holdout-ledger.v1"
HOLDOUT_CLAIM_SCHEMA = "evomind.holdout-claim.v1"

BOOTSTRAP_METHOD = "participant_cluster_percentile_bootstrap_v1"
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_ROUNDS = 10_000
MIN_PARTICIPANTS = 30
MAX_PARTICIPANTS = 100_000
MAX_FILES = 128
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
PSEUDONYM_RE = re.compile(r"^(?:p|participant)_[0-9a-f]{16,64}$")
REQUIRED_ROLES = frozenset(
    {
        "protocol",
        "task_manifest",
        "scorer",
        "human_baseline_source",
        "holdout_ledger",
        "holdout_claim",
        "participant_metrics",
        "human_baseline",
    }
)
JSON_ROLES = REQUIRED_ROLES - {"human_baseline_source"}
PII_KEYS = frozenset(
    {
        "name",
        "first_name",
        "middle_name",
        "last_name",
        "full_name",
        "legal_name",
        "display_name",
        "email",
        "email_address",
        "phone",
        "phone_number",
        "mobile",
        "mobile_number",
        "address",
        "street_address",
        "postal_address",
        "home_address",
        "ip",
        "ip_address",
        "mac_address",
        "dob",
        "date_of_birth",
        "birth_date",
        "ssn",
        "social_security_number",
        "passport",
        "passport_number",
        "national_id",
        "government_id",
        "username",
        "user_name",
        "handle",
        "biometric",
        "biometrics",
        "cookie",
        "session_cookie",
    }
)


class StudyValidationError(ValueError):
    """A deterministic fail-closed study package validation error."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise StudyValidationError(code, detail)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json_value(path: Path, *, label: str) -> Any:
    _require(path.stat().st_size <= MAX_JSON_BYTES, "JSON_FILE_TOO_LARGE", label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudyValidationError("JSON_INVALID", label) from exc
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_json_value(path, label=label)
    _require(isinstance(value, dict), "JSON_OBJECT_REQUIRED", label)
    return value


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _scan_pii_keys(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            _require(normalized not in PII_KEYS, "PII_KEY_FORBIDDEN", f"{location}.{key}")
            _scan_pii_keys(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_pii_keys(nested, location=f"{location}[{index}]")


def _scan_json_or_jsonl_for_pii(path: Path, *, relative_path: str) -> None:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        _scan_pii_keys(_read_json_value(path, label=relative_path))
        return
    if suffix != ".jsonl":
        return
    _require(path.stat().st_size <= MAX_JSON_BYTES, "JSON_FILE_TOO_LARGE", relative_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise StudyValidationError("JSONL_INVALID", relative_path) from exc
    for index, line in enumerate(lines, start=1):
        _require(bool(line.strip()), "JSONL_BLANK_LINE", f"{relative_path}:{index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudyValidationError("JSONL_INVALID", f"{relative_path}:{index}") from exc
        _scan_pii_keys(value, location=f"{relative_path}:{index}")


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
    normalized = text.replace("\\", "/").casefold()
    if write:
        _require("/programdata/evomind/" not in f"/{normalized.strip('/')}/", "PRODUCTION_WRITE_FORBIDDEN")


def _collect_actual_files(root: Path) -> dict[str, Path]:
    actual: dict[str, Path] = {}
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            _require(not (current_path / directory).is_symlink(), "SYMLINK_FORBIDDEN", directory)
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            _require(not path.is_symlink(), "SYMLINK_FORBIDDEN", relative)
            _require(path.is_file(), "REGULAR_FILE_REQUIRED", relative)
            actual[relative] = path
    return actual


def _load_closed_manifest(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Path]]:
    manifest_path = root / MANIFEST_NAME
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "MANIFEST_MISSING")
    manifest = _read_json(manifest_path, label=MANIFEST_NAME)
    _scan_pii_keys(manifest)
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "MANIFEST_SCHEMA_INVALID")
    _require(manifest.get("manifest_self_excluded") is True, "MANIFEST_SELF_EXCLUSION_MISSING")
    _require(manifest.get("synthetic_fixture") is not True, "SYNTHETIC_FIXTURE_FORBIDDEN")

    files = manifest.get("files")
    _require(isinstance(files, list), "MANIFEST_FILES_INVALID")
    _require(0 < len(files) <= MAX_FILES, "MANIFEST_FILE_COUNT_INVALID")
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
        _require(role and SAFE_ID_RE.fullmatch(role) is not None, "MANIFEST_ROLE_INVALID", relative)
        _require(SAFE_ID_RE.fullmatch(artifact_id) is not None, "MANIFEST_ARTIFACT_ID_INVALID", relative)
        _require(
            isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count >= 0,
            "MANIFEST_BYTES_INVALID",
            relative,
        )
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

    _require(REQUIRED_ROLES <= set(by_role), "MANIFEST_REQUIRED_ROLES_MISSING")
    _require(manifest.get("file_count") == len(files), "MANIFEST_FILE_COUNT_MISMATCH")
    _require(manifest.get("total_bytes") == total_bytes, "MANIFEST_TOTAL_BYTES_MISMATCH")
    _require(total_bytes <= MAX_TOTAL_BYTES, "PACKAGE_TOO_LARGE")

    actual = _collect_actual_files(root)
    _require(set(actual) == set(by_path) | {MANIFEST_NAME}, "MANIFEST_CLOSURE_MISMATCH")
    for relative, entry in by_path.items():
        path = actual[relative]
        _require(path.stat().st_size == entry["bytes"], "FILE_BYTES_MISMATCH", relative)
        _require(_sha256_file(path) == entry["sha256"], "FILE_SHA256_MISMATCH", relative)
        _scan_json_or_jsonl_for_pii(path, relative_path=relative)
    return manifest, by_role, actual


def _identity_fields(value: Mapping[str, Any], *, competition: str, study_id: str, label: str) -> None:
    _require(value.get("run_id") == competition_goal.FIXED_RUN_ID, "RUN_BINDING_INVALID", label)
    _require(value.get("allocation") == competition_goal.FIXED_ALLOCATION, "ALLOCATION_BINDING_INVALID", label)
    _require(value.get("competition") == competition, "COMPETITION_BINDING_INVALID", label)
    _require(value.get("study_id") == study_id, "STUDY_BINDING_INVALID", label)


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
    """Return a deterministic participant-cluster percentile 95% CI.

    The SHA-256 counter stream avoids dependence on platform PRNG internals.
    Each participant is one primary resampling unit and contributes one score.
    """

    _require(len(scores) >= MIN_PARTICIPANTS, "PARTICIPANT_COUNT_TOO_SMALL")
    sample_count = len(scores)
    replicates: list[float] = []
    seed = BOOTSTRAP_SEED.to_bytes(8, "big", signed=False)
    for replicate in range(BOOTSTRAP_ROUNDS):
        selected: list[float] = []
        counter = 0
        while len(selected) < sample_count:
            block = hashlib.sha256(
                seed + replicate.to_bytes(8, "big", signed=False) + counter.to_bytes(4, "big", signed=False)
            ).digest()
            counter += 1
            for offset in range(0, len(block), 8):
                selected.append(scores[int.from_bytes(block[offset : offset + 8], "big") % sample_count])
                if len(selected) == sample_count:
                    break
        replicates.append(fmean(selected))
    replicates.sort()
    return _percentile(replicates, 0.025), _percentile(replicates, 0.975)


def _load_role_json(root: Path, by_role: Mapping[str, Mapping[str, Any]], role: str) -> dict[str, Any]:
    entry = by_role[role]
    return _read_json(root / str(entry["path"]), label=role)


def _require_sha_binding(value: Mapping[str, Any], field: str, expected: str, *, label: str) -> None:
    _require(str(value.get(field) or "").casefold() == expected, "SHA_BINDING_MISMATCH", f"{label}.{field}")


def validate_study_package(
    package_dir: Path,
    *,
    competition: str,
    expected_metric: str,
    expected_direction: str,
) -> dict[str, Any]:
    """Validate a closed package and return an in-memory receipt payload."""

    _require(competition in competition_goal.COMPETITIONS, "COMPETITION_INVALID", competition)
    _require(bool(expected_metric), "EXPECTED_METRIC_REQUIRED")
    _require(expected_direction in competition_goal.VALID_DIRECTIONS, "EXPECTED_DIRECTION_INVALID")
    package_dir = package_dir.resolve(strict=True)
    _reject_nonlocal_or_production_path(package_dir, write=False)
    _require(package_dir.is_dir() and not package_dir.is_symlink(), "PACKAGE_DIRECTORY_INVALID")

    manifest, by_role, _actual = _load_closed_manifest(package_dir)
    _require(manifest.get("run_id") == competition_goal.FIXED_RUN_ID, "RUN_BINDING_INVALID", MANIFEST_NAME)
    _require(manifest.get("allocation") == competition_goal.FIXED_ALLOCATION, "ALLOCATION_BINDING_INVALID", MANIFEST_NAME)
    _require(manifest.get("competition") == competition, "COMPETITION_BINDING_INVALID", MANIFEST_NAME)
    study_id = str(manifest.get("study_id") or "")
    _require(SAFE_ID_RE.fullmatch(study_id) is not None, "STUDY_ID_INVALID")

    payloads = {role: _load_role_json(package_dir, by_role, role) for role in JSON_ROLES}
    for role, value in payloads.items():
        _identity_fields(value, competition=competition, study_id=study_id, label=role)

    protocol = payloads["protocol"]
    task = payloads["task_manifest"]
    scorer = payloads["scorer"]
    ledger = payloads["holdout_ledger"]
    claim = payloads["holdout_claim"]
    participant_metrics = payloads["participant_metrics"]
    human_baseline = payloads["human_baseline"]

    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_INVALID")
    _require(task.get("schema") == TASK_SCHEMA, "TASK_SCHEMA_INVALID")
    _require(scorer.get("schema") == SCORER_SCHEMA, "SCORER_SCHEMA_INVALID")
    _require(ledger.get("schema") == HOLDOUT_LEDGER_SCHEMA, "HOLDOUT_LEDGER_SCHEMA_INVALID")
    _require(claim.get("schema") == HOLDOUT_CLAIM_SCHEMA, "HOLDOUT_CLAIM_SCHEMA_INVALID")
    _require(participant_metrics.get("schema") == PARTICIPANT_METRICS_SCHEMA, "PARTICIPANT_METRICS_SCHEMA_INVALID")

    for label, value in (
        ("protocol", protocol),
        ("task_manifest", task),
        ("scorer", scorer),
        ("participant_metrics", participant_metrics),
        ("human_baseline", human_baseline),
    ):
        _require(value.get("metric") == expected_metric, "METRIC_MISMATCH", label)
        _require(value.get("direction") == expected_direction, "DIRECTION_MISMATCH", label)

    _require(protocol.get("participant_type") == "human", "PARTICIPANT_TYPE_INVALID")
    _require(
        protocol.get("participant_id_scheme") == "sha256_salted_pseudonym_v1",
        "PARTICIPANT_ID_SCHEME_INVALID",
    )
    protocol_text = str(protocol.get("protocol_text") or "")
    _require(len(protocol_text) >= 20, "PROTOCOL_TEXT_INCOMPLETE")
    _require(human_baseline.get("protocol") == protocol_text, "PROTOCOL_TEXT_MISMATCH")
    _require(
        human_baseline.get("protocol_sha256") == _sha256_text(protocol_text),
        "PROTOCOL_TEXT_SHA256_MISMATCH",
    )
    _require(human_baseline.get("protocol_id") == protocol.get("protocol_id"), "PROTOCOL_ID_MISMATCH")

    bootstrap = protocol.get("bootstrap")
    _require(isinstance(bootstrap, dict), "BOOTSTRAP_CONTRACT_MISSING")
    _require(bootstrap.get("method") == BOOTSTRAP_METHOD, "BOOTSTRAP_METHOD_INVALID")
    _require(bootstrap.get("seed") == BOOTSTRAP_SEED, "BOOTSTRAP_SEED_INVALID")
    _require(bootstrap.get("rounds") == BOOTSTRAP_ROUNDS, "BOOTSTRAP_ROUNDS_INVALID")
    _require(bootstrap.get("confidence_level") == 0.95, "BOOTSTRAP_CONFIDENCE_INVALID")
    _require(bootstrap.get("resampling_unit") == "human_participant", "BOOTSTRAP_UNIT_INVALID")

    role_hashes = {role: str(entry["sha256"]) for role, entry in by_role.items()}
    for field, role in (
        ("task_manifest_sha256", "task_manifest"),
        ("scorer_sha256", "scorer"),
        ("source_sha256", "human_baseline_source"),
        ("holdout_ledger_sha256", "holdout_ledger"),
        ("holdout_claim_sha256", "holdout_claim"),
    ):
        _require_sha_binding(protocol, field, role_hashes[role], label="protocol")
        _require_sha_binding(human_baseline, field, role_hashes[role], label="human_baseline")
    _require_sha_binding(human_baseline, "protocol_file_sha256", role_hashes["protocol"], label="human_baseline")
    _require_sha_binding(task, "scorer_sha256", role_hashes["scorer"], label="task_manifest")
    _require_sha_binding(ledger, "task_manifest_sha256", role_hashes["task_manifest"], label="holdout_ledger")
    _require_sha_binding(claim, "task_manifest_sha256", role_hashes["task_manifest"], label="holdout_claim")
    _require_sha_binding(claim, "ledger_sha256", role_hashes["holdout_ledger"], label="holdout_claim")

    _require(task.get("participant_blind_to_labels") is True, "PARTICIPANT_LABEL_BLINDING_MISSING")
    _require(task.get("participant_blind_to_candidate_outputs") is True, "PARTICIPANT_CANDIDATE_BLINDING_MISSING")
    _require(task.get("test_labels_exposed") is False, "TEST_LABEL_EXPOSURE_INVALID")
    _require(scorer.get("primary_aggregation_unit") == "human_participant", "SCORER_AGGREGATION_UNIT_INVALID")
    _require(scorer.get("participant_score_field") == "score", "SCORER_SCORE_FIELD_INVALID")
    _require(scorer.get("aggregation") == "arithmetic_mean", "SCORER_AGGREGATION_INVALID")

    ledger_id = str(ledger.get("ledger_id") or "")
    claim_id = str(claim.get("claim_id") or "")
    _require(SAFE_ID_RE.fullmatch(ledger_id) is not None, "HOLDOUT_LEDGER_ID_INVALID")
    _require(SAFE_ID_RE.fullmatch(claim_id) is not None, "HOLDOUT_CLAIM_ID_INVALID")
    _require(claim.get("ledger_id") == ledger_id, "HOLDOUT_LEDGER_BINDING_INVALID")
    _require(claim.get("unconsumed_at_claim") is True, "HOLDOUT_NOT_UNCONSUMED_AT_CLAIM")
    _require(claim.get("consumed_once") is True, "HOLDOUT_SINGLE_CONSUMPTION_NOT_PROVEN")
    _require(claim.get("selection_data_scope") == "development_only", "HOLDOUT_SELECTION_SCOPE_INVALID")
    _require(claim.get("prior_holdout_overlap") == 0, "HOLDOUT_OVERLAP_NONZERO")
    _require(claim.get("test_labels_exposed") is False, "HOLDOUT_TEST_LABEL_EXPOSURE_INVALID")

    records = participant_metrics.get("records")
    _require(isinstance(records, list), "PARTICIPANT_RECORDS_INVALID")
    _require(MIN_PARTICIPANTS <= len(records) <= MAX_PARTICIPANTS, "PARTICIPANT_COUNT_INVALID")
    participant_ids: set[str] = set()
    scores: list[float] = []
    for index, record in enumerate(records):
        _require(isinstance(record, dict), "PARTICIPANT_RECORD_INVALID", str(index))
        participant_id = str(record.get("participant_id") or "")
        _require(PSEUDONYM_RE.fullmatch(participant_id) is not None, "PARTICIPANT_ID_NOT_PSEUDONYMOUS", str(index))
        _require(participant_id not in participant_ids, "PARTICIPANT_ID_DUPLICATE", participant_id)
        participant_ids.add(participant_id)
        score = record.get("score")
        _require(_finite_number(score), "PARTICIPANT_SCORE_INVALID", participant_id)
        scores.append(float(score))
    _require(participant_metrics.get("participant_count") == len(records), "PARTICIPANT_COUNT_MISMATCH")

    recomputed_mean = fmean(scores)
    ci_lower, ci_upper = deterministic_bootstrap_ci(scores)
    _require(human_baseline.get("sample_size") == len(records), "HUMAN_BASELINE_SAMPLE_SIZE_MISMATCH")
    _require(
        _finite_number(human_baseline.get("mean"))
        and math.isclose(float(human_baseline["mean"]), recomputed_mean, rel_tol=0.0, abs_tol=1e-12),
        "HUMAN_BASELINE_MEAN_MISMATCH",
    )
    uncertainty = human_baseline.get("uncertainty")
    _require(isinstance(uncertainty, dict), "HUMAN_BASELINE_UNCERTAINTY_MISSING")
    _require(uncertainty.get("method") == BOOTSTRAP_METHOD, "HUMAN_BASELINE_BOOTSTRAP_METHOD_MISMATCH")
    _require(uncertainty.get("confidence_level") == 0.95, "HUMAN_BASELINE_CONFIDENCE_MISMATCH")
    _require(
        _finite_number(uncertainty.get("lower"))
        and math.isclose(float(uncertainty["lower"]), ci_lower, rel_tol=0.0, abs_tol=1e-12),
        "HUMAN_BASELINE_CI_LOWER_MISMATCH",
    )
    _require(
        _finite_number(uncertainty.get("upper"))
        and math.isclose(float(uncertainty["upper"]), ci_upper, rel_tol=0.0, abs_tol=1e-12),
        "HUMAN_BASELINE_CI_UPPER_MISMATCH",
    )
    _require(
        human_baseline.get("source_sha256") == role_hashes["human_baseline_source"],
        "HUMAN_BASELINE_SOURCE_SHA_MISMATCH",
    )
    _require(
        human_baseline.get("evidence_artifact_sha256") == role_hashes["human_baseline_source"],
        "HUMAN_BASELINE_EVIDENCE_SHA_MISMATCH",
    )
    _require(
        human_baseline.get("evidence_artifact_id") == by_role["human_baseline_source"]["artifact_id"],
        "HUMAN_BASELINE_EVIDENCE_ID_MISMATCH",
    )

    runtime_result = competition_goal.validate_human_baseline(
        human_baseline,
        expected_metric=expected_metric,
        expected_direction=expected_direction,
    )
    _require(
        runtime_result.valid,
        "RUNTIME_HUMAN_BASELINE_REJECTED",
        ",".join(issue.code for issue in runtime_result.issues),
    )

    manifest_sha256 = _sha256_file(package_dir / MANIFEST_NAME)
    tree_descriptor = [
        {"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
        for entry in sorted(by_role.values(), key=lambda item: str(item["path"]))
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "VALIDATED",
        "validated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": competition_goal.FIXED_RUN_ID,
        "allocation": competition_goal.FIXED_ALLOCATION,
        "competition": competition,
        "study_id": study_id,
        "package_manifest_sha256": manifest_sha256,
        "package_tree_sha256": hashlib.sha256(_canonical_json_bytes(tree_descriptor)).hexdigest(),
        "file_count": len(by_role),
        "total_bytes": manifest["total_bytes"],
        "participant_count": len(records),
        "participant_id_scheme": "sha256_salted_pseudonym_v1",
        "pii_keys_found": 0,
        "metric": expected_metric,
        "direction": expected_direction,
        "recomputed_mean": recomputed_mean,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "seed": BOOTSTRAP_SEED,
            "rounds": BOOTSTRAP_ROUNDS,
            "confidence_level": 0.95,
            "resampling_unit": "human_participant",
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "runtime_human_baseline_validation": runtime_result.to_dict(),
        "synthetic_evidence_accepted": False,
        "training_authorized_by_this_receipt": False,
        "side_effects": {
            "production_writes": 0,
            "hpc_accessed": False,
            "gpu_touched": False,
            "remote_writes": 0,
            "signals_sent": 0,
            "training_tool_calls": 0,
        },
    }


def write_validation_receipt(receipt: Mapping[str, Any], receipt_path: Path, *, package_dir: Path) -> None:
    """Atomically create a receipt outside the closed package directory."""

    package_root = package_dir.resolve(strict=True)
    parent = receipt_path.parent.resolve(strict=True)
    target = parent / receipt_path.name
    _reject_nonlocal_or_production_path(target, write=True)
    _require(not _is_descendant(target, package_root), "RECEIPT_INSIDE_PACKAGE_FORBIDDEN")
    _require(not target.exists(), "RECEIPT_ALREADY_EXISTS", str(target))
    payload = _canonical_json_bytes(dict(receipt)) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
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
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--competition", choices=competition_goal.COMPETITIONS, required=True)
    parser.add_argument("--expected-metric", required=True)
    parser.add_argument("--expected-direction", choices=sorted(competition_goal.VALID_DIRECTIONS), required=True)
    parser.add_argument("--receipt", type=Path, required=True, help="New local receipt path outside the package directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = validate_study_package(
            args.package_dir,
            competition=args.competition,
            expected_metric=args.expected_metric,
            expected_direction=args.expected_direction,
        )
        write_validation_receipt(receipt, args.receipt, package_dir=args.package_dir)
    except (OSError, StudyValidationError) as exc:
        if isinstance(exc, StudyValidationError):
            error = exc.to_dict()
        else:
            error = {"code": "LOCAL_IO_ERROR", "detail": type(exc).__name__}
        print(json.dumps({"status": "REJECTED", "error": error}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.receipt.resolve()),
                "competition": receipt["competition"],
                "participant_count": receipt["participant_count"],
                "package_manifest_sha256": receipt["package_manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
