"""Strict, side-effect-free contract for the fixed G21 five-competition goal.

The runtime already owns execution and artifact storage.  This module only
validates the evidence needed to call a goal item complete; it never starts a
process, changes a Run, consumes a holdout, or contacts an external service.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
# Compatibility alias for callers that use the shorter allocation name.
ALLOCATION = FIXED_ALLOCATION
COMPETITIONS = (
    "cure_bench",
    "e2lmc",
    "mindgames",
    "open_polymer",
    "ariel_2025",
)

GOAL_BOARD_SCHEMA = "evomind.goal-board.v2"
LEGACY_GOAL_CONTRACT_SCHEMA = "evomind.five_competition_goal.v1"
GOAL_CONTRACT_SCHEMA = "evomind.five_competition_goal.v2"
GOAL_RECORD_SCHEMA = "evomind.goal-record.v1"
FIXED_GOAL_KEY = "g21_five_competition"
FIXED_GOAL_ID = "goal_g21_five_competition"
VALID_GOAL_RECORD_STATUSES = frozenset({"active", "blocked", "completed"})
VALID_STATUSES = frozenset(
    {
        "WAITING_EXTERNAL_GATE",
        "READY_FOR_GPU",
        "VERIFYING_5_OF_5",
        "TRAINING",
        "INDEPENDENT_REVIEW",
        "VERIFIED_COMPLETE",
        "WAITING_EXACT_GATE",
    }
)
VALID_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})
PROHIBITED_COMPETITIONS = frozenset({"weather4cast"})
PROHIBITED_TOOLS = frozenset(
    {
        "competition_data_prepare",
        "competition_data_accelerate",
        "kaggle_download",
        "kaggle_submit",
        "join_competition",
        "accept_terms",
    }
)
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
INITIAL_NEGATIVE_HUMAN_BASELINE_EVIDENCE_SHA256 = (
    "5b6ed3d912ad1cd4655c96f832629cf9603a4b0ad58ab6e27c01861fadd69c5f"
)
INITIAL_GOAL_SPEC_FILE_SHA256 = (
    "380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f"
)
INITIAL_GOAL_SPEC_CANONICAL_SHA256 = (
    "a569bae731723fc7046817df737876d6dc7674eadd9f7b812fe7b4a87e521655"
)
CONDITIONAL_BASELINE_POLICY_MODE = "organizer_or_public_strong_with_conditional_human"
CONDITIONAL_BASELINE_MIGRATION_ID = "g21-conditional-strong-baseline-v2"
CONDITIONAL_BASELINE_AUTHORIZATION_SHA256 = (
    "f1edf39c1be5a03ff0da8e02acbfecfe906d72f39abb2e912f5fad6aafd754ae"
)
CONDITIONAL_BASELINE_POLICY_EVIDENCE_SHA256 = (
    "f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0"
)
CONDITIONAL_BASELINE_POLICY_EVIDENCE_FILE_SHA256 = (
    "47df860192502afcaacd7b1b6bc7fc4d86110c2e665278c8b7b3ea421893b7a7"
)
CONDITIONAL_GOAL_SPEC_CANONICAL_SHA256 = (
    "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
)
PRIMARY_BASELINE_TYPES = frozenset({"official_organizer", "publicly_reproducible_strong"})
AUTHORITATIVE_SOURCE_TYPES = frozenset({"official_organizer", "peer_reviewed", "official_dataset"})
OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA = "evomind.existing-candidate-adoption.v2"
OPEN_POLYMER_LEGACY_MANIFEST_SCHEMA = "evomind.open_polymer.artifact_manifest.v2"
OPEN_POLYMER_LEGACY_HOLDOUT_SCHEMA = "evomind.legacy-holdout-evidence.v1"
OPEN_POLYMER_LEGACY_REVIEW_BINDING_SCHEMA = "evomind.legacy-independent-review-binding.v1"
OPEN_POLYMER_LEGACY_SOURCE_CALL_ID = "call_54e3a187cfc179efd9060a283489691a"
OPEN_POLYMER_LEGACY_SOLUTION_ID = "open-polymer-v2-baseline-candidate-1"
OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY = "open-polymer-v2-baseline-candidate-1"
OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID = "artifact_8e4bd06da6d34f32a862a8a2966a48ce"
OPEN_POLYMER_LEGACY_MANIFEST_SHA256 = "98262bfca5a5dcb5720889d5d21e4880eea1b2a8dc5dbca6a50612aca4e67b9c"
OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID = "artifact_c2fc96d5c7ce47eaa831a56707866c4d"
OPEN_POLYMER_LEGACY_SPLIT_SHA256 = "8c53c5a3876b46fae65f9096ca5fbbe75ce221ab29c822812c50d49dc9c66fbb"
OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID = "artifact_38b633544092490c87b937f71c7423e4"
OPEN_POLYMER_LEGACY_REVIEW_SHA256 = "971e0edd49e7f6273f186fc08ca2c9c62778cdadaeb55fcb3c067e7b80b94b2c"
OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_ARTIFACT_ID = "artifact_c2e86ec7c0b44430845e227f7cc5e118"
OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_SHA256 = "94982c5452fafbc7bc9187cc91ccd7131056baeb842873b8d76b025cad5af9d6"
OPEN_POLYMER_LEGACY_MANIFEST_RECEIPT_BYTES = 140
OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ = 2844
OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ = 2865
OPEN_POLYMER_LEGACY_REQUIRED_FILES = frozenset(
    {
        "baseline-evidence.json",
        "baseline-reproduction.json",
        "candidate-history.jsonl",
        "candidate-vs-baseline.json",
        "dataset-audit.json",
        "environment-lock.json",
        "holdout-predictions.csv",
        "independent-verification.json",
        "metrics.json",
        "model-bundle.joblib",
        "retrospective-memory.json",
        "solution.py",
        "split-manifest.json",
        "task-contract-v2.json",
        "training-config.json",
        "training.log",
    }
)
ARTIFACT_ROLE_FIELDS = {
    "command": "command_sha256",
    "environment": "environment_sha256",
    "data_manifest": "data_manifest_sha256",
    "source_code": "source_code_sha256",
    "model": "model_sha256",
    "metrics": "metrics_sha256",
    "rollback": "rollback_sha256",
}
EVIDENCE_ARTIFACT_ROLES = frozenset(
    {*ARTIFACT_ROLE_FIELDS, "holdout_ledger", "holdout_claim", "human_baseline_source"}
)
BASE_EVIDENCE_ARTIFACT_ROLES = frozenset(
    {*ARTIFACT_ROLE_FIELDS, "holdout_ledger", "holdout_claim"}
)
DEFAULT_GOAL_SPEC_RELATIVE = "configs/g21_five_competition_goal.json"


@dataclass(frozen=True)
class GateIssue:
    """One deterministic reason an item cannot be promoted."""

    code: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class HumanBaselineResult:
    valid: bool
    mean: float | None = None
    direction: str = ""
    metric: str = ""
    issues: tuple[GateIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "mean": self.mean,
            "direction": self.direction,
            "metric": self.metric,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class PrimaryBaselineResult:
    valid: bool
    value: float | None = None
    direction: str = ""
    metric: str = ""
    baseline_type: str = ""
    issues: tuple[GateIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "value": self.value,
            "direction": self.direction,
            "metric": self.metric,
            "baseline_type": self.baseline_type,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class MetricGateResult:
    valid: bool
    normalized_delta: float | None = None
    normalized_ci_lower: float | None = None
    issues: tuple[GateIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "normalized_delta": self.normalized_delta,
            "normalized_ci_lower": self.normalized_ci_lower,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class GoalItemResult:
    competition: str
    declared_status: str
    strict_verified_complete: bool
    issues: tuple[GateIssue, ...] = ()
    human_baseline: HumanBaselineResult = field(
        default_factory=lambda: HumanBaselineResult(False)
    )
    primary_baseline: PrimaryBaselineResult = field(
        default_factory=lambda: PrimaryBaselineResult(False)
    )
    metric_gate: MetricGateResult = field(default_factory=lambda: MetricGateResult(False))
    human_metric_gate: MetricGateResult = field(default_factory=lambda: MetricGateResult(False))
    human_baseline_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition": self.competition,
            "declared_status": self.declared_status,
            "strict_verified_complete": self.strict_verified_complete,
            "issues": [item.to_dict() for item in self.issues],
            "human_baseline": self.human_baseline.to_dict(),
            "primary_baseline": self.primary_baseline.to_dict(),
            "metric_gate": self.metric_gate.to_dict(),
            "human_metric_gate": self.human_metric_gate.to_dict(),
            "human_baseline_required": self.human_baseline_required,
        }


@dataclass(frozen=True)
class GoalBoardResult:
    valid: bool
    run_id: str
    allocation: str
    complete_count: int
    target_count: int
    issues: tuple[GateIssue, ...]
    items: tuple[GoalItemResult, ...]

    @property
    def completion_proven(self) -> bool:
        return self.valid and self.complete_count == self.target_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "completion_proven": self.completion_proven,
            "run_id": self.run_id,
            "allocation": self.allocation,
            "complete_count": self.complete_count,
            "target_count": self.target_count,
            "issues": [item.to_dict() for item in self.issues],
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class GoalRecordResult:
    valid: bool
    goal_id: str = ""
    run_id: str = ""
    allocation: str = ""
    goal_key: str = ""
    status: str = ""
    spec_sha256: str = ""
    board_sha256: str = ""
    human_baseline_sha256: str = ""
    issues: tuple[GateIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "goal_id": self.goal_id,
            "run_id": self.run_id,
            "allocation": self.allocation,
            "goal_key": self.goal_key,
            "status": self.status,
            "spec_sha256": self.spec_sha256,
            "board_sha256": self.board_sha256,
            "human_baseline_sha256": self.human_baseline_sha256,
            "issues": [item.to_dict() for item in self.issues],
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 representation used for persisted evidence hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha(value: Any) -> bool:
    return bool(SHA256_RE.fullmatch(_text(value)))


def _first_mapping(item: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _authoritative_source_present(value: Mapping[str, Any]) -> bool:
    source_url = _text(value.get("source_url"))
    source_doi = _text(value.get("source_doi"))
    doi_ok = bool(DOI_RE.fullmatch(source_doi))
    url_ok = False
    if source_url:
        try:
            parsed = urlsplit(source_url)
        except ValueError:
            parsed = None
        if parsed is not None:
            hostname = str(parsed.hostname or "").casefold()
            reserved = (
                not hostname
                or hostname in {"localhost", "example.com", "example.org", "example.net"}
                or hostname.endswith((".invalid", ".localhost", ".test"))
            )
            url_ok = parsed.scheme.casefold() == "https" and not reserved
    return doi_ok or url_ok


def _safe_manifest_path(value: Any) -> bool:
    raw = _text(value)
    if not raw or "\\" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return False
    path = PurePosixPath(raw)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def validate_human_baseline(
    value: Any,
    *,
    expected_metric: str = "",
    expected_direction: str = "",
) -> HumanBaselineResult:
    """Validate a source-backed human-participant baseline record.

    Model-agent references, leaderboard values, and organizer baselines are
    intentionally rejected by requiring an explicit human participant type,
    sample size, comparable protocol, and source-bound uncertainty.
    """

    if not isinstance(value, Mapping):
        return HumanBaselineResult(False, issues=(GateIssue("HUMAN_BASELINE_MISSING"),))
    issues: list[GateIssue] = []
    status = _text(value.get("status")).upper()
    if status not in {"VERIFIED", "VERIFIED_HUMAN"}:
        issues.append(GateIssue("HUMAN_BASELINE_STATUS_INVALID"))
    participant_type = _text(
        value.get("participant_type") or value.get("population") or value.get("subject_type")
    ).casefold()
    if participant_type not in {"human", "human_participants", "people"}:
        issues.append(GateIssue("HUMAN_PARTICIPANTS_NOT_PROVEN"))
    if not _authoritative_source_present(value):
        issues.append(GateIssue("HUMAN_BASELINE_SOURCE_MISSING"))
    if _text(value.get("source_authority")) not in AUTHORITATIVE_SOURCE_TYPES:
        issues.append(GateIssue("HUMAN_BASELINE_SOURCE_AUTHORITY_INVALID"))
    source_sha = _text(value.get("source_sha256")).lower()
    evidence_sha = _text(value.get("evidence_artifact_sha256")).lower()
    evidence_id = _text(value.get("evidence_artifact_id"))
    if not _sha(source_sha):
        issues.append(GateIssue("HUMAN_BASELINE_SOURCE_SHA_INVALID"))
    if not SAFE_ID_RE.fullmatch(evidence_id):
        issues.append(GateIssue("HUMAN_BASELINE_EVIDENCE_ARTIFACT_ID_INVALID"))
    if not _sha(evidence_sha) or evidence_sha != source_sha:
        issues.append(GateIssue("HUMAN_BASELINE_EVIDENCE_ARTIFACT_SHA_MISMATCH"))
    if not _text(value.get("title")):
        issues.append(GateIssue("HUMAN_BASELINE_TITLE_MISSING"))
    sample_size = value.get("sample_size", value.get("participant_count"))
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 2:
        issues.append(GateIssue("HUMAN_BASELINE_SAMPLE_SIZE_INVALID"))
    protocol = _text(value.get("protocol") or value.get("evaluation_protocol"))
    if len(protocol) < 20:
        issues.append(GateIssue("HUMAN_BASELINE_PROTOCOL_INCOMPLETE"))
    if not SAFE_ID_RE.fullmatch(_text(value.get("protocol_id"))):
        issues.append(GateIssue("HUMAN_BASELINE_PROTOCOL_ID_INVALID"))
    if value.get("protocol_comparable") is not True:
        issues.append(GateIssue("HUMAN_BASELINE_PROTOCOL_NOT_COMPARABLE"))
    if (
        not protocol
        or not _sha(value.get("protocol_sha256"))
        or _text(value.get("protocol_sha256")).lower() != sha256_bytes(protocol.encode("utf-8"))
    ):
        issues.append(GateIssue("HUMAN_BASELINE_PROTOCOL_SHA_MISMATCH"))
    metric = _text(value.get("metric") or value.get("metric_name"))
    direction = _text(value.get("direction") or value.get("metric_direction"))
    if not metric or (expected_metric and metric != expected_metric):
        issues.append(GateIssue("HUMAN_BASELINE_METRIC_MISMATCH"))
    if direction not in VALID_DIRECTIONS or (expected_direction and direction != expected_direction):
        issues.append(GateIssue("HUMAN_BASELINE_DIRECTION_INVALID"))
    mean = value.get("mean", value.get("participant_mean"))
    if not _finite_number(mean):
        issues.append(GateIssue("HUMAN_BASELINE_MEAN_INVALID"))
    uncertainty = _first_mapping(value, "uncertainty", "ci95", "confidence_interval")
    has_interval = (
        _finite_number(uncertainty.get("lower"))
        and _finite_number(uncertainty.get("upper"))
        and float(uncertainty["lower"]) <= float(uncertainty["upper"])
        and _finite_number(mean)
        and float(uncertainty["lower"]) <= float(mean) <= float(uncertainty["upper"])
        and _finite_number(uncertainty.get("confidence_level"))
        and abs(float(uncertainty["confidence_level"]) - 0.95) < 1e-12
        and bool(_text(uncertainty.get("method")))
        and _text(uncertainty.get("source_sha256")).lower() == source_sha
    )
    if not has_interval:
        issues.append(GateIssue("HUMAN_BASELINE_UNCERTAINTY_MISSING"))
    return HumanBaselineResult(
        not issues,
        float(mean) if _finite_number(mean) else None,
        direction,
        metric,
        tuple(issues),
    )


def validate_primary_baseline(
    value: Any,
    *,
    expected_metric: str = "",
    expected_direction: str = "",
) -> PrimaryBaselineResult:
    """Validate the organizer or publicly reproducible strong baseline.

    A leaderboard number without source bytes and a reproduction contract is
    deliberately insufficient.  Human-participant evidence remains a separate
    contract and cannot be smuggled in as a primary model baseline.
    """

    if not isinstance(value, Mapping):
        return PrimaryBaselineResult(False, issues=(GateIssue("PRIMARY_BASELINE_MISSING"),))
    issues: list[GateIssue] = []
    if _text(value.get("status")).upper() != "VERIFIED":
        issues.append(GateIssue("PRIMARY_BASELINE_STATUS_INVALID"))
    baseline_type = _text(value.get("baseline_type")).casefold()
    if baseline_type not in PRIMARY_BASELINE_TYPES:
        issues.append(GateIssue("PRIMARY_BASELINE_TYPE_INVALID"))
    participant_type = _text(value.get("participant_type") or value.get("population")).casefold()
    if participant_type in {"human", "human_participants", "people"}:
        issues.append(GateIssue("HUMAN_BASELINE_MUST_USE_HUMAN_CONTRACT"))
    if not _authoritative_source_present(value):
        issues.append(GateIssue("PRIMARY_BASELINE_SOURCE_MISSING"))
    source_authority = _text(value.get("source_authority"))
    if source_authority not in AUTHORITATIVE_SOURCE_TYPES:
        issues.append(GateIssue("PRIMARY_BASELINE_SOURCE_AUTHORITY_INVALID"))
    if baseline_type == "official_organizer" and (
        source_authority != "official_organizer"
        or value.get("official_organizer_baseline") is not True
    ):
        issues.append(GateIssue("OFFICIAL_ORGANIZER_BASELINE_NOT_PROVEN"))
    source_sha = _text(value.get("source_sha256")).lower()
    evidence_sha = _text(value.get("evidence_artifact_sha256")).lower()
    if not _sha(source_sha):
        issues.append(GateIssue("PRIMARY_BASELINE_SOURCE_SHA_INVALID"))
    if not SAFE_ID_RE.fullmatch(_text(value.get("evidence_artifact_id"))):
        issues.append(GateIssue("PRIMARY_BASELINE_EVIDENCE_ARTIFACT_ID_INVALID"))
    if not _sha(evidence_sha) or evidence_sha != source_sha:
        issues.append(GateIssue("PRIMARY_BASELINE_EVIDENCE_ARTIFACT_SHA_MISMATCH"))
    if not _text(value.get("title")):
        issues.append(GateIssue("PRIMARY_BASELINE_TITLE_MISSING"))
    protocol = _text(value.get("protocol") or value.get("evaluation_protocol"))
    if len(protocol) < 20:
        issues.append(GateIssue("PRIMARY_BASELINE_PROTOCOL_INCOMPLETE"))
    if not SAFE_ID_RE.fullmatch(_text(value.get("protocol_id"))):
        issues.append(GateIssue("PRIMARY_BASELINE_PROTOCOL_ID_INVALID"))
    if value.get("protocol_comparable") is not True:
        issues.append(GateIssue("PRIMARY_BASELINE_PROTOCOL_NOT_COMPARABLE"))
    if (
        not protocol
        or not _sha(value.get("protocol_sha256"))
        or _text(value.get("protocol_sha256")).lower()
        != sha256_bytes(protocol.encode("utf-8"))
    ):
        issues.append(GateIssue("PRIMARY_BASELINE_PROTOCOL_SHA_MISMATCH"))
    metric = _text(value.get("metric") or value.get("metric_name"))
    direction = _text(value.get("direction") or value.get("metric_direction"))
    if not metric or (expected_metric and metric != expected_metric):
        issues.append(GateIssue("PRIMARY_BASELINE_METRIC_MISMATCH"))
    if direction not in VALID_DIRECTIONS or (expected_direction and direction != expected_direction):
        issues.append(GateIssue("PRIMARY_BASELINE_DIRECTION_INVALID"))
    baseline_value = value.get("value", value.get("mean", value.get("score")))
    if not _finite_number(baseline_value):
        issues.append(GateIssue("PRIMARY_BASELINE_VALUE_INVALID"))
    if baseline_type == "publicly_reproducible_strong":
        if value.get("publicly_reproducible") is not True:
            issues.append(GateIssue("STRONG_BASELINE_REPRODUCIBILITY_NOT_PROVEN"))
        for field_name in (
            "implementation_sha256",
            "data_manifest_sha256",
            "environment_sha256",
            "reproduction_receipt_sha256",
        ):
            if not _sha(value.get(field_name)):
                issues.append(GateIssue(f"STRONG_BASELINE_{field_name.upper()}_INVALID"))
    return PrimaryBaselineResult(
        not issues,
        float(baseline_value) if _finite_number(baseline_value) else None,
        direction,
        metric,
        baseline_type,
        tuple(issues),
    )


def compare_metric_to_primary(
    *,
    candidate_metric: Any,
    baseline_value: Any,
    ci95_lower: Any,
    ci95_upper: Any,
    direction: str,
) -> MetricGateResult:
    issues: list[GateIssue] = []
    if direction not in VALID_DIRECTIONS:
        issues.append(GateIssue("METRIC_DIRECTION_INVALID"))
    values = (candidate_metric, baseline_value, ci95_lower, ci95_upper)
    if not all(_finite_number(value) for value in values):
        issues.append(GateIssue("PRIMARY_BASELINE_METRIC_OR_CI_INVALID"))
    if not issues and float(ci95_lower) > float(ci95_upper):
        issues.append(GateIssue("CI_BOUNDS_REVERSED"))
    if issues:
        return MetricGateResult(False, issues=tuple(issues))
    candidate = float(candidate_metric)
    baseline = float(baseline_value)
    lower = float(ci95_lower)
    upper = float(ci95_upper)
    if direction == "higher_is_better":
        delta = candidate - baseline
        ci_lower = lower - baseline
    else:
        delta = baseline - candidate
        ci_lower = baseline - upper
    if delta <= 0:
        issues.append(GateIssue("PRIMARY_BASELINE_POINT_ESTIMATE_NOT_BETTER"))
    if ci_lower <= 0:
        issues.append(GateIssue("PRIMARY_BASELINE_CI_LOWER_NOT_ABOVE_ZERO"))
    return MetricGateResult(not issues, delta, ci_lower, tuple(issues))


def _human_baseline_requirement(
    item: Mapping[str, Any], baseline_value: Any
) -> tuple[bool, tuple[GateIssue, ...]]:
    availability = _first_mapping(item, "human_baseline_availability")
    baseline_present = isinstance(baseline_value, Mapping) and bool(baseline_value)
    status = _text(availability.get("status")).upper()
    issues: list[GateIssue] = []
    if baseline_present:
        if status == "NOT_VERIFIED_AVAILABLE":
            issues.append(GateIssue("HUMAN_BASELINE_AVAILABILITY_CONFLICT"))
        return True, tuple(issues)
    if status == "VERIFIED_AVAILABLE":
        return True, (GateIssue("HUMAN_BASELINE_REQUIRED_WHEN_VERIFIED_AVAILABLE"),)
    if status != "NOT_VERIFIED_AVAILABLE":
        issues.append(GateIssue("HUMAN_BASELINE_AVAILABILITY_MISSING"))
        return False, tuple(issues)
    if (
        availability.get("audit_complete") is not True
        or availability.get("protocol_search_complete") is not True
        or availability.get("human_baseline_verified") is not False
    ):
        issues.append(GateIssue("HUMAN_BASELINE_NEGATIVE_AUDIT_INVALID"))
    if not SAFE_ID_RE.fullmatch(_text(availability.get("evidence_artifact_id"))):
        issues.append(GateIssue("HUMAN_BASELINE_AVAILABILITY_ARTIFACT_ID_INVALID"))
    if not _sha(availability.get("evidence_artifact_sha256")):
        issues.append(GateIssue("HUMAN_BASELINE_AVAILABILITY_ARTIFACT_SHA_INVALID"))
    return False, tuple(issues)


def conditional_baseline_policy(spec: Mapping[str, Any] | None) -> bool:
    if not isinstance(spec, Mapping):
        return False
    metric_gate = spec.get("metric_gate") if isinstance(spec.get("metric_gate"), Mapping) else {}
    return (
        spec.get("schema") == GOAL_CONTRACT_SCHEMA
        and metric_gate.get("mode") == CONDITIONAL_BASELINE_POLICY_MODE
    )


def validate_policy_migration_evidence(
    value: Any,
    *,
    new_spec: Mapping[str, Any] | None = None,
) -> tuple[bool, tuple[GateIssue, ...]]:
    issues: list[GateIssue] = []
    if not isinstance(value, Mapping):
        return False, (GateIssue("POLICY_MIGRATION_EVIDENCE_INVALID"),)
    expected_identity = {
        "schema": "evomind.g21_goal_policy_migration_evidence.v1",
        "migration_id": CONDITIONAL_BASELINE_MIGRATION_ID,
        "goal_id": FIXED_GOAL_ID,
        "goal_key": FIXED_GOAL_KEY,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "previous_goal_spec_file_sha256": INITIAL_GOAL_SPEC_FILE_SHA256,
        "previous_goal_spec_canonical_sha256": INITIAL_GOAL_SPEC_CANONICAL_SHA256,
    }
    for field_name, expected in expected_identity.items():
        if _text(value.get(field_name)) != expected:
            issues.append(GateIssue(f"POLICY_MIGRATION_{field_name.upper()}_MISMATCH"))
    authorization = value.get("authorization") if isinstance(value.get("authorization"), Mapping) else {}
    if (
        authorization.get("source") != "explicit_user_confirmation"
        or _text(authorization.get("normalized_statement_sha256")).lower()
        != CONDITIONAL_BASELINE_AUTHORIZATION_SHA256
    ):
        issues.append(GateIssue("POLICY_MIGRATION_AUTHORIZATION_INVALID"))
    policy = value.get("authorized_policy") if isinstance(value.get("authorized_policy"), Mapping) else {}
    if (
        policy.get("primary_baseline_required") is not True
        or tuple(policy.get("accepted_primary_baseline_types") or ())
        != tuple(sorted(PRIMARY_BASELINE_TYPES))
        or policy.get("point_estimate_must_improve") is not True
        or policy.get("bootstrap_95_percent_ci_lower_bound_must_improve") is not True
        or policy.get("human_baseline_additionally_required_only_when_verified_available") is not True
        or policy.get("protocol_comparable_human_baseline_must_not_be_ignored") is not True
    ):
        issues.append(GateIssue("POLICY_MIGRATION_AUTHORIZED_POLICY_INVALID"))
    guards = value.get("preserved_guards") if isinstance(value.get("preserved_guards"), Mapping) else {}
    required_true_guards = {
        "fixed_run_only",
        "fixed_allocation_only",
        "single_gpu_serial",
        "test_labels_forbidden",
        "fake_human_evidence_forbidden",
        "weather_forbidden",
        "automatic_submission_forbidden",
        "join_forbidden",
        "terms_acceptance_forbidden",
        "other_gpu_process_modification_forbidden",
    }
    if any(guards.get(field_name) is not True for field_name in required_true_guards):
        issues.append(GateIssue("POLICY_MIGRATION_PRESERVED_GUARDS_INVALID"))
    if isinstance(new_spec, Mapping):
        if _text(new_spec.get("policy_evidence_sha256")).lower() != sha256_json(value):
            issues.append(GateIssue("POLICY_MIGRATION_EVIDENCE_SHA_MISMATCH"))
        if _text(new_spec.get("policy_migration_id")) != _text(value.get("migration_id")):
            issues.append(GateIssue("POLICY_MIGRATION_SPEC_ID_MISMATCH"))
    if sha256_json(value) != CONDITIONAL_BASELINE_POLICY_EVIDENCE_SHA256:
        issues.append(GateIssue("POLICY_MIGRATION_EVIDENCE_IMMUTABILITY_MISMATCH"))
    return not issues, tuple(dict.fromkeys(issues))


def compare_metric_to_human(
    *,
    candidate_metric: Any,
    human_mean: Any,
    ci95_lower: Any,
    ci95_upper: Any,
    direction: str,
) -> MetricGateResult:
    """Apply the conservative direction-normalized human gate."""

    issues: list[GateIssue] = []
    if direction not in VALID_DIRECTIONS:
        issues.append(GateIssue("METRIC_DIRECTION_INVALID"))
    values = (candidate_metric, human_mean, ci95_lower, ci95_upper)
    if not all(_finite_number(value) for value in values):
        issues.append(GateIssue("METRIC_OR_CI_INVALID"))
    if not issues and float(ci95_lower) > float(ci95_upper):
        issues.append(GateIssue("CI_BOUNDS_REVERSED"))
    if issues:
        return MetricGateResult(False, issues=tuple(issues))
    candidate = float(candidate_metric)
    mean = float(human_mean)
    lower = float(ci95_lower)
    upper = float(ci95_upper)
    if direction == "higher_is_better":
        delta = candidate - mean
        ci_lower = lower - mean
    else:
        delta = mean - candidate
        # For a lower-is-better metric the conservative bound is the upper CI.
        ci_lower = mean - upper
    if delta <= 0:
        issues.append(GateIssue("HUMAN_GATE_POINT_ESTIMATE_NOT_BETTER"))
    if ci_lower <= 0:
        issues.append(GateIssue("HUMAN_GATE_CI_LOWER_NOT_ABOVE_ZERO"))
    return MetricGateResult(not issues, delta, ci_lower, tuple(issues))


def _validate_holdout_contract(item: Mapping[str, Any]) -> list[GateIssue]:
    ledger = _first_mapping(item, "holdout_ledger")
    issues: list[GateIssue] = []
    if not SAFE_ID_RE.fullmatch(_text(ledger.get("ledger_id"))):
        issues.append(GateIssue("HOLDOUT_LEDGER_ID_INVALID"))
    if not _sha(ledger.get("ledger_sha256")):
        issues.append(GateIssue("HOLDOUT_LEDGER_SHA_INVALID"))
    if not SAFE_ID_RE.fullmatch(_text(ledger.get("claim_id"))):
        issues.append(GateIssue("HOLDOUT_CLAIM_ID_INVALID"))
    if not _sha(ledger.get("claim_sha256")):
        issues.append(GateIssue("HOLDOUT_CLAIM_SHA_INVALID"))
    if ledger.get("unconsumed_at_claim") is not True:
        issues.append(GateIssue("HOLDOUT_NOT_PROVEN_UNCONSUMED_AT_CLAIM"))
    if ledger.get("consumed_once") is not True:
        issues.append(GateIssue("HOLDOUT_SINGLE_CONSUMPTION_NOT_PROVEN"))
    if ledger.get("selection_data_scope") != "development_only":
        issues.append(GateIssue("HOLDOUT_SELECTION_SCOPE_INVALID"))
    return issues


def _validate_open_polymer_legacy_artifact_contract(
    item: Mapping[str, Any], *, solution_id: str
) -> list[GateIssue]:
    """Validate the honest, non-retroactive contract for the pre-Goal candidate.

    The 2026-08-29 Open Polymer candidate predates the fixed Goal and therefore
    has no atomic holdout-ledger claim.  This contract deliberately records that
    absence instead of manufacturing a claim after evaluation.  Store-backed
    verification still has to bind every listed file to the historical
    completed tool call before the board can be updated.
    """

    issues: list[GateIssue] = []
    manifest = _first_mapping(item, "artifact_manifest", "legacy_artifact_manifest")
    manifest_sha = _text(item.get("artifact_manifest_sha256")).lower()
    if manifest.get("schema") != OPEN_POLYMER_LEGACY_MANIFEST_SCHEMA:
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_MANIFEST_SCHEMA_INVALID"))
    if not SAFE_ID_RE.fullmatch(_text(item.get("artifact_manifest_artifact_id"))):
        issues.append(GateIssue("ARTIFACT_MANIFEST_ARTIFACT_ID_INVALID"))
    if not _sha(manifest_sha):
        issues.append(GateIssue("ARTIFACT_MANIFEST_SHA_MISSING"))
    if manifest.get("exit_code") != 0 or manifest.get("gate_passed") is not True:
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_MANIFEST_GATE_INVALID"))
    if _text(manifest.get("source_sha256")).lower() != _text(item.get("source_code_sha256")).lower():
        issues.append(GateIssue("SOURCE_CODE_SHA256_NOT_TRACEABLE"))

    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            issues.append(GateIssue("OPEN_POLYMER_LEGACY_MANIFEST_ENTRY_INVALID"))
            continue
        name = _text(raw.get("name"))
        byte_count = _count(raw.get("bytes"))
        if (
            not name
            or PurePosixPath(name).name != name
            or name in {".", ".."}
            or name in by_name
            or byte_count is None
            or not _sha(raw.get("sha256"))
        ):
            issues.append(GateIssue("OPEN_POLYMER_LEGACY_MANIFEST_ENTRY_INVALID", name))
            continue
        by_name[name] = raw
    if set(by_name) != set(OPEN_POLYMER_LEGACY_REQUIRED_FILES):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_MANIFEST_FILE_SET_INVALID"))
    bindings = item.get("legacy_artifact_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(by_name):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_ARTIFACT_BINDINGS_INCOMPLETE"))
        bindings = {}
    elif (
        any(not SAFE_ID_RE.fullmatch(_text(value)) for value in bindings.values())
        or len({_text(value) for value in bindings.values()}) != len(bindings)
    ):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_ARTIFACT_BINDING_ID_INVALID"))

    trace = {
        "training-config.json": "command_sha256",
        "environment-lock.json": "environment_sha256",
        "split-manifest.json": "data_manifest_sha256",
        "solution.py": "source_code_sha256",
        "model-bundle.joblib": "model_sha256",
        "metrics.json": "metrics_sha256",
        # No model is deployed by adoption.  The historical retrospective is
        # the immutable rollback/failure-strategy evidence for this legacy run.
        "retrospective-memory.json": "rollback_sha256",
    }
    for name, field_name in trace.items():
        declared = _text(item.get(field_name)).lower()
        if not _sha(declared):
            issues.append(GateIssue(f"{field_name.upper()}_MISSING"))
        elif _text(by_name.get(name, {}).get("sha256")).lower() != declared:
            issues.append(GateIssue(f"{field_name.upper()}_NOT_TRACEABLE"))
    if item.get("rollback_evidence_mode") != "legacy_completed_candidate_not_deployed":
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_ROLLBACK_MODE_INVALID"))

    holdout = _first_mapping(item, "legacy_holdout_evidence")
    if item.get("holdout_ledger") or item.get("holdout_claim"):
        issues.append(GateIssue("OPEN_POLYMER_POSTHOC_HOLDOUT_CLAIM_FORBIDDEN"))
    if holdout.get("schema") != OPEN_POLYMER_LEGACY_HOLDOUT_SCHEMA:
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_HOLDOUT_SCHEMA_INVALID"))
    if (
        holdout.get("atomic_claim_available") is not False
        or holdout.get("historical_claim_not_asserted") is not True
        or holdout.get("posthoc_ledger_created") is not False
        or holdout.get("selection_data_scope") != "development_only"
        or holdout.get("holdout_blind_for_selection") is not True
        or holdout.get("group_overlap") != 0
        or holdout.get("test_labels_used") is not False
    ):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_HOLDOUT_FACTS_INVALID"))
    if (
        not SAFE_ID_RE.fullmatch(_text(holdout.get("split_manifest_artifact_id")))
        or not _sha(holdout.get("split_manifest_sha256"))
        or _text(holdout.get("split_manifest_sha256")).lower()
        != _text(by_name.get("split-manifest.json", {}).get("sha256")).lower()
        or _text(bindings.get("split-manifest.json")).lower()
        != _text(holdout.get("split_manifest_artifact_id")).lower()
    ):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_SPLIT_BINDING_INVALID"))

    review = _first_mapping(item, "independent_review")
    if (
        review.get("schema") != OPEN_POLYMER_LEGACY_REVIEW_BINDING_SCHEMA
        or review.get("source_schema") != "evomind.open_polymer.independent_verification.v2"
        or not SAFE_ID_RE.fullmatch(_text(review.get("artifact_id")))
        or not _sha(review.get("sha256"))
        or review.get("verified") is not True
        or _text(review.get("sha256")).lower()
        != _text(by_name.get("independent-verification.json", {}).get("sha256")).lower()
        or _text(bindings.get("independent-verification.json")).lower()
        != _text(review.get("artifact_id")).lower()
    ):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_REVIEW_BINDING_INVALID"))

    primary = _first_mapping(item, "primary_baseline", "strong_baseline")
    if (
        _text(primary.get("evidence_artifact_id"))
        != _text(bindings.get("baseline-evidence.json"))
        or _text(primary.get("evidence_artifact_sha256")).lower()
        != _text(by_name.get("baseline-evidence.json", {}).get("sha256")).lower()
        or _text(primary.get("implementation_sha256")).lower()
        != _text(by_name.get("solution.py", {}).get("sha256")).lower()
        or _text(primary.get("data_manifest_sha256")).lower()
        != _text(by_name.get("split-manifest.json", {}).get("sha256")).lower()
        or _text(primary.get("environment_sha256")).lower()
        != _text(by_name.get("environment-lock.json", {}).get("sha256")).lower()
        or _text(primary.get("reproduction_receipt_sha256")).lower()
        != _text(by_name.get("baseline-reproduction.json", {}).get("sha256")).lower()
    ):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_PRIMARY_BASELINE_BINDING_INVALID"))

    adoption = _first_mapping(item, "existing_candidate_adoption")
    if (
        adoption.get("schema") != OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA
        or adoption.get("mode") != "legacy_pre_goal_completed_candidate"
        or adoption.get("run_id") != FIXED_RUN_ID
        or _text(adoption.get("allocation")).casefold() != FIXED_ALLOCATION.casefold()
        or adoption.get("competition") != "open_polymer"
        or adoption.get("source_tool_call_id") != OPEN_POLYMER_LEGACY_SOURCE_CALL_ID
        or adoption.get("source_idempotency_key") != OPEN_POLYMER_LEGACY_IDEMPOTENCY_KEY
        or adoption.get("source_solution_id") != OPEN_POLYMER_LEGACY_SOLUTION_ID
        or adoption.get("source_started_event_seq") != OPEN_POLYMER_LEGACY_STARTED_EVENT_SEQ
        or adoption.get("source_completed_event_seq") != OPEN_POLYMER_LEGACY_COMPLETED_EVENT_SEQ
        or adoption.get("source_artifact_manifest_artifact_id")
        != OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID
        or _text(adoption.get("source_artifact_manifest_sha256")).lower()
        != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
        or adoption.get("source_split_manifest_artifact_id")
        != OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID
        or _text(adoption.get("source_split_manifest_sha256")).lower()
        != OPEN_POLYMER_LEGACY_SPLIT_SHA256
        or adoption.get("source_independent_verification_artifact_id")
        != OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID
        or _text(adoption.get("source_independent_verification_sha256")).lower()
        != OPEN_POLYMER_LEGACY_REVIEW_SHA256
        or item.get("artifact_manifest_artifact_id") != OPEN_POLYMER_LEGACY_MANIFEST_ARTIFACT_ID
        or manifest_sha != OPEN_POLYMER_LEGACY_MANIFEST_SHA256
        or holdout.get("split_manifest_artifact_id") != OPEN_POLYMER_LEGACY_SPLIT_ARTIFACT_ID
        or _text(holdout.get("split_manifest_sha256")).lower() != OPEN_POLYMER_LEGACY_SPLIT_SHA256
        or review.get("artifact_id") != OPEN_POLYMER_LEGACY_REVIEW_ARTIFACT_ID
        or _text(review.get("sha256")).lower() != OPEN_POLYMER_LEGACY_REVIEW_SHA256
        or adoption.get("retraining_performed") is not False
        or adoption.get("historical_atomic_holdout_claim_present") is not False
        or adoption.get("historical_ledger_fabricated") is not False
        or _text(adoption.get("source_solution_id")) != solution_id
        or _text(adoption.get("source_artifact_manifest_sha256")).lower() != manifest_sha
        or item.get("official_metric_claimed") is not False
        or item.get("protocol_scope") != "publicly_reproducible_internal_protocol"
    ):
        issues.append(GateIssue("OPEN_POLYMER_LEGACY_ADOPTION_BINDING_INVALID"))
    return issues


def _validate_artifact_contract(
    item: Mapping[str, Any], *, competition: str, solution_id: str, conditional_policy: bool = False
) -> list[GateIssue]:
    issues: list[GateIssue] = []
    manifest = _first_mapping(item, "artifact_manifest", "manifest")
    manifest_sha = _text(item.get("artifact_manifest_sha256")).lower()
    if not manifest or not _sha(manifest_sha):
        issues.append(GateIssue("ARTIFACT_MANIFEST_SHA_MISSING"))
    elif manifest_sha != sha256_json(manifest):
        issues.append(GateIssue("ARTIFACT_MANIFEST_HASH_MISMATCH"))
    if not SAFE_ID_RE.fullmatch(_text(item.get("artifact_manifest_artifact_id"))):
        issues.append(GateIssue("ARTIFACT_MANIFEST_ARTIFACT_ID_INVALID"))
    if (
        manifest.get("schema") != "evomind.artifact-manifest.v1"
        or _text(manifest.get("run_id")) != FIXED_RUN_ID
        or _text(manifest.get("competition")) != competition
        or _text(manifest.get("solution_id")) != solution_id
    ):
        issues.append(GateIssue("ARTIFACT_MANIFEST_BINDING_INVALID"))
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, list) or not files:
        issues.append(GateIssue("ARTIFACT_MANIFEST_FILES_MISSING"))
        files = []
    by_role: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    observed_total = 0
    for raw in files:
        if not isinstance(raw, Mapping):
            issues.append(GateIssue("ARTIFACT_MANIFEST_ENTRY_INVALID"))
            continue
        path = _text(raw.get("path"))
        role = _text(raw.get("role"))
        artifact_id = _text(raw.get("artifact_id"))
        byte_count = _count(raw.get("bytes"))
        valid = (
            _safe_manifest_path(path)
            and bool(role)
            and SAFE_ID_RE.fullmatch(artifact_id) is not None
            and byte_count is not None
            and _sha(raw.get("sha256"))
            and raw.get("regular_file") is True
            and raw.get("symlink") is False
        )
        if not valid:
            issues.append(GateIssue("ARTIFACT_MANIFEST_ENTRY_INVALID", path))
            continue
        if path in seen_paths or artifact_id in seen_ids or role in by_role:
            issues.append(GateIssue("ARTIFACT_MANIFEST_DUPLICATE_ENTRY", path or artifact_id or role))
            continue
        seen_paths.add(path)
        seen_ids.add(artifact_id)
        by_role[role] = raw
        observed_total += int(byte_count)
    if _count(manifest.get("file_count")) != len(files) or _count(manifest.get("total_bytes")) != observed_total:
        issues.append(GateIssue("ARTIFACT_MANIFEST_CLOSURE_MISMATCH"))
    required_roles = set(BASE_EVIDENCE_ARTIFACT_ROLES)
    baseline = _first_mapping(item, "human_baseline", "human_baseline_evidence")
    if conditional_policy:
        required_roles.add("primary_baseline_source")
        required_roles.add(
            "human_baseline_source" if baseline else "human_baseline_availability"
        )
    else:
        required_roles.add("human_baseline_source")
    if not required_roles <= set(by_role):
        issues.append(GateIssue("ARTIFACT_MANIFEST_REQUIRED_ROLES_MISSING"))
    for role, field_name in ARTIFACT_ROLE_FIELDS.items():
        declared = _text(item.get(field_name)).lower()
        entry = by_role.get(role, {})
        if not _sha(declared):
            issues.append(GateIssue(f"{field_name.upper()}_MISSING"))
        elif _text(entry.get("sha256")).lower() != declared:
            issues.append(GateIssue(f"{field_name.upper()}_NOT_TRACEABLE"))
    ledger = _first_mapping(item, "holdout_ledger")
    for role, field_name in (
        ("holdout_ledger", "ledger_sha256"),
        ("holdout_claim", "claim_sha256"),
    ):
        if _text(by_role.get(role, {}).get("sha256")).lower() != _text(ledger.get(field_name)).lower():
            issues.append(GateIssue(f"{role.upper()}_NOT_TRACEABLE"))
    if baseline:
        source_entry = by_role.get("human_baseline_source", {})
        if (
            _text(source_entry.get("artifact_id")) != _text(baseline.get("evidence_artifact_id"))
            or _text(source_entry.get("sha256")).lower()
            != _text(baseline.get("evidence_artifact_sha256")).lower()
        ):
            issues.append(GateIssue("HUMAN_BASELINE_SOURCE_NOT_TRACEABLE"))
    elif conditional_policy:
        availability = _first_mapping(item, "human_baseline_availability")
        source_entry = by_role.get("human_baseline_availability", {})
        if (
            _text(source_entry.get("artifact_id"))
            != _text(availability.get("evidence_artifact_id"))
            or _text(source_entry.get("sha256")).lower()
            != _text(availability.get("evidence_artifact_sha256")).lower()
        ):
            issues.append(GateIssue("HUMAN_BASELINE_AVAILABILITY_NOT_TRACEABLE"))
    if conditional_policy:
        primary = _first_mapping(item, "primary_baseline", "strong_baseline")
        source_entry = by_role.get("primary_baseline_source", {})
        if (
            _text(source_entry.get("artifact_id")) != _text(primary.get("evidence_artifact_id"))
            or _text(source_entry.get("sha256")).lower()
            != _text(primary.get("evidence_artifact_sha256")).lower()
        ):
            issues.append(GateIssue("PRIMARY_BASELINE_SOURCE_NOT_TRACEABLE"))
    return issues


def _validate_leakage_contract(item: Mapping[str, Any]) -> list[GateIssue]:
    leakage = _first_mapping(item, "leakage_audit", "independent_verification", "verification")
    issues: list[GateIssue] = []
    if leakage.get("test_labels_used") is not False or leakage.get("hidden_test_labels_used") is not False:
        issues.append(GateIssue("TEST_LABEL_USAGE_NOT_FALSE"))
    overlap_values = (
        leakage.get("holdout_overlap"),
        leakage.get("id_overlap"),
        leakage.get("architecture_overlap"),
        leakage.get("point_overlap"),
    )
    if any(value is None for value in overlap_values):
        issues.append(GateIssue("HOLDOUT_OVERLAP_EVIDENCE_MISSING"))
    elif any(value != 0 for value in overlap_values):
        issues.append(GateIssue("HOLDOUT_OVERLAP_NONZERO"))
    if leakage.get("manifest_closed") is not True:
        issues.append(GateIssue("MANIFEST_CLOSURE_NOT_PROVEN"))
    if leakage.get("loader_smoke") not in (True, "passed", "PASS"):
        issues.append(GateIssue("LOADER_SMOKE_NOT_PROVEN"))
    return issues


def _validate_side_effect_contract(item: Mapping[str, Any]) -> list[GateIssue]:
    side = _first_mapping(item, "side_effects", "execution")
    issues: list[GateIssue] = []
    if side.get("other_processes_modified") is not False:
        issues.append(GateIssue("OTHER_PROCESSES_MODIFIED_NOT_FALSE"))
    if side.get("signals_sent") not in (0, "0"):
        issues.append(GateIssue("UNEXPECTED_SIGNALS"))
    return issues


def validate_goal_item(
    competition: str,
    item: Mapping[str, Any],
    *,
    conditional_policy: bool = False,
) -> GoalItemResult:
    declared = _text(item.get("goal_status"))
    issues: list[GateIssue] = []
    if _text(item.get("competition")) and _text(item.get("competition")) != competition:
        issues.append(GateIssue("COMPETITION_ID_MISMATCH"))
    if declared not in VALID_STATUSES:
        issues.append(GateIssue("GOAL_STATUS_INVALID"))
    solution_id = _text(item.get("solution_id"))
    if not SAFE_ID_RE.fullmatch(solution_id):
        issues.append(GateIssue("SOLUTION_ID_INVALID"))
    legacy_open_polymer = False
    if competition == "open_polymer":
        adoption = _first_mapping(item, "existing_candidate_adoption")
        legacy_open_polymer = adoption.get("schema") == OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA
        if adoption.get("schema") not in {
            "evomind.existing-candidate-adoption.v1",
            OPEN_POLYMER_LEGACY_ADOPTION_SCHEMA,
        }:
            issues.append(GateIssue("OPEN_POLYMER_ADOPTION_SCHEMA_INVALID"))
        if adoption.get("retraining_performed") is not False:
            issues.append(GateIssue("OPEN_POLYMER_RETRAINING_NOT_FALSE"))
        if not SAFE_ID_RE.fullmatch(_text(adoption.get("source_tool_call_id"))):
            issues.append(GateIssue("OPEN_POLYMER_SOURCE_TOOL_CALL_ID_INVALID"))
        if _text(adoption.get("source_solution_id")) != solution_id:
            issues.append(GateIssue("OPEN_POLYMER_SOURCE_SOLUTION_MISMATCH"))
        if (
            not _sha(adoption.get("source_artifact_manifest_sha256"))
            or _text(adoption.get("source_artifact_manifest_sha256")).lower()
            != _text(item.get("artifact_manifest_sha256")).lower()
        ):
            issues.append(GateIssue("OPEN_POLYMER_SOURCE_MANIFEST_MISMATCH"))
    else:
        execution = _first_mapping(item, "managed_execution")
        if execution.get("schema") != "evomind.managed-candidate-execution.v1":
            issues.append(GateIssue("MANAGED_EXECUTION_SCHEMA_INVALID"))
        for field_name in (
            "training_route_tool_call_id",
            "hpc_verify_tool_call_id",
            "hpc_execute_tool_call_id",
        ):
            if not SAFE_ID_RE.fullmatch(_text(execution.get(field_name))):
                issues.append(GateIssue(f"{field_name.upper()}_INVALID"))
    if not legacy_open_polymer:
        review = _first_mapping(item, "independent_review")
        if (
            review.get("schema") != "evomind.independent-review.v1"
            or not SAFE_ID_RE.fullmatch(_text(review.get("artifact_id")))
            or not _sha(review.get("sha256"))
            or review.get("verified") is not True
        ):
            issues.append(GateIssue("INDEPENDENT_REVIEW_EVIDENCE_INVALID"))
    direction = _text(item.get("metric_direction"))
    metric = _text(item.get("official_metric"))
    if direction not in VALID_DIRECTIONS:
        issues.append(GateIssue("OFFICIAL_METRIC_DIRECTION_MISSING"))
    if not metric:
        issues.append(GateIssue("OFFICIAL_METRIC_MISSING"))
    baseline_value = item.get("human_baseline")
    if not isinstance(baseline_value, Mapping):
        baseline_value = item.get("human_baseline_evidence")
    baseline = HumanBaselineResult(False)
    primary = PrimaryBaselineResult(False)
    human_metric_gate = MetricGateResult(False)
    human_required = not conditional_policy
    if conditional_policy:
        primary_value = item.get("primary_baseline")
        if not isinstance(primary_value, Mapping):
            primary_value = item.get("strong_baseline")
        primary = validate_primary_baseline(
            primary_value,
            expected_metric=metric,
            expected_direction=direction,
        )
        primary_metric_data = _first_mapping(
            item, "primary_comparison", "baseline_comparison", "metric_gate"
        )
        metric_gate = compare_metric_to_primary(
            candidate_metric=primary_metric_data.get("candidate_metric"),
            baseline_value=primary.value,
            ci95_lower=primary_metric_data.get("ci95_lower"),
            ci95_upper=primary_metric_data.get("ci95_upper"),
            direction=direction,
        )
        human_required, availability_issues = _human_baseline_requirement(item, baseline_value)
        issues.extend(primary.issues)
        issues.extend(metric_gate.issues)
        issues.extend(availability_issues)
        if human_required:
            baseline = validate_human_baseline(
                baseline_value,
                expected_metric=metric,
                expected_direction=direction,
            )
            metric_data = _first_mapping(item, "human_comparison", "human_gate")
            human_metric_gate = compare_metric_to_human(
                candidate_metric=metric_data.get("candidate_metric"),
                human_mean=baseline.mean,
                ci95_lower=metric_data.get("ci95_lower"),
                ci95_upper=metric_data.get("ci95_upper"),
                direction=direction,
            )
            issues.extend(baseline.issues)
            issues.extend(human_metric_gate.issues)
        else:
            human_metric_gate = MetricGateResult(True)
    else:
        baseline = validate_human_baseline(
            baseline_value,
            expected_metric=metric,
            expected_direction=direction,
        )
        metric_data = _first_mapping(item, "human_comparison", "human_gate", "metric_gate")
        metric_gate = compare_metric_to_human(
            candidate_metric=metric_data.get("candidate_metric"),
            human_mean=baseline.mean,
            ci95_lower=metric_data.get("ci95_lower"),
            ci95_upper=metric_data.get("ci95_upper"),
            direction=direction,
        )
        human_metric_gate = metric_gate
        issues.extend(baseline.issues)
        issues.extend(metric_gate.issues)
    if legacy_open_polymer:
        issues.extend(
            _validate_open_polymer_legacy_artifact_contract(
                item,
                solution_id=solution_id,
            )
        )
    else:
        issues.extend(_validate_holdout_contract(item))
        issues.extend(
            _validate_artifact_contract(
                item,
                competition=competition,
                solution_id=solution_id,
                conditional_policy=conditional_policy,
            )
        )
    issues.extend(_validate_leakage_contract(item))
    issues.extend(_validate_side_effect_contract(item))
    if declared == "VERIFIED_COMPLETE" and issues:
        issues.append(GateIssue("DECLARED_COMPLETE_CONTRACT_FAILED"))
    return GoalItemResult(
        competition=competition,
        declared_status=declared,
        strict_verified_complete=declared == "VERIFIED_COMPLETE" and not issues,
        issues=tuple(dict.fromkeys(issues)),
        human_baseline=baseline,
        primary_baseline=primary,
        metric_gate=metric_gate,
        human_metric_gate=human_metric_gate,
        human_baseline_required=human_required,
    )


def validate_goal_board(
    board: Mapping[str, Any],
    *,
    run_id: str = FIXED_RUN_ID,
    allocation: str = FIXED_ALLOCATION,
    spec: Mapping[str, Any] | None = None,
) -> GoalBoardResult:
    """Validate the complete fixed five-competition board without mutation."""

    issues: list[GateIssue] = []
    board_policy = _text(board.get("baseline_policy")) if isinstance(board, Mapping) else ""
    is_conditional = conditional_baseline_policy(spec) or board_policy == CONDITIONAL_BASELINE_POLICY_MODE
    if is_conditional:
        if board_policy != CONDITIONAL_BASELINE_POLICY_MODE:
            issues.append(GateIssue("GOAL_BOARD_BASELINE_POLICY_MISMATCH"))
        policy_evidence_sha = _text(board.get("policy_evidence_sha256")).lower()
        if not _sha(policy_evidence_sha):
            issues.append(GateIssue("GOAL_BOARD_POLICY_EVIDENCE_SHA_INVALID"))
        if isinstance(spec, Mapping) and policy_evidence_sha != _text(
            spec.get("policy_evidence_sha256")
        ).lower():
            issues.append(GateIssue("GOAL_BOARD_POLICY_EVIDENCE_SHA_MISMATCH"))
    if not isinstance(board, Mapping) or board.get("schema") != GOAL_BOARD_SCHEMA:
        issues.append(GateIssue("GOAL_BOARD_SCHEMA_INVALID"))
    board_run = _text(board.get("run_id") or board.get("fixed_run_id")) if isinstance(board, Mapping) else ""
    board_allocation = _text(board.get("allocation") or board.get("g21")) if isinstance(board, Mapping) else ""
    if board_run != run_id:
        issues.append(GateIssue("GOAL_BOARD_RUN_MISMATCH"))
    if board_allocation.casefold() != allocation.casefold():
        issues.append(GateIssue("GOAL_BOARD_ALLOCATION_MISMATCH"))
    rows = board.get("competitions") if isinstance(board, Mapping) else None
    if not isinstance(rows, list):
        rows = []
        issues.append(GateIssue("GOAL_BOARD_COMPETITIONS_MISSING"))
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            issues.append(GateIssue("GOAL_BOARD_ITEM_INVALID"))
            continue
        name = _text(raw.get("competition"))
        if name in by_name:
            issues.append(GateIssue("GOAL_BOARD_DUPLICATE_COMPETITION", name))
        by_name[name] = raw
    if set(by_name) != set(COMPETITIONS):
        issues.append(GateIssue("GOAL_BOARD_SCOPE_MISMATCH"))
    items: list[GoalItemResult] = []
    for competition in COMPETITIONS:
        item = by_name.get(competition, {})
        items.append(validate_goal_item(competition, item, conditional_policy=is_conditional))
    complete_count = sum(item.strict_verified_complete for item in items)
    target_count = len(COMPETITIONS)
    if isinstance(board, Mapping) and board.get("target_count") not in (None, target_count):
        issues.append(GateIssue("GOAL_BOARD_TARGET_COUNT_MISMATCH"))
    if isinstance(board, Mapping) and board.get("weather_actions") not in (None, 0):
        issues.append(GateIssue("WEATHER_ACTIONS_NONZERO"))
    completed_items = [item for item in items if item.declared_status == "VERIFIED_COMPLETE"]
    if completed_items and (
        not is_conditional or any(item.human_baseline_required for item in completed_items)
    ):
        if not SAFE_ID_RE.fullmatch(_text(board.get("human_baseline_evidence_artifact_id"))):
            issues.append(GateIssue("HUMAN_BASELINE_EVIDENCE_ARTIFACT_ID_INVALID"))
        if not _sha(board.get("human_baseline_evidence_sha256")):
            issues.append(GateIssue("HUMAN_BASELINE_EVIDENCE_SHA_INVALID"))
    # A board is structurally valid only when every item is contract-valid.
    for item in items:
        if item.issues:
            issues.extend(GateIssue(f"{item.competition}:{issue.code}", issue.detail) for issue in item.issues)
    return GoalBoardResult(
        valid=not issues,
        run_id=run_id,
        allocation=allocation,
        complete_count=complete_count,
        target_count=target_count,
        issues=tuple(dict.fromkeys(issues)),
        items=tuple(items),
    )


def validate_goal_board_identity(
    board: Any,
    *,
    run_id: str = FIXED_RUN_ID,
    allocation: str = FIXED_ALLOCATION,
) -> tuple[bool, tuple[GateIssue, ...]]:
    """Validate board binding/scope without requiring completion evidence."""

    issues: list[GateIssue] = []
    if not isinstance(board, Mapping) or board.get("schema") != GOAL_BOARD_SCHEMA:
        issues.append(GateIssue("GOAL_BOARD_SCHEMA_INVALID"))
    board_run = _text(board.get("run_id") or board.get("fixed_run_id")) if isinstance(board, Mapping) else ""
    board_allocation = _text(board.get("allocation") or board.get("g21")) if isinstance(board, Mapping) else ""
    if board_run != run_id:
        issues.append(GateIssue("GOAL_BOARD_RUN_MISMATCH"))
    if board_allocation.casefold() != allocation.casefold():
        issues.append(GateIssue("GOAL_BOARD_ALLOCATION_MISMATCH"))
    rows = board.get("competitions") if isinstance(board, Mapping) else None
    if not isinstance(rows, list):
        issues.append(GateIssue("GOAL_BOARD_COMPETITIONS_MISSING"))
        rows = []
    names: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            issues.append(GateIssue("GOAL_BOARD_ITEM_INVALID"))
            continue
        names.append(_text(raw.get("competition")))
    if len(names) != len(set(names)):
        issues.append(GateIssue("GOAL_BOARD_DUPLICATE_COMPETITION"))
    if set(names) != set(COMPETITIONS):
        issues.append(GateIssue("GOAL_BOARD_SCOPE_MISMATCH"))
    if isinstance(board, Mapping) and board.get("target_count") not in (None, len(COMPETITIONS)):
        issues.append(GateIssue("GOAL_BOARD_TARGET_COUNT_MISMATCH"))
    if isinstance(board, Mapping) and board.get("weather_actions") not in (None, 0):
        issues.append(GateIssue("WEATHER_ACTIONS_NONZERO"))
    return not issues, tuple(dict.fromkeys(issues))


def validate_goal_record(
    value: Any,
    *,
    run_id: str = FIXED_RUN_ID,
    allocation: str = FIXED_ALLOCATION,
    goal_key: str = FIXED_GOAL_KEY,
    goal_id: str = FIXED_GOAL_ID,
) -> GoalRecordResult:
    """Validate a persisted Goal identity and its hash-bound spec/board."""

    if not isinstance(value, Mapping):
        return GoalRecordResult(False, issues=(GateIssue("GOAL_RECORD_INVALID"),))
    issues: list[GateIssue] = []
    actual_id = _text(value.get("goal_id") or value.get("id"))
    actual_run = _text(value.get("run_id"))
    actual_allocation = _text(value.get("allocation"))
    actual_key = _text(value.get("goal_key"))
    status = _text(value.get("status")).casefold()
    if _text(value.get("schema")) != GOAL_RECORD_SCHEMA:
        issues.append(GateIssue("GOAL_RECORD_SCHEMA_INVALID"))
    if actual_id != goal_id or not SAFE_ID_RE.fullmatch(actual_id):
        issues.append(GateIssue("GOAL_RECORD_ID_INVALID"))
    if actual_run != run_id:
        issues.append(GateIssue("GOAL_RECORD_RUN_MISMATCH"))
    if actual_allocation.casefold() != allocation.casefold():
        issues.append(GateIssue("GOAL_RECORD_ALLOCATION_MISMATCH"))
    if actual_key != goal_key or not SAFE_ID_RE.fullmatch(actual_key):
        issues.append(GateIssue("GOAL_RECORD_KEY_INVALID"))
    if status not in VALID_GOAL_RECORD_STATUSES:
        issues.append(GateIssue("GOAL_RECORD_STATUS_INVALID"))
    spec = value.get("spec")
    spec_hash = _text(value.get("spec_sha256")).lower()
    if not isinstance(spec, Mapping):
        issues.append(GateIssue("GOAL_RECORD_SPEC_MISSING"))
    else:
        spec_ok, spec_issues = validate_goal_spec(spec)
        issues.extend(GateIssue(f"spec:{item.code}", item.detail) for item in spec_issues)
        if not _sha(spec_hash) or spec_hash != sha256_json(spec):
            issues.append(GateIssue("GOAL_RECORD_SPEC_HASH_MISMATCH"))
        if not spec_ok:
            # Keep the explicit prefixed issues above; this branch documents
            # that an invalid spec is never rescued by a matching hash.
            pass
    board = value.get("board")
    board_hash = _text(value.get("board_sha256")).lower()
    if not isinstance(board, Mapping):
        issues.append(GateIssue("GOAL_RECORD_BOARD_MISSING"))
    else:
        _, board_issues = validate_goal_board_identity(board, run_id=run_id, allocation=allocation)
        issues.extend(GateIssue(f"board:{item.code}", item.detail) for item in board_issues)
        if not _sha(board_hash) or board_hash != sha256_json(board):
            issues.append(GateIssue("GOAL_RECORD_BOARD_HASH_MISMATCH"))
        if status == "completed":
            strict_board = validate_goal_board(
                board,
                run_id=run_id,
                allocation=allocation,
                spec=spec if isinstance(spec, Mapping) else None,
            )
            if not strict_board.completion_proven:
                issues.append(GateIssue("GOAL_RECORD_COMPLETION_UNPROVEN"))
    human_baseline_hash = _text(value.get("human_baseline_sha256")).lower()
    if not _sha(human_baseline_hash):
        issues.append(GateIssue("GOAL_RECORD_HUMAN_BASELINE_SHA_INVALID"))
    bootstrap_human_baseline_hash = _text(
        spec.get("human_baseline_evidence_sha256") if isinstance(spec, Mapping) else ""
    ).lower()
    board_evidence_sha = _text(
        board.get("human_baseline_evidence_sha256") if isinstance(board, Mapping) else ""
    ).lower()
    if _sha(human_baseline_hash) and human_baseline_hash != bootstrap_human_baseline_hash:
        if board_evidence_sha != human_baseline_hash:
            issues.append(GateIssue("GOAL_RECORD_HUMAN_BASELINE_BOARD_MISMATCH"))
    if status == "completed":
        if board_evidence_sha != human_baseline_hash:
            issues.append(GateIssue("GOAL_RECORD_HUMAN_BASELINE_BOARD_MISMATCH"))
        if (
            human_baseline_hash == INITIAL_NEGATIVE_HUMAN_BASELINE_EVIDENCE_SHA256
            and not conditional_baseline_policy(spec if isinstance(spec, Mapping) else None)
        ):
            issues.append(GateIssue("GOAL_RECORD_HUMAN_BASELINE_NEGATIVE_EVIDENCE"))
    return GoalRecordResult(
        not issues,
        actual_id,
        actual_run,
        actual_allocation,
        actual_key,
        status,
        spec_hash,
        board_hash,
        human_baseline_hash,
        tuple(dict.fromkeys(issues)),
    )


def validate_run_preflight(
    snapshot: Mapping[str, Any], *, run_id: str = FIXED_RUN_ID
) -> tuple[bool, tuple[GateIssue, ...]]:
    """Check the non-mutating preconditions required before a new GPU action."""

    issues: list[GateIssue] = []
    if _text(snapshot.get("id") or snapshot.get("run_id")) != run_id:
        issues.append(GateIssue("RUN_ID_MISMATCH"))
    if "effective_active_count" in snapshot:
        active = _count(snapshot.get("effective_active_count"))
    elif "active_tool_calls" in snapshot:
        raw_active = snapshot.get("active_tool_calls")
        active = len(raw_active) if isinstance(raw_active, (list, tuple)) else None
    else:
        active = None
    if active is None or active != 0:
        issues.append(GateIssue("EFFECTIVE_ACTIVE_NONZERO"))
    if "pending_approval_count" in snapshot:
        pending = _count(snapshot.get("pending_approval_count"))
    elif "approvals" in snapshot:
        approvals = snapshot.get("approvals")
        pending = (
            sum(1 for item in approvals if isinstance(item, Mapping) and item.get("status") == "pending")
            if isinstance(approvals, (list, tuple))
            else None
        )
    else:
        pending = None
    if pending is None or pending != 0:
        issues.append(GateIssue("PENDING_APPROVAL_NONZERO"))
    return not issues, tuple(issues)


def validate_goal_spec(spec: Any) -> tuple[bool, tuple[GateIssue, ...]]:
    """Validate the immutable, non-secret G21 goal specification."""

    issues: list[GateIssue] = []
    if not isinstance(spec, Mapping):
        return False, (GateIssue("GOAL_SPEC_INVALID"),)
    schema = _text(spec.get("schema"))
    if schema not in {LEGACY_GOAL_CONTRACT_SCHEMA, GOAL_CONTRACT_SCHEMA}:
        issues.append(GateIssue("GOAL_SPEC_SCHEMA_INVALID"))
    if schema == GOAL_CONTRACT_SCHEMA:
        if _text(spec.get("previous_goal_spec_file_sha256")).lower() != INITIAL_GOAL_SPEC_FILE_SHA256:
            issues.append(GateIssue("GOAL_SPEC_PREVIOUS_FILE_SHA_MISMATCH"))
        if (
            _text(spec.get("previous_goal_spec_canonical_sha256")).lower()
            != INITIAL_GOAL_SPEC_CANONICAL_SHA256
        ):
            issues.append(GateIssue("GOAL_SPEC_PREVIOUS_CANONICAL_SHA_MISMATCH"))
        if _text(spec.get("policy_migration_id")) != CONDITIONAL_BASELINE_MIGRATION_ID:
            issues.append(GateIssue("GOAL_SPEC_POLICY_MIGRATION_ID_INVALID"))
        if _text(spec.get("policy_evidence_path")) != (
            "configs/g21_conditional_strong_baseline_policy_evidence.json"
        ):
            issues.append(GateIssue("GOAL_SPEC_POLICY_EVIDENCE_PATH_INVALID"))
        if (
            _text(spec.get("policy_evidence_sha256")).lower()
            != CONDITIONAL_BASELINE_POLICY_EVIDENCE_SHA256
        ):
            issues.append(GateIssue("GOAL_SPEC_POLICY_EVIDENCE_SHA_INVALID"))
        if (
            _text(spec.get("policy_evidence_file_sha256")).lower()
            != CONDITIONAL_BASELINE_POLICY_EVIDENCE_FILE_SHA256
        ):
            issues.append(GateIssue("GOAL_SPEC_POLICY_EVIDENCE_FILE_SHA_INVALID"))
    if _text(spec.get("goal_id")) != FIXED_GOAL_ID:
        issues.append(GateIssue("GOAL_SPEC_ID_MISMATCH"))
    if _text(spec.get("goal_key")) != FIXED_GOAL_KEY:
        issues.append(GateIssue("GOAL_SPEC_KEY_MISMATCH"))
    if not _sha(spec.get("human_baseline_evidence_sha256")):
        issues.append(GateIssue("GOAL_SPEC_HUMAN_BASELINE_SHA_INVALID"))
    if _text(spec.get("run_id")) != FIXED_RUN_ID:
        issues.append(GateIssue("GOAL_SPEC_RUN_MISMATCH"))
    if _text(spec.get("allocation")).casefold() != FIXED_ALLOCATION.casefold():
        issues.append(GateIssue("GOAL_SPEC_ALLOCATION_MISMATCH"))
    competitions = tuple(str(item) for item in spec.get("competitions") or ())
    if competitions != COMPETITIONS:
        issues.append(GateIssue("GOAL_SPEC_COMPETITION_SET_INVALID"))
    execution = spec.get("execution") if isinstance(spec.get("execution"), Mapping) else {}
    if execution.get("mode") != "single_gpu_serial":
        issues.append(GateIssue("GOAL_SPEC_EXECUTION_MODE_INVALID"))
    if execution.get("max_concurrent_solutions") != 1:
        issues.append(GateIssue("GOAL_SPEC_CONCURRENCY_INVALID"))
    if execution.get("hpc_identity_samples") != 5:
        issues.append(GateIssue("GOAL_SPEC_HPC_SAMPLE_COUNT_INVALID"))
    sequence = tuple(str(item) for item in execution.get("call_sequence") or ())
    if sequence != ("training_route", "hpc_verify", "hpc_execute_solution", "artifact_collect", "independent_verify"):
        issues.append(GateIssue("GOAL_SPEC_CALL_SEQUENCE_INVALID"))
    metric_gate = spec.get("metric_gate") if isinstance(spec.get("metric_gate"), Mapping) else {}
    if schema == LEGACY_GOAL_CONTRACT_SCHEMA:
        if metric_gate.get("human_baseline_required") is not True:
            issues.append(GateIssue("GOAL_SPEC_HUMAN_BASELINE_REQUIRED_FALSE"))
    elif schema == GOAL_CONTRACT_SCHEMA:
        if "human_baseline_required" in metric_gate:
            issues.append(GateIssue("GOAL_SPEC_UNCONDITIONAL_HUMAN_FLAG_FORBIDDEN"))
        if metric_gate.get("mode") != CONDITIONAL_BASELINE_POLICY_MODE:
            issues.append(GateIssue("GOAL_SPEC_BASELINE_POLICY_MODE_INVALID"))
        if metric_gate.get("primary_baseline_required") is not True:
            issues.append(GateIssue("GOAL_SPEC_PRIMARY_BASELINE_REQUIRED_FALSE"))
        if tuple(metric_gate.get("accepted_primary_baseline_types") or ()) != tuple(
            sorted(PRIMARY_BASELINE_TYPES)
        ):
            issues.append(GateIssue("GOAL_SPEC_PRIMARY_BASELINE_TYPES_INVALID"))
        if metric_gate.get("primary_baseline_source_required") is not True:
            issues.append(GateIssue("GOAL_SPEC_PRIMARY_BASELINE_SOURCE_REQUIRED_FALSE"))
        if metric_gate.get("human_baseline_required_when_verified_available") is not True:
            issues.append(GateIssue("GOAL_SPEC_CONDITIONAL_HUMAN_BASELINE_REQUIRED_FALSE"))
        if metric_gate.get("human_baseline_source_required_when_claimed") is not True:
            issues.append(GateIssue("GOAL_SPEC_HUMAN_BASELINE_SOURCE_REQUIRED_FALSE"))
    if metric_gate.get("point_estimate_must_improve") is not True:
        issues.append(GateIssue("GOAL_SPEC_POINT_ESTIMATE_REQUIRED_FALSE"))
    if metric_gate.get("confidence_interval_lower_bound_must_improve") is not True:
        issues.append(GateIssue("GOAL_SPEC_CI_BOUND_REQUIRED_FALSE"))
    forbidden = set(str(item) for item in spec.get("forbidden") or ())
    required_forbidden = {
        "weather4cast",
        "test_labels",
        "replacement_run",
        "online_dependency_install",
        "online_model_download",
        "other_gpu_process_modification",
        "competition_data_prepare",
        "competition_data_accelerate",
        "kaggle_download",
        "kaggle_submit",
        "join_competition",
        "accept_terms",
    }
    if not required_forbidden <= forbidden:
        issues.append(GateIssue("GOAL_SPEC_FORBIDDEN_SCOPE_INCOMPLETE"))
    required_artifacts = {
        "command_sha256",
        "environment_sha256",
        "data_manifest_sha256",
        "source_code_sha256",
        "model_sha256",
        "metrics_sha256",
        "rollback_sha256",
        "artifact_manifest_sha256",
    }
    artifact_requirements = set(str(item) for item in spec.get("artifact_requirements") or ())
    if not required_artifacts <= artifact_requirements:
        issues.append(GateIssue("GOAL_SPEC_ARTIFACT_REQUIREMENTS_INCOMPLETE"))
    persistence = spec.get("persistence") if isinstance(spec.get("persistence"), Mapping) else {}
    if persistence.get("mode") != "thread_goal" or persistence.get("heartbeat_enabled") is not False:
        issues.append(GateIssue("GOAL_SPEC_PERSISTENCE_INVALID"))
    external_gates = spec.get("external_gates")
    if not isinstance(external_gates, Mapping) or set(external_gates) != set(COMPETITIONS):
        issues.append(GateIssue("GOAL_SPEC_EXTERNAL_GATES_INCOMPLETE"))
    return not issues, tuple(dict.fromkeys(issues))


def load_goal_spec(path: str | None = None) -> dict[str, Any]:
    """Load a goal spec for local tooling; callers decide whether to persist it."""

    candidate = path or DEFAULT_GOAL_SPEC_RELATIVE
    with open(candidate, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    valid, issues = validate_goal_spec(value)
    if not valid:
        raise ValueError("invalid G21 goal specification: " + ",".join(item.code for item in issues))
    return dict(value)


def validate_serial_action_sequence(
    calls: Sequence[Mapping[str, Any]],
    *,
    run_id: str = FIXED_RUN_ID,
    competition: str = "",
    solution_id: str = "",
) -> tuple[bool, tuple[GateIssue, ...]]:
    """Validate one candidate's managed call sequence and retry identity."""

    issues: list[GateIssue] = []
    if competition in PROHIBITED_COMPETITIONS:
        issues.append(GateIssue("PROHIBITED_COMPETITION"))
    if competition and competition not in COMPETITIONS:
        issues.append(GateIssue("COMPETITION_OUT_OF_SCOPE"))
    if solution_id and not SAFE_ID_RE.fullmatch(solution_id):
        issues.append(GateIssue("SOLUTION_ID_INVALID"))
    seen_keys: set[str] = set()
    seen_solution = False
    solution_ids: set[str] = set()
    order = [str(item.get("tool_name") or item.get("tool")) for item in calls]
    for item in calls:
        if _text(item.get("run_id") or item.get("session_id")) != run_id:
            issues.append(GateIssue("CALL_RUN_MISMATCH"))
        tool = _text(item.get("tool_name") or item.get("tool"))
        if tool.casefold() in PROHIBITED_TOOLS:
            issues.append(GateIssue("PROHIBITED_TOOL", tool))
        if tool in {"artifact_collect", "independent_verify"}:
            issues.append(GateIssue("EVIDENCE_STAGE_IS_NOT_A_TOOL", tool))
        status = _text(item.get("status")).casefold()
        if status in {"running", "waiting_approval", "pending"}:
            issues.append(GateIssue("UNSETTLED_CALL", tool))
        key = _text(item.get("idempotency_key"))
        if not key or key in seen_keys:
            issues.append(GateIssue("IDEMPOTENCY_KEY_MISSING_OR_DUPLICATE"))
        seen_keys.add(key)
        if _text(item.get("solution_id")) == solution_id and tool in {"hpc_execute_solution", "training_route"}:
            if seen_solution and tool == "hpc_execute_solution":
                issues.append(GateIssue("SOLUTION_REPEATED"))
            if tool == "hpc_execute_solution":
                seen_solution = True
        candidate_id = _text(item.get("solution_id"))
        if candidate_id:
            solution_ids.add(candidate_id)
    if len(solution_ids) > 1:
        issues.append(GateIssue("MULTIPLE_SOLUTIONS_IN_SEQUENCE"))
    expected = ["training_route", "hpc_verify", "hpc_execute_solution"]
    filtered = [tool for tool in order if tool in set(expected)]
    if filtered != expected:
        issues.append(GateIssue("MANAGED_CALL_ORDER_INVALID"))
    return not issues, tuple(dict.fromkeys(issues))


__all__ = [
    "ALLOCATION",
    "COMPETITIONS",
    "FIXED_ALLOCATION",
    "FIXED_RUN_ID",
    "GOAL_BOARD_SCHEMA",
    "GOAL_CONTRACT_SCHEMA",
    "LEGACY_GOAL_CONTRACT_SCHEMA",
    "GOAL_RECORD_SCHEMA",
    "FIXED_GOAL_KEY",
    "FIXED_GOAL_ID",
    "INITIAL_NEGATIVE_HUMAN_BASELINE_EVIDENCE_SHA256",
    "INITIAL_GOAL_SPEC_FILE_SHA256",
    "INITIAL_GOAL_SPEC_CANONICAL_SHA256",
    "CONDITIONAL_BASELINE_POLICY_MODE",
    "CONDITIONAL_BASELINE_MIGRATION_ID",
    "DEFAULT_GOAL_SPEC_RELATIVE",
    "GoalBoardResult",
    "GoalRecordResult",
    "GoalItemResult",
    "HumanBaselineResult",
    "PrimaryBaselineResult",
    "MetricGateResult",
    "compare_metric_to_human",
    "compare_metric_to_primary",
    "conditional_baseline_policy",
    "canonical_json_bytes",
    "load_goal_spec",
    "sha256_bytes",
    "sha256_json",
    "validate_goal_board",
    "validate_goal_board_identity",
    "validate_goal_record",
    "validate_goal_item",
    "validate_goal_spec",
    "validate_human_baseline",
    "validate_primary_baseline",
    "validate_policy_migration_evidence",
    "validate_run_preflight",
    "validate_serial_action_sequence",
]
