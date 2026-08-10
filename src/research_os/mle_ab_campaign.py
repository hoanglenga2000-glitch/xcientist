"""Fail-closed, preregistered MLE-Bench Lite A/B campaign contracts.

This module deliberately separates inference-time search experiments from the
historical SIIM workflow.  It creates an immutable schedule, freezes every
candidate before a blind local grade, records one grade receipt per candidate,
and aggregates paired task/seed evidence without contacting Kaggle or any
private SIIM grader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
CAMPAIGN_SCHEMA = "evomind.mle_lite_ab.preregistration.v1"
SCHEDULE_SCHEMA = "evomind.mle_lite_ab.schedule.v1"
CAMPAIGN_LOCK_SCHEMA = "evomind.mle_lite_ab.campaign_lock.v1"
FREEZE_SCHEMA = "evomind.mle_lite_ab.candidate_freeze.v1"
GRADE_CLAIM_SCHEMA = "evomind.mle_lite_ab.local_grade_claim.v1"
GRADE_RECEIPT_SCHEMA = "evomind.mle_lite_ab.local_grade_receipt.v1"
AGGREGATE_SCHEMA = "evomind.mle_lite_ab.aggregate.v1"
SHADOW_SCHEMA = "evomind.mle_lite_ab.shadow_replay.v1"
INVALIDATION_SCHEMA = "evomind.mle_lite_ab.invalidation.v1"
INVALIDATION_LOG_NAME = "invalidation-events.jsonl"

LEGACY_ARM = "legacy_uct"
TREATMENT_ARM = "experience_mcgs_v1"
ARMS = (LEGACY_ARM, TREATMENT_ARM)
SEEDS = (42, 43, 44)
SCREEN_TASKS = (
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "spooky-author-identification",
    "jigsaw-toxic-comment-classification-challenge",
    "siim-isic-melanoma-classification",
    "leaf-classification",
)
UNIQUE_SIIM_RUN = "evomind_siim_isic_a800_job90353_20260730_095826"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_SAFE_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9._-]{7,95}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FORBIDDEN_RESULT_KEYS = {
    "private_feedback",
    "private_grader_feedback",
    "private_leaderboard",
    "kaggle_submission",
    "official_submission",
}


class CampaignContractError(RuntimeError):
    """Raised when immutable campaign evidence violates its contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignContractError(message)


def _exclusive_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CampaignContractError(f"immutable evidence already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write_json_exclusive(path: Path, payload: Any) -> Path:
    return _exclusive_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignContractError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignContractError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _safe_campaign_root(root: str | Path, namespace: str) -> Path:
    _require(bool(_SAFE_NAMESPACE.fullmatch(namespace)), "invalid campaign namespace")
    _require(namespace != UNIQUE_SIIM_RUN and "job90353" not in namespace.lower(), "campaign namespace aliases the frozen SIIM run")
    base = Path(root).expanduser().resolve()
    target = (base / namespace).resolve()
    _require(target.parent == base, "campaign path escapes campaign root")
    return target


def counterbalanced_order(task_id: str, seed: int) -> tuple[str, str]:
    _require(seed in SEEDS, f"unsupported seed: {seed}")
    digest = hashlib.sha256(f"{task_id}\0{seed}\0evomind-mle-ab-v1".encode("utf-8")).digest()
    return ARMS if digest[0] % 2 == 0 else (TREATMENT_ARM, LEGACY_ARM)


@dataclass(frozen=True)
class RunBudget:
    max_wall_seconds: int
    max_nodes: int
    max_total_tokens: int
    context_tokens: int = 32_000
    max_estimated_cost: float | None = None

    def validate(self) -> None:
        _require(self.max_wall_seconds > 0, "wall budget must be positive")
        _require(self.max_nodes > 0, "node budget must be positive")
        _require(self.max_total_tokens > 0, "token budget must be positive")
        _require(self.context_tokens == 32_000, "context must remain 32k")
        if self.max_estimated_cost is not None:
            _require(math.isfinite(self.max_estimated_cost) and self.max_estimated_cost > 0, "cost budget must be positive and finite")


@dataclass(frozen=True)
class CampaignConfig:
    namespace: str
    phase: str
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    model: str
    provider: str
    temperature: float
    budget: RunBudget
    source_hashes: Mapping[str, str]
    canonical_manifest_sha256: str
    created_at: str

    def validate(self) -> None:
        _require(bool(_SAFE_NAMESPACE.fullmatch(self.namespace)), "invalid campaign namespace")
        _require(self.namespace != UNIQUE_SIIM_RUN and "job90353" not in self.namespace.lower(), "SIIM run identity reuse is forbidden")
        _require(self.phase in {"screen", "formal"}, "phase must be screen or formal")
        expected_count = 6 if self.phase == "screen" else 22
        _require(len(self.tasks) == expected_count and len(set(self.tasks)) == expected_count, f"{self.phase} requires {expected_count} unique tasks")
        if self.phase == "screen":
            _require(self.tasks == SCREEN_TASKS, "screen task set or order drifted")
        _require(self.seeds == SEEDS, "seeds must be exactly 42,43,44")
        _require(self.model == "gpt-5.6-sol", "model drifted")
        _require(bool(self.provider.strip()), "provider must be explicit")
        _require(math.isclose(self.temperature, 0.4, abs_tol=1e-12), "temperature must be 0.4")
        self.budget.validate()
        _require(bool(_SHA256.fullmatch(self.canonical_manifest_sha256)), "canonical manifest hash is invalid")
        for name, digest in self.source_hashes.items():
            _require(bool(str(name).strip()) and bool(_SHA256.fullmatch(str(digest).lower())), "source hash record is invalid")

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "schema": CAMPAIGN_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "arms": list(ARMS),
                "counterbalance": "sha256(task_id + seed + fixed_salt) parity",
                "candidate_policy": "immutable_freeze_then_single_blind_local_grade",
                "kaggle_submission": "disabled",
                "siim_private_grader": "forbidden",
                "frozen_siim_run": UNIQUE_SIIM_RUN,
            }
        )
        return value


def build_config(
    *,
    namespace: str,
    phase: str,
    tasks: Sequence[str],
    provider: str,
    source_hashes: Mapping[str, str],
    canonical_manifest_sha256: str,
    created_at: str | None = None,
) -> CampaignConfig:
    budget = (
        RunBudget(max_wall_seconds=4 * 60 * 60, max_nodes=24, max_total_tokens=700_000)
        if phase == "screen"
        else RunBudget(max_wall_seconds=12 * 60 * 60, max_nodes=64, max_total_tokens=2_000_000)
    )
    config = CampaignConfig(
        namespace=namespace,
        phase=phase,
        tasks=tuple(tasks),
        seeds=SEEDS,
        model="gpt-5.6-sol",
        provider=provider,
        temperature=0.4,
        budget=budget,
        source_hashes={str(key): str(value).lower() for key, value in source_hashes.items()},
        canonical_manifest_sha256=canonical_manifest_sha256.lower(),
        created_at=created_at or utc_now(),
    )
    config.validate()
    return config


def run_id(namespace: str, phase: str, task_id: str, seed: int, arm: str) -> str:
    _require(arm in ARMS, f"invalid arm: {arm}")
    slug = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-")
    namespace_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8]
    task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
    arm_code = "base" if arm == LEGACY_ARM else "xmcgs"
    # Keep the evidence path below the legacy Windows MAX_PATH boundary even
    # when pytest or an enterprise profile contributes a long parent path.
    return f"mleab-{phase}-{namespace_hash}-{slug[:16]}-{task_hash}-s{seed}-{arm_code}"


def build_schedule(config: CampaignConfig) -> dict[str, Any]:
    config.validate()
    runs: list[dict[str, Any]] = []
    sequence = 0
    for task in config.tasks:
        for seed in config.seeds:
            for within_pair_order, arm in enumerate(counterbalanced_order(task, seed), start=1):
                sequence += 1
                runs.append(
                    {
                        "sequence": sequence,
                        "within_pair_order": within_pair_order,
                        "run_id": run_id(config.namespace, config.phase, task, seed, arm),
                        "task_id": task,
                        "seed": seed,
                        "arm": arm,
                        "search_mode": arm,
                        "model": config.model,
                        "provider": config.provider,
                        "temperature": config.temperature,
                        "budget": asdict(config.budget),
                        "status": "preregistered",
                    }
                )
    expected = len(config.tasks) * len(config.seeds) * len(ARMS)
    _require(len(runs) == expected and len({item["run_id"] for item in runs}) == expected, "schedule cardinality drifted")
    return {
        "schema": SCHEDULE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "namespace": config.namespace,
        "phase": config.phase,
        "expected_task_runs": expected,
        "runs": runs,
    }


def preregister(root: str | Path, config: CampaignConfig) -> dict[str, Any]:
    campaign = _safe_campaign_root(root, config.namespace)
    campaign.mkdir(parents=True, exist_ok=False)
    config_payload = config.payload()
    schedule = build_schedule(config)
    config_path = write_json_exclusive(campaign / "preregistration.json", config_payload)
    schedule["preregistration_sha256"] = sha256_file(config_path)
    schedule_path = write_json_exclusive(campaign / "schedule.json", schedule)
    lock_path = write_json_exclusive(
        campaign / "campaign-lock.json",
        {
            "schema": CAMPAIGN_LOCK_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "namespace": config.namespace,
            "preregistration_sha256": sha256_file(config_path),
            "schedule_sha256": sha256_file(schedule_path),
        },
    )
    return {
        "campaign_dir": str(campaign),
        "preregistration_path": str(config_path),
        "preregistration_sha256": sha256_file(config_path),
        "schedule_path": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "campaign_lock_path": str(lock_path),
        "campaign_lock_sha256": sha256_file(lock_path),
        "expected_task_runs": schedule["expected_task_runs"],
    }


def _read_invalidation_events(campaign: Path) -> list[dict[str, Any]]:
    """Read and verify the append-only campaign invalidation hash chain."""

    path = campaign / INVALIDATION_LOG_NAME
    if not path.exists():
        return []
    _require(path.is_file() and not path.is_symlink(), "campaign invalidation log is not a regular file")
    try:
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise CampaignContractError(f"invalid campaign invalidation log: {path}") from exc
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for expected_sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignContractError(f"invalid campaign invalidation log line {expected_sequence}") from exc
        _require(isinstance(event, dict), "campaign invalidation event must be an object")
        _require(event.get("schema") == INVALIDATION_SCHEMA, "campaign invalidation schema mismatch")
        _require(event.get("sequence") == expected_sequence, "campaign invalidation sequence drifted")
        _require(event.get("namespace") == campaign.name, "campaign invalidation namespace mismatch")
        _require(event.get("previous_event_sha256") == previous_hash, "campaign invalidation hash chain drifted")
        claimed_hash = str(event.get("event_sha256") or "").lower()
        _require(bool(_SHA256.fullmatch(claimed_hash)), "campaign invalidation event hash is invalid")
        hash_payload = dict(event)
        hash_payload.pop("event_sha256", None)
        _require(sha256_bytes(canonical_bytes(hash_payload)) == claimed_hash, "campaign invalidation event hash drifted")
        previous_hash = claimed_hash
        events.append(event)
    _require(bool(events), "campaign invalidation log is empty")
    return events


def _append_jsonl_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one bounded JSON record without rewriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_bytes(dict(payload))
    _require(len(line) <= 16 * 1024, "campaign invalidation event is unexpectedly large")
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # O_APPEND protects existing bytes.  A torn final line remains
        # detectable by _read_invalidation_events and therefore fails closed.
        raise


def load_campaign(
    campaign_dir: str | Path,
    *,
    verify_source_hashes: bool = True,
    allow_invalidated: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = Path(campaign_dir).expanduser().resolve()
    prereg = read_json(campaign / "preregistration.json", label="preregistration")
    schedule = read_json(campaign / "schedule.json", label="schedule")
    campaign_lock = read_json(campaign / "campaign-lock.json", label="campaign lock")
    _require(prereg.get("schema") == CAMPAIGN_SCHEMA, "preregistration schema mismatch")
    _require(schedule.get("schema") == SCHEDULE_SCHEMA, "schedule schema mismatch")
    _require(campaign_lock.get("schema") == CAMPAIGN_LOCK_SCHEMA, "campaign lock schema mismatch")
    _require(schedule.get("preregistration_sha256") == sha256_file(campaign / "preregistration.json"), "preregistration hash drifted")
    _require(prereg.get("namespace") == schedule.get("namespace") == campaign.name, "campaign namespace mismatch")
    _require(campaign_lock.get("namespace") == campaign.name, "campaign lock namespace mismatch")
    _require(campaign_lock.get("preregistration_sha256") == sha256_file(campaign / "preregistration.json"), "campaign lock preregistration hash drifted")
    _require(campaign_lock.get("schedule_sha256") == sha256_file(campaign / "schedule.json"), "campaign lock schedule hash drifted")
    _require(prereg.get("frozen_siim_run") == UNIQUE_SIIM_RUN, "frozen SIIM invariant missing")
    invalidation_events = _read_invalidation_events(campaign)
    if invalidation_events and not allow_invalidated:
        raise CampaignContractError(
            f"campaign invalidated by append-only event {invalidation_events[-1]['event_sha256']}"
        )
    source_hashes = prereg.get("source_hashes")
    _require(isinstance(source_hashes, dict) and bool(source_hashes), "frozen source hashes are missing")
    if verify_source_hashes:
        for relative_name, expected_digest in sorted(source_hashes.items()):
            source_path = (REPOSITORY_ROOT / str(relative_name)).resolve()
            _require(source_path == REPOSITORY_ROOT or REPOSITORY_ROOT in source_path.parents, "frozen source path escapes repository")
            _require(source_path.is_file() and not source_path.is_symlink(), f"frozen source is missing: {relative_name}")
            _require(sha256_file(source_path) == str(expected_digest).lower(), f"frozen source hash drifted: {relative_name}")
    return prereg, schedule


def invalidate_campaign(
    campaign_dir: str | Path,
    *,
    reason_code: str,
    reason: str,
    replacement_namespace: str | None = None,
    require_zero_task_runs: bool = True,
) -> dict[str, Any]:
    """Seal a preregistration as unusable without modifying frozen inputs.

    Invalidation deliberately skips *current* source-hash verification so a
    zero-run campaign can be closed after an implementation correction.  The
    immutable preregistration, schedule, and campaign lock are still verified,
    and their hashes are bound into the append-only event.
    """

    campaign = Path(campaign_dir).expanduser().resolve()
    prereg, _ = load_campaign(
        campaign,
        verify_source_hashes=False,
        allow_invalidated=True,
    )
    _require(not _read_invalidation_events(campaign), "campaign already invalidated")
    normalized_code = reason_code.strip().lower()
    _require(bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", normalized_code)), "invalid invalidation reason code")
    normalized_reason = reason.strip()
    _require(8 <= len(normalized_reason) <= 500, "invalidation reason must be 8..500 characters")
    if replacement_namespace is not None:
        replacement_namespace = replacement_namespace.strip()
        _require(bool(_SAFE_NAMESPACE.fullmatch(replacement_namespace)), "invalid replacement namespace")
        _require(replacement_namespace != campaign.name, "replacement namespace must be new")
        _require(
            replacement_namespace != UNIQUE_SIIM_RUN and "job90353" not in replacement_namespace.lower(),
            "replacement namespace aliases the frozen SIIM run",
        )

    run_root = campaign / "runs"
    run_directories = sorted(path.name for path in run_root.iterdir() if path.is_dir()) if run_root.is_dir() else []
    run_files = sorted(path for path in run_root.rglob("*") if path.is_file()) if run_root.is_dir() else []
    freeze_count = sum(path.name == "candidate_freeze.json" for path in run_files)
    claim_count = sum(path.name == "local_grade_claim.json" for path in run_files)
    receipt_count = sum(path.name == "local_grade_receipt.json" for path in run_files)
    zero_task_runs = not run_directories and not run_files
    if require_zero_task_runs:
        _require(zero_task_runs, "campaign has task-run evidence and cannot be marked zero-run")

    source_drift: list[dict[str, Any]] = []
    for relative_name, expected_digest in sorted(dict(prereg["source_hashes"]).items()):
        source_path = (REPOSITORY_ROOT / str(relative_name)).resolve()
        _require(source_path == REPOSITORY_ROOT or REPOSITORY_ROOT in source_path.parents, "frozen source path escapes repository")
        actual_digest = sha256_file(source_path) if source_path.is_file() and not source_path.is_symlink() else None
        if actual_digest != str(expected_digest).lower():
            source_drift.append(
                {
                    "path": str(relative_name),
                    "expected_sha256": str(expected_digest).lower(),
                    "actual_sha256": actual_digest,
                }
            )

    event_without_hash: dict[str, Any] = {
        "schema": INVALIDATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sequence": 1,
        "namespace": campaign.name,
        "status": "invalidated_failed_closed",
        "reason_code": normalized_code,
        "reason": normalized_reason,
        "replacement_namespace": replacement_namespace,
        "zero_task_runs_verified": zero_task_runs,
        "observed_run_directories": len(run_directories),
        "observed_run_files": len(run_files),
        "observed_candidate_freezes": freeze_count,
        "observed_grade_claims": claim_count,
        "observed_grade_receipts": receipt_count,
        "source_drift": source_drift,
        "preregistration_sha256": sha256_file(campaign / "preregistration.json"),
        "schedule_sha256": sha256_file(campaign / "schedule.json"),
        "campaign_lock_sha256": sha256_file(campaign / "campaign-lock.json"),
        "frozen_siim_run": UNIQUE_SIIM_RUN,
        "siim_private_grader_used": False,
        "kaggle_submission_performed": False,
        "previous_event_sha256": None,
        "invalidated_at": utc_now(),
    }
    event = {
        **event_without_hash,
        "event_sha256": sha256_bytes(canonical_bytes(event_without_hash)),
    }
    log_path = campaign / INVALIDATION_LOG_NAME
    _append_jsonl_fsync(log_path, event)
    verified = _read_invalidation_events(campaign)
    _require(len(verified) == 1 and verified[0]["event_sha256"] == event["event_sha256"], "invalidation write verification failed")
    return {
        "status": event["status"],
        "campaign_dir": str(campaign),
        "invalidation_log": str(log_path),
        **event,
    }


def _schedule_record(schedule: Mapping[str, Any], target_run_id: str) -> dict[str, Any]:
    matches = [item for item in list(schedule.get("runs") or []) if isinstance(item, dict) and item.get("run_id") == target_run_id]
    _require(len(matches) == 1, f"run is not uniquely preregistered: {target_run_id}")
    return matches[0]


def _safe_candidate_path(path: Path, allowed_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = allowed_root.expanduser().resolve()
    _require(resolved == root or root in resolved.parents, "candidate path escapes allowed root")
    _require(resolved.is_file() and not resolved.is_symlink(), "candidate is missing or symbolic")
    return resolved


def freeze_candidate(
    campaign_dir: str | Path,
    *,
    target_run_id: str,
    candidate_files: Sequence[str | Path],
    allowed_root: str | Path,
    public_validation: Mapping[str, Any],
) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    _, schedule = load_campaign(campaign)
    record = _schedule_record(schedule, target_run_id)
    _require(candidate_files, "candidate freeze requires files")
    files: list[dict[str, Any]] = []
    for raw in candidate_files:
        path = _safe_candidate_path(Path(raw), Path(allowed_root))
        files.append({"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    files.sort(key=lambda item: (item["name"], item["sha256"]))
    freeze = {
        "schema": FREEZE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "namespace": schedule["namespace"],
        "run_id": target_run_id,
        "task_id": record["task_id"],
        "seed": record["seed"],
        "arm": record["arm"],
        "status": "frozen_before_blind_local_grade",
        "files": files,
        "public_validation": dict(public_validation),
        "private_feedback_used": False,
        "kaggle_submission_performed": False,
        "siim_private_grader_used": False,
        "frozen_at": utc_now(),
    }
    path = write_json_exclusive(campaign / "runs" / target_run_id / "candidate_freeze.json", freeze)
    return {"freeze_path": str(path), "freeze_sha256": sha256_file(path), "candidate_files": files}


def record_blind_local_grade(
    campaign_dir: str | Path,
    *,
    target_run_id: str,
    freeze_sha256: str,
    grade: Mapping[str, Any],
) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    _, schedule = load_campaign(campaign)
    record = _schedule_record(schedule, target_run_id)
    run_dir = campaign / "runs" / target_run_id
    freeze_path = run_dir / "candidate_freeze.json"
    freeze = read_json(freeze_path, label="candidate freeze")
    _require(freeze.get("schema") == FREEZE_SCHEMA and freeze.get("run_id") == target_run_id, "candidate freeze binding mismatch")
    _require(bool(_SHA256.fullmatch(freeze_sha256.lower())) and sha256_file(freeze_path) == freeze_sha256.lower(), "candidate freeze hash mismatch")
    claim_path = run_dir / "local_grade_claim.json"
    receipt_path = run_dir / "local_grade_receipt.json"
    _require(not claim_path.exists() and not receipt_path.exists(), "blind local grade already claimed")
    claim = {
        "schema": GRADE_CLAIM_SCHEMA,
        "run_id": target_run_id,
        "task_id": record["task_id"],
        "candidate_freeze_sha256": freeze_sha256.lower(),
        "attempt": 1,
        "blind_to_arm": True,
        "claimed_at": utc_now(),
    }
    write_json_exclusive(claim_path, claim)
    try:
        sanitized = dict(grade)
        _reject_forbidden_result_keys(sanitized)
        score = sanitized.get("normalized_score")
        _require(score is None or (isinstance(score, (int, float)) and math.isfinite(float(score))), "normalized score must be finite or null")
        receipt = {
            "schema": GRADE_RECEIPT_SCHEMA,
            "run_id": target_run_id,
            "task_id": record["task_id"],
            "candidate_freeze_sha256": freeze_sha256.lower(),
            "claim_sha256": sha256_file(claim_path),
            "execution_count": 1,
            "evaluator": "blind_local_mle_grader",
            "official_kaggle_result": False,
            "siim_private_grader_used": False,
            "grade": sanitized,
            "graded_at": utc_now(),
        }
    except BaseException as exc:
        receipt = {
            "schema": GRADE_RECEIPT_SCHEMA,
            "run_id": target_run_id,
            "task_id": record["task_id"],
            "candidate_freeze_sha256": freeze_sha256.lower(),
            "claim_sha256": sha256_file(claim_path),
            "execution_count": 1,
            "evaluator": "blind_local_mle_grader",
            "official_kaggle_result": False,
            "siim_private_grader_used": False,
            "grade": {"status": "failed_closed", "normalized_score": None, "error_type": type(exc).__name__},
            "graded_at": utc_now(),
        }
        write_json_exclusive(receipt_path, receipt)
        raise
    write_json_exclusive(receipt_path, receipt)
    return {"receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path), **receipt}


def _reject_forbidden_result_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            _require(normalized not in _FORBIDDEN_RESULT_KEYS, f"forbidden result field: {'.'.join((*path, normalized))}")
            _reject_forbidden_result_keys(item, (*path, normalized))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_forbidden_result_keys(item, (*path, str(index)))


def _percentile(values: Sequence[float], quantile: float) -> float:
    _require(bool(values), "percentile requires values")
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_ci(task_deltas: Sequence[float], *, samples: int = 10_000, seed: int = 56_020_768) -> dict[str, Any]:
    _require(task_deltas and samples >= 100, "paired bootstrap requires task deltas and at least 100 samples")
    rng = random.Random(seed)
    values = [float(item) for item in task_deltas]
    estimates = [statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return {
        "method": "task_level_paired_bootstrap",
        "samples": samples,
        "seed": seed,
        "estimate": statistics.fmean(values),
        "ci95_lower": _percentile(estimates, 0.025),
        "ci95_upper": _percentile(estimates, 0.975),
    }


def _median(values: Iterable[float]) -> float:
    materialized = [float(item) for item in values]
    _require(bool(materialized), "median requires values")
    return float(statistics.median(materialized))


def aggregate(campaign_dir: str | Path, *, bootstrap_samples: int = 10_000) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    prereg, schedule = load_campaign(campaign)
    scheduled = list(schedule.get("runs") or [])
    observed: dict[tuple[str, int, str], dict[str, Any]] = {}
    audit_failures: list[str] = []
    receipt_times: list[str] = []
    for record in scheduled:
        target_run_id = str(record["run_id"])
        receipt_path = campaign / "runs" / target_run_id / "local_grade_receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path, label="local grade receipt")
        if isinstance(receipt.get("graded_at"), str):
            receipt_times.append(str(receipt["graded_at"]))
        if receipt.get("schema") != GRADE_RECEIPT_SCHEMA or receipt.get("execution_count") != 1:
            audit_failures.append(f"{target_run_id}:invalid_grade_receipt")
            continue
        freeze_path = campaign / "runs" / target_run_id / "candidate_freeze.json"
        if receipt.get("candidate_freeze_sha256") != sha256_file(freeze_path):
            audit_failures.append(f"{target_run_id}:hash_drift")
            continue
        grade = dict(receipt.get("grade") or {})
        required_flags = {
            "data_leakage": False,
            "private_feedback_used": False,
            "grader_reuse": False,
            "hash_drift": False,
            "unsupported_claim": False,
        }
        for key, expected in required_flags.items():
            if grade.get(key, False) is not expected:
                audit_failures.append(f"{target_run_id}:{key}")
        observed[(str(record["task_id"]), int(record["seed"]), str(record["arm"]))] = grade

    expected_count = int(schedule["expected_task_runs"])
    complete = len(observed) == expected_count
    pair_rows: list[dict[str, Any]] = []
    if complete:
        for task in prereg["tasks"]:
            for seed in prereg["seeds"]:
                baseline = observed[(task, int(seed), LEGACY_ARM)]
                treatment = observed[(task, int(seed), TREATMENT_ARM)]
                b_score, t_score = baseline.get("normalized_score"), treatment.get("normalized_score")
                valid_pair = isinstance(b_score, (int, float)) and isinstance(t_score, (int, float))
                pair_rows.append(
                    {
                        "task_id": task,
                        "seed": int(seed),
                        "baseline_valid": bool(baseline.get("valid", False)),
                        "treatment_valid": bool(treatment.get("valid", False)),
                        "baseline_score": float(b_score) if isinstance(b_score, (int, float)) else None,
                        "treatment_score": float(t_score) if isinstance(t_score, (int, float)) else None,
                        "delta": float(t_score) - float(b_score) if valid_pair else None,
                        "baseline_medal": bool(baseline.get("any_medal", False)),
                        "treatment_medal": bool(treatment.get("any_medal", False)),
                    }
                )

    task_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    seed_paired_deltas = [float(row["delta"]) for row in pair_rows if isinstance(row.get("delta"), (int, float))]
    for task in prereg["tasks"] if complete else []:
        rows = [row for row in pair_rows if row["task_id"] == task and row["delta"] is not None]
        if len(rows) == len(SEEDS):
            # The preregistered primary estimand is the per-task, three-seed
            # median for each arm.  Taking the median of seed-wise deltas is a
            # different statistic and can reverse a paired result when the two
            # arms have different seed ordering.
            baseline_median = _median(row["baseline_score"] for row in rows)
            treatment_median = _median(row["treatment_score"] for row in rows)
            delta = treatment_median - baseline_median
            task_rows.append(
                {
                    "task_id": task,
                    "baseline_median_score": baseline_median,
                    "treatment_median_score": treatment_median,
                    "median_paired_delta": delta,
                    "paired_win": delta > 0,
                    # For boolean evaluator outcomes, the median of three seeds
                    # is the majority result and is equivalent to evaluating the
                    # median-ranked candidate when the outcome is monotone.
                    "baseline_valid": _median(float(row["baseline_valid"]) for row in rows) >= 0.5,
                    "treatment_valid": _median(float(row["treatment_valid"]) for row in rows) >= 0.5,
                    "baseline_medal": _median(float(row["baseline_medal"]) for row in rows) >= 0.5,
                    "treatment_medal": _median(float(row["treatment_medal"]) for row in rows) >= 0.5,
                }
            )
    for seed in prereg["seeds"] if complete else []:
        rows = [row for row in pair_rows if row["seed"] == seed and row["delta"] is not None]
        if len(rows) == len(prereg["tasks"]):
            net = statistics.fmean(float(row["delta"]) for row in rows)
            seed_rows.append({"seed": seed, "net_delta": net, "positive": net > 0})

    baseline_grades = [observed[key] for key in observed if key[2] == LEGACY_ARM]
    treatment_grades = [observed[key] for key in observed if key[2] == TREATMENT_ARM]
    # Valid Rate and Medal Average are task-level primary metrics.  A task
    # contributes exactly once after its three-seed arm medians are frozen.
    baseline_valid_rate = statistics.fmean(bool(item["baseline_valid"]) for item in task_rows) if task_rows else 0.0
    treatment_valid_rate = statistics.fmean(bool(item["treatment_valid"]) for item in task_rows) if task_rows else 0.0
    baseline_medal_average = statistics.fmean(bool(item["baseline_medal"]) for item in task_rows) if task_rows else 0.0
    treatment_medal_average = statistics.fmean(bool(item["treatment_medal"]) for item in task_rows) if task_rows else 0.0

    def efficiency(items: Sequence[Mapping[str, Any]]) -> float:
        tokens = sum(max(0, int(item.get("total_tokens") or 0)) for item in items)
        new_bests = sum(max(0, int(item.get("new_best_count") or 0)) for item in items)
        return (new_bests * 1_000_000 / tokens) if tokens else 0.0

    baseline_efficiency = efficiency(baseline_grades)
    treatment_efficiency = efficiency(treatment_grades)
    efficiency_gain = (treatment_efficiency / baseline_efficiency - 1.0) if baseline_efficiency > 0 else float("-inf")
    baseline_prompt_p99 = _percentile([float(item.get("prompt_tokens_p99") or 0) for item in baseline_grades], 0.99) if baseline_grades else 0.0
    treatment_prompt_p99 = _percentile([float(item.get("prompt_tokens_p99") or 0) for item in treatment_grades], 0.99) if treatment_grades else 0.0
    prompt_reduction = (1.0 - treatment_prompt_p99 / baseline_prompt_p99) if baseline_prompt_p99 > 0 else float("-inf")
    task_paired_deltas = [float(row["median_paired_delta"]) for row in task_rows]
    bootstrap = paired_bootstrap_ci(task_paired_deltas, samples=bootstrap_samples) if task_paired_deltas else None

    common_checks = {
        "all_runs_terminal": complete,
        "valid_rate_non_decreasing": treatment_valid_rate >= baseline_valid_rate,
        "two_of_three_seed_net_positive": sum(bool(row["positive"]) for row in seed_rows) >= 2,
        "median_paired_delta_positive": _median(task_paired_deltas) > 0 if task_paired_deltas else False,
        "audit_clean": not audit_failures,
    }
    if prereg["phase"] == "screen":
        checks = {
            **common_checks,
            "four_of_six_task_wins": sum(bool(row["paired_win"]) for row in task_rows) >= 4,
            "new_best_per_million_tokens_gain_at_least_10pct": efficiency_gain >= 0.10,
            "prompt_p99_reduction_at_least_25pct": prompt_reduction >= 0.25,
        }
    else:
        checks = {
            **common_checks,
            "twenty_two_tasks_three_seeds_complete": len(task_rows) == 22 and complete,
            "medal_average_gain_at_least_4_55pp": (treatment_medal_average - baseline_medal_average) >= 0.0455,
            "thirteen_of_twenty_two_task_wins": sum(bool(row["paired_win"]) for row in task_rows) >= 13,
            "paired_bootstrap_ci_lower_nonnegative": bool(bootstrap and bootstrap["ci95_lower"] >= 0),
            "new_best_per_million_tokens_gain_at_least_20pct": efficiency_gain >= 0.20,
            "prompt_p99_reduction_at_least_40pct": prompt_reduction >= 0.40,
        }
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "namespace": prereg["namespace"],
        "phase": prereg["phase"],
        "status": "passed" if checks and all(checks.values()) else "failed_closed",
        "expected_task_runs": expected_count,
        "observed_task_runs": len(observed),
        "checks": checks,
        "audit_failures": sorted(audit_failures),
        "metrics": {
            "baseline_valid_rate": baseline_valid_rate,
            "treatment_valid_rate": treatment_valid_rate,
            "baseline_medal_average": baseline_medal_average,
            "treatment_medal_average": treatment_medal_average,
            "medal_average_delta_percentage_points": (treatment_medal_average - baseline_medal_average) * 100,
            "median_paired_delta": _median(task_paired_deltas) if task_paired_deltas else None,
            "seed_pair_count": len(seed_paired_deltas),
            "baseline_new_best_per_million_tokens": baseline_efficiency,
            "treatment_new_best_per_million_tokens": treatment_efficiency,
            "new_best_efficiency_gain": efficiency_gain if math.isfinite(efficiency_gain) else None,
            "baseline_prompt_p99": baseline_prompt_p99,
            "treatment_prompt_p99": treatment_prompt_p99,
            "prompt_p99_reduction": prompt_reduction if math.isfinite(prompt_reduction) else None,
        },
        "paired_bootstrap": bootstrap,
        "task_results": task_rows,
        "seed_results": seed_rows,
        # Deterministic from immutable inputs: repeated aggregation must produce
        # identical bytes and may never manufacture a fresh result timestamp.
        "generated_at": max(receipt_times) if receipt_times else str(prereg.get("created_at") or ""),
        "claim_boundary": "local MLE-Bench evaluation; not an official Kaggle result",
    }
    output = campaign / "aggregate.json"
    if output.exists():
        _require(sha256_file(output) == sha256_bytes(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"), "aggregate already exists with different bytes")
    else:
        write_json_exclusive(output, payload)
    return payload


def shadow_replay(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    replay: Any,
) -> dict[str, Any]:
    """Run a deterministic, no-LLM shadow adapter twice and compare bytes."""

    first = replay([dict(item) for item in trajectory])
    second = replay([dict(item) for item in trajectory])
    first_bytes, second_bytes = canonical_bytes(first), canonical_bytes(second)
    serialized = first_bytes.decode("utf-8", errors="replace").lower()
    forbidden = [token for token in ("private_grader", "private_feedback", "private_leaderboard") if token in serialized]
    return {
        "schema": SHADOW_SCHEMA,
        "status": "passed" if first_bytes == second_bytes and not forbidden else "failed_closed",
        "deterministic": first_bytes == second_bytes,
        "private_feedback_absent": not forbidden,
        "forbidden_tokens": forbidden,
        "trajectory_sha256": sha256_bytes(canonical_bytes(list(trajectory))),
        "replay_sha256": sha256_bytes(first_bytes),
        "result": first,
    }
