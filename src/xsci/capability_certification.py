"""Fail-closed release certification for externally evaluated agent capability.

The certificate is deliberately a verifier, not a benchmark runner.  A release
pipeline must receive a report and its SHA-256 digest from an independent
evaluator, then bind that report to the exact Git source and release artifacts.
Self-reported scores or an unattached internal benchmark are never sufficient.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any, Mapping, Sequence

EVIDENCE_SCHEMA = "evomind.capability_certification_evidence.v1"
RESULT_SCHEMA = "evomind.capability_certification_result.v1"
CAMPAIGN_SCHEMA = "evomind.self_upgrade_campaign_evidence.v1"
SOURCE_DIGEST_ALGORITHM = "evomind.git_source_manifest.sha256.v1"
RAW_TRIAL_SCHEMA = "evomind.capability_raw_trial.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class CertificationInputError(ValueError):
    """Raised for an invalid verifier policy or an unreadable repository."""


@dataclass(frozen=True)
class CertificationPolicy:
    """Trust anchors and quantitative gates supplied by the release owner."""

    expected_report_sha256: str
    expected_suite_id: str
    expected_suite_sha256: str
    expected_evaluator_id: str
    expected_evaluator_sha256: str
    required_baseline_agents: tuple[str, ...]
    expected_candidate_name: str = "EvoMind"
    minimum_hidden_tasks: int = 100
    minimum_domains: int = 8
    minimum_tasks_per_domain: int = 3
    minimum_repeats: int = 3
    minimum_candidate_success_rate: float = 0.80
    minimum_candidate_wilson_lower_bound: float = 0.75
    confidence_level: float = 0.95
    baseline_noninferiority_margin: float = 0.05
    maximum_timeout_rate: float = 0.0
    minimum_upgrade_candidates: int = 2
    maximum_evidence_age_days: int = 30
    required_artifact_roles: tuple[str, ...] = (
        "wheel",
        "sdist",
        "source_archive",
        "benchmark_raw_results",
        "self_upgrade_campaign",
    )

    def __post_init__(self) -> None:
        digest_fields = (
            "expected_report_sha256",
            "expected_suite_sha256",
            "expected_evaluator_sha256",
        )
        for field_name in digest_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise CertificationInputError(f"{field_name} must be a lowercase SHA-256 digest")
        for field_name in ("expected_suite_id", "expected_evaluator_id", "expected_candidate_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise CertificationInputError(f"{field_name} must be non-empty")
        if not self.required_baseline_agents or any(
            not isinstance(name, str) or not name.strip() for name in self.required_baseline_agents
        ):
            raise CertificationInputError("required_baseline_agents must contain named agents")
        if len({_normalized_name(name) for name in self.required_baseline_agents}) != len(
            self.required_baseline_agents
        ):
            raise CertificationInputError("required_baseline_agents must be unique")
        integer_minimums = {
            "minimum_hidden_tasks": self.minimum_hidden_tasks,
            "minimum_domains": self.minimum_domains,
            "minimum_tasks_per_domain": self.minimum_tasks_per_domain,
            "minimum_repeats": self.minimum_repeats,
            "minimum_upgrade_candidates": self.minimum_upgrade_candidates,
            "maximum_evidence_age_days": self.maximum_evidence_age_days,
        }
        for field_name, value in integer_minimums.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CertificationInputError(f"{field_name} must be a positive integer")
        probability_fields = {
            "minimum_candidate_success_rate": self.minimum_candidate_success_rate,
            "minimum_candidate_wilson_lower_bound": self.minimum_candidate_wilson_lower_bound,
            "confidence_level": self.confidence_level,
            "baseline_noninferiority_margin": self.baseline_noninferiority_margin,
            "maximum_timeout_rate": self.maximum_timeout_rate,
        }
        for field_name, value in probability_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise CertificationInputError(f"{field_name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise CertificationInputError(f"{field_name} must be between 0 and 1")
        if not 0.5 < self.confidence_level < 1.0:
            raise CertificationInputError("confidence_level must be greater than 0.5 and less than 1")
        if not self.required_artifact_roles or any(
            not isinstance(role, str) or not role.strip() for role in self.required_artifact_roles
        ):
            raise CertificationInputError("required_artifact_roles must be non-empty")
        if len(set(self.required_artifact_roles)) != len(self.required_artifact_roles):
            raise CertificationInputError("required_artifact_roles must be unique")


@dataclass(frozen=True)
class RepositoryIdentity:
    commit_sha: str
    tree_sha: str
    source_digest_sha256: str
    clean: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "source_digest_algorithm": SOURCE_DIGEST_ALGORITHM,
            "source_digest_sha256": self.source_digest_sha256,
            "worktree_clean": self.clean,
        }


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compute_evidence_payload_sha256(report: Mapping[str, Any]) -> str:
    """Hash the semantic report payload, excluding its self-describing attestation."""

    payload = dict(report)
    payload.pop("attestation", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *args: str, timeout: int = 30) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CertificationInputError("git command could not be executed") from exc
    if completed.returncode != 0:
        raise CertificationInputError("repository identity could not be read")
    return completed.stdout


def compute_source_digest(repository: Path) -> str:
    """Compute a SHA-256 digest over paths, modes, types, and committed bytes."""

    listing = _git(repository, "ls-tree", "-r", "-z", "--full-tree", "HEAD", timeout=60)
    entries: list[tuple[bytes, bytes, bytes, bytes]] = []
    for raw_entry in listing.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as exc:
            raise CertificationInputError("git tree contained an invalid entry") from exc
        if object_type not in {b"blob", b"commit"} or not path:
            raise CertificationInputError("git tree contained an unsupported entry")
        entries.append((path, mode, object_type, object_id))

    blob_ids = [object_id for _, _, object_type, object_id in entries if object_type == b"blob"]
    blob_payloads: dict[bytes, bytes] = {}
    if blob_ids:
        batch_input = b"".join(object_id + b"\n" for object_id in blob_ids)
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), "cat-file", "--batch"],
                input=batch_input,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CertificationInputError("committed source objects could not be read") from exc
        if completed.returncode != 0:
            raise CertificationInputError("committed source objects could not be read")
        output = completed.stdout
        offset = 0
        for requested_id in blob_ids:
            newline = output.find(b"\n", offset)
            if newline < 0:
                raise CertificationInputError("git object batch output was truncated")
            header = output[offset:newline].split(b" ")
            if len(header) != 3 or header[1] != b"blob":
                raise CertificationInputError("git object batch output was invalid")
            try:
                size = int(header[2])
            except ValueError as exc:
                raise CertificationInputError("git object size was invalid") from exc
            start = newline + 1
            end = start + size
            if end >= len(output) or output[end : end + 1] != b"\n":
                raise CertificationInputError("git object batch payload was truncated")
            blob_payloads[requested_id] = output[start:end]
            offset = end + 1
        if offset != len(output):
            raise CertificationInputError("git object batch output had trailing data")

    digest = hashlib.sha256(b"evomind-git-source-manifest-v1\0")
    for path, mode, object_type, object_id in entries:
        content = blob_payloads[object_id] if object_type == b"blob" else object_id
        for field in (path, mode, object_type):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def compute_repository_identity(repository: Path) -> RepositoryIdentity:
    repository = repository.resolve()
    commit_sha = _git(repository, "rev-parse", "HEAD").decode("ascii", errors="strict").strip().lower()
    tree_sha = _git(repository, "rev-parse", "HEAD^{tree}").decode("ascii", errors="strict").strip().lower()
    if GIT_OID_RE.fullmatch(commit_sha) is None or GIT_OID_RE.fullmatch(tree_sha) is None:
        raise CertificationInputError("repository returned an invalid commit or tree identifier")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all", timeout=60)
    return RepositoryIdentity(
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        source_digest_sha256=compute_source_digest(repository),
        clean=not bool(status.strip()),
    )


def wilson_lower_bound(successes: int, attempts: int, confidence_level: float) -> float:
    if attempts <= 0 or successes < 0 or successes > attempts:
        raise ValueError("invalid binomial counts")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / attempts
    z2 = z * z
    denominator = 1.0 + z2 / attempts
    centre = proportion + z2 / (2.0 * attempts)
    adjustment = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * attempts)) / attempts)
    return (centre - adjustment) / denominator


def paired_difference_lower_bound(
    candidate_only: int,
    baseline_only: int,
    total: int,
    confidence_level: float,
) -> float:
    """Two-sided normal lower bound for paired {-1, 0, 1} outcomes."""

    if total <= 0 or min(candidate_only, baseline_only) < 0 or candidate_only + baseline_only > total:
        raise ValueError("invalid paired counts")
    mean = (candidate_only - baseline_only) / total
    if total == 1:
        return mean
    sum_squares = candidate_only + baseline_only
    sample_variance = max(0.0, (sum_squares - total * mean * mean) / (total - 1))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return mean - z * math.sqrt(sample_variance / total)


def _strict_json_loads(data: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        data.decode("utf-8-sig"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _same_float(actual: Any, expected: float, tolerance: float = 1e-9) -> bool:
    return _is_number(actual) and math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance)


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, check_id: str, passed: bool, detail: str) -> bool:
        self.items.append({"id": check_id, "passed": bool(passed), "detail": detail})
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item["passed"] for item in self.items)


def _exact_keys(checks: _Checks, check_id: str, value: Mapping[str, Any], expected: set[str]) -> bool:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    detail = "exact fields present"
    if missing or unexpected:
        detail = f"missing={missing}; unexpected={unexpected}"
    return checks.add(check_id, not missing and not unexpected, detail)


_METRIC_KEYS = {
    "agent_name",
    "agent_version",
    "provider",
    "model",
    "attempted_trials",
    "successful_trials",
    "failed_trials",
    "timed_out_trials",
    "scope_violations",
    "unsupported_claims",
    "success_rate",
    "confidence_level",
    "wilson_lower_bound",
}


def _validate_metric_record(
    checks: _Checks,
    *,
    prefix: str,
    record: Any,
    expected_trials: int | None,
    policy: CertificationPolicy,
    expected_name: str | None = None,
) -> dict[str, Any] | None:
    if not checks.add(f"{prefix}.object", isinstance(record, dict), "metric record is an object"):
        return None
    assert isinstance(record, dict)
    _exact_keys(checks, f"{prefix}.fields", record, _METRIC_KEYS)

    identity_ok = all(
        isinstance(record.get(field), str) and bool(record[field].strip())
        for field in ("agent_name", "agent_version", "provider", "model")
    )
    if expected_name is not None:
        identity_ok = identity_ok and _normalized_name(str(record.get("agent_name", ""))) == _normalized_name(
            expected_name
        )
    checks.add(f"{prefix}.identity", identity_ok, "agent identity, version, provider, and model are bound")

    counter_names = (
        "attempted_trials",
        "successful_trials",
        "failed_trials",
        "timed_out_trials",
        "scope_violations",
        "unsupported_claims",
    )
    counters_ok = all(_is_int(record.get(name)) and record[name] >= 0 for name in counter_names)
    checks.add(f"{prefix}.counters", counters_ok, "all counters are non-negative integers")
    if not counters_ok:
        return record

    attempts = record["attempted_trials"]
    successes = record["successful_trials"]
    failures = record["failed_trials"]
    timed_out = record["timed_out_trials"]
    expected_ok = expected_trials is not None and attempts == expected_trials
    checks.add(f"{prefix}.trial_scope", expected_ok, "attempt count equals the complete held-out trial matrix")
    checks.add(
        f"{prefix}.outcome_partition",
        attempts > 0 and successes + failures == attempts and timed_out <= failures,
        "successes and failures partition attempts; timeouts are failures",
    )
    checks.add(
        f"{prefix}.zero_scope_violations",
        record["scope_violations"] == 0,
        "scope violations must be zero",
    )
    checks.add(
        f"{prefix}.zero_unsupported_claims",
        record["unsupported_claims"] == 0,
        "unsupported claims must be zero",
    )

    if attempts > 0 and 0 <= successes <= attempts:
        success_rate = successes / attempts
        lower_bound = wilson_lower_bound(successes, attempts, policy.confidence_level)
        checks.add(
            f"{prefix}.success_rate_recomputed",
            _same_float(record.get("success_rate"), success_rate),
            "reported success rate matches counts",
        )
        checks.add(
            f"{prefix}.confidence_level",
            _same_float(record.get("confidence_level"), policy.confidence_level),
            "reported confidence level matches release policy",
        )
        checks.add(
            f"{prefix}.wilson_lower_bound_recomputed",
            _same_float(record.get("wilson_lower_bound"), lower_bound),
            "reported Wilson lower bound matches independent recomputation",
        )
    else:
        checks.add(f"{prefix}.success_rate_recomputed", False, "invalid trial counts")
        checks.add(f"{prefix}.confidence_level", False, "invalid trial counts")
        checks.add(f"{prefix}.wilson_lower_bound_recomputed", False, "invalid trial counts")
    return record


def _validate_external_evaluation(
    checks: _Checks,
    evaluation: Any,
    policy: CertificationPolicy,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not checks.add("external_evaluation.object", isinstance(evaluation, dict), "external evaluation is an object"):
        return None, {}
    assert isinstance(evaluation, dict)
    _exact_keys(
        checks,
        "external_evaluation.fields",
        evaluation,
        {"suite", "evaluator", "timeout_policy", "candidate", "baselines", "paired_comparisons"},
    )

    suite = evaluation.get("suite")
    expected_trials: int | None = None
    if checks.add("suite.object", isinstance(suite, dict), "suite identity is an object"):
        assert isinstance(suite, dict)
        _exact_keys(
            checks,
            "suite.fields",
            suite,
            {
                "id",
                "version",
                "digest_sha256",
                "hidden_task_count",
                "domains",
                "domain_task_counts",
                "repeat_count",
                "total_trials",
                "held_out",
                "selection_locked_before_evaluation",
                "tasks_not_used_for_development",
            },
        )
        checks.add(
            "suite.trust_anchor",
            suite.get("id") == policy.expected_suite_id
            and suite.get("digest_sha256") == policy.expected_suite_sha256
            and isinstance(suite.get("version"), str)
            and bool(suite.get("version", "").strip()),
            "suite identity and digest match external trust anchors",
        )
        tasks = suite.get("hidden_task_count")
        repeats = suite.get("repeat_count")
        total_trials = suite.get("total_trials")
        task_shape_ok = (
            _is_int(tasks)
            and tasks >= policy.minimum_hidden_tasks
            and _is_int(repeats)
            and repeats >= policy.minimum_repeats
            and _is_int(total_trials)
            and total_trials == tasks * repeats
        )
        checks.add(
            "suite.minimum_task_repeat_scope",
            task_shape_ok,
            "hidden task, repeat, and total trial counts satisfy policy",
        )
        if task_shape_ok:
            expected_trials = total_trials

        domains = suite.get("domains")
        domain_counts = suite.get("domain_task_counts")
        domain_ok = (
            isinstance(domains, list)
            and all(isinstance(domain, str) and bool(domain.strip()) for domain in domains)
            and len({_normalized_name(domain) for domain in domains}) == len(domains)
            and len(domains) >= policy.minimum_domains
            and isinstance(domain_counts, dict)
            and set(domain_counts) == set(domains)
            and all(_is_int(count) and count >= policy.minimum_tasks_per_domain for count in domain_counts.values())
            and _is_int(tasks)
            and sum(domain_counts.values()) == tasks
        )
        checks.add(
            "suite.minimum_domain_scope",
            domain_ok,
            "domain allocation is unique, complete, and satisfies per-domain minimums",
        )
        checks.add(
            "suite.held_out_controls",
            suite.get("held_out") is True
            and suite.get("selection_locked_before_evaluation") is True
            and suite.get("tasks_not_used_for_development") is True,
            "suite is held out and locked before evaluation",
        )

    evaluator = evaluation.get("evaluator")
    if checks.add("evaluator.object", isinstance(evaluator, dict), "evaluator identity is an object"):
        assert isinstance(evaluator, dict)
        _exact_keys(
            checks,
            "evaluator.fields",
            evaluator,
            {
                "id",
                "version",
                "digest_sha256",
                "organization",
                "independent",
                "execution_isolated",
                "all_trials_reported",
            },
        )
        checks.add(
            "evaluator.trust_anchor",
            evaluator.get("id") == policy.expected_evaluator_id
            and evaluator.get("digest_sha256") == policy.expected_evaluator_sha256
            and isinstance(evaluator.get("version"), str)
            and bool(evaluator.get("version", "").strip())
            and isinstance(evaluator.get("organization"), str)
            and bool(evaluator.get("organization", "").strip()),
            "evaluator identity and digest match external trust anchors",
        )
        checks.add(
            "evaluator.independence",
            evaluator.get("independent") is True
            and evaluator.get("execution_isolated") is True
            and evaluator.get("all_trials_reported") is True,
            "evaluator is independent, isolated, and reports all trials",
        )

    timeout_policy = evaluation.get("timeout_policy")
    reported_timeout_limit: float | None = None
    if checks.add("timeout_policy.object", isinstance(timeout_policy, dict), "timeout policy is an object"):
        assert isinstance(timeout_policy, dict)
        _exact_keys(
            checks,
            "timeout_policy.fields",
            timeout_policy,
            {
                "timeouts_count_as_failures",
                "max_timeout_rate",
                "retry_on_timeout",
                "timeout_seconds",
                "uniform_across_agents",
            },
        )
        if _is_number(timeout_policy.get("max_timeout_rate")):
            reported_timeout_limit = float(timeout_policy["max_timeout_rate"])
        timeout_controls_ok = (
            timeout_policy.get("timeouts_count_as_failures") is True
            and reported_timeout_limit is not None
            and 0.0 <= reported_timeout_limit <= policy.maximum_timeout_rate
            and timeout_policy.get("retry_on_timeout") is False
            and _is_number(timeout_policy.get("timeout_seconds"))
            and float(timeout_policy["timeout_seconds"]) > 0.0
            and timeout_policy.get("uniform_across_agents") is True
        )
        checks.add(
            "timeout_policy.controls",
            timeout_controls_ok,
            "timeouts are uniform, non-retried failures within the release limit",
        )

    candidate = _validate_metric_record(
        checks,
        prefix="candidate",
        record=evaluation.get("candidate"),
        expected_trials=expected_trials,
        policy=policy,
        expected_name=policy.expected_candidate_name,
    )
    if (
        candidate is not None
        and _is_int(candidate.get("attempted_trials"))
        and candidate["attempted_trials"] > 0
        and _is_int(candidate.get("timed_out_trials"))
        and candidate["timed_out_trials"] >= 0
    ):
        timeout_rate = candidate.get("timed_out_trials", candidate["attempted_trials"]) / candidate["attempted_trials"]
        effective_limit = policy.maximum_timeout_rate
        if reported_timeout_limit is not None:
            effective_limit = min(effective_limit, reported_timeout_limit)
        checks.add("candidate.timeout_rate", timeout_rate <= effective_limit, "candidate timeout rate satisfies policy")
        checks.add(
            "candidate.absolute_thresholds",
            _is_number(candidate.get("success_rate"))
            and float(candidate["success_rate"]) >= policy.minimum_candidate_success_rate
            and _is_number(candidate.get("wilson_lower_bound"))
            and float(candidate["wilson_lower_bound"]) >= policy.minimum_candidate_wilson_lower_bound,
            "candidate success rate and confidence lower bound satisfy release thresholds",
        )
    else:
        checks.add("candidate.timeout_rate", False, "candidate attempts are invalid")
        checks.add("candidate.absolute_thresholds", False, "candidate attempts are invalid")

    baselines = evaluation.get("baselines")
    baseline_by_name: dict[str, dict[str, Any]] = {}
    baselines_list_ok = isinstance(baselines, list) and bool(baselines)
    checks.add("baselines.array", baselines_list_ok, "baseline results are a non-empty array")
    if isinstance(baselines, list):
        for index, baseline in enumerate(baselines):
            validated = _validate_metric_record(
                checks,
                prefix=f"baseline.{index}",
                record=baseline,
                expected_trials=expected_trials,
                policy=policy,
            )
            if validated is None or not isinstance(validated.get("agent_name"), str):
                continue
            normalized = _normalized_name(validated["agent_name"])
            if normalized in baseline_by_name:
                checks.add(f"baseline.{index}.unique_name", False, "baseline agent names must be unique")
            else:
                checks.add(f"baseline.{index}.unique_name", True, "baseline agent name is unique")
                baseline_by_name[normalized] = validated
            if (
                _is_int(validated.get("attempted_trials"))
                and validated["attempted_trials"] > 0
                and _is_int(validated.get("timed_out_trials"))
                and validated["timed_out_trials"] >= 0
            ):
                timeout_rate = validated.get("timed_out_trials", validated["attempted_trials"]) / validated[
                    "attempted_trials"
                ]
                effective_limit = policy.maximum_timeout_rate
                if reported_timeout_limit is not None:
                    effective_limit = min(effective_limit, reported_timeout_limit)
                checks.add(
                    f"baseline.{index}.timeout_rate",
                    timeout_rate <= effective_limit,
                    "baseline timeout rate satisfies the same policy",
                )
    missing_baselines = sorted(
        name
        for name in policy.required_baseline_agents
        if _normalized_name(name) not in baseline_by_name
    )
    checks.add(
        "baselines.required_agents",
        not missing_baselines,
        "all release-policy baseline agents are present" if not missing_baselines else f"missing={missing_baselines}",
    )

    comparisons = evaluation.get("paired_comparisons")
    comparison_names: set[str] = set()
    checks.add("comparisons.array", isinstance(comparisons, list) and bool(comparisons), "paired comparisons are present")
    if isinstance(comparisons, list):
        comparison_keys = {
            "baseline_agent_name",
            "total_pairs",
            "both_passed",
            "candidate_only_passed",
            "baseline_only_passed",
            "both_failed",
            "candidate_minus_baseline_rate",
            "confidence_level",
            "difference_lower_bound",
        }
        for index, comparison in enumerate(comparisons):
            prefix = f"comparison.{index}"
            if not checks.add(f"{prefix}.object", isinstance(comparison, dict), "comparison is an object"):
                continue
            assert isinstance(comparison, dict)
            _exact_keys(checks, f"{prefix}.fields", comparison, comparison_keys)
            name = comparison.get("baseline_agent_name")
            normalized = _normalized_name(name) if isinstance(name, str) else ""
            name_ok = bool(normalized) and normalized in baseline_by_name and normalized not in comparison_names
            checks.add(f"{prefix}.baseline_binding", name_ok, "comparison binds one unique baseline result")
            if name_ok:
                comparison_names.add(normalized)
            count_fields = (
                "total_pairs",
                "both_passed",
                "candidate_only_passed",
                "baseline_only_passed",
                "both_failed",
            )
            counts_ok = all(_is_int(comparison.get(field)) and comparison[field] >= 0 for field in count_fields)
            checks.add(f"{prefix}.counts", counts_ok, "paired outcome counts are non-negative integers")
            if not counts_ok:
                continue
            total = comparison["total_pairs"]
            both = comparison["both_passed"]
            candidate_only = comparison["candidate_only_passed"]
            baseline_only = comparison["baseline_only_passed"]
            neither = comparison["both_failed"]
            partition_ok = (
                expected_trials is not None
                and total == expected_trials
                and both + candidate_only + baseline_only + neither == total
            )
            checks.add(f"{prefix}.partition", partition_ok, "paired outcomes partition the complete trial matrix")
            difference = (candidate_only - baseline_only) / total if total > 0 else float("nan")
            try:
                lower = paired_difference_lower_bound(
                    candidate_only,
                    baseline_only,
                    total,
                    policy.confidence_level,
                )
            except ValueError:
                lower = float("nan")
            checks.add(
                f"{prefix}.difference_recomputed",
                math.isfinite(difference)
                and _same_float(comparison.get("candidate_minus_baseline_rate"), difference)
                and _same_float(comparison.get("confidence_level"), policy.confidence_level)
                and math.isfinite(lower)
                and _same_float(comparison.get("difference_lower_bound"), lower),
                "paired difference and confidence lower bound match independent recomputation",
            )
            checks.add(
                f"{prefix}.noninferiority",
                math.isfinite(lower) and lower >= -policy.baseline_noninferiority_margin,
                "paired confidence lower bound satisfies the baseline noninferiority margin",
            )
            if name_ok and candidate is not None:
                baseline = baseline_by_name[normalized]
                checks.add(
                    f"{prefix}.score_binding",
                    candidate.get("successful_trials") == both + candidate_only
                    and baseline.get("successful_trials") == both + baseline_only,
                    "paired counts reproduce candidate and baseline scores",
                )

    missing_comparisons = sorted(
        name
        for name in policy.required_baseline_agents
        if _normalized_name(name) not in comparison_names
    )
    checks.add(
        "comparisons.required_agents",
        not missing_comparisons,
        "all required baselines have paired comparisons"
        if not missing_comparisons
        else f"missing={missing_comparisons}",
    )
    return candidate, baseline_by_name


def compute_campaign_payload_sha256(campaign: Mapping[str, Any]) -> str:
    payload = dict(campaign)
    payload.pop("attestation", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _resolve_artifact_path(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative:
        return None
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return None
    try:
        resolved_root = root.resolve(strict=True)
        candidate = root.joinpath(*pure.parts)
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _validate_artifacts(
    checks: _Checks,
    artifacts: Any,
    *,
    artifact_root: Path,
    policy: CertificationPolicy,
) -> dict[str, tuple[dict[str, Any], Path]]:
    result: dict[str, tuple[dict[str, Any], Path]] = {}
    if not checks.add("artifacts.array", isinstance(artifacts, list) and bool(artifacts), "artifact records are present"):
        return result
    assert isinstance(artifacts, list)
    seen_paths: set[str] = set()
    for index, item in enumerate(artifacts):
        prefix = f"artifact.{index}"
        if not checks.add(f"{prefix}.object", isinstance(item, dict), "artifact record is an object"):
            continue
        assert isinstance(item, dict)
        _exact_keys(checks, f"{prefix}.fields", item, {"role", "path", "sha256", "size_bytes"})
        role = item.get("role")
        path_value = item.get("path")
        identity_ok = (
            isinstance(role, str)
            and bool(role.strip())
            and role not in result
            and isinstance(path_value, str)
            and path_value not in seen_paths
            and isinstance(item.get("sha256"), str)
            and SHA256_RE.fullmatch(item["sha256"]) is not None
            and _is_int(item.get("size_bytes"))
            and item["size_bytes"] >= 0
        )
        checks.add(f"{prefix}.identity", identity_ok, "artifact role, path, digest, and size are well formed and unique")
        if not identity_ok:
            continue
        seen_paths.add(path_value)
        path = _resolve_artifact_path(artifact_root, path_value)
        checks.add(f"{prefix}.path", path is not None, "artifact is a regular file contained by artifact root")
        if path is None:
            continue
        try:
            actual_size = path.stat().st_size
            actual_digest = sha256_file(path)
        except OSError:
            checks.add(f"{prefix}.content", False, "artifact could not be read")
            continue
        checks.add(
            f"{prefix}.content",
            actual_size == item["size_bytes"] and actual_digest == item["sha256"],
            "artifact size and SHA-256 match the externally bound record",
        )
        result[role] = (item, path)

    missing = sorted(role for role in policy.required_artifact_roles if role not in result)
    checks.add(
        "artifacts.required_roles",
        not missing,
        "all required release evidence artifacts are verified" if not missing else f"missing={missing}",
    )
    return result


_RAW_TRIAL_KEYS = {
    "schema",
    "task_id",
    "domain",
    "repeat",
    "agent_name",
    "outcome",
    "scope_violation",
    "unsupported_claim",
}


def _validate_raw_results(
    checks: _Checks,
    *,
    artifact: tuple[dict[str, Any], Path] | None,
    evaluation: Any,
) -> None:
    """Recompute aggregate and paired metrics from the complete trial matrix."""

    setup_ok = artifact is not None and isinstance(evaluation, dict)
    suite = evaluation.get("suite") if isinstance(evaluation, dict) else None
    candidate = evaluation.get("candidate") if isinstance(evaluation, dict) else None
    baselines = evaluation.get("baselines") if isinstance(evaluation, dict) else None
    comparisons = evaluation.get("paired_comparisons") if isinstance(evaluation, dict) else None
    setup_ok = setup_ok and isinstance(suite, dict) and isinstance(candidate, dict) and isinstance(baselines, list)
    checks.add("raw_results.setup", setup_ok, "raw trial artifact and evaluation identities are available")
    if not setup_ok or artifact is None:
        checks.add("raw_results.strict_records", False, "raw trial records cannot be parsed")
        checks.add("raw_results.complete_matrix", False, "raw trial matrix cannot be established")
        checks.add("raw_results.metric_binding", False, "raw metrics cannot be recomputed")
        checks.add("raw_results.paired_binding", False, "raw paired outcomes cannot be recomputed")
        return

    assert isinstance(suite, dict)
    assert isinstance(candidate, dict)
    assert isinstance(baselines, list)
    metric_records = [candidate, *[item for item in baselines if isinstance(item, dict)]]
    agent_records: dict[str, dict[str, Any]] = {}
    identities_ok = len(metric_records) == 1 + len(baselines)
    for record in metric_records:
        name = record.get("agent_name")
        normalized = _normalized_name(name) if isinstance(name, str) else ""
        if not normalized or normalized in agent_records:
            identities_ok = False
            continue
        agent_records[normalized] = record

    hidden_tasks = suite.get("hidden_task_count")
    repeat_count = suite.get("repeat_count")
    total_trials = suite.get("total_trials")
    domains = suite.get("domains")
    domain_counts = suite.get("domain_task_counts")
    domains_ok = (
        isinstance(domains, list)
        and bool(domains)
        and all(isinstance(domain, str) and bool(domain.strip()) for domain in domains)
        and len(set(domains)) == len(domains)
    )
    shape_ok = (
        identities_ok
        and bool(agent_records)
        and _is_int(hidden_tasks)
        and hidden_tasks > 0
        and _is_int(repeat_count)
        and repeat_count > 0
        and _is_int(total_trials)
        and total_trials == hidden_tasks * repeat_count
        and domains_ok
        and isinstance(domain_counts, dict)
        and set(domain_counts) == set(domains)
        and all(_is_int(count) and count > 0 for count in domain_counts.values())
    )
    expected_records = total_trials * len(agent_records) if shape_ok else 0
    shape_ok = shape_ok and 0 < expected_records <= 2_000_000
    if not shape_ok:
        checks.add("raw_results.strict_records", False, "raw trial matrix declaration is invalid or too large")
        checks.add("raw_results.complete_matrix", False, "raw trial matrix cannot be established")
        checks.add("raw_results.metric_binding", False, "raw metrics cannot be recomputed")
        checks.add("raw_results.paired_binding", False, "raw paired outcomes cannot be recomputed")
        return

    expected_domains = set(domains)
    records: dict[tuple[str, str, int], str] = {}
    task_domains: dict[str, str] = {}
    raw_counts = {
        name: {
            "attempted_trials": 0,
            "successful_trials": 0,
            "failed_trials": 0,
            "timed_out_trials": 0,
            "scope_violations": 0,
            "unsupported_claims": 0,
        }
        for name in agent_records
    }
    records_ok = True
    try:
        with artifact[1].open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or len(line) > 64 * 1024 or len(records) >= expected_records:
                    records_ok = False
                    break
                try:
                    item = _strict_json_loads(line)
                except (UnicodeError, ValueError, json.JSONDecodeError):
                    records_ok = False
                    break
                if not isinstance(item, dict) or set(item) != _RAW_TRIAL_KEYS:
                    records_ok = False
                    break
                task_id = item.get("task_id")
                domain = item.get("domain")
                repeat = item.get("repeat")
                agent_name = item.get("agent_name")
                normalized_agent = _normalized_name(agent_name) if isinstance(agent_name, str) else ""
                outcome = item.get("outcome")
                item_ok = (
                    item.get("schema") == RAW_TRIAL_SCHEMA
                    and isinstance(task_id, str)
                    and 0 < len(task_id) <= 512
                    and isinstance(domain, str)
                    and domain in expected_domains
                    and _is_int(repeat)
                    and 1 <= repeat <= repeat_count
                    and normalized_agent in agent_records
                    and outcome in {"passed", "failed", "timeout"}
                    and isinstance(item.get("scope_violation"), bool)
                    and isinstance(item.get("unsupported_claim"), bool)
                )
                key = (normalized_agent, task_id, repeat) if item_ok else ("", str(line_number), 0)
                if not item_ok or key in records or task_id in task_domains and task_domains[task_id] != domain:
                    records_ok = False
                    break
                task_domains[task_id] = domain
                records[key] = outcome
                counters = raw_counts[normalized_agent]
                counters["attempted_trials"] += 1
                if outcome == "passed":
                    counters["successful_trials"] += 1
                else:
                    counters["failed_trials"] += 1
                if outcome == "timeout":
                    counters["timed_out_trials"] += 1
                counters["scope_violations"] += int(item["scope_violation"])
                counters["unsupported_claims"] += int(item["unsupported_claim"])
    except OSError:
        records_ok = False
    checks.add("raw_results.strict_records", records_ok, "raw trial JSONL is strict, bounded, and duplicate-free")

    observed_domain_counts = {
        domain: sum(observed == domain for observed in task_domains.values())
        for domain in expected_domains
    }
    complete_matrix = (
        records_ok
        and len(records) == expected_records
        and len(task_domains) == hidden_tasks
        and observed_domain_counts == domain_counts
        and all(
            (agent, task_id, repeat) in records
            for agent in agent_records
            for task_id in task_domains
            for repeat in range(1, repeat_count + 1)
        )
    )
    checks.add(
        "raw_results.complete_matrix",
        complete_matrix,
        "raw trials cover every declared task, domain, repeat, and agent exactly once",
    )

    counter_fields = tuple(next(iter(raw_counts.values())))
    metric_binding = complete_matrix and all(
        all(record.get(field) == raw_counts[name][field] for field in counter_fields)
        for name, record in agent_records.items()
    )
    checks.add(
        "raw_results.metric_binding",
        metric_binding,
        "raw outcomes reproduce all aggregate success, failure, timeout, and violation counters",
    )

    candidate_name = _normalized_name(str(candidate.get("agent_name") or ""))
    comparison_binding = complete_matrix and isinstance(comparisons, list) and bool(comparisons)
    if comparison_binding:
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                comparison_binding = False
                break
            baseline_name = _normalized_name(str(comparison.get("baseline_agent_name") or ""))
            if baseline_name not in agent_records or baseline_name == candidate_name:
                comparison_binding = False
                break
            paired = {
                "both_passed": 0,
                "candidate_only_passed": 0,
                "baseline_only_passed": 0,
                "both_failed": 0,
            }
            for task_id in task_domains:
                for repeat in range(1, repeat_count + 1):
                    candidate_passed = records[(candidate_name, task_id, repeat)] == "passed"
                    baseline_passed = records[(baseline_name, task_id, repeat)] == "passed"
                    if candidate_passed and baseline_passed:
                        paired["both_passed"] += 1
                    elif candidate_passed:
                        paired["candidate_only_passed"] += 1
                    elif baseline_passed:
                        paired["baseline_only_passed"] += 1
                    else:
                        paired["both_failed"] += 1
            comparison_binding = comparison_binding and comparison.get("total_pairs") == total_trials and all(
                comparison.get(field) == value for field, value in paired.items()
            )
    checks.add(
        "raw_results.paired_binding",
        comparison_binding,
        "raw trial pairs reproduce every candidate-to-baseline contingency table",
    )


def _validate_campaign(
    checks: _Checks,
    *,
    campaign_reference: Any,
    artifact_records: Mapping[str, tuple[dict[str, Any], Path]],
    subject: Mapping[str, Any] | None,
    policy: CertificationPolicy,
) -> None:
    reference_ok = isinstance(campaign_reference, dict)
    checks.add("self_upgrade_reference.object", reference_ok, "self-upgrade campaign reference is an object")
    if not reference_ok:
        return
    assert isinstance(campaign_reference, dict)
    _exact_keys(
        checks,
        "self_upgrade_reference.fields",
        campaign_reference,
        {"artifact_role", "manifest_schema", "manifest_sha256"},
    )
    artifact_tuple = artifact_records.get("self_upgrade_campaign")
    reference_matches = (
        campaign_reference.get("artifact_role") == "self_upgrade_campaign"
        and campaign_reference.get("manifest_schema") == CAMPAIGN_SCHEMA
        and isinstance(campaign_reference.get("manifest_sha256"), str)
        and SHA256_RE.fullmatch(campaign_reference["manifest_sha256"]) is not None
        and artifact_tuple is not None
        and artifact_tuple[0].get("sha256") == campaign_reference.get("manifest_sha256")
    )
    checks.add(
        "self_upgrade_reference.artifact_binding",
        reference_matches,
        "campaign reference matches the verified campaign artifact",
    )
    if artifact_tuple is None:
        return
    try:
        campaign = _strict_json_loads(artifact_tuple[1].read_bytes())
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        checks.add("self_upgrade_campaign.json", False, "campaign artifact is not strict JSON")
        return
    if not checks.add("self_upgrade_campaign.json", isinstance(campaign, dict), "campaign artifact is a JSON object"):
        return
    assert isinstance(campaign, dict)
    _exact_keys(
        checks,
        "self_upgrade_campaign.fields",
        campaign,
        {
            "schema",
            "campaign_id",
            "created_at",
            "subject",
            "evaluator_lock",
            "candidate_generation_started_at",
            "baseline",
            "candidates",
            "selection",
            "promotion",
            "rollback",
            "attestation",
        },
    )
    checks.add(
        "self_upgrade_campaign.identity",
        campaign.get("schema") == CAMPAIGN_SCHEMA
        and isinstance(campaign.get("campaign_id"), str)
        and bool(campaign.get("campaign_id", "").strip()),
        "campaign schema and identifier are valid",
    )

    attestation = campaign.get("attestation")
    attestation_ok = (
        isinstance(attestation, dict)
        and set(attestation) == {"algorithm", "payload_sha256"}
        and attestation.get("algorithm") == "sha256"
        and isinstance(attestation.get("payload_sha256"), str)
        and SHA256_RE.fullmatch(attestation["payload_sha256"]) is not None
        and attestation["payload_sha256"] == compute_campaign_payload_sha256(campaign)
    )
    checks.add(
        "self_upgrade_campaign.attestation",
        attestation_ok,
        "campaign semantic payload SHA-256 matches independent recomputation",
    )

    campaign_subject = campaign.get("subject")
    subject_ok = isinstance(campaign_subject, dict) and isinstance(subject, Mapping)
    if subject_ok:
        assert isinstance(campaign_subject, dict)
        subject_ok = set(campaign_subject) == {
            "commit_sha",
            "tree_sha",
            "source_digest_algorithm",
            "source_digest_sha256",
        } and all(
            campaign_subject.get(field) == subject.get(field)
            for field in ("commit_sha", "tree_sha", "source_digest_algorithm", "source_digest_sha256")
        )
    checks.add(
        "self_upgrade_campaign.subject_binding",
        subject_ok,
        "campaign promotion binds the exact certified source",
    )

    created_at = _parse_utc_timestamp(campaign.get("created_at"))
    generation_started_at = _parse_utc_timestamp(campaign.get("candidate_generation_started_at"))
    evaluator_lock = campaign.get("evaluator_lock")
    lock_ok = isinstance(evaluator_lock, dict)
    locked_at: datetime | None = None
    if lock_ok:
        assert isinstance(evaluator_lock, dict)
        lock_ok = set(evaluator_lock) == {
            "evaluator_id",
            "evaluator_digest_sha256",
            "commands_digest_sha256",
            "files_digest_sha256",
            "locked_at",
            "locked_before_candidate_generation",
        }
        locked_at = _parse_utc_timestamp(evaluator_lock.get("locked_at"))
        lock_ok = lock_ok and all(
            isinstance(evaluator_lock.get(field), str)
            and SHA256_RE.fullmatch(evaluator_lock[field]) is not None
            for field in ("evaluator_digest_sha256", "commands_digest_sha256", "files_digest_sha256")
        )
        lock_ok = (
            lock_ok
            and isinstance(evaluator_lock.get("evaluator_id"), str)
            and bool(evaluator_lock.get("evaluator_id", "").strip())
            and evaluator_lock.get("locked_before_candidate_generation") is True
            and locked_at is not None
            and generation_started_at is not None
            and locked_at < generation_started_at
            and created_at is not None
            and generation_started_at <= created_at
        )
    checks.add(
        "self_upgrade_campaign.frozen_evaluator",
        lock_ok,
        "evaluator commands and files were hash-locked before candidate generation",
    )

    baseline = campaign.get("baseline")
    baseline_ok = isinstance(baseline, dict)
    if baseline_ok:
        assert isinstance(baseline, dict)
        baseline_ok = (
            set(baseline) == {"status", "commit_sha", "tree_sha", "metrics_digest_sha256"}
            and baseline.get("status") == "passed"
            and GIT_OID_RE.fullmatch(str(baseline.get("commit_sha", ""))) is not None
            and GIT_OID_RE.fullmatch(str(baseline.get("tree_sha", ""))) is not None
            and SHA256_RE.fullmatch(str(baseline.get("metrics_digest_sha256", ""))) is not None
        )
    checks.add("self_upgrade_campaign.baseline", baseline_ok, "campaign has a successful hash-bound baseline")

    candidates = campaign.get("candidates")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    candidates_ok = isinstance(candidates, list) and len(candidates) >= policy.minimum_upgrade_candidates
    if isinstance(candidates, list):
        candidate_keys = {
            "candidate_id",
            "patch_digest_sha256",
            "evaluation_digest_sha256",
            "status",
            "evaluator_files_modified",
            "strictly_improves_baseline",
        }
        for candidate in candidates:
            item_ok = isinstance(candidate, dict) and set(candidate) == candidate_keys
            if item_ok:
                assert isinstance(candidate, dict)
                candidate_id = candidate.get("candidate_id")
                item_ok = (
                    isinstance(candidate_id, str)
                    and bool(candidate_id.strip())
                    and candidate_id not in candidate_by_id
                    and SHA256_RE.fullmatch(str(candidate.get("patch_digest_sha256", ""))) is not None
                    and SHA256_RE.fullmatch(str(candidate.get("evaluation_digest_sha256", ""))) is not None
                    and candidate.get("status") in {"passed", "failed"}
                    and candidate.get("evaluator_files_modified") is False
                    and isinstance(candidate.get("strictly_improves_baseline"), bool)
                )
                if item_ok:
                    candidate_by_id[candidate_id] = candidate
            candidates_ok = candidates_ok and item_ok
    checks.add(
        "self_upgrade_campaign.candidates",
        candidates_ok,
        "multiple unique candidates were evaluated without modifying frozen evaluator files",
    )

    selection = campaign.get("selection")
    selection_ok = isinstance(selection, dict)
    selected_id = ""
    if selection_ok:
        assert isinstance(selection, dict)
        selected_id = str(selection.get("candidate_id", ""))
        selected = candidate_by_id.get(selected_id)
        selection_ok = (
            set(selection) == {"candidate_id", "decision", "strictly_improves_baseline"}
            and selection.get("decision") == "promoted"
            and selection.get("strictly_improves_baseline") is True
            and selected is not None
            and selected.get("status") == "passed"
            and selected.get("strictly_improves_baseline") is True
        )
    checks.add(
        "self_upgrade_campaign.selection",
        selection_ok,
        "only a passing, strictly improving candidate was selected",
    )

    promotion = campaign.get("promotion")
    promotion_ok = isinstance(promotion, dict) and isinstance(subject, Mapping)
    if promotion_ok:
        assert isinstance(promotion, dict)
        promotion_ok = (
            set(promotion)
            == {
                "candidate_id",
                "human_approved",
                "verified",
                "promoted_commit_sha",
                "promoted_tree_sha",
            }
            and promotion.get("candidate_id") == selected_id
            and promotion.get("human_approved") is True
            and promotion.get("verified") is True
            and promotion.get("promoted_commit_sha") == subject.get("commit_sha")
            and promotion.get("promoted_tree_sha") == subject.get("tree_sha")
        )
    checks.add(
        "self_upgrade_campaign.promotion",
        promotion_ok,
        "human-approved verified promotion matches the certified commit and tree",
    )

    rollback = campaign.get("rollback")
    rollback_ok = isinstance(rollback, dict) and isinstance(baseline, dict)
    if rollback_ok:
        assert isinstance(rollback, dict)
        rollback_ok = (
            set(rollback) == {"tested", "passed", "restored_tree_sha"}
            and rollback.get("tested") is True
            and rollback.get("passed") is True
            and rollback.get("restored_tree_sha") == baseline.get("tree_sha")
        )
    checks.add(
        "self_upgrade_campaign.rollback",
        rollback_ok,
        "rollback was tested and restored the baseline tree",
    )


def _result(
    checks: _Checks,
    *,
    policy: CertificationPolicy,
    report_sha256: str,
    evidence_id: str,
    repository_identity: RepositoryIdentity | None,
    artifact_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    failures = [item["id"] for item in checks.items if not item["passed"]]
    passed = checks.passed
    return {
        "schema": RESULT_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "release_allowed": passed,
        "evidence_id": evidence_id,
        "report_sha256": report_sha256,
        "repository_identity": repository_identity.to_dict() if repository_identity is not None else None,
        "artifact_bindings": {
            str(role): dict(binding)
            for role, binding in sorted((artifact_bindings or {}).items())
        },
        "policy": {
            "expected_suite_id": policy.expected_suite_id,
            "expected_suite_sha256": policy.expected_suite_sha256,
            "expected_evaluator_id": policy.expected_evaluator_id,
            "expected_evaluator_sha256": policy.expected_evaluator_sha256,
            "required_baseline_agents": list(policy.required_baseline_agents),
            "minimum_hidden_tasks": policy.minimum_hidden_tasks,
            "minimum_domains": policy.minimum_domains,
            "minimum_tasks_per_domain": policy.minimum_tasks_per_domain,
            "minimum_repeats": policy.minimum_repeats,
            "minimum_candidate_success_rate": policy.minimum_candidate_success_rate,
            "minimum_candidate_wilson_lower_bound": policy.minimum_candidate_wilson_lower_bound,
            "confidence_level": policy.confidence_level,
            "baseline_noninferiority_margin": policy.baseline_noninferiority_margin,
            "maximum_timeout_rate": policy.maximum_timeout_rate,
            "minimum_upgrade_candidates": policy.minimum_upgrade_candidates,
            "maximum_evidence_age_days": policy.maximum_evidence_age_days,
            "required_artifact_roles": list(policy.required_artifact_roles),
        },
        "checks_passed": sum(1 for item in checks.items if item["passed"]),
        "checks_total": len(checks.items),
        "failures": failures,
        "checks": checks.items,
    }


def verify_capability_certification(
    report_path: Path,
    *,
    repository: Path,
    policy: CertificationPolicy,
    artifact_root: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify external capability evidence and return a deterministic gate result.

    ``expected_report_sha256`` is intentionally supplied by the caller.  It is
    the trust anchor obtained out of band from the external evaluator; accepting
    a digest stored only inside the report would not detect intentional edits.
    """

    checks = _Checks()
    report_path = report_path.resolve()
    artifact_root = (artifact_root or report_path.parent).resolve()
    report_sha256 = ""
    evidence_id = ""
    repository_identity: RepositoryIdentity | None = None

    try:
        raw_report = report_path.read_bytes()
    except OSError:
        checks.add("report.readable", False, "external certification report is missing or unreadable")
        return _result(
            checks,
            policy=policy,
            report_sha256=report_sha256,
            evidence_id=evidence_id,
            repository_identity=repository_identity,
        )
    checks.add("report.readable", True, "external certification report is readable")
    report_sha256 = hashlib.sha256(raw_report).hexdigest()
    checks.add(
        "report.external_digest",
        report_sha256 == policy.expected_report_sha256,
        "raw report SHA-256 matches the out-of-band trust anchor",
    )
    try:
        report = _strict_json_loads(raw_report)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        checks.add("report.strict_json", False, "report is not unambiguous strict JSON")
        return _result(
            checks,
            policy=policy,
            report_sha256=report_sha256,
            evidence_id=evidence_id,
            repository_identity=repository_identity,
        )
    if not checks.add("report.strict_json", isinstance(report, dict), "report is a JSON object"):
        return _result(
            checks,
            policy=policy,
            report_sha256=report_sha256,
            evidence_id=evidence_id,
            repository_identity=repository_identity,
        )
    assert isinstance(report, dict)
    _exact_keys(
        checks,
        "report.fields",
        report,
        {
            "schema",
            "evidence_id",
            "issued_at",
            "expires_at",
            "evaluation_completed_at",
            "subject",
            "external_evaluation",
            "self_upgrade_campaign",
            "artifacts",
            "attestation",
        },
    )
    evidence_id = report.get("evidence_id") if isinstance(report.get("evidence_id"), str) else ""
    checks.add(
        "report.identity",
        report.get("schema") == EVIDENCE_SCHEMA and bool(evidence_id.strip()),
        "report schema and evidence identifier are valid",
    )

    attestation = report.get("attestation")
    attestation_ok = (
        isinstance(attestation, dict)
        and set(attestation) == {"algorithm", "payload_sha256"}
        and attestation.get("algorithm") == "sha256"
        and isinstance(attestation.get("payload_sha256"), str)
        and SHA256_RE.fullmatch(attestation["payload_sha256"]) is not None
        and attestation["payload_sha256"] == compute_evidence_payload_sha256(report)
    )
    checks.add(
        "report.semantic_attestation",
        attestation_ok,
        "semantic payload SHA-256 matches independent recomputation",
    )

    effective_now = now or datetime.now(timezone.utc)
    if effective_now.tzinfo is None or effective_now.utcoffset() is None:
        raise CertificationInputError("now must be timezone-aware")
    effective_now = effective_now.astimezone(timezone.utc)
    issued_at = _parse_utc_timestamp(report.get("issued_at"))
    expires_at = _parse_utc_timestamp(report.get("expires_at"))
    completed_at = _parse_utc_timestamp(report.get("evaluation_completed_at"))
    timestamps_ok = issued_at is not None and expires_at is not None and completed_at is not None
    checks.add("report.timestamps", timestamps_ok, "report timestamps use canonical UTC seconds")
    if timestamps_ok:
        assert issued_at is not None and expires_at is not None and completed_at is not None
        max_age = timedelta(days=policy.maximum_evidence_age_days)
        chronology_ok = (
            completed_at <= issued_at <= effective_now <= expires_at
            and effective_now - completed_at <= max_age
            and expires_at - issued_at <= max_age
        )
        checks.add(
            "report.freshness",
            chronology_ok,
            "evaluation is current, issued after completion, and not expired or overlong",
        )
    else:
        checks.add("report.freshness", False, "report chronology cannot be established")

    try:
        repository_identity = compute_repository_identity(repository)
    except (CertificationInputError, UnicodeError):
        checks.add("repository.identity", False, "repository identity could not be independently computed")
    else:
        checks.add("repository.identity", True, "repository commit, tree, and source digest were recomputed")
        checks.add(
            "repository.clean",
            repository_identity.clean,
            "release repository has no tracked, staged, or untracked changes",
        )

    subject = report.get("subject")
    subject_mapping: Mapping[str, Any] | None = subject if isinstance(subject, dict) else None
    if checks.add("subject.object", subject_mapping is not None, "certified subject is an object"):
        assert isinstance(subject_mapping, dict)
        _exact_keys(
            checks,
            "subject.fields",
            subject_mapping,
            {
                "repository_url",
                "commit_sha",
                "tree_sha",
                "source_digest_algorithm",
                "source_digest_sha256",
                "source_archive_sha256",
                "worktree_clean",
            },
        )
        subject_format_ok = (
            isinstance(subject_mapping.get("repository_url"), str)
            and bool(subject_mapping.get("repository_url", "").strip())
            and isinstance(subject_mapping.get("commit_sha"), str)
            and GIT_OID_RE.fullmatch(subject_mapping["commit_sha"]) is not None
            and isinstance(subject_mapping.get("tree_sha"), str)
            and GIT_OID_RE.fullmatch(subject_mapping["tree_sha"]) is not None
            and subject_mapping.get("source_digest_algorithm") == SOURCE_DIGEST_ALGORITHM
            and isinstance(subject_mapping.get("source_digest_sha256"), str)
            and SHA256_RE.fullmatch(subject_mapping["source_digest_sha256"]) is not None
            and isinstance(subject_mapping.get("source_archive_sha256"), str)
            and SHA256_RE.fullmatch(subject_mapping["source_archive_sha256"]) is not None
            and subject_mapping.get("worktree_clean") is True
        )
        checks.add("subject.format", subject_format_ok, "subject identity and source digests are well formed")
        identity_matches = repository_identity is not None and all(
            subject_mapping.get(field) == getattr(repository_identity, field)
            for field in ("commit_sha", "tree_sha", "source_digest_sha256")
        )
        checks.add(
            "subject.repository_binding",
            identity_matches,
            "report binds the exact current commit, tree, and committed source bytes",
        )

    external_evaluation = report.get("external_evaluation")
    _validate_external_evaluation(checks, external_evaluation, policy)
    artifacts = _validate_artifacts(
        checks,
        report.get("artifacts"),
        artifact_root=artifact_root,
        policy=policy,
    )
    _validate_raw_results(
        checks,
        artifact=artifacts.get("benchmark_raw_results"),
        evaluation=external_evaluation,
    )
    source_artifact = artifacts.get("source_archive")
    checks.add(
        "subject.source_archive_binding",
        source_artifact is not None
        and subject_mapping is not None
        and source_artifact[0].get("sha256") == subject_mapping.get("source_archive_sha256"),
        "certified source archive digest matches the subject",
    )
    _validate_campaign(
        checks,
        campaign_reference=report.get("self_upgrade_campaign"),
        artifact_records=artifacts,
        subject=subject_mapping,
        policy=policy,
    )
    artifact_bindings = {
        role: {
            "sha256": str(record.get("sha256") or ""),
            "size_bytes": int(record.get("size_bytes") or 0),
        }
        for role, (record, _path) in artifacts.items()
    }
    return _result(
        checks,
        policy=policy,
        report_sha256=report_sha256,
        evidence_id=evidence_id,
        repository_identity=repository_identity,
        artifact_bindings=artifact_bindings,
    )


def deterministic_json(result: Mapping[str, Any]) -> str:
    """Serialize a certification result with stable keys and newline."""

    return json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "CAMPAIGN_SCHEMA",
    "EVIDENCE_SCHEMA",
    "RAW_TRIAL_SCHEMA",
    "RESULT_SCHEMA",
    "SOURCE_DIGEST_ALGORITHM",
    "CertificationInputError",
    "CertificationPolicy",
    "RepositoryIdentity",
    "compute_campaign_payload_sha256",
    "compute_evidence_payload_sha256",
    "compute_repository_identity",
    "compute_source_digest",
    "deterministic_json",
    "paired_difference_lower_bound",
    "sha256_file",
    "verify_capability_certification",
    "wilson_lower_bound",
]
