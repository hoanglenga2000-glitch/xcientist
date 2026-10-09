"""Strict evidence verifier for an externally executed agent-parity campaign.

This module deliberately does not launch paid agents and does not treat rows in
``benchmark_runs`` as capability evidence.  It verifies a complete, held-out
trial matrix produced by an independent evaluator and re-reads every oracle and
artifact evidence file before computing Wilson and paired non-inferiority
statistics.

The final release certificate remains the responsibility of
``xsci.capability_certification`` because only that verifier binds an
out-of-band evaluator/suite digest to clean release source bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any, Iterable, Mapping

SUITE_SCHEMA = "evomind.parity_hidden_suite.v1"
TRIAL_SCHEMA = "evomind.parity_trial_evidence.v1"
RESULT_SCHEMA = "evomind.parity_campaign_verification.v1"
REQUIRED_AGENTS = ("evomind", "codex", "claude-code")
MINIMUM_HIDDEN_TASKS = 100
MINIMUM_DOMAINS = 8
MINIMUM_TASKS_PER_DOMAIN = 3
REQUIRED_REPETITIONS = 3
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_CANDIDATE_SUCCESS_RATE = 0.80
DEFAULT_CANDIDATE_WILSON_LOWER_BOUND = 0.75
DEFAULT_NONINFERIORITY_MARGIN = 0.05
SHA256_HEX = frozenset("0123456789abcdef")

_SUITE_KEYS = {
    "schema",
    "suite_id",
    "version",
    "held_out",
    "selection_locked_before_evaluation",
    "tasks_not_used_for_development",
    "tasks",
}
_TASK_KEYS = {"task_id", "domain", "prompt_sha256"}
_TRIAL_KEYS = {
    "schema",
    "campaign_id",
    "suite_id",
    "task_id",
    "domain",
    "repetition",
    "agent",
    "agent_version",
    "provider",
    "model",
    "outcome",
    "timed_out",
    "duration_seconds",
    "scope_violation",
    "unsupported_claims",
    "tool_calls",
    "oracle",
    "artifacts",
}
_TOOL_CALL_KEYS = {"total", "succeeded", "failed"}
_ORACLE_KEYS = {"id", "passed", "evidence_path", "evidence_sha256", "evidence_bytes"}
_ARTIFACT_KEYS = {"role", "path", "sha256", "bytes"}
_REQUIRED_ARTIFACT_ROLES = {"workspace_result", "tool_trace"}
_ORACLE_EVIDENCE_KEYS = {
    "schema",
    "campaign_id",
    "suite_id",
    "task_id",
    "domain",
    "repetition",
    "agent",
    "oracle_id",
    "passed",
    "fixture_sha256",
    "workspace_snapshot_sha256",
}
_TOOL_TRACE_KEYS = {"tool_call_id", "tool_name", "status"}
ORACLE_EVIDENCE_SCHEMA = "evomind.private_oracle_evidence.v1"


class ParityEvidenceError(ValueError):
    """Raised when the campaign evidence is malformed or incomplete."""


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA256_HEX
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)


def _relative_evidence_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ParityEvidenceError(f"{label} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ParityEvidenceError(f"{label} escapes the evidence root")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ParityEvidenceError(f"{label} is unavailable or outside the evidence root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ParityEvidenceError(f"{label} must identify a regular non-symlink file")
    return resolved


def _verify_file_record(root: Path, record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != _ARTIFACT_KEYS:
        raise ParityEvidenceError(f"{label} must contain exactly {sorted(_ARTIFACT_KEYS)}")
    path = _relative_evidence_file(root, record.get("path"), label=f"{label}.path")
    size = path.stat().st_size
    digest = _sha256_file(path)
    if not _is_int(record.get("bytes")) or record["bytes"] < 0 or record["bytes"] != size:
        raise ParityEvidenceError(f"{label} byte count does not match the file")
    if not _is_sha256(record.get("sha256")) or record["sha256"] != digest:
        raise ParityEvidenceError(f"{label} SHA-256 does not match the file")
    role = record.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ParityEvidenceError(f"{label} role must be non-empty")
    return {"role": role, "path": record["path"], "bytes": size, "sha256": digest}


def _verify_oracle_binding(path: Path, row: Mapping[str, Any], oracle: Mapping[str, Any], *, label: str) -> None:
    try:
        evidence = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ParityEvidenceError(f"{label} is not strict UTF-8 JSON") from exc
    expected = {
        "schema": ORACLE_EVIDENCE_SCHEMA,
        "campaign_id": row["campaign_id"],
        "suite_id": row["suite_id"],
        "task_id": row["task_id"],
        "domain": row["domain"],
        "repetition": row["repetition"],
        "agent": row["agent"],
        "oracle_id": oracle["id"],
        "passed": oracle["passed"],
    }
    if not isinstance(evidence, dict) or set(evidence) != _ORACLE_EVIDENCE_KEYS:
        raise ParityEvidenceError(f"{label} has an invalid exact field set")
    if any(evidence.get(field) != value for field, value in expected.items()):
        raise ParityEvidenceError(f"{label} does not bind the exact trial and oracle result")
    for field in ("fixture_sha256", "workspace_snapshot_sha256"):
        if not _is_sha256(evidence.get(field)):
            raise ParityEvidenceError(f"{label} has an invalid {field}")


def _verify_tool_trace(path: Path, expected: Mapping[str, Any], *, label: str) -> None:
    seen_ids: set[str] = set()
    succeeded = failed = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip() or len(raw) > 64 * 1024:
                    raise ParityEvidenceError(f"{label} row {line_number} is blank or oversized")
                item = _strict_json_loads(raw.decode("utf-8"))
                if not isinstance(item, dict) or set(item) != _TOOL_TRACE_KEYS:
                    raise ParityEvidenceError(f"{label} row {line_number} has an invalid exact field set")
                call_id = item.get("tool_call_id")
                tool_name = item.get("tool_name")
                status = item.get("status")
                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or call_id in seen_ids
                    or not isinstance(tool_name, str)
                    or not tool_name.strip()
                    or status not in {"succeeded", "failed"}
                ):
                    raise ParityEvidenceError(f"{label} row {line_number} has an invalid call identity/outcome")
                seen_ids.add(call_id)
                succeeded += int(status == "succeeded")
                failed += int(status == "failed")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ParityEvidenceError):
            raise
        raise ParityEvidenceError(f"{label} is not strict UTF-8 JSONL") from exc
    actual = {"total": len(seen_ids), "succeeded": succeeded, "failed": failed}
    if actual != dict(expected):
        raise ParityEvidenceError(f"{label} does not reproduce the declared tool-call metrics")


def load_suite_manifest(path: Path | str) -> tuple[dict[str, Any], str]:
    """Load a public held-out manifest without exposing oracle answers."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        manifest = _strict_json_loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ParityEvidenceError("suite manifest is not strict UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _SUITE_KEYS:
        raise ParityEvidenceError(f"suite manifest must contain exactly {sorted(_SUITE_KEYS)}")
    if manifest.get("schema") != SUITE_SCHEMA:
        raise ParityEvidenceError("suite manifest schema mismatch")
    for field in ("suite_id", "version"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ParityEvidenceError(f"suite {field} must be non-empty")
    if not all(
        manifest.get(field) is True
        for field in ("held_out", "selection_locked_before_evaluation", "tasks_not_used_for_development")
    ):
        raise ParityEvidenceError("suite must be held out, selection-locked, and unused for development")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < MINIMUM_HIDDEN_TASKS:
        raise ParityEvidenceError(f"suite must contain at least {MINIMUM_HIDDEN_TASKS} hidden tasks")
    task_ids: set[str] = set()
    domain_counts: Counter[str] = Counter()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != _TASK_KEYS:
            raise ParityEvidenceError(f"suite task {index} has an invalid public descriptor")
        task_id = task.get("task_id")
        domain = task.get("domain")
        if not isinstance(task_id, str) or not task_id.strip() or task_id in task_ids:
            raise ParityEvidenceError(f"suite task {index} has an invalid or duplicate task_id")
        if not isinstance(domain, str) or not domain.strip():
            raise ParityEvidenceError(f"suite task {index} has an invalid domain")
        if not _is_sha256(task.get("prompt_sha256")):
            raise ParityEvidenceError(f"suite task {index} has an invalid prompt SHA-256")
        task_ids.add(task_id)
        domain_counts[domain] += 1
    if len(domain_counts) < MINIMUM_DOMAINS:
        raise ParityEvidenceError(f"suite must cover at least {MINIMUM_DOMAINS} domains")
    if min(domain_counts.values()) < MINIMUM_TASKS_PER_DOMAIN:
        raise ParityEvidenceError(f"every domain must contain at least {MINIMUM_TASKS_PER_DOMAIN} tasks")
    return manifest, hashlib.sha256(raw).hexdigest()


def load_trial_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load bounded, strict JSONL without accepting blank or duplicate-key rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or len(raw) > 256 * 1024:
                raise ParityEvidenceError(f"trial row {line_number} is blank or oversized")
            try:
                row = _strict_json_loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise ParityEvidenceError(f"trial row {line_number} is not strict UTF-8 JSON") from exc
            if not isinstance(row, dict):
                raise ParityEvidenceError(f"trial row {line_number} is not an object")
            rows.append(row)
            if len(rows) > 2_000_000:
                raise ParityEvidenceError("trial matrix exceeds the bounded verifier limit")
    return rows


def wilson_lower_bound(successes: int, attempts: int, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL) -> float:
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
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> float:
    if total <= 0 or min(candidate_only, baseline_only) < 0 or candidate_only + baseline_only > total:
        raise ValueError("invalid paired counts")
    mean = (candidate_only - baseline_only) / total
    if total == 1:
        return mean
    sample_variance = max(
        0.0,
        (candidate_only + baseline_only - total * mean * mean) / (total - 1),
    )
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return mean - z * math.sqrt(sample_variance / total)


def verify_campaign(
    suite_manifest: Mapping[str, Any],
    suite_manifest_sha256: str,
    trial_rows: Iterable[Mapping[str, Any]],
    *,
    evidence_root: Path | str,
    required_agents: tuple[str, ...] = REQUIRED_AGENTS,
    candidate_agent: str = "evomind",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    minimum_candidate_success_rate: float = DEFAULT_CANDIDATE_SUCCESS_RATE,
    minimum_candidate_wilson_lower_bound: float = DEFAULT_CANDIDATE_WILSON_LOWER_BOUND,
    noninferiority_margin: float = DEFAULT_NONINFERIORITY_MARGIN,
) -> dict[str, Any]:
    """Verify a complete 100 x 3 x 3 campaign from real evidence files.

    A failed validation returns ``NO-GO`` with explicit blockers.  It never
    upgrades a local campaign to an external release certificate.
    """

    root = Path(evidence_root).resolve(strict=True)
    tasks = suite_manifest.get("tasks") if isinstance(suite_manifest, Mapping) else None
    if not isinstance(tasks, list):
        raise ParityEvidenceError("suite manifest tasks are unavailable")
    suite_id = suite_manifest.get("suite_id")
    task_domains = {
        str(task["task_id"]): str(task["domain"])
        for task in tasks
        if isinstance(task, Mapping) and "task_id" in task and "domain" in task
    }
    normalized_agents = tuple(_normalized(name) for name in required_agents)
    if len(set(normalized_agents)) != len(normalized_agents) or _normalized(candidate_agent) not in normalized_agents:
        raise ParityEvidenceError("required agents must be unique and include the candidate")
    expected = {
        (agent, task_id, repetition)
        for agent in normalized_agents
        for task_id in task_domains
        for repetition in range(1, REQUIRED_REPETITIONS + 1)
    }
    observed: dict[tuple[str, str, int], dict[str, Any]] = {}
    blockers: list[str] = []
    campaign_ids: set[str] = set()
    evidence_files: set[str] = set()
    primary_evidence_paths: set[str] = set()
    agent_identity: dict[str, tuple[str, str, str]] = {}

    for index, raw_row in enumerate(trial_rows, start=1):
        label = f"trial[{index}]"
        if not isinstance(raw_row, Mapping) or set(raw_row) != _TRIAL_KEYS:
            blockers.append(f"{label}: invalid exact field set")
            continue
        row = dict(raw_row)
        task_id = row.get("task_id")
        domain = row.get("domain")
        repetition = row.get("repetition")
        agent = _normalized(str(row.get("agent") or ""))
        outcome = row.get("outcome")
        campaign_id = row.get("campaign_id")
        if isinstance(campaign_id, str) and campaign_id.strip():
            campaign_ids.add(campaign_id)
        key = (agent, str(task_id), int(repetition) if _is_int(repetition) else 0)
        structural_ok = (
            row.get("schema") == TRIAL_SCHEMA
            and row.get("suite_id") == suite_id
            and isinstance(campaign_id, str)
            and bool(campaign_id.strip())
            and task_id in task_domains
            and domain == task_domains.get(str(task_id))
            and _is_int(repetition)
            and 1 <= repetition <= REQUIRED_REPETITIONS
            and agent in normalized_agents
            and outcome in {"passed", "failed", "timeout"}
            and isinstance(row.get("timed_out"), bool)
            and row["timed_out"] is (outcome == "timeout")
            and _is_number(row.get("duration_seconds"))
            and float(row["duration_seconds"]) >= 0.0
            and isinstance(row.get("scope_violation"), bool)
            and _is_int(row.get("unsupported_claims"))
            and row["unsupported_claims"] >= 0
        )
        if not structural_ok:
            blockers.append(f"{label}: invalid trial identity or outcome")
            continue
        identity = (str(row.get("agent_version") or ""), str(row.get("provider") or ""), str(row.get("model") or ""))
        if not all(value.strip() for value in identity):
            blockers.append(f"{label}: agent version/provider/model identity is incomplete")
            continue
        previous_identity = agent_identity.setdefault(agent, identity)
        if previous_identity != identity:
            blockers.append(f"{label}: agent identity drifted within the campaign")
            continue
        calls = row.get("tool_calls")
        if not isinstance(calls, Mapping) or set(calls) != _TOOL_CALL_KEYS:
            blockers.append(f"{label}: tool-call metrics are missing")
            continue
        if not all(_is_int(calls.get(field)) and calls[field] >= 0 for field in _TOOL_CALL_KEYS):
            blockers.append(f"{label}: tool-call counters are invalid")
            continue
        if calls["total"] < 1 or calls["succeeded"] + calls["failed"] != calls["total"]:
            blockers.append(f"{label}: tool-call counters do not partition a real tool trace")
            continue
        oracle = row.get("oracle")
        if not isinstance(oracle, Mapping) or set(oracle) != _ORACLE_KEYS:
            blockers.append(f"{label}: oracle evidence is missing")
            continue
        try:
            oracle_record = {
                "role": "oracle_evidence",
                "path": oracle.get("evidence_path"),
                "sha256": oracle.get("evidence_sha256"),
                "bytes": oracle.get("evidence_bytes"),
            }
            verified_oracle = _verify_file_record(root, oracle_record, label=f"{label}.oracle")
            oracle_path = _relative_evidence_file(root, verified_oracle["path"], label=f"{label}.oracle.path")
            _verify_oracle_binding(oracle_path, row, oracle, label=f"{label}.oracle")
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ParityEvidenceError(f"{label}.artifacts must contain real evidence files")
            verified_artifacts = [
                _verify_file_record(root, item, label=f"{label}.artifacts[{artifact_index}]")
                for artifact_index, item in enumerate(artifacts)
            ]
            roles = [item["role"] for item in verified_artifacts]
            if len(set(roles)) != len(roles) or not _REQUIRED_ARTIFACT_ROLES <= set(roles):
                raise ParityEvidenceError(
                    f"{label}.artifacts must contain unique roles including {sorted(_REQUIRED_ARTIFACT_ROLES)}"
                )
            trace = next(item for item in verified_artifacts if item["role"] == "tool_trace")
            trace_path = _relative_evidence_file(root, trace["path"], label=f"{label}.tool_trace.path")
            _verify_tool_trace(trace_path, calls, label=f"{label}.tool_trace")
            unique_paths = {verified_oracle["path"], trace["path"]}
            if primary_evidence_paths & unique_paths:
                raise ParityEvidenceError(f"{label} reuses another trial's oracle or tool trace")
        except ParityEvidenceError as exc:
            blockers.append(str(exc))
            continue
        if not isinstance(oracle.get("id"), str) or not oracle["id"].strip() or not isinstance(oracle.get("passed"), bool):
            blockers.append(f"{label}: oracle identity/result is invalid")
            continue
        if oracle["passed"] is not (outcome == "passed"):
            blockers.append(f"{label}: outcome does not match the private oracle result")
            continue
        if key in observed:
            blockers.append(f"{label}: duplicate agent/task/repetition cell")
            continue
        observed[key] = {
            **row,
            "agent": agent,
            "verified_oracle": verified_oracle,
            "verified_artifacts": verified_artifacts,
        }
        evidence_files.add(verified_oracle["path"])
        evidence_files.update(item["path"] for item in verified_artifacts)
        primary_evidence_paths.update(unique_paths)

    missing = expected - set(observed)
    unexpected = set(observed) - expected
    if len(campaign_ids) != 1:
        blockers.append("trial rows must bind exactly one campaign_id")
    if missing:
        blockers.append(f"complete trial matrix is missing {len(missing)} cells")
    if unexpected:
        blockers.append(f"trial matrix contains {len(unexpected)} unexpected cells")
    if len(tasks) < MINIMUM_HIDDEN_TASKS:
        blockers.append(f"suite has fewer than {MINIMUM_HIDDEN_TASKS} tasks")
    domain_counts = Counter(task_domains.values())
    if len(domain_counts) < MINIMUM_DOMAINS or (domain_counts and min(domain_counts.values()) < MINIMUM_TASKS_PER_DOMAIN):
        blockers.append("suite domain coverage is below the locked minimum")
    if not _is_sha256(suite_manifest_sha256):
        blockers.append("suite manifest SHA-256 is invalid")

    summaries: dict[str, dict[str, Any]] = {}
    for agent in normalized_agents:
        cells = [row for (name, _task, _repeat), row in observed.items() if name == agent]
        attempts = len(cells)
        successes = sum(row["outcome"] == "passed" for row in cells)
        timeouts = sum(bool(row["timed_out"]) for row in cells)
        scope_violations = sum(bool(row["scope_violation"]) for row in cells)
        unsupported = sum(int(row["unsupported_claims"]) for row in cells)
        calls_total = sum(int(row["tool_calls"]["total"]) for row in cells)
        calls_succeeded = sum(int(row["tool_calls"]["succeeded"]) for row in cells)
        summaries[agent] = {
            "attempted_trials": attempts,
            "successful_trials": successes,
            "failed_trials": attempts - successes,
            "timed_out_trials": timeouts,
            "scope_violations": scope_violations,
            "unsupported_claims": unsupported,
            "success_rate": successes / attempts if attempts else None,
            "wilson_lower_bound": wilson_lower_bound(successes, attempts, confidence_level) if attempts else None,
            "tool_calls_total": calls_total,
            "tool_calls_succeeded": calls_succeeded,
            "tool_call_success_rate": calls_succeeded / calls_total if calls_total else None,
            "mean_duration_seconds": (
                sum(float(row["duration_seconds"]) for row in cells) / attempts if attempts else None
            ),
            "identity": {
                "agent_version": agent_identity.get(agent, (None, None, None))[0],
                "provider": agent_identity.get(agent, (None, None, None))[1],
                "model": agent_identity.get(agent, (None, None, None))[2],
            },
        }
        if scope_violations:
            blockers.append(f"{agent} has {scope_violations} scope violations")
        if unsupported:
            blockers.append(f"{agent} has {unsupported} unsupported claims")

    candidate = _normalized(candidate_agent)
    candidate_summary = summaries[candidate]
    if (
        candidate_summary["success_rate"] is None
        or candidate_summary["success_rate"] < minimum_candidate_success_rate
        or candidate_summary["wilson_lower_bound"] is None
        or candidate_summary["wilson_lower_bound"] < minimum_candidate_wilson_lower_bound
    ):
        blockers.append("candidate does not satisfy absolute success/Wilson thresholds")

    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in normalized_agents:
        if baseline == candidate:
            continue
        both = candidate_only = baseline_only = neither = 0
        for task_id in task_domains:
            for repetition in range(1, REQUIRED_REPETITIONS + 1):
                candidate_row = observed.get((candidate, task_id, repetition))
                baseline_row = observed.get((baseline, task_id, repetition))
                candidate_passed = bool(candidate_row and candidate_row["outcome"] == "passed")
                baseline_passed = bool(baseline_row and baseline_row["outcome"] == "passed")
                if candidate_passed and baseline_passed:
                    both += 1
                elif candidate_passed:
                    candidate_only += 1
                elif baseline_passed:
                    baseline_only += 1
                else:
                    neither += 1
        total = both + candidate_only + baseline_only + neither
        lower = paired_difference_lower_bound(candidate_only, baseline_only, total, confidence_level)
        comparisons[baseline] = {
            "total_pairs": total,
            "both_passed": both,
            "candidate_only_passed": candidate_only,
            "baseline_only_passed": baseline_only,
            "both_failed": neither,
            "candidate_minus_baseline_rate": (candidate_only - baseline_only) / total,
            "difference_lower_bound": lower,
            "noninferiority_margin": noninferiority_margin,
            "noninferior": lower >= -noninferiority_margin,
        }
        if lower < -noninferiority_margin:
            blockers.append(f"candidate is not noninferior to {baseline}")

    matrix_complete = not missing and not unexpected and len(observed) == len(expected)
    evidence_projection_ready = not blockers and matrix_complete
    return {
        "schema": RESULT_SCHEMA,
        "status": "evidence_projection_ready" if evidence_projection_ready else "no_go",
        "release_gate": "NO-GO",
        "release_gate_reason": (
            "requires out-of-band suite/evaluator trust anchors and capability_certification source binding"
        ),
        "campaign_id": next(iter(campaign_ids)) if len(campaign_ids) == 1 else None,
        "suite_id": suite_id,
        "suite_manifest_sha256": suite_manifest_sha256,
        "hidden_task_count": len(task_domains),
        "domain_task_counts": dict(sorted(domain_counts.items())),
        "required_repetitions": REQUIRED_REPETITIONS,
        "required_agents": list(required_agents),
        "expected_trials": len(expected),
        "verified_trials": len(observed),
        "matrix_complete": matrix_complete,
        "verified_evidence_files": len(evidence_files),
        "agent_summaries": summaries,
        "paired_comparisons": comparisons,
        "blockers": sorted(set(blockers)),
        "next_gate": "xsci.capability_certification.verify_capability_certification",
    }


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "MINIMUM_DOMAINS",
    "MINIMUM_HIDDEN_TASKS",
    "ORACLE_EVIDENCE_SCHEMA",
    "ParityEvidenceError",
    "REQUIRED_AGENTS",
    "REQUIRED_REPETITIONS",
    "RESULT_SCHEMA",
    "SUITE_SCHEMA",
    "TRIAL_SCHEMA",
    "deterministic_json",
    "load_suite_manifest",
    "load_trial_rows",
    "paired_difference_lower_bound",
    "verify_campaign",
    "wilson_lower_bound",
]
