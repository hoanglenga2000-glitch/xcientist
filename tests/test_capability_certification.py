from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

from scripts.verify_capability_certification import main as certification_cli
from xsci.capability_certification import (
    CAMPAIGN_SCHEMA,
    EVIDENCE_SCHEMA,
    RAW_TRIAL_SCHEMA,
    SOURCE_DIGEST_ALGORITHM,
    CertificationPolicy,
    compute_repository_identity,
    verify_capability_certification,
)

NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
SUITE_SHA = hashlib.sha256(b"independent-held-out-suite-v7").hexdigest()
EVALUATOR_SHA = hashlib.sha256(b"independent-evaluator-container-v4").hexdigest()


@dataclass(frozen=True)
class _Fixture:
    report_path: Path
    repository: Path
    artifact_root: Path
    policy: CertificationPolicy


def _canonical_digest(payload: dict[str, object], excluded: str) -> str:
    semantic = dict(payload)
    semantic.pop(excluded, None)
    data = json.dumps(
        semantic,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _wilson(successes: int, attempts: int, confidence: float = 0.95) -> float:
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / attempts
    z2 = z * z
    return (p + z2 / (2 * attempts) - z * math.sqrt((p * (1 - p) + z2 / (4 * attempts)) / attempts)) / (
        1 + z2 / attempts
    )


def _metric(name: str, trials: int) -> dict[str, object]:
    return {
        "agent_name": name,
        "agent_version": "2026.07",
        "provider": "independent-evaluation-provider",
        "model": f"{name.casefold().replace(' ', '-')}-pinned",
        "attempted_trials": trials,
        "successful_trials": trials,
        "failed_trials": 0,
        "timed_out_trials": 0,
        "scope_violations": 0,
        "unsupported_claims": 0,
        "success_rate": 1.0,
        "confidence_level": 0.95,
        "wilson_lower_bound": _wilson(trials, trials),
    }


def _comparison(name: str, trials: int) -> dict[str, object]:
    return {
        "baseline_agent_name": name,
        "total_pairs": trials,
        "both_passed": trials,
        "candidate_only_passed": 0,
        "baseline_only_passed": 0,
        "both_failed": 0,
        "candidate_minus_baseline_rate": 0.0,
        "confidence_level": 0.95,
        "difference_lower_bound": 0.0,
    }


def _artifact(path: Path, root: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _build_external_fixture(tmp_path: Path) -> _Fixture:
    repository = tmp_path / "subject-repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "External Test")
    _git(repository, "config", "user.email", "external@example.invalid")
    (repository / "agent.py").write_text("CAPABILITY_CONTRACT = 'r16'\n", encoding="utf-8", newline="\n")
    _git(repository, "add", "agent.py")
    _git(repository, "commit", "-q", "-m", "exact evaluated source")
    identity = compute_repository_identity(repository)

    artifact_root = tmp_path / "external-evidence"
    artifact_root.mkdir()
    source_archive = artifact_root / "xcientist-source.zip"
    source_archive.write_bytes(b"independently-built-source-archive\n")
    wheel = artifact_root / "xcientist-0.2.2-py3-none-any.whl"
    wheel.write_bytes(b"independently-built-wheel\n")
    sdist = artifact_root / "xcientist-0.2.2.tar.gz"
    sdist.write_bytes(b"independently-built-sdist\n")
    hidden_tasks = 104
    repeat_count = 3
    total_trials = hidden_tasks * repeat_count
    domains = [f"research-domain-{index}" for index in range(1, 9)]
    raw_results = artifact_root / "held-out-raw-results.jsonl"
    raw_rows = []
    for task_index in range(hidden_tasks):
        task_id = f"hidden-task-{task_index + 1:03d}"
        domain = domains[task_index // 13]
        for repeat in range(1, repeat_count + 1):
            for agent_name in ("EvoMind", "Claude Code", "Codex"):
                raw_rows.append(json.dumps({
                    "schema": RAW_TRIAL_SCHEMA,
                    "task_id": task_id,
                    "domain": domain,
                    "repeat": repeat,
                    "agent_name": agent_name,
                    "outcome": "passed",
                    "scope_violation": False,
                    "unsupported_claim": False,
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    raw_results.write_text("\n".join(raw_rows) + "\n", encoding="utf-8", newline="\n")

    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": "campaign-external-20260714-01",
        "created_at": "2026-07-14T01:00:00Z",
        "subject": {
            "commit_sha": identity.commit_sha,
            "tree_sha": identity.tree_sha,
            "source_digest_algorithm": SOURCE_DIGEST_ALGORITHM,
            "source_digest_sha256": identity.source_digest_sha256,
        },
        "evaluator_lock": {
            "evaluator_id": "upgrade-evaluator-v3",
            "evaluator_digest_sha256": hashlib.sha256(b"upgrade-evaluator").hexdigest(),
            "commands_digest_sha256": hashlib.sha256(b"locked-commands").hexdigest(),
            "files_digest_sha256": hashlib.sha256(b"locked-files").hexdigest(),
            "locked_at": "2026-07-14T00:00:00Z",
            "locked_before_candidate_generation": True,
        },
        "candidate_generation_started_at": "2026-07-14T00:10:00Z",
        "baseline": {
            "status": "passed",
            "commit_sha": "1" * 40,
            "tree_sha": "2" * 40,
            "metrics_digest_sha256": hashlib.sha256(b"baseline-metrics").hexdigest(),
        },
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "patch_digest_sha256": hashlib.sha256(b"patch-a").hexdigest(),
                "evaluation_digest_sha256": hashlib.sha256(b"evaluation-a").hexdigest(),
                "status": "passed",
                "evaluator_files_modified": False,
                "strictly_improves_baseline": True,
            },
            {
                "candidate_id": "candidate-b",
                "patch_digest_sha256": hashlib.sha256(b"patch-b").hexdigest(),
                "evaluation_digest_sha256": hashlib.sha256(b"evaluation-b").hexdigest(),
                "status": "failed",
                "evaluator_files_modified": False,
                "strictly_improves_baseline": False,
            },
        ],
        "selection": {
            "candidate_id": "candidate-a",
            "decision": "promoted",
            "strictly_improves_baseline": True,
        },
        "promotion": {
            "candidate_id": "candidate-a",
            "human_approved": True,
            "verified": True,
            "promoted_commit_sha": identity.commit_sha,
            "promoted_tree_sha": identity.tree_sha,
        },
        "rollback": {"tested": True, "passed": True, "restored_tree_sha": "2" * 40},
    }
    campaign["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _canonical_digest(campaign, "attestation"),
    }
    campaign_path = artifact_root / "self-upgrade-campaign.json"
    _write_json(campaign_path, campaign)

    artifacts = [
        _artifact(wheel, artifact_root, "wheel"),
        _artifact(sdist, artifact_root, "sdist"),
        _artifact(source_archive, artifact_root, "source_archive"),
        _artifact(raw_results, artifact_root, "benchmark_raw_results"),
        _artifact(campaign_path, artifact_root, "self_upgrade_campaign"),
    ]
    report = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_id": "external-evidence-20260714-01",
        "issued_at": "2026-07-14T02:05:00Z",
        "expires_at": "2026-07-30T02:05:00Z",
        "evaluation_completed_at": "2026-07-14T02:00:00Z",
        "subject": {
            "repository_url": "https://example.invalid/independent/xcientist",
            "commit_sha": identity.commit_sha,
            "tree_sha": identity.tree_sha,
            "source_digest_algorithm": SOURCE_DIGEST_ALGORITHM,
            "source_digest_sha256": identity.source_digest_sha256,
            "source_archive_sha256": _sha(source_archive),
            "worktree_clean": True,
        },
        "external_evaluation": {
            "suite": {
                "id": "external-research-agent-heldout",
                "version": "7.0.0",
                "digest_sha256": SUITE_SHA,
                "hidden_task_count": hidden_tasks,
                "domains": domains,
                "domain_task_counts": {domain: 13 for domain in domains},
                "repeat_count": repeat_count,
                "total_trials": total_trials,
                "held_out": True,
                "selection_locked_before_evaluation": True,
                "tasks_not_used_for_development": True,
            },
            "evaluator": {
                "id": "external-evaluator-service",
                "version": "4.1.0",
                "digest_sha256": EVALUATOR_SHA,
                "organization": "Independent Evaluation Lab",
                "independent": True,
                "execution_isolated": True,
                "all_trials_reported": True,
            },
            "timeout_policy": {
                "timeouts_count_as_failures": True,
                "max_timeout_rate": 0.0,
                "retry_on_timeout": False,
                "timeout_seconds": 900,
                "uniform_across_agents": True,
            },
            "candidate": _metric("EvoMind", total_trials),
            "baselines": [_metric("Claude Code", total_trials), _metric("Codex", total_trials)],
            "paired_comparisons": [
                _comparison("Claude Code", total_trials),
                _comparison("Codex", total_trials),
            ],
        },
        "self_upgrade_campaign": {
            "artifact_role": "self_upgrade_campaign",
            "manifest_schema": CAMPAIGN_SCHEMA,
            "manifest_sha256": _sha(campaign_path),
        },
        "artifacts": artifacts,
    }
    report["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _canonical_digest(report, "attestation"),
    }
    report_path = artifact_root / "capability-certification.json"
    _write_json(report_path, report)
    policy = CertificationPolicy(
        expected_report_sha256=_sha(report_path),
        expected_suite_id="external-research-agent-heldout",
        expected_suite_sha256=SUITE_SHA,
        expected_evaluator_id="external-evaluator-service",
        expected_evaluator_sha256=EVALUATOR_SHA,
        required_baseline_agents=("Claude Code", "Codex"),
    )
    return _Fixture(report_path, repository, artifact_root, policy)


def test_good_independently_generated_temp_report_passes_and_is_deterministic(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)

    first = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=fixture.policy,
        now=NOW,
    )
    second = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=fixture.policy,
        now=NOW,
    )

    assert first == second
    assert first["status"] == "PASS"
    assert first["release_allowed"] is True
    assert first["failures"] == []
    assert first["checks_passed"] == first["checks_total"]
    report = json.loads(fixture.report_path.read_text(encoding="utf-8"))
    campaign_artifact = next(item for item in report["artifacts"] if item["role"] == "self_upgrade_campaign")
    assert first["artifact_bindings"]["self_upgrade_campaign"] == {
        "sha256": campaign_artifact["sha256"],
        "size_bytes": campaign_artifact["size_bytes"],
    }


def test_missing_external_report_fails_closed(tmp_path: Path) -> None:
    policy = CertificationPolicy(
        expected_report_sha256="a" * 64,
        expected_suite_id="suite",
        expected_suite_sha256="b" * 64,
        expected_evaluator_id="evaluator",
        expected_evaluator_sha256="c" * 64,
        required_baseline_agents=("Claude Code", "Codex"),
    )

    result = verify_capability_certification(
        tmp_path / "missing.json",
        repository=tmp_path,
        policy=policy,
        now=NOW,
    )

    assert result["status"] == "FAIL"
    assert result["release_allowed"] is False
    assert result["failures"] == ["report.readable"]


def test_report_tampering_breaks_external_and_semantic_digests(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)
    report = json.loads(fixture.report_path.read_text(encoding="utf-8"))
    report["external_evaluation"]["candidate"]["successful_trials"] -= 1
    _write_json(fixture.report_path, report)

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=fixture.policy,
        now=NOW,
    )

    assert result["release_allowed"] is False
    assert "report.external_digest" in result["failures"]
    assert "report.semantic_attestation" in result["failures"]


def test_artifact_tampering_fails_even_when_report_is_unchanged(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)
    (fixture.artifact_root / "held-out-raw-results.jsonl").write_text(
        '{"trial":"altered-after-issuance"}\n',
        encoding="utf-8",
    )

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=fixture.policy,
        now=NOW,
    )

    assert result["release_allowed"] is False
    assert any(failure.endswith(".content") for failure in result["failures"])


def test_raw_trials_are_recomputed_instead_of_trusting_signed_aggregates(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)
    raw_path = fixture.artifact_root / "held-out-raw-results.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    candidate_row = next(row for row in rows if row["agent_name"] == "EvoMind")
    candidate_row["outcome"] = "failed"
    raw_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = json.loads(fixture.report_path.read_text(encoding="utf-8"))
    raw_record = next(item for item in report["artifacts"] if item["role"] == "benchmark_raw_results")
    raw_record["sha256"] = _sha(raw_path)
    raw_record["size_bytes"] = raw_path.stat().st_size
    report["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _canonical_digest(report, "attestation"),
    }
    _write_json(fixture.report_path, report)
    policy = replace(fixture.policy, expected_report_sha256=_sha(fixture.report_path))

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=policy,
        now=NOW,
    )

    assert result["release_allowed"] is False
    assert result["failures"][-2:] == [
        "raw_results.metric_binding",
        "raw_results.paired_binding",
    ]


def test_malformed_signed_metric_fails_closed_without_crashing(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)
    report = json.loads(fixture.report_path.read_text(encoding="utf-8"))
    report["external_evaluation"]["candidate"]["timed_out_trials"] = "zero"
    report["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _canonical_digest(report, "attestation"),
    }
    _write_json(fixture.report_path, report)
    policy = replace(fixture.policy, expected_report_sha256=_sha(fixture.report_path))

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=policy,
        now=NOW,
    )

    assert result["release_allowed"] is False
    assert "candidate.counters" in result["failures"]
    assert "candidate.timeout_rate" in result["failures"]


def test_malformed_signed_suite_domains_fail_closed_without_crashing(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)
    report = json.loads(fixture.report_path.read_text(encoding="utf-8"))
    report["external_evaluation"]["suite"]["domains"] = [{"not": "a domain name"}]
    report["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _canonical_digest(report, "attestation"),
    }
    _write_json(fixture.report_path, report)
    policy = replace(fixture.policy, expected_report_sha256=_sha(fixture.report_path))

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=policy,
        now=NOW,
    )

    assert result["release_allowed"] is False
    assert "suite.minimum_domain_scope" in result["failures"]
    assert "raw_results.strict_records" in result["failures"]


def test_insufficient_hidden_task_scope_fails_policy_gate(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)
    stricter_policy = replace(fixture.policy, minimum_hidden_tasks=105)

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=stricter_policy,
        now=NOW,
    )

    assert result["release_allowed"] is False
    assert "suite.minimum_task_repeat_scope" in result["failures"]


def test_expired_evidence_fails_freshness_gate(tmp_path: Path) -> None:
    fixture = _build_external_fixture(tmp_path)

    result = verify_capability_certification(
        fixture.report_path,
        repository=fixture.repository,
        artifact_root=fixture.artifact_root,
        policy=fixture.policy,
        now=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )

    assert result["release_allowed"] is False
    assert "report.freshness" in result["failures"]


def test_cli_returns_zero_only_for_certified_evidence(tmp_path: Path, capsys) -> None:
    fixture = _build_external_fixture(tmp_path)
    output = tmp_path / "gate-result.json"
    args = [
        str(fixture.report_path),
        "--repo-root",
        str(fixture.repository),
        "--artifact-root",
        str(fixture.artifact_root),
        "--expected-report-sha256",
        fixture.policy.expected_report_sha256,
        "--expected-suite-id",
        fixture.policy.expected_suite_id,
        "--expected-suite-sha256",
        fixture.policy.expected_suite_sha256,
        "--expected-evaluator-id",
        fixture.policy.expected_evaluator_id,
        "--expected-evaluator-sha256",
        fixture.policy.expected_evaluator_sha256,
        "--baseline-agent",
        "Claude Code",
        "--baseline-agent",
        "Codex",
        "--as-of",
        "2026-07-15T12:00:00Z",
        "--output",
        str(output),
    ]

    assert certification_cli(args) == 0
    rendered = capsys.readouterr().out
    assert json.loads(rendered)["release_allowed"] is True
    assert output.read_text(encoding="utf-8") == rendered

    args[args.index(fixture.policy.expected_report_sha256)] = "0" * 64
    assert certification_cli(args) == 1
    assert json.loads(capsys.readouterr().out)["release_allowed"] is False
