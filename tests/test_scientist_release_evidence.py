from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from xsci.capability_certification import RESULT_SCHEMA, compute_repository_identity
from xsci.scientist_release_evidence import (
    PARITY_SCORE_CAP_WITHOUT_CERTIFICATION,
    install_capability_certification,
    read_capability_certification_status,
    read_research_parity_gate,
)
from xsci.scientist_upgrade_controller import CHAMPION_REF, CONTROLLER_SCHEMA


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _digest(payload: dict[str, object], *, excluded: str | None = None) -> str:
    value = dict(payload)
    if excluded:
        value.pop(excluded, None)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "EvoMind Test")
    _git(repository, "config", "user.email", "evomind@example.invalid")
    (repository / ".gitignore").write_text(".xsci/\n", encoding="utf-8")
    (repository / "source.txt").write_text("baseline\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "source.txt")
    _git(repository, "commit", "-q", "-m", "baseline")
    return repository


def _write_certification(repository: Path) -> tuple[Path, str]:
    identity = compute_repository_identity(repository)
    campaign_evidence = repository / ".xsci" / "self_upgrade_campaign_evidence.json"
    campaign_evidence.parent.mkdir(parents=True, exist_ok=True)
    if not campaign_evidence.is_file():
        campaign_evidence.write_text('{"campaign_id":"external-fixture"}', encoding="utf-8")
    payload: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "release_allowed": True,
        "evidence_id": "external-suite-fixture",
        "report_sha256": "a" * 64,
        "repository_identity": identity.to_dict(),
        "artifact_bindings": {
            "self_upgrade_campaign": {
                "sha256": hashlib.sha256(campaign_evidence.read_bytes()).hexdigest(),
                "size_bytes": campaign_evidence.stat().st_size,
            }
        },
        "policy": {
            "required_baseline_agents": ["OpenAI Codex", "Anthropic Claude Code"],
            "minimum_hidden_tasks": 100,
            "minimum_domains": 8,
            "minimum_repeats": 3,
        },
        "checks_passed": 2,
        "checks_total": 2,
        "failures": [],
        "checks": [
            {"id": "external.hidden_suite", "passed": True, "detail": "fixture"},
            {"id": "baseline.noninferiority", "passed": True, "detail": "fixture"},
        ],
    }
    path = repository / ".xsci" / "capability_certification_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _activate_campaign(repository: Path) -> Path:
    (repository / "source.txt").write_text("candidate\n", encoding="utf-8")
    _git(repository, "add", "source.txt")
    _git(repository, "commit", "-q", "-m", "candidate")
    commit = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    _git(repository, "update-ref", CHAMPION_REF, commit)
    baseline_evaluation: dict[str, object] = {
        "label": "baseline",
        "passed": True,
        "score": 1.0,
        "metrics": {"quality": 1.0},
        "missing_required_metrics": [],
        "details": "fixture baseline",
        "evaluator_id": "fixture-evaluator",
        "primary_metric": "quality",
        "direction": "maximize",
    }
    baseline_evaluation["evaluation_digest_sha256"] = _digest(baseline_evaluation)
    candidate_evaluation: dict[str, object] = {
        "label": "candidate-1",
        "passed": True,
        "score": 1.1,
        "metrics": {"quality": 1.1},
        "missing_required_metrics": [],
        "details": "fixture candidate",
        "evaluator_id": "fixture-evaluator",
        "primary_metric": "quality",
        "direction": "maximize",
    }
    candidate_evaluation["evaluation_digest_sha256"] = _digest(candidate_evaluation)
    campaign_evidence = repository / ".xsci" / "self_upgrade_campaign_evidence.json"
    campaign_evidence.parent.mkdir(parents=True, exist_ok=True)
    campaign_evidence.write_text(
        json.dumps({"campaign_id": "upgrade-fixture", "commit_sha": commit, "tree_sha": tree}, sort_keys=True),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema": CONTROLLER_SCHEMA,
        "campaign_id": "upgrade-fixture",
        "status": "active",
        "baseline": baseline_evaluation,
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "evaluation": candidate_evaluation,
                "evaluation_digest_sha256": candidate_evaluation["evaluation_digest_sha256"],
                "strictly_improves_baseline": True,
            }
        ],
        "selection": {
            "candidate_id": "candidate-1",
            "decision": "promoted",
            "strictly_improves_baseline": True,
        },
        "promotion": {
            "candidate_id": "candidate-1",
            "human_approved": True,
            "verified": True,
            "promoted_commit_sha": commit,
            "promoted_tree_sha": tree,
        },
        "rollback": {"tested": True, "passed": True, "restored_tree_sha": tree},
        "certification_manifest_path": str(campaign_evidence),
    }
    payload["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _digest(payload),
    }
    path = repository / ".xsci" / "scientist_upgrade_campaign.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_parity_gate_fails_closed_without_external_evidence(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    result = read_research_parity_gate(repository)

    assert result["status"] == "blocked"
    assert result["parity_claim_allowed"] is False
    assert result["score_cap"] == PARITY_SCORE_CAP_WITHOUT_CERTIFICATION
    assert set(result["blockers"]) == {
        "external_capability_certification_not_verified",
        "active_self_upgrade_campaign_not_verified",
    }


def test_certification_requires_out_of_band_result_digest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write_certification(repository)

    result = read_capability_certification_status(repository)

    assert result["status"] == "trust_anchor_missing"
    assert result["verified"] is False


def test_hash_anchored_malformed_policy_fails_closed_without_exception(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    path, _ = _write_certification(repository)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy"]["minimum_hidden_tasks"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = read_capability_certification_status(repository, expected_result_sha256=digest)

    assert result["status"] == "invalid_or_wrong_source"
    assert result["semantic_checks_passed"] is False
    assert result["verified"] is False


def test_hash_anchored_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    path = repository / ".xsci" / "capability_certification_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = read_capability_certification_status(repository, expected_result_sha256=digest)

    assert result["status"] == "invalid_or_wrong_source"
    assert result["semantic_checks_passed"] is False
    assert result["verified"] is False


def test_hash_anchored_exact_source_and_active_campaign_open_parity_gate(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _activate_campaign(repository)
    _, digest = _write_certification(repository)

    result = read_research_parity_gate(repository, expected_result_sha256=digest)

    assert result["status"] == "certified_research_parity", json.dumps(result, indent=2)
    assert result["parity_claim_allowed"] is True
    assert result["score_cap"] == 100
    assert result["certification"]["source_identity_matches"] is True
    assert result["upgrade_campaign"]["champion_ref_matches"] is True
    assert result["certification_campaign_source_binding_verified"] is True
    assert result["certification_campaign_artifact_binding_verified"] is True


def test_certificate_and_active_campaign_must_bind_the_same_source(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    certified_commit = _git(repository, "rev-parse", "HEAD")
    _, digest = _write_certification(repository)
    _activate_campaign(repository)
    _git(repository, "checkout", "-q", "--detach", certified_commit)

    result = read_research_parity_gate(repository, expected_result_sha256=digest)

    assert result["certification"]["verified"] is True
    assert result["upgrade_campaign"]["active_and_verified"] is True
    assert result["certification_campaign_source_binding_verified"] is False
    assert result["status"] == "blocked"
    assert result["parity_claim_allowed"] is False
    assert "certification_campaign_source_mismatch" in result["blockers"]


def test_active_campaign_artifact_must_match_external_certification(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _activate_campaign(repository)
    _, digest = _write_certification(repository)
    (repository / ".xsci" / "self_upgrade_campaign_evidence.json").write_text(
        '{"campaign_id":"locally-forged"}',
        encoding="utf-8",
    )

    result = read_research_parity_gate(repository, expected_result_sha256=digest)

    assert result["certification"]["verified"] is True
    assert result["upgrade_campaign"]["active_and_verified"] is True
    assert result["certification_campaign_source_binding_verified"] is True
    assert result["certification_campaign_artifact_binding_verified"] is False
    assert result["status"] == "blocked"
    assert "certification_campaign_artifact_mismatch" in result["blockers"]


def test_source_change_after_certification_closes_gate(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _activate_campaign(repository)
    _, digest = _write_certification(repository)
    (repository / "source.txt").write_text("post-certification drift\n", encoding="utf-8")

    result = read_research_parity_gate(repository, expected_result_sha256=digest)

    assert result["status"] == "blocked"
    assert result["certification"]["status"] == "invalid_or_wrong_source"
    assert result["certification"]["source_identity_matches"] is False


def test_tampered_campaign_attestation_closes_gate(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    campaign_path = _activate_campaign(repository)
    _, digest = _write_certification(repository)
    payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    payload["promotion"]["verified"] = False
    campaign_path.write_text(json.dumps(payload), encoding="utf-8")

    result = read_research_parity_gate(repository, expected_result_sha256=digest)

    assert result["status"] == "blocked"
    assert result["upgrade_campaign"]["attestation_verified"] is False
    assert result["upgrade_campaign"]["active_and_verified"] is False


def test_active_campaign_requires_attested_candidate_evaluations(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    campaign_path = _activate_campaign(repository)
    _, digest = _write_certification(repository)
    payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    payload["candidates"][0]["evaluation"]["score"] = 0.5
    payload["attestation"] = {
        "algorithm": "sha256",
        "payload_sha256": _digest(payload, excluded="attestation"),
    }
    campaign_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = read_research_parity_gate(repository, expected_result_sha256=digest)

    assert result["status"] == "blocked"
    assert result["upgrade_campaign"]["attestation_verified"] is True
    assert result["upgrade_campaign"]["strict_improvement_verified"] is False
    assert result["upgrade_campaign"]["active_and_verified"] is False


def test_explicit_hash_import_persists_verified_runtime_anchor(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _activate_campaign(repository)
    generated_path, digest = _write_certification(repository)
    external_result = tmp_path / "external-capability-result.json"
    external_result.write_bytes(generated_path.read_bytes())
    generated_path.unlink()

    installed = install_capability_certification(
        repository,
        external_result,
        expected_result_sha256=digest,
    )
    observed = read_capability_certification_status(repository)

    assert installed["status"] == "installed"
    assert observed["status"] == "certified"
    assert observed["trust_anchor_source"] == "explicitly_installed_local_anchor"
    assert observed["verified"] is True
