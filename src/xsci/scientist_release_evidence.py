"""Read-only trust gate for research-parity and self-upgrade claims."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .capability_certification import RESULT_SCHEMA, compute_repository_identity
from .scientist_upgrade_controller import CHAMPION_REF, CONTROLLER_SCHEMA, evaluation_attestation_matches

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
DEFAULT_RESULT_PATH = Path(".xsci") / "capability_certification_result.json"
DEFAULT_ANCHOR_PATH = Path(".xsci") / "capability_certification_anchor.json"
DEFAULT_CAMPAIGN_PATH = Path(".xsci") / "scientist_upgrade_campaign.json"
ANCHOR_SCHEMA = "evomind.capability_certification_anchor.v1"
PARITY_SCORE_CAP_WITHOUT_CERTIFICATION = 84


def _canonical_digest(value: Mapping[str, Any], *, excluded_key: str | None = None) -> str:
    payload = dict(value)
    if excluded_key is not None:
        payload.pop(excluded_key, None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes().decode("utf-8-sig")
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _resolve_path(repository: Path, value: Path | str | None, *, default: Path) -> Path:
    path = Path(value) if value is not None else default
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _git(repository: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", "core.quotepath=false", *args],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _expected_result_digest(repository: Path, explicit: str | None) -> tuple[str, str]:
    value = explicit or os.environ.get("EVOMIND_CAPABILITY_CERTIFICATION_SHA256", "")
    if value.strip():
        return value.strip().lower(), "explicit_argument_or_environment"
    anchor_path = repository / DEFAULT_ANCHOR_PATH
    anchor = _read_object(anchor_path)
    if anchor.get("schema") == ANCHOR_SCHEMA and isinstance(anchor.get("result_sha256"), str):
        return str(anchor["result_sha256"]).strip().lower(), "explicitly_installed_local_anchor"
    return "", "missing"


def _baseline_names(policy: Mapping[str, Any]) -> list[str]:
    values = policy.get("required_baseline_agents")
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _named_frontier_baselines(values: list[str]) -> bool:
    normalized = [re.sub(r"[^a-z0-9]+", "", value.lower()) for value in values]
    return any("codex" in value for value in normalized) and any("claude" in value for value in normalized)


def read_capability_certification_status(
    repository: Path | str,
    *,
    result_path: Path | str | None = None,
    expected_result_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify an externally produced certification result against the current source."""

    root = Path(repository).resolve()
    configured_path = result_path or os.environ.get("EVOMIND_CAPABILITY_CERTIFICATION_RESULT")
    path = _resolve_path(root, configured_path, default=DEFAULT_RESULT_PATH)
    expected_digest, anchor_source = _expected_result_digest(root, expected_result_sha256)
    base = {
        "tool": "capability_certification_status",
        "artifact_path": str(path),
        "expected_result_sha256_configured": bool(expected_digest),
        "trust_anchor_source": anchor_source,
        "verified": False,
        "release_allowed": False,
        "parity_claim_allowed": False,
    }
    if not path.is_file():
        return {
            **base,
            "status": "not_certified",
            "reason": "external capability certification result is not installed",
        }
    if not SHA256_RE.fullmatch(expected_digest):
        return {
            **base,
            "status": "trust_anchor_missing",
            "result_sha256": _file_sha256(path),
            "reason": "an out-of-band certification result SHA-256 is required",
        }

    observed_digest = _file_sha256(path)
    if observed_digest != expected_digest:
        return {
            **base,
            "status": "digest_mismatch",
            "result_sha256": observed_digest,
            "reason": "certification result bytes do not match the configured trust anchor",
        }
    payload = _read_object(path)
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    failures = payload.get("failures") if isinstance(payload.get("failures"), list) else ["malformed_failures"]
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    identity = payload.get("repository_identity") if isinstance(payload.get("repository_identity"), dict) else {}
    artifact_bindings = (
        payload.get("artifact_bindings")
        if isinstance(payload.get("artifact_bindings"), dict)
        else {}
    )
    campaign_artifact = (
        artifact_bindings.get("self_upgrade_campaign")
        if isinstance(artifact_bindings.get("self_upgrade_campaign"), dict)
        else {}
    )
    campaign_artifact_ok = (
        set(campaign_artifact) == {"sha256", "size_bytes"}
        and isinstance(campaign_artifact.get("sha256"), str)
        and SHA256_RE.fullmatch(str(campaign_artifact.get("sha256"))) is not None
        and _integer_at_least(campaign_artifact.get("size_bytes"), 1)
    )
    baselines = _baseline_names(policy)
    semantic_ok = (
        payload.get("schema") == RESULT_SCHEMA
        and payload.get("status") == "PASS"
        and payload.get("release_allowed") is True
        and _integer_at_least(payload.get("checks_total"), 1)
        and _integer_at_least(payload.get("checks_passed"), 1)
        and payload.get("checks_passed") == payload.get("checks_total") == len(checks)
        and not failures
        and all(isinstance(item, dict) and item.get("passed") is True for item in checks)
        and campaign_artifact_ok
        and _named_frontier_baselines(baselines)
        and _integer_at_least(policy.get("minimum_hidden_tasks"), 100)
        and _integer_at_least(policy.get("minimum_domains"), 8)
        and _integer_at_least(policy.get("minimum_repeats"), 3)
    )
    try:
        current = compute_repository_identity(root)
        current_identity = current.to_dict()
    except Exception:
        current_identity = {}
    identity_fields = ("commit_sha", "tree_sha", "source_digest_algorithm", "source_digest_sha256")
    certified_identity = {
        key: str(identity.get(key) or "")
        for key in identity_fields
    }
    identity_mismatches = [
        key
        for key in identity_fields
        if current_identity.get(key) != identity.get(key)
    ]
    identity_ok = (
        bool(current_identity)
        and current_identity.get("worktree_clean") is True
        and identity.get("worktree_clean") is True
        and not identity_mismatches
    )
    verified = semantic_ok and identity_ok
    return {
        **base,
        "status": "certified" if verified else "invalid_or_wrong_source",
        "result_sha256": observed_digest,
        "evidence_id": str(payload.get("evidence_id") or ""),
        "report_sha256": str(payload.get("report_sha256") or ""),
        "baseline_agents": baselines,
        "certified_repository_identity": certified_identity,
        "certified_campaign_artifact": dict(campaign_artifact) if campaign_artifact_ok else {},
        "semantic_checks_passed": semantic_ok,
        "source_identity_matches": identity_ok,
        "source_identity_mismatches": identity_mismatches,
        "current_worktree_clean": current_identity.get("worktree_clean") is True,
        "certified_worktree_clean": identity.get("worktree_clean") is True,
        "verified": verified,
        "release_allowed": verified,
        "parity_claim_allowed": verified,
        "reason": (
            "external hidden-suite certification is hash-anchored and matches the exact source"
            if verified
            else "certification semantics or exact source identity did not verify"
        ),
    }


def install_capability_certification(
    repository: Path | str,
    result_path: Path | str,
    *,
    expected_result_sha256: str,
) -> dict[str, Any]:
    """Install an explicitly hash-anchored certificate for subsequent local checks."""

    root = Path(repository).resolve()
    source = Path(result_path).resolve()
    verified = read_capability_certification_status(
        root,
        result_path=source,
        expected_result_sha256=expected_result_sha256,
    )
    if verified.get("verified") is not True:
        return {
            "ok": False,
            "tool": "capability_certification_install",
            "status": "rejected",
            "verification": verified,
        }
    target = root / DEFAULT_RESULT_PATH
    anchor_path = root / DEFAULT_ANCHOR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target_temp = target.with_suffix(target.suffix + ".tmp")
    anchor_temp = anchor_path.with_suffix(anchor_path.suffix + ".tmp")
    if source != target:
        shutil.copyfile(source, target_temp)
        target_temp.replace(target)
    anchor = {
        "schema": ANCHOR_SCHEMA,
        "result_sha256": str(verified.get("result_sha256") or ""),
        "evidence_id": str(verified.get("evidence_id") or ""),
        "report_sha256": str(verified.get("report_sha256") or ""),
        "source": "explicit_out_of_band_digest_import",
    }
    anchor_temp.write_text(
        json.dumps(anchor, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    anchor_temp.replace(anchor_path)
    try:
        target.chmod(0o600)
        anchor_path.chmod(0o600)
    except OSError:
        pass
    installed = read_capability_certification_status(root)
    return {
        "ok": installed.get("verified") is True,
        "tool": "capability_certification_install",
        "status": "installed" if installed.get("verified") is True else "post_install_verification_failed",
        "artifact_path": str(target),
        "anchor_path": str(anchor_path),
        "verification": installed,
    }


def read_active_upgrade_campaign_status(
    repository: Path | str,
    *,
    campaign_path: Path | str | None = None,
    champion_ref: str = CHAMPION_REF,
) -> dict[str, Any]:
    """Verify that the locally active champion matches an attested campaign."""

    root = Path(repository).resolve()
    path = _resolve_path(root, campaign_path, default=DEFAULT_CAMPAIGN_PATH)
    base = {
        "tool": "scientist_upgrade_campaign_status",
        "artifact_path": str(path),
        "champion_ref": champion_ref,
        "active_and_verified": False,
    }
    if not path.is_file():
        return {**base, "status": "not_run", "reason": "no upgrade campaign is active"}
    payload = _read_object(path)
    if not payload:
        return {**base, "status": "invalid", "reason": "upgrade campaign artifact is unreadable"}

    attestation = payload.get("attestation") if isinstance(payload.get("attestation"), dict) else {}
    attestation_ok = (
        attestation.get("algorithm") == "sha256"
        and attestation.get("payload_sha256") == _canonical_digest(payload, excluded_key="attestation")
    )
    promotion = payload.get("promotion") if isinstance(payload.get("promotion"), dict) else {}
    rollback = payload.get("rollback") if isinstance(payload.get("rollback"), dict) else {}
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    selected_id = str(selection.get("candidate_id") or "")
    selected = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("candidate_id") == selected_id
        ),
        {},
    )
    selected_evaluation = selected.get("evaluation") if isinstance(selected.get("evaluation"), dict) else {}
    baseline_evidence_ok = evaluation_attestation_matches(baseline)
    candidate_evidence_ok = (
        evaluation_attestation_matches(selected_evaluation)
        and selected.get("evaluation_digest_sha256") == selected_evaluation.get("evaluation_digest_sha256")
        and selected.get("strictly_improves_baseline") is True
    )
    promoted_commit = str(promotion.get("promoted_commit_sha") or "")
    promoted_tree = str(promotion.get("promoted_tree_sha") or "")
    certification_manifest_value = payload.get("certification_manifest_path")
    certification_manifest = Path(certification_manifest_value) if isinstance(certification_manifest_value, str) else None
    if certification_manifest is not None and not certification_manifest.is_absolute():
        certification_manifest = root / certification_manifest
    certification_manifest_sha256 = ""
    certification_manifest_size = 0
    try:
        if certification_manifest is not None:
            certification_manifest = certification_manifest.resolve(strict=True)
            certification_manifest_size = certification_manifest.stat().st_size
            if certification_manifest.is_file() and 0 < certification_manifest_size <= 16 * 1024 * 1024:
                certification_manifest_sha256 = _file_sha256(certification_manifest)
    except OSError:
        certification_manifest_sha256 = ""
        certification_manifest_size = 0
    ref_commit = _git(root, "rev-parse", "--verify", champion_ref)
    ref_tree = _git(root, "rev-parse", f"{champion_ref}^{{tree}}") if ref_commit else ""
    active = (
        payload.get("schema") == CONTROLLER_SCHEMA
        and payload.get("status") == "active"
        and attestation_ok
        and promotion.get("human_approved") is True
        and promotion.get("verified") is True
        and selection.get("decision") == "promoted"
        and selection.get("strictly_improves_baseline") is True
        and baseline_evidence_ok
        and candidate_evidence_ok
        and rollback.get("tested") is True
        and rollback.get("passed") is True
        and COMMIT_RE.fullmatch(promoted_commit) is not None
        and COMMIT_RE.fullmatch(promoted_tree) is not None
        and ref_commit == promoted_commit
        and ref_tree == promoted_tree
    )
    return {
        **base,
        "status": "active_verified" if active else "inactive_or_invalid",
        "campaign_status": str(payload.get("status") or "unknown"),
        "campaign_id": str(payload.get("campaign_id") or ""),
        "attestation_verified": attestation_ok,
        "promotion_verified": promotion.get("verified") is True,
        "rollback_verified": rollback.get("tested") is True and rollback.get("passed") is True,
        "strict_improvement_verified": (
            selection.get("strictly_improves_baseline") is True
            and baseline_evidence_ok
            and candidate_evidence_ok
        ),
        "champion_ref_matches": ref_commit == promoted_commit and ref_tree == promoted_tree,
        "promoted_commit_sha": promoted_commit,
        "promoted_tree_sha": promoted_tree,
        "certification_manifest_path": str(certification_manifest) if certification_manifest is not None else "",
        "certification_manifest_sha256": certification_manifest_sha256,
        "certification_manifest_size_bytes": certification_manifest_size,
        "active_and_verified": active,
        "reason": (
            "strictly improved champion is active with canary and rollback evidence"
            if active
            else "campaign, attestation, champion ref, canary, or rollback evidence did not verify"
        ),
    }


def read_research_parity_gate(
    repository: Path | str,
    *,
    certification_result_path: Path | str | None = None,
    expected_result_sha256: str | None = None,
    campaign_path: Path | str | None = None,
) -> dict[str, Any]:
    certification = read_capability_certification_status(
        repository,
        result_path=certification_result_path,
        expected_result_sha256=expected_result_sha256,
    )
    campaign = read_active_upgrade_campaign_status(repository, campaign_path=campaign_path)
    certification_verified = certification.get("verified") is True
    campaign_verified = campaign.get("active_and_verified") is True
    certified_identity = (
        certification.get("certified_repository_identity")
        if isinstance(certification.get("certified_repository_identity"), dict)
        else {}
    )
    source_binding_evaluated = certification_verified and campaign_verified
    source_binding_verified = source_binding_evaluated and (
        campaign.get("promoted_commit_sha") == certified_identity.get("commit_sha")
        and campaign.get("promoted_tree_sha") == certified_identity.get("tree_sha")
    )
    certified_campaign_artifact = (
        certification.get("certified_campaign_artifact")
        if isinstance(certification.get("certified_campaign_artifact"), dict)
        else {}
    )
    campaign_artifact_binding_evaluated = certification_verified and campaign_verified
    campaign_artifact_binding_verified = campaign_artifact_binding_evaluated and (
        campaign.get("certification_manifest_sha256") == certified_campaign_artifact.get("sha256")
        and campaign.get("certification_manifest_size_bytes") == certified_campaign_artifact.get("size_bytes")
    )
    ready = (
        certification_verified
        and campaign_verified
        and source_binding_verified
        and campaign_artifact_binding_verified
    )
    blockers = []
    if not certification_verified:
        blockers.append("external_capability_certification_not_verified")
    if not campaign_verified:
        blockers.append("active_self_upgrade_campaign_not_verified")
    if source_binding_evaluated and not source_binding_verified:
        blockers.append("certification_campaign_source_mismatch")
    if campaign_artifact_binding_evaluated and not campaign_artifact_binding_verified:
        blockers.append("certification_campaign_artifact_mismatch")
    return {
        "tool": "research_parity_gate",
        "status": "certified_research_parity" if ready else "blocked",
        "parity_claim_allowed": ready,
        "score_cap": 100 if ready else PARITY_SCORE_CAP_WITHOUT_CERTIFICATION,
        "certification": certification,
        "upgrade_campaign": campaign,
        "certification_campaign_source_binding_verified": source_binding_verified,
        "certification_campaign_artifact_binding_verified": campaign_artifact_binding_verified,
        "blockers": blockers,
        "claim": (
            "externally certified non-inferiority against named Codex and Claude baselines"
            if ready
            else "research parity is not externally certified"
        ),
    }


__all__ = [
    "DEFAULT_CAMPAIGN_PATH",
    "DEFAULT_ANCHOR_PATH",
    "DEFAULT_RESULT_PATH",
    "PARITY_SCORE_CAP_WITHOUT_CERTIFICATION",
    "install_capability_certification",
    "read_active_upgrade_campaign_status",
    "read_capability_certification_status",
    "read_research_parity_gate",
]
