from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow() -> tuple[dict, str]:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text), text


def test_tag_release_requires_external_capability_certification_job() -> None:
    data, text = workflow()
    jobs = data["jobs"]
    certification = jobs["capability-certification"]
    assert certification["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert set(certification["needs"]) == {"python-release", "frontend-release"}
    assert set(jobs["release-artifacts"]["needs"]) == {
        "python-release",
        "frontend-release",
        "capability-certification",
    }
    assert "secrets.EVOMIND_CAPABILITY_EVIDENCE_URL" in text
    assert "vars.EVOMIND_CAPABILITY_REPORT_SHA256" in text
    assert "vars.EVOMIND_CAPABILITY_SUITE_SHA256" in text
    assert "vars.EVOMIND_CAPABILITY_EVALUATOR_SHA256" in text
    assert "At least two external baseline agents are required" in text


def test_release_gate_enforces_scope_statistics_and_certified_source_bytes() -> None:
    _data, text = workflow()
    for token in (
        "--minimum-hidden-tasks', '100'",
        "--minimum-domains', '8'",
        "--minimum-tasks-per-domain', '3'",
        "--minimum-repeats', '3'",
        "--minimum-wilson-lower-bound', '0.75'",
        "--maximum-timeout-rate', '0'",
        "--minimum-upgrade-candidates', '2'",
    ):
        assert token in text
    assert "Built workstation source bytes do not match externally certified source bytes" in text
    assert "capability-certification-result.json" in text
    assert "extract_capability_evidence_bundle.py" in text


def test_capability_evidence_artifacts_use_immutable_action_versions() -> None:
    _data, text = workflow()
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in text
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in text
    assert "persist-credentials: false" in text


def test_runtime_exposes_fail_closed_certification_campaign_and_parity_gate() -> None:
    route = (
        ROOT
        / "web"
        / "research-agent-workstation"
        / "src"
        / "app"
        / "api"
        / "scientist"
        / "upgrade-campaign"
        / "route.ts"
    ).read_text(encoding="utf-8")
    ui = (
        ROOT
        / "web"
        / "research-agent-workstation"
        / "src"
        / "components"
        / "workstation"
        / "AiControlConsole.tsx"
    ).read_text(encoding="utf-8")
    gateway = (ROOT / "src" / "xsci" / "scientist_upgrade_gateway.py").read_text(encoding="utf-8")
    evidence = (ROOT / "src" / "xsci" / "scientist_release_evidence.py").read_text(encoding="utf-8")

    assert '["status", "run", "promote", "rollback"]' in route
    assert "Explicit human approval is required for promotion" in route
    assert "activation_command" not in route
    assert "Verified Upgrade Campaign" in ui
    assert "scientistPromotionApproved" in ui
    assert 'scientistParityCertified ? "green" : "red"' in ui
    assert "EVOMIND_SOURCE_REPOSITORY" in gateway
    assert "initialize_upgrade_repository" in gateway
    assert "PARITY_SCORE_CAP_WITHOUT_CERTIFICATION = 84" in evidence
    assert "explicit_out_of_band_digest_import" in evidence


def test_release_bundle_verifier_pins_all_r16_trust_boundary_sources() -> None:
    verifier = (ROOT / "scripts" / "verify_release_artifacts.py").read_text(encoding="utf-8")
    for path in (
        ".github/workflows/ci.yml",
        "scripts/extract_capability_evidence_bundle.py",
        "scripts/verify_capability_certification.py",
        "src/xsci/capability_certification.py",
        "src/xsci/scientist_hypothesis_panel.py",
        "src/xsci/scientist_release_evidence.py",
        "src/xsci/scientist_upgrade_controller.py",
        "src/xsci/scientist_upgrade_gateway.py",
    ):
        assert f'"{path}"' in verifier


def test_public_docs_reserve_frontier_parity_claim_for_external_evidence() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    assert "Research-Parity Certification" in readme
    assert "evomind parity-status" in readme
    assert "Local artifacts, proxy benchmarks" in readme
    assert "never open the research-parity gate" in readme
    assert "Stable Research-Parity Gate" in checklist
    assert "at least 100 tasks, 8 domains, 3 tasks per domain, and 3 repeats" in checklist
