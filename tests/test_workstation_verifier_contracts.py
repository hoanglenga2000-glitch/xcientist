from __future__ import annotations

import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_navigation_only_reads_nav_items(tmp_path, monkeypatch):
    verifier = load_script("verify_workstation_runtime_navigation")
    navigation = tmp_path / "navigation.ts"
    navigation.write_text(
        """
        export const navItems = [
          { id: "assistant", label: "Assistant" },
          { id: "runtime", label: "Runtime" },
        ] as const;
        export const navSections = [
          { id: "workbench", label: "Workbench" },
        ] as const;
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "NAVIGATION_TS", navigation)

    assert verifier.extract_nav_ids() == ["assistant", "runtime"]


def test_frontend_route_parser_supports_encoded_dynamic_segments():
    verifier = load_script("verify_workstation_frontend_api_contract")

    route, reason = verifier.endpoint_to_route(
        "/api/multi-agent/runs/${encodeURIComponent(runId)}/events"
        "?after_seq=${Math.max(0, afterSeq)}"
    )

    assert reason is None
    assert route == (
        verifier.SRC
        / "app"
        / "api"
        / "multi-agent"
        / "runs"
        / "[runId]"
        / "events"
        / "route.ts"
    )


def test_frontend_route_parser_supports_plain_dynamic_segments():
    verifier = load_script("verify_workstation_frontend_api_contract")

    route, reason = verifier.endpoint_to_route(
        "/api/multi-agent/runs/${encodeURIComponent(runId)}/${action}"
    )

    assert reason is None
    assert route == (
        verifier.SRC
        / "app"
        / "api"
        / "multi-agent"
        / "runs"
        / "[runId]"
        / "[action]"
        / "route.ts"
    )


def test_frontend_runtime_event_precedence_contract_matches_live_source():
    verifier = load_script("verify_workstation_frontend_api_contract")

    contract = verifier.build_terminal_task_sync_contract()

    assert contract["checks"]["runtime_prefers_current_run_events"] is True


def test_release_verifier_accepts_only_a_packed_cli_artifact():
    source = (ROOT / "scripts" / "verify_release_bundle.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--cli-tgz"' in source
    assert not re.search(r'parser\.add_argument\("--cli"\s*,', source)
    assert "install_cli_from_tgz" in source
    assert "npm-tgz-disposable-prefix" in source
    assert 'packages" / "evomind-cli" / "bin"' not in source


def test_release_ci_binds_manifest_cli_tgz_and_zip_with_fixture_and_production_gates():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    invocations = workflow.split("python scripts/verify_release_bundle.py")[1:]

    assert len(invocations) == 2
    for invocation in invocations:
        command = invocation[:500]
        assert "--manifest" in command
        assert "--cli-tgz" in command
        assert "--zip" in command
    for marker in ("npm test", "npm run pack:check", "npm pack"):
        assert marker in workflow
    assert "tests/fixtures/release/fixture-ed25519-private.pem" in workflow
    assert "tests/fixtures/release/fixture-ed25519-public.pem" in workflow
    assert "git ls-files --error-unmatch" in workflow
    assert "packages/evomind-cli/keys/release-ed25519-public.pem" in workflow

    protected_job = workflow.split("  release-artifacts:", 1)[1]
    ordinary_job = workflow.split("  release-artifacts:", 1)[0]
    assert "if: startsWith(github.ref, 'refs/tags/v')" in protected_job
    assert "environment: production-release" in protected_job
    assert "secrets.EVOMIND_RELEASE_PRIVATE_KEY_PEM" in protected_job
    assert "secrets.EVOMIND_RELEASE_PRIVATE_KEY_PEM" not in ordinary_job
