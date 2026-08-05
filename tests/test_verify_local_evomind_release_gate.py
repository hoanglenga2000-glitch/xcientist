from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import verify_local_evomind_release_gate as gate


def test_release_source_tracking_manifest_includes_agent_ux_smoke_files() -> None:
    tracked_paths = {
        path.relative_to(gate.ROOT).as_posix()
        for path in gate.RELEASE_RELEVANT_SOURCE_FILES
    }

    assert "scripts/verify_local_evomind_release_gate.py" in tracked_paths
    assert "scripts/verify_live_assistant_demo_smoke.py" in tracked_paths
    assert "scripts/build_agent_ux_release_source_manifest.py" in tracked_paths
    assert "tests/test_verify_local_evomind_release_gate.py" in tracked_paths
    assert "tests/test_verify_live_assistant_demo_smoke.py" in tracked_paths
    assert "tests/test_build_agent_ux_release_source_manifest.py" in tracked_paths
    assert "src/xsci/assistant_behavior_distillation.py" in tracked_paths
    assert "src/xsci/kaggle_conversation.py" in tracked_paths
    assert "scripts/manage_hpc_proxy_bridge.ps1" in tracked_paths
    assert "tests/test_hpc_connection_memory_core.py" in tracked_paths
    assert "scripts/manage_workstation_dashboard.py" in tracked_paths
    assert "tests/test_dashboard_build_transaction.py" in tracked_paths
    assert "web/research-agent-workstation/src/app/api/assistant/stream/route.ts" in tracked_paths
    assert "web/research-agent-workstation/src/components/workstation/screens/AssistantScreen.tsx" in tracked_paths


def test_release_source_tracking_covers_agent_runtime_and_session_dependency_closure() -> None:
    tracked_paths = [
        path.relative_to(gate.ROOT).as_posix()
        for path in gate.RELEASE_RELEVANT_SOURCE_FILES
    ]
    tracked_set = set(tracked_paths)
    required = {
        "configs/evaluation/assistant_behavior_board_v1.json",
        "configs/evaluation/assistant_novice_v1.json",
        "configs/hpc_connection_memory_core.json",
        "docs/HPC_CONNECTION_MEMORY_CORE.md",
        "scripts/hpc_socks_bridge.py",
        "scripts/run_authenticated_web_contract.py",
        "scripts/verify_web_isolated_build.py",
        "tests/test_verify_web_isolated_build.py",
        "src/research_os/agent/messaging.py",
        "src/research_os/llm_client.py",
        "src/xsci/assistant_context.py",
        "src/xsci/assistant_quality_evaluation.py",
        "src/xsci/assistant_stream.py",
        "src/xsci/config.py",
        "src/xsci/kaggle_intent.py",
        "src/xsci/kaggle_session.py",
        "src/xsci/terminal_agent.py",
        "src/xsci/terminal_tools.py",
        "src/xsci/user_request.py",
        "tests/test_agent_messaging.py",
        "tests/test_assistant_behavior_distillation.py",
        "tests/test_assistant_context.py",
        "tests/test_assistant_quality_evaluation.py",
        "tests/test_assistant_stream.py",
        "tests/test_hpc_socks_bridge.py",
        "tests/test_user_request_protocol.py",
        "web/research-agent-workstation/src/app/api/session/bootstrap/route.ts",
        "web/research-agent-workstation/src/app/api/session/status/route.ts",
        "web/research-agent-workstation/src/app/layout.tsx",
        "web/research-agent-workstation/src/components/workstation/LocalSessionBootstrap.tsx",
        "web/research-agent-workstation/src/lib/server/local-session-security.test.ts",
        "web/research-agent-workstation/src/lib/server/local-session.ts",
        "web/research-agent-workstation/src/lib/server/loopback-session.test.ts",
        "web/research-agent-workstation/src/lib/server/paths.ts",
        "web/research-agent-workstation/src/proxy.ts",
        "web/research-agent-workstation/scripts/lib/loopback-session.mjs",
        "web/research-agent-workstation/scripts/verify-localhost-security-contract.mjs",
        "web/research-agent-workstation/scripts/verify-report-studio-contract.mjs",
        "web/research-agent-workstation/package.json",
        "web/research-agent-workstation/package-lock.json",
    }
    runtime_sources = {
        path.relative_to(gate.ROOT).as_posix()
        for path in (gate.ROOT / "src/evomind_runtime").glob("*.py")
    }
    runtime_tests = {
        path.relative_to(gate.ROOT).as_posix()
        for path in (gate.ROOT / "tests").glob("test_evomind_runtime*.py")
    }

    missing = sorted((required | runtime_sources | runtime_tests) - tracked_set)
    assert not missing, f"release source scope omitted runtime dependencies: {missing}"
    assert len(tracked_paths) == len(tracked_set), "release source scope contains duplicate paths"


def _git_result(stdout: str = "") -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    proc = subprocess.CompletedProcess(["git"], 0, stdout, "")
    return proc, {"command": ["git"], "returncode": 0, "stdout_sha256": "", "stderr_tail": ""}


def test_release_source_tracking_warns_for_untracked_file(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent_ux.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('agent ux fix')\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "RELEASE_RELEVANT_SOURCE_FILES", (source,))

    def fake_run_git(args: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        if args[:2] == ["ls-files", "--stage"]:
            return _git_result("")
        if args[:2] == ["status", "--porcelain=v1"]:
            return _git_result("?? src/agent_ux.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(gate, "run_git", fake_run_git)
    checks: list[gate.Check] = []

    gate.check_release_source_tracking(checks, require_release_source_tracked=False)

    check = checks[0].as_dict()
    assert check["id"] == "release_source:agent_ux_files_tracked_for_clean_checkout"
    assert check["ok"] is False
    assert check["severity"] == "warning"
    assert check["untracked"] == ["src/agent_ux.py"]


def test_release_source_tracking_blocks_formal_release_for_untracked_file(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent_ux.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('agent ux fix')\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "RELEASE_RELEVANT_SOURCE_FILES", (source,))

    def fake_run_git(args: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        if args[:2] == ["ls-files", "--stage"]:
            return _git_result("")
        if args[:2] == ["status", "--porcelain=v1"]:
            return _git_result("?? src/agent_ux.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(gate, "run_git", fake_run_git)
    checks: list[gate.Check] = []

    gate.check_release_source_tracking(checks, require_release_source_tracked=True)

    check = checks[0].as_dict()
    assert check["ok"] is False
    assert check["severity"] == "blocker"
    assert check["untracked"] == ["src/agent_ux.py"]


def test_release_source_tracking_warns_for_tracked_dirty_file_in_local_demo(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent_ux.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('agent ux fix')\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "RELEASE_RELEVANT_SOURCE_FILES", (source,))

    def fake_run_git(args: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        if args[:2] == ["ls-files", "--stage"]:
            return _git_result("100644 0123456789012345678901234567890123456789 0\tsrc/agent_ux.py\n")
        if args[:2] == ["status", "--porcelain=v1"]:
            return _git_result(" M src/agent_ux.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(gate, "run_git", fake_run_git)
    checks: list[gate.Check] = []

    gate.check_release_source_tracking(checks, require_release_source_tracked=False)

    check = checks[0].as_dict()
    assert check["ok"] is True
    assert check["untracked"] == []
    assert check["tracked_dirty"] == ["src/agent_ux.py"]
    assert check["strict_dirty_blockers"] == []


def test_release_source_tracking_blocks_formal_release_for_tracked_dirty_file(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent_ux.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('agent ux fix')\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "RELEASE_RELEVANT_SOURCE_FILES", (source,))

    def fake_run_git(args: list[str]) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        if args[:2] == ["ls-files", "--stage"]:
            return _git_result("100644 0123456789012345678901234567890123456789 0\tsrc/agent_ux.py\n")
        if args[:2] == ["status", "--porcelain=v1"]:
            return _git_result(" M src/agent_ux.py\n")
        raise AssertionError(args)

    monkeypatch.setattr(gate, "run_git", fake_run_git)
    checks: list[gate.Check] = []

    gate.check_release_source_tracking(checks, require_release_source_tracked=True)

    check = checks[0].as_dict()
    assert check["ok"] is False
    assert check["severity"] == "blocker"
    assert check["untracked"] == []
    assert check["tracked_dirty"] == ["src/agent_ux.py"]
    assert check["strict_dirty_blockers"] == ["src/agent_ux.py"]


def _source_tracking_fixture() -> dict[str, object]:
    return {
        "missing": [],
        "untracked": ["src/agent_ux.py"],
        "tracked_dirty": ["src/tracked_dirty.py"],
        "files": [
            {"path": "src/agent_ux.py"},
            {"path": "src/tracked_dirty.py"},
        ],
    }


def _scope_manifest_fixture() -> dict[str, object]:
    return {
        "schema": "evomind.agent_ux_release_source_scope.v1",
        "status": "needs_release_source_review",
        "claim_boundary": "Source-scope audit only.",
        "secret_scan": {
            "status": "passed",
            "scanned_files": 2,
            "finding_count": 0,
            "findings": [],
        },
        "missing": [],
        "untracked": ["src/agent_ux.py"],
        "dirty_tracked": ["src/tracked_dirty.py"],
        "files": [
            {"path": "src/agent_ux.py"},
            {"path": "src/tracked_dirty.py"},
        ],
    }


def test_release_source_scope_manifest_matches_live_tracking(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "scope.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(gate.OPTIONAL_EVIDENCE, "agent_ux_release_source_scope", manifest_path)
    evidence = {"agent_ux_release_source_scope": _scope_manifest_fixture()}
    checks: list[gate.Check] = []

    gate.check_release_source_scope_manifest(
        evidence,
        checks,
        _source_tracking_fixture(),
        max_age_hours=48,
        require_release_source_clean=True,
    )

    check = checks[0].as_dict()
    assert check["id"] == "release_source:scope_manifest_fresh_and_consistent"
    assert check["ok"] is True
    assert check["severity"] == "blocker"
    assert check["manifest_paths_match_source_tracking"] is True


def test_release_source_scope_manifest_detects_tracking_drift(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "scope.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(gate.OPTIONAL_EVIDENCE, "agent_ux_release_source_scope", manifest_path)
    manifest = _scope_manifest_fixture()
    manifest["untracked"] = ["src/other.py"]
    evidence = {"agent_ux_release_source_scope": manifest}
    checks: list[gate.Check] = []

    gate.check_release_source_scope_manifest(
        evidence,
        checks,
        _source_tracking_fixture(),
        max_age_hours=48,
        require_release_source_clean=True,
    )

    check = checks[0].as_dict()
    assert check["ok"] is False
    assert check["severity"] == "blocker"
    assert check["manifest_untracked"] == ["src/other.py"]
    assert check["source_untracked"] == ["src/agent_ux.py"]


def test_release_source_scope_manifest_secret_scan_is_blocking() -> None:
    manifest = _scope_manifest_fixture()
    evidence = {"agent_ux_release_source_scope": manifest}
    checks: list[gate.Check] = []

    gate.check_release_source_manifest_secret_scan(
        evidence,
        checks,
        require_release_source_clean=False,
    )

    check = checks[0].as_dict()
    assert check["id"] == "release_source:scope_manifest_secret_scan_passed"
    assert check["ok"] is True
    assert check["severity"] == "blocker"


def test_release_source_scope_manifest_secret_scan_rejects_findings() -> None:
    manifest = _scope_manifest_fixture()
    manifest["secret_scan"] = {
        "status": "failed",
        "scanned_files": 2,
        "finding_count": 1,
        "findings": [{"path": "src/agent_ux.py", "line": 7, "pattern_id": "literal_password_assignment"}],
    }
    evidence = {"agent_ux_release_source_scope": manifest}
    checks: list[gate.Check] = []

    gate.check_release_source_manifest_secret_scan(
        evidence,
        checks,
        require_release_source_clean=True,
    )

    check = checks[0].as_dict()
    assert check["ok"] is False
    assert check["severity"] == "blocker"
    assert check["finding_count"] == 1


def _web_contract_evidence(*, npm_audit_ok: bool | None) -> dict[str, object]:
    build: dict[str, object] = {
        "status": "passed",
        "npm_ci": {"ok": True},
        "prisma_generate": {"ok": True},
        "build": {"ok": True},
        "standalone_server_exists": True,
        "protected_ports_unchanged": True,
        "stage_removed": True,
        "build_id": "isolated-build",
    }
    if npm_audit_ok is not None:
        build["npm_audit"] = {"ok": npm_audit_ok}
    return {
        "authenticated_web_report": {
            "ok": True,
            "isolated_runtime": True,
            "protected_ports_unchanged": True,
            "contract": {"ok": True, "checks": 1},
        },
        "authenticated_web_security": {
            "ok": True,
            "isolated_runtime": True,
            "protected_ports_unchanged": True,
            "contract": {
                "ok": True,
                "checks": 1,
                "grader_invoked": False,
                "mutations_with_product_side_effects": 0,
            },
        },
        "web_isolated_build": build,
    }


def test_web_build_gate_requires_successful_full_dependency_audit() -> None:
    checks: list[gate.Check] = []
    gate.check_web_contracts(_web_contract_evidence(npm_audit_ok=None), checks)

    build_check = next(check.as_dict() for check in checks if check.id == "web_build:isolated_production_build_passed")
    assert build_check["ok"] is False
    assert build_check["npm_audit_ok"] is None

    checks = []
    gate.check_web_contracts(_web_contract_evidence(npm_audit_ok=True), checks)
    build_check = next(check.as_dict() for check in checks if check.id == "web_build:isolated_production_build_passed")
    assert build_check["ok"] is True
    assert build_check["npm_audit_ok"] is True
