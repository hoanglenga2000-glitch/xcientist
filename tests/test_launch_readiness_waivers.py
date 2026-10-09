"""External-condition waivers are explicit, allowlisted and recorded."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def readiness(monkeypatch):
    module = load("launch_readiness_waivers", "scripts/verify_workstation_launch_readiness.py")
    monkeypatch.setattr(module, "COMMANDS", [{"id": "server_health"}, {"id": "workstation_training_progress"}, {"id": "learning_loop_readiness"}])
    monkeypatch.setattr(module, "with_runtime_base_url", lambda item, base: item)
    status = {"server_health": "passed", "workstation_training_progress": "failed", "learning_loop_readiness": "failed"}
    monkeypatch.setattr(module, "run_command", lambda item, timeout: {"id": item["id"], "critical": True, "status": status[item["id"]]})
    monkeypatch.setattr(module, "write_figma_gate", lambda args: {"id": "figma_gate", "critical": False, "status": "blocked"})
    monkeypatch.setattr(module, "http_check", lambda base, path: {"ok": True})
    monkeypatch.setattr(module, "normalize_figma_status", lambda value: {"status": "blocked_figma_auth", "blocked": True})
    monkeypatch.setattr(module, "resource_snapshot", lambda base: {})
    monkeypatch.setattr(module, "summarize_inventory", lambda: {})
    return module


def args(**extra):
    return argparse.Namespace(base_url="http://127.0.0.1:18088", include_frontend=False, include_build=False,
                              command_timeout=5, figma_status=None, waive_external=extra.get("waive", []))


def test_without_waivers_external_conditions_still_fail(readiness) -> None:
    report = readiness.build_report(args())
    assert report["status"] == "failed"
    assert report["critical_failures"] == ["workstation_training_progress", "learning_loop_readiness"]
    assert report["waived_external"] == []


def test_explicit_waivers_are_recorded_not_dropped(readiness) -> None:
    report = readiness.build_report(args(waive=["workstation_training_progress", "learning_loop_readiness"]))
    assert report["status"] == "passed"
    assert report["critical_failures"] == []
    assert [item["id"] for item in report["waived_external"]] == ["workstation_training_progress", "learning_loop_readiness"]
    assert all(item["status"] == "failed" and item["reason"] for item in report["waived_external"])
    assert "figma_auth_blocked" in report["blockers"]


def test_only_allowlisted_external_checks_can_be_waived(readiness) -> None:
    with pytest.raises(SystemExit, match="unsupported --waive-external"):
        readiness.build_report(args(waive=["browser_render_smoke"]))
    assert "browser_render_smoke" not in readiness.EXTERNAL_WAIVABLE_CHECKS
    assert "server_health" not in readiness.EXTERNAL_WAIVABLE_CHECKS


def test_acceptance_records_waivers_in_its_report() -> None:
    text = (ROOT / "scripts" / "run_new_user_release_acceptance.ps1").read_text(encoding="utf-8-sig")
    assert "[string[]]$WaiveExternal = @()" in text
    assert '"--waive-external"; $_' in text
    assert "waived_external = @($WaiveExternal" in text


def test_new_user_readiness_checks_the_dashboard_on_the_requested_port(monkeypatch) -> None:
    module = load("new_user_readiness_port", "scripts/verify_new_user_release_readiness.py")
    seen = []
    monkeypatch.setattr(module, "run", lambda cmd, **kw: seen.append(cmd) or {"ok": False, "stdout_tail": "", "stderr_tail": ""})
    module.check_existing_launch_gate("http://127.0.0.1:18088")
    assert seen[0][seen[0].index("--port") + 1] == "18088"


def test_retired_dashboard_markers_are_not_applicable_not_passed() -> None:
    verifier = load("frontend_contract_redirect", "scripts/verify_workstation_frontend_api_contract.py")
    nav = verifier.build_navigation_contract()
    sync = verifier.build_terminal_task_sync_contract()
    assert nav["mode"] == "workspace_redirect" and nav["ok"] is True
    assert sync["not_applicable"] == sorted(verifier.LEGACY_PAGE_CHECKS)
    assert not set(verifier.LEGACY_PAGE_CHECKS) & set(sync["checks"])
    assert sync["checks"]["runtime_prefers_current_run_events"] is True


def test_every_port_bound_launch_check_targets_the_server_under_test() -> None:
    module = load("launch_readiness_ports", "scripts/verify_workstation_launch_readiness.py")
    for item in module.COMMANDS:
        if item["id"] in {"deepseek_cache_probe_api", "server_health", "task_api_matrix"}:
            resolved = module.with_runtime_base_url(item, "http://127.0.0.1:18088")
            assert "http://127.0.0.1:18088" in resolved["cmd"], item["id"]
