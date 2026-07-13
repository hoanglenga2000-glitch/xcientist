from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from scripts import verify_verified_workstation_launch_audit as audit


def _write_audit(
    tmp_path,
    *,
    deepseek: bool,
    claude: bool,
    include_deepseek_smoke: bool = False,
    generated_at: str | None = None,
) -> tuple:
    json_path = tmp_path / "verified_workstation_launch_audit.json"
    markdown_path = tmp_path / "verified_workstation_launch_audit.md"
    app_dir = tmp_path / "web" / "research-agent-workstation"
    (app_dir / "src").mkdir(parents=True)
    (app_dir / "src" / "page.ts").write_text("export {};\n", encoding="utf-8")
    (app_dir / ".next").mkdir()
    (app_dir / ".next" / "BUILD_ID").write_text("test-build-id\n", encoding="utf-8")
    runtime_dir = app_dir / ".runtime-logs"
    runtime_dir.mkdir()
    source_digest = audit.source_tree_digest(app_dir)
    runtime = {
        "pid": os.getpid(),
        "port": 8089,
        "mode": "start",
        "build_id": "test-build-id",
        "source_digest": source_digest,
        "build_requested": True,
    }
    (runtime_dir / "dashboard.pid").write_text(str(os.getpid()), encoding="utf-8")
    (runtime_dir / "dashboard.state.json").write_text(json.dumps(runtime), encoding="utf-8")
    labels = [
        {
            "label": "backend_resource_status",
            "ok": True,
            "signals": {},
        },
        {
            "label": "external_gateway_smoke",
            "ok": True,
            "signals": {"code_agent_configured_not_invoked": True},
        },
        {
            "label": "kaggle_secret_smoke",
            "ok": True,
            "signals": {},
        },
        {
            "label": "plaintext_secret_scan",
            "ok": True,
            "signals": {},
        },
    ]
    if include_deepseek_smoke:
        labels.append({"label": "deepseek_smoke", "ok": True, "signals": {}})
    provider_verified = deepseek and include_deepseek_smoke
    json_path.write_text(
        json.dumps(
            {
                "status": "passed" if provider_verified else "local_ready_external_unverified",
                "run_id": "a" * 32,
                "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
                "dashboard_url": "http://127.0.0.1:8089",
                "dashboard_runtime": runtime,
                "dpapi_loaded": {
                    "deepseek": deepseek,
                    "claude": claude,
                    "kaggle": False,
                    "hpc_ssh": False,
                },
                "external_provider_runtime_verified": provider_verified,
                "secret_policy": "No secret values or raw command output are written to this audit report.",
                "claim_boundary": (
                    "The provider passed."
                    if provider_verified
                    else "The local production gateway passed, but no external LLM provider was invoked successfully in this run."
                ),
                "allow_real_external": False,
                "result_summaries": labels,
            }
        ),
        encoding="utf-8",
    )
    markdown_path.write_text("# Verified Workstation Launch Audit\n", encoding="utf-8")
    return json_path, markdown_path


def test_verified_launch_accepts_claude_as_the_protected_llm(monkeypatch, tmp_path, capsys) -> None:
    json_path, markdown_path = _write_audit(tmp_path, deepseek=False, claude=True)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    audit.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert "deepseek_smoke" not in payload["smoke_labels"]
    assert payload["external_provider_runtime_verified"] is False


def test_verified_launch_rejects_when_no_protected_llm_is_loaded(monkeypatch, tmp_path) -> None:
    json_path, markdown_path = _write_audit(tmp_path, deepseek=False, claude=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    with pytest.raises(SystemExit, match="At least one protected LLM provider"):
        audit.main()


def test_verified_launch_requires_deepseek_smoke_when_deepseek_is_loaded(monkeypatch, tmp_path) -> None:
    json_path, markdown_path = _write_audit(tmp_path, deepseek=True, claude=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    with pytest.raises(SystemExit, match="missing required smoke labels"):
        audit.main()


def test_verified_launch_accepts_current_deepseek_smoke(monkeypatch, tmp_path, capsys) -> None:
    json_path, markdown_path = _write_audit(
        tmp_path,
        deepseek=True,
        claude=False,
        include_deepseek_smoke=True,
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    audit.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["external_provider_runtime_verified"] is True


def test_verified_launch_rejects_stale_audit(monkeypatch, tmp_path) -> None:
    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    json_path, markdown_path = _write_audit(tmp_path, deepseek=False, claude=True, generated_at=stale)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    with pytest.raises(SystemExit, match="stale or future-dated"):
        audit.main()


def test_verified_launch_rejects_raw_command_excerpt(monkeypatch, tmp_path) -> None:
    json_path, markdown_path = _write_audit(tmp_path, deepseek=False, claude=True)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["result_summaries"][0]["output_excerpt"] = "GPU_SSH_PASSWORD=sentinel"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    with pytest.raises(SystemExit, match="must not persist raw command output"):
        audit.main()


def test_verified_launch_rejects_source_changed_after_build(monkeypatch, tmp_path) -> None:
    json_path, markdown_path = _write_audit(tmp_path, deepseek=False, claude=True)
    (tmp_path / "web" / "research-agent-workstation" / "src" / "page.ts").write_text(
        "export const changed = true;\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    with pytest.raises(SystemExit, match="source changed"):
        audit.main()


def test_verified_launch_does_not_treat_gpu_smoke_as_llm_verification(monkeypatch, tmp_path) -> None:
    json_path, markdown_path = _write_audit(tmp_path, deepseek=False, claude=True)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["status"] = "passed"
    report["external_provider_runtime_verified"] = True
    external = next(item for item in report["result_summaries"] if item["label"] == "external_gateway_smoke")
    external["signals"] = {"gpu_smoke_tested": True, "code_agent_configured_not_invoked": True}
    json_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_JSON", json_path)
    monkeypatch.setattr(audit, "AUDIT_MD", markdown_path)

    with pytest.raises(SystemExit, match="verification flag does not match"):
        audit.main()
