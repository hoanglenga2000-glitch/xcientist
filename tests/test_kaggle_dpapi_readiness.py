from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_kaggle_dpapi_readiness as readiness


def _manager_status(*, installed: bool, credential_path: str = "") -> dict[str, object]:
    return {
        "status": "configured" if installed else "not_configured",
        "credential_installed": installed,
        "credential_path": credential_path,
        "token_type": "access_token" if installed else "none",
        "tool_status": {
            "python_package_installed": True,
            "python_package_version": "2.2.3",
            "cli_path": "C:/runtime/Scripts/kaggle.exe",
        },
    }


def test_configured_credential_without_real_smoke_remains_auth_pending(tmp_path: Path) -> None:
    report = readiness.build_report(
        _manager_status(installed=True, credential_path=str(tmp_path / "kaggle_api_token.xml"))
    )

    serialized = json.dumps(report)
    assert report["status"] == "auth_pending"
    assert report["credential_status"] == "configured_dpapi_unverified"
    assert report["authenticated"] is False
    assert str(tmp_path) not in serialized
    assert "EvoMind-" + "release-validation" not in serialized


def test_only_explicit_real_api_smoke_can_mark_authenticated(tmp_path: Path) -> None:
    report = readiness.build_report(
        _manager_status(installed=True, credential_path=str(tmp_path / "kaggle_api_token.xml")),
        real_smoke={"status": "passed", "real_external_called": True},
    )

    assert report["status"] == "passed"
    assert report["credential_status"] == "authenticated_real_api"
    assert report["authenticated"] is True


def test_not_configured_is_truthful_tooling_only_state() -> None:
    report = readiness.build_report(_manager_status(installed=False))

    assert report["status"] == "auth_pending"
    assert report["credential_status"] == "not_configured"
    assert report["credential_installed"] is False
    assert report["authenticated"] is False


def test_manager_parse_failure_is_not_a_configured_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(returncode=0, stdout=b"not-json", stderr=b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(SystemExit) as exc_info:
        readiness.run_manager("status")

    payload = json.loads(str(exc_info.value))
    assert payload["status"] == "failed"
    assert "clean JSON" in payload["message"]


def test_manager_environment_restores_windows_security_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WINDIR", r"C:\Windows")
    monkeypatch.setenv("PSModulePath", r"D:\PowerShell7\Modules")

    env = readiness._manager_environment()
    expected = str(
        Path(r"C:\Windows")
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "Modules"
    )

    assert env["PSModulePath"].split(os.pathsep)[0].casefold() == expected.casefold()


def test_manager_failure_reports_only_sanitized_error_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {
                "status": "failed",
                "error_code": "credential_operation_failed",
                "error_type": "CommandNotFoundException",
                "credential_installed": False,
                "credential_path": r"C:\private\credential.xml",
            }
        ).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(SystemExit) as exc_info:
        readiness.run_manager("status")

    payload = json.loads(str(exc_info.value))
    evidence = payload["evidence"]
    assert evidence["manager_error"] == {
        "status": "failed",
        "error_code": "credential_operation_failed",
        "error_type": "CommandNotFoundException",
        "credential_installed": False,
    }
    assert "credential_path" not in json.dumps(evidence)
