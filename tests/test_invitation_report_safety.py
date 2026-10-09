from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_invitation_acceptance.py"
    spec = importlib.util.spec_from_file_location("invitation_report_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_omits_environment_and_arbitrary_diagnostic_content(tmp_path):
    runner = load_runner()
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    case = ET.SubElement(suite, "testcase", {"name": "test_bound[" + "x" * 40000 + "]", "classname": "tests.fixture", "time": "0.1"})
    error = ET.SubElement(case, "error", {"message": "AUTH_TOKEN=fixture-private-value"})
    error.text = "self = environ({'PASSWORD': 'fixture-private-value'})"
    ET.SubElement(case, "system-out").text = "fixture-private-value"
    source, destination = tmp_path / "input.xml", tmp_path / "safe.xml"
    source.write_bytes(ET.tostring(root))
    result = runner.sanitize_junit(source, destination)
    data = destination.read_text()
    assert "fixture-private-value" not in data
    assert "environ" not in data and "AUTH_TOKEN" not in data
    assert result["errors"] == 1
    assert len(ET.fromstring(data).find("testsuite/testcase").get("name")) < 200


def test_acceptance_environment_does_not_inherit_live_credentials(monkeypatch):
    runner = load_runner()
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-private-value")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "fixture-private-value")
    monkeypatch.setenv("WORKSTATION_SESSION_SECRET", "fixture-private-value")
    monkeypatch.setenv("HTTPS_PROXY", "https://fixture.invalid")
    environment = runner.test_environment()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "WORKSTATION_SESSION_SECRET", "HTTPS_PROXY"):
        assert key not in environment


def test_sanitized_reports_are_derivatives_and_existing_files_are_immutable(tmp_path):
    runner = load_runner()
    source = tmp_path / "original.xml"
    raw = b'<testsuites><testsuite><testcase name="test_one[fixture-value]"/></testsuite></testsuites>'
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="source_overwrite"):
        runner.sanitize_junit(source, source)
    first, second = tmp_path / "first.xml", tmp_path / "second.xml"
    one = runner.sanitize_junit(source, first)
    two = runner.sanitize_junit(source, second)
    assert source.read_bytes() == raw
    assert first.read_bytes() == second.read_bytes()
    assert one == two
    with pytest.raises(ValueError, match="destination_exists"):
        runner.sanitize_junit(source, first)


def test_main_keeps_prior_evidence_and_rejects_reusing_an_output_directory(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ARTIFACT_ROOT", tmp_path / "artifacts")
    prior = tmp_path / "artifacts/invitation-beta-20260905/release-repairs-junit.xml"
    prior.parent.mkdir(parents=True)
    original = b'<testsuites><testsuite><testcase name="test_prior[fixture-value]"/></testsuite></testsuites>'
    prior.write_bytes(original)
    calls = []
    def fake_test(command, **_kwargs):
        calls.append(command)
        target = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        target.write_bytes(b'<testsuites><testsuite><testcase name="test_current"/></testsuite></testsuites>')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake_test)
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--report-dir", "artifacts/new-acceptance"])
    assert runner.main() == 0
    assert prior.read_bytes() == original
    result = json.loads((tmp_path / "artifacts/new-acceptance/acceptance-summary.json").read_text())
    assert result["prior_reports_sanitized"][0]["sanitized_path"].startswith("artifacts/new-acceptance/prior-reports/")
    with pytest.raises(FileExistsError):
        runner.main()
    assert len(calls) == 1
