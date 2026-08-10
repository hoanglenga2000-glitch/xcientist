from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ci_checks.py"


def load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_tests_keeps_outer_pytest_args_out_of_nested_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner("run_ci_checks_nested_env")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unsafe-inherited-option")
    monkeypatch.setattr(runner, "_pytest_command", lambda: (["python", "-m", "pytest"], "python"))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert "PYTEST_ADDOPTS" not in environment
        if "--collect-only" in command:
            return subprocess.CompletedProcess(command, 0, stdout="tests/test_example.py: 1\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=".\n1 passed\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    ok, detail = runner.gate_tests(
        False,
        pytest_run_args=["--junitxml=C:/outside/result.xml", "@C:/outside/order.args"],
        pytest_collect_args=["--basetemp=C:/outside/collect"],
    )

    assert ok is True
    assert "1 tests collected" in detail
    assert calls[0][0][-1] == "--basetemp=C:/outside/collect"
    assert calls[1][0][-2:] == ["--junitxml=C:/outside/result.xml", "@C:/outside/order.args"]


def test_main_rejects_junit_output_inside_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner("run_ci_checks_repository_guard")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--skip-tests",
            "--pytest-junitxml",
            str(ROOT / "artifacts" / "invalid.xml"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        runner.main()
