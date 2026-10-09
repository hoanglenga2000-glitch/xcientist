"""P0-1: production hotfixes D8/D9/D10 live in source, not only on the server (D1/D2B superseded).

The 2026-09-20 hotfixes were applied in place on the Shanghai node and were
reverted by the 2026-10-08 redeploy because they never reached the repo.
These checks make that regression visible in CI.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from evomind_runtime import assistant_runs, super_agent_runtime
from evomind_runtime.runtime import _message_max_output_tokens


def test_d1_d2b_are_superseded_by_session_scoped_capabilities_not_ported():
    # D1 would let every Run use the runtime-wide ``workspace`` capability, which
    # the current design deliberately forbids (tests/test_super_agent_runtime_integration.py:
    # a tenant session gets its own run-scoped capability and is denied ``workspace``).
    source = inspect.getsource(super_agent_runtime)
    assert "Hotfix D1" not in source and "Hotfix D2B" not in source


def test_d8_assistant_runs_have_long_training_step_budget():
    source = inspect.getsource(assistant_runs)
    assert "max_steps=48)" in source and "max_steps=24)" not in source
    assert "Hotfix D8: training-run tuning" in source


def test_d9_flash_output_budget_and_parallel_data_rule():
    assert _message_max_output_tokens({}, SimpleNamespace(model="deepseek-flash")) == 32768
    assert "Hotfix D9: long-run tuning" in inspect.getsource(assistant_runs)


def test_d10_retries_only_transient_file_errors(monkeypatch):
    monkeypatch.setattr(assistant_runs.time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "held by scanner")
        return "ok"

    assert assistant_runs._retry_transient_file_op(flaky) == "ok" and calls["n"] == 3

    def broken():
        raise FileNotFoundError("missing")

    with pytest.raises(FileNotFoundError):
        assistant_runs._retry_transient_file_op(broken)

    def always_denied():
        raise PermissionError(13, "denied")

    with pytest.raises(PermissionError):
        assistant_runs._retry_transient_file_op(always_denied, attempts=2)
