"""`run_agent` render wiring: the terminal path must suppress the plan/banner
wall and route events through the injected staged renderer, while the standalone
`xsci agent` path keeps its full plan dump. Heavy deps (task resolution, plan
build, message client, session) are faked so this is a fast, token-free unit test
of the branching in ``run_agent`` only."""
from __future__ import annotations

from pathlib import Path

import pytest

from xsci import agent as xagent


class _FakePlan:
    warnings: list[str] = []
    task_name = "titanic"
    compute = "local"
    exp_dir = Path("exp/evolution/titanic_local_20260709_000000")

    def render(self) -> str:
        return "PLAN-WALL-SENTINEL: task/metric/compute/artifacts..."


class _FakeSession:
    exp_dir = "exp/titanic"

    def __init__(self, on_event):
        self.on_event = on_event

    def run(self, goal):
        # Emit two events so we can assert the injected renderer received them.
        self.on_event({"type": "run_begin", "task": "t", "metric": "auc",
                       "metric_direction": "maximize"})
        self.on_event({"type": "run_end", "task": "t", "best_exp_id": "EXP1",
                       "best_cv_score": 0.9, "n_promotions": 1, "n_iterations": 1})
        return {"best_exp_id": "EXP1", "best_cv_score": 0.9,
                "n_promotions": 1, "n_iterations": 1, "turns_used": 3}


@pytest.fixture()
def patched_agent(monkeypatch):
    # resolve_task is imported *inside* run_agent (`from .tasks import ...`), so
    # patch it at the source module, not on xagent.
    import xsci.tasks as xtasks
    monkeypatch.setattr(xtasks, "resolve_task", lambda task: {"name": task})
    monkeypatch.setattr(xagent, "build_plan",
                        lambda *a, **k: _FakePlan(), raising=True)

    captured = {}

    def fake_build_session(plan, *, quiet, mcgs, resume, event_renderer=None):
        captured["event_renderer"] = event_renderer
        return _FakeSession(on_event=event_renderer or (lambda e: None))

    monkeypatch.setattr(xagent, "_build_session", fake_build_session)
    monkeypatch.setattr(
        xagent, "_record_evolution_summary",
        lambda root, summary, *, task="", events_path=None: captured.setdefault("evolution_records", []).append(
            (root, summary, task, events_path)
        ),
    )
    monkeypatch.setattr(
        xagent,
        "build_execution_contract_for_task",
        lambda *a, **k: {
            "ok": True,
            "go_no_go": "go",
            "agent_session_ready": True,
            "model_training_ready": True,
            "data_contract_status": "ready",
            "execution_command": "evomind run titanic",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        },
    )
    monkeypatch.setattr(xagent, "_banner", lambda *a, **k: "BANNER-SENTINEL")

    class _FakeClient:
        def is_available(self):
            return True

    import research_os.agent as ra
    monkeypatch.setattr(ra, "AgentMessageClient", _FakeClient)
    return captured


def test_terminal_path_suppresses_plan_and_banner(patched_agent, capsys):
    seen = []
    rc = xagent.run_agent("titanic", goal="go", cfg=object(),
                          event_renderer=lambda e: seen.append(e["type"]),
                          show_plan=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "PLAN-WALL-SENTINEL" not in out          # no plan dump
    assert "BANNER-SENTINEL" not in out             # no banner wall
    assert "[goal]" not in out and "[summary]" not in out  # no raw log lines
    assert seen == ["run_begin", "run_end"]         # events reached the renderer
    assert patched_agent["evolution_records"][0][2] == "titanic"
    assert str(patched_agent["evolution_records"][0][3]).endswith("events.jsonl")


def test_standalone_path_still_shows_plan(patched_agent, capsys):
    rc = xagent.run_agent("titanic", goal="go", cfg=object())  # defaults
    out = capsys.readouterr().out
    assert rc == 0
    assert "PLAN-WALL-SENTINEL" in out              # full plan preserved
    assert "BANNER-SENTINEL" in out
    assert patched_agent["event_renderer"] is None  # uses built-in raw renderer
    assert patched_agent["evolution_records"][0][2] == "titanic"
    assert str(patched_agent["evolution_records"][0][3]).endswith("events.jsonl")
