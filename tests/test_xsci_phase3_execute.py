"""Phase-3 execution-path test: drives `execute_plan` with a faked engine so
the artifact-writing wiring (summary.json / best_solution.py / search_graph.json)
is verified WITHOUT spending API tokens or touching the network."""
from __future__ import annotations

import json

import pytest

from xsci import config as xcfg
from xsci import engine as xengine


class _FakeGraph:
    def export_json(self, path):
        from pathlib import Path
        Path(path).write_text('{"nodes": []}', encoding="utf-8")


class _FakeLoop:
    """Mimics EvolutionLoop's surface used by execute_plan."""

    last_kwargs = None

    def __init__(self, ctx, *, data_dir, work_dir, runner, memory, config, selector,
                 on_event=None, run_meta=None):
        type(self).last_kwargs = dict(
            task=ctx.task_name, data_dir=data_dir, work_dir=str(work_dir),
            runner=type(runner).__name__, selector=selector,
            on_event=on_event, run_meta=run_meta,
        )
        self.best_code = "print('best solution')"
        self.graph = _FakeGraph()
        self._on_event = on_event

    def run(self, *, strategies=None):
        # Mirror the real loop: emit a minimal, well-formed stream through the sink
        # execute_plan handed us, so the JSONL-persist + fan-out wiring is exercised.
        if self._on_event is not None:
            self._on_event({"seq": 1, "ts": "t", "type": "run_begin", "task": "titanic"})
            self._on_event({"seq": 2, "ts": "t", "type": "repair", "failure_pattern": "timeout"})
            self._on_event({"seq": 3, "ts": "t", "type": "lesson", "reusable_strategy": "gbm", "failure_pattern": ""})
            self._on_event({"seq": 4, "ts": "t", "type": "run_end", "best_exp_id": "exp_003"})
        return {
            "best_exp_id": "exp_003", "best_cv_score": 0.912,
            "n_promotions": 2, "n_iterations": 3, "strategies": strategies,
            "iterations": [],
        }


@pytest.fixture()
def patched_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOMIND_HPC_REMOTE_WORKSPACE", "/srv/evomind/test-phase3")
    monkeypatch.setattr("research_os.evolution_loop.EvolutionLoop", _FakeLoop)
    monkeypatch.setattr("research_os.gpu_runner.GPURunner",
                        lambda *a, **k: type("R", (), {})())
    monkeypatch.setattr("research_os.retrospective_memory.RetrospectiveMemoryStore",
                        lambda *a, **k: object())
    return tmp_path


def _task_json(path, name="titanic"):
    path.write_text(json.dumps({
        "task_name": name, "modality": "tabular", "metric": "accuracy",
        "local_data_dir": str(path.parent / "data"), "n_train": 891, "n_test": 418,
    }), encoding="utf-8")
    return path


def test_execute_plan_writes_artifacts(patched_engine, tmp_path):
    cfg_path = _task_json(tmp_path / "titanic.json")
    plan = xengine.build_plan(
        cfg_path, cfg=xcfg.load_config(), compute="gpu",
        iterations=3, mcgs=False, project_root=tmp_path,
    )
    summary = xengine.execute_plan(plan)

    assert summary["best_exp_id"] == "exp_003"
    assert (plan.exp_dir / "summary.json").exists()
    assert (plan.exp_dir / "best_solution.py").read_text(encoding="utf-8") == "print('best solution')"
    graph = json.loads((plan.exp_dir / "search_graph.json").read_text(encoding="utf-8"))
    assert graph == {"nodes": []}
    # summary.json round-trips
    on_disk = json.loads((plan.exp_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["best_cv_score"] == 0.912
    tracker = json.loads((tmp_path / ".xsci" / "evolution_tracker.json").read_text(encoding="utf-8"))
    latest = tracker["history"][-1]
    assert latest["total_runs"] == 1
    assert latest["total_promotions"] == 2
    assert latest["repair_attempts"] == 1
    assert latest["repair_successes"] == 1
    assert latest["lessons_recorded"] == 1
    assert latest["reusable_lessons"] == 1


def test_execute_plan_gpu_uses_remote_runner(patched_engine, tmp_path):
    cfg_path = _task_json(tmp_path / "titanic.json")
    plan = xengine.build_plan(cfg_path, cfg=xcfg.load_config(), compute="gpu",
                              iterations=3, mcgs=False, project_root=tmp_path)
    xengine.execute_plan(plan)
    assert _FakeLoop.last_kwargs["selector"] is None  # mcgs=False -> no selector
    assert _FakeLoop.last_kwargs["task"] == "titanic"


def test_execute_plan_mcgs_builds_selector(patched_engine, tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("research_os.mcgs_selector.MCGSSelector",
                        lambda **k: captured.setdefault("sel", object()) or captured["sel"])
    cfg_path = _task_json(tmp_path / "titanic.json")
    plan = xengine.build_plan(cfg_path, cfg=xcfg.load_config(), compute="gpu",
                              iterations=5, mcgs=True, project_root=tmp_path)
    xengine.execute_plan(plan)
    assert _FakeLoop.last_kwargs["selector"] is captured["sel"]


def test_execute_plan_persists_events_jsonl(patched_engine, tmp_path):
    """The event stream is ALWAYS written to <exp_dir>/events.jsonl, even with no
    live renderer, so watch/dashboard/replay have a durable source of truth."""
    from research_os import events as ev
    cfg_path = _task_json(tmp_path / "titanic.json")
    plan = xengine.build_plan(cfg_path, cfg=xcfg.load_config(), compute="gpu",
                              iterations=3, mcgs=False, project_root=tmp_path)
    xengine.execute_plan(plan)
    events_path = plan.exp_dir / "events.jsonl"
    assert events_path.exists()
    events = ev.read_events(events_path)
    assert [e["type"] for e in events] == [ev.RUN_BEGIN, ev.REPAIR, ev.LESSON, ev.RUN_END]
    # run_meta is threaded into the loop so the dashboard has compute context
    assert _FakeLoop.last_kwargs["run_meta"]["compute"] == "gpu"


def test_execute_plan_fans_out_to_live_renderer(patched_engine, tmp_path):
    """A caller-supplied renderer (xsci run's terminal view) receives every event
    AND the JSONL file is still written -- fan-out feeds both sinks."""
    from research_os import events as ev
    seen = []
    cfg_path = _task_json(tmp_path / "titanic.json")
    plan = xengine.build_plan(cfg_path, cfg=xcfg.load_config(), compute="gpu",
                              iterations=3, mcgs=False, project_root=tmp_path)
    xengine.execute_plan(plan, on_event=lambda e: seen.append(e["type"]))
    assert seen == [ev.RUN_BEGIN, ev.REPAIR, ev.LESSON, ev.RUN_END]
    assert (plan.exp_dir / "events.jsonl").exists()
