"""Tests for the research-event stream (research_os.events + engine emission).

Offline and deterministic: a FakeLLMClient + LocalSubprocessRunner drive a REAL
EvolutionLoop, and an in-memory sink captures the events. We assert the stream is
well-formed (monotonic seq, lifecycle bracketing) and faithful (a promotion emits
promoted=True, a failure emits a repair with a real pattern). No tokens, no net.
"""
from __future__ import annotations

import json
from typing import Optional

from research_os import events as ev
from research_os.evolution_loop import EvolutionConfig, EvolutionLoop, LocalSubprocessRunner
from research_os.variation_generator import TaskContext, VariationGenerator


class FakeLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, user, *, system=None, max_tokens=4096, temperature=None, provider=None):
        from research_os.llm_client import LLMResponse
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return LLMResponse(text=text, provider="fake", model="fake-1", input_tokens=10, output_tokens=20)


def _full_script(score: Optional[float]) -> str:
    if score is None:
        body = "raise ValueError('injected: no cv score')"
    else:
        body = (
            "import argparse, json, os\n"
            "ap=argparse.ArgumentParser(); ap.add_argument('--data-dir'); ap.add_argument('--out-dir')\n"
            "a=ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)\n"
            "rows='id,y' + chr(10) + '1,0' + chr(10)\n"
            "open(os.path.join(a.out_dir,'submission.csv'),'w').write(rows)\n"
            f"json.dump({{'cv_score':{score},'metric':'accuracy'}}, open(os.path.join(a.out_dir,'metrics.json'),'w'))\n"
            f"print('CV_SCORE={score}')"
        )
    return f"hypothesis for this candidate\n```python\n{body}\n```"


def _run(tmp_path, scripts, *, iterations, sink):
    ctx = TaskContext("t", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient(scripts))
    runner = LocalSubprocessRunner(tmp_path / "work", timeout=60)
    loop = EvolutionLoop(
        ctx, data_dir=str(tmp_path / "data"), work_dir=tmp_path / "work",
        runner=runner, generator=gen,
        config=EvolutionConfig(max_iterations=iterations),
        on_event=sink,
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    summary = loop.run()
    return summary


# ── pure events module ───────────────────────────────────────────────────────
def test_default_loop_emits_nothing(tmp_path):
    """With no on_event (the default), the engine must be byte-for-byte unchanged:
    a run completes and NOTHING is emitted (proven by the absence of a sink)."""
    ctx = TaskContext("t", "tabular", "classification", "accuracy", "maximize", target_column="y")
    gen = VariationGenerator(client=FakeLLMClient([_full_script(0.9)]))
    loop = EvolutionLoop(ctx, data_dir=str(tmp_path / "d"), work_dir=tmp_path / "w",
                         runner=LocalSubprocessRunner(tmp_path / "w", timeout=60),
                         generator=gen, config=EvolutionConfig(max_iterations=1))
    (tmp_path / "d").mkdir()
    assert loop._on_event is None
    loop.run()  # must not raise; nothing to assert on a null sink beyond "no crash"


def test_jsonl_sink_and_read_roundtrip(tmp_path):
    sink = ev.JsonlEventSink(tmp_path / "events.jsonl")
    sink({"seq": 1, "type": ev.RUN_BEGIN, "task": "t"})
    sink({"seq": 2, "type": ev.RUN_END, "task": "t"})
    back = ev.read_events(tmp_path / "events.jsonl")
    assert [e["type"] for e in back] == [ev.RUN_BEGIN, ev.RUN_END]


def test_read_events_tolerates_partial_trailing_line(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(json.dumps({"seq": 1, "type": ev.RUN_BEGIN}) + "\n{ half-written",
                 encoding="utf-8")
    back = ev.read_events(p)
    assert len(back) == 1 and back[0]["type"] == ev.RUN_BEGIN


def test_fan_out_isolates_a_failing_sink(tmp_path):
    seen = []
    def bad(_e): raise RuntimeError("boom")
    def good(e): seen.append(e["type"])
    ev.fan_out(bad, good)({"type": ev.SCORE})
    assert seen == [ev.SCORE]  # good sink still fired despite bad one raising


# ── engine emission ──────────────────────────────────────────────────────────
def test_stream_brackets_and_is_monotonic(tmp_path):
    captured = []
    _run(tmp_path, [_full_script(0.90)], iterations=1, sink=captured.append)
    seqs = [e["seq"] for e in captured]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))  # strictly ordered, unique
    types = [e["type"] for e in captured]
    assert types[0] == ev.RUN_BEGIN and types[-1] == ev.RUN_END
    assert types.count(ev.ITER_BEGIN) == types.count(ev.ITER_END) == 1


def test_promotion_emits_promote_true(tmp_path):
    captured = []
    _run(tmp_path, [_full_script(0.90)], iterations=1, sink=captured.append)
    promos = [e for e in captured if e["type"] == ev.PROMOTE]
    assert promos and promos[0]["promoted"] is True
    assert promos[0]["best_cv_score"] == 0.90


def test_failure_emits_repair_with_pattern(tmp_path):
    captured = []
    # first proposal fails (no score), so a REPAIR event must carry a real pattern
    _run(tmp_path, [_full_script(None), _full_script(0.8)], iterations=2, sink=captured.append)
    repairs = [e for e in captured if e["type"] == ev.REPAIR]
    assert repairs, "a failed run must emit a repair event"
    assert repairs[0]["failure_pattern"]  # non-empty bucket
    assert "seq" in repairs[0] and "ts" in repairs[0]


def test_select_event_only_when_mcgs_on(tmp_path):
    """SELECT is emitted only when the MCGS brain produced a plan. Linear mode
    (no selector) must never emit SELECT."""
    captured = []
    _run(tmp_path, [_full_script(0.90), _full_script(0.91)], iterations=2, sink=captured.append)
    assert not any(e["type"] == ev.SELECT for e in captured)


def test_format_event_covers_all_types():
    """Every declared type renders a non-empty line (no KeyError, no blank)."""
    samples = {
        ev.RUN_BEGIN: {"task": "t", "metric": "accuracy", "metric_direction": "maximize",
                       "max_iterations": 6, "mcgs": True},
        ev.ITER_BEGIN: {"iteration": 0, "exp_id": "EXP000"},
        ev.SELECT: {"node_exp_id": "EXP000", "expansion_type": "intra_branch",
                    "coding_mode": "Diff", "reference_exp_ids": ["EXP001"]},
        ev.PROPOSE: {"mode": "Base", "expansion_type": "primary", "hypothesis": "try X"},
        ev.EXEC_BEGIN: {"exp_id": "EXP000", "runner": "LocalSubprocessRunner"},
        ev.SCORE: {"success": True, "cv_score": 0.9},
        ev.PROMOTE: {"promoted": True, "best_exp_id": "EXP000", "best_cv_score": 0.9, "delta": 0.01},
        ev.REPAIR: {"failure_pattern": "oom", "error": "CUDA out of memory"},
        ev.LESSON: {"failure_pattern": "", "reusable_strategy": "kfold"},
        ev.ITER_END: {"exp_id": "EXP000", "mode": "Base", "cv_score": 0.9, "promoted": True},
        ev.RUN_END: {"best_exp_id": "EXP000", "best_cv_score": 0.9, "n_promotions": 1, "n_iterations": 1},
    }
    for t, fields in samples.items():
        line = ev.format_event({"seq": 1, "ts": "now", "type": t, **fields})
        assert line and isinstance(line, str) and len(line) > 3
