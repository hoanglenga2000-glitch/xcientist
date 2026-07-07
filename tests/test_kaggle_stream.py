"""Tests for the staged research-event renderer and the thinking indicator.

These lock the *view* contract: the raw event stream (run_begin -> agent_msg ->
tool_call -> select -> propose -> exec -> score -> promote -> lesson -> run_end)
must render as the nine named research stages, carry the key facts (score,
promote/hold, lesson, best), and stay clean (no colour, no cursor escapes) when
piped / under NO_COLOR — so pipes and test capture are never polluted.
"""
from __future__ import annotations

import io

from research_os import events as ev
from xsci.kaggle_stream import STAGES, StageRenderer, thinking


def _render(events, *, color=False) -> str:
    buf = io.StringIO()
    r = StageRenderer(color=color, stream=buf)
    for e in events:
        r(e)
    return buf.getvalue()


# A scripted run that touches every stage, in the order the agent really emits.
SCRIPTED = [
    {"type": ev.RUN_BEGIN, "task": "titanic", "metric": "accuracy",
     "metric_direction": "maximize", "max_iterations": 5},
    {"type": ev.AGENT_MSG, "text": "Let me ground myself in the data first."},
    {"type": ev.TOOL_CALL, "tool": "inspect_data", "args_brief": "rows=5"},
    {"type": ev.TOOL_RESULT, "tool": "inspect_data", "ok": True, "summary": "12 columns, 891 rows"},
    {"type": ev.TOOL_CALL, "tool": "plan_next_experiment", "args_brief": ""},
    {"type": ev.SELECT, "node_exp_id": "EXP001", "expansion_type": "primary", "coding_mode": "Base"},
    {"type": ev.TOOL_CALL, "tool": "run_experiment", "args_brief": "hypothesis=..."},
    {"type": ev.PROPOSE, "mode": "Base", "expansion_type": "primary",
     "hypothesis": "A gradient-boosted baseline on raw features."},
    {"type": ev.EXEC_BEGIN, "exp_id": "EXP002", "runner": "local"},
    {"type": ev.SCORE, "success": True, "cv_score": 0.8421},
    {"type": ev.TOOL_CALL, "tool": "evaluate_promotion", "args_brief": "exp_id=EXP002"},
    {"type": ev.PROMOTE, "promoted": True, "best_exp_id": "EXP002", "best_cv_score": 0.8421, "delta": 0.0421},
    {"type": ev.TOOL_CALL, "tool": "record_lesson", "args_brief": "exp_id=EXP002"},
    {"type": ev.LESSON, "reusable_strategy": "gbm baseline"},
    {"type": ev.RUN_END, "task": "titanic", "best_exp_id": "EXP002",
     "best_cv_score": 0.8421, "n_promotions": 1, "n_iterations": 1},
]


def test_render_walks_the_named_stages_in_order():
    out = _render(SCRIPTED)
    # Every stage that the script exercises must be named as a labelled header.
    for stage in ("Understanding task", "Data audit", "Search decision",
                  "Hypothesis", "Training", "Gate", "Memory", "Report"):
        assert f"▸ {stage}" in out, f"missing stage header: {stage}"
    # Stage headers appear in narrative order (first occurrence of each).
    order = ["Understanding task", "Data audit", "Search decision", "Hypothesis",
             "Training", "Gate", "Memory", "Report"]
    positions = [out.index(f"▸ {s}") for s in order]
    assert positions == sorted(positions), "stage headers out of narrative order"


def test_render_surfaces_the_key_facts():
    out = _render(SCRIPTED)
    assert "0.8421" in out                      # the CV score
    assert "EXP002" in out                      # the promoted / best node
    assert "primary" in out or "Base" in out    # the search decision detail
    assert "gradient-boosted baseline" in out   # the hypothesis
    # Report footer carries the summary essentials.
    assert "promotions=1/1" in out
    assert "best=EXP002" in out


def test_render_has_a_progress_counter_over_nine_stages():
    out = _render(SCRIPTED)
    assert f"/{len(STAGES)}]" in out            # e.g. "[2/9]"
    assert f"[1/{len(STAGES)}]" in out          # Understanding task is stage 1


def test_first_agent_msg_opens_understanding_later_msgs_are_continuation():
    events = [
        {"type": ev.RUN_BEGIN, "task": "t", "metric": "auc", "metric_direction": "maximize"},
        {"type": ev.AGENT_MSG, "text": "first thought"},
        {"type": ev.AGENT_MSG, "text": "second thought"},
    ]
    out = _render(events)
    assert out.count("▸ Understanding task") == 1   # only the FIRST opens the stage
    assert "first thought" in out and "second thought" in out


def test_no_color_output_is_clean_plain_text():
    out = _render(SCRIPTED, color=False)
    assert "\033[" not in out                   # no ANSI colour codes
    assert "\r" not in out                      # no cursor returns


def test_renderer_defaults_to_no_color_under_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()                          # StringIO is not a TTY
    r = StageRenderer(stream=buf)
    r({"type": ev.RUN_BEGIN, "task": "t", "metric": "auc", "metric_direction": "maximize"})
    assert "\033[" not in buf.getvalue()


def test_renderer_never_raises_on_garbage_events():
    buf = io.StringIO()
    r = StageRenderer(color=False, stream=buf)
    for bad in ({}, {"type": "unknown"}, {"type": ev.SCORE}, {"type": ev.AGENT_MSG}):
        r(bad)                                   # must not raise


def test_thinking_is_noop_on_non_tty_stream():
    buf = io.StringIO()                          # not a TTY -> silent
    with thinking("thinking", stream=buf):
        pass
    assert buf.getvalue() == ""


class _GbkBuffer(io.StringIO):
    """A StringIO that reports a legacy gbk encoding, like a Windows console."""

    encoding = "gbk"


def test_render_degrades_to_ascii_on_gbk_console():
    buf = _GbkBuffer()
    r = StageRenderer(color=False, stream=buf)
    for e in SCRIPTED:
        r(e)
    out = buf.getvalue()
    # Fancy glyphs are replaced by ASCII, not dropped: the stages still render.
    assert "▸" not in out and "●" not in out and "→" not in out
    assert "> Understanding task" in out            # ASCII arrow header
    assert "* Research run" in out                  # ASCII bullet banner
    assert "0.8421" in out and "best=EXP002" in out  # facts survive intact
    # And every line must actually be gbk-encodable (would have raised otherwise).
    out.encode("gbk")


def test_thinking_honors_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    class _FakeTTY(io.StringIO):
        def isatty(self):  # pretend to be a terminal
            return True

    buf = _FakeTTY()
    with thinking("thinking", stream=buf):
        pass
    assert buf.getvalue() == ""                  # NO_COLOR suppresses the spinner


def test_jsonl_event_sink_writes_verifiable_events(tmp_path):
    """Verify JsonlEventSink actually writes to disk and events can be read back."""
    from research_os.events import JsonlEventSink, RUN_BEGIN, SCORE, PROMOTE, RUN_END

    path = tmp_path / "test_events.jsonl"
    sink = JsonlEventSink(path)
    events = [
        {"type": RUN_BEGIN, "task": "titanic", "metric": "accuracy", "seq": 1},
        {"type": SCORE, "exp_id": "EXP001", "cv_score": 0.85, "seq": 2},
        {"type": PROMOTE, "promoted": True, "best_exp_id": "EXP001", "seq": 3},
        {"type": RUN_END, "best_cv_score": 0.85, "n_promotions": 1, "seq": 4},
    ]
    for e in events:
        sink(e)

    import json
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    for i, (line, expected) in enumerate(zip(lines, events)):
        parsed = json.loads(line)
        assert parsed["type"] == expected["type"]
        assert parsed["seq"] == expected["seq"]


def test_preflight_stages_render_all_six():
    """Verify that preflight() renders all 6 preflight stages."""
    buf = io.StringIO()
    r = StageRenderer(color=False, stream=buf)
    stages = [
        ("Inspecting task", "task=titanic, metric=accuracy", "passed"),
        ("Checking data", "train.csv found, test.csv missing", "blocked"),
        ("Checking config", "provider=anthropic, model=claude-opus-4-8, ready=yes", "passed"),
        ("Selecting compute", "compute=local", "passed"),
        ("Planning experiment", "goal=improve baseline CV", "passed"),
        ("Entering workstation agent", "compute=local, events → events.jsonl", "passed"),
    ]
    for stage, detail, status in stages:
        r.preflight(stage, detail, status)

    out = buf.getvalue()
    for stage_name in ("Inspecting task", "Checking data", "Checking config",
                       "Selecting compute", "Planning experiment", "Entering workstation agent"):
        assert stage_name in out, f"missing preflight stage: {stage_name}"
    # Check status marks
    assert "✓" in out  # passed
    assert "⊘" in out  # blocked
    assert "titanic" in out
    assert "claude-opus-4-8" in out
    assert "local" in out
    assert "1/6" in out
    assert "6/6" in out
    # No ANSI in non-colour output
    assert "\033[" not in out
