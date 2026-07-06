"""Context-compaction tests: the search graph is the durable state.

Prove: the trigger respects threshold + anti-thrash; the deterministic state block
reflects the graph (best/failed/pending) without inventing scores; compaction
protects the head, splices the state block, keeps a safe tail (no orphaned
tool_result), and the session compacts mid-run when the prompt grows.
"""
from __future__ import annotations

import sys
from pathlib import Path

from research_os.agent.context import (
    build_research_state_block, compact_messages, estimate_tokens, should_compact,
)
from research_os.agent.session import AgentSession, AgentSessionConfig
from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import RunResult
from research_os.variation_generator import TaskContext


def _ctx():
    return TaskContext(task_name="t", modality="tabular", task_type="classification",
                       metric="accuracy", metric_direction="maximize")


# ── trigger logic ─────────────────────────────────────────────────────────────
def test_should_compact_respects_threshold_and_min_messages():
    msgs = [{"role": "user", "content": "x"}] * 10
    # below threshold → no
    assert should_compact(prompt_tokens=100, messages=msgs,
                          threshold_tokens=1000, last_compact_tokens=0) is False
    # above threshold → yes
    assert should_compact(prompt_tokens=2000, messages=msgs,
                          threshold_tokens=1000, last_compact_tokens=0) is True
    # too few messages → no, even if huge
    assert should_compact(prompt_tokens=9999, messages=msgs[:3],
                          threshold_tokens=1000, last_compact_tokens=0) is False


def test_should_compact_anti_thrash():
    msgs = [{"role": "user", "content": "x"}] * 10
    # just barely above the last compaction size → don't recompact yet
    assert should_compact(prompt_tokens=1050, messages=msgs,
                          threshold_tokens=1000, last_compact_tokens=1000) is False
    # grown >10% past last compaction → recompact
    assert should_compact(prompt_tokens=1200, messages=msgs,
                          threshold_tokens=1000, last_compact_tokens=1000) is True


def test_estimate_tokens_counts_content():
    msgs = [{"role": "user", "content": "a" * 400}]
    assert estimate_tokens(msgs) == 100


# ── deterministic state block from the graph ────────────────────────────────────
def _toolbox_with_history(tmp_path):
    class _Runner:
        def __init__(self):
            self._scores = iter([0.80, 0.85])
        def run(self, code, *, data_dir, out_dir, exp_id):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return RunResult(success=True, cv_score=next(self._scores, 0.85), exit_code=0,
                             artifacts=["/x/metrics.json", "/x/submission.csv"])

    from research_os.mcgs_selector import MCGSSelector
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "exp",
                         runner=_Runner(), selector=MCGSSelector(total_steps=8))
    for _ in range(2):
        tb.dispatch("plan_next_experiment", {})
        exp_id = f"EXP{tb._exp_counter:03d}"
        tb.dispatch("run_experiment", {"hypothesis": "gbm", "code": "print(1)"})
        tb.dispatch("evaluate_promotion", {"exp_id": exp_id})
    return tb


def test_state_block_reflects_graph(tmp_path):
    tb = _toolbox_with_history(tmp_path)
    block = build_research_state_block(tb)
    assert "RESEARCH STATE SO FAR" in block
    assert "BEST:" in block
    assert "EXPERIMENTS DONE" in block
    assert "EXP000" in block and "EXP001" in block
    # the best is a real promoted node, not invented
    assert tb.best_exp_id in block
    assert "you cannot fake either" in block  # invariant reminder present


def test_state_block_lists_failures(tmp_path):
    class _Fail:
        def run(self, code, *, data_dir, out_dir, exp_id):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return RunResult(success=False, cv_score=None, exit_code=1, error="boom")

    from research_os.mcgs_selector import MCGSSelector
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "exp",
                         runner=_Fail(), selector=MCGSSelector(total_steps=8))
    tb.dispatch("plan_next_experiment", {})
    tb.dispatch("run_experiment", {"hypothesis": "risky idea", "code": "print(1)"})
    block = build_research_state_block(tb)
    assert "FAILED" in block
    assert "risky idea" in block
    assert "BEST: (none promoted yet)" in block


# ── compaction preserves boundaries ─────────────────────────────────────────────
def test_compact_protects_head_and_splices_state():
    messages = [{"role": "user", "content": "SEED goal"}]
    for i in range(10):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]})
        messages.append({"role": "user", "content": [{"type": "tool_result",
                         "tool_use_id": f"t{i}", "content": "r", "is_error": False}]})
    out = compact_messages(messages, state_block="STATE-BLOCK", tail_turns=4)
    # head preserved + state block spliced in
    assert out[0]["role"] == "user"
    assert "SEED goal" in out[0]["content"] and "STATE-BLOCK" in out[0]["content"]
    # much shorter than the original
    assert len(out) < len(messages)


def test_compact_tail_starts_on_assistant_no_orphan_tool_result():
    """The tail must not begin on a tool_result (that would orphan it from its
    tool_use, which the API rejects)."""
    messages = [{"role": "user", "content": "SEED"}]
    for i in range(8):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]})
        messages.append({"role": "user", "content": [{"type": "tool_result",
                         "tool_use_id": f"t{i}", "content": "r"}]})
    out = compact_messages(messages, state_block="S", tail_turns=5)
    # first message after the head must be an assistant turn
    assert out[1]["role"] == "assistant"


def test_compact_noop_when_short():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert compact_messages(messages, state_block="S", tail_turns=6) == messages


# ── session compacts mid-run ────────────────────────────────────────────────────
class _BigPromptClient:
    """Reports a large prompt-token count so the session triggers compaction, then
    finishes. Records the message count it saw on the final send."""
    def __init__(self):
        self.sends = 0
        self.saw_lengths = []

    def is_available(self):
        return True

    def send(self, messages, *, system, tools, max_tokens=0, temperature=0.0):
        from research_os.agent.messaging import AssistantTurn, ToolCall
        self.sends += 1
        self.saw_lengths.append(len(messages))
        if self.sends == 1:
            # first turn: several tool calls to grow the history, report big tokens
            return AssistantTurn(text="working", tool_calls=[ToolCall("t1", "read_search_tree", {})],
                                 stop_reason="tool_use", raw_content=[{"type": "text", "text": "working"}],
                                 model="m", input_tokens=200_000)
        return AssistantTurn(text="done", tool_calls=[ToolCall("t2", "finish", {"summary": "ok"})],
                             stop_reason="tool_use", raw_content=[{"type": "text", "text": "done"}],
                             model="m", input_tokens=1000)


def test_session_compacts_when_prompt_grows(tmp_path):
    exp_dir = tmp_path / "exp"
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=exp_dir,
                         runner=type("R", (), {"run": lambda *a, **k: RunResult(True, 0.5, exit_code=0)})())
    # pre-load the history so there are enough messages to compact
    session = AgentSession(context=_ctx(), toolbox=tb, exp_dir=exp_dir,
                           client=_BigPromptClient(),
                           config=AgentSessionConfig(compact_threshold_tokens=100_000,
                                                     compact_tail_turns=4))
    for i in range(10):
        session.messages.append({"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]})
        session.messages.append({"role": "user", "content": "u"})
    session._last_prompt_tokens = 200_000  # pretend the last send was big
    session.run("go")
    from research_os import events as ev
    types = [e["type"] for e in ev.read_events(exp_dir / "events.jsonl")]
    assert ev.COMPACTION in types
