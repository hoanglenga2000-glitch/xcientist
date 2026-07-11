"""End-to-end AgentSession test with a SCRIPTED fake LLM (no network/keys).

The fake client returns a fixed transcript of assistant turns — inspect_data,
run_experiment, evaluate_promotion, finish — exactly as a real model would drive
the tool-use loop. We assert:
  * the loop runs tools in order and feeds tool_results back as user turns;
  * a real (local subprocess) experiment executes and is promoted;
  * the dashboard-visible artifacts land: events.jsonl, search_graph.json,
    summary.json, best_solution.py — in the same layout the fixed loop uses.
"""
from __future__ import annotations

import json
import sys

from research_os import events as ev
from research_os.agent.messaging import AssistantTurn, ToolCall
from research_os.agent.session import AgentSession, AgentSessionConfig
from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import LocalSubprocessRunner, RunResult
from research_os.variation_generator import TaskContext

# A tiny but contract-honoring solution: prints CV_SCORE and writes both artifacts.
_GOOD_SOLUTION = """
import argparse, json, csv, os
p = argparse.ArgumentParser()
p.add_argument("--data-dir"); p.add_argument("--out-dir")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
print("CV_SCORE=0.8123")
with open(os.path.join(a.out_dir, "metrics.json"), "w") as f:
    json.dump({"cv_score": 0.8123, "metric": "accuracy"}, f)
with open(os.path.join(a.out_dir, "submission.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "target"]); w.writerow([1, 0])
"""


class _ScriptedClient:
    """Replays a fixed list of assistant turns; ignores the messages it's sent
    except to record that tool_results were threaded back."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.sends = 0
        self.saw_tool_result_turn = False

    def is_available(self):
        return True

    def send(self, messages, *, system, tools, max_tokens=0, temperature=0.0):
        # After the first tool call, the last message should be a user turn of
        # tool_result blocks — proof the loop fed results back.
        if messages and messages[-1]["role"] == "user":
            content = messages[-1]["content"]
            if isinstance(content, list) and content and content[0].get("type") == "tool_result":
                self.saw_tool_result_turn = True
        turn = self._turns[self.sends]
        self.sends += 1
        return turn


def _turn(text, tool_calls, stop):
    return AssistantTurn(text=text, tool_calls=tool_calls, stop_reason=stop,
                         raw_content=[{"type": "text", "text": text}], model="fake-model")


def _ctx():
    return TaskContext(task_name="titanic", modality="tabular", task_type="classification",
                       metric="accuracy", metric_direction="maximize", n_train=891, n_test=418)


def test_agent_session_runs_tools_and_writes_artifacts(tmp_path):
    exp_dir = tmp_path / "experiments" / "evolution" / "titanic_local_agent"
    runner = LocalSubprocessRunner(exp_dir / "runs", timeout=120, python_exe=sys.executable)
    toolbox = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"),
                              work_dir=exp_dir, runner=runner)

    script = [
        _turn("I'll ground myself, then plan.",
              [ToolCall("t1", "read_search_tree", {})], "tool_use"),
        _turn("Ask the MCGS brain what to expand.",
              [ToolCall("t2", "plan_next_experiment", {})], "tool_use"),
        _turn("Running a baseline for the planned node.",
              [ToolCall("t3", "run_experiment",
                        {"hypothesis": "GBM baseline", "code": _GOOD_SOLUTION})], "tool_use"),
        _turn("Let the gate rule.",
              [ToolCall("t4", "evaluate_promotion", {"exp_id": "EXP000"})], "tool_use"),
        _turn("Recording what I learned.",
              [ToolCall("t5", "record_lesson",
                        {"exp_id": "EXP000", "what_worked": "GBM baseline", "reusable_strategy": "gbm"})], "tool_use"),
        _turn("Done.", [ToolCall("t6", "finish", {"summary": "baseline cv=0.8123 promoted"})], "tool_use"),
    ]
    client = _ScriptedClient(script)
    session = AgentSession(context=_ctx(), toolbox=toolbox, exp_dir=exp_dir, client=client)
    summary = session.run("Establish a baseline and promote it.")

    # the loop consumed the whole script and fed tool_results back as user turns
    assert client.sends == 6
    assert client.saw_tool_result_turn is True

    # a real experiment ran and was promoted by the deterministic gate
    assert summary["best_exp_id"] == "EXP000"
    assert summary["best_cv_score"] == 0.8123
    assert summary["n_promotions"] == 1
    assert summary["finished_by_agent"] is True
    assert summary["mode"] == "agent"

    # dashboard-visible artifacts, same layout as the fixed loop
    assert (exp_dir / "events.jsonl").exists()
    assert (exp_dir / "summary.json").exists()
    assert (exp_dir / "search_graph.json").exists()
    assert (exp_dir / "best_solution.py").exists()
    graph = json.loads((exp_dir / "search_graph.json").read_text(encoding="utf-8"))
    assert graph["best_exp_id"] == "EXP000"


def test_agent_session_emits_expected_event_stream(tmp_path):
    exp_dir = tmp_path / "exp"
    runner = LocalSubprocessRunner(exp_dir / "runs", timeout=120, python_exe=sys.executable)
    toolbox = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"),
                              work_dir=exp_dir, runner=runner)
    script = [
        _turn("Plan.", [ToolCall("t1", "plan_next_experiment", {})], "tool_use"),
        _turn("Running.", [ToolCall("t2", "run_experiment",
              {"hypothesis": "h", "code": _GOOD_SOLUTION})], "tool_use"),
        _turn("Ruling.", [ToolCall("t3", "evaluate_promotion", {"exp_id": "EXP000"})], "tool_use"),
        _turn("Done.", [ToolCall("t4", "finish", {"summary": "ok"})], "tool_use"),
    ]
    session = AgentSession(context=_ctx(), toolbox=toolbox, exp_dir=exp_dir,
                           client=_ScriptedClient(script))
    session.run("baseline")

    types = [e["type"] for e in ev.read_events(exp_dir / "events.jsonl")]
    # lifecycle brackets the run; agent turns, tool calls/results, and the reused
    # score/promote joints all appear on the ONE stream the dashboard reads.
    assert types[0] == ev.RUN_BEGIN
    assert types[-1] == ev.RUN_END
    for expected in (ev.AGENT_MSG, ev.TOOL_CALL, ev.TOOL_RESULT, ev.SCORE, ev.PROMOTE):
        assert expected in types, f"missing {expected} in {types}"


def test_agent_session_stops_at_turn_budget(tmp_path):
    """If the model never calls finish, the session stops at max_turns (safety)."""
    exp_dir = tmp_path / "exp"
    toolbox = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir,
                              runner=LocalSubprocessRunner(exp_dir / "runs", timeout=30))
    # every turn just reads the (empty) tree and never finishes
    loop_turn = _turn("still going", [ToolCall("t", "read_search_tree", {})], "tool_use")
    client = _ScriptedClient([loop_turn] * 10)
    session = AgentSession(context=_ctx(), toolbox=toolbox, exp_dir=exp_dir, client=client,
                           config=AgentSessionConfig(max_turns=3))
    summary = session.run("go forever")
    assert summary["turns_used"] == 3
    assert summary["finished_by_agent"] is False
    assert summary["needs_continuation"] is True
    assert summary["stop_reason"] == "turn_budget_exhausted"


def test_agent_session_nudges_text_only_turn_then_requires_finish(tmp_path):
    exp_dir = tmp_path / "exp"
    toolbox = ResearchToolbox(
        _ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir,
        runner=LocalSubprocessRunner(exp_dir / "runs", timeout=30),
    )
    client = _ScriptedClient([
        _turn("I think this is done.", [], "end_turn"),
        _turn("Done honestly.", [ToolCall("finish", "finish", {"summary": "blocked: no experiment"})], "tool_use"),
    ])
    session = AgentSession(context=_ctx(), toolbox=toolbox, exp_dir=exp_dir, client=client)

    summary = session.run("inspect and finish")

    assert client.sends == 2
    assert summary["finished_by_agent"] is True
    assert summary["needs_continuation"] is False
    assert summary["stop_reason"] == "finished"
    assert any(
        message.get("role") == "user" and "not complete until you call finish" in str(message.get("content"))
        for message in session.messages
    )


def test_agent_session_marks_repeated_text_only_turn_as_continuation(tmp_path):
    exp_dir = tmp_path / "exp"
    toolbox = ResearchToolbox(
        _ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir,
        runner=LocalSubprocessRunner(exp_dir / "runs", timeout=30),
    )
    client = _ScriptedClient([
        _turn("Maybe done.", [], "end_turn"),
        _turn("Still no finish.", [], "end_turn"),
    ])
    session = AgentSession(context=_ctx(), toolbox=toolbox, exp_dir=exp_dir, client=client)

    summary = session.run("inspect and finish")

    assert summary["finished_by_agent"] is False
    assert summary["needs_continuation"] is True
    assert summary["stop_reason"] == "text_only_without_finish"


def test_cli_run_once_returns_nonzero_for_unfinished_session(tmp_path, monkeypatch):
    from xsci import agent as agent_cli

    class _Session:
        exp_dir = tmp_path / "exp"

        def run(self, goal):
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            return {"finished_by_agent": False, "needs_continuation": True, "stop_reason": "provider_error"}

    monkeypatch.setattr(agent_cli, "_record_evolution_summary", lambda *args, **kwargs: None)
    assert agent_cli._run_once(_Session(), "continue", quiet_summary=True, task_name="demo") == 2


def test_agent_session_halts_on_consecutive_tool_failures(tmp_path):
    """The guardrail halts the loop after N consecutive failing tool calls, rather
    than burning the whole turn budget flailing."""
    from research_os.agent.guardrails import ToolGuardrailController

    exp_dir = tmp_path / "exp"
    # a runner that always fails -> every run_experiment is a failure
    class _AlwaysFail:
        def run(self, code, *, data_dir, out_dir, exp_id):
            from pathlib import Path as _P
            _P(out_dir).mkdir(parents=True, exist_ok=True)
            return RunResult(success=False, cv_score=None, exit_code=1, error="boom")

    toolbox = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir,
                              runner=_AlwaysFail())
    # each turn plans then runs (the run fails); distinct code each time so the
    # halt is driven by the CONSECUTIVE-failure counter, not the identical-repeat block.
    turns = []
    for i in range(10):
        turns.append(_turn("plan", [ToolCall(f"p{i}", "plan_next_experiment", {})], "tool_use"))
        turns.append(_turn("run", [ToolCall(f"r{i}", "run_experiment",
                     {"hypothesis": "h", "code": f"print({i})  # variant {i}\n"})], "tool_use"))
    session = AgentSession(context=_ctx(), toolbox=toolbox, exp_dir=exp_dir,
                           client=_ScriptedClient(turns),
                           config=AgentSessionConfig(max_turns=40),
                           guardrails=ToolGuardrailController(consecutive_failure_halt=3))
    summary = session.run("keep failing")
    # halted well before the 40-turn budget (plan ok, run fail -> 3 fails halts it)
    assert summary["turns_used"] < 40
    assert session.guardrails.halted is True
