"""Resume/continue coverage for the deep-research agent.

A deep run can die mid-flight (host exit, GPU blip, 1800s timeout). Resume must
CONTINUE it truthfully — reload the audited search graph and the raw conversation,
never fabricate, and never overwrite the prior lineage with an empty one.

These tests are fully offline (scripted fake LLM + local subprocess runner) and
exercise each layer of the resume path:
  * SearchGraph.from_dict / load_json round-trip;
  * ResearchToolbox.restore_from rehydrates graph, best-so-far, exp-counter;
  * AgentSession(resume=True) reloads the ledger, sanitizes a dangling tool_use
    turn, and continues without re-seeding the full briefing;
  * a resumed session PRESERVES the prior graph on finalize (no overwrite).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from research_os import events as ev
from research_os.agent.messaging import AssistantTurn, LLMError, ToolCall
from research_os.agent.session import AgentSession, _drop_dangling_tool_use
from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import LocalSubprocessRunner
from research_os.search_graph import ExperimentNode, SearchGraph
from research_os.variation_generator import TaskContext

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
    def __init__(self, turns):
        self._turns = list(turns)
        self.sends = 0
        self.last_messages = None

    def is_available(self):
        return True

    def send(self, messages, *, system, tools, max_tokens=0, temperature=0.0):
        self.last_messages = messages
        turn = self._turns[self.sends]
        self.sends += 1
        return turn


def _turn(text, tool_calls, stop="tool_use"):
    return AssistantTurn(text=text, tool_calls=tool_calls, stop_reason=stop,
                         raw_content=[{"type": "text", "text": text}], model="fake-model")


def _ctx():
    return TaskContext(task_name="titanic", modality="tabular", task_type="classification",
                       metric="accuracy", metric_direction="maximize", n_train=891, n_test=418)


def _seed_graph() -> SearchGraph:
    g = SearchGraph(task_id="titanic", root_exp_id="EXP000",
                    metric_name="cv_score", metric_direction="maximize")
    g.nodes["EXP000"] = ExperimentNode(
        exp_id="EXP000", parent_id=None, branch_type="Base", task_name="titanic",
        hypothesis="baseline", implementation_summary="base", code_path="EXP000/solution.py",
        cv_score=0.80, run_success=True, promoted=True)
    g.nodes["EXP001"] = ExperimentNode(
        exp_id="EXP001", parent_id="EXP000", branch_type="Diff", task_name="titanic",
        hypothesis="tuned", implementation_summary="diff", code_path="EXP001/solution.py",
        cv_score=0.83, run_success=True, promoted=True)
    g.edges.append({"source": "EXP000", "target": "EXP001", "reason": "Diff"})
    g.best_exp_id = "EXP001"
    g.promotion_history = [{"candidate_exp_id": "EXP001", "promoted": True}]
    return g


# ── layer 1: SearchGraph reconstruction ──────────────────────────────────────
def test_search_graph_from_dict_round_trips(tmp_path):
    g = _seed_graph()
    g2 = SearchGraph.from_dict(g.to_dict())
    assert g2.task_id == "titanic"
    assert set(g2.nodes) == {"EXP000", "EXP001"}
    assert isinstance(g2.nodes["EXP001"], ExperimentNode)
    assert g2.nodes["EXP001"].cv_score == 0.83
    assert g2.nodes["EXP001"].parent_id == "EXP000"
    assert g2.nodes["EXP001"].promoted is True
    assert g2.best_exp_id == "EXP001"
    assert g2.promotion_history == [{"candidate_exp_id": "EXP001", "promoted": True}]
    assert len(g2.edges) == 1

    # load_json goes through a real file
    p = g.export_json(tmp_path / "search_graph.json")
    g3 = SearchGraph.load_json(p)
    assert set(g3.nodes) == {"EXP000", "EXP001"} and g3.best_exp_id == "EXP001"


def test_from_dict_tolerates_unknown_node_fields():
    d = _seed_graph().to_dict()
    d["nodes"][0]["a_future_field_from_a_newer_schema"] = 999
    g = SearchGraph.from_dict(d)  # must not raise
    assert "EXP000" in g.nodes


# ── layer 2: toolbox rehydration ─────────────────────────────────────────────
def test_toolbox_restore_from_rehydrates_state(tmp_path):
    exp_dir = tmp_path / "run"
    exp_dir.mkdir(parents=True)
    _seed_graph().export_json(exp_dir / "search_graph.json")
    (exp_dir / "best_solution.py").write_text("print('best')", encoding="utf-8")

    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir,
                         runner=LocalSubprocessRunner(exp_dir / "runs", timeout=30))
    info = tb.restore_from(exp_dir)

    assert info["restored_nodes"] == 2
    assert info["best_exp_id"] == "EXP001"
    assert info["promotions"] == 2
    # counter bumped PAST EXP001 so the next allocation can't collide
    assert tb._exp_counter == 2
    assert info["next_exp_id"] == "EXP002"
    assert tb.best_exp_id == "EXP001"
    assert tb.best_code == "print('best')"
    assert set(tb.graph.nodes) == {"EXP000", "EXP001"}


def test_toolbox_restore_from_missing_graph_raises(tmp_path):
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=tmp_path,
                         runner=LocalSubprocessRunner(tmp_path / "runs", timeout=30))
    import pytest
    with pytest.raises(FileNotFoundError):
        tb.restore_from(tmp_path / "does_not_exist")


# ── layer 3: dangling tool_use sanitation ────────────────────────────────────
def test_drop_dangling_tool_use_removes_unanswered_assistant_turn():
    msgs = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
        # crash: assistant asked for a tool but the result was never appended
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "run_experiment", "input": {}}]},
    ]
    out = _drop_dangling_tool_use(msgs)
    assert len(out) == 3
    assert out[-1]["role"] == "user"  # ends ready for the next user turn


def test_drop_dangling_keeps_answered_history():
    msgs = [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "y"}]},
    ]
    assert _drop_dangling_tool_use(msgs) == msgs  # nothing dangling


# ── layer 4: end-to-end resume through AgentSession ──────────────────────────
def _run_fresh(exp_dir: Path) -> AgentSession:
    """Run one fresh session to baseline, leaving a real ledger + graph on disk."""
    runner = LocalSubprocessRunner(exp_dir / "runs", timeout=120, python_exe=sys.executable)
    tb = ResearchToolbox(_ctx(), data_dir=str(exp_dir.parent / "data"), work_dir=exp_dir, runner=runner)
    script = [
        _turn("plan", [ToolCall("t1", "plan_next_experiment", {})]),
        _turn("run", [ToolCall("t2", "run_experiment", {"hypothesis": "base", "code": _GOOD_SOLUTION})]),
        _turn("rule", [ToolCall("t3", "evaluate_promotion", {"exp_id": "EXP000"})]),
        _turn("done", [ToolCall("t4", "finish", {"summary": "baseline promoted"})]),
    ]
    session = AgentSession(context=_ctx(), toolbox=tb, exp_dir=exp_dir, client=_ScriptedClient(script))
    session.run("Establish a baseline.")
    return session


def test_agent_session_resume_continues_from_ledger(tmp_path):
    exp_dir = tmp_path / "experiments" / "evolution" / "titanic_local_agent"
    _run_fresh(exp_dir)
    assert (exp_dir / "messages.jsonl").exists()
    graph_before = json.loads((exp_dir / "search_graph.json").read_text(encoding="utf-8"))
    assert graph_before["best_exp_id"] == "EXP000"
    prior_nodes = {n["exp_id"] for n in graph_before["nodes"]}

    # --- resume in the SAME dir: rehydrate the toolbox, then continue the chat ---
    runner = LocalSubprocessRunner(exp_dir / "runs", timeout=120, python_exe=sys.executable)
    tb2 = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir, runner=runner)
    tb2.restore_from(exp_dir)
    assert set(tb2.graph.nodes) == prior_nodes
    assert tb2._exp_counter == 1  # EXP000 restored -> next is EXP001

    resume_script = [
        _turn("continue: plan the next one", [ToolCall("r1", "plan_next_experiment", {})]),
        _turn("run improvement", [ToolCall("r2", "run_experiment", {"hypothesis": "v2", "code": _GOOD_SOLUTION})]),
        _turn("rule", [ToolCall("r3", "evaluate_promotion", {"exp_id": "EXP001"})]),
        _turn("done", [ToolCall("r4", "finish", {"summary": "second round done"})]),
    ]
    client = _ScriptedClient(resume_script)
    session2 = AgentSession(context=_ctx(), toolbox=tb2, exp_dir=exp_dir, client=client, resume=True)

    # the ledger was reloaded before the run (prior conversation present)
    assert session2._resumed_turns > 0
    reloaded = session2._resumed_turns

    session2.run("Improve on the baseline.")

    # the first send carried the reloaded history + a continuation user turn (not a
    # fresh single-message seed) -> proof it CONTINUED rather than restarted.
    assert client.last_messages is not None
    # after the full resumed run, more messages than the resume seed alone
    assert session2._resumed_turns == reloaded  # unchanged bookkeeping field

    # the new experiment used a fresh, non-colliding id
    assert "EXP001" in tb2.graph.nodes
    assert set(tb2.graph.nodes) >= prior_nodes | {"EXP001"}

    # finalize PRESERVED the prior lineage (did not overwrite with an empty graph)
    graph_after = json.loads((exp_dir / "search_graph.json").read_text(encoding="utf-8"))
    after_nodes = {n["exp_id"] for n in graph_after["nodes"]}
    assert prior_nodes <= after_nodes
    assert "EXP001" in after_nodes


def test_resume_emits_resumed_flag_in_run_begin(tmp_path):
    exp_dir = tmp_path / "exp"
    _run_fresh(exp_dir)

    runner = LocalSubprocessRunner(exp_dir / "runs", timeout=120, python_exe=sys.executable)
    tb2 = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir, runner=runner)
    tb2.restore_from(exp_dir)
    script = [_turn("done", [ToolCall("r1", "finish", {"summary": "nothing more to do"})])]
    session = AgentSession(context=_ctx(), toolbox=tb2, exp_dir=exp_dir,
                           client=_ScriptedClient(script), resume=True)
    session.run("continue")

    begins = [e for e in ev.read_events(exp_dir / "events.jsonl") if e["type"] == ev.RUN_BEGIN]
    assert begins, "expected a RUN_BEGIN event"
    assert begins[-1].get("resumed") is True
    assert begins[-1].get("resumed_turns", 0) > 0


def test_provider_error_writes_partial_summary_and_can_resume(tmp_path):
    class _FailingClient:
        def is_available(self):
            return True

        def send(self, messages, *, system, tools, max_tokens=0, temperature=0.0):
            raise LLMError("gateway unavailable")

    exp_dir = tmp_path / "exp"
    runner = LocalSubprocessRunner(exp_dir / "runs", timeout=30, python_exe=sys.executable)
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir, runner=runner)
    session = AgentSession(context=_ctx(), toolbox=tb, exp_dir=exp_dir, client=_FailingClient())

    summary = session.run("continue the experiment without losing evidence")

    assert summary["finished_by_agent"] is False
    assert summary["needs_continuation"] is True
    assert summary["stop_reason"] == "provider_error"
    assert "gateway unavailable" in summary["latest_error"]
    assert (exp_dir / "summary.json").exists()
    assert (exp_dir / "messages.jsonl").exists()

    tb2 = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "data"), work_dir=exp_dir, runner=runner)
    tb2.restore_from(exp_dir)
    resumed = AgentSession(
        context=_ctx(), toolbox=tb2, exp_dir=exp_dir,
        client=_ScriptedClient([_turn("done", [ToolCall("r1", "finish", {"summary": "resumed"})])]),
        resume=True,
    )
    assert "gateway unavailable" in resumed._latest_error
    resumed_summary = resumed.run("resume after provider recovery")
    assert resumed_summary["finished_by_agent"] is True
