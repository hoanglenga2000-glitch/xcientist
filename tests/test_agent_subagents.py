"""Sub-agent tests: the read-only audit agent CANNOT mutate (hard whitelist).

Prove: a whitelisted toolbox advertises only allowed tools and hard-denies the
rest (run_experiment/evaluate_promotion refused even if the model calls them); the
audit agent shares the parent's real graph; and spawn returns a structured summary
only (never the child transcript), with a depth cap.
"""
from __future__ import annotations

from pathlib import Path

from research_os.agent.messaging import AssistantTurn, ToolCall
from research_os.agent.subagents import AUDIT_TOOLS, spawn_audit_agent
from research_os.agent.tools import ResearchToolbox
from research_os.evolution_loop import RunResult
from research_os.variation_generator import TaskContext


def _ctx():
    return TaskContext(task_name="t", modality="tabular", task_type="classification",
                       metric="accuracy", metric_direction="maximize")


class _Runner:
    def run(self, code, *, data_dir, out_dir, exp_id):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return RunResult(success=True, cv_score=0.9, exit_code=0,
                         artifacts=["/x/metrics.json", "/x/submission.csv"])


def _scripted(turns):
    class _C:
        def __init__(self):
            self.sends = 0
        def is_available(self):
            return True
        def send(self, messages, *, system, tools, max_tokens=0, temperature=0.0):
            t = turns[self.sends]
            self.sends += 1
            return t
    return _C()


def _turn(text, calls, stop="tool_use"):
    return AssistantTurn(text=text, tool_calls=calls, stop_reason=stop,
                         raw_content=[{"type": "text", "text": text}], model="m")


def test_whitelisted_toolbox_only_advertises_allowed(tmp_path):
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e",
                         runner=_Runner(), allowed_tools=AUDIT_TOOLS)
    names = {s.name for s in tb.specs()}
    # only audit tools (+ finish), never the mutating ones
    assert names <= (AUDIT_TOOLS | {"finish"})
    assert "run_experiment" not in names
    assert "evaluate_promotion" not in names
    assert "submit_to_kaggle" not in names


def test_whitelisted_toolbox_hard_denies_mutating_tools(tmp_path):
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e",
                         runner=_Runner(), allowed_tools=AUDIT_TOOLS)
    # even if the model calls a forbidden tool directly, dispatch refuses it
    for forbidden in ("run_experiment", "evaluate_promotion", "record_lesson", "submit_to_kaggle"):
        out = tb.dispatch(forbidden, {"hypothesis": "h", "code": "print(1)", "exp_id": "EXP000"})
        assert out.ok is False
        assert "not permitted" in out.content
    # nothing ran / mutated
    assert tb.graph.nodes == {}


def test_audit_tools_can_still_read(tmp_path):
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e",
                         runner=_Runner(), allowed_tools=AUDIT_TOOLS)
    out = tb.dispatch("read_search_tree", {})
    assert out.ok is True  # reads are allowed


def _parent_with_one_result(tmp_path):
    tb = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e",
                         runner=_Runner())
    tb.dispatch("plan_next_experiment", {})
    tb.dispatch("run_experiment", {"hypothesis": "gbm baseline", "code": "print(1)"})
    tb.dispatch("evaluate_promotion", {"exp_id": "EXP000"})
    return tb


def test_spawn_audit_shares_parent_graph_and_returns_summary(tmp_path):
    parent = _parent_with_one_result(tmp_path)
    # scripted audit child: read the tree, audit the conclusion, finish
    child_client = _scripted([
        _turn("Reading the tree.", [ToolCall("a1", "read_search_tree", {})]),
        _turn("Auditing.", [ToolCall("a2", "audit_conclusion",
              {"exp_id": "EXP000", "claim": "EXP000 improved the baseline"})]),
        _turn("Verdict in.", [ToolCall("a3", "finish", {"summary": "conclusion supported by EXP000"})]),
    ])
    result = spawn_audit_agent(parent, goal="audit the current best claim",
                               client=child_client, exp_dir=tmp_path / "audit")
    assert result.role == "audit"
    assert result.status == "ok"
    assert "supported" in result.summary
    # the audit saw the REAL parent node
    assert result.detail.get("best_exp_id") == "EXP000"
    # it returned a compact brief, not a transcript
    assert "sub-agent audit" in result.to_brief()


def test_audit_child_cannot_mutate_parent_graph(tmp_path):
    """Even if the audit child tries run_experiment/evaluate_promotion, the shared
    graph is unchanged — the whitelist denies mutation."""
    parent = _parent_with_one_result(tmp_path)
    nodes_before = dict(parent.graph.nodes)
    best_before = parent.best_exp_id
    child_client = _scripted([
        _turn("Trying to run (should be denied).",
              [ToolCall("a1", "run_experiment", {"hypothesis": "sneaky", "code": "print(2)"})]),
        _turn("Trying to promote (should be denied).",
              [ToolCall("a2", "evaluate_promotion", {"exp_id": "EXP000"})]),
        _turn("Give up.", [ToolCall("a3", "finish", {"summary": "could not mutate (as designed)"})]),
    ])
    spawn_audit_agent(parent, goal="try to cheat", client=child_client, exp_dir=tmp_path / "audit")
    # parent graph is byte-for-byte the same set of nodes; best unchanged
    assert set(parent.graph.nodes) == set(nodes_before)
    assert parent.best_exp_id == best_before


def test_spawn_depth_cap(tmp_path):
    parent = _parent_with_one_result(tmp_path)
    result = spawn_audit_agent(parent, goal="x", client=_scripted([]),
                               depth=1, max_spawn_depth=1)
    assert result.status == "error"
    assert "depth" in result.summary


# ── request_audit: the top-level agent can delegate to the auditor, the auditor cannot ──
def test_request_audit_advertised_to_top_level_but_not_to_auditor(tmp_path):
    top = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e",
                          runner=_Runner())
    assert "request_audit" in {s.name for s in top.specs()}
    # the read-only auditor's whitelist excludes it -> no nested audits are offered
    auditor = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e2",
                              runner=_Runner(), allowed_tools=AUDIT_TOOLS)
    assert "request_audit" not in {s.name for s in auditor.specs()}
    assert auditor.dispatch("request_audit", {}).ok is False  # hard-denied too


def test_request_audit_without_spawner_degrades_gracefully(tmp_path):
    # no session wired => no spawner; the tool must refuse cleanly, not crash
    tb = _parent_with_one_result(tmp_path)
    assert tb.audit_spawner is None
    out = tb.dispatch("request_audit", {"focus": "is EXP000 real?"})
    assert out.ok is False
    assert "not available" in out.content


def test_request_audit_invokes_spawner_and_returns_brief(tmp_path):
    tb = _parent_with_one_result(tmp_path)
    seen = {}

    def _fake_spawner(focus: str) -> str:
        seen["focus"] = focus
        return "[sub-agent audit] status=ok turns=3\nSUPPORTED: EXP000 gain is real."

    tb.audit_spawner = _fake_spawner  # what AgentSession wires for the top-level agent
    out = tb.dispatch("request_audit", {"focus": "check CV leakage"})
    assert out.ok is True
    assert seen["focus"] == "check CV leakage"
    assert "SUPPORTED" in out.content


def test_session_wires_audit_spawner_for_top_level_only(tmp_path):
    """AgentSession gives the unrestricted toolbox a spawner; a restricted (audit)
    toolbox is left with none, so audits can never nest."""
    from research_os.agent.session import AgentSession

    top = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e",
                          runner=_Runner())
    AgentSession(context=_ctx(), toolbox=top, exp_dir=tmp_path / "s", client=_scripted([]))
    assert callable(top.audit_spawner)

    auditor = ResearchToolbox(_ctx(), data_dir=str(tmp_path / "d"), work_dir=tmp_path / "e2",
                              runner=_Runner(), allowed_tools=AUDIT_TOOLS)
    AgentSession(context=_ctx(), toolbox=auditor, exp_dir=tmp_path / "s2", client=_scripted([]))
    assert auditor.audit_spawner is None
