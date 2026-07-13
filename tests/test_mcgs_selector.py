"""Offline, deterministic tests for the MCGS selection brain.

No LLM, no runner: these drive MCGSSelector against small hand-built SearchGraphs
so the selection logic is provable in isolation. The strongest checks target the
exact defects found in the old workstation engine: visit counts must actually
increment (its UCT was dead), and exploitation reward must respect metric
direction (its score/visits was wrong for minimize).
"""
from __future__ import annotations

from research_os.mcgs_selector import MCGSSelector
from research_os.search_graph import ExperimentNode, SearchGraph


def _node(exp_id: str, parent: str | None, cv: float | None) -> ExperimentNode:
    return ExperimentNode(
        exp_id=exp_id, parent_id=parent, branch_type="Base", task_name="t",
        hypothesis="h", implementation_summary="s", code_path=f"{exp_id}/solution.py",
        cv_score=cv, metric_name="cv_score", metric_direction="maximize",
    )


def _graph(direction: str = "maximize") -> SearchGraph:
    return SearchGraph(task_id="t", root_exp_id="EXP000",
                       metric_name="cv_score", metric_direction=direction)


def test_backprop_increments_visits_up_the_chain():
    """Locks the exact bug in the old engine: visit_count never incremented."""
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.80)
    g.nodes["EXP001"] = _node("EXP001", "EXP000", 0.82)
    sel = MCGSSelector(total_steps=6)
    assert sel.visits.get("EXP001", 0) == 0
    sel.backpropagate(g, "EXP001", improved=True)
    # both the node and its ancestor must have been visited
    assert sel.visits["EXP001"] == 1
    assert sel.visits["EXP000"] == 1
    sel.backpropagate(g, "EXP001", improved=False)
    assert sel.visits["EXP001"] == 2
    assert sel.visits["EXP000"] == 2  # accumulates up the chain


def test_reward_respects_minimize_direction():
    """For a minimize metric (e.g. RMSLE), the LOWER score must reward higher."""
    g = _graph(direction="minimize")
    g.nodes["EXP000"] = _node("EXP000", None, 0.10)   # worse (higher) for minimize
    g.nodes["EXP001"] = _node("EXP001", "EXP000", 0.02)  # better (lower)
    g.metric_direction = "minimize"
    sel = MCGSSelector(total_steps=6)
    r_worse = sel._reward(g, g.nodes["EXP000"])
    r_better = sel._reward(g, g.nodes["EXP001"])
    assert r_better > r_worse, "minimize: lower score should give higher reward"


def test_uct_prefers_unvisited_child_first():
    """An unvisited frontier node returns +inf UCT so it is tried once before
    exploitation kicks in (the correct MCTS behavior the old engine faked)."""
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.80)
    g.nodes["EXP001"] = _node("EXP001", "EXP000", 0.90)  # visited, high score
    g.nodes["EXP002"] = _node("EXP002", "EXP000", 0.50)  # unvisited
    sel = MCGSSelector(total_steps=6)
    sel.visits["EXP000"] = 2
    sel.visits["EXP001"] = 2   # EXP001 visited; EXP002 not
    plan = sel.select(g, step=0)
    assert plan.node_exp_id == "EXP002"  # unvisited beats even the higher scorer


def test_root_only_graph_returns_seed_plan():
    """Before any node exists, select() must return a safe primary/Base seed plan
    on the root id (so the loop can bootstrap a baseline without crashing)."""
    g = _graph()
    sel = MCGSSelector(total_steps=6)
    plan = sel.select(g, step=0)
    assert plan.node_exp_id == "EXP000"
    assert plan.expansion_type == "primary"
    assert plan.coding_mode == "Base"


def test_register_child_assigns_branch_membership():
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.80)
    sel = MCGSSelector(total_steps=6)
    plan = sel.select(g, step=0)          # primary on EXP000, branch_0
    sel.register_child(plan, "EXP001")
    assert sel.branch_of["EXP001"] == plan.branch_id


def test_branch_stall_triggers_cross_then_global_stall_triggers_aggregation():
    """The core evolution behavior: a stalled branch reaches to other branches
    (cross_branch); when the WHOLE search stalls with multiple branches, it fuses
    (aggregation). This is exactly what the old system could not do."""
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.80)
    g.nodes["EXP001"] = _node("EXP001", "EXP000", 0.95)  # a strong node in ANOTHER branch
    sel = MCGSSelector(total_steps=8, branch_stagnation_patience=2, global_stagnation_patience=3)
    # put EXP000 and EXP001 in different branches
    sel.branch_of["EXP000"] = "branch_0"
    sel.branch_of["EXP001"] = "branch_1"
    # stall branch_0 twice -> cross_branch expansion, referencing the better branch_1
    sel.branch_stagnation["branch_0"] = 2
    plan = sel._plan_expansion(g, g.nodes["EXP000"], phase="balanced")
    assert plan.expansion_type == "cross_branch"
    assert "EXP001" in plan.reference_exp_ids  # reached to the stronger other branch

    # now drive a GLOBAL stall with >=2 branches -> aggregation (fuse top nodes)
    sel.global_stagnation = 3
    plan2 = sel._plan_expansion(g, g.nodes["EXP000"], phase="exploitation")
    assert plan2.expansion_type == "aggregation"
    assert len(plan2.reference_exp_ids) >= 1


def test_single_branch_stall_diversifies_into_a_new_branch():
    """The bootstrap fix: in a single-branch world a stalled branch has nothing to
    cross to, so it must OPEN A NEW BRANCH (diversify from the best). Without this
    the search can never leave branch_0 and cross_branch/aggregation are unreachable
    dead code — the multi-branch brain needs a second branch to bootstrap itself."""
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.80)
    g.nodes["EXP001"] = _node("EXP001", "EXP000", 0.80)   # flat: branch stalled
    sel = MCGSSelector(total_steps=8, branch_stagnation_patience=2, max_branches=3)
    sel.branch_of = {"EXP000": "branch_0", "EXP001": "branch_0"}
    sel.branch_stagnation["branch_0"] = 2                 # branch_0 has stalled
    plan = sel._plan_expansion(g, g.nodes["EXP001"], phase="balanced")
    assert plan.expansion_type == "primary"               # a fresh line of attack
    assert plan.branch_id not in ("", "branch_0")         # on a genuinely NEW branch
    assert len(set(sel.branch_of.values())) == 1          # (child not registered yet)


def test_diversify_respects_max_branches_cap():
    """Diversification must not run away: once branches hit the cap, a stalled branch
    with nothing to cross to falls through to intra_branch instead of spawning more."""
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.80)
    g.nodes["EXP001"] = _node("EXP001", "EXP000", 0.80)
    sel = MCGSSelector(total_steps=8, branch_stagnation_patience=2, max_branches=1)
    sel.branch_of = {"EXP000": "branch_0", "EXP001": "branch_0"}
    sel.branch_stagnation["branch_0"] = 2
    plan = sel._plan_expansion(g, g.nodes["EXP001"], phase="balanced")
    assert plan.expansion_type != "primary"               # no new branch at the cap
    assert len(set(sel.branch_of.values())) == 1


def test_cold_start_reaches_multi_branch_and_aggregation():
    """End-to-end (no hand-injected branches): drive select->register->backprop with
    flat scores. The selector must escape single-branch on its own and, once the whole
    search stalls with >=2 branches, fire aggregation. This is the capability my trace
    proved was previously impossible (nbranch stuck at 1 forever)."""
    g = _graph()
    g.nodes["EXP000"] = _node("EXP000", None, 0.50)
    sel = MCGSSelector(total_steps=12, branch_stagnation_patience=2,
                       global_stagnation_patience=4, max_branches=3)
    seen: set[str] = set()
    for i in range(1, 11):
        plan = sel.select(g, step=i)
        child = f"EXP{i:03d}"
        g.nodes[child] = _node(child, plan.node_exp_id, 0.50)  # flat -> forces stalls
        g.add_edge(plan.node_exp_id, child, "Base")
        sel.register_child(plan, child)
        sel.backpropagate(g, child, improved=False)
        seen.add(plan.expansion_type)
    assert len(set(sel.branch_of.values())) >= 2   # escaped the single-branch trap
    assert "aggregation" in seen                    # global stall -> fusion fired


