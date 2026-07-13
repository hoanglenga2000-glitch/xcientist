"""Correct, tested MCGS selection brain for the evolution loop.

This is the *selection* half of a Monte-Carlo-Graph-Search evolutionary loop:
given the current ``research_os.search_graph.SearchGraph`` it decides WHICH node
to expand next, WHAT expansion type to use (primary / intra-branch / cross-branch
/ aggregation) and WHICH coding mode (base / stepwise / diff). Execution and the
promotion gate stay entirely in ``evolution_loop`` / ``search_graph`` — this file
never runs code and never mutates the audited node schema.

Why this exists (design note): the workstation already had a MCGS engine
(``strategy/mlevolve_search.py``) with the right *ideas* — piecewise exploration
decay, four expansion types, branch/global stagnation. But its UCT was
non-functional: ``visit_count`` was declared, read, and never incremented, so
``_uct_value`` always hit the ``visits==0 -> inf`` path and selection collapsed to
"first child". It also divided a final CV score by visits (direction-unaware, wrong
for minimize) and had zero tests. Rather than import that liability across packages
(and inherit a second ``SearchGraph`` name clash), we port the *sound* ideas here,
operate on B's own graph, keep a private visit side-table, and use B's
direction-aware ``_is_better`` so UCT is correct for maximize AND minimize.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ── plan the loop asks for each round ────────────────────────────────────────
@dataclass
class ExpansionPlan:
    node_exp_id: str                 # node to expand FROM (not necessarily global best)
    expansion_type: str              # primary | intra_branch | cross_branch | aggregation
    coding_mode: str                 # Base | Stepwise | Diff  (matches VariationGenerator)
    reference_exp_ids: list[str] = field(default_factory=list)
    branch_id: str = ""              # branch the *new* child will belong to
    phase: str = "exploration"       # exploration | balanced | exploitation (for observability)


# ── progressive exploration schedule (ported from the sound part of A) ───────
def _phase_for(progress: float) -> str:
    """Map normalized search progress [0,1] to a coarse phase."""
    if progress < 0.3:
        return "exploration"
    if progress < 0.7:
        return "balanced"
    return "exploitation"


def _piecewise_decay(step: int, total_steps: int, *, initial_c: float = 1.414,
                     lower_bound: float = 0.5) -> float:
    """Exploration constant C decays from 1.414 -> 0.5 across the budget.

    Flat during early exploration, linear decay through the middle, floored late.
    Scales to whatever total budget the caller uses (unlike A's hard-coded 500).
    """
    if total_steps <= 1:
        return lower_bound
    t1 = max(1, round(total_steps * 0.3))
    t2 = max(t1 + 1, round(total_steps * 0.7))
    if step < t1:
        return initial_c
    if step <= t2:
        frac = (step - t1) / max(1, (t2 - t1))
        return max(initial_c - (initial_c - lower_bound) * frac, lower_bound)
    return lower_bound


class MCGSSelector:
    """Selection brain over a ``research_os.search_graph.SearchGraph``.

    Keeps its OWN state (visit counts, branch membership, stagnation) in private
    side-tables keyed by ``exp_id`` so B's audited node schema is never touched.
    """

    def __init__(self, *, total_steps: int = 6, exploration_c: float = 1.414,
                 branch_stagnation_patience: int = 2, global_stagnation_patience: int = 4,
                 min_delta: float = 1e-4, max_branches: int = 3) -> None:
        self.total_steps = max(1, total_steps)
        self.exploration_c = exploration_c
        self.branch_stagnation_patience = branch_stagnation_patience
        self.global_stagnation_patience = global_stagnation_patience
        self.min_delta = min_delta
        self.max_branches = max_branches
        # private side-tables (never written back to ExperimentNode)
        self.visits: dict[str, int] = {}
        self.branch_of: dict[str, str] = {}          # exp_id -> branch_id
        self.branch_stagnation: dict[str, int] = {}  # branch_id -> consecutive no-improve
        self.global_stagnation = 0
        self._branch_counter = 0

    # ── reward normalization: make "better" always mean higher, any direction ──
    def _reward(self, graph, node) -> float:
        """Map a node's score to [0,1] where 1 is best under the metric direction.

        This is the fix for A's bug: A used ``score/visits`` which is wrong for a
        final CV score and ignores minimize metrics. We normalize against the range
        of evaluated scores and flip for minimize, using the graph's own direction.
        """
        score = graph._node_score(node)
        if score is None:
            return 0.0
        scored = [graph._node_score(n) for n in graph.nodes.values()]
        scored = [s for s in scored if s is not None]
        if len(scored) < 2:
            return 0.5
        lo, hi = min(scored), max(scored)
        if hi <= lo:
            return 0.5
        direction = (graph.metric_direction or "maximize").lower()
        if direction in {"minimize", "lower", "lower_is_better"}:
            return (hi - score) / (hi - lo)
        return (score - lo) / (hi - lo)

    def _uct(self, graph, node, parent_visits: int, c: float) -> float:
        """UCT = exploitation(reward) + C * sqrt(ln(parent_visits)/visits).

        Unvisited nodes return +inf so every frontier gets tried once (the correct
        MCTS behavior A only *looked* like it had).
        """
        v = self.visits.get(node.exp_id, 0)
        if v == 0:
            return float("inf")
        exploitation = self._reward(graph, node)
        exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / v)
        return exploitation + exploration

    def _children(self, graph, exp_id: str) -> list:
        return [n for n in graph.nodes.values() if n.parent_id == exp_id]

    def select(self, graph, *, step: int) -> ExpansionPlan:
        """Walk from root by UCT to a frontier node, then decide expansion + mode.

        Falls back gracefully: if the graph has no root yet, the caller should be
        seeding the baseline (this returns a primary/Base plan on the root id).
        """
        c = _piecewise_decay(step, self.total_steps, initial_c=self.exploration_c)
        phase = _phase_for(step / self.total_steps)
        root_id = graph.root_exp_id
        # traverse to a leaf by best UCT child
        node = graph.nodes.get(root_id)
        if node is None:
            return ExpansionPlan(node_exp_id=root_id, expansion_type="primary",
                                 coding_mode="Base", branch_id="branch_0", phase=phase)
        while True:
            children = self._children(graph, node.exp_id)
            if not children:
                break
            parent_visits = self.visits.get(node.exp_id, 0)
            node = max(children, key=lambda ch: self._uct(graph, ch, parent_visits, c))
        return self._plan_expansion(graph, node, phase=phase)

    # ── helpers on B's graph ────────────────────────────────────────────────
    def _depth(self, graph, node) -> int:
        d, cur = 0, node
        seen = set()
        while cur.parent_id and cur.parent_id in graph.nodes and cur.exp_id not in seen:
            seen.add(cur.exp_id)
            cur = graph.nodes[cur.parent_id]
            d += 1
        return d

    def _branch(self, exp_id: str) -> str:
        return self.branch_of.get(exp_id, "branch_0")

    def _ancestors_in_branch(self, graph, node, k: int = 3) -> list[str]:
        out, cur, bid = [], node, self._branch(node.exp_id)
        seen = {node.exp_id}   # cycle-guard, consistent with _depth: never loop forever
        while cur.parent_id and cur.parent_id in graph.nodes and len(out) < k:
            if cur.parent_id in seen:
                break
            seen.add(cur.parent_id)
            cur = graph.nodes[cur.parent_id]
            if self._branch(cur.exp_id) == bid:
                out.append(cur.exp_id)
        return out

    def _top_exp_ids(self, graph, *, n: int, exclude_branch: Optional[str] = None) -> list[str]:
        scored = [(self._reward(graph, nd), nd.exp_id) for nd in graph.nodes.values()
                  if graph._node_score(nd) is not None
                  and (exclude_branch is None or self._branch(nd.exp_id) != exclude_branch)]
        scored.sort(reverse=True)
        return [eid for _, eid in scored[:n]]

    def _coding_mode(self, graph, node, *, phase: str) -> str:
        depth = self._depth(graph, node)
        branch_age = sum(1 for e in self.branch_of.values() if e == self._branch(node.exp_id))
        if phase == "exploration" and depth <= 1:
            return "Base"
        if branch_age >= 4 and phase != "exploration":
            return "Diff"
        if depth <= 2:
            return "Stepwise"
        return "Diff"

    def _plan_expansion(self, graph, node, *, phase: str) -> ExpansionPlan:
        bid = self._branch(node.exp_id)
        n_branches = len(set(self.branch_of.values())) or 1
        # 1) aggregation: global stall + enough branches to fuse
        if self.global_stagnation >= self.global_stagnation_patience and n_branches >= 2:
            refs = self._top_exp_ids(graph, n=4)
            new_bid = self._new_branch() if n_branches < self.max_branches else bid
            return ExpansionPlan(node.exp_id, "aggregation", "Base", refs,
                                 branch_id=new_bid, phase=phase)
        # 2) this branch stalled. Two sub-cases:
        #    (a) other branches exist -> borrow their ideas (cross_branch).
        #    (b) single-branch world -> DIVERSIFY: open a fresh branch from the global
        #        best. Without (b) the search can never leave branch_0, so cross_branch
        #        AND aggregation are unreachable dead code (the multi-branch brain needs
        #        a second branch to bootstrap, but only these paths create one).
        if self.branch_stagnation.get(bid, 0) >= self.branch_stagnation_patience:
            refs = self._top_exp_ids(graph, n=3, exclude_branch=bid)
            if refs:
                new_bid = self._new_branch() if n_branches < self.max_branches else bid
                return ExpansionPlan(node.exp_id, "cross_branch", "Stepwise", refs,
                                     branch_id=new_bid, phase=phase)
            if n_branches < self.max_branches:
                seed = (self._top_exp_ids(graph, n=1) or [node.exp_id])[0]
                return ExpansionPlan(seed, "primary", "Base", [],
                                     branch_id=self._new_branch(), phase=phase)
        # 3) intra-branch: refine using this branch's own history (post-exploration)
        if phase != "exploration":
            hist = self._ancestors_in_branch(graph, node, k=3)
            if hist:
                mode = self._coding_mode(graph, node, phase=phase)
                return ExpansionPlan(node.exp_id, "intra_branch", mode, hist,
                                     branch_id=bid, phase=phase)
        # 4) primary: straightforward improvement of the selected node
        mode = self._coding_mode(graph, node, phase=phase)
        return ExpansionPlan(node.exp_id, "primary", mode, [], branch_id=bid, phase=phase)

    def _new_branch(self) -> str:
        self._branch_counter += 1
        return f"branch_{self._branch_counter}"

    # ── backpropagation: the increment A never did + stagnation bookkeeping ──
    def register_child(self, plan: ExpansionPlan, child_exp_id: str) -> None:
        """Record which branch the newly-created child belongs to."""
        self.branch_of[child_exp_id] = plan.branch_id or self._branch(plan.node_exp_id)

    def backpropagate(self, graph, exp_id: str, *, improved: bool) -> None:
        """Increment visit counts up the ancestor chain and update stagnation.

        ``improved`` is whether this node improved the global best (the caller
        already knows this from the promotion decision, so we don't recompute it).
        """
        # visit increment along the path to root (this is what makes UCT live)
        cur = graph.nodes.get(exp_id)
        seen = set()
        while cur is not None and cur.exp_id not in seen:
            seen.add(cur.exp_id)
            self.visits[cur.exp_id] = self.visits.get(cur.exp_id, 0) + 1
            cur = graph.nodes.get(cur.parent_id) if cur.parent_id else None
        # stagnation bookkeeping
        bid = self._branch(exp_id)
        if improved:
            self.global_stagnation = 0
            self.branch_stagnation[bid] = 0
        else:
            self.global_stagnation += 1
            self.branch_stagnation[bid] = self.branch_stagnation.get(bid, 0) + 1



