"""Read-only MLEvolve reference adapter for the Research OS.

The adapter extracts configuration-level search policy from the local
`external-projects/MLEvolve` checkout and converts it into workstation-friendly
control records. It never launches MLEvolve, trains models, or submits to
Kaggle; execution must still go through AgentOrchestrator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MLEvolveSearchPolicy:
    source_repo: str
    time_budget_hours: float
    initial_drafts: int
    parallel_search_num: int
    num_drafts: int
    num_improves: int
    top_candidates_size: int
    branch_stagnation_threshold: int
    topk_stagnation_threshold: int
    stagnation_window: int
    explore_switch_start: float
    explore_switch_end: float
    min_exploration_weight: float
    fusion_min_time_hours: float
    fusion_max_time_hours: float
    fusion_min_successful_nodes: int
    fusion_min_branches: int
    use_diff_mode: bool
    use_stepwise_generation: bool
    use_evolution: bool
    use_fusion: bool
    use_aggregation: bool
    use_global_memory: bool
    memory_similarity_threshold: float
    workstation_constraints: list[str] = field(default_factory=lambda: [
        "workstation_agent_orchestrator_required",
        "no_codex_direct_training",
        "no_official_submit_without_human_gate",
        "all_claims_require_artifacts",
    ])


def _parse_scalar(raw: str) -> Any:
    value = raw.split("#", 1)[0].strip().strip('"').strip("'")
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_mlevolve_config(config_path: str | Path) -> dict[str, Any]:
    """Parse the small subset of YAML needed for policy extraction.

    The project already depends on YAML elsewhere, but this parser keeps the
    adapter dependency-light and avoids importing MLEvolve runtime modules.
    """

    path = Path(config_path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            continue
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip() == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            value_text = raw_value.strip()
            if value_text.startswith("[") and value_text.endswith("]"):
                parent[key] = [
                    _parse_scalar(item.strip())
                    for item in value_text.strip("[]").split(",")
                    if item.strip()
                ]
            else:
                parent[key] = _parse_scalar(value_text)
    return root


def extract_policy(repo_root: str | Path) -> MLEvolveSearchPolicy:
    repo = Path(repo_root)
    config = load_mlevolve_config(repo / "config" / "config.yaml")
    agent = config.get("agent", {})
    search = agent.get("search", {})
    return MLEvolveSearchPolicy(
        source_repo=str(repo),
        time_budget_hours=float(agent.get("time_limit", 43200)) / 3600.0,
        initial_drafts=int(agent.get("initial_drafts", 3)),
        parallel_search_num=int(search.get("parallel_search_num", 3)),
        num_drafts=int(search.get("num_drafts", 5)),
        num_improves=int(search.get("num_improves", 3)),
        top_candidates_size=int(search.get("top_candidates_size", 20)),
        branch_stagnation_threshold=int(search.get("branch_stagnation_threshold", 3)),
        topk_stagnation_threshold=int(search.get("topk_stagnation_threshold", 6)),
        stagnation_window=int(search.get("stagnation_window", 4)),
        explore_switch_start=float(search.get("explore_switch_start", 0.5)),
        explore_switch_end=float(search.get("explore_switch_end", 0.7)),
        min_exploration_weight=float(search.get("min_exploration_weight", 0.2)),
        fusion_min_time_hours=float(search.get("fusion_min_time_hours", 6)),
        fusion_max_time_hours=float(search.get("fusion_max_time_hours", 10)),
        fusion_min_successful_nodes=int(search.get("fusion_min_successful_nodes", 2)),
        fusion_min_branches=int(search.get("fusion_min_branches", 2)),
        use_diff_mode=bool(agent.get("use_diff_mode", True)),
        use_stepwise_generation=bool(agent.get("use_stepwise_generation", True)),
        use_evolution=bool(agent.get("use_evolution", True)),
        use_fusion=bool(agent.get("use_fusion", True)),
        use_aggregation=bool(agent.get("use_aggregation", True)),
        use_global_memory=bool(agent.get("use_global_memory", True)),
        memory_similarity_threshold=float(agent.get("memory_similarity_threshold", 0.7)),
    )


def build_workstation_alignment(policy: MLEvolveSearchPolicy) -> dict[str, Any]:
    return {
        "schema": "academic_research_os.mlevolve_alignment_matrix.v1",
        "source_repo": policy.source_repo,
        "mlevolve_reference_policy": asdict(policy),
        "workstation_mapping": {
            "Progressive MCGS": {
                "workstation_component": "SearchController + SearchGraph",
                "status": "partially_integrated",
                "next_required_work": [
                    "persist branch score trajectories per task",
                    "use exploration stage to schedule AgentOrchestrator runs",
                    "turn stagnation into cross-branch fusion work orders",
                ],
            },
            "Retrospective Memory": {
                "workstation_component": "retrospective_memory + task_benchmark_state",
                "status": "partially_integrated",
                "next_required_work": [
                    "write every hold, timeout, and top30_failed into reusable memory",
                    "retrieve success and failure records before code-agent prompts",
                ],
            },
            "Adaptive Code Generation": {
                "workstation_component": "DeepSeek/Claude Code Agent gated draft flow",
                "status": "partially_integrated",
                "next_required_work": [
                    "bind Base/Stepwise/Diff to branch stage and failure count",
                    "show generated code artifacts in Code Agent workspace",
                ],
            },
            "Cross-Branch Fusion": {
                "workstation_component": "top30_next_evolution_orders + branch-diverse candidates",
                "status": "planned",
                "next_required_work": [
                    "select representatives from promoted nodes only",
                    "use held nodes as negative evidence, not blend inputs",
                ],
            },
            "MLE-Bench Evaluation": {
                "workstation_component": "benchmark_manager + benchmark/mle_bench_75",
                "status": "skeleton_ready_not_comparable",
                "next_required_work": [
                    "register real 75-task list",
                    "run same-budget task batches",
                    "record official/private medal evidence where available",
                ],
            },
        },
        "claim_boundary": "This matrix supports migration planning only. It is not benchmark performance evidence.",
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
