from __future__ import annotations

import json

from research_os.evolution_loop import EvolutionConfig, EvolutionLoop, RunResult
from research_os.mcgs_selector import MCGSSelector
from research_os.variation_generator import TaskContext, VariationProposal
from research_os.search_graph import ExperimentNode


class Generator:
    def __init__(self):
        self.calls = []

    def propose(self, context, **kwargs):
        self.calls.append(kwargs)
        exp_id = kwargs["exp_id"]
        return VariationProposal(
            exp_id=exp_id,
            code=f"print('candidate {exp_id}')",
            hypothesis=f"hypothesis {exp_id}",
            changes_summary=f"change {exp_id}",
            applied_strategies=["tree" if exp_id != "EXP003" else "linear"],
            parent_exp_id=kwargs.get("parent_exp_id"),
            code_generation_mode=kwargs.get("mode", "Base"),
            llm_input_tokens=100,
            llm_output_tokens=50,
            prompt=f"prompt {exp_id}",
        )


class Runner:
    def __init__(self):
        self.index = 0

    def run(self, code, **kwargs):
        outcomes = [RunResult(True, 0.50, artifacts=["metrics.json", "submission.csv"]),
                    RunResult(False, None, error="ValueError: missing column", artifacts=[]),
                    RunResult(True, 0.60, artifacts=["metrics.json", "submission.csv"]),
                    RunResult(True, 0.61, artifacts=["metrics.json", "submission.csv"])]
        result = outcomes[self.index]
        self.index += 1
        return result


def test_evolution_loop_writes_experience_board_trace_and_budget(tmp_path):
    events = []
    generator = Generator()
    selector = MCGSSelector(total_steps=4, search_mode="experience_mcgs_v1", max_total_tokens=10_000, max_wall_seconds=60)
    loop = EvolutionLoop(
        TaskContext(task_name="fixture", modality="tabular", task_type="classification", metric="auc", metric_direction="maximize"),
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "run",
        runner=Runner(),
        generator=generator,
        selector=selector,
        config=EvolutionConfig(max_iterations=4),
        on_event=events.append,
    )
    summary = loop.run()
    assert summary["n_iterations"] == 4
    board = json.loads((tmp_path / "run" / "experience-board.json").read_text(encoding="utf-8"))
    ledger = json.loads((tmp_path / "run" / "budget-ledger.json").read_text(encoding="utf-8"))
    traces = (tmp_path / "run" / "selection-traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(board["cards"]) == 4
    assert {card["operator"] for card in board["cards"]} >= {"Draft", "Improve", "Debug"}
    assert ledger["total_tokens"] == 600  # all four executions, including the root Draft, are hard-budgeted
    assert ledger["nodes"] == 4
    assert ledger["terminal_reason"] == "node_budget_exhausted"
    assert len(traces) == 3
    parsed_traces = [json.loads(line) for line in traces]
    assert all(trace["operator"] in {"Improve", "Debug", "Crossover"} for trace in parsed_traces)
    assert all(trace["selection_reason"] for trace in parsed_traces)
    assert all("max_nodes" in trace["budget_snapshot"] for trace in parsed_traces)
    select_events = [event for event in events if event["type"].endswith("select") or event["type"] == "select"]
    assert any(event.get("operator") == "Debug" for event in select_events)
    assert any(call.get("experience_context") for call in generator.calls[1:])
    assert all("operator" in call for call in generator.calls)
    cache_stats = json.loads((tmp_path / "run" / "summary-cache-stats.json").read_text(encoding="utf-8"))
    assert cache_stats["hits"] + cache_stats["misses"] > 0


def test_experience_mode_selector_failure_is_audited_terminal_not_linear_fallback(tmp_path):
    class BrokenSelector(MCGSSelector):
        def select(self, graph, *, step):
            raise RuntimeError("selector evidence unavailable")

    runner = Runner()
    loop = EvolutionLoop(
        TaskContext(task_name="fixture", modality="tabular", task_type="classification", metric="auc", metric_direction="maximize"),
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "run",
        runner=runner,
        generator=Generator(),
        selector=BrokenSelector(total_steps=4, search_mode="experience_mcgs_v1"),
        config=EvolutionConfig(max_iterations=4),
    )
    summary = loop.run()
    assert summary["n_iterations"] == 0
    assert summary["terminal_reason"] == "selector_failed:RuntimeError"
    assert runner.index == 0
    assert loop.graph.nodes == {}


def test_crossover_integration_persists_two_parent_lineage_and_reference_edge(tmp_path):
    loop = EvolutionLoop(
        TaskContext(task_name="fixture", modality="tabular", task_type="classification", metric="auc", metric_direction="maximize"),
        data_dir=str(tmp_path / "data"),
        work_dir=tmp_path / "run",
        runner=Runner(),
        generator=Generator(),
        config=EvolutionConfig(max_iterations=1),
    )
    for exp_id, family, score in (("EXP000", "linear", 0.5), ("EXP001", "tree", 0.6)):
        loop.graph.add_node(ExperimentNode(
            exp_id=exp_id, parent_id=None, branch_type=family, task_name="fixture",
            hypothesis="h", implementation_summary=family, code_path=f"{exp_id}/solution.py",
            cv_score=score,
        ))
    loop.best_exp_id = "EXP001"
    proposal = VariationProposal(
        exp_id="EXP002", code="print(1)", hypothesis="cross", changes_summary="combine linear and tree",
        applied_strategies=["crossover"], parent_exp_id="EXP001", code_generation_mode="Stepwise",
    )
    loop._integrate(
        proposal,
        RunResult(True, 0.7, artifacts=["metrics.json", "submission.csv"]),
        tree_parent="EXP001",
        parent_exp_ids=["EXP001", "EXP000"],
    )
    node = loop.graph.nodes["EXP002"]
    assert node.parent_id == "EXP001"
    assert node.reference_parent_ids == ["EXP001", "EXP000"]
    assert any(edge["source"] == "EXP000" and edge["target"] == "EXP002" and edge["reference_type"] == "crossover_parent" for edge in loop.graph.reference_edges)
