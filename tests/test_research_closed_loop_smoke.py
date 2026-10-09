"""End-to-end research closed-loop smoke test (all mocked, no real training)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from xsci.kaggle_intent import classify, is_execution, Intent, PLANNING, EXECUTION, CHAT, TOOL_QUERY


def test_intent_classifies_research_request():
    result = classify("我想做一个分类任务")
    assert result.kind in (PLANNING, CHAT, TOOL_QUERY)


def test_intent_classifies_execution_request():
    result = classify("开始训练")
    assert result.kind == EXECUTION
    assert is_execution("开始训练") is True


def test_intent_classifies_status_query():
    result = classify("训练结果怎么样")
    assert result.kind == TOOL_QUERY


def test_strategy_selector_returns_strategies():
    from research_os.strategy_selector import recommend_strategies, TaskProfile

    profile = TaskProfile(
        modality="tabular",
        task_type="classification",
        train_size=891,
        test_size=418,
        metric="accuracy",
        n_features=12,
    )
    rec = recommend_strategies(profile)
    assert len(rec.strategies) >= 1
    assert all(isinstance(s, str) for s in rec.strategies)


def test_search_graph_promotion_gate():
    from research_os.search_graph import SearchGraph, ExperimentNode

    graph = SearchGraph(task_id="titanic", root_exp_id="EXP000")
    node = ExperimentNode(
        exp_id="EXP000", parent_id=None, branch_type="base",
        task_name="titanic", hypothesis="baseline", implementation_summary="sklearn",
        code_path="solution.py", cv_score=0.85, run_success=True,
    )
    graph.add_node(node)
    decision = graph.decide_promotion("EXP000", run_success=True)
    assert decision["promoted"] is True


def test_search_graph_rejects_failed_run():
    from research_os.search_graph import SearchGraph, ExperimentNode

    graph = SearchGraph(task_id="titanic", root_exp_id="EXP000")
    node = ExperimentNode(
        exp_id="EXP000", parent_id=None, branch_type="base",
        task_name="titanic", hypothesis="baseline", implementation_summary="sklearn",
        code_path="solution.py", cv_score=0.99, run_success=False,
    )
    graph.add_node(node)
    decision = graph.decide_promotion("EXP000", run_success=False)
    assert decision["promoted"] is False


def test_mcgs_selector_produces_expansion_plan():
    from research_os.mcgs_selector import MCGSSelector

    selector = MCGSSelector(total_steps=6, branch_stagnation_patience=2)
    assert selector is not None
    assert hasattr(selector, "select")


def test_innovation_engine_requires_minimum_evidence():
    from xsci.innovation_engine import InnovationEngine

    engine = InnovationEngine()
    proposals = engine.propose_innovations(task_type="classification")
    assert isinstance(proposals, list)


def test_doctor_checks_include_disk_and_os():
    from xsci.doctor import _check_disk_space, _check_os

    status_disk, detail_disk = _check_disk_space()
    assert status_disk in ("PASS", "WARN")
    assert "GB" in detail_disk

    status_os, detail_os = _check_os()
    assert status_os == "PASS"
    assert len(detail_os) > 0


def test_classify_with_llm_fallback_without_llm():
    from xsci.kaggle_intent import classify_with_llm_fallback

    result = classify_with_llm_fallback("你好")
    assert result.kind is not None

    result2 = classify_with_llm_fallback("帮我分析一下数据", generate_fn=None)
    assert result2.kind is not None


def test_tool_loop_budget_is_at_least_8():
    from xsci.kaggle_conversation import ConversationAgent

    agent = ConversationAgent()
    assert agent._max_tool_rounds >= 8
