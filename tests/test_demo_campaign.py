from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_os.demo_campaign import (
    DEMO_ITERATIONS,
    DEMO_MAX_COST_USD,
    DEMO_MAX_TOKENS,
    DEMO_MAX_WALL_SECONDS,
    DEMO_TASK_ID,
    DemoVariationGenerator,
    freeze_and_review_demo_candidate,
    prepare_demo_dataset,
    verify_demo_evidence_chain,
)
from research_os.evolution_loop import EvolutionConfig, EvolutionLoop, LocalSubprocessRunner
from research_os.mcgs_selector import MCGSSelector
from research_os.variation_generator import TaskContext


def _execute_demo(tmp_path: Path, *, iterations: int):
    campaign_root = tmp_path / "campaign"
    manifest = prepare_demo_dataset(campaign_root)
    assert manifest["train_rows"] == 2400
    assert not (campaign_root / "data" / "holdout_labels.csv").exists()

    work_dir = tmp_path / "experiments" / DEMO_TASK_ID
    selector = MCGSSelector(
        total_steps=iterations,
        search_mode="experience_mcgs_v1",
        max_total_tokens=DEMO_MAX_TOKENS,
        max_wall_seconds=DEMO_MAX_WALL_SECONDS,
        max_cost_usd=DEMO_MAX_COST_USD,
    )
    context = TaskContext(
        task_name=DEMO_TASK_ID,
        modality="tabular",
        task_type="binary_classification",
        metric="roc_auc",
        metric_direction="maximize",
        target_column="churned",
        id_column="customer_id",
        data_schema="fixed synthetic customer churn demo",
        n_train=2400,
        n_test=600,
        compute_backend="cpu",
    )
    loop = EvolutionLoop(
        context,
        data_dir=str(campaign_root / "data"),
        work_dir=work_dir,
        runner=LocalSubprocessRunner(work_dir / "runs", timeout=120),
        generator=DemoVariationGenerator(),
        config=EvolutionConfig(max_iterations=iterations),
        selector=selector,
    )
    summary = loop.run(strategies=[])
    selector.export_observability(work_dir)
    loop.graph.export_json(work_dir / "search_graph.json")
    return campaign_root, work_dir, summary


def test_demo_campaign_executes_all_four_operators_and_sealed_review(tmp_path: Path):
    campaign_root, work_dir, summary = _execute_demo(tmp_path, iterations=DEMO_ITERATIONS)

    assert summary["n_iterations"] == DEMO_ITERATIONS
    assert summary["iterations"][2]["success"] is False
    assert summary["best_exp_id"]

    board = json.loads((work_dir / "experience-board.json").read_text(encoding="utf-8"))
    operators = {card["operator"] for card in board["cards"]}
    assert operators == {"Draft", "Improve", "Debug", "Crossover"}
    assert len(board["cards"]) == 8

    evidence = freeze_and_review_demo_candidate(work_dir, summary, campaign_root)
    assert evidence["review"]["status"] == "passed"
    assert evidence["review"]["rows"] == 600
    assert 0.5 <= evidence["review"]["score"] <= 1.0
    assert evidence["claim_audit"]["status"] == "passed"
    assert {item["decision"] for item in evidence["claim_audit"]["claims"]} == {"supported"}
    assert evidence["claim_audit"]["evidence"]["execution"]["executed_candidate_count"] == DEMO_ITERATIONS
    valid_crossovers = [
        item
        for item in evidence["claim_audit"]["evidence"]["search"]["crossover_lineages"]
        if item["valid"]
    ]
    assert valid_crossovers
    assert len(valid_crossovers[-1]["parent_exp_ids"]) == 2
    assert evidence["claim_audit"]["unsupported_claims"] == []
    assert (work_dir / "candidate-freeze.json").is_file()
    assert (work_dir / "independent-review.json").is_file()

    verified = verify_demo_evidence_chain(work_dir, campaign_root, summary=summary)
    assert verified["freeze_sha256"] == evidence["freeze_sha256"]
    assert verified["independent_review_sha256"] == evidence["independent_review_sha256"]
    assert verified["claim_audit_sha256"] == evidence["claim_audit_sha256"]

    best_submission = work_dir / str(summary["best_exp_id"]) / "out" / "submission.csv"
    original_submission = best_submission.read_bytes()
    best_submission.write_bytes(original_submission + b"\n")
    with pytest.raises(ValueError, match="candidate freeze artifact hash mismatch"):
        verify_demo_evidence_chain(work_dir, campaign_root, summary=summary)
    best_submission.write_bytes(original_submission)

    review_path = work_dir / "independent-review.json"
    original_review = review_path.read_bytes()
    review = json.loads(original_review.decode("utf-8"))
    review["score"] = float(review["score"]) - 0.001
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="claim audit is not bound to the current independent review"):
        verify_demo_evidence_chain(work_dir, campaign_root, summary=summary)
    review_path.write_bytes(original_review)

    claim_path = work_dir / "claim-audit.json"
    original_claim = claim_path.read_bytes()
    claim = json.loads(original_claim.decode("utf-8"))
    claim["evidence"]["candidate_freeze_sha256"] = "0" * 64
    claim_path.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="claim audit is not bound to the current candidate freeze"):
        verify_demo_evidence_chain(work_dir, campaign_root, summary=summary)
    claim_path.write_bytes(original_claim)


def test_three_node_run_is_held_without_an_eight_execution_claim(tmp_path: Path):
    campaign_root, work_dir, summary = _execute_demo(tmp_path, iterations=3)

    assert summary["n_iterations"] == 3
    evidence = freeze_and_review_demo_candidate(work_dir, summary, campaign_root)
    audit = evidence["claim_audit"]
    decisions = {item["claim_id"]: item for item in audit["claims"]}

    assert audit["status"] == "hold"
    assert decisions["execution_count"]["decision"] == "hold"
    assert decisions["operators_and_crossover"]["decision"] == "hold"
    assert decisions["independent_review"]["decision"] == "supported"
    assert audit["evidence"]["execution"]["executed_candidate_count"] == 3
    assert "3 candidate execution records" in decisions["execution_count"]["claim"]
    assert "Eight" not in json.dumps(audit["claims"], ensure_ascii=False)
    assert audit["unsupported_claims"] == audit["held_claims"]
    assert verify_demo_evidence_chain(work_dir, campaign_root, summary=summary)["claim_audit"]["status"] == "hold"
