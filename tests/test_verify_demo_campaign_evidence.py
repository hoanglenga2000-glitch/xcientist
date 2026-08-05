from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from research_os.experience_mcgs import (
    ExperienceBoard,
    ExperienceCardBuilder,
    SearchOperator,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_demo_campaign_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_demo_campaign_evidence", SCRIPT)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_board_verifier_recomputes_hash_chain_and_detects_tampering() -> None:
    first = ExperienceCardBuilder.build(
        task_id="fixture",
        node_id="EXP000",
        operator=SearchOperator.DRAFT,
        method_family="linear",
        code="def solve():\n    return 0\n",
        status="success",
        public_validation_score=0.5,
        provenance={"split": "public_validation"},
    )
    second = ExperienceCardBuilder.build(
        task_id="fixture",
        node_id="EXP001",
        parent_ids=["EXP000"],
        operator=SearchOperator.IMPROVE,
        method_family="tree",
        code="def solve():\n    return 1\n",
        status="success",
        public_validation_score=0.6,
        provenance={"split": "public_validation"},
    )
    current = ExperienceBoard("fixture")
    current.append(first)
    current.append(second)
    board = current.to_dict()
    result = verifier.verify_board(board)
    assert result["status"] == "passed"
    assert result["card_count"] == 2
    assert result["lineage_edge_count"] == 1

    board["cards"][0]["quality"] += 0.01
    tampered = verifier.verify_board(board)
    assert tampered["status"] == "failed"
    assert any("content-derived id" in error for error in tampered["errors"])
    assert "append_chain_head mismatch" in tampered["errors"]
    assert "board_hash mismatch" in tampered["errors"]


def test_cache_evidence_requires_real_card_miss_hit_and_retrieval() -> None:
    records = [
        {
            "cache_hit": False,
            "card_cache_hits": 0,
            "card_cache_misses": 1,
            "card_ids": ["exp_a"],
        },
        {
            # A bundle can be a top-level miss because one newly appended card
            # needs a summary while earlier cards are genuine cache hits.
            "cache_hit": False,
            "card_cache_hits": 1,
            "card_cache_misses": 0,
            "card_ids": ["exp_a"],
        },
    ]
    result = verifier.verify_cache_evidence(records, {"hits": 1, "misses": 1})
    assert result["summary_stats_consistent"] is True
    assert result["miss_observed"] is True
    assert result["hit_observed"] is True
    assert result["memory_used"] is True
    assert result["retrieval_bundle_hits"] == 0

    empty = verifier.verify_cache_evidence(
        [{"cache_hit": False, "card_cache_hits": 0, "card_cache_misses": 0, "card_ids": []}],
        {"hits": 0, "misses": 0},
    )
    assert empty["summary_stats_consistent"] is True
    assert empty["miss_observed"] is False
    assert empty["hit_observed"] is False
    assert empty["memory_used"] is False


def test_receipt_paths_default_to_run_root_and_keep_legacy_overrides(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    legacy_cli = tmp_path / "legacy-cli.json"
    legacy_input = tmp_path / "legacy-input.json"

    defaults = verifier.resolve_evidence_paths(run_dir)
    assert defaults == {
        "cli_receipt": (run_dir / "cli-receipt.json").resolve(),
        "input_contract": (run_dir / "input-contract.json").resolve(),
        "approval_receipt": (run_dir / "approval-receipt.json").resolve(),
        "cycle_receipt": (run_dir / "cycle-receipt.json").resolve(),
    }

    overridden = verifier.resolve_evidence_paths(
        run_dir,
        cli_log=legacy_cli,
        input_contract=legacy_input,
    )
    assert overridden["cli_receipt"] == legacy_cli.resolve()
    assert overridden["input_contract"] == legacy_input.resolve()
    assert overridden["approval_receipt"] == defaults["approval_receipt"]
    assert overridden["cycle_receipt"] == defaults["cycle_receipt"]


def test_approval_and_cycle_receipts_bind_plan_request_files_and_db_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "evomind_demo_customer_churn_local_fixture"
    run_dir.mkdir()
    input_contract = {
        "schema": verifier.INPUT_CONTRACT_SCHEMA,
        "task_id": verifier.EXPECTED_TASK_ID,
        "runner": "local",
        "iterations": 8,
        "requested_iterations": 8,
        "mcgs": True,
        "search_mode": "experience_mcgs_v1",
        "max_nodes": 8,
        "max_tokens": 100_000,
        "max_wall_seconds": 600,
        "max_cost": 0.01,
        "demo_campaign": True,
        "official_submission": "disabled",
        "created_at": "2026-08-02T17:00:01+00:00",
    }
    request_contract = {
        "task_id": verifier.EXPECTED_TASK_ID,
        "engine": "research_os",
        "runner": "local",
        "iterations": 8,
        "mcgs": True,
        "search_mode": "experience_mcgs_v1",
        "max_nodes": 8,
        "max_tokens": 100_000,
        "max_wall_seconds": 600,
        "max_cost": 0.01,
    }
    plan = {
        "task_id": verifier.EXPECTED_TASK_ID,
        "official_submit_allowed": False,
        "selected_branch": "EXP001",
    }
    unsigned_approval = {
        "schema": verifier.APPROVAL_SCHEMA,
        "hash_canonicalization": verifier.CANONICAL_HASH_SCHEMA,
        "plan_id": "50ba785a-bfb1-42be-a2b6-8e7faa57a3cc",
        "task_id": verifier.EXPECTED_TASK_ID,
        "status": "approved",
        "request_fingerprint": verifier.canonical_sha256(request_contract),
        "request_contract": request_contract,
        "plan_sha256": verifier.canonical_sha256(plan),
        "plan": plan,
        "created_at": "2026-08-02T16:58:10.646Z",
        "expires_at": "2026-08-02T17:58:10.646Z",
        "approved_at": "2026-08-02T16:58:42.150Z",
    }
    approval = {
        **unsigned_approval,
        "receipt_sha256": verifier.canonical_sha256(unsigned_approval),
    }

    def write_json(name: str, value: object) -> Path:
        path = run_dir / name
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    write_json("input-contract.json", input_contract)
    write_json("cli-receipt.json", {"schema": verifier.CLI_RECEIPT_SCHEMA})
    write_json("candidate-freeze.json", {"status": "frozen_before_independent_review"})
    write_json("independent-review.json", {"status": "passed"})
    write_json("claim-audit.json", {"status": "passed"})
    write_json("approval-receipt.json", approval)

    approval_result = verifier.verify_approval_receipt(approval, input_contract)
    assert approval_result["status"] == "passed"

    tampered_approval = {**approval, "request_contract": {**request_contract, "max_nodes": 9}}
    tampered_approval_result = verifier.verify_approval_receipt(
        tampered_approval,
        input_contract,
    )
    assert tampered_approval_result["status"] == "failed"
    assert "approval request_fingerprint mismatch" in tampered_approval_result["errors"]
    assert "approval receipt_sha256 mismatch" in tampered_approval_result["errors"]
    assert "approval request_contract does not match the fixed demo request" in tampered_approval_result["errors"]

    artifact_names = {
        "input-contract.json",
        "cli-receipt.json",
        "candidate-freeze.json",
        "independent-review.json",
        "claim-audit.json",
        "approval-receipt.json",
    }
    cycle = {
        "schema": verifier.CYCLE_RECEIPT_SCHEMA,
        "task_id": verifier.EXPECTED_TASK_ID,
        "run_id": "run_fixture_db_id",
        "output_dir_name": run_dir.name,
        "status": "completed",
        "plan_id": approval["plan_id"],
        "plan_sha256": approval["plan_sha256"],
        "request_fingerprint": approval["request_fingerprint"],
        "request_contract_sha256": verifier.canonical_sha256(request_contract),
        "approval_receipt_sha256": approval["receipt_sha256"],
        "artifact_hashes": {
            name: verifier.sha256_path(run_dir / name) for name in sorted(artifact_names)
        },
        "official_submit_allowed": False,
        "completed_at": "2026-08-02T17:05:00.000Z",
    }
    write_json("cycle-receipt.json", cycle)

    passed = verifier.verify_cycle_receipt(
        cycle,
        approval,
        approval_result,
        run_dir,
        db_run_id="run_fixture_db_id",
    )
    assert passed["status"] == "passed"
    assert passed["errors"] == []

    tampered = dict(cycle)
    tampered["artifact_hashes"] = dict(cycle["artifact_hashes"])
    tampered["artifact_hashes"]["claim-audit.json"] = "0" * 64
    failed = verifier.verify_cycle_receipt(
        tampered,
        approval,
        approval_result,
        run_dir,
        db_run_id="different_db_run",
    )
    assert failed["status"] == "failed"
    assert "cycle receipt artifact SHA-256 mismatch" in failed["errors"]
    assert "cycle receipt run_id does not match exact Prisma experiment run" in failed["errors"]


def test_private_feedback_is_rejected_in_keys_and_string_values() -> None:
    assert verifier.find_forbidden_evaluation_values({"notes": "public validation only"}) == []
    findings = verifier.find_forbidden_evaluation_values(
        {"safe_key": ["copy private grader feedback", {"leaderboard_score": 0.9}]}
    )
    assert "root.safe_key[0]" in findings
    assert "root.safe_key[1].leaderboard_score" in findings


def test_dual_parent_crossover_is_cross_artifact_not_card_only() -> None:
    cards = [
        {"node_id": "EXP000", "operator": "Draft", "status": "success", "method_family": "linear", "parent_ids": []},
        {"node_id": "EXP001", "operator": "Improve", "status": "success", "method_family": "tree", "parent_ids": ["EXP000"]},
        {"node_id": "EXP007", "operator": "Crossover", "status": "success", "method_family": "voting", "parent_ids": ["EXP000", "EXP001"]},
    ]
    graph = {
        "nodes": [
            {"exp_id": "EXP000", "reference_parent_ids": []},
            {"exp_id": "EXP001", "reference_parent_ids": ["EXP000"]},
            {"exp_id": "EXP007", "reference_parent_ids": ["EXP000", "EXP001"]},
        ],
        "reference_edges": [
            {"source": "EXP001", "target": "EXP007", "reference_type": "crossover_parent"}
        ],
    }
    traces = [
        {"operator": "Crossover", "selected_parent_ids": ["EXP000", "EXP001"]}
    ]
    events = [
        {"type": "select", "exp_id": "EXP007", "operator": "Crossover", "parent_exp_ids": ["EXP000", "EXP001"]},
        {"type": "propose", "exp_id": "EXP007", "parent_exp_ids": ["EXP000", "EXP001"]},
    ]
    assert verifier.verify_dual_parent_crossover(cards, graph, traces, events)["status"] == "passed"

    graph["reference_edges"] = []
    failed = verifier.verify_dual_parent_crossover(cards, graph, traces, events)
    assert failed["status"] == "failed"
    assert any("secondary crossover reference edge" in error for error in failed["errors"])


def test_legacy_r4_report_fails_closed_on_hash_and_recording_memory_gates() -> None:
    run_dir = (
        ROOT
        / "experiments"
        / "evolution"
        / "evomind_demo_customer_churn_local_20260802_194345_607945"
    )
    campaign_dir = ROOT / "workspace" / "demo_campaigns" / "evomind_demo_customer_churn"
    cli_log = ROOT / "artifacts" / "release-0.3.0-20260802" / "qa" / "demo-r4-cli.log"
    input_contract = campaign_dir / "r4-demo-input.json"
    workstation_db = ROOT / "web" / "research-agent-workstation" / "prisma" / "workstation.db"
    if not all(path.is_file() if path.suffix else path.is_dir() for path in (run_dir, campaign_dir, cli_log, input_contract, workstation_db)):
        pytest.skip("r4 demo evidence is not present in this checkout")

    report = verifier.verify_demo_campaign_evidence(
        run_dir,
        campaign_dir,
        cli_log=cli_log,
        input_contract=input_contract,
        workstation_db=workstation_db,
    )
    assert report["status"] == "failed_closed"
    assert report["recording_ready"] is False
    failures = {item["id"] for item in report["checks"] if item["blocking"] and item["status"] == "failed"}
    assert failures == {
        "cache_miss_observed",
        "cache_hit_observed",
        "prompt_memory_retrieval_used",
        "ui_run_binding",
        "board_append_order_hash_chain",
        "approval_receipt_binding",
        "cycle_receipt_binding",
    }
    assert report["candidate_execution"] == {"scheduled_and_executed": 8, "successful": 7, "failed": 1}
    assert report["verifier_actions"][-2:] == ["no_private_grader", "no_kaggle_submission"]
