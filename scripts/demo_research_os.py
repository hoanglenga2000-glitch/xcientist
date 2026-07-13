"""Run a local Research OS skeleton demo without external services."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research_os import (  # noqa: E402
    ExperimentNode,
    SearchGraph,
    audit_claim,
    check_required_artifacts,
    create_contract,
    evaluate_acceptance,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    demo_dir = ROOT / "examples" / "research_os_demo"
    nodes_payload = load_json(demo_dir / "sample_experiment_nodes.json")
    graph_payload = load_json(demo_dir / "sample_search_graph.json")
    contract_payload = load_json(demo_dir / "sample_validation_contract.json")

    graph = SearchGraph(
        task_id=graph_payload["task_id"],
        root_exp_id=graph_payload["root_exp_id"],
        selected_next_branch=graph_payload["selected_next_branch"],
        exploration_stage=graph_payload["exploration_stage"],
    )

    for payload in nodes_payload:
        graph.add_node(ExperimentNode(**payload))

    for edge in graph_payload["edges"]:
        graph.add_edge(edge["source"], edge["target"], edge.get("reason", ""))

    top_candidates = graph.get_top_candidates(limit=3)
    print("Top candidates:")
    for node in top_candidates:
        print(f"- {node.exp_id}: cv_score={node.cv_score}, decision={node.decision}")

    contract = create_contract(**contract_payload)
    available_artifacts = ["metrics.json", "oof_predictions.parquet", "submission_audit.json"]
    artifact_check = check_required_artifacts(contract, available_artifacts)
    acceptance = evaluate_acceptance(
        contract,
        {"cv_score": 0.7653, "fold_std": 0.0035, "schema_valid": True},
    )

    audit = audit_claim(
        claim_id="CLAIM_EXP005_RUNTIME_DEMO",
        claim_text=contract.claim,
        related_exp_ids=["EXP003", "EXP004", "EXP005"],
        contract=asdict(contract),
        supporting_metrics={"EXP005_cv_score": 0.7653, "EXP003_cv_score": 0.7624},
        required_ablations=contract.ablation_plan,
        completed_ablations=["compare_against_EXP003_lightgbm_only"],
        evidence={
            "has_required_experiments": True,
            "has_mechanistic_evidence": False,
            "missing_evidence": artifact_check["missing_artifacts"],
        },
    )

    print("\nValidation contract:")
    print(f"- contract_id={contract.contract_id}")
    print(f"- required_artifacts_passed={artifact_check['passed']}")
    print(f"- missing_artifacts={artifact_check['missing_artifacts']}")
    print(f"- acceptance_passed={acceptance['passed']}")

    print("\nClaim audit:")
    print(f"- audit_result={audit.audit_result}")
    print(f"- drift_type={audit.drift_type}")
    print(f"- missing_evidence={audit.missing_evidence}")
    print(f"- allowed_conclusion={audit.allowed_conclusion}")

    output_path = ROOT / "workspace" / "research_os_demo_search_graph.json"
    graph.export_json(output_path)
    print(f"\nExported graph: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
