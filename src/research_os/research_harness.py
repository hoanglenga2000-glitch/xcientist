"""Phase-0 three-layer Research OS harness skeleton (P0-HN-01).

Layers (local CPU-only demo / tests — no HPC):
  1. Multi-Agent Research OS surface (fixture search graph + optional audit role)
  2. Search Controller decision artifact
  3. Research Harness claim audit + benchmark claim gate

Gated by ``EVOMIND_FF_RESEARCH_OS_HARNESS`` (default off). Insufficient evidence
always blocks promotion claims; never emits external medal / "beat AI2/MLEvolve"
language.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .claim_audit import audit_claim
from .feature_flags import ff_benchmark_claim_gate, ff_research_os_harness
from .mlevolve_controller import build_benchmark_claim_gate, build_search_controller_decision
from .search_graph import ExperimentNode, SearchGraph


def run_research_os_harness(
    work_dir: str | Path,
    *,
    task_id: str = "harness_demo",
    run_id: str = "harness-run-001",
    evaluated_tasks: int = 0,
    claim_text: str = "",
) -> dict[str, Any]:
    """Run the local three-layer skeleton and write artifacts under ``work_dir``.

    When the harness flag is off, returns a skipped payload and writes nothing.
    When on, always produces ``search_graph.json``, ``search_controller_decision.json``,
    ``claim_audit.json``, and ``benchmark_claim_gate.json``. Thin evidence ⇒ claim
    blocked / reject.
    """
    enabled = ff_research_os_harness()
    root = Path(work_dir)
    if not enabled:
        return {
            "schema": "academic_research_os.research_harness.v1",
            "enabled": False,
            "skipped": True,
            "feature_flag": "EVOMIND_FF_RESEARCH_OS_HARNESS",
            "claim_blocked": True,
            "artifacts": {},
        }

    root.mkdir(parents=True, exist_ok=True)
    graph = SearchGraph(
        task_id=task_id,
        root_exp_id="EXP000",
        metric_name="cv_score",
        metric_direction="maximize",
    )
    graph.nodes["EXP000"] = ExperimentNode(
        exp_id="EXP000",
        parent_id=None,
        branch_type="Base",
        task_name=task_id,
        hypothesis="harness skeleton baseline",
        implementation_summary="fixture node for Phase-0 demo",
        code_path="EXP000/solution.py",
        cv_score=None,
        metric_name="cv_score",
        metric_direction="maximize",
        run_success=False,
    )
    graph_path = root / "search_graph.json"
    graph_payload = {
        "schema": "academic_research_os.search_graph.v1",
        "task_id": task_id,
        "root_exp_id": graph.root_exp_id,
        "nodes": {
            eid: {
                "exp_id": n.exp_id,
                "parent_id": n.parent_id,
                "branch_type": n.branch_type,
                "cv_score": n.cv_score,
                "run_success": n.run_success,
                "hypothesis": n.hypothesis,
            }
            for eid, n in graph.nodes.items()
        },
        "edges": list(getattr(graph, "edges", []) or []),
        "layer": "search_controller_input",
    }
    graph_path.write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    controller = build_search_controller_decision(
        task_id=task_id,
        run_id=run_id,
        selected_branch="b0",
        exploration_stage="harness_skeleton",
        metric="cv_score",
        metric_direction="maximize",
        has_parent=False,
    )
    controller_path = root / "search_controller_decision.json"
    controller_path.write_text(json.dumps(controller, ensure_ascii=False, indent=2), encoding="utf-8")

    # Insufficient evidence by design for the skeleton demo.
    text = (claim_text or f"{task_id} harness skeleton produced a fixture search graph").strip()
    audit = audit_claim(
        claim_id=f"{task_id}:{run_id}:harness",
        claim_text=text,
        related_exp_ids=["EXP000"],
        contract={"hypothesis": "harness skeleton baseline", "conclusion_boundary": "demo only"},
        supporting_metrics={},
        required_ablations=["independent_holdout"],
        completed_ablations=[],
        evidence={
            "has_required_experiments": False,
            "has_mechanistic_evidence": False,
            "missing_evidence": ["no_verified_cv_score", "no_independent_holdout"],
        },
    )
    from dataclasses import asdict

    audit_payload = {
        "schema": "academic_research_os.claim_audit.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": task_id,
        "layer": "research_harness",
        "promotion_claim_allowed": False,
        **asdict(audit),
    }
    audit_path = root / "claim_audit.json"
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = build_benchmark_claim_gate(evaluated_tasks=evaluated_tasks)
    gate_path = root / "benchmark_claim_gate.json"
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    assert audit.audit_result in {"reject", "revise"}, "harness must block thin claims"
    summary = {
        "schema": "academic_research_os.research_harness.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "enabled": True,
        "skipped": False,
        "feature_flag": "EVOMIND_FF_RESEARCH_OS_HARNESS",
        "layers": ["multi_agent_research_os", "search_controller", "research_harness"],
        "task_id": task_id,
        "run_id": run_id,
        "claim_blocked": True,  # skeleton never allows promotion claims
        "audit_result": audit.audit_result,
        "benchmark_claim_status": gate.get("claim_status"),
        "claim_label": gate.get("claim_label", "no"),
        "artifacts": {
            "search_graph": str(graph_path),
            "search_controller_decision": str(controller_path),
            "claim_audit": str(audit_path),
            "benchmark_claim_gate": str(gate_path),
        },
        "boundary": (
            "Local Phase-0 skeleton only; insufficient evidence blocks claims; "
            "not comparable to MLEvolve or AI2 medal rates."
        ),
    }
    summary_path = root / "research_harness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["artifacts"]["summary"] = str(summary_path)
    return summary
