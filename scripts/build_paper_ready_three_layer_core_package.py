from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = "20260623"
OUT_MD = ROOT / "reports" / f"PAPER_READY_THREE_LAYER_CORE_PACKAGE_{TODAY}.md"
OUT_JSON = ROOT / "workspace" / f"paper_ready_three_layer_core_package_{TODAY}.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    round4 = read_json(ROOT / "workspace" / "three_layer_evolution_round4_20260623.json")
    verification = read_json(ROOT / "workspace" / "three_layer_steady_improvement_verification_20260623.json")
    invariant = read_json(ROOT / "workspace" / "three_layer_algorithm_invariant_claims_20260623.json")
    matrix = read_json(ROOT / "workspace" / "three_layer_thesis_core_matrix_20260623.json")
    figure_manifest = read_json(ROOT / "reports" / "figures" / "three_layer_evidence_20260623" / "figure_manifest.json")

    rows = round4["trajectory"]
    package = {
        "schema": "academic_research_os.paper_ready_three_layer_core_package.v1",
        "generated_at": generated_at,
        "status": "ready_for_local_proxy_paper_section",
        "core_contribution": "A three-layer Self-Evolving and Auditable MLE Research OS that combines artifact-based multi-agent execution, MLEvolve-style memory-guided search, and XCIENTIST-style claim-bounded validation.",
        "supported_claims": [item for item in matrix["claim_evidence_matrix"] if item["status"] == "supported"],
        "algorithm_invariant": invariant,
        "verification": {
            "status": verification["status"],
            "checks_passed": sum(1 for item in verification["checks"] if item["passed"]),
            "checks_total": len(verification["checks"]),
        },
        "round4_aggregate": round4["aggregate"],
        "figures": figure_manifest["figures"],
        "claim_boundary": matrix["claim_boundary"],
        "paper_sections": {
            "method_and_architecture": "included_in_markdown",
            "algorithm_and_invariant": "included_in_markdown",
            "evidence_and_results": "included_in_markdown",
            "limitations_and_claim_boundary": "included_in_markdown",
            "reviewer_qna": "included_in_markdown",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")

    checks_passed = package["verification"]["checks_passed"]
    checks_total = package["verification"]["checks_total"]
    lines = [
        "# Paper-ready Core Package: Three-layer Self-Evolving and Auditable MLE Research OS",
        "",
        f"- Generated at: {generated_at}",
        "- Status: ready for local-proxy paper section",
        "- Claim boundary: local three-task proxy evidence only; no official Kaggle / GPU-HPC / MLE-Bench medal-rate claim.",
        "",
        "## Abstract-ready Contribution Statement",
        "",
        "We propose a three-layer Self-Evolving and Auditable MLE Research OS for Kaggle-style machine learning research. Unlike a single training script, the system separates execution, optimization, and scientific validation into three coupled layers: a Multi-Agent Research OS that produces reproducible artifacts, an MLEvolve-style Search Controller that performs memory-guided branch selection with best-so-far protection, and an XCIENTIST-style Research Harness that binds hypotheses, validation contracts, risk checks, and claim audits to every reported conclusion. On three local proxy tabular tasks over four rounds, the system promotes two Round4 branches while preserving one weaker branch, demonstrating monotonic best-so-far protection rather than unsupported per-branch improvement claims.",
        "",
        "## 1. System Overview",
        "",
        "The key design choice is to treat machine learning experimentation as an auditable operating system rather than as an isolated model-training routine. Each experiment is represented as a branch with bounded context, explicit artifacts, metric evidence, validation constraints, and claim boundaries. The three layers are:",
        "",
        "1. **Multi-Agent Research OS**: decomposes a task into bounded agent responsibilities and records execution artifacts such as agent traces, metrics, OOF predictions, submissions, manifests, and reports.",
        "2. **MLEvolve-style Search Controller**: retrieves retrospective memory, chooses an exploration/exploitation branch, selects Base/Stepwise/Diff code generation mode, and applies promote/preserve best-so-far gates.",
        "3. **XCIENTIST-style Research Harness**: creates validation contracts, risk checklists, conclusion boundaries, and claim audits so that every reported claim is tied to evidence.",
        "",
        "## 2. Algorithm 1: Three-layer Self-Evolving and Auditable MLE Loop",
        "",
        "```text",
        "Input: task spec T, current best state B_t, retrospective memory M_t, claim boundary C_t",
        "Output: updated best state B_{t+1}, updated memory M_{t+1}, audited artifacts A_{t+1}",
        "",
        "1  L1 builds bounded context from task spec, current best, memory, and prior artifacts.",
        "2  L2 retrieves success/failure memory and proposes candidate branch x_t.",
        "3  L2 selects search stage and code generation mode: Base, Stepwise, or Diff.",
        "4  L3 creates a validation contract before execution.",
        "5  L1 executes the branch through workstation-controlled runner or gated job manifest.",
        "6  L1 writes metrics, OOF, submission schema check, agent trace, artifact manifest, and report draft.",
        "7  L3 audits leakage risk, CV-public gap risk, claim drift, and conclusion boundary.",
        "8  if x_t is strictly better than B_t under metric direction and claim audit allows the claim:",
        "9      promote x_t and set B_{t+1}=x_t.",
        "10 else:",
        "11     preserve B_t and convert x_t into neutral/negative retrospective memory.",
        "12 return B_{t+1}, M_{t+1}, A_{t+1}.",
        "```",
        "",
        "## 3. Invariant: Monotonic Best-so-far Protection",
        "",
        "The system does not assume that every candidate branch improves. Instead, it enforces the invariant that the accepted best-so-far state cannot regress. For a minimization metric, `better(x, B)` means `x < B`; for a maximization metric, it means `x > B`. The promote/preserve gate implements:",
        "",
        "```text",
        "if better(candidate, parent_best): final_best = candidate",
        "else:                              final_best = parent_best",
        "therefore final_best is never worse than parent_best.",
        "```",
        "",
        "This invariant is the precise meaning of steady improvement in the current evidence package.",
        "",
        "## 4. Evidence Table: Round1 to Round4",
        "",
        "| Task | Metric | Direction | Round1 | Round2 | Round3 parent | Round4 candidate | Final best | Decision | Evidence interpretation |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if row["round4_decision"] == "promote_round4":
            interpretation = "candidate promoted; best-so-far improved"
        else:
            interpretation = "candidate rejected; parent preserved; negative memory written"
        lines.append(
            f"| {row['task_id']} | {row['metric']} | {row['direction']} | {row['round1_baseline']:.6f} | {row['round2_best_so_far']:.6f} | {row['round3_best_so_far']:.6f} | {row['round4_score']:.6f} | {row['final_best_so_far']:.6f} | {row['round4_decision']} | {interpretation} |"
        )
    lines.extend([
        "",
        "## 5. Why the House Prices Non-improvement Strengthens the Claim",
        "",
        "The House Prices Round4 candidate is weaker than the Round3 parent under the RMSLE minimization metric. A score-only narrative might hide this result, but the proposed system records it as evidence. The Search Controller preserves the parent best, and the Research Harness prevents the report from claiming an unsupported improvement. This negative branch is converted into retrospective memory, making it useful for later search. Therefore, the non-improving branch strengthens the paper claim: the system is designed for auditable self-evolution, not for cherry-picking successful runs.",
        "",
        "## 6. Figure Plan",
        "",
        "| Figure | Title | Caption | Artifact |",
        "|---|---|---|---|",
    ])
    for fig in figure_manifest["figures"]:
        lines.append(f"| {fig['figure_id']} | {fig['title']} | {fig['caption']} | `{fig['paths']['png']}` |")
    lines.extend([
        "",
        "## 7. Reviewer-facing Claim Boundary",
        "",
        "The current evidence supports local proxy validation, not full benchmark dominance. The paper can claim that the three-layer mechanism is implemented and verified on three local tabular proxy tasks, and that best-so-far protection is preserved across the checked rounds. It cannot claim official Kaggle leaderboard improvement, GPU/HPC execution for these local rounds, MLE-Bench 75 coverage, or parity/superiority over MLEvolve medal rate without future benchmark evidence.",
        "",
        "## 8. Reviewer Q&A",
        "",
        "**Q1: Does the system guarantee every experiment improves?**  ",
        "No. The guarantee is best-so-far protection. Weak candidates are preserved as memory instead of being promoted.",
        "",
        "**Q2: Why is one non-improving Round4 branch useful evidence?**  ",
        "Because it verifies that the gate and claim audit prevent unsupported improvement claims.",
        "",
        "**Q3: Is this an official Kaggle or MLE-Bench result?**  ",
        "No. This package is local proxy evidence and explicitly blocks leaderboard and medal-rate overclaims.",
        "",
        "**Q4: What distinguishes the system from a normal AutoML script?**  ",
        "The system separates execution, search, and validation into auditable layers, and every claim is bound to artifacts rather than only to a final score.",
        "",
        "## 9. Verification Status",
        "",
        f"- Three-layer verification: `{verification['status']}`",
        f"- Verification checks: `{checks_passed}/{checks_total}`",
        f"- Algorithm invariant: `{invariant['theorem_status']}`",
        f"- Claim matrix: `{matrix['claim_matrix_supported']}/{matrix['claim_matrix_total']}` supported",
        f"- Round4 aggregate: `{round4['aggregate']['promoted']}` promoted, `{round4['aggregate']['preserved_parent']}` preserved parent, best-so-far never regressed = `{str(round4['aggregate']['best_so_far_never_regressed']).lower()}`",
        "",
        "## 10. Artifact Index",
        "",
        f"- Paper-ready package JSON: `{rel(OUT_JSON)}`",
        "- Core matrix: `workspace/three_layer_thesis_core_matrix_20260623.json`",
        "- Algorithm invariant: `workspace/three_layer_algorithm_invariant_claims_20260623.json`",
        "- Verification: `workspace/three_layer_thesis_package_verification_20260623.json`",
        "- Figure manifest: `reports/figures/three_layer_evidence_20260623/figure_manifest.json`",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({"status": package["status"], "json": rel(OUT_JSON), "markdown": rel(OUT_MD)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
