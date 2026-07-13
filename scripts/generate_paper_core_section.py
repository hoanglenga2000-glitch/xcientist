from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "PAPER_CORE_THREE_LAYER_STEADY_IMPROVEMENT_SECTION_20260623.md"
TABLE_PATH = ROOT / "reports" / "tables" / "three_layer_steady_improvement_table_20260623.csv"
JSON_PATH = ROOT / "workspace" / "paper_core_three_layer_claims_20260623.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def better_delta(direction: str, final_score: float, base_score: float) -> float:
    return base_score - final_score if direction == "minimize" else final_score - base_score


def percent_gain(direction: str, final_score: float, base_score: float) -> float:
    if base_score == 0:
        return 0.0
    return better_delta(direction, final_score, base_score) / abs(base_score) * 100.0


def format_score(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    bundle = read_json(ROOT / "workspace" / "paper_evidence_bundle_20260623.json")
    verification = read_json(ROOT / "workspace" / "three_layer_steady_improvement_verification_20260623.json")
    round4 = read_json(ROOT / "workspace" / "three_layer_evolution_round4_20260623.json")
    rows = round4["trajectory"]

    task_claims = []
    table_lines = ["task_id,metric,direction,round1_baseline,round2_best,round3_parent,round4_score,final_best,round4_decision,final_gain_vs_round1_percent,claim_audit"]
    for row in rows:
        gain_pct = percent_gain(row["direction"], row["final_best_so_far"], row["round1_baseline"])
        task_claims.append({
            "task_id": row["task_id"],
            "metric": row["metric"],
            "direction": row["direction"],
            "round1_baseline": row["round1_baseline"],
            "round2_best_so_far": row["round2_best_so_far"],
            "round3_parent_best": row["round3_best_so_far"],
            "round4_score": row["round4_score"],
            "final_best_so_far": row["final_best_so_far"],
            "round4_decision": row["round4_decision"],
            "final_gain_vs_round1_percent": gain_pct,
            "claim_boundary": "promoted_local_proxy_improvement" if row["round4_decision"] == "promote_round4" else "preserved_parent_negative_memory",
            "artifact_binding": {
                "experiment_dir": row["output_dir"],
                "validation_contract": row["validation_contract"],
                "claim_audit": row["claim_audit"],
                "artifact_manifest": row["artifact_manifest"],
            },
        })
        table_lines.append(
            f"{row['task_id']},{row['metric']},{row['direction']},{row['round1_baseline']:.9f},{row['round2_best_so_far']:.9f},{row['round3_best_so_far']:.9f},{row['round4_score']:.9f},{row['final_best_so_far']:.9f},{row['round4_decision']},{gain_pct:.4f},{row['claim_audit']}"
        )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(table_lines), encoding="utf-8-sig")

    passed_checks = sum(1 for item in verification["checks"] if item["passed"])
    total_checks = len(verification["checks"])
    promoted = round4["aggregate"]["promoted"]
    preserved = round4["aggregate"]["preserved_parent"]
    monotonic = round4["aggregate"]["best_so_far_never_regressed"]

    claim_json = {
        "schema": "academic_research_os.paper_core_claims.v1",
        "generated_at": generated_at,
        "core_claim": "A three-layer AI Research Workstation can convert baseline experiments into a self-evolving and auditable MLE workflow by combining execution orchestration, memory-guided search, and claim-bounded validation.",
        "verified_scope": "local_proxy_three_task_round1_to_round4",
        "verification_status": verification["status"],
        "verification_checks_passed": passed_checks,
        "verification_checks_total": total_checks,
        "round4_promoted": promoted,
        "round4_preserved_parent": preserved,
        "best_so_far_never_regressed": monotonic,
        "layer_claims": {
            "layer_1_multi_agent_research_os": {
                "paper_role": "execution substrate and artifact ledger",
                "evidence": "Each Round3/Round4 branch writes agent_trace, metrics, OOF, submission, manifest and report artifacts.",
            },
            "layer_2_mlevolve_style_search_controller": {
                "paper_role": "self-evolving search and best-so-far protection",
                "evidence": "Round4 consumes Round3 memory, promotes 2 branches, preserves 1 weaker branch, and keeps best-so-far monotonic.",
            },
            "layer_3_xcientist_style_research_harness": {
                "paper_role": "validation contract, risk check and claim audit",
                "evidence": "Every branch has validation_contract.json and claim_audit.json; unsupported improvement claims are revised.",
            },
        },
        "task_claims": task_claims,
        "claim_boundary": bundle["claim_boundary"],
        "table_path": str(TABLE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "section_path": str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(JSON_PATH, claim_json)

    lines = [
        "# 论文核心章节：三层架构与稳步提升机制",
        "",
        f"- 生成时间：{generated_at}",
        "- 证据范围：三个本地 tabular proxy 任务，Round1 → Round4。",
        "- 结论边界：不声称官方 Kaggle 排名、不声称 GPU/HPC 本轮执行、不声称 MLE-Bench 75 任务 medal rate。",
        "",
        "## 1. 核心论点",
        "",
        "本文系统不是单一 Kaggle 训练脚本，而是一个 Self-Evolving and Auditable MLE Research OS。它由三层组成：底层 Multi-Agent Research OS 负责执行与留痕，中层 MLEvolve-style Search Controller 负责自进化搜索与 best-so-far 保护，上层 XCIENTIST-style Research Harness 负责 validation contract、risk check 与 claim audit。",
        "",
        "关键点是：系统不承诺每个候选实验都提升；系统承诺低分候选不会覆盖当前最优，并且失败会转化为 retrospective memory，进入下一轮搜索约束。",
        "",
        "## 2. 三层架构如何在 Round4 中被验证",
        "",
        "| Layer | Role | Round4 evidence |",
        "|---|---|---|",
        "| Multi-Agent Research OS | 执行、留痕、artifact ledger | 每个 Round4 分支均生成 agent_trace、metrics、OOF、submission、artifact_manifest、report |",
        "| MLEvolve-style Search Controller | memory-guided branch search + best-so-far gate | Round4 根据 Round3 memory 发起 3 个分支，2 个 promote，1 个 preserve parent |",
        "| XCIENTIST-style Research Harness | contract、risk、claim audit | 每个分支都有 validation_contract 与 claim_audit，House Prices 未提升时被禁止宣传为提升 |",
        "",
        "## 3. 稳步提升轨迹",
        "",
        "| Task | Metric | Direction | Round1 | Round2 | Round3 parent | Round4 candidate | Final best | Decision | Gain vs Round1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for claim in task_claims:
        lines.append(
            f"| {claim['task_id']} | {claim['metric']} | {claim['direction']} | {format_score(claim['round1_baseline'])} | {format_score(claim['round2_best_so_far'])} | {format_score(claim['round3_parent_best'])} | {format_score(claim['round4_score'])} | {format_score(claim['final_best_so_far'])} | {claim['round4_decision']} | {claim['final_gain_vs_round1_percent']:.2f}% |"
        )
    lines.extend([
        "",
        "## 4. 为什么 House Prices 未提升反而能证明系统优势",
        "",
        "Round4 的 House Prices 分支分数为 0.123128，弱于 Round3 parent best 0.122591。系统没有把这个分支 promote，也没有改写最终最优，而是保留 Round3 parent，并将该分支沉淀为 negative/neutral memory。这个案例直接证明了 best-so-far protection 与 XCIENTIST-style claim boundary：系统不会为了制造“每轮都提升”的叙事而虚假宣传。",
        "",
        "## 5. 自动验证结论",
        "",
        f"- Machine verification: `{verification['status']}`",
        f"- Checks passed: `{passed_checks}/{total_checks}`",
        f"- Round4 promoted branches: `{promoted}`",
        f"- Round4 preserved parent branches: `{preserved}`",
        f"- Best-so-far never regressed: `{str(monotonic).lower()}`",
        "",
        "## 6. 可直接放入论文的表述",
        "",
        "> Across three local proxy tabular tasks and four rounds, the proposed three-layer Research OS promoted two Round4 branches while preserving one weaker branch without overwriting the current best. This demonstrates that the system optimizes the best-so-far trajectory through memory-guided search and claim-bounded validation, rather than relying on unsupported claims that every candidate branch improves.",
        "",
        "中文表述：",
        "",
        "> 在三个本地代理 tabular 任务和四轮实验中，系统在 Round4 提升了两个分支，并对一个弱于父节点的分支执行 preserve parent。该结果证明，系统的稳步提升来自 memory-guided search、best-so-far gate 与 claim audit 的协同，而不是虚假假设每个实验都必然提分。",
        "",
        "## 7. Artifact 索引",
        "",
        f"- Core claims JSON: `{JSON_PATH.relative_to(ROOT)}`",
        f"- Table CSV: `{TABLE_PATH.relative_to(ROOT)}`",
        "- Verification JSON: `workspace/three_layer_steady_improvement_verification_20260623.json`",
        "- Paper evidence bundle: `workspace/paper_evidence_bundle_20260623.json`",
        "- Round4 summary: `workspace/three_layer_evolution_round4_20260623.json`",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({"claims_json": str(JSON_PATH), "section": str(REPORT_PATH), "table": str(TABLE_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
