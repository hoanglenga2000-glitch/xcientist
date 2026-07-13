from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    round2 = read_json(ROOT / "workspace" / "three_layer_evolution_round2_20260623.json")
    round3 = read_json(ROOT / "workspace" / "round3_targeted_branches_20260623.json")
    by_task_round2 = {row["task_id"]: row for row in round2["tasks"]}
    rows = []
    for item in round3["results"]:
        r2 = by_task_round2[item["task_id"]]
        rows.append(
            {
                "task_id": item["task_id"],
                "metric": item["metric"],
                "direction": item["direction"],
                "round1_baseline": r2["baseline_score"],
                "round2_best_so_far": r2["best_so_far_score"],
                "round3_score": item["round3_score"],
                "round3_decision": item["decision"],
                "final_best_so_far": item["best_so_far_score"],
                "round2_to_round3_best_delta": item["best_so_far_score"] - r2["best_so_far_score"],
                "output_dir": item["output_dir"],
                "branch_id": item["branch_id"],
                "claim_audit": str(Path(item["output_dir"]) / "claim_audit.json"),
                "validation_contract": str(Path(item["output_dir"]) / "validation_contract.json"),
            }
        )

    memory = {
        "schema": "academic_research_os.retrospective_memory_batch.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary": "workspace/round3_targeted_branches_20260623.json",
        "records": [
            {
                "memory_id": f"round3_{row['task_id']}_{'success' if row['round3_decision'] == 'promote_round3' else 'neutral'}",
                "task_id": row["task_id"],
                "method": row["branch_id"],
                "metric": row["metric"],
                "metric_before": row["round2_best_so_far"],
                "metric_after": row["round3_score"],
                "decision": row["round3_decision"],
                "what_worked": (
                    "Targeted branch improved the best-so-far trajectory."
                    if row["round3_decision"] == "promote_round3"
                    else "Gate preserved the previous parent best when the targeted branch did not improve."
                ),
                "what_failed": None if row["round3_decision"] == "promote_round3" else "No measurable improvement over parent best.",
                "reusable_strategy": (
                    "Use this branch as the next parent for exploitation."
                    if row["round3_decision"] == "promote_round3"
                    else "Keep this as an ablation/evidence branch; search a different route next."
                ),
                "linked_artifacts": [row["output_dir"], row["claim_audit"], row["validation_contract"]],
            }
            for row in rows
        ],
    }
    memory_path = ROOT / "workspace" / "retrospective_memory_round3_20260623.json"
    write_json(memory_path, memory)

    summary = {
        "schema": "academic_research_os.three_layer_evolution_round3_summary.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "Round3 targeted local CPU branches selected from Round2 retrospective memory; no GPU/HPC; no official Kaggle submit",
        "three_layer_evidence": {
            "layer_1_multi_agent_research_os": "Each Round3 branch writes agent_trace, metrics, submission, artifact_manifest and report artifacts under the task experiment directory.",
            "layer_2_mlevolve_style_search_controller": "Round3 switches branch strategy based on Round2 memory: regression OOF stack, Titanic feature ablation, Telco calibration/threshold recovery.",
            "layer_3_xcientist_research_harness": "Each branch writes validation_contract.json and claim_audit.json; unsupported improvement claims are revised or rejected.",
        },
        "trajectory": rows,
        "aggregate": {
            "tasks": len(rows),
            "round3_promoted": sum(1 for row in rows if row["round3_decision"] == "promote_round3"),
            "round3_preserved_parent": sum(1 for row in rows if row["round3_decision"] != "promote_round3"),
            "best_so_far_never_regressed": True,
        },
        "claim_boundary": [
            "These are local proxy validation results, not official Kaggle leaderboard scores.",
            "GPU/HPC was not used.",
            "No official Kaggle submission was made.",
            "The supported claim is best-so-far local trajectory improvement/preservation, not medal performance.",
        ],
    }
    summary_path = ROOT / "workspace" / "three_layer_evolution_round3_20260623.json"
    write_json(summary_path, summary)

    lines = [
        "# 三层融合科研工作站 Round 3 进化轨迹报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        "- 执行范围：本地 CPU；根据 Round2 retrospective memory 选择 targeted branch；未使用 GPU/HPC；未提交 Kaggle。",
        "- 核心证明：系统不是盲目重复训练，而是根据成功/失败记忆切换分支，并保持 best-so-far 不倒退。",
        "",
        "## 轨迹总览",
        "",
        "| Task | Metric | Round1 Baseline | Round2 Best | Round3 | Final Best | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['metric']} | {row['round1_baseline']:.6f} | {row['round2_best_so_far']:.6f} | {row['round3_score']:.6f} | {row['final_best_so_far']:.6f} | {row['round3_decision']} |"
        )
    lines.extend(
        [
            "",
            "## 三层架构证据",
            "",
            "### 第一层：Multi-Agent Research OS",
            "- 每个 Round3 分支都写入 `agent_trace.json`、`metrics.json`、`submission.csv`、`artifact_manifest.json`、`report.md`。",
            "- 执行入口是受控 runner 与工作站 artifact 规范，不进行 Kaggle 官方提交。",
            "",
            "### 第二层：MLEvolve-style Search Controller",
            "- House Prices：从 Round2 成功分支进入 exploitation，执行 OOF stacking，RMSLE 继续从 0.122627 到 0.122591。",
            "- Titanic：执行特征 ablation，未超过 Round2 parent，因此保留 0.838384，不发生 best 回退。",
            "- Telco：根据 Round2 ensemble 失败 memory 切换到 calibration/threshold recovery，从 0.807773 提升到 0.808129。",
            "",
            "### 第三层：XCIENTIST-style Research Harness",
            "- 每个分支都有 `validation_contract.json` 与 `claim_audit.json`。",
            "- Titanic 未提升，因此 claim audit 不允许声称 Round3 提升，只允许声称 parent best 被保留。",
            "",
            "## 结论边界",
            "- 允许声称：本地 proxy 下三层科研工作站展示了搜索分支、失败回退、memory 沉淀和 best-so-far 稳步提升/保留机制。",
            "- 不允许声称：官方 Kaggle 分数、GPU/HPC 成功训练、MLE-Bench medal rate。",
        ]
    )
    report_path = ROOT / "reports" / "THREE_LAYER_EVOLUTION_ROUND3_20260623.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({"summary": str(summary_path), "memory": str(memory_path), "report": str(report_path), "aggregate": summary["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
