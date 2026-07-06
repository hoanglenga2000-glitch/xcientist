from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = "20260623"
OUT_MD = ROOT / "reports" / f"PAPER_THREE_LAYER_ALGORITHM_AND_INVARIANT_SECTION_{TODAY}.md"
OUT_JSON = ROOT / "workspace" / f"three_layer_algorithm_invariant_claims_{TODAY}.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def metric_better(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    return candidate < reference - eps if direction == "minimize" else candidate > reference + eps


def metric_not_worse(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    return candidate <= reference + eps if direction == "minimize" else candidate >= reference - eps


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    round4 = read_json(ROOT / "workspace" / "three_layer_evolution_round4_20260623.json")
    verification = read_json(ROOT / "workspace" / "three_layer_steady_improvement_verification_20260623.json")
    memory = read_json(ROOT / "workspace" / "retrospective_memory_round4_20260623.json")
    matrix = read_json(ROOT / "workspace" / "three_layer_thesis_core_matrix_20260623.json")

    task_invariants = []
    for row in round4["trajectory"]:
        parent = float(row["round3_best_so_far"])
        candidate = float(row["round4_score"])
        final_best = float(row["final_best_so_far"])
        direction = row["direction"]
        should_promote = metric_better(direction, candidate, parent)
        final_ok = metric_not_worse(direction, final_best, parent)
        task_invariants.append({
            "task_id": row["task_id"],
            "direction": direction,
            "parent_best": parent,
            "candidate_score": candidate,
            "final_best": final_best,
            "decision": row["round4_decision"],
            "should_promote_by_metric": should_promote,
            "final_not_worse_than_parent": final_ok,
            "proof_status": "passed" if final_ok and ((should_promote and row["round4_decision"] == "promote_round4") or ((not should_promote) and row["round4_decision"] != "promote_round4")) else "failed",
            "evidence": {
                "metrics": f"{row['output_dir']}/metrics.json",
                "search_controller_decision": row["search_controller_decision"],
                "validation_contract": row["validation_contract"],
                "claim_audit": row["claim_audit"],
            },
        })

    theorem_status = "passed" if all(item["proof_status"] == "passed" for item in task_invariants) and round4["aggregate"]["best_so_far_never_regressed"] else "failed"
    payload = {
        "schema": "academic_research_os.three_layer_algorithm_invariant_claims.v1",
        "generated_at": generated_at,
        "paper_section": rel(OUT_MD),
        "theorem": "Best-so-far monotonicity under promote/preserve gate",
        "theorem_status": theorem_status,
        "algorithm_name": "Three-Layer Self-Evolving and Auditable MLE Loop",
        "layers": [
            {
                "layer": "L1 Multi-Agent Research OS",
                "responsibility": "Construct bounded task contexts, execute branch manifests, produce metrics/OOF/submission/report artifacts, and record agent trace.",
                "evidence": "branch_artifacts_complete verifier gate",
            },
            {
                "layer": "L2 MLEvolve-style Search Controller",
                "responsibility": "Retrieve retrospective memory, select exploration/exploitation branch, choose Base/Stepwise/Diff generation mode, and apply promote/preserve decision.",
                "evidence": "round4_search_plan + search_controller_decision artifacts",
            },
            {
                "layer": "L3 XCIENTIST-style Research Harness",
                "responsibility": "Create validation contract, risk checklist, conclusion boundary, and claim audit before any paper claim is allowed.",
                "evidence": "validation_contract + claim_audit artifacts",
            },
        ],
        "task_invariants": task_invariants,
        "round4_aggregate": round4["aggregate"],
        "verification_status": verification["status"],
        "claim_matrix_supported": f"{matrix['claim_matrix_supported']}/{matrix['claim_matrix_total']}",
        "memory_records": len(memory.get("records", [])),
        "safe_claim": "The system guarantees best-so-far protection on the verified local proxy trajectory; it does not guarantee every candidate branch improves.",
        "blocked_overclaims": [
            "Official Kaggle leaderboard improvement",
            "GPU/HPC execution for the local proxy rounds",
            "MLE-Bench 75 medal rate or MLEvolve parity",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 论文算法化表述：三层自进化可审计 MLE Loop 与不回退不变量",
        "",
        f"- 生成时间：{generated_at}",
        "- 用途：作为论文 Method / System Design / Evidence 部分的算法盒与不变量证明草稿。",
        "- 证据边界：当前只证明三个本地 tabular proxy 任务 Round1→Round4 的 best-so-far 不回退。",
        "",
        "## 1. Algorithm 1：Three-Layer Self-Evolving and Auditable MLE Loop",
        "",
        "```text",
        "Input: task spec T, current best state B_t, retrospective memory M_t, claim boundary C_t",
        "Output: updated best state B_{t+1}, new memory M_{t+1}, audited artifacts A_{t+1}",
        "",
        "1  L1 Research OS builds bounded context from T, B_t, M_t, previous artifacts.",
        "2  L2 Search Controller retrieves success/failure memory and selects branch b_t.",
        "3  L2 chooses search stage in {exploration, exploitation} and code mode in {Base, Stepwise, Diff}.",
        "4  L3 Research Harness creates validation contract V_t before execution.",
        "5  L1 executes the branch through workstation-controlled runner/job manifest.",
        "6  L1 writes metrics, OOF, submission schema check, agent trace, artifact manifest and report draft.",
        "7  L3 audits claim drift, leakage risk, CV-public gap risk and conclusion boundary.",
        "8  if candidate score strictly improves over B_t under metric direction and audit allows claim:",
        "9      promote candidate; set B_{t+1} = candidate.",
        "10 else:",
        "11     preserve parent; set B_{t+1} = B_t and write neutral/negative memory.",
        "12 return B_{t+1}, M_{t+1}, A_{t+1}.",
        "```",
        "",
        "## 2. 三层职责的论文表达",
        "",
        "| Layer | Paper role | Operational evidence |",
        "|---|---|---|",
        "| L1 Multi-Agent Research OS | 负责执行、上下文分治、artifact ledger 与可复现记录 | agent_trace、artifact_manifest、metrics、OOF、submission、report |",
        "| L2 MLEvolve-style Search Controller | 负责 memory-guided branch search、Base/Stepwise/Diff 代码生成模式选择、promote/preserve gate | round4_search_plan、search_controller_decision、retrospective_memory |",
        "| L3 XCIENTIST-style Research Harness | 负责 hypothesis、validation contract、risk checklist、claim audit 与 conclusion boundary | validation_contract、claim_audit、claim_boundary |",
        "",
        "## 3. 不变量：Best-so-far Monotonicity",
        "",
        "定义：给定指标方向 d，如果 d=minimize，则更小为更优；如果 d=maximize，则更大为更优。设父节点最优为 B_t，候选分支为 x_t，系统最终状态为 B_{t+1}。promote/preserve gate 保证：",
        "",
        "```text",
        "if better_d(x_t, B_t): B_{t+1} = x_t",
        "else:                  B_{t+1} = B_t",
        "therefore not_worse_d(B_{t+1}, B_t) always holds.",
        "```",
        "",
        "这就是本文中“稳步提升”的精确定义：不是每个候选分支都提升，而是当前 best-so-far 轨迹不会被低质量候选覆盖。",
        "",
        "## 4. Round4 不变量验证表",
        "",
        "| Task | Direction | Parent best | Candidate | Final best | Decision | Candidate better | Final not worse | Proof |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for item in task_invariants:
        lines.append(
            f"| {item['task_id']} | {item['direction']} | {item['parent_best']:.6f} | {item['candidate_score']:.6f} | {item['final_best']:.6f} | {item['decision']} | {item['should_promote_by_metric']} | {item['final_not_worse_than_parent']} | {item['proof_status']} |"
        )
    lines.extend([
        "",
        "## 5. House Prices 反例的作用",
        "",
        "House Prices Round4 candidate 弱于 Round3 parent。系统没有 promote，而是 preserve parent，并把该结果写入 retrospective memory。这一负例是论文中最重要的可信度证据之一：它证明系统不会为了叙事而把未提升实验包装成提升，而是用 L2 gate 与 L3 claim audit 共同约束结论。",
        "",
        "## 6. 可直接写入论文的英文表述",
        "",
        "> We define steady improvement as monotonic best-so-far protection rather than per-branch improvement. At each iteration, the MLEvolve-style Search Controller proposes a candidate branch, while the XCIENTIST-style Research Harness audits whether the claimed improvement is supported. A candidate is promoted only when it is strictly better than the parent under the task metric and passes the claim boundary; otherwise, the parent is preserved and the candidate is converted into retrospective memory. This invariant prevents weak branches from overwriting the current best and turns failures into future search constraints.",
        "",
        "## 7. 可直接写入论文的中文表述",
        "",
        "> 本文将“稳步提升”定义为 best-so-far 轨迹不回退，而不是每个候选实验都必然提升。每一轮中，MLEvolve-style Search Controller 依据历史成功/失败记忆提出候选分支，XCIENTIST-style Research Harness 审计该分支是否满足 validation contract 与 claim boundary。只有当候选分支在任务指标方向上严格优于父节点且审计通过时，系统才执行 promote；否则 preserve parent，并将该分支沉淀为 retrospective memory。该不变量阻止弱分支覆盖当前最优，同时把失败转化为下一轮搜索约束。",
        "",
        "## 8. 证据索引",
        "",
        f"- Algorithm invariant JSON: `{rel(OUT_JSON)}`",
        "- Round4 summary: `workspace/three_layer_evolution_round4_20260623.json`",
        "- Round4 memory: `workspace/retrospective_memory_round4_20260623.json`",
        "- Verification: `workspace/three_layer_steady_improvement_verification_20260623.json`",
        "- Thesis matrix: `workspace/three_layer_thesis_core_matrix_20260623.json`",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({"status": theorem_status, "json": rel(OUT_JSON), "markdown": rel(OUT_MD)}, ensure_ascii=False, indent=2))
    if theorem_status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
