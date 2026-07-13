from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = "20260623"
OUT_JSON = ROOT / "workspace" / f"three_layer_thesis_core_matrix_{TODAY}.json"
OUT_MD = ROOT / "reports" / f"THREE_LAYER_THESIS_CORE_MATRIX_{TODAY}.md"
OUT_CSV = ROOT / "reports" / "tables" / f"three_layer_claim_evidence_matrix_{TODAY}.csv"

PATHS = {
    "verification": ROOT / "workspace" / "three_layer_steady_improvement_verification_20260623.json",
    "bundle": ROOT / "workspace" / "paper_evidence_bundle_20260623.json",
    "round4": ROOT / "workspace" / "three_layer_evolution_round4_20260623.json",
    "round4_memory": ROOT / "workspace" / "retrospective_memory_round4_20260623.json",
    "round4_plan": ROOT / "workspace" / "round4_search_plan_20260623.json",
    "protocol": ROOT / "workspace" / "steady_improvement_protocol_20260623.json",
    "core_claims": ROOT / "workspace" / "paper_core_three_layer_claims_20260623.json",
    "core_section": ROOT / "reports" / "PAPER_CORE_THREE_LAYER_STEADY_IMPROVEMENT_SECTION_20260623.md",
    "evidence_bundle_report": ROOT / "reports" / "PAPER_THREE_LAYER_EVIDENCE_BUNDLE_20260623.md",
    "verification_report": ROOT / "reports" / "THREE_LAYER_STEADY_IMPROVEMENT_VERIFICATION_20260623.md",
    "figure_manifest": ROOT / "reports" / "figures" / "three_layer_evidence_20260623" / "figure_manifest.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path),
    }


def metric_not_worse(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    return candidate <= reference + eps if direction == "minimize" else candidate >= reference - eps


def metric_better(direction: str, candidate: float, reference: float, eps: float = 1e-12) -> bool:
    return candidate < reference - eps if direction == "minimize" else candidate > reference + eps


def gain_percent(direction: str, final_score: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    delta = baseline - final_score if direction == "minimize" else final_score - baseline
    return delta / abs(baseline) * 100.0


def format_score(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def build_task_rows(round4: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in round4.get("trajectory", []):
        final_not_worse = metric_not_worse(row["direction"], float(row["final_best_so_far"]), float(row["round3_best_so_far"]))
        candidate_better = metric_better(row["direction"], float(row["round4_score"]), float(row["round3_best_so_far"]))
        rows.append({
            **row,
            "final_not_worse_than_parent": final_not_worse,
            "candidate_better_than_parent": candidate_better,
            "gain_vs_round1_percent": gain_percent(row["direction"], float(row["final_best_so_far"]), float(row["round1_baseline"])),
            "evidence_files": [
                file_record(ROOT / row["output_dir"] / "agent_trace.json"),
                file_record(ROOT / row["output_dir"] / "artifact_manifest.json"),
                file_record(ROOT / row["output_dir"] / "metrics.json"),
                file_record(ROOT / row["output_dir"] / "oof_predictions.csv"),
                file_record(ROOT / row["output_dir"] / "submission.csv"),
                file_record(ROOT / row["search_controller_decision"]),
                file_record(ROOT / row["validation_contract"]),
                file_record(ROOT / row["claim_audit"]),
            ],
        })
    return rows


def build_claim_matrix(verification: dict[str, Any], bundle: dict[str, Any], round4: dict[str, Any], task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verification_checks = {item["id"]: item for item in verification.get("checks", [])}
    artifacts_ok = bool(verification_checks.get("branch_artifacts_complete", {}).get("passed"))
    monotonic_ok = bool(verification_checks.get("best_so_far_monotonic", {}).get("passed"))
    memory_ok = bool(verification_checks.get("retrospective_memory_complete", {}).get("passed"))
    claim_boundary_ok = bool(verification_checks.get("claim_boundary_enforced", {}).get("passed"))
    round4_execution_ok = bool(verification_checks.get("round4_execution_evidence", {}).get("passed"))
    return [
        {
            "claim_id": "C1_three_layer_architecture_exists",
            "paper_claim": "系统由 Multi-Agent Research OS、MLEvolve-style Search Controller、XCIENTIST-style Research Harness 三层构成。",
            "status": "supported" if verification_checks.get("three_layer_summary_present", {}).get("passed") and artifacts_ok else "needs_revision",
            "evidence": [rel(PATHS["round4"]), rel(PATHS["verification"]), rel(PATHS["bundle"])],
            "mechanism": "三层分别负责执行留痕、搜索选择、科研审计。",
            "claim_boundary": "支持本地 proxy 证据；不扩展为官方 Kaggle 或 75 任务结论。",
        },
        {
            "claim_id": "C2_best_so_far_steady_improvement",
            "paper_claim": "系统的稳步提升体现在 best-so-far 轨迹不回退，而不是每个候选分支都提升。",
            "status": "supported" if monotonic_ok and all(row["final_not_worse_than_parent"] for row in task_rows) else "needs_revision",
            "evidence": [rel(PATHS["verification"]), rel(PATHS["core_claims"]), rel(PATHS["core_section"])],
            "mechanism": "promote/preserve gate 根据指标方向阻止弱分支覆盖当前最优。",
            "claim_boundary": "可写 best-so-far never regressed；不可写 every branch improves。",
        },
        {
            "claim_id": "C3_failure_to_memory_conversion",
            "paper_claim": "失败或未提升分支被转化为 retrospective memory，用于约束下一轮搜索。",
            "status": "supported" if memory_ok else "needs_revision",
            "evidence": [rel(PATHS["round4_memory"]), rel(PATHS["round4_plan"])],
            "mechanism": "House Prices Round4 未提升，系统 preserve parent 并写入 neutral/negative memory。",
            "claim_boundary": "支持 failure-to-memory；不宣称失败分支本身提分。",
        },
        {
            "claim_id": "C4_agent_artifact_workflow",
            "paper_claim": "每个实验分支均产生可审计 artifacts：agent trace、metrics、OOF、submission、manifest、contract、audit、report。",
            "status": "supported" if artifacts_ok else "needs_revision",
            "evidence": [row["output_dir"] for row in task_rows],
            "mechanism": "底层 Research OS 把训练过程转成 artifact ledger，支持复核与复现。",
            "claim_boundary": "支持 artifact-level audit；不等同于官方线上提交。",
        },
        {
            "claim_id": "C5_claim_drift_control",
            "paper_claim": "XCIENTIST-style harness 通过 validation contract 与 claim audit 阻断 claim drift。",
            "status": "supported" if claim_boundary_ok else "needs_revision",
            "evidence": [rel(PATHS["verification"]), rel(PATHS["bundle"]), *[row["claim_audit"] for row in task_rows]],
            "mechanism": "claim audit 区分 confirmed、weak evidence、unsupported/blocked overclaim。",
            "claim_boundary": "禁止从本地 proxy 推导 official leaderboard、GPU/HPC 或 MLE-Bench medal rate。",
        },
        {
            "claim_id": "C6_workstation_controlled_round4",
            "paper_claim": "Round4 由工作站控制的本地 proxy runner 执行，并输出三层证据。",
            "status": "supported" if round4_execution_ok else "needs_revision",
            "evidence": [rel(PATHS["round4"]), "scripts/run_round4_workstation_branches.py"],
            "mechanism": "工作站 runner 生成每个分支的决策、artifact 与审计，不绕过 evidence ledger。",
            "claim_boundary": "本轮不能写成 GPU/HPC 执行。",
        },
    ]


def write_csv(claim_matrix: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "status", "paper_claim", "mechanism", "claim_boundary", "evidence"])
        writer.writeheader()
        for row in claim_matrix:
            writer.writerow({**row, "evidence": " | ".join(row["evidence"])})


def write_markdown(payload: dict[str, Any]) -> None:
    task_rows = payload["task_rows"]
    claim_matrix = payload["claim_evidence_matrix"]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 三层架构论文核心证据矩阵",
        "",
        f"- 生成时间：{payload['generated_at']}",
        "- 目标：突出论文核心三层架构，并用现有 Round1→Round4 证据证明 best-so-far 稳步提升机制。",
        "- 证据边界：三个本地 tabular proxy 任务；不声称官方 Kaggle 排名、GPU/HPC 本轮执行、MLE-Bench 75 medal rate。",
        "",
        "## 1. 一句话论文核心",
        "",
        "> 本系统的核心贡献不是直接替代某个训练脚本，而是把机器学习实验组织成一个三层闭环：底层 Multi-Agent Research OS 负责可复现执行与 artifact ledger，中层 MLEvolve-style Search Controller 负责记忆驱动的多分支自进化搜索与 best-so-far gate，上层 XCIENTIST-style Research Harness 负责 validation contract、risk check 与 claim audit，从而保证系统在多轮实验中稳步优化当前最优轨迹，并阻断无证据的提分叙事。",
        "",
        "## 2. 三层机制到证据的对应",
        "",
        "| 层级 | 机制 | 当前证据 | 论文可写结论 |",
        "|---|---|---|---|",
        "| Layer 1 Multi-Agent Research OS | Agent trace + artifact manifest + metrics/OOF/submission/report | Round4 每个分支均有完整 artifact 文件 | 系统具备可审计执行底座 |",
        "| Layer 2 MLEvolve-style Search Controller | Retrospective memory + branch selection + promote/preserve gate | Round4 2 个 promote，1 个 preserve parent | 稳步提升指 best-so-far 不回退 |",
        "| Layer 3 XCIENTIST-style Research Harness | validation contract + risk checklist + claim audit | 每个分支都有 contract/audit，House Prices 未提升被限制结论 | 系统能阻断 claim drift |",
        "",
        "## 3. Claim-Evidence Matrix",
        "",
        "| Claim ID | Status | Paper claim | Evidence | Boundary |",
        "|---|---|---|---|---|",
    ]
    for row in claim_matrix:
        evidence = "<br>".join(f"`{item}`" for item in row["evidence"][:6])
        lines.append(f"| {row['claim_id']} | {row['status']} | {row['paper_claim']} | {evidence} | {row['claim_boundary']} |")

    lines.extend([
        "",
        "## 4. Best-so-far 稳步提升轨迹",
        "",
        "| Task | Metric | Direction | Round1 | Round2 | Round3 parent | Round4 candidate | Final best | Decision | Gain vs Round1 | Final not worse |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ])
    for row in task_rows:
        lines.append(
            f"| {row['task_id']} | {row['metric']} | {row['direction']} | {format_score(row['round1_baseline'])} | {format_score(row['round2_best_so_far'])} | {format_score(row['round3_best_so_far'])} | {format_score(row['round4_score'])} | {format_score(row['final_best_so_far'])} | {row['round4_decision']} | {row['gain_vs_round1_percent']:.2f}% | {row['final_not_worse_than_parent']} |"
        )

    lines.extend([
        "",
        "## 5. 关键反例证明：House Prices 未提升为什么反而重要",
        "",
        "House Prices Round4 candidate = 0.123128，高于/弱于 Round3 parent = 0.122591（RMSLE 越低越好）。系统没有把该分支 promote，而是 preserve parent，并将其写入 retrospective memory。这个反例证明系统核心不是包装每个实验都提升，而是通过 best-so-far gate 与 claim audit 保证最终轨迹不被弱实验污染。",
        "",
        "## 6. 自动验证状态",
        "",
        f"- Verification status: `{payload['verification_status']}`",
        f"- Verification checks: `{payload['verification_checks_passed']}/{payload['verification_checks_total']}`",
        f"- Claim matrix supported: `{payload['claim_matrix_supported']}/{payload['claim_matrix_total']}`",
        f"- Best-so-far never regressed: `{str(payload['best_so_far_never_regressed']).lower()}`",
        "",
        "## 7. 论文中必须保留的边界",
        "",
    ])
    for item in payload["claim_boundary"]["not_allowed"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 8. Artifact 索引",
        "",
        f"- JSON: `{rel(OUT_JSON)}`",
        f"- CSV: `{rel(OUT_CSV)}`",
        f"- Core section: `{rel(PATHS['core_section'])}`",
        f"- Verification: `{rel(PATHS['verification'])}`",
        f"- Evidence bundle: `{rel(PATHS['bundle'])}`",
        f"- Figure manifest: `{rel(PATHS['figure_manifest'])}`",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    verification = read_json(PATHS["verification"])
    bundle = read_json(PATHS["bundle"])
    round4 = read_json(PATHS["round4"])
    task_rows = build_task_rows(round4)
    claim_matrix = build_claim_matrix(verification, bundle, round4, task_rows)
    supported = sum(1 for row in claim_matrix if row["status"] == "supported")
    payload = {
        "schema": "academic_research_os.three_layer_thesis_core_matrix.v1",
        "generated_at": generated_at,
        "objective": "突出论文核心三层架构，并证明稳步提升机制来自 best-so-far protection + retrospective memory + claim audit。",
        "verification_status": verification.get("status"),
        "verification_checks_passed": sum(1 for item in verification.get("checks", []) if item.get("passed")),
        "verification_checks_total": len(verification.get("checks", [])),
        "best_so_far_never_regressed": bool(round4.get("aggregate", {}).get("best_so_far_never_regressed")),
        "claim_matrix_supported": supported,
        "claim_matrix_total": len(claim_matrix),
        "claim_evidence_matrix": claim_matrix,
        "task_rows": task_rows,
        "claim_boundary": bundle.get("claim_boundary", {}),
        "source_files": {name: file_record(path) for name, path in PATHS.items()},
        "outputs": {
            "json": rel(OUT_JSON),
            "markdown": rel(OUT_MD),
            "csv": rel(OUT_CSV),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(claim_matrix)
    write_markdown(payload)
    print(json.dumps({
        "status": "passed" if supported == len(claim_matrix) and payload["best_so_far_never_regressed"] else "needs_review",
        "json": rel(OUT_JSON),
        "markdown": rel(OUT_MD),
        "csv": rel(OUT_CSV),
        "supported_claims": f"{supported}/{len(claim_matrix)}",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
