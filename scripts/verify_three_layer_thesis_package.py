from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = "20260623"
OUT_JSON = ROOT / "workspace" / f"three_layer_thesis_package_verification_{TODAY}.json"
OUT_MD = ROOT / "reports" / f"THREE_LAYER_THESIS_PACKAGE_VERIFICATION_{TODAY}.md"

REQUIRED_FILES = [
    "workspace/three_layer_steady_improvement_verification_20260623.json",
    "workspace/paper_evidence_bundle_20260623.json",
    "workspace/paper_core_three_layer_claims_20260623.json",
    "workspace/three_layer_thesis_core_matrix_20260623.json",
    "workspace/three_layer_evolution_round4_20260623.json",
    "workspace/retrospective_memory_round4_20260623.json",
    "reports/PAPER_CORE_THREE_LAYER_STEADY_IMPROVEMENT_SECTION_20260623.md",
    "reports/PAPER_THREE_LAYER_EVIDENCE_BUNDLE_20260623.md",
    "reports/THREE_LAYER_THESIS_CORE_MATRIX_20260623.md",
    "reports/PAPER_THREE_LAYER_ALGORITHM_AND_INVARIANT_SECTION_20260623.md",
    "reports/PAPER_READY_THREE_LAYER_CORE_PACKAGE_20260623.md",
    "reports/tables/three_layer_steady_improvement_table_20260623.csv",
    "reports/tables/three_layer_claim_evidence_matrix_20260623.csv",
    "reports/figures/three_layer_evidence_20260623/figure_manifest.json",
    "workspace/three_layer_algorithm_invariant_claims_20260623.json",
    "workspace/paper_ready_three_layer_core_package_20260623.json",
]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"KGAT_[A-Za-z0-9]{16,}"),
    re.compile(r"Bearer\s+(?!\$\{|`\$\{|['\"]?\s*\$)[A-Za-z0-9_\-.]{20,}"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-.]{16,}['\"]"),
]

OVERCLAIM_PHRASES = [
    "超过 MLEvolve",
    "官方 Kaggle 排名提升",
    "GPU/HPC 执行已证明",
    "75 任务已经完成",
    "medal rate 已达到",
]


def read_json(rel_path: str) -> Any:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8-sig"))


def read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8-sig", errors="replace")


def file_check(rel_path: str) -> dict[str, Any]:
    path = ROOT / rel_path
    return {
        "path": rel_path,
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "passed": path.exists() and path.stat().st_size > 0,
    }


def scan_secrets(files: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel_path in files:
        path = ROOT / rel_path
        if not path.exists() or path.suffix.lower() not in {".py", ".json", ".md", ".csv"}:
            continue
        for line_no, line in enumerate(read_text(rel_path).splitlines(), start=1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    hits.append({"path": rel_path, "line": line_no, "redaction": "value_not_recorded"})
    return hits


def scan_overclaims(files: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel_path in files:
        path = ROOT / rel_path
        if not path.exists() or path.suffix.lower() not in {".md", ".json"}:
            continue
        text = read_text(rel_path)
        for phrase in OVERCLAIM_PHRASES:
            if phrase in text:
                hits.append({"path": rel_path, "phrase": phrase})
    return hits


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    file_checks = [file_check(item) for item in REQUIRED_FILES]
    verification = read_json("workspace/three_layer_steady_improvement_verification_20260623.json")
    matrix = read_json("workspace/three_layer_thesis_core_matrix_20260623.json")
    invariant = read_json("workspace/three_layer_algorithm_invariant_claims_20260623.json")
    paper_ready = read_json("workspace/paper_ready_three_layer_core_package_20260623.json")
    round4 = read_json("workspace/three_layer_evolution_round4_20260623.json")
    bundle = read_json("workspace/paper_evidence_bundle_20260623.json")

    checks = [
        {
            "id": "required_files_present",
            "passed": all(item["passed"] for item in file_checks),
            "evidence": file_checks,
        },
        {
            "id": "verification_passed",
            "passed": verification.get("status") == "passed" and sum(1 for item in verification.get("checks", []) if item.get("passed")) == len(verification.get("checks", [])),
            "evidence": {"status": verification.get("status"), "checks": len(verification.get("checks", []))},
        },
        {
            "id": "claim_matrix_supported",
            "passed": matrix.get("claim_matrix_supported") == matrix.get("claim_matrix_total") and matrix.get("claim_matrix_total") >= 6,
            "evidence": {"supported": matrix.get("claim_matrix_supported"), "total": matrix.get("claim_matrix_total")},
        },
        {
            "id": "round4_best_so_far_never_regressed",
            "passed": bool(round4.get("aggregate", {}).get("best_so_far_never_regressed")),
            "evidence": round4.get("aggregate", {}),
        },
        {
            "id": "algorithm_invariant_proved",
            "passed": invariant.get("theorem_status") == "passed" and all(item.get("proof_status") == "passed" for item in invariant.get("task_invariants", [])),
            "evidence": {
                "theorem": invariant.get("theorem"),
                "theorem_status": invariant.get("theorem_status"),
                "task_invariants": len(invariant.get("task_invariants", [])),
            },
        },
        {
            "id": "claim_boundary_present",
            "passed": bool(bundle.get("claim_boundary", {}).get("allowed")) and bool(bundle.get("claim_boundary", {}).get("not_allowed")),
            "evidence": bundle.get("claim_boundary", {}),
        },
        {
            "id": "paper_ready_package_present",
            "passed": paper_ready.get("status") == "ready_for_local_proxy_paper_section"
            and bool(paper_ready.get("paper_sections", {}).get("reviewer_qna"))
            and len(paper_ready.get("figures", [])) >= 3,
            "evidence": {
                "status": paper_ready.get("status"),
                "figures": len(paper_ready.get("figures", [])),
                "paper_sections": paper_ready.get("paper_sections", {}),
            },
        },
    ]
    secret_hits = scan_secrets(REQUIRED_FILES + ["scripts/build_three_layer_thesis_core_matrix.py", "scripts/verify_three_layer_thesis_package.py"])
    overclaim_hits = scan_overclaims(REQUIRED_FILES)
    checks.extend([
        {"id": "secret_scan_clean", "passed": len(secret_hits) == 0, "evidence": {"hits": secret_hits}},
        {"id": "overclaim_scan_clean", "passed": len(overclaim_hits) == 0, "evidence": {"hits": overclaim_hits}},
    ])
    status = "passed" if all(item["passed"] for item in checks) else "failed"
    payload = {
        "schema": "academic_research_os.three_layer_thesis_package_verification.v1",
        "generated_at": generated_at,
        "status": status,
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for item in checks if item["passed"]),
            "secret_hits": len(secret_hits),
            "overclaim_hits": len(overclaim_hits),
        },
        "thesis_safe_status": "ready_for_local_proxy_paper_section" if status == "passed" else "needs_revision",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Three-layer Thesis Package Verification",
        "",
        f"- Generated at: {generated_at}",
        f"- Status: `{status}`",
        f"- Checks: `{payload['summary']['checks_passed']}/{payload['summary']['checks_total']}`",
        f"- Secret hits: `{payload['summary']['secret_hits']}`",
        f"- Overclaim hits: `{payload['summary']['overclaim_hits']}`",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['id']} | {'PASSED' if item['passed'] else 'FAILED'} |")
    lines.extend([
        "",
        "## Conclusion",
        "",
        "当前论文证据包可以支持：三层架构、本地 proxy 三任务 Round1→Round4 best-so-far 稳步提升、算法化 promote/preserve 不回退不变量、失败转记忆、claim audit 阻断过度宣称，并已生成成稿级 Method/Evidence/Reviewer Q&A 核心章节包。仍不能支持：官方 Kaggle 排名、GPU/HPC 本轮执行、MLE-Bench 75 medal rate 或超过 MLEvolve。",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({"status": status, "json": str(OUT_JSON.relative_to(ROOT)).replace('\\', '/'), "md": str(OUT_MD.relative_to(ROOT)).replace('\\', '/')}, ensure_ascii=False, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
