from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_os.mlevolve_adapter import build_workstation_alignment, extract_policy, save_json  # noqa: E402


OUT_JSON = ROOT / "workspace" / "mlevolve_alignment_matrix_20260625.json"
OUT_MD = ROOT / "reports" / "MLEVOLVE_ALIGNMENT_MATRIX_20260625.md"


def main() -> None:
    repo = ROOT / "external-projects" / "MLEvolve"
    policy = extract_policy(repo)
    payload = build_workstation_alignment(policy)
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["codex_role"] = "reference_reverse_engineering_only; no direct training or Kaggle submission"
    save_json(OUT_JSON, payload)

    lines = [
        "# MLEvolve 对齐矩阵",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Source repo: `{repo.relative_to(ROOT).as_posix()}`",
        f"- Reference time budget: `{policy.time_budget_hours:.1f}h`",
        f"- Parallel search branches: `{policy.parallel_search_num}`",
        f"- Top candidates size: `{policy.top_candidates_size}`",
        f"- Branch stagnation threshold: `{policy.branch_stagnation_threshold}`",
        f"- Fusion window: `{policy.fusion_min_time_hours}h~{policy.fusion_max_time_hours}h`",
        f"- Global memory enabled: `{policy.use_global_memory}`",
        f"- Claim boundary: {payload['claim_boundary']}",
        "",
        "## Mapping",
        "",
        "| MLEvolve concept | Workstation component | Status | Next required work |",
        "|---|---|---|---|",
    ]
    for concept, item in payload["workstation_mapping"].items():
        next_work = "; ".join(item.get("next_required_work", []))
        lines.append(f"| {concept} | `{item['workstation_component']}` | `{item['status']}` | {next_work} |")
    lines.extend(
        [
            "",
            "## Execution Rule",
            "",
            "- MLEvolve is used as a read-only engineering reference.",
            "- Workstation AgentOrchestrator remains the only allowed execution subject.",
            "- Top30 and medal claims require official response artifacts and benchmark claim gates.",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": OUT_JSON.relative_to(ROOT).as_posix(), "md": OUT_MD.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
