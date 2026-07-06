from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from research_os.mlevolve_controller import build_benchmark_claim_gate, evaluate_rank_gate  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    records: list[dict] = []
    for official_path in sorted((ROOT / "experiments").rglob("kaggle_official_submission.json")):
        run_dir = official_path.parent
        try:
            official = json.loads(official_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            records.append(
                {
                    "task_id": official_path.parent.parent.name,
                    "run_id": official_path.parent.name,
                    "error": f"failed_to_read_official_submission:{type(exc).__name__}",
                    "artifact": str(official_path.relative_to(ROOT)),
                }
            )
            continue
        task_id = str(official.get("task_id") or run_dir.parent.name)
        run_id = str(official.get("run_id") or run_dir.name)
        rank_gate = evaluate_rank_gate(
            task_id=task_id,
            run_id=run_id,
            official_submission=official,
        )
        benchmark_gate = build_benchmark_claim_gate(
            evaluated_tasks=1,
            medal_rate=None,
        )
        rank_gate_path = run_dir / "rank_promotion_gate.json"
        benchmark_gate_path = run_dir / "benchmark_claim_gate.json"
        write_json(rank_gate_path, rank_gate)
        write_json(benchmark_gate_path, benchmark_gate)
        records.append(
            {
                "task_id": task_id,
                "run_id": run_id,
                "rank_gate": str(rank_gate_path.relative_to(ROOT)),
                "benchmark_claim_gate": str(benchmark_gate_path.relative_to(ROOT)),
                "rank_percentile": rank_gate.get("rank_percentile"),
                "top30_reached": rank_gate.get("top30_reached"),
            }
        )

    report = {
        "schema": "academic_research_os.official_rank_gate_backfill.v1",
        "records": records,
        "policy": "Backfill only reads existing Kaggle response artifacts; it does not submit, train, or call external services.",
    }
    out = ROOT / "workspace" / "official_rank_gate_backfill_20260624.json"
    write_json(out, report)
    print(json.dumps({"status": "completed", "records": len(records), "report": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
