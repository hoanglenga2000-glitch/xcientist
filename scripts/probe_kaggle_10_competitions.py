from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = "20260623"
TASKS_PATH = ROOT / "benchmark" / "kaggle_10_self_evolution" / f"tasks_{TODAY}.json"
OUT_JSON = ROOT / "workspace" / f"kaggle_10_access_probe_{TODAY}.json"
OUT_MD = ROOT / "reports" / f"KAGGLE_10_ACCESS_PROBE_{TODAY}.md"


def load_tasks() -> list[dict[str, Any]]:
    payload = json.loads(TASKS_PATH.read_text(encoding="utf-8-sig"))
    return payload.get("tasks", [])


def main() -> int:
    generated_at = datetime.now().isoformat(timespec="seconds")
    tasks = load_tasks()
    results: list[dict[str, Any]] = []
    status = "passed"
    error = None

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as exc:  # no secrets in output
        status = "not_configured_or_failed"
        error = str(exc)[:240]
        api = None

    if api is not None:
        for task in tasks:
            slug = task["competition_name"]
            try:
                response = api.competitions_list(search=slug)
                competitions = getattr(response, "competitions", response) or []
                compact = []
                for item in competitions[:5]:
                    compact.append({
                        "ref": getattr(item, "ref", None),
                        "title": getattr(item, "title", None),
                        "category": getattr(item, "category", None),
                        "deadline": str(getattr(item, "deadline", None)) if getattr(item, "deadline", None) else None,
                    })
                exact = any((entry.get("ref") or "").lower() == slug.lower() for entry in compact)
                results.append({
                    "task_id": task["task_id"],
                    "competition_name": slug,
                    "probe_status": "found" if exact or compact else "not_found_in_search",
                    "exact_ref_found": exact,
                    "result_count": len(competitions),
                    "top_results": compact,
                })
            except Exception as exc:
                results.append({
                    "task_id": task["task_id"],
                    "competition_name": slug,
                    "probe_status": "probe_failed",
                    "error": str(exc)[:240],
                })

    payload = {
        "ok": status == "passed",
        "schema": "academic_research_os.kaggle_10_access_probe.v1",
        "created_at": generated_at,
        "status": status,
        "secret_policy": "No token, username, cookie, API key or credential value is written to this artifact.",
        "task_count": len(tasks),
        "results": results,
        "error": error,
        "claim_boundary": [
            "This probe only checks competition search visibility through Kaggle API.",
            "It does not download data, run training, or submit predictions.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Kaggle 10 Access Probe",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Status: `{status}`",
        f"- Task count: `{len(tasks)}`",
        "- Action: Kaggle API search only; no download, no training, no submission.",
        "",
        "| task_id | competition | probe status | exact ref | result count |",
        "|---|---|---|---:|---:|",
    ]
    for row in results:
        lines.append(f"| `{row['task_id']}` | `{row['competition_name']}` | {row.get('probe_status')} | {row.get('exact_ref_found', False)} | {row.get('result_count', 0)} |")
    if error:
        lines += ["", "## Probe Error", "", error]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({
        "status": status,
        "task_count": len(tasks),
        "results": len(results),
        "json": str(OUT_JSON.relative_to(ROOT)).replace("\\", "/"),
        "md": str(OUT_MD.relative_to(ROOT)).replace("\\", "/"),
    }, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
