from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = Path(os.environ.get("RESEARCH_INTEGRITY_GATE_PATH", ROOT / "docs" / "research_integrity_gate.json"))


TASKS = {
    "titanic": ROOT / "experiments" / "titanic" / "20260606_192118",
    "house_prices": ROOT / "experiments" / "house_prices" / "20260606_192030",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_check(path: Path, label: str) -> dict[str, Any]:
    return {"label": label, "path": str(path.relative_to(ROOT)), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}


def task_checks(name: str, exp_dir: Path) -> dict[str, Any]:
    files = [
        file_check(exp_dir / "validation_gate.json", "validation gate"),
        file_check(exp_dir / "workflow_stage_audit.json", "stage audit"),
        file_check(exp_dir / "task_scaffold.json", "task scaffold"),
        file_check(exp_dir / "experiment_log.json", "experiment log"),
        file_check(exp_dir / "data_quality.json", "data quality"),
        file_check(exp_dir / "model_results.json", "model results"),
        file_check(exp_dir / "submission.csv", "submission"),
    ]
    if name == "house_prices":
        files.append(file_check(exp_dir / "post_scaffold_improvement.json", "post scaffold"))
        files.append(file_check(exp_dir / "local_report.md", "markdown report"))
        files.append(file_check(exp_dir / "local_report.docx", "docx report"))
    else:
        files.append(file_check(exp_dir / "titanic_local_report.md", "markdown report"))
        files.append(file_check(exp_dir / "titanic_local_report.docx", "docx report"))

    missing = [item for item in files if not item["exists"] or item["size"] <= 0]
    gate = read_json(exp_dir / "validation_gate.json")
    audit = read_json(exp_dir / "workflow_stage_audit.json")
    return {
        "task": name,
        "experiment_dir": str(exp_dir.relative_to(ROOT)),
        "status": "passed" if not missing and gate.get("status") == "passed" and audit.get("all_stages_passed") else "failed",
        "files": files,
        "missing": missing,
        "gate_status": gate.get("status"),
        "all_stages_passed": audit.get("all_stages_passed"),
    }


def main() -> None:
    source_config = yaml.safe_load((ROOT / "configs" / "research_sources.yaml").read_text(encoding="utf-8"))
    roadmap = yaml.safe_load((ROOT / "configs" / "long_term_roadmap.yaml").read_text(encoding="utf-8"))
    final_audit = (ROOT / "docs" / "科研Agent工作站最终完成审计.md").read_text(encoding="utf-8")

    task_results = [task_checks(name, path) for name, path in TASKS.items()]
    checks = [
        {
            "dimension": "provenance",
            "status": "passed" if source_config.get("sources") and all(task["status"] == "passed" for task in task_results) else "failed",
            "evidence": ["configs/research_sources.yaml", "data source notes", "experiment logs"],
        },
        {
            "dimension": "reproducibility",
            "status": "passed" if "最新验收命令" in final_audit and all(task["gate_status"] == "passed" for task in task_results) else "failed",
            "evidence": ["validation_gate.json", "run_full_acceptance.py", "final audit commands"],
        },
        {
            "dimension": "validity",
            "status": "passed" if all(task["all_stages_passed"] for task in task_results) else "failed",
            "evidence": ["workflow_stage_audit.json", "metric thresholds", "submission checks"],
        },
        {
            "dimension": "human_oversight",
            "status": "passed" if "人工" in final_audit or "human" in final_audit.lower() else "failed",
            "evidence": ["计划文档", "最终审计", "服务器/GPU/Kaggle 边界"],
        },
        {
            "dimension": "limitations",
            "status": "passed" if "当前限制" in final_audit and roadmap.get("items") else "failed",
            "evidence": ["docs/科研Agent工作站最终完成审计.md", "configs/long_term_roadmap.yaml"],
        },
    ]

    failed = [check for check in checks if check["status"] != "passed"]
    result = {
        "status": "passed" if not failed and all(task["status"] == "passed" for task in task_results) else "failed",
        "generated_by": "scripts/verify_research_integrity.py",
        "dimensions": checks,
        "tasks": task_results,
        "roadmap_items": [item["id"] for item in roadmap.get("items", [])],
    }
    GATE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if result["status"] != "passed":
        raise SystemExit("RESEARCH_INTEGRITY_VALIDATION_FAILED: " + json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
