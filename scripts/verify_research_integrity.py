from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.verify_research_sources import validate_source_metadata
except ModuleNotFoundError:  # Direct execution from scripts/.
    from verify_research_sources import validate_source_metadata


ROOT = Path(__file__).resolve().parents[1]

TASK_REQUIRED_FILES = {
    "titanic": (
        "validation_gate.json",
        "workflow_stage_audit.json",
        "task_scaffold.json",
        "experiment_log.json",
        "data_quality.json",
        "model_results.json",
        "submission.csv",
        "titanic_local_report.md",
        "titanic_local_report.docx",
    ),
    "house_prices": (
        "validation_gate.json",
        "workflow_stage_audit.json",
        "task_scaffold.json",
        "experiment_log.json",
        "data_quality.json",
        "model_results.json",
        "submission.csv",
        "post_scaffold_improvement.json",
        "local_report.md",
        "local_report.docx",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configured_runtime_path(value: str | None, default: Path) -> Path:
    configured = Path(value) if value else default
    return (configured if configured.is_absolute() else ROOT / configured).resolve()


def configured_experiment_root(explicit: str | None = None) -> Path:
    evidence_root = os.environ.get("RESEARCH_EVIDENCE_ROOT")
    return configured_runtime_path(
        explicit or os.environ.get("RESEARCH_EXPERIMENT_ROOT") or (str(Path(evidence_root) / "experiments") if evidence_root else None),
        ROOT / "experiments",
    )


def display_evidence_path(path: Path, experiment_root: Path) -> str:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    evidence_root = experiment_root.resolve()
    if resolved.is_relative_to(evidence_root):
        return f"runtime/experiments/{resolved.relative_to(evidence_root).as_posix()}"
    return "runtime/experiments/external-artifact"


def file_check(path: Path, label: str, experiment_root: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": display_evidence_path(path, experiment_root),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
    }


def latest_complete_experiment(task_root: Path, required_files: tuple[str, ...]) -> Path:
    try:
        runs = sorted(path for path in task_root.iterdir() if path.is_dir())
    except OSError as error:
        raise SystemExit(
            "RESEARCH_INTEGRITY_VALIDATION_FAILED: "
            f"experiment root unavailable for task {task_root.name!r}"
        ) from error
    for run in reversed(runs):
        if all((run / name).is_file() and (run / name).stat().st_size > 0 for name in required_files):
            return run
    raise SystemExit(
        "RESEARCH_INTEGRITY_VALIDATION_FAILED: "
        f"no complete experiment for task {task_root.name!r}; required_files={list(required_files)}"
    )


def task_checks(name: str, exp_dir: Path, experiment_root: Path) -> dict[str, Any]:
    files = [
        file_check(exp_dir / "validation_gate.json", "validation gate", experiment_root),
        file_check(exp_dir / "workflow_stage_audit.json", "stage audit", experiment_root),
        file_check(exp_dir / "task_scaffold.json", "task scaffold", experiment_root),
        file_check(exp_dir / "experiment_log.json", "experiment log", experiment_root),
        file_check(exp_dir / "data_quality.json", "data quality", experiment_root),
        file_check(exp_dir / "model_results.json", "model results", experiment_root),
        file_check(exp_dir / "submission.csv", "submission", experiment_root),
    ]
    if name == "house_prices":
        files.append(file_check(exp_dir / "post_scaffold_improvement.json", "post scaffold", experiment_root))
        files.append(file_check(exp_dir / "local_report.md", "markdown report", experiment_root))
        files.append(file_check(exp_dir / "local_report.docx", "docx report", experiment_root))
    else:
        files.append(file_check(exp_dir / "titanic_local_report.md", "markdown report", experiment_root))
        files.append(file_check(exp_dir / "titanic_local_report.docx", "docx report", experiment_root))

    missing = [item for item in files if not item["exists"] or item["size"] <= 0]
    gate = read_json(exp_dir / "validation_gate.json")
    audit = read_json(exp_dir / "workflow_stage_audit.json")
    return {
        "task": name,
        "experiment_dir": display_evidence_path(exp_dir, experiment_root),
        "status": "passed"
        if not missing and gate.get("status") == "passed" and audit.get("all_stages_passed")
        else "failed",
        "files": files,
        "missing": missing,
        "gate_status": gate.get("status"),
        "all_stages_passed": audit.get("all_stages_passed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify research-integrity evidence from a local or external runtime root.")
    parser.add_argument(
        "--experiment-root",
        default=None,
        help="Directory containing per-task experiment folders; defaults to RESEARCH_EXPERIMENT_ROOT or ./experiments.",
    )
    args = parser.parse_args()
    experiment_root = configured_experiment_root(args.experiment_root)
    gate_path = configured_runtime_path(
        os.environ.get("RESEARCH_INTEGRITY_GATE_PATH"),
        ROOT / "docs" / "research_integrity_gate.json",
    )
    source_config = yaml.safe_load((ROOT / "configs" / "research_sources.yaml").read_text(encoding="utf-8"))
    roadmap = yaml.safe_load((ROOT / "configs" / "long_term_roadmap.yaml").read_text(encoding="utf-8"))
    validate_source_metadata(source_config.get("sources", []))

    task_results = [
        task_checks(
            name,
            latest_complete_experiment(experiment_root / name, required_files),
            experiment_root,
        )
        for name, required_files in TASK_REQUIRED_FILES.items()
    ]
    roadmap_items = roadmap.get("items", [])
    status_policy = roadmap.get("status_policy", {})
    controlled_items = [item for item in roadmap_items if item.get("status") == "controlled_pending"]
    controlled_policy = str(status_policy.get("controlled_pending", "")).lower()
    human_gate_documented = bool(controlled_items) and any(
        marker in controlled_policy for marker in ("approval", "human", "user")
    )
    limitations_documented = bool(roadmap_items) and all(item.get("status") in status_policy for item in roadmap_items)
    checks = [
        {
            "dimension": "provenance",
            "status": "passed"
            if source_config.get("sources") and all(task["status"] == "passed" for task in task_results)
            else "failed",
            "evidence": ["configs/research_sources.yaml", "data source notes", "experiment logs"],
        },
        {
            "dimension": "reproducibility",
            "status": "passed"
            if all(task["status"] == "passed" and task["gate_status"] == "passed" for task in task_results)
            else "failed",
            "evidence": ["experiment_log.json", "validation_gate.json", "run_full_acceptance.py"],
        },
        {
            "dimension": "validity",
            "status": "passed" if all(task["all_stages_passed"] for task in task_results) else "failed",
            "evidence": ["workflow_stage_audit.json", "metric thresholds", "submission checks"],
        },
        {
            "dimension": "human_oversight",
            "status": "passed" if human_gate_documented else "failed",
            "evidence": ["configs/long_term_roadmap.yaml", "controlled_pending approval policy"],
        },
        {
            "dimension": "limitations",
            "status": "passed" if limitations_documented else "failed",
            "evidence": ["configs/long_term_roadmap.yaml", "status_policy"],
        },
    ]

    failed = [check for check in checks if check["status"] != "passed"]
    result = {
        "status": "passed" if not failed and all(task["status"] == "passed" for task in task_results) else "failed",
        "generated_by": "scripts/verify_research_integrity.py",
        "dimensions": checks,
        "tasks": task_results,
        "roadmap_items": [item["id"] for item in roadmap_items],
        "experiment_evidence_source": (
            "repository_workspace" if experiment_root.is_relative_to(ROOT.resolve()) else "external_runtime_root"
        ),
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if result["status"] != "passed":
        raise SystemExit("RESEARCH_INTEGRITY_VALIDATION_FAILED: " + json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
