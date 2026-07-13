from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..schemas.task import TaskProfile
from ..core.json_utils import write_json


class TaskService:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def load_config(self, config_path: Path) -> dict[str, Any]:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def import_from_config(self, config_path: Path) -> TaskProfile:
        config = self.load_config(config_path)
        task = config["task"]
        data = config["data"]
        return TaskProfile(
            task_id=task["name"],
            name=task.get("competition", task["name"]),
            task_type=task["type"],
            target=task["target"],
            metric=task["metric"],
            task_dir=self.workspace_root / data["task_dir"],
            train_path=self.workspace_root / data["train"],
            test_path=self.workspace_root / data["test"],
            sample_submission_path=self.workspace_root / data["sample_submission"],
            overview_path=self.workspace_root / data["overview"] if data.get("overview") else None,
            metadata={"config_path": str(config_path), "id_column": task.get("id_column"), "prediction_column": task.get("prediction_column")},
        )

    def generate_scaffold(self, task_profile: TaskProfile, config_path: Path, output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
        config = self.load_config(config_path)
        task = config["task"]
        model = config.get("model", {})
        validation = config.get("validation", {})
        scaffold = {
            "research_question": f"Can a reproducible {task_profile.task_type} baseline improve {task_profile.metric} for {task_profile.name}?",
            "task_type": task_profile.task_type,
            "dataset_type": "tabular",
            "target_column": task_profile.target,
            "id_column": task.get("id_column"),
            "metric": task_profile.metric,
            "validation_strategy": validation or {"strategy": "holdout_or_cv"},
            "baseline_strategy": "local_template_sklearn_baseline",
            "preprocessing_strategy": "numeric/categorical automatic preprocessing with missing-value handling",
            "candidate_models": model.get("candidates", ["linear_or_logistic_baseline", "tree_ensemble_baseline"]),
            "risk_points": [
                "data leakage",
                "submission column mismatch",
                "missing predictions",
                "metric mismatch",
                "unapproved plan or patch",
            ],
            "required_gates": ["PLAN_APPROVAL", "SUBMISSION_APPROVAL", "FINAL_CLAIM_APPROVAL", "REPORT_EXPORT_APPROVAL"],
            "expected_artifacts": [
                "task_profile.json",
                "scaffold.json",
                "data_quality.json",
                "model_results.json",
                "submission.csv",
                "validation_gate.json",
                "agent_trace.jsonl",
                "artifact_manifest.json",
                "evidence_index.json",
                "report.md",
                "reflection.json",
            ],
            "first_run_plan": [
                "inspect input files",
                "train local baseline",
                "write metrics and submission",
                "run validation gate",
                "bind evidence and draft report",
            ],
            "fallback_plan": [
                "if training fails, enter WAITING_FIX and export context for Codex/Claude Code",
                "if validation fails, block submission approval and request reviewer evidence",
            ],
        }
        scaffold_json = write_json(output_dir / "scaffold.json", scaffold)
        scaffold_md = output_dir / "scaffold.md"
        scaffold_md.write_text(
            "\n".join(
                [
                    "# Research Task Scaffold",
                    "",
                    f"- Research question: {scaffold['research_question']}",
                    f"- Task type: {scaffold['task_type']}",
                    f"- Dataset type: {scaffold['dataset_type']}",
                    f"- Target: {scaffold['target_column']}",
                    f"- Metric: {scaffold['metric']}",
                    "",
                    "## Required Gates",
                    *[f"- {gate}" for gate in scaffold["required_gates"]],
                    "",
                    "## First Run Plan",
                    *[f"- {step}" for step in scaffold["first_run_plan"]],
                ]
            ),
            encoding="utf-8",
        )
        return scaffold_json, scaffold_md, scaffold
