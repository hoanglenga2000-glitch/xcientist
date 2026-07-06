from __future__ import annotations

import difflib
import json
from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .base import Adapter
from ..schemas.agent import CodeArtifact, CodePlan, ExperimentPlan, PatchResult, ReviewResult
from ..schemas.task import TaskProfile


class CodeAgentAdapter(Adapter):
    provider = "code_agent"

    @abstractmethod
    def generate_plan(self, task_profile: TaskProfile, eda_summary: dict) -> CodePlan:
        raise NotImplementedError

    @abstractmethod
    def generate_code(self, plan: CodePlan, workspace_context: dict) -> CodeArtifact:
        raise NotImplementedError

    @abstractmethod
    def review_code(self, code_path: Path, task_context: dict) -> ReviewResult:
        raise NotImplementedError

    @abstractmethod
    def fix_error(self, code_path: Path, error_log: str, run_context: dict) -> PatchResult:
        raise NotImplementedError

    @abstractmethod
    def suggest_next_experiment(self, experiment_history: list[dict]) -> ExperimentPlan:
        raise NotImplementedError

    def review_patch(self, patch: PatchResult, task_context: dict) -> ReviewResult:
        findings: list[str] = []
        if not patch.patch_diff.strip():
            findings.append("Patch diff is empty.")
        if ".." in patch.patch_diff:
            findings.append("Patch contains parent-directory traversal markers.")
        return ReviewResult("passed" if not findings else "warning", findings)

    def apply_patch(self, patch: PatchResult, run_context: dict) -> PatchResult:
        patch.review_status = "recorded"
        patch.metadata.update(
            {
                "apply_status": "recorded",
                "apply_mode": "human_gate_required",
                "note": "Adapter records patch approval; source mutation is performed only by an approved code agent workflow.",
            }
        )
        return patch

    def rollback_patch(self, patch: PatchResult, run_context: dict) -> PatchResult:
        patch.metadata.update({"apply_status": "rolled_back"})
        return patch

    def compare_runs_before_after_patch(self, experiment_history: list[dict]) -> dict:
        if len(experiment_history) < 2:
            return {"status": "insufficient_runs", "before": None, "after": experiment_history[-1] if experiment_history else None}
        return {"status": "compared", "before": experiment_history[-2], "after": experiment_history[-1]}


class LocalTemplateCodeAgentAdapter(CodeAgentAdapter):
    provider = "local_template"

    def generate_plan(self, task_profile: TaskProfile, eda_summary: dict) -> CodePlan:
        steps = [
            "load Kaggle-style train/test/sample files",
            "generate task_profile and EDA summary",
            "train configured tabular baseline models",
            "write metrics, submission and evidence manifest",
            "route submission through validation and human gate",
        ]
        risks = [
            "demo tasks must not be hard-coded into adapter logic",
            "generated code must be saved as artifacts before apply",
            "Kaggle submission remains disabled until credentials and human approval exist",
        ]
        return CodePlan(f"plan_{uuid4().hex[:10]}", task_profile.task_id, self.provider, steps, risks)

    def generate_code(self, plan: CodePlan, workspace_context: dict) -> CodeArtifact:
        task_id = plan.task_id
        code_dir = Path(workspace_context["workspace_dir"]) / "tasks" / task_id / "code" / "generated"
        code_dir.mkdir(parents=True, exist_ok=True)
        script_path = code_dir / "baseline_runner.py"
        script = (
            "from research_agent_workstation.tabular_pipeline import load_yaml, run\n"
            "from pathlib import Path\n\n"
            "def main():\n"
            "    config = load_yaml(Path('config.yaml'))\n"
            "    print(run(config, Path('experiments'), 42))\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        script_path.write_text(script, encoding="utf-8")
        (code_dir / "plan.json").write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")
        return CodeArtifact(f"artifact_{uuid4().hex[:10]}", self.provider, [script_path], "Generated local template baseline runner.")

    def review_code(self, code_path: Path, task_context: dict) -> ReviewResult:
        findings: list[str] = []
        if not code_path.exists():
            findings.append(f"Missing code path: {code_path}")
        if code_path.exists() and code_path.is_file() and "subprocess" in code_path.read_text(encoding="utf-8", errors="ignore"):
            findings.append("Subprocess usage should stay inside PythonRunnerAdapter.")
        return ReviewResult("passed" if not findings else "warning", findings)

    def fix_error(self, code_path: Path, error_log: str, run_context: dict) -> PatchResult:
        patch_id = f"patch_{uuid4().hex[:10]}"
        patch_dir = Path(run_context["workspace_dir"]) / "tasks" / run_context["task_id"] / "code" / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        original = code_path.read_text(encoding="utf-8", errors="ignore") if code_path.exists() else ""
        proposed = original + "\n# Local template note: inspect error_log.txt before applying changes.\n"
        diff = "\n".join(difflib.unified_diff(original.splitlines(), proposed.splitlines(), fromfile=str(code_path), tofile=f"{code_path}.proposed", lineterm=""))
        (patch_dir / f"{patch_id}.diff").write_text(diff, encoding="utf-8")
        return PatchResult(patch_id, self.provider, diff, "pending", [code_path], {"error_log_excerpt": error_log[:1000]})

    def suggest_next_experiment(self, experiment_history: list[dict]) -> ExperimentPlan:
        return ExperimentPlan(
            f"experiment_plan_{uuid4().hex[:10]}",
            "Improve the current tabular baseline only after evidence and validation gates pass.",
            ["compare feature presets", "review failed submission checks", "export context to external code agent if local template plateaus"],
        )
