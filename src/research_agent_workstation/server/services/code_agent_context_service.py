from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from ..adapters.storage_adapter import StorageAdapter
from ..schemas.task import TaskProfile


class CodeAgentContextService:
    def __init__(self, storage: StorageAdapter, workspace_root: Path) -> None:
        self.storage = storage
        self.workspace_root = workspace_root

    def export_context(self, task: TaskProfile, latest_output_dir: Path | None, experiment_plan: str) -> Path:
        context_dir = self.workspace_root / "workspace" / "tasks" / task.task_id / "agent_context"
        self.storage.ensure_dir(context_dir)
        self.storage.write_json(context_dir / "task_profile.json", task)

        eda_path = latest_output_dir / "data_quality.json" if latest_output_dir else None
        metrics_path = latest_output_dir / "model_results.json" if latest_output_dir else None
        error_path = latest_output_dir / "error_log.txt" if latest_output_dir else None
        current_code_dir = context_dir / "current_code"
        self.storage.ensure_dir(current_code_dir)

        if eda_path and eda_path.exists():
            shutil.copy2(eda_path, context_dir / "eda_summary.json")
        else:
            self.storage.write_json(context_dir / "eda_summary.json", {})
        if metrics_path and metrics_path.exists():
            shutil.copy2(metrics_path, context_dir / "metrics.json")
        else:
            self.storage.write_json(context_dir / "metrics.json", {})
        (context_dir / "error_log.txt").write_text(error_path.read_text(encoding="utf-8") if error_path and error_path.exists() else "", encoding="utf-8")
        (context_dir / "experiment_plan.md").write_text(experiment_plan, encoding="utf-8")

        instructions = self._instructions(task)
        (context_dir / "instructions_for_codex.md").write_text(instructions.replace("{agent}", "Codex"), encoding="utf-8")
        (context_dir / "instructions_for_claude_code.md").write_text(instructions.replace("{agent}", "Claude Code"), encoding="utf-8")
        return context_dir

    def import_patch(self, task_id: str, patch_text: str, source_agent: str = "external") -> Path:
        patch_dir = self.workspace_root / "workspace" / "tasks" / task_id / "code" / "patches"
        self.storage.ensure_dir(patch_dir)
        patch_id = f"patch_{len(list(patch_dir.glob('*.diff'))) + 1:04d}"
        patch_path = patch_dir / f"{patch_id}.diff"
        patch_path.write_text(patch_text, encoding="utf-8")
        metadata = {
            "patch_id": patch_id,
            "source_agent": source_agent,
            "review_status": "pending",
            "applied_at": None,
            "rollback_path": None,
        }
        self.storage.write_json(patch_dir / f"{patch_id}.json", metadata)
        return patch_path

    def review_patch(self, task_id: str, patch_id: str, reviewer: str = "ReviewerAgent") -> dict:
        patch_path, metadata_path, metadata = self._patch_record(task_id, patch_id)
        patch_text = patch_path.read_text(encoding="utf-8", errors="ignore")
        findings: list[str] = []
        if not patch_text.strip():
            findings.append("Patch is empty.")
        if ".." in patch_text or "\\.." in patch_text:
            findings.append("Patch contains parent-directory traversal markers.")
        if any(marker in patch_text for marker in [" /", " C:\\", " D:\\"]):
            findings.append("Patch appears to contain absolute paths.")
        if not patch_text.startswith(("diff --git", "--- ", "*** Begin Patch")):
            findings.append("Patch format is not a standard git/unified/apply_patch diff.")
        metadata.update(
            {
                "review_status": "passed" if not findings else "warning",
                "reviewer": reviewer,
                "review_findings": findings,
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.storage.write_json(metadata_path, metadata)
        return metadata

    def apply_patch(self, task_id: str, patch_id: str, reviewer: str = "Research Admin") -> dict:
        patch_path, metadata_path, metadata = self._patch_record(task_id, patch_id)
        if metadata.get("review_status") not in {"passed", "warning"}:
            metadata = self.review_patch(task_id, patch_id)
        apply_dir = self.workspace_root / "workspace" / "tasks" / task_id / "code" / "applied_patches"
        self.storage.ensure_dir(apply_dir)
        applied_copy = apply_dir / patch_path.name
        shutil.copy2(patch_path, applied_copy)
        rollback_path = apply_dir / f"{patch_id}_rollback.json"
        rollback_payload = {
            "patch_id": patch_id,
            "task_id": task_id,
            "source_patch": str(patch_path),
            "applied_copy": str(applied_copy),
            "rollback_mode": "metadata_only",
            "note": "The MVP records approved external patches without mutating source files automatically. Apply source edits only after a human gate.",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.storage.write_json(rollback_path, rollback_payload)
        metadata.update(
            {
                "apply_status": "recorded",
                "applied_at": datetime.now().isoformat(timespec="seconds"),
                "applied_by": reviewer,
                "applied_copy": str(applied_copy),
                "rollback_path": str(rollback_path),
            }
        )
        self.storage.write_json(metadata_path, metadata)
        return metadata

    def rollback_patch(self, task_id: str, patch_id: str, reviewer: str = "Research Admin") -> dict:
        _patch_path, metadata_path, metadata = self._patch_record(task_id, patch_id)
        metadata.update(
            {
                "apply_status": "rolled_back",
                "rolled_back_at": datetime.now().isoformat(timespec="seconds"),
                "rolled_back_by": reviewer,
            }
        )
        self.storage.write_json(metadata_path, metadata)
        return metadata

    def compare_runs_before_after_patch(self, task_id: str, before_run: Path | None = None, after_run: Path | None = None) -> dict:
        run_root = self.workspace_root / "experiments" / task_id
        runs = sorted(path for path in run_root.glob("*") if path.is_dir())
        if not before_run and len(runs) >= 2:
            before_run = runs[-2]
        if not after_run and runs:
            after_run = runs[-1]
        comparison = {
            "task_id": task_id,
            "before_run": self._display_run_path(task_id, before_run),
            "after_run": self._display_run_path(task_id, after_run),
            "before_metrics": self._read_best_metrics(before_run) if before_run else {},
            "after_metrics": self._read_best_metrics(after_run) if after_run else {},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        compare_dir = self.workspace_root / "workspace" / "tasks" / task_id / "code" / "comparisons"
        self.storage.ensure_dir(compare_dir)
        self.storage.write_json(compare_dir / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", comparison)
        return comparison

    def _display_run_path(self, task_id: str, run_dir: Path | None) -> str | None:
        if not run_dir:
            return None
        return (Path("runtime") / "experiments" / Path(task_id).name / run_dir.name).as_posix()

    def _patch_record(self, task_id: str, patch_id: str) -> tuple[Path, Path, dict]:
        patch_dir = self.workspace_root / "workspace" / "tasks" / task_id / "code" / "patches"
        patch_path = patch_dir / f"{patch_id}.diff"
        metadata_path = patch_dir / f"{patch_id}.json"
        if not patch_path.exists():
            raise FileNotFoundError(patch_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"patch_id": patch_id}
        return patch_path, metadata_path, metadata

    def _read_best_metrics(self, run_dir: Path | None) -> dict:
        if not run_dir:
            return {}
        path = run_dir / "model_results.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        best_model = payload.get("best_model")
        return payload.get("model_results", {}).get(best_model, {})

    def _instructions(self, task: TaskProfile) -> str:
        try:
            data_path = task.task_dir.relative_to(self.workspace_root)
        except ValueError:
            data_path = task.task_dir
        return f"""# Instructions for {{agent}}

## Current Task

- Task goal: {task.name}
- Data path: {data_path}
- Target: {task.target}
- Metric: {task.metric}

## Rules

- Do not overwrite original data.
- Output a patch or generated artifact, not direct destructive edits.
- Save experiment records for every run.
- Pass submission check before requesting submission approval.
- Bind every conclusion to evidence artifacts.
- Keep rollback path for any applied patch.
"""
