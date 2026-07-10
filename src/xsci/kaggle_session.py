"""Long-lived session state for the `kaggle` research terminal.

This is the terminal agent's working memory: workspace, selected competition,
last goal, backend readiness, recent run digest, and memory summary. It is safe
to expose to the 8088 workstation because it stores no secret values.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config, PROJECT_DIRNAME, load_config

MODE_CHAT = "chat"
MODE_PLANNING = "planning"
MODE_EXECUTING = "executing"


def _has_llm(cfg: Config) -> bool:
    return bool(
        cfg.get("secrets.anthropic_api_key")
        or cfg.get("secrets.deepseek_api_key")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )


def _has_kaggle(cfg: Config) -> bool:
    return bool(
        cfg.get("secrets.kaggle_api_token")
        or (cfg.get("secrets.kaggle_username") and cfg.get("secrets.kaggle_key"))
        or os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )


def _has_gpu(cfg: Config) -> bool:
    return bool(cfg.get("gpu_ssh.host") and cfg.get("gpu_ssh.user"))


def _strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _external_resource_manifest(root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    for base in [Path.cwd(), root, Path(__file__).resolve().parents[2]]:
        candidates.append(base / "configs" / "external_resources.yaml")
        candidates.extend(parent / "configs" / "external_resources.yaml" for parent in base.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _gpu_manifest(root: Path) -> dict[str, str]:
    manifest = _external_resource_manifest(root)
    if manifest is None:
        return {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    in_hpc = False
    result: dict[str, str] = {"manifest_path": str(manifest)}
    for raw in lines:
        stripped = raw.strip()
        if raw.startswith("  hpc_gpu_ssh:"):
            in_hpc = True
            continue
        if in_hpc and raw.startswith("  ") and not raw.startswith("    ") and stripped.endswith(":"):
            break
        if not in_hpc or not raw.startswith("    ") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key in {"status", "current_blocker", "current_runtime_note"}:
            result[key] = _strip_yaml_scalar(value)
    return result


def _same_task(candidate: str, selected_task: str) -> bool:
    if not selected_task:
        return True

    def norm(value: str) -> str:
        return (value or "").lower().replace("-", "").replace("_", "").replace(" ", "")

    c = norm(candidate)
    s = norm(selected_task)
    return c == s or c.startswith(s) or s in c


@dataclass
class SessionState:
    workspace_root: str = ""
    selected_task: Optional[str] = None
    last_goal: str = ""
    llm_ready: bool = False
    llm_provider: str = "unset"
    kaggle_ready: bool = False
    compute_backend: str = "local"
    gpu_ready: bool = False
    gpu_status: str = ""
    gpu_blocker: str = ""
    gpu_blocked: bool = False
    recent_run_id: str = ""
    recent_events_path: str = ""
    recent_best_cv: Optional[float] = None
    current_mode: str = MODE_CHAT
    memory_summary: str = ""
    task_brief: str = ""
    n_tasks: int = 0
    updated_at: str = ""
    # ── EvoMind terminal-agent extensions ──────────────────────────────
    tool_readiness: str = ""                # "idle" | "inspecting" | "training" | "reporting"
    current_compute_override: str = ""      # "local" | "gpu" | "" (use default)
    last_action: str = ""                   # last action type
    last_artifact: str = ""                 # last artifact path
    last_event_path: str = ""               # last events.jsonl path

    @classmethod
    def from_root(cls, root: Path, *, cfg: Optional[Config] = None) -> "SessionState":
        root = Path(root)
        cfg = cfg or load_config(root)
        gpu_configured = _has_gpu(cfg)
        gpu_manifest = _gpu_manifest(root)
        gpu_status = gpu_manifest.get("status", "")
        gpu_blocker = gpu_manifest.get("current_blocker", "")
        gpu_blocked = bool(gpu_configured and (gpu_blocker or gpu_status.endswith("_closed") or "blocked" in gpu_status.lower()))
        state = cls(
            workspace_root=str(root),
            llm_ready=_has_llm(cfg),
            llm_provider=str(cfg.get("llm.brand") or cfg.get("llm.provider", "unset") or "unset"),
            kaggle_ready=_has_kaggle(cfg),
            compute_backend=str(cfg.get("compute.backend", "local") or "local"),
            gpu_ready=gpu_configured,
            gpu_status=gpu_status,
            gpu_blocker=gpu_blocker,
            gpu_blocked=gpu_blocked,
        )
        state._restore_choices(root)
        state.refresh_tasks(root)
        state.refresh_recent_run(root)
        state.refresh_memory(root)
        state.refresh_task_brief(root)
        state.updated_at = datetime.now().isoformat(timespec="seconds")
        return state

    def _restore_choices(self, root: Path) -> None:
        snap = Path(root) / PROJECT_DIRNAME / "session.json"
        if not snap.exists():
            return
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        saved_task = data.get("selected_task")
        if isinstance(saved_task, str) and saved_task:
            self.selected_task = saved_task
        saved_goal = data.get("last_goal")
        if isinstance(saved_goal, str):
            self.last_goal = saved_goal

    def refresh_tasks(self, root: Path) -> None:
        from .tasks import list_tasks

        tasks = list_tasks(root)
        self.n_tasks = len(tasks)
        slugs = {slug for slug, _ in tasks}
        if self.selected_task and self.selected_task not in slugs:
            self.selected_task = None
        if self.selected_task is None and tasks:
            self.selected_task = tasks[0][0]

    def refresh_task_brief(self, root: Path) -> None:
        self.task_brief = ""
        if not self.selected_task:
            return
        try:
            from .tasks import resolve_task

            path = resolve_task(self.selected_task, project_root=root)
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        schema = str(data.get("data_schema", "") or "")
        needs_schema = (not schema) or schema.startswith("TODO")
        parts = [
            f"name={data.get('task_name', self.selected_task)}",
            f"modality={data.get('modality', '?')}",
            f"task_type={data.get('task_type', '?')}",
            f"metric={data.get('metric', '?')}({data.get('metric_direction', '?')})",
        ]
        target = data.get("target_column")
        if target:
            parts.append(f"target={target}")
        if not needs_schema:
            snippet = schema.strip().replace("\n", " ")
            parts.append("schema=" + (snippet[:160] + "..." if len(snippet) > 160 else snippet))
        else:
            parts.append("schema=UNFILLED (scaffold TODO; ask user or inspect data)")
        self.task_brief = " | ".join(parts)

    def refresh_recent_run(self, root: Path) -> None:
        self.recent_run_id = ""
        self.recent_events_path = ""
        self.recent_best_cv = None
        base = Path(root) / "experiments" / "evolution"
        if not base.is_dir():
            return
        run_dirs = [d for d in base.iterdir() if d.is_dir()]
        if not run_dirs:
            return
        candidates: list[Path] = []
        for run_dir in run_dirs:
            if not self.selected_task:
                candidates.append(run_dir)
                continue
            summary = run_dir / "summary.json"
            task_name = run_dir.name
            if summary.exists():
                try:
                    payload = json.loads(summary.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        task_name = str(payload.get("task") or payload.get("task_id") or task_name)
                except (json.JSONDecodeError, OSError):
                    task_name = run_dir.name
            if _same_task(task_name, self.selected_task) or _same_task(run_dir.name, self.selected_task):
                candidates.append(run_dir)
        if not candidates:
            return
        newest = max(candidates, key=lambda d: d.name)
        self.recent_run_id = newest.name
        events = newest / "events.jsonl"
        if events.exists():
            self.recent_events_path = str(events)
        summary = newest / "summary.json"
        if summary.exists():
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                cv = data.get("best_cv_score")
                self.recent_best_cv = float(cv) if isinstance(cv, (int, float)) else None
            except (json.JSONDecodeError, OSError, ValueError):
                self.recent_best_cv = None

    def refresh_memory(self, root: Path) -> None:
        mem = Path(root) / "experiments" / "evolution" / "retrospective_memory.json"
        if not mem.exists():
            self.memory_summary = "empty (no prior lessons yet)"
            return
        try:
            records = json.loads(mem.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.memory_summary = "unreadable"
            return
        if isinstance(records, dict):
            records = records.get("records", []) or list(records.values())
        if not isinstance(records, list) or not records:
            self.memory_summary = "empty (no prior lessons yet)"
            return
        successes = sum(1 for r in records if isinstance(r, dict) and r.get("success"))
        failures = len(records) - successes
        self.memory_summary = f"{len(records)} lessons ({successes} reuse, {failures} avoid)"

    def missing_setup(self, *, compute_override: Optional[str] = None) -> list[str]:
        effective_compute = compute_override or self.compute_backend
        gaps: list[str] = []
        if not self.llm_ready:
            gaps.append(
                "LLM API: hypothesis generation, code generation, failure attribution, and multi-round evolution need it. "
                "Run `setup` or `/setup` to configure Anthropic or DeepSeek."
            )
        if not self.kaggle_ready:
            gaps.append(
                "Kaggle API: needed for official downloads, competition metadata, and submit candidates. "
                "Run `setup` to import kaggle.json or configure a token. You may skip it for local data."
            )
        if effective_compute == "gpu" and not self.gpu_ready:
            gaps.append(
                "GPU/SSH: compute=gpu is selected, but SSH host/user is missing. Configure it in `setup`, "
                "or switch back to local for small controlled tests."
            )
        if effective_compute == "gpu" and self.gpu_ready and self.gpu_blocked:
            detail = f" Manifest status: {self.gpu_status}." if self.gpu_status else ""
            blocker = f" Blocker: {self.gpu_blocker}" if self.gpu_blocker else ""
            gaps.append(
                "GPU/SSH: metadata is configured, but the external-resource manifest blocks real training until "
                f"a fresh GPU smoke passes.{detail}{blocker}"
            )
        if not self.selected_task:
            gaps.append(
                "Task: no competition is selected. Use `task add https://www.kaggle.com/competitions/<slug>` first."
            )
        return gaps

    def blocking_setup(self, *, compute_override: Optional[str] = None) -> list[str]:
        """Return only the gates that must block a run.

        Kaggle API is useful for official downloads/submissions, but it should not
        block a local run when the task data/config already exists. GPU blockers
        apply only when the effective compute backend is gpu.
        """
        effective_compute = compute_override or self.compute_backend
        gaps: list[str] = []
        if not self.llm_ready:
            gaps.append(
                "LLM API: code generation and experiment reasoning need it. "
                "Run `setup` or `/setup` to configure Anthropic, DeepSeek, or a compatible gateway."
            )
        if not self.selected_task:
            gaps.append(
                "Task: no competition is selected. Use `task add https://www.kaggle.com/competitions/<slug>` first."
            )
        if effective_compute == "gpu" and not self.gpu_ready:
            gaps.append(
                "GPU/SSH: this run requested compute=gpu, but SSH host/user is missing. "
                "Configure GPU/HPC or request local compute for a small controlled test."
            )
        if effective_compute == "gpu" and self.gpu_ready and self.gpu_blocked:
            detail = f" Manifest status: {self.gpu_status}." if self.gpu_status else ""
            blocker = f" Blocker: {self.gpu_blocker}" if self.gpu_blocker else ""
            gaps.append(
                "GPU/SSH: this run requested compute=gpu, but the external-resource manifest blocks real training "
                f"until a fresh GPU smoke passes.{detail}{blocker}"
            )
        return gaps

    def can_execute(self, *, compute_override: Optional[str] = None) -> bool:
        return not self.blocking_setup(compute_override=compute_override)

    def status_rows(self) -> list[tuple[str, str]]:
        if self.gpu_blocked:
            gpu_label = f"blocked ({self.gpu_status or 'manifest blocker'})"
        elif self.gpu_ready:
            gpu_label = "configured"
        else:
            gpu_label = "optional"
        return [
            ("workspace", self.workspace_root),
            ("task", self.selected_task or "(none selected)"),
            ("llm", f"{self.llm_provider} ({'ready' if self.llm_ready else 'setup needed'})"),
            ("kaggle", "ready" if self.kaggle_ready else "setup needed"),
            ("compute", self.compute_backend),
            ("gpu/ssh", gpu_label),
            ("memory", self.memory_summary or "-"),
            ("recent run", self.recent_run_id or "(none yet)"),
            ("tool status", self.tool_readiness or "idle"),
            ("last action", self.last_action or "(none)"),
            ("last artifact", self.last_artifact or "(none)"),
        ]

    def persist(self, root: Optional[Path] = None) -> Optional[Path]:
        base = Path(root or self.workspace_root)
        target_dir = base / PROJECT_DIRNAME
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = asdict(self)
            payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
            path = target_dir / "session.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            return path
        except OSError:
            return None
