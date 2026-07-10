"""Lightweight read-only tools for the EvoMind research terminal.

Every tool returns a structured ``dict`` (never just ``print``) so the caller
can decide whether to render it in the terminal, push it to the dashboard, or
feed it into the LLM's context.  No tool reads or prints secrets, API keys,
Kaggle tokens, SSH passwords, or cookies.

These are the "terminal tools" — they inspect, list, and explain, but they
never start training or submit to Kaggle on their own.  Training goes through
``_run_agent`` / ``AgentSession`` in the main kaggle.py dispatch.
"""
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config, load_config, _SECRET_ENV, _PLAIN_ENV
from .kaggle_session import SessionState
from .tasks import list_tasks


def _redact_url(url: str) -> str:
    """Strip any embedded credentials or keys from a URL for display."""
    if not url:
        return url
    # Crude but safe: never echo full base URLs that might carry query-param tokens.
    if "?" in url:
        return url.split("?")[0] + "?<redacted>"
    return url


def get_model_status(session: SessionState, root: Path) -> dict[str, Any]:
    """Return the current LLM provider, model, readiness, and capabilities.

    Does NOT read or echo the API key.
    """
    cfg = load_config(root)
    provider = str(cfg.get("llm.brand") or cfg.get("llm.provider") or session.llm_provider or "unset")
    model = str(cfg.get("llm.model") or "")
    if not model:
        family = str(cfg.get("llm.provider") or "").lower()
        model = (
            os.environ.get("CLAUDE_CODE_MODEL")
            if family == "anthropic"
            else os.environ.get("DEEPSEEK_MODEL")
        ) or "(provider default)"
    base_url = (
        cfg.get("llm.anthropic_base_url")
        if str(cfg.get("llm.provider") or "").lower() == "anthropic"
        else cfg.get("llm.deepseek_base_url")
    ) or "(provider default)"

    supports_tool_use = str(cfg.get("llm.provider") or "").lower() in ("anthropic",)
    supports_streaming = True  # the gateway supports streaming for both families

    return {
        "ok": True,
        "tool": "model_status",
        "provider": provider,
        "model": model,
        "base_url": _redact_url(str(base_url)),
        "ready": session.llm_ready,
        "tool_use": supports_tool_use,
        "streaming": supports_streaming,
        "note": "API key is never displayed.",
    }


def get_system_status(session: SessionState, root: Path) -> dict[str, Any]:
    """Full readiness summary: LLM, Kaggle, GPU, tasks."""
    return {
        "ok": True,
        "tool": "system_status",
        "llm_ready": session.llm_ready,
        "llm_provider": session.llm_provider,
        "kaggle_ready": session.kaggle_ready,
        "compute_backend": session.compute_backend,
        "gpu_configured": session.gpu_ready,
        "gpu_blocked": session.gpu_blocked,
        "gpu_status": session.gpu_status or "(not declared)",
        "tasks_count": session.n_tasks,
        "selected_task": session.selected_task or "(none)",
        "workspace_root": session.workspace_root,
        "recent_run_id": session.recent_run_id or "(none)",
        "memory_summary": session.memory_summary or "empty",
        "blockers": session.blocking_setup(),
    }


def list_registered_tasks(session: SessionState, root: Path) -> dict[str, Any]:
    """Return all registered tasks with their briefs."""
    tasks = list_tasks(root)
    result = []
    for slug, task_path in tasks:
        brief = ""
        try:
            data = json.loads(task_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                parts = [
                    f"modality={data.get('modality', '?')}",
                    f"task_type={data.get('task_type', '?')}",
                    f"metric={data.get('metric', '?')}",
                ]
                brief = ", ".join(parts)
        except (json.JSONDecodeError, OSError):
            pass
        result.append({"slug": slug, "path": str(task_path), "brief": brief})
    return {
        "ok": True,
        "tool": "task_list",
        "tasks": result,
        "count": len(result),
        "selected": session.selected_task or "(none)",
    }


def inspect_selected_task(session: SessionState, root: Path) -> dict[str, Any]:
    """Detailed info about the currently selected task."""
    if not session.selected_task:
        return {"ok": False, "tool": "inspect_task", "message": "No task selected.", "blockers": ["task"]}
    from .tasks import resolve_task
    try:
        task_path = resolve_task(session.selected_task, project_root=root)
        data = json.loads(task_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "tool": "inspect_task", "message": str(exc)}

    if not isinstance(data, dict):
        return {"ok": False, "tool": "inspect_task", "message": "Task config is not a valid dict."}

    return {
        "ok": True,
        "tool": "inspect_task",
        "slug": session.selected_task,
        "name": data.get("task_name", session.selected_task),
        "modality": data.get("modality", "?"),
        "task_type": data.get("task_type", "?"),
        "metric": data.get("metric", "?"),
        "metric_direction": data.get("metric_direction", "?"),
        "target_column": data.get("target_column", ""),
        "id_column": data.get("id_column", ""),
        "data_schema": (data.get("data_schema", "") or "(unfilled)")[:200],
        "n_train": data.get("n_train", 0),
        "n_test": data.get("n_test", 0),
        "local_data_dir": data.get("local_data_dir", ""),
        "gpu_data_dir": data.get("gpu_data_dir", ""),
        "task_brief": session.task_brief or "(no brief)",
    }


def inspect_data_availability(session: SessionState, root: Path) -> dict[str, Any]:
    """Check whether train.csv, test.csv, and sample_submission.csv exist for the
    selected task's data dir."""
    if not session.selected_task:
        return {"ok": False, "tool": "data_check", "message": "No task selected."}

    from .tasks import resolve_task
    result: dict[str, Any] = {"ok": True, "tool": "data_check", "slug": session.selected_task}
    try:
        task_path = resolve_task(session.selected_task, project_root=root)
        data = json.loads(task_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        result["ok"] = False
        result["message"] = "Cannot read task config."
        return result

    local_dir = data.get("local_data_dir", "") if isinstance(data, dict) else ""
    if not local_dir:
        result["data_dir"] = "(not set)"
        result["train_csv"] = False
        result["test_csv"] = False
        result["sample_submission"] = False
        result["message"] = "No local_data_dir configured. Run `evomind download <task>` or set local_data_dir in the task config."
        return result

    data_path = Path(local_dir)
    files_present = []
    for fname in ("train.csv", "test.csv", "sample_submission.csv"):
        exists = (data_path / fname).exists()
        result[fname.replace(".", "_")] = exists
        if exists:
            files_present.append(fname)

    result["data_dir"] = str(data_path)
    result["files"] = files_present
    if not files_present:
        result["message"] = f"Data dir {data_path} exists but contains no expected CSVs."
    else:
        result["message"] = f"Found: {', '.join(files_present)}"
    return result


def inspect_recent_run(session: SessionState, root: Path) -> dict[str, Any]:
    """Show the latest training run's results."""
    if not session.recent_run_id:
        return {
            "ok": True,
            "tool": "recent_run",
            "run_id": "(none)",
            "message": "No training runs yet. Use `evomind run <task>` or describe your training goal.",
        }
    return {
        "ok": True,
        "tool": "recent_run",
        "run_id": session.recent_run_id,
        "best_cv": session.recent_best_cv,
        "events_path": session.recent_events_path or "",
        "memory_summary": session.memory_summary,
        "message": f"Latest run: {session.recent_run_id}" +
                   (f", best CV: {session.recent_best_cv:.4f}" if session.recent_best_cv is not None else ""),
    }


def inspect_gpu_status(session: SessionState, root: Path) -> dict[str, Any]:
    """GPU configuration and manifest blocker status.  Never prints keys or passwords."""
    cfg = load_config(root)
    host = cfg.get("gpu_ssh.host") or ""
    # Redact: never show the real hostname in full.
    host_display = (host[:4] + "..." if len(host) > 4 else host) if host else "(not configured)"
    return {
        "ok": True,
        "tool": "gpu_status",
        "configured": session.gpu_ready,
        "blocked": session.gpu_blocked,
        "manifest_status": session.gpu_status or "(not declared)",
        "blocker": session.gpu_blocker or "",
        "host": host_display,
        "compute_backend": session.compute_backend,
        "can_execute_gpu": session.gpu_ready and not session.gpu_blocked,
        "suggestion": (
            "GPU blocked — run a fresh GPU smoke test to unblock."
            if session.gpu_blocked
            else "GPU ready."
            if session.gpu_ready
            else "GPU not configured. Use `evomind setup` to add SSH/HPC details."
        ),
    }


def inspect_kaggle_status(session: SessionState, root: Path) -> dict[str, Any]:
    """Kaggle API configuration status. Never prints token or key values."""
    cfg = load_config(root)
    has_token = bool(cfg.get("secrets.kaggle_api_token") or os.environ.get("KAGGLE_API_TOKEN"))
    has_creds = bool(
        (cfg.get("secrets.kaggle_username") and cfg.get("secrets.kaggle_key"))
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    return {
        "ok": True,
        "tool": "kaggle_status",
        "configured": session.kaggle_ready,
        "has_token": has_token,
        "has_credentials": has_creds,
        "message": (
            "Kaggle API ready for downloads and human-gated submissions."
            if session.kaggle_ready
            else "Kaggle API not configured. Run `evomind setup` to configure."
        ),
    }


def open_dashboard_url(session: SessionState, root: Path) -> dict[str, Any]:
    """Return the workstation dashboard URL."""
    cfg = load_config(root)
    url = str(
        cfg.get("workstation.dashboard_url")
        or cfg.get("dashboard.url")
        or "http://127.0.0.1:8088/?page=control"
    )
    return {
        "ok": True,
        "tool": "dashboard",
        "url": url,
        "message": f"Workstation dashboard: {url}",
    }


def explain_next_steps(session: SessionState, root: Path) -> dict[str, Any]:
    """Figure out what gates are blocking and suggest the next action."""
    gaps = session.missing_setup()
    blockers = session.blocking_setup()
    suggestions: list[str] = []

    if not session.selected_task:
        suggestions.append("Select a task: `competitions` to browse, then `task add <url>`.")
    if not session.llm_ready:
        suggestions.append("Configure LLM: `evomind setup` to add an Anthropic or DeepSeek key.")
    if not session.kaggle_ready and session.selected_task:
        suggestions.append("Configure Kaggle API: `evomind setup` for data downloads.")
    if session.selected_task and session.llm_ready:
        if not blockers:
            suggestions.append("Ready to train! Type your research goal or `run`.")
        else:
            suggestions.append(f"Fix blockers before training: {'; '.join(blockers)}")

    return {
        "ok": True,
        "tool": "next_steps",
        "blockers": blockers if blockers else [],
        "gaps": gaps,
        "suggestions": suggestions,
        "selected_task": session.selected_task or "(none)",
        "ready_to_train": len(blockers) == 0 and bool(session.selected_task),
    }


def inspect_evolution_status(session: SessionState, root: Path) -> dict[str, Any]:
    """Show whether EvoMind has durable self-evolution evidence."""
    root = Path(root)
    tracker_path = root / ".xsci" / "evolution_tracker.json"
    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    innovation_path = root / ".xsci" / "innovation_log.json"
    scientist_turns_path = root / ".xsci" / "scientist_turns.jsonl"

    try:
        from .evolution_tracker import EvolutionTracker

        snap = EvolutionTracker(root).current_snapshot()
        tracker = {
            "skill_level": snap.skill_level,
            "total_runs": snap.total_runs,
            "total_promotions": snap.total_promotions,
            "repair_attempts": snap.repair_attempts,
            "repair_successes": snap.repair_successes,
            "innovations_tried": snap.innovations_tried,
            "innovation_successes": snap.innovation_successes,
            "lessons_recorded": snap.lessons_recorded,
            "reusable_lessons": snap.reusable_lessons,
            "failure_lessons": snap.failure_lessons,
            "cross_task_transfers": snap.cross_task_transfers,
            "artifact": str(tracker_path),
        }
    except Exception as exc:
        tracker = {"skill_level": "unknown", "error": type(exc).__name__, "artifact": str(tracker_path)}

    memory_records = 0
    memory_successes = 0
    memory_failures = 0
    try:
        if memory_path.exists():
            payload = json.loads(memory_path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("records", [])
            if isinstance(records, list):
                memory_records = len(records)
                memory_successes = sum(1 for r in records if isinstance(r, dict) and (r.get("what_worked") or r.get("reusable_strategy")))
                memory_failures = sum(1 for r in records if isinstance(r, dict) and (r.get("what_failed") or r.get("failure_pattern")))
    except (json.JSONDecodeError, OSError):
        pass

    innovation = {}
    try:
        if innovation_path.exists():
            payload = json.loads(innovation_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                innovation = {
                    "proposals": len(payload.get("proposals", []) or []),
                    "tried": len(payload.get("tried", []) or []),
                    "successes": payload.get("successes", 0),
                    "failures": payload.get("failures", 0),
                    "artifact": str(innovation_path),
                }
    except (json.JSONDecodeError, OSError):
        innovation = {"artifact": str(innovation_path)}
    if not innovation:
        innovation = {"proposals": 0, "tried": 0, "successes": 0, "failures": 0, "artifact": str(innovation_path)}

    scientist_turns = {
        "turns": 0,
        "latest_turn_id": "",
        "artifact": str(scientist_turns_path),
    }
    try:
        from .scientist_turns import load_recent_scientist_turns

        turns = load_recent_scientist_turns(root, limit=50)
        scientist_turns["turns"] = len(turns)
        if turns:
            scientist_turns["latest_turn_id"] = str(turns[-1].get("turn_id") or "")
    except Exception:
        pass

    return {
        "ok": True,
        "tool": "evolution_status",
        "selected_task": session.selected_task or "(none)",
        "tracker": tracker,
        "retrospective_memory": {
            "records": memory_records,
            "success_records": memory_successes,
            "failure_records": memory_failures,
            "artifact": str(memory_path),
        },
        "innovation": innovation,
        "scientist_turns": scientist_turns,
        "message": (
            "Self-evolution evidence is present."
            if (tracker.get("total_runs", 0) or memory_records or innovation.get("tried", 0) or scientist_turns.get("turns", 0))
            else "No durable self-evolution evidence yet; complete an audited run to create it."
        ),
    }


def get_scientist_checkpoint(session: SessionState, root: Path) -> dict[str, Any]:
    """Return a read-only Observe/Analyze/Propose/Gate/Act checkpoint."""
    from .scientist_state import build_scientist_checkpoint

    return build_scientist_checkpoint(session, root)


def get_research_decision(session: SessionState, root: Path) -> dict[str, Any]:
    """Return and persist the next audited experiment decision."""
    from .scientist_state import build_research_decision

    return build_research_decision(session, root, persist=True)


def get_scientist_workplan(session: SessionState, root: Path) -> dict[str, Any]:
    """Return and persist the multi-step AI Scientist workplan."""
    from .scientist_state import build_scientist_workplan

    return build_scientist_workplan(session, root, persist=True)


def get_scientist_step_trace(session: SessionState, root: Path) -> dict[str, Any]:
    """Return recent step-level AI Scientist events."""
    from .scientist_trace import load_recent_scientist_step_events, scientist_step_trace_path

    events = load_recent_scientist_step_events(root, limit=50)
    return {
        "ok": True,
        "tool": "scientist_step_trace",
        "selected_task": session.selected_task or "",
        "artifact_path": str(scientist_step_trace_path(root)),
        "count": len(events),
        "recent": events,
        "message": (
            "Recent AI Scientist step trace events are available."
            if events
            else "No AI Scientist step trace yet; run `evomind autopilot` or `evomind workplan`."
        ),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }


def get_scientist_turn_plan(session: SessionState, root: Path) -> dict[str, Any]:
    """Return and persist the current per-turn AI Scientist control plan."""
    from .scientist_turn_planner import build_scientist_turn_plan

    return build_scientist_turn_plan(
        session,
        root,
        user_text=session.last_goal or "",
        persist=True,
        record_turn=True,
    )


def _read_json_artifact(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_tail(path: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines[-max(1, limit):]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_json_payload(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _short_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _split_strategy_components(text: str) -> list[str]:
    import re

    raw = _short_text(text, limit=500)
    if not raw:
        return []
    chunks = re.split(r"\s*(?:\+|,|;|\||/| with | and | then |->)\s*", raw, flags=re.IGNORECASE)
    components: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        item = re.sub(r"\s+", " ", chunk).strip(" .:-")
        if len(item) < 3:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        components.append(item[:80])
    return components


def _default_strategy_components(task_type: str, modality: str, metric: str) -> list[str]:
    task_type_l = task_type.lower()
    modality_l = modality.lower()
    metric_l = metric.lower()
    if "time" in task_type_l or "forecast" in task_type_l or "time" in modality_l:
        return [
            "leakage-safe backtesting",
            "calendar and lag feature audit",
            "hierarchical residual model",
            "forecast ensemble with fold stability gate",
        ]
    if "image" in modality_l or "vision" in modality_l:
        return [
            "pretrained backbone fine-tuning",
            "augmentation ablation",
            "test-time augmentation",
            "calibrated ensemble",
        ]
    if "regression" in task_type_l or metric_l in {"rmse", "rmsle", "mae"}:
        return [
            "target transform and inverse audit",
            "robust feature interactions",
            "GBDT and linear residual blend",
            "outlier-aware validation split",
        ]
    return [
        "OOF calibrated ensemble",
        "target/statistical encoding with leakage guard",
        "model-family diversity blend",
        "CV-public gap risk probe",
    ]


def _selected_task_profile(session: SessionState, root: Path) -> dict[str, Any]:
    if not session.selected_task:
        return {}
    try:
        from .tasks import resolve_task

        task_path = resolve_task(session.selected_task, project_root=root)
        payload = _read_json_payload(task_path)
        if isinstance(payload, dict):
            return {
                "task_name": str(payload.get("task_name") or session.selected_task),
                "task_slug": session.selected_task,
                "task_path": str(task_path),
                "modality": str(payload.get("modality") or "tabular"),
                "task_type": str(payload.get("task_type") or "classification"),
                "metric": str(payload.get("metric") or "accuracy"),
                "metric_direction": str(payload.get("metric_direction") or "maximize"),
                "target_column": str(payload.get("target_column") or ""),
                "id_column": str(payload.get("id_column") or ""),
                "data_schema": _short_text(payload.get("data_schema") or "", limit=260),
                "extra_notes": _short_text(payload.get("extra_notes") or "", limit=260),
            }
    except Exception:
        pass
    return {
        "task_name": session.selected_task,
        "task_slug": session.selected_task,
        "modality": "tabular",
        "task_type": "classification",
        "metric": "accuracy",
        "metric_direction": "maximize",
    }


def _memory_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records") or payload.get("items") or payload.get("memory") or []
    else:
        records = []
    return [item for item in records if isinstance(item, dict)]


def _stable_memory_id(prefix: str, *parts: Any) -> str:
    text = "|".join(_short_text(part, limit=700) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _memory_schema_record(
    *,
    memory_id: str,
    task_type: str,
    dataset_profile: dict[str, Any],
    method: str,
    what_worked: str,
    what_failed: str,
    reusable_strategy: str,
    failure_pattern: str,
    linked_exp_ids: list[str],
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "task_type": _short_text(task_type or "unknown", limit=80),
        "dataset_profile": dataset_profile,
        "method": _short_text(method, limit=180),
        "what_worked": _short_text(what_worked, limit=500),
        "what_failed": _short_text(what_failed, limit=500),
        "metric_delta": None,
        "reusable_strategy": _short_text(reusable_strategy, limit=700),
        "failure_pattern": _short_text(failure_pattern, limit=300),
        "linked_exp_ids": [
            _short_text(item, limit=120)
            for item in linked_exp_ids
            if _short_text(item, limit=120)
        ][:8],
    }


def _memory_relevance(record: dict[str, Any], task: dict[str, Any]) -> tuple[int, list[str]]:
    profile = record.get("dataset_profile") if isinstance(record.get("dataset_profile"), dict) else {}
    task_slug = str(task.get("task_slug") or "").strip().lower().replace("_", "-")
    record_slug = str(
        profile.get("task_slug")
        or record.get("task")
        or record.get("task_id")
        or record.get("task_name")
        or ""
    ).strip().lower().replace("_", "-")
    task_type = str(task.get("task_type") or "").strip().lower()
    record_type = str(record.get("task_type") or profile.get("task_type") or "").strip().lower()
    modality = str(task.get("modality") or "").strip().lower()
    record_modality = str(profile.get("modality") or record.get("modality") or "").strip().lower()
    metric = str(task.get("metric") or "").strip().lower()
    record_metric = str(profile.get("metric") or record.get("metric") or "").strip().lower()

    score = 0
    reasons: list[str] = []
    if task_slug and record_slug == task_slug:
        score += 70
        reasons.append("same_task")
    elif task_slug and record_slug and (task_slug in record_slug or record_slug in task_slug):
        score += 50
        reasons.append("task_slug_overlap")
    if task_type and record_type == task_type:
        score += 20
        reasons.append("same_task_type")
    if modality and record_modality == modality:
        score += 15
        reasons.append("same_modality")
    if metric and record_metric == metric:
        score += 10
        reasons.append("same_metric")
    if not record_slug and not record_modality:
        score += 4
        reasons.append("generic_record")

    method_text = " ".join(
        str(record.get(field) or "")
        for field in ("method", "reusable_strategy", "what_worked", "hypothesis")
    ).lower()
    incompatible_markers = {
        "tabular": ("rna", "base-pair", "tf-idf", "essay", "tokenizer", "image augmentation", "backbone"),
        "image": ("tf-idf", "target encoding", "lag feature", "rna"),
        "text": ("image augmentation", "lag feature", "rna base-pair"),
        "time_series": ("tf-idf", "image augmentation", "rna base-pair"),
    }
    modality_key = (
        "time_series"
        if "time" in modality or "forecast" in task_type
        else "image"
        if "image" in modality or "vision" in modality
        else "text"
        if "text" in modality or "nlp" in modality
        else "tabular"
    )
    if record_slug != task_slug and any(marker in method_text for marker in incompatible_markers[modality_key]):
        score -= 60
        reasons.append("cross_modality_method_penalty")
    return score, reasons


def _record_matches_selected_task(record: dict[str, Any], task_slug: str) -> bool:
    if not task_slug:
        return True
    record_task = str(
        record.get("task")
        or record.get("task_id")
        or record.get("selected_task")
        or ""
    ).strip().lower().replace("_", "-")
    if not record_task:
        return True
    expected = task_slug.strip().lower().replace("_", "-")
    return record_task == expected or record_task in expected or expected in record_task


def _innovation_source_records(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    xsci = root / ".xsci"
    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    memory_records = _memory_records_from_payload(_read_json_payload(memory_path))
    task_type = str(task.get("task_type") or "")
    task_slug = str(task.get("task_slug") or "")
    same_type = [
        item for item in memory_records
        if not task_type or str(item.get("task_type") or "").lower() == task_type.lower()
    ]
    ranked_memory: list[tuple[int, int, dict[str, Any], list[str]]] = []
    for index, item in enumerate(memory_records):
        score, reasons = _memory_relevance(item, task)
        ranked_memory.append((score, index, item, reasons))
    ranked_memory.sort(key=lambda row: (-row[0], -row[1]))
    selected_memory: list[dict[str, Any]] = []
    for score, _, item, reasons in ranked_memory:
        if score < 20:
            continue
        enriched = dict(item)
        enriched["_memory_relevance"] = {"score": score, "reasons": reasons}
        selected_memory.append(enriched)
        if len(selected_memory) >= 30:
            break
    if not selected_memory:
        selected_memory = same_type[-12:] if same_type else memory_records[-8:]

    def filtered_tail(name: str, limit: int) -> list[dict[str, Any]]:
        records = _read_jsonl_tail(xsci / name, limit=limit * 3)
        return [
            record
            for record in records
            if isinstance(record, dict) and _record_matches_selected_task(record, task_slug)
        ][-limit:]

    autopilot = _read_json_artifact(xsci / "scientist_autopilot.json")
    if isinstance(autopilot, dict) and not _record_matches_selected_task(autopilot, task_slug):
        autopilot = None
    workplan = _read_json_artifact(xsci / "scientist_workplan.json")
    if isinstance(workplan, dict) and not _record_matches_selected_task(workplan, task_slug):
        workplan = None

    return {
        "retrospective_memory": {
            "path": str(memory_path),
            "records": selected_memory,
            "total_records": len(memory_records),
            "matched_task_type_records": len(same_type),
            "relevant_records": len(selected_memory),
            "exact_task_records": sum(
                1
                for item in selected_memory
                if "same_task" in ((item.get("_memory_relevance") or {}).get("reasons") or [])
            ),
            "rejected_irrelevant_records": max(0, len(memory_records) - len(selected_memory)),
        },
        "loop_lessons": {
            "path": str(xsci / "scientist_loop_lessons.jsonl"),
            "records": filtered_tail("scientist_loop_lessons.jsonl", 40),
        },
        "turns": {
            "path": str(xsci / "scientist_turns.jsonl"),
            "records": filtered_tail("scientist_turns.jsonl", 30),
        },
        "steps": {
            "path": str(xsci / "scientist_step_trace.jsonl"),
            "records": filtered_tail("scientist_step_trace.jsonl", 40),
        },
        "autopilot": {
            "path": str(xsci / "scientist_autopilot.json"),
            "record": autopilot,
        },
        "workplan": {
            "path": str(xsci / "scientist_workplan.json"),
            "record": workplan,
        },
    }


def _collect_strategy_signals(sources: dict[str, Any], defaults: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = (
        "reusable_strategy", "what_worked", "method", "strategy", "hypothesis",
        "selected_branch", "branch_type", "selected_action", "action", "message",
        "answer_preview", "lesson", "fix", "root_cause",
    )
    counts: dict[str, int] = {}
    evidence: list[dict[str, Any]] = []

    def add_component(component: str, source: str, record: dict[str, Any], field: str) -> None:
        key = component.lower()
        counts[key] = counts.get(key, 0) + 1
        if len(evidence) < 18:
            evidence.append({
                "source": source,
                "field": field,
                "component": component,
                "task": _short_text(record.get("task") or record.get("task_id") or record.get("task_name") or "", limit=60),
                "record_id": _short_text(record.get("memory_id") or record.get("turn_id") or record.get("trace_run_id") or record.get("id") or "", limit=80),
                "snippet": _short_text(record.get(field) or record.get("message") or record.get("answer_preview") or "", limit=160),
            })

    for source_name, source in sources.items():
        records = source.get("records") if isinstance(source, dict) else []
        if isinstance(source, dict) and isinstance(source.get("record"), dict):
            records = [source["record"], *(records or [])]
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in fields:
                value = record.get(field)
                if isinstance(value, dict):
                    value = " ".join(str(v) for v in value.values() if not isinstance(v, (dict, list)))
                elif isinstance(value, list):
                    value = " ".join(str(v) for v in value[:6] if not isinstance(v, (dict, list)))
                for component in _split_strategy_components(str(value or "")):
                    add_component(component, source_name, record, field)

    for item in defaults:
        key = item.lower()
        counts.setdefault(key, 1)
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    components = [next((d for d in defaults if d.lower() == key), key) for key in ranked[:10]]
    return components, evidence


def _redacted_memory_text(value: Any, *, limit: int = 220) -> str:
    """Return a compact memory snippet that is safe for terminal/UI display."""
    import re

    text = _short_text(value, limit=limit)
    if not text:
        return ""
    text = re.sub(
        r"(?i)(api[_-]?key|token|cookie|password|passwd|secret|ssh[_-]?key)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}\b", "[redacted-key]", text)
    text = re.sub(r"\bagt_codex_[A-Za-z0-9_-]{6,}\b", "[redacted-token]", text)
    return text


def _memory_record_id(record: dict[str, Any], fallback: str) -> str:
    return _redacted_memory_text(
        record.get("memory_id")
        or record.get("turn_id")
        or record.get("trace_run_id")
        or record.get("id")
        or fallback,
        limit=100,
    )


def _compact_memory_reuse_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    return {
        "status": plan.get("status") or "missing",
        "gate": plan.get("gate") or "memory_reuse_before_execution",
        "reuse_rules": (plan.get("reuse_rules") or [])[:3],
        "avoid_patterns": (plan.get("avoid_patterns") or [])[:3],
        "supporting_memory_ids": (plan.get("supporting_memory_ids") or [])[:8],
        "apply_before": plan.get("apply_before") or [],
    }


def _memory_reuse_plan_has_content(plan: dict[str, Any] | None) -> bool:
    """Return whether a memory plan can actually guide the next branch."""
    if not isinstance(plan, dict):
        return False
    return bool(plan.get("reuse_rules") or plan.get("avoid_patterns"))


def _build_memory_reuse_plan(
    *,
    sources: dict[str, Any],
    components: list[str],
    task_type: str,
    metric: str,
) -> dict[str, Any]:
    """Build the forward-looking part of the self-evolution loop.

    Retrospective memory is useful only if the next planner can see exactly
    which strategies to reuse and which failure modes to avoid before a run is
    allowed to spend compute.
    """
    reuse_rules: list[dict[str, Any]] = []
    avoid_patterns: list[dict[str, Any]] = []
    supporting_ids: list[str] = []

    def remember_id(record_id: str) -> None:
        if record_id and record_id not in supporting_ids and record_id != "default_strategy_components":
            supporting_ids.append(record_id)

    def add_rule(source: str, record: dict[str, Any], field: str, text: Any) -> None:
        snippet = _redacted_memory_text(text, limit=260)
        if not snippet:
            return
        record_id = _memory_record_id(record, f"{source}_{len(reuse_rules) + 1}")
        if any(item.get("strategy") == snippet for item in reuse_rules):
            remember_id(record_id)
            return
        reuse_rules.append({
            "source": source,
            "record_id": record_id,
            "field": field,
            "strategy": snippet,
            "components": _split_strategy_components(snippet)[:5],
            "apply_to": [
                "search_controller_decision",
                "experiment_blueprint",
                "candidate_validation_plan",
            ],
        })
        remember_id(record_id)

    def add_avoid(source: str, record: dict[str, Any], field: str, text: Any) -> None:
        snippet = _redacted_memory_text(text, limit=240)
        if not snippet:
            return
        record_id = _memory_record_id(record, f"{source}_{len(avoid_patterns) + 1}")
        if any(item.get("pattern") == snippet for item in avoid_patterns):
            remember_id(record_id)
            return
        avoid_patterns.append({
            "source": source,
            "record_id": record_id,
            "field": field,
            "pattern": snippet,
            "prevent_by": [
                "same-split validation before promotion",
                "score_promotion_gate hold instead of overwrite",
                "claim_audit before any public claim",
            ],
        })
        remember_id(record_id)

    for source_name, source in sources.items():
        records = source.get("records") if isinstance(source, dict) else []
        if isinstance(source, dict) and isinstance(source.get("record"), dict):
            records = [source["record"], *(records or [])]
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            add_rule(source_name, record, "reusable_strategy", record.get("reusable_strategy"))
            add_rule(source_name, record, "what_worked", record.get("what_worked"))
            add_rule(source_name, record, "lesson", record.get("lesson"))
            add_rule(source_name, record, "strategy", record.get("strategy"))
            add_avoid(source_name, record, "failure_pattern", record.get("failure_pattern"))
            add_avoid(source_name, record, "what_failed", record.get("what_failed"))
            add_avoid(source_name, record, "root_cause", record.get("root_cause"))
            add_avoid(source_name, record, "stop_reason", record.get("stop_reason"))
            add_avoid(source_name, record, "blockers", record.get("blockers"))
            if len(reuse_rules) >= 8 and len(avoid_patterns) >= 8:
                break
        if len(reuse_rules) >= 8 and len(avoid_patterns) >= 8:
            break

    if not reuse_rules:
        for component in components[:4]:
            snippet = _redacted_memory_text(component, limit=160)
            if snippet:
                reuse_rules.append({
                    "source": "default_strategy_components",
                    "record_id": "default_strategy_components",
                    "field": "component",
                    "strategy": snippet,
                    "components": [snippet],
                    "apply_to": [
                        "search_controller_decision",
                        "experiment_blueprint",
                        "candidate_validation_plan",
                    ],
                })

    source_paths = {
        name: source.get("path")
        for name, source in sources.items()
        if isinstance(source, dict) and source.get("path")
    }
    matched_records = 0
    total_records = 0
    retrospective = sources.get("retrospective_memory")
    if isinstance(retrospective, dict):
        matched_records = int(retrospective.get("matched_task_type_records") or 0)
        total_records = int(retrospective.get("total_records") or 0)

    return {
        "status": "ready" if reuse_rules or avoid_patterns else "default_only",
        "gate": "memory_reuse_before_execution",
        "task_type": _redacted_memory_text(task_type or "unknown", limit=80),
        "metric": _redacted_memory_text(metric or "unknown", limit=80),
        "reuse_rules": reuse_rules[:6],
        "avoid_patterns": avoid_patterns[:6],
        "supporting_memory_ids": supporting_ids[:12],
        "matched_task_type_records": matched_records,
        "total_retrospective_records": total_records,
        "source_paths": source_paths,
        "apply_before": [
            "scientist_hypothesis_review",
            "scientist_experiment_blueprint",
            "run_gated_candidate",
        ],
        "instructions": [
            "Prefer branches that reuse at least one memory reuse_rule.",
            "Before any run, check avoid_patterns and add explicit mitigation to validation_plan.",
            "Do not promote or claim improvement unless gates prove the remembered failure mode did not recur.",
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }


def _build_current_memory_reuse_plan(session: SessionState, root: Path) -> dict[str, Any]:
    """Build a fresh memory reuse plan for the currently selected task.

    Older artifacts may predate the memory-reuse bridge and therefore carry an
    empty ``memory_reuse_plan`` even when retrospective memory exists.  This
    helper lets downstream review/blueprint tools repair that stale state
    without starting training.
    """
    task = _selected_task_profile(session, root)
    task_type = str(task.get("task_type") or "classification")
    modality = str(task.get("modality") or "tabular")
    metric = str(task.get("metric") or "accuracy")
    defaults = _default_strategy_components(task_type, modality, metric)
    sources = _innovation_source_records(root, task)
    components, _ = _collect_strategy_signals(sources, defaults)
    if len(components) < 4:
        components = list(dict.fromkeys([*components, *defaults]))
    return _build_memory_reuse_plan(
        sources=sources,
        components=components,
        task_type=task_type,
        metric=metric,
    )


def get_scientist_memory_consolidation(session: SessionState, root: Path) -> dict[str, Any]:
    """Consolidate safe Scientist artifacts into durable retrospective memory.

    This tool closes the self-evolution loop. It reads only non-secret local
    artifacts produced by the Scientist tools, converts lessons and gate
    outcomes into schema-compatible retrospective memory records, and stops
    before training, downloads, or official submission.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_memory_consolidation.json"
    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    task = _selected_task_profile(session, root)
    task_type = str(task.get("task_type") or "unknown")
    task_slug = str(task.get("task_slug") or session.selected_task or "")
    dataset_profile = {
        "task_slug": _short_text(task_slug, limit=120),
        "task_name": _short_text(task.get("task_name") or task_slug, limit=160),
        "modality": _short_text(task.get("modality") or "", limit=80),
        "metric": _short_text(task.get("metric") or "", limit=80),
        "metric_direction": _short_text(task.get("metric_direction") or "", limit=40),
        "source": "scientist_memory_consolidation",
    }

    loop = _read_json_artifact(xsci / "scientist_loop.json") or {}
    lessons = _read_jsonl_tail(xsci / "scientist_loop_lessons.jsonl", limit=12)
    steps = _read_jsonl_tail(xsci / "scientist_step_trace.jsonl", limit=80)
    hypothesis_review = _read_json_artifact(xsci / "scientist_hypothesis_review.json") or {}
    experiment_blueprint = _read_json_artifact(xsci / "scientist_experiment_blueprint.json") or {}
    execution_contract = _read_json_artifact(xsci / "scientist_execution_contract.json") or {}
    action_queue = _read_json_artifact(xsci / "scientist_action_queue.json") or {}
    autopilot = _read_json_artifact(xsci / "scientist_autopilot.json") or {}
    patch_work_order = _read_json_artifact(xsci / "scientist_patch_work_order.json") or {}
    patch_action_queue = _read_json_artifact(xsci / "scientist_patch_action_queue.json") or {}
    patch_trials = _read_jsonl_tail(xsci / "scientist_patch_trials.jsonl", limit=10)
    continuation_resume = _read_json_artifact(xsci / "scientist_continuation_resume.json") or {}

    candidates: list[dict[str, Any]] = []

    if isinstance(loop, dict) and loop:
        lesson = loop.get("lesson") if isinstance(loop.get("lesson"), dict) else {}
        lesson_text = str(lesson.get("lesson") or loop.get("message") or "")
        stop_reason = str(loop.get("stop_reason") or loop.get("mode") or "")
        final_autopilot = loop.get("final_autopilot") if isinstance(loop.get("final_autopilot"), dict) else {}
        blockers = final_autopilot.get("blockers") if isinstance(final_autopilot, dict) else []
        blockers_text = "; ".join(str(item) for item in blockers[:5]) if isinstance(blockers, list) else str(blockers or "")
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_loop", task_slug, loop.get("trace_run_id"), stop_reason, lesson_text),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="bounded_ai_scientist_loop",
            what_worked="EvoMind executed only safe read-only tools, wrote loop and lesson artifacts, and preserved submit/training gates.",
            what_failed=blockers_text or stop_reason,
            reusable_strategy=lesson_text or "Run bounded observe-plan-act-reflect loop before spending compute.",
            failure_pattern=stop_reason or "unknown_loop_stop_reason",
            linked_exp_ids=[str(loop.get("trace_run_id") or ""), str(loop.get("artifact_path") or "")],
        ))

    for item in lessons[-5:]:
        if not isinstance(item, dict):
            continue
        lesson_text = str(item.get("lesson") or "")
        if not lesson_text:
            continue
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_lesson", task_slug, item.get("trace_run_id"), item.get("stop_reason"), lesson_text),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="scientist_loop_lesson",
            what_worked="A bounded Scientist turn produced a reusable lesson without starting training.",
            what_failed=str(item.get("stop_reason") or ""),
            reusable_strategy=lesson_text,
            failure_pattern=str(item.get("stop_reason") or "lesson_recorded"),
            linked_exp_ids=[str(item.get("trace_run_id") or "")],
        ))

    selected_hypothesis = hypothesis_review.get("selected_hypothesis") if isinstance(hypothesis_review.get("selected_hypothesis"), dict) else {}
    blueprint = experiment_blueprint.get("experiment_blueprint") if isinstance(experiment_blueprint.get("experiment_blueprint"), dict) else {}
    if selected_hypothesis or blueprint:
        strategy = str(
            selected_hypothesis.get("strategy_name")
            or blueprint.get("strategy")
            or blueprint.get("branch_type")
            or "reviewed_scientist_hypothesis"
        )
        branch_type = str(blueprint.get("branch_type") or selected_hypothesis.get("proposed_branch_type") or "")
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_blueprint", task_slug, selected_hypothesis.get("hypothesis_id"), blueprint.get("blueprint_id"), strategy),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="hypothesis_review_to_experiment_blueprint",
            what_worked="Hypothesis review and experiment blueprint converted memory-guided ideas into gated, auditable branch plans.",
            what_failed=str(experiment_blueprint.get("status") or hypothesis_review.get("status") or ""),
            reusable_strategy=f"{strategy}; branch_type={branch_type}; require validation_contract, score gate, rollback, and claim audit.",
            failure_pattern=str(experiment_blueprint.get("status") or "blueprint_ready_or_blocked"),
            linked_exp_ids=[
                str(selected_hypothesis.get("hypothesis_id") or ""),
                str(blueprint.get("blueprint_id") or ""),
                str(experiment_blueprint.get("artifact_path") or ""),
            ],
        ))

    if execution_contract:
        root_causes = execution_contract.get("root_causes") or execution_contract.get("blocking_gates") or execution_contract.get("blockers") or []
        if isinstance(root_causes, list):
            failure = "; ".join(str(item) for item in root_causes[:6])
        else:
            failure = str(root_causes or "")
        go_no_go = str(execution_contract.get("go_no_go") or execution_contract.get("status") or "")
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_contract", task_slug, go_no_go, failure),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="execution_contract_gate",
            what_worked="Execution contract made go/no-go, rollback, human gate, and required artifacts explicit before training.",
            what_failed=failure or go_no_go,
            reusable_strategy="Build and honor execution_contract before launching any candidate run; repair no_go root causes first.",
            failure_pattern=go_no_go or "execution_contract_snapshot",
            linked_exp_ids=[str(execution_contract.get("artifact_path") or "")],
        ))

    patch_order_body = patch_work_order.get("work_order") if isinstance(patch_work_order.get("work_order"), dict) else {}
    if patch_work_order or patch_order_body:
        patch_status = str(patch_work_order.get("status") or patch_order_body.get("status") or "")
        issue_id = str(patch_work_order.get("selected_issue_id") or patch_order_body.get("issue_id") or "scientist_patch_work_order")
        title = str(patch_work_order.get("selected_title") or patch_order_body.get("title") or issue_id)
        rationale = str(patch_order_body.get("rationale") or "")
        files_to_edit = patch_order_body.get("files_to_edit") if isinstance(patch_order_body.get("files_to_edit"), list) else []
        acceptance_checks = patch_order_body.get("acceptance_checks") if isinstance(patch_order_body.get("acceptance_checks"), list) else []
        safe_next = str(patch_order_body.get("safe_next_command") or "")
        code_prompt = str(patch_order_body.get("code_agent_prompt") or "")
        if patch_status == "blocked_external_gate":
            what_worked = "EvoMind identified that the latest blocker is external resources/data, so source patching stayed blocked."
            reusable_strategy = (
                f"Do not edit source for {issue_id}; clear the external gate first via {safe_next or 'evomind ready'} "
                "and refresh resource proof before retrying code-agent work."
            )
            what_failed = rationale or title
        else:
            what_worked = (
                "EvoMind converted failure evidence into a code-agent work order with scoped files, "
                "acceptance checks, rollback condition, and human gate."
            )
            checks_text = "; ".join(str(item) for item in acceptance_checks[:5])
            files_text = "; ".join(str(item) for item in files_to_edit[:5])
            reusable_strategy = (
                code_prompt
                or f"Patch {issue_id} by editing {files_text or 'the scoped files'} and validating with {checks_text or 'the recorded acceptance checks'}."
            )
            what_failed = rationale or title
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_patch_work_order", task_slug, issue_id, patch_status, title),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="scientist_patch_work_order",
            what_worked=what_worked,
            what_failed=what_failed,
            reusable_strategy=reusable_strategy,
            failure_pattern=patch_status or issue_id,
            linked_exp_ids=[
                str(patch_work_order.get("artifact_path") or ""),
                str(patch_work_order.get("action_queue_path") or ""),
                str(patch_work_order.get("trials_path") or ""),
            ],
        ))

    for item in patch_trials[-5:]:
        if not isinstance(item, dict):
            continue
        lesson_text = str(item.get("lesson") or "")
        if not lesson_text:
            continue
        issue_id = str(item.get("selected_issue_id") or "patch_trial")
        status = str(item.get("status") or "recorded")
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_patch_trial", task_slug, issue_id, status, lesson_text),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="scientist_patch_trial_lesson",
            what_worked="Patch-order trial log captured the repair decision and preserved no-training/no-submit gates.",
            what_failed=lesson_text if status != "ready_for_code_agent" else "",
            reusable_strategy=lesson_text,
            failure_pattern=status,
            linked_exp_ids=[str(item.get("source") or "scientist_patch_work_order")],
        ))

    if continuation_resume:
        resume_status = str(continuation_resume.get("status") or "")
        stop_reason = str(continuation_resume.get("stop_reason") or resume_status or "unknown")
        executed_tools = [
            str(tool)
            for tool in (continuation_resume.get("executed_tools") or [])
            if str(tool)
        ]
        remaining_tools = [
            str(tool)
            for tool in (continuation_resume.get("remaining_safe_tools") or [])
            if str(tool)
        ]
        resume_steps = continuation_resume.get("steps") if isinstance(continuation_resume.get("steps"), list) else []
        step_tools = [
            str(item.get("executed_tool") or "")
            for item in resume_steps
            if isinstance(item, dict) and str(item.get("executed_tool") or "")
        ]
        tool_sequence = list(dict.fromkeys([*executed_tools, *step_tools]))
        worked = (
            "Continuation resume closed the incomplete Scientist turn by consuming only read-only safe tools."
            if resume_status == "closed" and not remaining_tools
            else "Continuation resume preserved the read-only gate and stopped before unsafe execution."
        )
        failed = (
            f"stop_reason={stop_reason}; remaining_safe_tools={', '.join(remaining_tools[:6]) or '(none)'}"
            if resume_status != "closed" or remaining_tools
            else ""
        )
        reusable = (
            "When a Scientist turn exhausts its tool budget, run evomind resume-continuation before training; "
            f"expected safe sequence={', '.join(tool_sequence[:8]) or 'read-only continuation tools'}; "
            "stop immediately if blocked_by_gate, stalled, or remaining tools do not shrink."
        )
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id(
                "scientist_continuation_resume",
                task_slug,
                continuation_resume.get("generated_at"),
                resume_status,
                stop_reason,
                tool_sequence,
                remaining_tools,
            ),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="scientist_continuation_resume",
            what_worked=worked,
            what_failed=failed,
            reusable_strategy=reusable,
            failure_pattern=stop_reason,
            linked_exp_ids=[
                str(continuation_resume.get("artifact_path") or ""),
                str(continuation_resume.get("continuation_status_artifact_path") or ""),
                str(continuation_resume.get("continuation_artifact_path") or ""),
            ],
        ))

    blocked_events = [
        event for event in steps
        if isinstance(event, dict) and str(event.get("status") or "").lower() in {"blocked", "failed"}
    ]
    if blocked_events:
        event = blocked_events[-1]
        candidates.append(_memory_schema_record(
            memory_id=_stable_memory_id("scientist_trace", task_slug, event.get("phase"), event.get("tool"), event.get("message")),
            task_type=task_type,
            dataset_profile=dataset_profile,
            method="step_trace_blocker_pattern",
            what_worked="Step trace preserved the blocked tool, phase, message, and artifact path for recovery.",
            what_failed=str(event.get("message") or event.get("status") or ""),
            reusable_strategy="When a step trace blocks, use the artifact path and phase to select repair-plan or execution-contract before retrying.",
            failure_pattern=str(event.get("phase") or event.get("tool") or "blocked_step"),
            linked_exp_ids=[str(event.get("trace_run_id") or ""), str(event.get("artifact_path") or "")],
        ))

    existing = _memory_records_from_payload(_read_json_payload(memory_path))
    by_id: dict[str, dict[str, Any]] = {
        str(item.get("memory_id") or _stable_memory_id("legacy_memory", item)): item
        for item in existing
        if isinstance(item, dict)
    }
    added_ids: list[str] = []
    for record in candidates:
        memory_id = str(record.get("memory_id") or "")
        if not memory_id:
            continue
        if memory_id not in by_id:
            added_ids.append(memory_id)
        by_id[memory_id] = record

    merged_records = list(by_id.values())
    write_ok = True
    write_error = ""
    try:
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = memory_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged_records, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(memory_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        write_ok = False
        write_error = str(exc)

    source_counts = {
        "loop_present": bool(loop),
        "lessons": len(lessons),
        "step_events": len(steps),
        "blocked_step_events": len(blocked_events),
        "hypothesis_review_present": bool(hypothesis_review),
        "experiment_blueprint_present": bool(experiment_blueprint),
        "execution_contract_present": bool(execution_contract),
        "action_queue_present": bool(action_queue),
        "autopilot_present": bool(autopilot),
        "patch_work_order_present": bool(patch_work_order),
        "patch_action_queue_present": bool(patch_action_queue),
        "patch_trials": len(patch_trials),
        "continuation_resume_present": bool(continuation_resume),
    }
    payload: dict[str, Any] = {
        "ok": write_ok,
        "tool": "scientist_memory_consolidation",
        "generated_at": generated_at,
        "selected_task": task_slug,
        "task_profile": task,
        "source_counts": source_counts,
        "records_before": len(existing),
        "candidate_records": len(candidates),
        "records_added": len(added_ids),
        "records_total": len(merged_records),
        "added_memory_ids": added_ids,
        "memory_path": str(memory_path),
        "artifact_path": str(artifact_path),
        "message": (
            f"Consolidated {len(added_ids)} new Scientist memory records."
            if write_ok else f"Could not write memory artifact: {write_error}"
        ),
        "next_safe_commands": [
            "evomind innovate-plan",
            "evomind self-audit",
            "evomind situation",
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    if write_error:
        payload["write_error"] = write_error

    try:
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(artifact_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write consolidation artifact: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"memory_consolidation_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_memory_consolidation",
            "task": task_slug,
            "phase": "memory_consolidation",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_memory_consolidation",
            "message": payload["message"],
            "artifact_path": str(artifact_path),
            "details": {"records_added": len(added_ids), "records_total": len(merged_records)},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": task_slug,
            "route": "scientist_memory_consolidation",
            "user": "scientist_memory_consolidation",
            "forced_tools": ["scientist_loop", "scientist_step_trace", "scientist_execution_contract"],
            "executed_tools": [{"tool": "scientist_memory_consolidation", "ok": payload.get("ok", True)}],
            "mode": "memory_writeback",
            "decision": {"records_added": len(added_ids), "records_total": len(merged_records)},
            "blockers": [],
            "next_actions": payload["next_safe_commands"],
            "artifacts": [str(artifact_path), str(memory_path)],
            "answer_preview": payload["message"],
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def get_scientist_innovation_backlog(session: SessionState, root: Path) -> dict[str, Any]:
    """Generate memory-guided innovation hypotheses without training.

    This tool is the missing bridge between "self-audit says we need memory
    reuse" and "start a costly run".  It mines existing non-secret artifacts,
    proposes auditable branches, updates proposal memory, and stops at gates.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_innovation_backlog.json"
    innovation_log_path = xsci / "innovation_log.json"
    task = _selected_task_profile(session, root)
    task_type = str(task.get("task_type") or "classification")
    modality = str(task.get("modality") or "tabular")
    metric = str(task.get("metric") or "accuracy")
    metric_direction = str(task.get("metric_direction") or "maximize")
    defaults = _default_strategy_components(task_type, modality, metric)
    sources = _innovation_source_records(root, task)
    components, evidence = _collect_strategy_signals(sources, defaults)
    if len(components) < 4:
        components = list(dict.fromkeys([*components, *defaults]))
    memory_reuse_plan = _build_memory_reuse_plan(
        sources=sources,
        components=components,
        task_type=task_type,
        metric=metric,
    )

    def hypothesis(idx: int, strategy_name: str, selected: list[str],
                   branch_type: str, code_mode: str, rationale: str) -> dict[str, Any]:
        return {
            "id": f"innov_{generated_at[:10].replace('-', '')}_{idx:02d}",
            "strategy_name": strategy_name,
            "components": selected,
            "rationale": rationale,
            "evidence_records": evidence[:8],
            "risk_controls": [
                "start from current best-so-far; never overwrite promoted artifacts",
                f"evaluate with declared metric={metric} direction={metric_direction}",
                "require same-split CV/OOF evidence before promotion",
                "write validation_contract, score_promotion_gate, and claim_audit",
                "block official Kaggle submit until explicit human approval and Kaggle response artifact",
            ],
            "expected_artifacts": [
                "search_controller_decision.json",
                "agent_trace.jsonl",
                "metrics.json",
                "oof_predictions or validation predictions",
                "submission.csv when task format is known",
                "validation_contract.json",
                "score_promotion_gate.json",
                "claim_audit.json",
            ],
            "gate": "proposal_only_requires_execution_contract_before_training",
            "proposed_branch_type": branch_type,
            "code_generation_mode": code_mode,
            "memory_reuse_plan": _compact_memory_reuse_plan(memory_reuse_plan),
            "ready_for_training": False,
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }

    c = components
    proposals = [
        hypothesis(
            1,
            "memory_guided_frontier_blend",
            c[:3],
            "ensemble_blend",
            "stepwise",
            "Combine the strongest reusable signals into a low-cost blend first, because this spends less compute and protects best-so-far.",
        ),
        hypothesis(
            2,
            "validation_gap_risk_probe",
            [c[0], "adversarial validation", "fold stability audit"],
            "validation_risk",
            "diff",
            "Before scaling training, test whether CV evidence is trustworthy and whether the candidate is likely to transfer to leaderboard data.",
        ),
        hypothesis(
            3,
            "feature_family_ablation_ladder",
            [c[1] if len(c) > 1 else defaults[1], "single-factor ablation", "leakage guard"],
            "feature_engineering",
            "stepwise",
            "Promote only feature families whose isolated ablation improves the declared metric under the same split.",
        ),
        hypothesis(
            4,
            "model_family_diversity_asset",
            [c[2] if len(c) > 2 else defaults[2], "probability asset export", "best-so-far blend gate"],
            "model_family",
            "base",
            "Create a diverse probability or prediction asset for future blending, but do not submit or promote it without gate evidence.",
        ),
    ]

    memory_summary = {
        "retrospective_memory_records": sources["retrospective_memory"]["total_records"],
        "matched_task_type_records": sources["retrospective_memory"]["matched_task_type_records"],
        "task_relevant_records": sources["retrospective_memory"].get("relevant_records", 0),
        "exact_task_records": sources["retrospective_memory"].get("exact_task_records", 0),
        "rejected_irrelevant_records": sources["retrospective_memory"].get("rejected_irrelevant_records", 0),
        "loop_lessons": len(sources["loop_lessons"]["records"]),
        "turns_considered": len(sources["turns"]["records"]),
        "step_events_considered": len(sources["steps"]["records"]),
        "strategy_components_considered": len(components),
    }
    source_paths = {
        name: source.get("path")
        for name, source in sources.items()
        if isinstance(source, dict) and source.get("path")
    }

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_innovation_backlog",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "task_profile": task,
        "memory_summary": memory_summary,
        "memory_reuse_plan": memory_reuse_plan,
        "source_paths": source_paths,
        "innovation_hypotheses": proposals,
        "next_safe_commands": [
            "evomind innovate-plan",
            "evomind contract",
            "evomind autopilot",
            "evomind loop",
        ],
        "artifact_path": str(artifact_path),
        "innovation_log_path": str(innovation_log_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
        "message": "Innovation hypotheses generated from existing memory artifacts; training has not started.",
    }

    try:
        xsci.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_payload = _read_json_artifact(innovation_log_path) or {}
        existing = log_payload.get("proposals") if isinstance(log_payload.get("proposals"), list) else []
        existing_keys = {
            str(item.get("id") or item.get("strategy_name") or "")
            for item in existing
            if isinstance(item, dict)
        }
        stamped = []
        for item in proposals:
            proposal = dict(item)
            proposal["status"] = "proposed"
            proposal["source_tool"] = "scientist_innovation_backlog"
            proposal["created_at"] = generated_at
            key = str(proposal.get("id") or proposal.get("strategy_name"))
            if key not in existing_keys:
                stamped.append(proposal)
        merged = [*(existing if isinstance(existing, list) else []), *stamped][-50:]
        log_payload.update({
            "proposals": merged,
            "tried": log_payload.get("tried") if isinstance(log_payload.get("tried"), list) else [],
            "successes": int(log_payload.get("successes") or 0),
            "failures": int(log_payload.get("failures") or 0),
            "updated_at": generated_at,
            "no_training_started": True,
        })
        innovation_log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["innovation_log_updated"] = True
        payload["proposal_count_written"] = len(stamped)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write innovation backlog: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"innovation_backlog_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_innovation_backlog",
            "task": session.selected_task or "",
            "phase": "memory_guided_innovation",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_innovation_backlog",
            "message": f"Generated {len(proposals)} innovation hypotheses from memory_summary={memory_summary}",
            "artifact_path": str(artifact_path),
            "details": {"memory_summary": memory_summary, "proposal_count": len(proposals)},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_innovation_backlog",
            "user": "scientist_innovation_backlog",
            "forced_tools": ["evolution_status", "scientist_workplan", "scientist_step_trace"],
            "executed_tools": [{"tool": "scientist_innovation_backlog", "ok": payload.get("ok", True)}],
            "mode": "proposal_only_no_training",
            "decision": {"proposal_count": len(proposals), "memory_summary": memory_summary},
            "blockers": [],
            "next_actions": payload["next_safe_commands"],
            "artifacts": [str(artifact_path), str(innovation_log_path)],
            "answer_preview": f"innovation backlog generated; proposals={len(proposals)}; no_training_started=True",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def get_scientist_hypothesis_review(session: SessionState, root: Path) -> dict[str, Any]:
    """Critique and rank memory-guided hypotheses before any training."""
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_hypothesis_review.json"
    backlog_path = xsci / "scientist_innovation_backlog.json"

    backlog = _read_json_artifact(backlog_path)
    if not isinstance(backlog, dict) or not isinstance(backlog.get("innovation_hypotheses"), list):
        backlog = get_scientist_innovation_backlog(session, root)
    elif not _memory_reuse_plan_has_content(backlog.get("memory_reuse_plan")):
        # Stale backlog artifacts from older builds may contain proposals but no
        # forward memory plan. Refresh them before ranking hypotheses so the
        # next branch cannot silently ignore retrospective lessons.
        refreshed_backlog = get_scientist_innovation_backlog(session, root)
        if isinstance(refreshed_backlog, dict) and isinstance(refreshed_backlog.get("innovation_hypotheses"), list):
            backlog = refreshed_backlog

    hypotheses = [
        item for item in (backlog.get("innovation_hypotheses") or [])
        if isinstance(item, dict)
    ]
    task = backlog.get("task_profile") if isinstance(backlog.get("task_profile"), dict) else _selected_task_profile(session, root)
    memory_summary = backlog.get("memory_summary") if isinstance(backlog.get("memory_summary"), dict) else {}
    memory_reuse_plan = backlog.get("memory_reuse_plan") if isinstance(backlog.get("memory_reuse_plan"), dict) else {}
    if not _memory_reuse_plan_has_content(memory_reuse_plan):
        memory_reuse_plan = _build_current_memory_reuse_plan(session, root)

    try:
        data = inspect_data_availability(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        data = {"ok": False, "tool": "data_check", "message": type(exc).__name__}
    try:
        contract = get_scientist_execution_contract(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        contract = {"ok": False, "tool": "scientist_execution_contract", "go_no_go": "no_go", "root_causes": [type(exc).__name__]}

    data_ready = bool(data.get("train_csv")) or "Remote GPU training data is declared." in " ".join(
        str(item) for item in (contract.get("analyze") or [])
    )
    contract_go = str(contract.get("go_no_go") or "no_go")
    memory_records = int(memory_summary.get("retrospective_memory_records") or 0)
    matched_records = int(memory_summary.get("matched_task_type_records") or 0)
    components_considered = int(memory_summary.get("strategy_components_considered") or 0)

    def score_hypothesis(item: dict[str, Any], rank: int) -> dict[str, Any]:
        components = [str(x) for x in (item.get("components") or []) if str(x).strip()]
        evidence_records = [x for x in (item.get("evidence_records") or []) if isinstance(x, dict)]
        branch = str(item.get("proposed_branch_type") or "")
        mode = str(item.get("code_generation_mode") or "")

        evidence_score = min(35, len(evidence_records) * 4 + min(10, matched_records) + (5 if memory_records else 0))
        readiness_score = 10 + (15 if session.selected_task else 0) + (15 if data_ready else 0) + (10 if contract_go != "no_go" else 0)
        impact_score = 10 + min(15, len(components) * 4)
        if branch in {"ensemble_blend", "model_family"}:
            impact_score += 8
        if branch in {"validation_risk", "feature_engineering"}:
            impact_score += 5
        risk_penalty = 0
        if not data_ready:
            risk_penalty += 12
        if contract_go == "no_go":
            risk_penalty += 14
        if branch == "model_family" and mode == "base":
            risk_penalty += 4
        if len(evidence_records) < 2:
            risk_penalty += 6

        total = max(0, min(100, evidence_score + readiness_score + impact_score - risk_penalty))
        status = (
            "ready_for_execution_contract" if total >= 75 and data_ready and contract_go != "no_go"
            else "hold_until_gate_clear" if total >= 55
            else "needs_more_evidence"
        )
        blockers = []
        if not data_ready:
            blockers.append("training data is not locally available or declared through the remote data contract")
        if contract_go == "no_go":
            blockers.append("execution contract is no_go")
        if len(evidence_records) < 2:
            blockers.append("insufficient hypothesis-specific evidence records")

        return {
            "rank": rank,
            "hypothesis_id": item.get("id") or f"hypothesis_{rank}",
            "strategy_name": item.get("strategy_name") or item.get("id") or f"hypothesis_{rank}",
            "branch_type": branch,
            "code_generation_mode": mode,
            "score": total,
            "evidence_score": evidence_score,
            "readiness_score": readiness_score,
            "impact_score": impact_score,
            "risk_penalty": risk_penalty,
            "risk_level": "low" if risk_penalty <= 6 else "medium" if risk_penalty <= 18 else "high",
            "status": status,
            "reasons": [
                f"evidence_records={len(evidence_records)}",
                f"components={len(components)}",
                f"data_ready={data_ready}",
                f"contract={contract_go}",
                f"branch={branch or 'unknown'}",
            ],
            "blockers": blockers,
            "next_gate": (
                "evomind contract" if status == "ready_for_execution_contract"
                else "evomind repair" if blockers
                else "evomind innovate-plan"
            ),
            "memory_reuse_plan": _compact_memory_reuse_plan(memory_reuse_plan),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }

    reviews = [score_hypothesis(item, idx) for idx, item in enumerate(hypotheses, start=1)]
    reviews.sort(key=lambda row: (-int(row.get("score") or 0), str(row.get("strategy_name") or "")))
    for idx, row in enumerate(reviews, start=1):
        row["rank"] = idx

    selected = reviews[0] if reviews else None
    recommendation = (
        "execute_contract_then_gated_run" if selected and selected.get("status") == "ready_for_execution_contract"
        else "clear_gates_before_training" if selected
        else "generate_innovation_backlog_first"
    )
    next_safe_commands = ["evomind review-hypotheses", "evomind contract", "evomind autopilot"]
    if recommendation != "execute_contract_then_gated_run":
        next_safe_commands.insert(1, "evomind repair")

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_hypothesis_review",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "task_profile": task,
        "memory_summary": memory_summary,
        "memory_reuse_plan": memory_reuse_plan,
        "hypotheses_reviewed": len(reviews),
        "reviews": reviews,
        "selected_hypothesis": selected,
        "recommendation": recommendation,
        "gate_summary": {
            "data_ready": data_ready,
            "execution_contract": contract_go,
            "contract_artifact_path": contract.get("artifact_path") if isinstance(contract, dict) else "",
            "memory_records": memory_records,
            "matched_task_type_records": matched_records,
            "strategy_components_considered": components_considered,
        },
        "next_safe_commands": next_safe_commands,
        "artifact_path": str(artifact_path),
        "source_backlog_path": str(backlog_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
        "message": "Hypotheses reviewed and ranked without starting training.",
    }
    try:
        xsci.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["ok"] = False
        payload["artifact_error"] = str(exc)

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"hypothesis_review_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_hypothesis_review",
            "task": session.selected_task or "",
            "phase": "hypothesis_review",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_hypothesis_review",
            "message": f"Reviewed {len(reviews)} hypotheses; recommendation={recommendation}",
            "artifact_path": str(artifact_path),
            "details": {"selected_hypothesis": selected, "gate_summary": payload["gate_summary"]},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_hypothesis_review",
            "user": "scientist_hypothesis_review",
            "forced_tools": ["scientist_innovation_backlog", "data_check", "scientist_execution_contract"],
            "executed_tools": [{"tool": "scientist_hypothesis_review", "ok": payload.get("ok", True)}],
            "mode": recommendation,
            "decision": {"selected_hypothesis": selected, "hypotheses_reviewed": len(reviews)},
            "blockers": selected.get("blockers", []) if isinstance(selected, dict) else [],
            "next_actions": next_safe_commands,
            "artifacts": [str(artifact_path), str(backlog_path)],
            "answer_preview": f"hypothesis review complete; recommendation={recommendation}; no_training_started=True",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def get_scientist_experiment_blueprint(session: SessionState, root: Path) -> dict[str, Any]:
    """Turn the reviewed hypothesis into a gated experiment blueprint.

    This is the bridge between "thinking" and "doing": it creates the exact
    branch/code/resource/artifact contract that a future run must satisfy, but it
    never starts training and never submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_experiment_blueprint.json"
    review_path = xsci / "scientist_hypothesis_review.json"

    review = _read_json_artifact(review_path)
    if not isinstance(review, dict) or not isinstance(review.get("selected_hypothesis"), dict):
        review = get_scientist_hypothesis_review(session, root)
    selected = review.get("selected_hypothesis") if isinstance(review, dict) else {}
    if not isinstance(selected, dict):
        selected = {}
    memory_reuse_plan = (
        review.get("memory_reuse_plan")
        if isinstance(review, dict) and isinstance(review.get("memory_reuse_plan"), dict)
        else selected.get("memory_reuse_plan") if isinstance(selected.get("memory_reuse_plan"), dict)
        else {}
    )
    if not _memory_reuse_plan_has_content(memory_reuse_plan):
        review = get_scientist_hypothesis_review(session, root)
        selected = review.get("selected_hypothesis") if isinstance(review, dict) else {}
        if not isinstance(selected, dict):
            selected = {}
        memory_reuse_plan = (
            review.get("memory_reuse_plan")
            if isinstance(review, dict) and isinstance(review.get("memory_reuse_plan"), dict)
            else selected.get("memory_reuse_plan") if isinstance(selected.get("memory_reuse_plan"), dict)
            else {}
        )
    if not _memory_reuse_plan_has_content(memory_reuse_plan):
        memory_reuse_plan = _build_current_memory_reuse_plan(session, root)
    try:
        decision_result = get_research_decision(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        decision_result = {"ok": False, "tool": "research_decision", "message": type(exc).__name__, "decision": {}}
    decision = decision_result.get("decision") if isinstance(decision_result, dict) else {}
    if not isinstance(decision, dict):
        decision = {}
    try:
        contract = get_scientist_execution_contract(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        contract = {"ok": False, "tool": "scientist_execution_contract", "go_no_go": "no_go", "root_causes": [type(exc).__name__]}

    task = session.selected_task or ""
    branch = str(selected.get("branch_type") or decision.get("selected_branch") or "baseline")
    code_mode = str(selected.get("code_generation_mode") or decision.get("code_generation_mode") or "Stepwise")
    strategy = str(selected.get("strategy_name") or selected.get("hypothesis_id") or "unreviewed_hypothesis")
    resource_mode = str(getattr(session, "current_compute_override", None) or session.compute_backend or "local")
    go_no_go = str(contract.get("go_no_go") or "no_go") if isinstance(contract, dict) else "no_go"
    blockers = list(selected.get("blockers") or []) if isinstance(selected.get("blockers"), list) else []
    if isinstance(contract, dict):
        blockers.extend(str(x) for x in (contract.get("root_causes") or []) if str(x))
    blockers = list(dict.fromkeys(blockers))
    ready = bool(selected) and go_no_go != "no_go" and not blockers
    blueprint_status = "ready_for_gated_execution" if ready else "blocked_until_gates_clear"
    expected_delta = decision.get("expected_delta") or review.get("expected_delta") or "evidence_bound_local_delta_required"
    rollback_condition = str(
        (contract.get("rollback_condition") if isinstance(contract, dict) else "")
        or decision.get("rollback_condition")
        or "hold candidate and preserve best-so-far if score_promotion_gate fails"
    )
    required_artifacts = [
        "agent_trace.jsonl",
        "metrics.json",
        "oof_predictions.*",
        "submission.csv",
        "artifact_manifest.json",
        "validation_contract.json",
        "submission_audit.json",
        "score_promotion_gate.json",
        "claim_audit.json",
        "retrospective_memory_update.json",
    ]
    blueprint_id = (
        f"bp_{task or 'no_task'}_"
        f"{str(selected.get('hypothesis_id') or strategy).lower().replace(' ', '_')[:40]}_"
        f"{generated_at.replace(':', '').replace('+', 'Z')}"
    )
    experiment_blueprint = {
        "blueprint_id": blueprint_id,
        "task_id": task,
        "hypothesis_id": selected.get("hypothesis_id") or "",
        "strategy_name": strategy,
        "branch_type": branch,
        "code_generation_mode": code_mode,
        "resource_mode": resource_mode,
        "run_command": f"evomind run {task}" if task else "evomind task add <kaggle-url>",
        "dry_run_command": "evomind contract",
        "expected_delta": expected_delta,
        "rollback_condition": rollback_condition,
        "validation_plan": [
            "Verify train/test/schema or declared remote data contract before execution.",
            "Apply memory_reuse_plan rules before code generation and document every reused strategy.",
            "Mitigate memory_reuse_plan avoid_patterns before training; hold candidates that repeat known failures.",
            "Use OOF or fold validation tied to the competition metric.",
            "Generate a submission candidate but keep official Kaggle submit blocked.",
            "Promote only when score_promotion_gate and claim_audit pass.",
        ],
        "required_artifacts": required_artifacts,
        "promotion_gates": [
            "execution_contract_go_or_conditional_go",
            "validation_contract_passed",
            "submission_audit_passed",
            "score_promotion_gate_promoted_or_hold",
            "claim_audit_passed_without_official_overclaim",
        ],
        "memory_writeback_plan": {
            "target": "experiments/evolution/retrospective_memory.json",
            "write_when": "after run artifact_manifest, metrics, gate status, and claim_audit exist",
            "fields": [
                "task_id",
                "blueprint_id",
                "hypothesis_id",
                "branch_type",
                "code_generation_mode",
                "metric_delta",
                "gate_status",
                "failure_reason_or_lesson",
                "reusable_strategy",
            ],
        },
        "memory_reuse_plan": memory_reuse_plan,
        "claim_boundary": "No rank, medal, or official-score claim is allowed without a Kaggle response artifact.",
    }
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_experiment_blueprint",
        "generated_at": generated_at,
        "selected_task": task,
        "blueprint_status": blueprint_status,
        "selected_hypothesis": selected,
        "memory_reuse_plan": memory_reuse_plan,
        "experiment_blueprint": experiment_blueprint,
        "gate_summary": {
            "hypothesis_review": review.get("recommendation") if isinstance(review, dict) else "missing",
            "execution_contract": go_no_go,
            "blockers": blockers,
            "ready_for_gated_execution": ready,
        },
        "next_safe_commands": (
            [f"evomind run {task}", "evomind live", "evomind report"]
            if ready and task else
            ["evomind repair", "evomind contract", "evomind review-hypotheses"]
        ),
        "artifact_path": str(artifact_path),
        "source_review_path": str(review_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "message": "Experiment blueprint generated without starting training.",
    }
    try:
        xsci.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["ok"] = False
        payload["artifact_error"] = str(exc)

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"experiment_blueprint_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_experiment_blueprint",
            "task": task,
            "phase": "experiment_blueprint",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_experiment_blueprint",
            "message": f"Generated blueprint status={blueprint_status}; strategy={strategy}",
            "artifact_path": str(artifact_path),
            "details": {"blueprint": experiment_blueprint, "gate_summary": payload["gate_summary"]},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": task,
            "route": "scientist_experiment_blueprint",
            "user": "scientist_experiment_blueprint",
            "forced_tools": ["scientist_hypothesis_review", "research_decision", "scientist_execution_contract"],
            "executed_tools": [{"tool": "scientist_experiment_blueprint", "ok": payload.get("ok", True)}],
            "mode": blueprint_status,
            "decision": {"blueprint": experiment_blueprint, "selected_hypothesis": selected},
            "blockers": blockers,
            "next_actions": payload["next_safe_commands"],
            "artifacts": [str(artifact_path), str(review_path)],
            "answer_preview": f"experiment blueprint complete; status={blueprint_status}; no_training_started=True",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def get_scientist_innovation_trial_feedback(session: SessionState, root: Path) -> dict[str, Any]:
    """Record hypothesis/blueprint gate feedback into the innovation log.

    This closes a safe self-evolution loop without starting training: proposal
    -> hypothesis review -> experiment blueprint -> gate outcome -> feedback.
    It records whether the current innovation is ready for explicit gated
    execution or blocked by gates, then writes the lesson back into
    ``.xsci/innovation_log.json`` idempotently.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_innovation_trial_feedback.json"
    innovation_log_path = xsci / "innovation_log.json"
    review_path = xsci / "scientist_hypothesis_review.json"
    blueprint_path = xsci / "scientist_experiment_blueprint.json"
    contract_path = xsci / "scientist_execution_contract.json"
    action_queue_path = xsci / "scientist_action_queue.json"
    loop_path = xsci / "scientist_loop.json"

    review = _read_json_artifact(review_path)
    if not isinstance(review, dict) or not isinstance(review.get("selected_hypothesis"), dict):
        review = get_scientist_hypothesis_review(session, root)
    selected = review.get("selected_hypothesis") if isinstance(review, dict) else {}
    if not isinstance(selected, dict):
        selected = {}

    blueprint_payload = _read_json_artifact(blueprint_path)
    if not isinstance(blueprint_payload, dict) or not isinstance(blueprint_payload.get("experiment_blueprint"), dict):
        blueprint_payload = get_scientist_experiment_blueprint(session, root)
    blueprint = blueprint_payload.get("experiment_blueprint") if isinstance(blueprint_payload, dict) else {}
    if not isinstance(blueprint, dict):
        blueprint = {}

    contract = _read_json_artifact(contract_path)
    if not isinstance(contract, dict):
        try:
            contract = get_scientist_execution_contract(session, root)
        except Exception as exc:  # pragma: no cover - defensive only
            contract = {"ok": False, "tool": "scientist_execution_contract", "go_no_go": "no_go", "root_causes": [type(exc).__name__]}
    action_queue = _read_json_artifact(action_queue_path) or {}
    loop = _read_json_artifact(loop_path) or {}

    gate_summary = blueprint_payload.get("gate_summary") if isinstance(blueprint_payload, dict) else {}
    if not isinstance(gate_summary, dict):
        gate_summary = {}
    ready = bool(gate_summary.get("ready_for_gated_execution"))
    go_no_go = str(contract.get("go_no_go") or gate_summary.get("execution_contract") or "no_go") if isinstance(contract, dict) else "no_go"
    blockers: list[str] = []
    for source in (
        selected.get("blockers") if isinstance(selected.get("blockers"), list) else [],
        gate_summary.get("blockers") if isinstance(gate_summary.get("blockers"), list) else [],
        contract.get("root_causes") if isinstance(contract, dict) and isinstance(contract.get("root_causes"), list) else [],
    ):
        blockers.extend(_redacted_memory_text(item, limit=240) for item in source if str(item).strip())
    blockers = list(dict.fromkeys(item for item in blockers if item))

    hypothesis_id = str(selected.get("hypothesis_id") or blueprint.get("hypothesis_id") or selected.get("id") or "unknown_hypothesis")
    strategy_name = str(selected.get("strategy_name") or blueprint.get("strategy_name") or hypothesis_id)
    blueprint_id = str(blueprint.get("blueprint_id") or "no_blueprint")
    branch_type = str(blueprint.get("branch_type") or selected.get("branch_type") or "")
    code_generation_mode = str(blueprint.get("code_generation_mode") or selected.get("code_generation_mode") or "")
    gate_status = "ready_for_gated_execution" if ready and go_no_go != "no_go" and not blockers else "blocked_by_gate"
    outcome = "ready_for_gated_execution" if gate_status == "ready_for_gated_execution" else "blocked_by_gate"

    memory_reuse_plan = (
        blueprint.get("memory_reuse_plan")
        if isinstance(blueprint.get("memory_reuse_plan"), dict)
        else blueprint_payload.get("memory_reuse_plan") if isinstance(blueprint_payload, dict) and isinstance(blueprint_payload.get("memory_reuse_plan"), dict)
        else selected.get("memory_reuse_plan") if isinstance(selected.get("memory_reuse_plan"), dict)
        else review.get("memory_reuse_plan") if isinstance(review, dict) and isinstance(review.get("memory_reuse_plan"), dict)
        else {}
    )
    memory_rule_count = len(memory_reuse_plan.get("reuse_rules") or []) if isinstance(memory_reuse_plan, dict) else 0
    memory_avoid_count = len(memory_reuse_plan.get("avoid_patterns") or []) if isinstance(memory_reuse_plan, dict) else 0

    if outcome == "ready_for_gated_execution":
        lesson = (
            f"Strategy {strategy_name} reached gated-execution readiness for hypothesis={hypothesis_id}; "
            "the next run still requires an explicit user run command, full artifact manifest, score gate, and claim audit."
        )
        next_safe_commands = [
            f"evomind run {session.selected_task}" if session.selected_task else "evomind task add <kaggle-url>",
            "evomind live",
            "evomind report",
        ]
    else:
        reason = "; ".join(blockers[:4]) or f"execution_contract={go_no_go}"
        lesson = (
            f"Strategy {strategy_name} is held before training because {reason}. "
            "Preserve the proposal, repair the blocking gates, and rerun blueprint before spending compute."
        )
        next_safe_commands = ["evomind repair", "evomind contract", "evomind blueprint"]

    trial_id = _stable_memory_id("innovation_trial", session.selected_task or "", hypothesis_id, blueprint_id, gate_status)
    feedback_record = {
        "trial_id": trial_id,
        "generated_at": generated_at,
        "task_id": session.selected_task or "",
        "hypothesis_id": hypothesis_id,
        "strategy_name": strategy_name,
        "branch_type": branch_type,
        "code_generation_mode": code_generation_mode,
        "blueprint_id": blueprint_id,
        "gate_status": gate_status,
        "outcome": outcome,
        "execution_contract": go_no_go,
        "blockers": blockers,
        "lesson": _redacted_memory_text(lesson, limit=700),
        "memory_reuse_rule_count": memory_rule_count,
        "avoid_pattern_count": memory_avoid_count,
        "supporting_memory_ids": list((memory_reuse_plan.get("supporting_memory_ids") or [])[:12]) if isinstance(memory_reuse_plan, dict) else [],
        "source_artifacts": [
            str(review_path),
            str(blueprint_path),
            str(contract_path),
            str(action_queue_path),
            str(loop_path),
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_innovation_trial_feedback",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "trial_feedback": feedback_record,
        "hypothesis_id": hypothesis_id,
        "strategy_name": strategy_name,
        "branch_type": branch_type,
        "code_generation_mode": code_generation_mode,
        "blueprint_id": blueprint_id,
        "gate_status": gate_status,
        "outcome": outcome,
        "lesson": feedback_record["lesson"],
        "memory_reuse_rule_count": memory_rule_count,
        "avoid_pattern_count": memory_avoid_count,
        "source_artifacts": feedback_record["source_artifacts"],
        "next_safe_commands": next_safe_commands,
        "artifact_path": str(artifact_path),
        "innovation_log_path": str(innovation_log_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
        "message": "Innovation trial feedback recorded without starting training.",
    }

    try:
        xsci.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_payload = _read_json_artifact(innovation_log_path) or {}
        proposals = log_payload.get("proposals") if isinstance(log_payload.get("proposals"), list) else []
        tried = log_payload.get("tried") if isinstance(log_payload.get("tried"), list) else []
        updated_tried: list[Any] = []
        replaced = False
        for item in tried:
            if isinstance(item, dict) and str(item.get("trial_id") or "") == trial_id:
                updated_tried.append(feedback_record)
                replaced = True
            else:
                updated_tried.append(item)
        if not replaced:
            updated_tried.append(feedback_record)
        log_payload.update({
            "proposals": proposals,
            "tried": updated_tried[-100:],
            "successes": int(log_payload.get("successes") or 0),
            "failures": int(log_payload.get("failures") or 0),
            "trial_feedback_count": len(updated_tried[-100:]),
            "last_trial_feedback_at": generated_at,
            "last_trial_feedback": feedback_record,
            "updated_at": generated_at,
            "no_training_started": True,
        })
        innovation_log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["innovation_log_updated"] = True
        payload["idempotent_update"] = replaced
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write innovation trial feedback: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"innovation_trial_feedback_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_innovation_trial_feedback",
            "task": session.selected_task or "",
            "phase": "innovation_trial_feedback",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_innovation_trial_feedback",
            "message": f"Recorded innovation feedback outcome={outcome}; gate_status={gate_status}",
            "artifact_path": str(artifact_path),
            "details": {
                "trial_id": trial_id,
                "hypothesis_id": hypothesis_id,
                "blueprint_id": blueprint_id,
                "outcome": outcome,
                "blockers": blockers[:6],
            },
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_innovation_trial_feedback",
            "user": "scientist_innovation_trial_feedback",
            "forced_tools": ["scientist_hypothesis_review", "scientist_experiment_blueprint", "scientist_execution_contract"],
            "executed_tools": [{"tool": "scientist_innovation_trial_feedback", "ok": payload.get("ok", True)}],
            "mode": outcome,
            "decision": {"trial_feedback": feedback_record},
            "blockers": blockers,
            "next_actions": next_safe_commands,
            "artifacts": [str(artifact_path), str(innovation_log_path)],
            "answer_preview": f"innovation feedback recorded; outcome={outcome}; no_training_started=True",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _classify_scientist_blocker(blocker: str) -> dict[str, Any]:
    text = str(blocker or "").strip()
    low = text.lower()
    if not text:
        category = "unknown"
        repair = "evomind recovery"
        severity = "low"
    elif any(token in low for token in ("data", "train", "test", "schema", "remote data")):
        category = "data_contract"
        repair = "evomind contract"
        severity = "high"
    elif any(token in low for token in ("gpu", "hpc", "compute", "ssh")):
        category = "resource_gate"
        repair = "evomind repair"
        severity = "high"
    elif any(token in low for token in ("llm", "api", "setup", "token", "credential")):
        category = "setup_gate"
        repair = "evomind setup"
        severity = "high"
    elif any(token in low for token in ("claim", "rank", "medal", "submit", "kaggle response")):
        category = "claim_gate"
        repair = "evomind contract"
        severity = "medium"
    elif any(token in low for token in ("score", "quality", "improvement", "promotion")):
        category = "quality_gate"
        repair = "evomind review-hypotheses"
        severity = "medium"
    else:
        category = "research_gate"
        repair = "evomind repair"
        severity = "medium"
    return {"blocker": text, "category": category, "severity": severity, "repair_command": repair}


def get_scientist_situation_model(session: SessionState, root: Path) -> dict[str, Any]:
    """Build a high-level AI Scientist situation model.

    This artifact is the bridge from tool outputs to researcher judgment. It
    synthesizes state, evidence, uncertainty, blockers, strategy, and memory so
    the terminal/UI can explain why the next action is chosen. It is read-only:
    no model training, no Kaggle download, and no official submission.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_situation_model.json"

    def safe_call(name: str, fn: Callable[[SessionState, Path], dict[str, Any]]) -> dict[str, Any]:
        try:
            result = fn(session, root)
            return result if isinstance(result, dict) else {"ok": False, "tool": name, "message": "tool returned non-dict"}
        except Exception as exc:  # pragma: no cover - defensive only
            return {"ok": False, "tool": name, "message": f"{type(exc).__name__}: {exc}"}

    system = safe_call("system_status", get_system_status)
    task = safe_call("inspect_task", inspect_selected_task) if session.selected_task else {"ok": False, "tool": "inspect_task", "message": "No task selected."}
    data = safe_call("data_check", inspect_data_availability) if session.selected_task else {"ok": False, "tool": "data_check", "message": "No task selected."}
    recent = safe_call("recent_run", inspect_recent_run)
    evolution = safe_call("evolution_status", inspect_evolution_status)
    checkpoint = safe_call("scientist_checkpoint", get_scientist_checkpoint)
    decision_result = safe_call("research_decision", get_research_decision)

    review = _read_json_artifact(xsci / "scientist_hypothesis_review.json")
    if not isinstance(review, dict):
        review = safe_call("scientist_hypothesis_review", get_scientist_hypothesis_review)
    blueprint_payload = _read_json_artifact(xsci / "scientist_experiment_blueprint.json")
    if not isinstance(blueprint_payload, dict):
        blueprint_payload = safe_call("scientist_experiment_blueprint", get_scientist_experiment_blueprint)
    workplan = _read_json_artifact(xsci / "scientist_workplan.json") or {}
    repair = _read_json_artifact(xsci / "scientist_repair_plan.json") or {}
    contract = _read_json_artifact(xsci / "scientist_execution_contract.json")
    if not isinstance(contract, dict):
        contract = safe_call("scientist_execution_contract", get_scientist_execution_contract)
    self_audit = _read_json_artifact(xsci / "scientist_self_audit.json") or {}

    decision = decision_result.get("decision") if isinstance(decision_result.get("decision"), dict) else {}
    selected_hypothesis = review.get("selected_hypothesis") if isinstance(review.get("selected_hypothesis"), dict) else {}
    experiment_blueprint = blueprint_payload.get("experiment_blueprint") if isinstance(blueprint_payload.get("experiment_blueprint"), dict) else {}
    gate = checkpoint.get("gate") if isinstance(checkpoint.get("gate"), dict) else {}
    tracker = evolution.get("tracker") if isinstance(evolution.get("tracker"), dict) else {}
    memory = evolution.get("retrospective_memory") if isinstance(evolution.get("retrospective_memory"), dict) else {}
    innovation = evolution.get("innovation") if isinstance(evolution.get("innovation"), dict) else {}

    blockers: list[str] = []
    blockers.extend(str(x) for x in (system.get("blockers") or []) if str(x))
    blockers.extend(str(x) for x in (gate.get("blockers") or []) if str(x))
    blockers.extend(str(x) for x in (gate.get("warnings") or []) if "missing" in str(x).lower())
    blockers.extend(str(x) for x in (selected_hypothesis.get("blockers") or []) if str(x))
    blueprint_gate = blueprint_payload.get("gate_summary") if isinstance(blueprint_payload, dict) else {}
    if not isinstance(blueprint_gate, dict):
        blueprint_gate = {}
    blockers.extend(str(x) for x in (blueprint_gate.get("blockers") or []) if str(x))
    contract_root_causes = contract.get("root_causes") if isinstance(contract, dict) else []
    blockers.extend(str(x) for x in (contract_root_causes or []) if str(x))
    blockers = list(dict.fromkeys(blockers))
    blocker_model = [_classify_scientist_blocker(item) for item in blockers]

    data_ready = bool(data.get("train_csv") and data.get("test_csv")) or str(contract.get("data_contract_status") or "").startswith("remote")
    setup_ready = bool(session.selected_task) and bool(session.llm_ready)
    gates_ready = bool(gate.get("can_execute")) and str(contract.get("go_no_go") or "") != "no_go" and not blockers
    hypothesis_ready = bool(selected_hypothesis) and str(selected_hypothesis.get("status") or "") in {"ready_for_gated_execution", "hold_until_gate_clear"}
    blueprint_ready = bool(experiment_blueprint.get("blueprint_id"))
    readiness_checks = {
        "task_selected": bool(session.selected_task),
        "llm_ready": bool(session.llm_ready),
        "kaggle_ready": bool(session.kaggle_ready),
        "data_ready": bool(data_ready),
        "memory_available": bool(memory.get("records", 0) or tracker.get("lessons_recorded", 0)),
        "hypothesis_reviewed": bool(selected_hypothesis),
        "blueprint_available": bool(blueprint_ready),
        "execution_contract_not_no_go": str(contract.get("go_no_go") or "") != "no_go",
        "no_blockers": not blockers,
    }
    readiness_score = round(100 * sum(1 for passed in readiness_checks.values() if passed) / max(1, len(readiness_checks)))

    uncertainties: list[str] = []
    if not session.selected_task:
        uncertainties.append("No selected Kaggle/MLE task; research objective is underspecified.")
    if not data_ready:
        uncertainties.append("Training/test data availability or remote data contract is not proven.")
    if not recent.get("run_id") or recent.get("run_id") == "(none)":
        uncertainties.append("No recent run evidence; local score trend is unknown.")
    if not memory.get("records", 0):
        uncertainties.append("Retrospective memory has no reusable records for cross-run learning.")
    if not selected_hypothesis:
        uncertainties.append("No ranked hypothesis has been selected yet.")
    if not blueprint_ready:
        uncertainties.append("No experiment blueprint exists for resource/artifact/rollback planning.")
    uncertainties.append("Official rank/medal/top30 status is unknown without Kaggle response artifact.")
    uncertainties = list(dict.fromkeys(uncertainties))

    evidence_map = [
        {
            "signal": "system_readiness",
            "status": "ready" if setup_ready else "blocked",
            "evidence": f"llm={session.llm_ready}; kaggle={session.kaggle_ready}; compute={session.compute_backend}",
            "artifact": "",
        },
        {
            "signal": "task_context",
            "status": "ready" if task.get("ok") else "missing",
            "evidence": f"task={session.selected_task or '(none)'}; metric={task.get('metric', '?')}",
            "artifact": str(xsci / "tasks" / f"{session.selected_task}.json") if session.selected_task else "",
        },
        {
            "signal": "data_contract",
            "status": "ready" if data_ready else "blocked",
            "evidence": data.get("message") or contract.get("data_contract_status") or "data contract not proven",
            "artifact": str(contract.get("artifact_path") or ""),
        },
        {
            "signal": "self_evolution_memory",
            "status": "present" if (memory.get("records", 0) or tracker.get("lessons_recorded", 0)) else "empty",
            "evidence": f"memory_records={memory.get('records', 0)}; lessons={tracker.get('lessons_recorded', 0)}; proposals={innovation.get('proposals', 0)}",
            "artifact": str(memory.get("artifact") or ""),
        },
        {
            "signal": "hypothesis_review",
            "status": str(review.get("recommendation") or "not_run"),
            "evidence": selected_hypothesis.get("strategy_name") or selected_hypothesis.get("hypothesis_id") or "no selected hypothesis",
            "artifact": str(review.get("artifact_path") or xsci / "scientist_hypothesis_review.json"),
        },
        {
            "signal": "experiment_blueprint",
            "status": str(blueprint_payload.get("blueprint_status") or "not_run"),
            "evidence": experiment_blueprint.get("blueprint_id") or "no blueprint id",
            "artifact": str(blueprint_payload.get("artifact_path") or xsci / "scientist_experiment_blueprint.json"),
        },
        {
            "signal": "execution_contract",
            "status": str(contract.get("go_no_go") or "not_run"),
            "evidence": f"agent_session_ready={contract.get('agent_session_ready')}; model_training_ready={contract.get('model_training_ready')}",
            "artifact": str(contract.get("artifact_path") or xsci / "scientist_execution_contract.json"),
        },
    ]

    if not session.selected_task:
        recommended_tool_sequence = ["evomind task add <kaggle-url>", "evomind situation"]
        posture = "needs_task"
    elif blockers:
        primary = blocker_model[0]["repair_command"] if blocker_model else "evomind repair"
        recommended_tool_sequence = [primary, "evomind contract", "evomind situation"]
        posture = "blocked_repair_first"
    elif not selected_hypothesis:
        recommended_tool_sequence = ["evomind innovate-plan", "evomind review-hypotheses", "evomind situation"]
        posture = "needs_hypothesis_review"
    elif not blueprint_ready:
        recommended_tool_sequence = ["evomind blueprint", "evomind situation"]
        posture = "needs_blueprint"
    elif gates_ready:
        recommended_tool_sequence = [str(experiment_blueprint.get("run_command") or f"evomind run {session.selected_task}"), "evomind live", "evomind report"]
        posture = "ready_for_gated_execution"
    else:
        recommended_tool_sequence = ["evomind contract", "evomind repair", "evomind situation"]
        posture = "needs_gate_refresh"
    recommended_tool_sequence = list(dict.fromkeys(recommended_tool_sequence))

    situation_model = {
        "research_question": (
            f"How should EvoMind improve {session.selected_task} under metric {task.get('metric', '?')} "
            "while preserving best-so-far, audit evidence, and claim boundaries?"
            if session.selected_task else
            "Which Kaggle/MLE task should EvoMind select before starting an auditable research loop?"
        ),
        "reasoning_mode": "observe_orient_decide_act_with_gates",
        "posture": posture,
        "readiness_score": readiness_score,
        "readiness_checks": readiness_checks,
        "evidence_map": evidence_map,
        "uncertainties": uncertainties,
        "blocker_model": blocker_model,
        "strategy_model": {
            "decision": decision,
            "selected_hypothesis": selected_hypothesis,
            "experiment_blueprint": experiment_blueprint,
            "workplan_focus": workplan.get("current_focus") if isinstance(workplan, dict) else {},
            "repair_focus": repair.get("safe_next_command") if isinstance(repair, dict) else "",
        },
        "self_evolution_model": {
            "skill_level": tracker.get("skill_level") or "unknown",
            "total_runs": tracker.get("total_runs", 0),
            "lessons_recorded": tracker.get("lessons_recorded", 0),
            "memory_records": memory.get("records", 0),
            "innovation_proposals": innovation.get("proposals", 0),
            "blueprint_memory_writeback": experiment_blueprint.get("memory_writeback_plan") if isinstance(experiment_blueprint, dict) else {},
            "self_audit_score": self_audit.get("overall_score") if isinstance(self_audit, dict) else None,
        },
        "recommended_tool_sequence": recommended_tool_sequence,
        "stop_conditions": [
            "Do not start model training while blocker_model is non-empty.",
            "Do not promote a candidate without score_promotion_gate and claim_audit artifacts.",
            "Do not claim official rank, medal, or top30 without Kaggle response artifact.",
        ],
    }

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_situation_model",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "situation_status": posture,
        "situation_model": situation_model,
        "readiness_score": readiness_score,
        "blockers": blockers,
        "next_safe_commands": recommended_tool_sequence,
        "artifact_path": str(artifact_path),
        "source_artifacts": [
            str(xsci / "scientist_hypothesis_review.json"),
            str(xsci / "scientist_experiment_blueprint.json"),
            str(xsci / "scientist_execution_contract.json"),
            str(root / "experiments" / "evolution" / "retrospective_memory.json"),
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "message": f"Situation model built: posture={posture}; readiness_score={readiness_score}.",
    }
    try:
        xsci.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["ok"] = False
        payload["artifact_error"] = str(exc)

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"situation_model_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_situation_model",
            "task": session.selected_task or "",
            "phase": "situation_model",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_situation_model",
            "message": payload["message"],
            "artifact_path": str(artifact_path),
            "details": {"posture": posture, "readiness_score": readiness_score, "blockers": blocker_model[:6]},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_situation_model",
            "user": "scientist_situation_model",
            "forced_tools": ["system_status", "data_check", "evolution_status", "scientist_hypothesis_review", "scientist_experiment_blueprint", "scientist_execution_contract"],
            "executed_tools": [{"tool": "scientist_situation_model", "ok": payload.get("ok", True)}],
            "mode": posture,
            "decision": {"readiness_score": readiness_score, "recommended_tool_sequence": recommended_tool_sequence},
            "blockers": blockers,
            "next_actions": recommended_tool_sequence,
            "artifacts": [str(artifact_path)],
            "answer_preview": f"situation model complete; posture={posture}; no_training_started=True",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _score_capability(name: str, checks: list[tuple[str, int, bool]]) -> dict[str, Any]:
    total = sum(weight for _, weight, _ in checks) or 1
    earned = sum(weight for _, weight, passed in checks if passed)
    score = round(100 * earned / total)
    missing = [label for label, _, passed in checks if not passed]
    if score >= 85:
        status = "strong"
    elif score >= 70:
        status = "usable_needs_polish"
    elif score >= 50:
        status = "weak_needs_upgrade"
    else:
        status = "critical_gap"
    return {
        "name": name,
        "score": score,
        "status": status,
        "passed_checks": [label for label, _, passed in checks if passed],
        "missing_checks": missing,
    }


def get_scientist_self_audit(session: SessionState, root: Path) -> dict[str, Any]:
    """Audit EvoMind's own AI Scientist capability and persist an upgrade backlog.

    This is a read-only meta-agent tool. It inspects existing traces, recovery
    snapshots, action queues, memory, and frontend bridge files. It never starts
    training, never submits to Kaggle, and never reads secret stores.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_path = root / ".xsci" / "scientist_self_audit.json"
    backlog_path = root / ".xsci" / "scientist_upgrade_backlog.json"
    trend_path = root / ".xsci" / "scientist_capability_trend.jsonl"
    xsci = root / ".xsci"

    autopilot = _read_json_artifact(xsci / "scientist_autopilot.json")
    loop = _read_json_artifact(xsci / "scientist_loop.json")
    action_queue = _read_json_artifact(xsci / "scientist_action_queue.json")
    next_action = _read_json_artifact(xsci / "scientist_next_action.json")
    recovery = _read_json_artifact(xsci / "scientist_recovery_snapshot.json")
    workplan = _read_json_artifact(xsci / "scientist_workplan.json")
    repair_plan = _read_json_artifact(xsci / "scientist_repair_plan.json")
    contract = _read_json_artifact(xsci / "scientist_execution_contract.json")
    innovation = _read_json_artifact(xsci / "innovation_log.json")
    hypothesis_review = _read_json_artifact(xsci / "scientist_hypothesis_review.json")
    experiment_blueprint = _read_json_artifact(xsci / "scientist_experiment_blueprint.json")
    reasoning_synthesis = _read_json_artifact(xsci / "scientist_reasoning_synthesis.json")
    reasoning_cache_stats = _read_json_artifact(xsci / "scientist_reasoning_cache_stats_llm.json")
    turns = _read_jsonl_tail(xsci / "scientist_turns.jsonl", limit=80)
    steps = _read_jsonl_tail(xsci / "scientist_step_trace.jsonl", limit=120)
    lessons = _read_jsonl_tail(xsci / "scientist_loop_lessons.jsonl", limit=80)
    parity_loops = _read_jsonl_tail(xsci / "scientist_parity_loop.jsonl", limit=80)

    try:
        evolution = inspect_evolution_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        evolution = {"ok": False, "tool": "evolution_status", "message": type(exc).__name__}
    try:
        system = get_system_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        system = {"ok": False, "tool": "system_status", "message": type(exc).__name__, "blockers": []}

    action_items = action_queue.get("actions") if isinstance(action_queue, dict) else []
    if not isinstance(action_items, list):
        action_items = []
    tool_trace = autopilot.get("tool_trace") if isinstance(autopilot, dict) else []
    if not isinstance(tool_trace, list):
        tool_trace = []
    trace_with_choice_metadata = [
        item for item in tool_trace
        if isinstance(item, dict)
        and isinstance(item.get("rationale"), str)
        and bool(str(item.get("rationale") or "").strip())
        and isinstance(item.get("confidence"), (int, float))
        and 0 < float(item.get("confidence") or 0) <= 1
        and bool(str(item.get("evidence_signal") or "").strip())
    ]
    tool_choice_explanations_ready = len(trace_with_choice_metadata) >= min(5, len(tool_trace)) and len(tool_trace) >= 5
    loop_steps = loop.get("steps") if isinstance(loop, dict) else []
    if not isinstance(loop_steps, list):
        loop_steps = []
    workplan_steps = workplan.get("steps") if isinstance(workplan, dict) else []
    if not isinstance(workplan_steps, list):
        workplan_steps = []
    latest_parity = parity_loops[-1] if parity_loops and isinstance(parity_loops[-1], dict) else {}
    latest_parity_lifecycle = latest_parity.get("lifecycle") if isinstance(latest_parity.get("lifecycle"), dict) else {}
    latest_phase_status = latest_parity.get("phase_status") if isinstance(latest_parity.get("phase_status"), dict) else {}
    required_parity_phases = {"observe", "plan", "act", "reflect", "improve"}
    completed_parity_records = 0
    for item in parity_loops:
        if not isinstance(item, dict):
            continue
        phase_status = item.get("phase_status") if isinstance(item.get("phase_status"), dict) else {}
        lifecycle = item.get("lifecycle") if isinstance(item.get("lifecycle"), dict) else {}
        phases = lifecycle.get("phases") if isinstance(lifecycle.get("phases"), list) else []
        phase_names = {
            str(phase.get("phase") or "")
            for phase in phases
            if isinstance(phase, dict)
        } | {str(key) for key in phase_status.keys()}
        if required_parity_phases.issubset(phase_names):
            completed_parity_records += 1

    memory = evolution.get("retrospective_memory", {}) if isinstance(evolution, dict) else {}
    tracker = evolution.get("tracker", {}) if isinstance(evolution, dict) else {}
    memory_records = int(memory.get("records") or 0) if isinstance(memory, dict) else 0
    reusable_lessons = int(tracker.get("reusable_lessons") or 0) if isinstance(tracker, dict) else 0
    innovation_proposals = len(innovation.get("proposals") or []) if isinstance(innovation, dict) else 0
    innovation_tried = len(innovation.get("tried") or []) if isinstance(innovation, dict) else 0
    raw_system_blockers = system.get("blockers") if isinstance(system.get("blockers"), list) else []
    system_blockers = [_redacted_memory_text(item, limit=500) for item in raw_system_blockers]
    hard_setup_blocked = bool(raw_system_blockers)
    gpu_blocked = bool(system.get("gpu_blocked")) or any(
        "gpu" in str(item).lower() or "ssh" in str(item).lower()
        for item in raw_system_blockers
    )

    api_dir = root / "web" / "research-agent-workstation" / "src" / "app" / "api" / "scientist"
    ui_console = root / "web" / "research-agent-workstation" / "src" / "components" / "workstation" / "AiControlConsole.tsx"
    summary_file = root / "web" / "research-agent-workstation" / "src" / "lib" / "server" / "summary.ts"
    upgrade_plan_route = api_dir / "upgrade-plan" / "route.ts"
    summary_text = ""
    ui_console_text = ""
    try:
        summary_text = summary_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    try:
        ui_console_text = ui_console.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass

    human_gate_action = any(
        isinstance(item, dict)
        and str(item.get("gate") or "").lower().find("human") >= 0
        for item in action_items
    )
    official_blocked = any(
        isinstance(source, dict)
        and str(source.get("official_submit") or "").startswith("blocked")
        for source in (autopilot, loop, action_queue, next_action, recovery, contract)
    )
    no_training_flags = any(
        isinstance(source, dict) and source.get("no_training_started") is True
        for source in (autopilot, loop, action_queue, next_action, recovery, contract)
    )
    contract_has_gate = isinstance(contract, dict) and bool(contract.get("go_no_go") or contract.get("human_gate"))
    contract_go_no_go = str(contract.get("go_no_go") or "").lower() if isinstance(contract, dict) else ""
    setup_gate_enforced = (
        no_training_flags
        and official_blocked
        and contract_has_gate
        and (
            not hard_setup_blocked
            or contract_go_no_go == "no_go"
            or bool(system_blockers)
        )
    )
    runtime_execution_ready = not hard_setup_blocked and not gpu_blocked
    execution_readiness_status = (
        "ready_for_gated_training"
        if runtime_execution_ready
        else "blocked_by_external_resource_or_data_gate"
    )
    decision = autopilot.get("decision") if isinstance(autopilot, dict) else {}
    if not isinstance(decision, dict):
        decision = {}
    reviewed_hypothesis = hypothesis_review.get("selected_hypothesis") if isinstance(hypothesis_review, dict) else {}
    if not isinstance(reviewed_hypothesis, dict):
        reviewed_hypothesis = {}
    blueprint_body = experiment_blueprint.get("experiment_blueprint") if isinstance(experiment_blueprint, dict) else {}
    if not isinstance(blueprint_body, dict):
        blueprint_body = {}
    review_memory_reuse_plan = (
        hypothesis_review.get("memory_reuse_plan")
        if isinstance(hypothesis_review, dict) and isinstance(hypothesis_review.get("memory_reuse_plan"), dict)
        else reviewed_hypothesis.get("memory_reuse_plan") if isinstance(reviewed_hypothesis.get("memory_reuse_plan"), dict)
        else {}
    )
    blueprint_memory_reuse_plan = (
        blueprint_body.get("memory_reuse_plan")
        if isinstance(blueprint_body.get("memory_reuse_plan"), dict)
        else experiment_blueprint.get("memory_reuse_plan") if isinstance(experiment_blueprint, dict) and isinstance(experiment_blueprint.get("memory_reuse_plan"), dict)
        else {}
    )
    active_memory_reuse_plan = (
        review_memory_reuse_plan
        if _memory_reuse_plan_has_content(review_memory_reuse_plan)
        else blueprint_memory_reuse_plan
        if _memory_reuse_plan_has_content(blueprint_memory_reuse_plan)
        else {}
    )
    memory_reuse_ready = _memory_reuse_plan_has_content(active_memory_reuse_plan)
    memory_reuse_rule_count = len(active_memory_reuse_plan.get("reuse_rules") or []) if isinstance(active_memory_reuse_plan, dict) else 0
    memory_reuse_avoid_count = len(active_memory_reuse_plan.get("avoid_patterns") or []) if isinstance(active_memory_reuse_plan, dict) else 0
    reasoning_quality = (
        reasoning_synthesis.get("reasoning_quality")
        if isinstance(reasoning_synthesis, dict) and isinstance(reasoning_synthesis.get("reasoning_quality"), dict)
        else {}
    )
    reasoning_quality_score = int(reasoning_quality.get("score") or 0)
    reasoning_checks = reasoning_quality.get("checks") if isinstance(reasoning_quality.get("checks"), dict) else {}
    reasoning_hypotheses = (
        reasoning_synthesis.get("hypotheses")
        if isinstance(reasoning_synthesis, dict) and isinstance(reasoning_synthesis.get("hypotheses"), list)
        else []
    )
    reasoning_next_action = (
        reasoning_synthesis.get("next_safe_action")
        if isinstance(reasoning_synthesis, dict) and isinstance(reasoning_synthesis.get("next_safe_action"), dict)
        else {}
    )
    reasoning_cache_ratio = (
        float(reasoning_cache_stats.get("hit_ratio") or 0.0)
        if isinstance(reasoning_cache_stats, dict)
        else 0.0
    )
    reasoning_cache_requests = (
        int(reasoning_cache_stats.get("requests") or 0)
        if isinstance(reasoning_cache_stats, dict)
        else 0
    )

    capabilities = [
        _score_capability("context_recovery", [
            ("recovery snapshot artifact exists", 20, isinstance(recovery, dict)),
            ("recovery decision is explicit", 15, isinstance(recovery, dict) and bool(recovery.get("recovery_decision"))),
            ("turn ledger has recent turns", 20, len(turns) > 0),
            ("step trace has recent tool events", 20, len(steps) > 0),
            ("recovery guard file exists", 10, (xsci / "recovery_guard.md").exists()),
            ("resume commands are recorded", 15, isinstance(recovery, dict) and bool(recovery.get("resume_commands"))),
        ]),
        _score_capability("tool_orchestration", [
            ("autopilot called at least five tools", 20, len(tool_trace) >= 5),
            ("tool-choice confidence and rationale are recorded", 15, tool_choice_explanations_ready),
            ("bounded loop recorded multiple steps", 20, len(loop_steps) >= 2),
            ("action queue contains next commands", 15, len(action_items) > 0),
            ("next-action artifact exists", 10, isinstance(next_action, dict)),
            ("terminal exposes a broad tool registry", 20, len(TerminalTools.list_tool_names()) >= 18),
        ]),
        _score_capability("safety_and_claim_gates", [
            ("execution contract exists", 20, contract_has_gate),
            ("no-training flags are present", 20, no_training_flags),
            ("official submit is blocked by default", 20, official_blocked),
            ("setup blockers are surfaced", 20, isinstance(system.get("blockers"), list)),
            ("human-gated action is present", 20, human_gate_action),
        ]),
        _score_capability("self_evolution_memory", [
            ("retrospective memory records exist", 25, memory_records > 0),
            ("retrospective memory has reusable lessons or an active reuse plan", 20, reusable_lessons > 0 or memory_reuse_ready),
            ("loop lessons exist", 15, len(lessons) > 0),
            ("innovation proposals exist", 10, innovation_proposals > 0),
            ("innovation trials have feedback", 15, innovation_tried > 0),
            ("hypothesis review selected a candidate", 15, bool(reviewed_hypothesis.get("strategy_name") or reviewed_hypothesis.get("hypothesis_id"))),
            ("workplan artifact exists", 10, isinstance(workplan, dict)),
            ("repair plan artifact exists", 5, isinstance(repair_plan, dict)),
        ]),
        _score_capability("frontend_observability", [
            ("autopilot API route exists", 12, (api_dir / "autopilot" / "route.ts").exists()),
            ("loop API route exists", 12, (api_dir / "loop" / "route.ts").exists()),
            ("upgrade-plan API route exists", 12, upgrade_plan_route.exists()),
            ("control console exists", 16, ui_console.exists()),
            ("control console exposes scientist_upgrade_plan", 18, "scientist_upgrade_plan" in ui_console_text),
            ("summary loader contains scientist loop", 14, "scientist_loop" in summary_text),
            ("summary loader contains scientist autopilot", 10, "scientist_autopilot" in summary_text),
            ("latest scientist artifacts are present", 6, isinstance(autopilot, dict) or isinstance(loop, dict)),
        ]),
        _score_capability("research_autonomy", [
            ("selected task is known", 15, bool(session.selected_task)),
            ("research decision exists", 15, bool(decision.get("selected_action"))),
            ("reviewed hypothesis is available", 15, bool(reviewed_hypothesis.get("strategy_name") or reviewed_hypothesis.get("hypothesis_id"))),
            ("experiment blueprint is available", 15, bool(blueprint_body.get("blueprint_id"))),
            ("workplan has steps", 20, len(workplan_steps) > 0),
            ("next action or queue is available", 10, isinstance(next_action, dict) or len(action_items) > 0),
            ("execute/readiness mode is explicit", 5, isinstance(autopilot, dict) and bool(autopilot.get("mode"))),
            ("hard setup gates are clear before training", 5, not hard_setup_blocked),
        ]),
        _score_capability("scientific_reasoning_quality", [
            ("reasoning synthesis artifact exists", 10, isinstance(reasoning_synthesis, dict)),
            ("latest turn directly answers the research question", 15, bool(reasoning_synthesis.get("direct_answer")) if isinstance(reasoning_synthesis, dict) else False),
            ("reasoning contract score is strong", 20, reasoning_quality_score >= 85),
            ("requested hypotheses are complete and falsifiable", 20, bool(reasoning_checks.get("hypothesis_count")) and bool(reasoning_checks.get("falsifiability"))),
            ("evidence/risk/cost comparison is present", 10, bool(reasoning_checks.get("comparison"))),
            ("a hypothesis is selected with rationale", 10, bool(reasoning_synthesis.get("selected_hypothesis_id")) and bool(reasoning_synthesis.get("selected_rationale")) if isinstance(reasoning_synthesis, dict) else False),
            ("next action is explicit and gated", 10, bool(reasoning_next_action.get("command")) and bool(reasoning_next_action.get("gate"))),
            ("reasoning cache hit ratio is at least 80 percent", 5, reasoning_cache_requests >= 5 and reasoning_cache_ratio >= 0.8),
        ]),
        _score_capability("capability_gap_management", [
            ("self-audit artifact is generated now", 20, True),
            ("upgrade backlog is generated now", 20, True),
            ("blockers are classified", 20, isinstance(system.get("blockers"), list)),
            ("capability scores are computed", 20, True),
            ("safe next commands are provided", 20, True),
        ]),
        _score_capability("ai_scientist_parity", [
            ("observe: current state has trace and recovery artifacts", 10, len(steps) > 0 and isinstance(recovery, dict)),
            ("plan: turn/workplan artifacts exist", 10, isinstance(workplan, dict) and len(workplan_steps) > 0),
            ("act: safe next action queue exists", 10, isinstance(next_action, dict) or len(action_items) > 0),
            ("reflect: loop lessons and retrospective memory exist", 10, len(lessons) > 0 and memory_records > 0),
            ("improve: active memory reuse and tried innovations exist", 10, memory_reuse_ready and innovation_tried > 0),
            ("parity lifecycle records cover observe-plan-act-reflect-improve", 10, completed_parity_records > 0),
            ("behavioral reasoning contract is strong", 10, reasoning_quality_score >= 85),
            ("surface: frontend exposes self-audit and upgrade-plan controls", 10, "scientist_self_audit" in ui_console_text and "scientist_upgrade_plan" in ui_console_text),
            ("execute: compute/data gates are either clear or hard-blocked with a contract", 20, runtime_execution_ready or setup_gate_enforced),
        ]),
    ]
    overall_score = round(sum(item["score"] for item in capabilities) / len(capabilities))

    gaps: list[dict[str, Any]] = []
    for cap in capabilities:
        severity = (
            "critical" if cap["score"] < 50 else
            "high" if cap["score"] < 70 else
            "medium" if cap["score"] < 85 else
            "low"
        )
        if severity != "low":
            gaps.append({
                "capability": cap["name"],
                "severity": severity,
                "score": cap["score"],
                "missing_checks": cap["missing_checks"][:5],
            })

    backlog: list[dict[str, Any]] = []
    def add_backlog(item_id: str, title: str, priority: str, why: str,
                    command: str, evidence: list[str]) -> None:
        backlog.append({
            "id": item_id,
            "title": title,
            "priority": priority,
            "status": "proposed",
            "why": why,
            "safe_next_command": command,
            "expected_artifacts": evidence,
            "gate": "engineering_review_required",
            "no_training_started": True,
        })

    if any(gap["capability"] == "self_evolution_memory" for gap in gaps):
        if not memory_reuse_ready and reusable_lessons <= 0:
            add_backlog(
                "memory_reuse_before_each_run",
                "Reuse retrospective memory before selecting any new experiment branch",
                "P0",
                "A scientist agent should carry lessons across tasks and rounds before spending compute.",
                "evomind innovate-plan",
                ["experiments/evolution/retrospective_memory.json", ".xsci/scientist_loop_lessons.jsonl", ".xsci/scientist_innovation_backlog.json"],
            )
        if innovation_tried <= 0:
            add_backlog(
                "innovation_trial_feedback_loop",
                "Close the proposal-to-trial feedback loop before claiming self-evolution",
                "P0",
                "The agent has proposals and memory artifacts, but no tried-innovation feedback evidence yet.",
                "evomind innovation-feedback",
                [".xsci/innovation_log.json", ".xsci/scientist_innovation_trial_feedback.json", ".xsci/scientist_experiment_blueprint.json"],
            )
    if any(gap["capability"] == "tool_orchestration" for gap in gaps):
        add_backlog(
            "planner_executor_observer_loop",
            "Strengthen planner-executor-observer loop with anti-stagnation checks",
            "P0",
            "Codex/Claude-Code-like behavior needs visible tool steps, recovery, and bounded continuation.",
            "evomind loop",
            [".xsci/scientist_loop.json", ".xsci/scientist_step_trace.jsonl"],
        )
    if any(gap["capability"] == "frontend_observability" for gap in gaps):
        add_backlog(
            "frontend_self_audit_card",
            "Expose self-audit scores and upgrade backlog on the Control page",
            "P1",
            "The UI should prove that the agent is learning and not just chatting.",
            "evomind dashboard start",
            [".xsci/scientist_self_audit.json"],
        )
    if any(gap["capability"] == "research_autonomy" for gap in gaps):
        add_backlog(
            "task_goal_to_gated_run_contract",
            "Convert natural-language research goals into execution contracts before training",
            "P0",
            "Training should only start after a task-aware decision, rollback condition, and artifact contract exist.",
            "evomind contract",
            [".xsci/scientist_execution_contract.json", ".xsci/scientist_action_queue.json"],
        )
    if any(gap["capability"] == "scientific_reasoning_quality" for gap in gaps):
        add_backlog(
            "evidence_grounded_reasoning_synthesis",
            "Make every Scientist turn directly satisfy the user's scientific deliverables",
            "P0",
            "Tool traces and gates do not prove intelligence unless the turn produces a direct, falsifiable, evidence-grounded answer.",
            "evomind ask \"analyze the current task and propose falsifiable hypotheses\"",
            [
                ".xsci/scientist_reasoning_synthesis.json",
                ".xsci/scientist_reasoning_synthesis.md",
                ".xsci/scientist_reasoning_history.jsonl",
            ],
        )
    if hard_setup_blocked or gpu_blocked:
        add_backlog(
            "resource_gate_truthfulness",
            "Keep training blocked until compute/data gates have fresh proof",
            "P0",
            "A Codex/Claude-Code-like scientist must not execute long training when GPU or setup gates are blocked.",
            "evomind ready",
            [".xsci/scientist_repair_plan.json", "docs/verified_workstation_launch_audit.json", "workspace/gpu"],
        )
    if any(gap["capability"] == "ai_scientist_parity" for gap in gaps):
        add_backlog(
            "codex_claude_parity_loop",
            "Implement a stricter observe-plan-act-reflect-improve parity loop",
            "P0",
            "The agent should keep improving from real traces, tried experiments, and gate outcomes instead of passing on artifact existence.",
            "evomind loop",
            [".xsci/scientist_turns.jsonl", ".xsci/scientist_step_trace.jsonl", ".xsci/scientist_parity_loop.jsonl", ".xsci/scientist_upgrade_plan.json"],
        )
    if not tool_choice_explanations_ready:
        add_backlog(
            "streaming_tool_confidence",
            "Add per-turn confidence and tool-choice rationale to terminal streaming",
            "P1",
            "A strong research agent should explain why each tool is selected and what evidence changed.",
            "evomind autopilot",
            [".xsci/terminal_events.jsonl", ".xsci/scientist_turns.jsonl"],
        )

    evidence_sources = {
        "autopilot": {
            "path": str(xsci / "scientist_autopilot.json"),
            "present": isinstance(autopilot, dict),
            "tool_calls": len(tool_trace),
            "tool_choice_explanations": len(trace_with_choice_metadata),
            "tool_choice_explanations_ready": tool_choice_explanations_ready,
        },
        "loop": {"path": str(xsci / "scientist_loop.json"), "present": isinstance(loop, dict), "steps": len(loop_steps), "lessons": len(lessons)},
        "action_queue": {"path": str(xsci / "scientist_action_queue.json"), "present": isinstance(action_queue, dict), "actions": len(action_items)},
        "recovery": {"path": str(xsci / "scientist_recovery_snapshot.json"), "present": isinstance(recovery, dict), "turns": len(turns), "step_events": len(steps)},
        "memory": {
            "path": str(root / "experiments" / "evolution" / "retrospective_memory.json"),
            "records": memory_records,
            "reusable_lessons": reusable_lessons,
            "active_reuse_plan": memory_reuse_ready,
            "reuse_rules": memory_reuse_rule_count,
            "avoid_patterns": memory_reuse_avoid_count,
            "reuse_plan_gate": active_memory_reuse_plan.get("gate") if isinstance(active_memory_reuse_plan, dict) else "",
        },
        "innovation": {
            "path": str(xsci / "innovation_log.json"),
            "feedback_path": str(xsci / "scientist_innovation_trial_feedback.json"),
            "proposals": innovation_proposals,
            "tried": innovation_tried,
            "last_trial_feedback": bool(isinstance(innovation, dict) and innovation.get("last_trial_feedback")),
        },
        "frontend": {
            "control_console": str(ui_console),
            "api_dir": str(api_dir),
            "upgrade_plan_route": str(upgrade_plan_route),
            "summary_loader": str(summary_file),
        },
        "parity": {
            "reusable_lessons": reusable_lessons,
            "innovation_tried": innovation_tried,
            "parity_loop_records": len(parity_loops),
            "completed_parity_records": completed_parity_records,
            "latest_phase_status": latest_phase_status,
            "latest_lifecycle_schema": latest_parity_lifecycle.get("schema") if isinstance(latest_parity_lifecycle, dict) else "",
            "hard_setup_blocked": hard_setup_blocked,
            "gpu_blocked": gpu_blocked,
            "setup_gate_enforced": setup_gate_enforced,
            "runtime_execution_ready": runtime_execution_ready,
            "active_memory_reuse_plan": memory_reuse_ready,
            "upgrade_plan_route_present": upgrade_plan_route.exists(),
            "upgrade_plan_ui_present": "scientist_upgrade_plan" in ui_console_text,
        },
        "reasoning_synthesis": {
            "path": str(xsci / "scientist_reasoning_synthesis.json"),
            "markdown_path": str(xsci / "scientist_reasoning_synthesis.md"),
            "present": isinstance(reasoning_synthesis, dict),
            "quality_score": reasoning_quality_score,
            "quality_status": reasoning_quality.get("status") if isinstance(reasoning_quality, dict) else "",
            "hypotheses": len(reasoning_hypotheses),
            "selected_hypothesis_id": reasoning_synthesis.get("selected_hypothesis_id") if isinstance(reasoning_synthesis, dict) else "",
            "next_safe_command": reasoning_next_action.get("command") if isinstance(reasoning_next_action, dict) else "",
            "cache_stats_path": str(xsci / "scientist_reasoning_cache_stats_llm.json"),
            "cache_requests": reasoning_cache_requests,
            "cache_hit_ratio": reasoning_cache_ratio,
        },
    }
    next_safe_commands = [
        "evomind self-audit",
        "evomind innovate-plan",
        "evomind innovation-feedback",
        "evomind recovery",
        "evomind loop",
        "evomind autopilot",
    ]
    capability_readiness = (
        "strong_local_agent_ready" if overall_score >= 85 else
        "usable_but_needs_agent_upgrades" if overall_score >= 70 else
        "not_ready_for_strong_ai_scientist_demo"
    )
    launch_readiness = (
        "strong_local_agent_ready"
        if overall_score >= 85 and runtime_execution_ready
        else "capability_ready_but_execution_blocked"
        if overall_score >= 85
        else "usable_but_needs_agent_upgrades"
        if overall_score >= 70
        else "not_ready_for_strong_ai_scientist_demo"
    )
    claim_readiness = {
        "capability_claim": (
            "strong_local_agent_capability_supported"
            if overall_score >= 85
            else "partial_capability_supported"
            if overall_score >= 70
            else "insufficient_capability_evidence"
        ),
        "training_readiness_claim": (
            "ready_for_gated_training"
            if runtime_execution_ready
            else "blocked_by_external_resource_or_data_gate"
        ),
        "ai_scientist_parity_claim": (
            "blocked_without_end_to_end_training_and_recovery_evidence"
            if not runtime_execution_ready
            else "local_agent_parity_proxy_only_requires_more_tasks"
        ),
        "rank_or_medal_claim": "blocked_without_kaggle_response_artifact",
        "official_submit_claim": "blocked_until_explicit_human_approval",
        "reason": (
            "Capability score is separated from execution readiness; blocked compute/data gates prevent broad launch or benchmark claims."
            if not runtime_execution_ready
            else "Execution gate is currently clear, but official submit/rank/medal claims still require human approval and Kaggle response artifacts."
        ),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    trend_history_before = _read_jsonl_tail(trend_path, limit=20)
    previous_trend = next((item for item in reversed(trend_history_before) if isinstance(item, dict)), {})
    previous_score = previous_trend.get("overall_score") if isinstance(previous_trend, dict) else None
    score_delta = (
        int(overall_score) - int(previous_score)
        if isinstance(previous_score, int)
        else None
    )
    capability_score_map = {
        str(item.get("name") or "unknown"): int(item.get("score") or 0)
        for item in capabilities
        if isinstance(item, dict)
    }
    previous_capabilities = previous_trend.get("capability_scores") if isinstance(previous_trend, dict) and isinstance(previous_trend.get("capability_scores"), dict) else {}
    capability_deltas = {
        name: score - int(previous_capabilities.get(name, score))
        for name, score in capability_score_map.items()
        if isinstance(previous_capabilities, dict)
    }
    trend_entry = {
        "schema": "evomind.ai_scientist.capability_trend.v1",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "overall_score": overall_score,
        "previous_score": previous_score if isinstance(previous_score, int) else None,
        "score_delta": score_delta,
        "capability_readiness": capability_readiness,
        "launch_readiness": launch_readiness,
        "claim_readiness": claim_readiness,
        "capability_scores": capability_score_map,
        "capability_deltas": capability_deltas,
        "gap_count": len(gaps),
        "backlog_count": len(backlog),
        "top_gap_capabilities": [str(item.get("capability") or "") for item in gaps[:5] if isinstance(item, dict)],
        "evidence_snapshot": {
            "memory_records": memory_records,
            "loop_lessons": len(lessons),
            "step_events": len(steps),
            "tool_trace_events": len(tool_trace),
            "parity_loop_records": len(parity_loops),
            "completed_parity_records": completed_parity_records,
            "continuation_resume_present": (xsci / "scientist_continuation_resume.json").exists(),
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    recent_trend = [item for item in trend_history_before[-4:] if isinstance(item, dict)] + [trend_entry]
    capability_trend = {
        "path": str(trend_path),
        "records_before": len([item for item in trend_history_before if isinstance(item, dict)]),
        "records_after": len([item for item in trend_history_before if isinstance(item, dict)]) + 1,
        "previous_score": previous_score if isinstance(previous_score, int) else None,
        "current_score": overall_score,
        "score_delta": score_delta,
        "latest_capability_readiness": capability_readiness,
        "latest_readiness": launch_readiness,
        "recent": recent_trend[-5:],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_self_audit",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "last_goal": session.last_goal or "",
        "overall_score": overall_score,
        "capability_readiness": capability_readiness,
        "launch_readiness": launch_readiness,
        "claim_readiness": claim_readiness,
        "capabilities": capabilities,
        "gaps": gaps,
        "upgrade_backlog": backlog,
        "capability_trend": capability_trend,
        "evidence_sources": evidence_sources,
        "execution_readiness": {
            "status": execution_readiness_status,
            "runtime_execution_ready": runtime_execution_ready,
            "gate_enforced": setup_gate_enforced,
            "hard_setup_blocked": hard_setup_blocked,
            "gpu_blocked": gpu_blocked,
            "blockers": system_blockers[:5],
            "message": (
                "Training can proceed only through the gated AgentSession/workstation path."
                if runtime_execution_ready
                else "Training remains blocked by external resource or data gates; this is a truthful gate, not a silent failure."
            ),
        },
        "system_blockers": system_blockers,
        "next_safe_commands": next_safe_commands,
        "artifact_path": str(artifact_path),
        "backlog_artifact_path": str(backlog_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        backlog_path.write_text(json.dumps({
            "generated_at": generated_at,
            "tool": "scientist_upgrade_backlog",
            "source": "scientist_self_audit",
            "overall_score": overall_score,
            "items": backlog,
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        with trend_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trend_entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write self-audit artifact: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"self_audit_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_self_audit",
            "task": session.selected_task or "",
            "phase": "capability_audit",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_self_audit",
            "message": f"Self-audit score={overall_score}; gaps={len(gaps)}; backlog={len(backlog)}",
            "artifact_path": str(artifact_path),
            "details": {"overall_score": overall_score, "launch_readiness": launch_readiness},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_self_audit",
            "user": "scientist_self_audit",
            "forced_tools": ["system_status", "evolution_status", "artifact_inventory"],
            "executed_tools": [{"tool": "scientist_self_audit", "ok": payload.get("ok", True)}],
            "mode": launch_readiness,
            "decision": {"overall_score": overall_score, "gaps": len(gaps)},
            "blockers": payload.get("system_blockers", []),
            "next_actions": next_safe_commands,
            "artifacts": [str(artifact_path), str(backlog_path)],
            "answer_preview": f"self-audit score={overall_score}; readiness={launch_readiness}",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _readiness_gate_status(name: str, ok: bool, evidence: str, action: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "ready" if ok else "blocked",
        "ok": bool(ok),
        "evidence": _redacted_memory_text(evidence, limit=320),
        "next_action": action,
    }


def _write_scientist_readiness_markdown(payload: dict[str, Any], path: Path) -> None:
    """Write a compact teacher/demo friendly readiness report."""
    lines: list[str] = [
        "# EvoMind AI Scientist Readiness Report",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- selected_task: {payload.get('selected_task') or '(none)'}",
        f"- overall_score: {payload.get('overall_score')}",
        f"- capability_readiness: {payload.get('capability_readiness')}",
        f"- launch_readiness: {payload.get('launch_readiness')}",
        f"- no_training_started: {payload.get('no_training_started', True)}",
        f"- official_submit: {payload.get('official_submit')}",
        "",
        "## Claim Readiness",
    ]
    claim = payload.get("claim_readiness") if isinstance(payload.get("claim_readiness"), dict) else {}
    for key in (
        "capability_claim",
        "training_readiness_claim",
        "ai_scientist_parity_claim",
        "rank_or_medal_claim",
        "official_submit_claim",
    ):
        lines.append(f"- {key}: {claim.get(key, 'unknown')}")
    if claim.get("reason"):
        lines.append(f"- reason: {_redacted_memory_text(claim.get('reason'), limit=500)}")

    lines.extend(["", "## Gate Matrix"])
    for item in payload.get("readiness_matrix") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- "
            f"{item.get('name')}: {item.get('status')} | "
            f"evidence={item.get('evidence')} | "
            f"next={item.get('next_action')}"
        )

    lines.extend(["", "## Recommended Next Commands"])
    for command in payload.get("recommended_next_commands") or []:
        lines.append(f"- `{command}`")

    blockers = payload.get("blocking_reasons") if isinstance(payload.get("blocking_reasons"), list) else []
    lines.extend(["", "## Blocking Reasons"])
    if blockers:
        for item in blockers:
            lines.append(f"- {_redacted_memory_text(item, limit=500)}")
    else:
        lines.append("- none")

    lines.extend(["", "## Artifact Evidence"])
    for item in payload.get("artifact_evidence") or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('name')}: {item.get('path')} ({item.get('present')})")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_scientist_readiness_report(session: SessionState, root: Path) -> dict[str, Any]:
    """Create one unified AI Scientist readiness report.

    The report intentionally separates local agent capability from execution,
    official submission, rank, and medal claims. It refreshes read-only audit
    artifacts only; it never trains models, downloads data, or submits Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_readiness_report.json"
    markdown_path = xsci / "scientist_readiness_report.md"

    self_audit = get_scientist_self_audit(session, root)
    continuation_status = get_scientist_continuation_status(session, root)
    try:
        system = get_system_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        system = {"ok": False, "tool": "system_status", "message": type(exc).__name__, "blockers": []}
    try:
        evolution = inspect_evolution_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        evolution = {"ok": False, "tool": "evolution_status", "message": type(exc).__name__}
    try:
        kaggle = inspect_kaggle_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        kaggle = {"ok": False, "tool": "kaggle_status", "message": type(exc).__name__}
    try:
        gpu = inspect_gpu_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        gpu = {"ok": False, "tool": "gpu_status", "message": type(exc).__name__, "blocked": True}

    claim_readiness = self_audit.get("claim_readiness") if isinstance(self_audit.get("claim_readiness"), dict) else {}
    execution_readiness = self_audit.get("execution_readiness") if isinstance(self_audit.get("execution_readiness"), dict) else {}
    memory = evolution.get("retrospective_memory", {}) if isinstance(evolution, dict) else {}
    memory_records = int(memory.get("records") or 0) if isinstance(memory, dict) else 0
    blockers = [
        _redacted_memory_text(item, limit=500)
        for item in (system.get("blockers") if isinstance(system.get("blockers"), list) else [])
        if _redacted_memory_text(item, limit=500)
    ]
    gpu_blocked = bool(gpu.get("blocked")) or bool(system.get("gpu_blocked"))
    runtime_ready = bool(execution_readiness.get("runtime_execution_ready"))
    llm_ready = bool(system.get("llm_ready"))
    kaggle_ready = bool(kaggle.get("ready") or system.get("kaggle_ready"))
    continuation_closed = str(continuation_status.get("status") or "") in {"closed", "no_continuation"}

    readiness_matrix = [
        _readiness_gate_status(
            "llm_model",
            llm_ready,
            f"provider={system.get('llm_provider') or 'unset'}",
            "evomind setup",
        ),
        _readiness_gate_status(
            "kaggle_api",
            kaggle_ready,
            str(kaggle.get("message") or ("configured" if kaggle_ready else "not configured")),
            "evomind setup",
        ),
        _readiness_gate_status(
            "compute_resource_gate",
            runtime_ready and not gpu_blocked,
            str(gpu.get("suggestion") or gpu.get("blocker") or system.get("gpu_status") or "unknown"),
            "evomind ready",
        ),
        _readiness_gate_status(
            "self_evolution_memory",
            memory_records > 0,
            f"retrospective_memory_records={memory_records}",
            "evomind memory-consolidate",
        ),
        _readiness_gate_status(
            "continuation_closed",
            continuation_closed,
            f"status={continuation_status.get('status')}; remaining={continuation_status.get('remaining_count', 0)}",
            "evomind resume-continuation",
        ),
        _readiness_gate_status(
            "rank_medal_claim_gate",
            False,
            "Kaggle rank/medal claims require official response artifacts.",
            "wait_for_kaggle_response_artifact",
        ),
        _readiness_gate_status(
            "official_submit_gate",
            False,
            "Official submit is blocked until explicit human approval.",
            "human_approval_required",
        ),
    ]

    recommended_next_commands = []
    if not llm_ready:
        recommended_next_commands.append("evomind setup")
    if not continuation_closed:
        recommended_next_commands.append("evomind resume-continuation")
    if not memory_records:
        recommended_next_commands.append("evomind memory-consolidate")
    recommended_next_commands.extend([
        "evomind self-audit",
        "evomind upgrade-plan",
        "evomind loop",
    ])
    recommended_next_commands = list(dict.fromkeys(recommended_next_commands))[:8]

    artifact_evidence = [
        {"name": "self_audit", "path": self_audit.get("artifact_path") or str(xsci / "scientist_self_audit.json"), "present": (xsci / "scientist_self_audit.json").exists()},
        {"name": "upgrade_backlog", "path": self_audit.get("backlog_artifact_path") or str(xsci / "scientist_upgrade_backlog.json"), "present": (xsci / "scientist_upgrade_backlog.json").exists()},
        {"name": "capability_trend", "path": (self_audit.get("capability_trend") or {}).get("path") if isinstance(self_audit.get("capability_trend"), dict) else str(xsci / "scientist_capability_trend.jsonl"), "present": (xsci / "scientist_capability_trend.jsonl").exists()},
        {"name": "continuation_status", "path": continuation_status.get("artifact_path") or str(xsci / "scientist_continuation_status.json"), "present": (xsci / "scientist_continuation_status.json").exists()},
        {"name": "readiness_report", "path": str(artifact_path), "present": True},
        {"name": "readiness_markdown", "path": str(markdown_path), "present": True},
    ]

    payload = {
        "ok": True,
        "tool": "scientist_readiness_report",
        "schema": "evomind.ai_scientist.readiness_report.v1",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "overall_score": self_audit.get("overall_score", 0),
        "capability_readiness": self_audit.get("capability_readiness") or "unknown",
        "launch_readiness": self_audit.get("launch_readiness") or "unknown",
        "claim_readiness": claim_readiness,
        "execution_readiness": execution_readiness,
        "readiness_matrix": readiness_matrix,
        "blocking_reasons": blockers[:8],
        "recommended_next_commands": recommended_next_commands,
        "artifact_evidence": artifact_evidence,
        "source_artifacts": {
            "self_audit": self_audit.get("artifact_path") or str(xsci / "scientist_self_audit.json"),
            "continuation_status": continuation_status.get("artifact_path") or str(xsci / "scientist_continuation_status.json"),
            "capability_trend": (self_audit.get("capability_trend") or {}).get("path") if isinstance(self_audit.get("capability_trend"), dict) else str(xsci / "scientist_capability_trend.jsonl"),
            "retrospective_memory": str(root / "experiments" / "evolution" / "retrospective_memory.json"),
        },
        "source_summaries": {
            "self_audit": {
                "ok": self_audit.get("ok", True),
                "gaps": len(self_audit.get("gaps") or []),
                "backlog": len(self_audit.get("upgrade_backlog") or []),
            },
            "continuation_status": {
                "status": continuation_status.get("status"),
                "remaining_count": continuation_status.get("remaining_count", 0),
                "completion_ratio": continuation_status.get("completion_ratio", 0),
            },
            "evolution": {
                "memory_records": memory_records,
                "reusable_lessons": (evolution.get("tracker") or {}).get("reusable_lessons") if isinstance(evolution.get("tracker"), dict) else 0,
            },
            "system": {
                "llm_ready": llm_ready,
                "kaggle_ready": kaggle_ready,
                "gpu_blocked": gpu_blocked,
                "runtime_execution_ready": runtime_ready,
            },
        },
        "artifact_path": str(artifact_path),
        "markdown_artifact_path": str(markdown_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_scientist_readiness_markdown(payload, markdown_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write readiness report artifact: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"readiness_report_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_readiness_report",
            "task": session.selected_task or "",
            "phase": "readiness_report",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_readiness_report",
            "message": f"launch={payload.get('launch_readiness')}; training={claim_readiness.get('training_readiness_claim')}",
            "artifact_path": str(artifact_path),
            "details": {"readiness_matrix": readiness_matrix[:5]},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_readiness_report",
            "user": "scientist_readiness_report",
            "forced_tools": ["scientist_self_audit", "scientist_continuation_status", "system_status", "evolution_status"],
            "executed_tools": [{"tool": "scientist_readiness_report", "ok": payload.get("ok", True)}],
            "mode": payload.get("launch_readiness"),
            "decision": {"training_readiness_claim": claim_readiness.get("training_readiness_claim")},
            "blockers": blockers[:8],
            "next_actions": recommended_next_commands,
            "artifacts": [str(artifact_path), str(markdown_path)],
            "answer_preview": f"readiness report: launch={payload.get('launch_readiness')}; score={payload.get('overall_score')}",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _write_scientist_causal_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# EvoMind AI Scientist Causal Diagnosis",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- selected_task: {payload.get('selected_task') or '(none)'}",
        f"- posture: {payload.get('posture')}",
        f"- no_training_started: {payload.get('no_training_started', True)}",
        f"- official_submit: {payload.get('official_submit')}",
        "",
        "## Symptoms",
    ]
    for item in payload.get("symptoms") or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('id')}: {item.get('summary')} ({item.get('severity')})")

    lines.extend(["", "## Root Causes"])
    for item in payload.get("root_causes") or []:
        if isinstance(item, dict):
            lines.append(
                f"- {item.get('id')}: {item.get('summary')} | "
                f"confidence={item.get('confidence')} | gate={item.get('gate')}"
            )

    lines.extend(["", "## Interventions"])
    for item in payload.get("interventions") or []:
        if isinstance(item, dict):
            lines.append(
                f"- {item.get('id')}: {item.get('title')} | "
                f"command=`{item.get('safe_next_command')}` | gate={item.get('gate')}"
            )

    lines.extend(["", "## Causal Edges"])
    graph = payload.get("causal_graph") if isinstance(payload.get("causal_graph"), dict) else {}
    for edge in graph.get("edges") or []:
        if isinstance(edge, dict):
            lines.append(f"- {edge.get('from')} -> {edge.get('to')} ({edge.get('relation')})")

    lines.extend(["", "## Evidence"])
    for item in payload.get("evidence_refs") or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('name')}: {item.get('path')} ({item.get('present')})")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_scientist_causal_diagnosis(session: SessionState, root: Path) -> dict[str, Any]:
    """Build a causal diagnosis graph for the current AI Scientist state.

    This is the "why" layer: symptoms are linked to root causes, evidence, and
    interventions. It never starts training or official Kaggle submission.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_causal_diagnosis.json"
    markdown_path = xsci / "scientist_causal_diagnosis.md"

    readiness = _read_json_artifact(xsci / "scientist_readiness_report.json") or {}
    if not readiness:
        readiness = get_scientist_readiness_report(session, root)
    self_audit = _read_json_artifact(xsci / "scientist_self_audit.json") or {}
    situation = _read_json_artifact(xsci / "scientist_situation_model.json") or {}
    repair = _read_json_artifact(xsci / "scientist_repair_plan.json") or {}
    contract = _read_json_artifact(xsci / "scientist_execution_contract.json") or {}
    queue = _read_json_artifact(xsci / "scientist_action_queue.json") or {}
    review = _read_json_artifact(xsci / "scientist_hypothesis_review.json") or {}
    blueprint = _read_json_artifact(xsci / "scientist_experiment_blueprint.json") or {}
    continuation = _read_json_artifact(xsci / "scientist_continuation_status.json") or {}

    try:
        data_status = inspect_data_availability(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        data_status = {"ok": False, "tool": "data_check", "message": type(exc).__name__}
    try:
        system = get_system_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        system = {"ok": False, "tool": "system_status", "message": type(exc).__name__, "blockers": []}
    try:
        evolution = inspect_evolution_status(session, root)
    except Exception as exc:  # pragma: no cover - defensive only
        evolution = {"ok": False, "tool": "evolution_status", "message": type(exc).__name__}

    claim = readiness.get("claim_readiness") if isinstance(readiness.get("claim_readiness"), dict) else {}
    execution = readiness.get("execution_readiness") if isinstance(readiness.get("execution_readiness"), dict) else {}
    selected_hypothesis = review.get("selected_hypothesis") if isinstance(review.get("selected_hypothesis"), dict) else {}
    blueprint_body = blueprint.get("experiment_blueprint") if isinstance(blueprint.get("experiment_blueprint"), dict) else {}
    action_items = queue.get("actions") if isinstance(queue.get("actions"), list) else []
    memory = evolution.get("retrospective_memory", {}) if isinstance(evolution, dict) else {}
    memory_records = int(memory.get("records") or 0) if isinstance(memory, dict) else 0

    symptoms: list[dict[str, Any]] = []
    root_causes: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, kind: str, status: str, summary: str, evidence: list[str] | None = None) -> None:
        if any(item.get("id") == node_id for item in nodes):
            return
        nodes.append({
            "id": node_id,
            "kind": kind,
            "status": status,
            "summary": _redacted_memory_text(summary, limit=360),
            "evidence": evidence or [],
        })

    def add_edge(source: str, target: str, relation: str) -> None:
        edge = {"from": source, "to": target, "relation": relation}
        if edge not in edges:
            edges.append(edge)

    def add_symptom(item_id: str, summary: str, severity: str, evidence: list[str]) -> None:
        symptoms.append({
            "id": item_id,
            "summary": _redacted_memory_text(summary, limit=360),
            "severity": severity,
            "evidence": evidence,
        })
        add_node(item_id, "symptom", severity, summary, evidence)

    def add_cause(item_id: str, summary: str, confidence: float, gate: str, evidence: list[str]) -> None:
        root_causes.append({
            "id": item_id,
            "summary": _redacted_memory_text(summary, limit=420),
            "confidence": round(float(confidence), 2),
            "gate": gate,
            "evidence": evidence,
        })
        add_node(item_id, "root_cause", gate, summary, evidence)

    def add_intervention(item_id: str, title: str, command: str, gate: str, expected_artifacts: list[str], addresses: list[str]) -> None:
        interventions.append({
            "id": item_id,
            "title": title,
            "safe_next_command": command,
            "gate": gate,
            "expected_artifacts": expected_artifacts,
            "addresses": addresses,
            "no_training_started": True,
        })
        add_node(item_id, "intervention", gate, title, expected_artifacts)
        for cause_id in addresses:
            add_edge(cause_id, item_id, "addressed_by")

    runtime_ready = bool(execution.get("runtime_execution_ready"))
    gpu_blocked = bool(system.get("gpu_blocked")) or str(claim.get("training_readiness_claim")) == "blocked_by_external_resource_or_data_gate"
    continuation_open = str(continuation.get("status") or "") not in {"", "closed", "no_continuation"}
    data_missing = data_status.get("ok") is False or (
        bool(session.selected_task)
        and not (data_status.get("train_csv") and data_status.get("test_csv") and data_status.get("sample_submission"))
    )
    no_hypothesis = not bool(selected_hypothesis)
    no_blueprint = not bool(blueprint_body)
    no_queue = len(action_items) == 0

    if not runtime_ready:
        add_symptom(
            "training_execution_blocked",
            "Training cannot honestly start because the runtime execution gate is not ready.",
            "high",
            [str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json")],
        )
    if gpu_blocked:
        add_cause(
            "gpu_or_external_resource_gate_blocked",
            "The compute/HPC gate is blocked or lacks fresh proof, so training must remain behind the workstation gate.",
            0.9,
            "resource_gate",
            [str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json")],
        )
        add_edge("gpu_or_external_resource_gate_blocked", "training_execution_blocked", "causes")
    if data_missing:
        add_symptom(
            "task_data_contract_incomplete",
            str(data_status.get("message") or "Selected task data contract is incomplete."),
            "high" if session.selected_task else "medium",
            [str(xsci / "scientist_execution_contract.json")],
        )
        add_cause(
            "data_or_task_contract_not_ready",
            "The selected task has missing or unverified train/test/sample submission evidence.",
            0.75,
            "data_contract_gate",
            [str(xsci / "scientist_execution_contract.json")],
        )
        add_edge("data_or_task_contract_not_ready", "task_data_contract_incomplete", "causes")
    if no_hypothesis:
        add_symptom(
            "no_ranked_research_hypothesis",
            "No reviewed hypothesis is selected, so the agent cannot choose a high-quality experiment branch yet.",
            "medium",
            [str(xsci / "scientist_hypothesis_review.json")],
        )
        add_cause(
            "hypothesis_review_missing_or_stale",
            "The innovation backlog has not been converted into a ranked, evidence-scored hypothesis.",
            0.7,
            "hypothesis_review_gate",
            [str(xsci / "scientist_innovation_backlog.json"), str(xsci / "scientist_hypothesis_review.json")],
        )
        add_edge("hypothesis_review_missing_or_stale", "no_ranked_research_hypothesis", "causes")
    if no_blueprint:
        add_symptom(
            "no_executable_experiment_blueprint",
            "No gated experiment blueprint is available to map a hypothesis into branch, code mode, resource mode, artifacts, and rollback.",
            "medium",
            [str(xsci / "scientist_experiment_blueprint.json")],
        )
        add_cause(
            "experiment_blueprint_missing",
            "The selected or future hypothesis has not been translated into an auditable execution plan.",
            0.7,
            "blueprint_gate",
            [str(xsci / "scientist_hypothesis_review.json"), str(xsci / "scientist_experiment_blueprint.json")],
        )
        add_edge("experiment_blueprint_missing", "no_executable_experiment_blueprint", "causes")
    if continuation_open:
        add_symptom(
            "previous_scientist_turn_incomplete",
            f"Previous Scientist turn is still {continuation.get('status')} with {continuation.get('remaining_count', 0)} remaining safe tools.",
            "medium",
            [str(continuation.get("artifact_path") or xsci / "scientist_continuation_status.json")],
        )
        add_cause(
            "deferred_safe_tools_not_closed",
            "The agent still has safe read-only tools to run before spending compute or declaring readiness.",
            0.8,
            "continuation_gate",
            [str(xsci / "scientist_continuation.json"), str(xsci / "scientist_continuation_status.json")],
        )
        add_edge("deferred_safe_tools_not_closed", "previous_scientist_turn_incomplete", "causes")
    if memory_records <= 0:
        add_symptom(
            "weak_retrospective_memory",
            "No retrospective memory records are available, so branch choice cannot reuse prior lessons.",
            "medium",
            [str(root / "experiments" / "evolution" / "retrospective_memory.json")],
        )
        add_cause(
            "memory_writeback_missing",
            "The learn/reflect layer has not accumulated reusable evidence yet.",
            0.65,
            "memory_gate",
            [str(root / "experiments" / "evolution" / "retrospective_memory.json")],
        )
        add_edge("memory_writeback_missing", "weak_retrospective_memory", "causes")
    if str(claim.get("rank_or_medal_claim")) == "blocked_without_kaggle_response_artifact":
        add_symptom(
            "rank_or_medal_claim_blocked",
            "Rank, medal, and top-percentile claims are blocked without official Kaggle response artifacts.",
            "high",
            [str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json")],
        )
        add_cause(
            "official_leaderboard_evidence_missing",
            "No official Kaggle response artifact proves public score, rank, team count, medal, or percentile.",
            0.95,
            "claim_audit_gate",
            [str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json")],
        )
        add_edge("official_leaderboard_evidence_missing", "rank_or_medal_claim_blocked", "causes")

    if not symptoms:
        add_symptom(
            "no_critical_symptom_detected",
            "No critical blocker was detected from current artifacts; proceed through the audited action queue only.",
            "low",
            [str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json")],
        )

    if continuation_open:
        add_intervention(
            "close_deferred_safe_tools",
            "Close the previous Scientist turn by running remaining safe tools.",
            "evomind resume-continuation",
            "read_only_continuation_gate",
            [str(xsci / "scientist_continuation_resume.json"), str(xsci / "scientist_continuation_status.json")],
            ["deferred_safe_tools_not_closed"],
        )
    if gpu_blocked or data_missing:
        add_intervention(
            "repair_execution_gates",
            "Refresh compute/data readiness before any training run.",
            "evomind ready",
            "resource_and_data_gate",
            ["docs/verified_workstation_launch_audit.json", str(xsci / "scientist_execution_contract.json")],
            ["gpu_or_external_resource_gate_blocked", "data_or_task_contract_not_ready"],
        )
    if no_hypothesis:
        add_intervention(
            "rank_research_hypotheses",
            "Generate and rank memory-guided research hypotheses before choosing a branch.",
            "evomind hypothesis-review",
            "hypothesis_review_gate",
            [str(xsci / "scientist_innovation_backlog.json"), str(xsci / "scientist_hypothesis_review.json")],
            ["hypothesis_review_missing_or_stale"],
        )
    if no_blueprint:
        add_intervention(
            "build_gated_blueprint",
            "Translate the reviewed hypothesis into an auditable experiment blueprint.",
            "evomind experiment-blueprint",
            "blueprint_gate",
            [str(xsci / "scientist_experiment_blueprint.json")],
            ["experiment_blueprint_missing"],
        )
    if memory_records <= 0:
        add_intervention(
            "consolidate_scientist_memory",
            "Write safe lessons and prior gate outcomes into retrospective memory.",
            "evomind memory-consolidate",
            "memory_writeback_gate",
            [str(xsci / "scientist_memory_consolidation.json"), str(root / "experiments" / "evolution" / "retrospective_memory.json")],
            ["memory_writeback_missing"],
        )
    add_intervention(
        "maintain_claim_gate",
        "Keep official rank, medal, and submit claims blocked until human approval and Kaggle response evidence exist.",
        "evomind readiness-report",
        "claim_audit_gate",
        [str(xsci / "scientist_readiness_report.json"), str(xsci / "scientist_causal_diagnosis.json")],
        ["official_leaderboard_evidence_missing"],
    )
    if no_queue:
        add_intervention(
            "refresh_action_queue",
            "Refresh the action queue so the next safe command is explicit and auditable.",
            "evomind autopilot",
            "action_queue_gate",
            [str(xsci / "scientist_action_queue.json"), str(xsci / "scientist_autopilot.json")],
            [],
        )

    if gpu_blocked or data_missing:
        posture = "repair_execution_gates_before_training"
    elif continuation_open:
        posture = "close_deferred_tools_before_next_decision"
    elif no_hypothesis or no_blueprint:
        posture = "complete_research_planning_chain"
    else:
        posture = "ready_for_human_gated_workstation_run_proxy"

    next_safe_command = next(
        (item["safe_next_command"] for item in interventions if item.get("safe_next_command")),
        "evomind autopilot",
    )
    evidence_refs = [
        {"name": "readiness_report", "path": str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json"), "present": bool(readiness)},
        {"name": "self_audit", "path": str(xsci / "scientist_self_audit.json"), "present": bool(self_audit)},
        {"name": "situation_model", "path": str(xsci / "scientist_situation_model.json"), "present": bool(situation)},
        {"name": "repair_plan", "path": str(xsci / "scientist_repair_plan.json"), "present": bool(repair)},
        {"name": "execution_contract", "path": str(xsci / "scientist_execution_contract.json"), "present": bool(contract)},
        {"name": "hypothesis_review", "path": str(xsci / "scientist_hypothesis_review.json"), "present": bool(review)},
        {"name": "experiment_blueprint", "path": str(xsci / "scientist_experiment_blueprint.json"), "present": bool(blueprint)},
    ]

    payload = {
        "ok": True,
        "tool": "scientist_causal_diagnosis",
        "schema": "evomind.ai_scientist.causal_diagnosis.v1",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "posture": posture,
        "symptoms": symptoms,
        "root_causes": root_causes,
        "interventions": interventions,
        "causal_graph": {"nodes": nodes, "edges": edges},
        "next_safe_command": next_safe_command,
        "claim_boundary": {
            "training": claim.get("training_readiness_claim") or "blocked_until_execution_gate_clear",
            "rank_or_medal": "blocked_without_kaggle_response_artifact",
            "official_submit": "blocked_until_explicit_human_approval",
        },
        "source_summaries": {
            "readiness_launch": readiness.get("launch_readiness"),
            "self_audit_score": self_audit.get("overall_score"),
            "memory_records": memory_records,
            "continuation_status": continuation.get("status") or "unknown",
            "action_count": len(action_items),
            "data_status": data_status.get("message") or data_status.get("tool"),
        },
        "evidence_refs": evidence_refs,
        "artifact_path": str(artifact_path),
        "markdown_artifact_path": str(markdown_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_scientist_causal_markdown(payload, markdown_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write causal diagnosis artifact: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"causal_diagnosis_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_causal_diagnosis",
            "task": session.selected_task or "",
            "phase": "causal_diagnosis",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_causal_diagnosis",
            "message": f"posture={posture}; root_causes={len(root_causes)}; next={next_safe_command}",
            "artifact_path": str(artifact_path),
            "details": {"root_causes": [item.get("id") for item in root_causes[:6]]},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_causal_diagnosis",
            "user": "scientist_causal_diagnosis",
            "forced_tools": ["scientist_readiness_report", "system_status", "data_check", "evolution_status"],
            "executed_tools": [{"tool": "scientist_causal_diagnosis", "ok": payload.get("ok", True)}],
            "mode": posture,
            "decision": {"next_safe_command": next_safe_command, "root_causes": len(root_causes)},
            "blockers": [item.get("summary") for item in symptoms[:8]],
            "next_actions": [item.get("safe_next_command") for item in interventions[:8] if item.get("safe_next_command")],
            "artifacts": [str(artifact_path), str(markdown_path)],
            "answer_preview": f"causal diagnosis: posture={posture}; next={next_safe_command}",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _write_scientist_strategy_optimizer_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# EvoMind AI Scientist Strategy Optimizer",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- selected_task: {payload.get('selected_task') or '(none)'}",
        f"- strategy_posture: {payload.get('strategy_posture')}",
        f"- next_safe_command: {payload.get('next_safe_command')}",
        f"- no_training_started: {payload.get('no_training_started', True)}",
        f"- official_submit: {payload.get('official_submit')}",
        "",
        "## Selected Strategy",
    ]
    selected = payload.get("selected_strategy") if isinstance(payload.get("selected_strategy"), dict) else {}
    if selected:
        lines.extend([
            f"- id: {selected.get('id')}",
            f"- title: {selected.get('title')}",
            f"- command: {selected.get('safe_next_command')}",
            f"- total_score: {selected.get('total_score')}",
            f"- gate_status: {selected.get('gate_status')}",
            f"- rationale: {selected.get('rationale')}",
            "",
        ])
    lines.append("## Intervention Ranking")
    for item in payload.get("intervention_ranking") or []:
        if not isinstance(item, dict):
            continue
        lines.extend([
            "",
            f"### {item.get('rank')}. {item.get('title')}",
            f"- id: {item.get('id')}",
            f"- command: {item.get('safe_next_command')}",
            f"- total_score: {item.get('total_score')}",
            f"- expected_impact: {item.get('expected_impact')}",
            f"- evidence_strength: {item.get('evidence_strength')}",
            f"- cost: {item.get('cost')}",
            f"- risk_penalty: {item.get('risk_penalty')}",
            f"- gate_status: {item.get('gate_status')}",
            f"- rationale: {item.get('rationale')}",
        ])
    lines.extend([
        "",
        "## Claim Boundary",
        "- Training remains behind EvoMind/workstation execution gates.",
        "- Official Kaggle submission remains blocked until explicit human approval.",
        "- Rank, top30, and medal claims remain blocked without Kaggle response artifacts.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strategy_optimizer_cost(command: str, gate: str, action: dict[str, Any]) -> tuple[int, str]:
    text = f"{command} {gate} {action.get('autonomy') or ''}".lower()
    if "run " in text or "download" in text or "submit" in text:
        return 35, "high"
    if "setup" in text or "ready" in text or "repair" in text:
        return 25, "medium"
    if "loop" in text or "autopilot" in text or "resume" in text:
        return 20, "medium"
    return 10, "low"


def _strategy_optimizer_risk(command: str, gate: str, status: str) -> tuple[int, str]:
    text = f"{command} {gate} {status}".lower()
    if "official" in text or "submit" in text or "medal" in text or "rank" in text:
        return 55, "human_gate_only"
    if "run " in text or "train" in text or "training" in text:
        return 45, "human_run_gate"
    if "download" in text:
        return 35, "data_gate"
    if "blocked" in text:
        return 30, "blocked_or_needs_gate"
    return 8, "read_only"


def _strategy_optimizer_gate_status(command: str, gate: str, status: str) -> str:
    text = f"{command} {gate} {status}".lower()
    if "official" in text or "submit" in text:
        return "blocked_until_explicit_human_approval"
    if "run " in text or "train" in text:
        return "blocked_until_evomind_run_gate"
    if "blocked" in text:
        return "blocked_or_needs_repair"
    return "safe_read_only"


def _strategy_optimizer_impact(
    *,
    action_id: str,
    gate: str,
    addresses: list[str],
    causal_posture: str,
    readiness_blocked: bool,
) -> int:
    text = f"{action_id} {gate} {' '.join(addresses)}".lower()
    score = 45
    if "resource" in text or "data" in text or "execution" in text:
        score += 28
    if "continuation" in text:
        score += 22
    if "hypothesis" in text or "blueprint" in text or "strategy" in text:
        score += 18
    if "memory" in text:
        score += 14
    if "claim" in text or "submit" in text or "rank" in text:
        score += 10
    if causal_posture == "repair_execution_gates_before_training" and ("resource" in text or "data" in text or "execution" in text):
        score += 20
    if readiness_blocked and ("resource" in text or "ready" in text or "repair" in text):
        score += 12
    return max(0, min(100, score))


def _strategy_optimizer_evidence_strength(action: dict[str, Any], source_count: int) -> int:
    evidence = action.get("evidence") if isinstance(action.get("evidence"), list) else []
    artifacts = action.get("expected_artifacts") if isinstance(action.get("expected_artifacts"), list) else []
    addresses = action.get("addresses") if isinstance(action.get("addresses"), list) else []
    score = 25 + min(25, source_count * 4)
    score += min(20, len([item for item in evidence if item]) * 4)
    score += min(20, len([item for item in artifacts if item]) * 4)
    score += min(10, len([item for item in addresses if item]) * 3)
    return max(0, min(100, score))


def _strategy_optimizer_candidates(
    *,
    causal: dict[str, Any],
    readiness: dict[str, Any],
    action_queue: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add_candidate(raw: dict[str, Any], *, source: str, fallback_id: str) -> None:
        action_id = _redacted_memory_text(raw.get("id") or fallback_id, limit=120)
        command = _redacted_memory_text(raw.get("safe_next_command") or raw.get("command") or "", limit=180)
        title = _redacted_memory_text(raw.get("title") or action_id or command, limit=240)
        if not (action_id or command or title):
            return
        candidates.append({
            "id": action_id or fallback_id,
            "title": title or action_id or fallback_id,
            "safe_next_command": command or "evomind strategy",
            "gate": _redacted_memory_text(raw.get("gate") or "", limit=120),
            "status": _redacted_memory_text(raw.get("status") or "", limit=80),
            "risk": _redacted_memory_text(raw.get("risk") or "", limit=240),
            "why": _redacted_memory_text(raw.get("why") or raw.get("summary") or raw.get("rationale") or "", limit=500),
            "expected_artifacts": _safe_string_list(raw.get("expected_artifacts"), limit=220, max_items=8),
            "evidence": _safe_string_list(raw.get("evidence"), limit=220, max_items=8),
            "addresses": _safe_string_list(raw.get("addresses"), limit=120, max_items=8),
            "source": source,
            "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        })

    for index, item in enumerate(causal.get("interventions") if isinstance(causal.get("interventions"), list) else []):
        if isinstance(item, dict):
            add_candidate(item, source="scientist_causal_diagnosis", fallback_id=f"causal_intervention_{index}")
    for index, item in enumerate(action_queue.get("actions") if isinstance(action_queue.get("actions"), list) else []):
        if isinstance(item, dict):
            add_candidate(item, source="scientist_action_queue", fallback_id=f"queue_action_{index}")
    for index, command in enumerate(readiness.get("recommended_next_commands") if isinstance(readiness.get("recommended_next_commands"), list) else []):
        text = _redacted_memory_text(command, limit=180)
        if text:
            add_candidate(
                {
                    "id": f"readiness_command_{index}",
                    "title": f"Run {text}",
                    "safe_next_command": text,
                    "gate": "readiness_gate",
                    "expected_artifacts": [str(readiness.get("artifact_path") or ".xsci/scientist_readiness_report.json")],
                    "why": "Recommended by Scientist readiness report.",
                },
                source="scientist_readiness_report",
                fallback_id=f"readiness_command_{index}",
            )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        key = (str(item.get("safe_next_command") or item.get("id") or "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def get_scientist_strategy_optimizer(session: SessionState, root: Path) -> dict[str, Any]:
    """Rank safe interventions by impact, evidence, risk, and gate status.

    This is the strategy-choice layer after causal diagnosis. It does not train,
    download data, modify code, or submit to Kaggle; it only makes the next
    read-only/gated command explicit and auditable.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_strategy_optimizer.json"
    markdown_path = xsci / "scientist_strategy_optimizer.md"

    causal = _read_json_artifact(xsci / "scientist_causal_diagnosis.json") or {}
    if not causal:
        causal = get_scientist_causal_diagnosis(session, root)
    readiness = _read_json_artifact(xsci / "scientist_readiness_report.json") or {}
    if not readiness:
        readiness = get_scientist_readiness_report(session, root)
    action_queue = _read_json_artifact(xsci / "scientist_action_queue.json") or {}
    if not action_queue:
        action_queue = get_scientist_action_queue(session, root)
    self_audit = _read_json_artifact(xsci / "scientist_self_audit.json") or {}
    situation = _read_json_artifact(xsci / "scientist_situation_model.json") or {}
    hypothesis_review = _read_json_artifact(xsci / "scientist_hypothesis_review.json") or {}
    experiment_blueprint = _read_json_artifact(xsci / "scientist_experiment_blueprint.json") or {}

    source_presence = {
        "readiness_report": bool(readiness),
        "causal_diagnosis": bool(causal),
        "action_queue": bool(action_queue),
        "self_audit": bool(self_audit),
        "situation_model": bool(situation),
        "hypothesis_review": bool(hypothesis_review),
        "experiment_blueprint": bool(experiment_blueprint),
    }
    source_count = len([ok for ok in source_presence.values() if ok])
    claim = readiness.get("claim_readiness") if isinstance(readiness.get("claim_readiness"), dict) else {}
    causal_posture = str(causal.get("posture") or "unknown")
    readiness_blocked = (
        str(readiness.get("launch_readiness") or "").lower().endswith("blocked")
        or str(claim.get("training_readiness_claim") or "") == "blocked_by_external_resource_or_data_gate"
        or bool(readiness.get("blocking_reasons"))
    )

    ranked: list[dict[str, Any]] = []
    for candidate in _strategy_optimizer_candidates(
        causal=causal,
        readiness=readiness,
        action_queue=action_queue,
    ):
        action_id = str(candidate.get("id") or "")
        gate = str(candidate.get("gate") or "")
        status = str(candidate.get("status") or "")
        command = str(candidate.get("safe_next_command") or "")
        addresses = [str(item) for item in (candidate.get("addresses") or [])]
        impact = _strategy_optimizer_impact(
            action_id=action_id,
            gate=gate,
            addresses=addresses,
            causal_posture=causal_posture,
            readiness_blocked=readiness_blocked,
        )
        evidence = _strategy_optimizer_evidence_strength(candidate, source_count)
        cost_score, cost = _strategy_optimizer_cost(command, gate, candidate)
        risk_penalty, risk_level = _strategy_optimizer_risk(command, gate, status)
        gate_status = _strategy_optimizer_gate_status(command, gate, status)
        total = round((impact * 0.45) + (evidence * 0.25) + ((100 - cost_score) * 0.15) - (risk_penalty * 0.15), 2)
        if gate_status == "safe_read_only":
            total += 6
        if causal_posture == "repair_execution_gates_before_training" and command in {"evomind ready", "evomind repair"}:
            total += 10
        rationale_parts = [
            f"source={candidate.get('source')}",
            f"impact={impact}",
            f"evidence={evidence}",
            f"cost={cost}",
            f"risk={risk_level}",
        ]
        if candidate.get("why"):
            rationale_parts.append(str(candidate.get("why")))
        ranked.append({
            "rank": 0,
            "id": action_id,
            "title": candidate.get("title"),
            "safe_next_command": command,
            "source": candidate.get("source"),
            "gate": gate,
            "gate_status": gate_status,
            "status": status or "candidate",
            "expected_impact": impact,
            "evidence_strength": evidence,
            "cost": cost,
            "cost_score": cost_score,
            "risk_level": risk_level,
            "risk_penalty": risk_penalty,
            "total_score": round(total, 2),
            "expected_artifacts": candidate.get("expected_artifacts") or [],
            "evidence": candidate.get("evidence") or [],
            "addresses": addresses,
            "rationale": _redacted_memory_text("; ".join(rationale_parts), limit=700),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        })
    ranked.sort(key=lambda item: (-float(item.get("total_score") or 0), str(item.get("id") or "")))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    selected_strategy = ranked[0] if ranked else {
        "id": "refresh_causal_diagnosis",
        "title": "Refresh causal diagnosis before strategy selection",
        "safe_next_command": "evomind causal-diagnosis",
        "gate_status": "safe_read_only",
        "total_score": 0,
        "rationale": "No ranked intervention candidates were available.",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    strategy_posture = (
        "repair_first" if readiness_blocked else
        "planning_chain_first" if any(str(item.get("id") or "").startswith(("rank_", "build_", "prepare_")) for item in ranked[:2]) else
        "safe_gated_run_candidate" if any(item.get("gate_status") == "blocked_until_evomind_run_gate" for item in ranked[:2]) else
        "observe_and_refine"
    )
    next_safe_command = str(selected_strategy.get("safe_next_command") or "evomind causal-diagnosis")
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_strategy_optimizer",
        "schema": "evomind.ai_scientist.strategy_optimizer.v1",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "strategy_posture": strategy_posture,
        "source_posture": causal_posture,
        "selected_strategy": selected_strategy,
        "intervention_ranking": ranked[:12],
        "decision_matrix": {
            "criteria": [
                {"name": "expected_impact", "weight": 0.45},
                {"name": "evidence_strength", "weight": 0.25},
                {"name": "low_cost", "weight": 0.15},
                {"name": "risk_penalty", "weight": -0.15},
                {"name": "safe_read_only_bonus", "weight": 6},
            ],
            "candidate_count": len(ranked),
            "source_presence": source_presence,
            "readiness_blocked": readiness_blocked,
            "claim_training_readiness": claim.get("training_readiness_claim"),
        },
        "next_safe_command": next_safe_command,
        "next_decision": {
            "selected_action": selected_strategy.get("id"),
            "selected_command": next_safe_command,
            "gate_status": selected_strategy.get("gate_status"),
            "requires_human_or_resource_gate": selected_strategy.get("gate_status") != "safe_read_only",
            "why": selected_strategy.get("rationale"),
        },
        "source_artifacts": {
            "readiness_report": str(readiness.get("artifact_path") or xsci / "scientist_readiness_report.json"),
            "causal_diagnosis": str(causal.get("artifact_path") or xsci / "scientist_causal_diagnosis.json"),
            "action_queue": str(action_queue.get("artifact_path") or xsci / "scientist_action_queue.json"),
            "self_audit": str(xsci / "scientist_self_audit.json"),
            "situation_model": str(xsci / "scientist_situation_model.json"),
            "hypothesis_review": str(xsci / "scientist_hypothesis_review.json"),
            "experiment_blueprint": str(xsci / "scientist_experiment_blueprint.json"),
        },
        "claim_boundary": {
            "training": claim.get("training_readiness_claim") or "blocked_until_execution_gate_clear",
            "rank_or_medal": "blocked_without_kaggle_response_artifact",
            "official_submit": "blocked_until_explicit_human_approval",
        },
        "artifact_path": str(artifact_path),
        "markdown_artifact_path": str(markdown_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_scientist_strategy_optimizer_markdown(payload, markdown_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write strategy optimizer artifact: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"strategy_optimizer_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_strategy_optimizer",
            "task": session.selected_task or "",
            "phase": "strategy_optimizer",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_strategy_optimizer",
            "message": f"selected={selected_strategy.get('id')}; next={next_safe_command}",
            "artifact_path": str(artifact_path),
            "details": {
                "strategy_posture": strategy_posture,
                "candidate_count": len(ranked),
                "selected_strategy": selected_strategy,
            },
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_strategy_optimizer",
            "user": "scientist_strategy_optimizer",
            "forced_tools": ["scientist_readiness_report", "scientist_causal_diagnosis", "scientist_action_queue"],
            "executed_tools": [{"tool": "scientist_strategy_optimizer", "ok": payload.get("ok", True)}],
            "mode": strategy_posture,
            "decision": payload["next_decision"],
            "blockers": readiness.get("blocking_reasons") if isinstance(readiness.get("blocking_reasons"), list) else [],
            "next_actions": [next_safe_command],
            "artifacts": [str(artifact_path), str(markdown_path)],
            "answer_preview": f"strategy optimizer selected={selected_strategy.get('id')}; next={next_safe_command}",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        })
    except Exception:
        pass
    return payload


def _upgrade_plan_files_for(item_id: str, expected_artifacts: list[str]) -> list[str]:
    files = list(expected_artifacts)
    mapping = {
        "memory_reuse_before_each_run": [
            "src/xsci/scientist_state.py",
            "src/xsci/terminal_tools.py",
            "experiments/evolution/retrospective_memory.json",
        ],
        "planner_executor_observer_loop": [
            "src/xsci/terminal_agent.py",
            "src/xsci/terminal_tools.py",
            "src/xsci/scientist_trace.py",
            "tests/test_autokaggle_cli.py",
        ],
        "frontend_self_audit_card": [
            "web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx",
            "web/research-agent-workstation/src/lib/server/summary.ts",
            "web/research-agent-workstation/src/lib/api/types.ts",
        ],
        "task_goal_to_gated_run_contract": [
            "src/xsci/scientist_state.py",
            "src/xsci/scientist_execution_gate.py",
            "src/xsci/terminal_tools.py",
            "tests/test_autokaggle_cli.py",
        ],
        "streaming_tool_confidence": [
            "src/xsci/terminal_agent.py",
            "src/xsci/terminal_events.py",
            "src/xsci/scientist_trace.py",
            "tests/test_autokaggle_cli.py",
        ],
        "innovation_trial_feedback_loop": [
            "src/xsci/terminal_tools.py",
            "src/xsci/evolution_tracker.py",
            "experiments/evolution/retrospective_memory.json",
            "tests/test_autokaggle_cli.py",
        ],
        "resource_gate_truthfulness": [
            "src/xsci/scientist_state.py",
            "src/xsci/scientist_execution_gate.py",
            "src/xsci/terminal_tools.py",
            "scripts/verify_launch_resource_readiness.py",
            "tests/test_autokaggle_cli.py",
        ],
        "codex_claude_parity_loop": [
            "src/xsci/terminal_agent.py",
            "src/xsci/terminal_tools.py",
            "src/xsci/scientist_turn_planner.py",
            "src/xsci/scientist_turns.py",
            "web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx",
            "tests/test_autokaggle_cli.py",
        ],
    }
    files.extend(mapping.get(item_id, ["src/xsci/terminal_tools.py", "tests/test_autokaggle_cli.py"]))
    return list(dict.fromkeys(_redacted_memory_text(path, limit=180) for path in files if path))[:10]


def _upgrade_plan_acceptance_for(item_id: str) -> list[str]:
    checks = [
        "python -m py_compile src\\xsci\\kaggle.py src\\xsci\\terminal_agent.py src\\xsci\\terminal_tools.py src\\xsci\\scientist_state.py",
        "python -m pytest tests\\test_autokaggle_cli.py -q --tb=short -k \"scientist_self_audit or scientist_upgrade or research_decision\"",
        "python scripts\\verify_no_plaintext_secrets.py",
    ]
    if item_id == "frontend_self_audit_card":
        checks.extend([
            "cd web\\research-agent-workstation && npm run typecheck",
            "cd web\\research-agent-workstation && npm run build",
        ])
    if item_id in {"codex_claude_parity_loop", "resource_gate_truthfulness"}:
        checks.append("python -m pytest tests\\test_autokaggle_cli.py -q --tb=short -k \"scientist_loop or scientist_turn or resource_gate or self_audit\"")
    if item_id == "codex_claude_parity_loop":
        checks.extend([
            "cd web\\research-agent-workstation && npm run typecheck",
            "cd web\\research-agent-workstation && npm run build",
        ])
    return checks


def _safe_string_list(values: Any, *, limit: int = 160, max_items: int = 12) -> list[str]:
    if not isinstance(values, list):
        return []
    items: list[str] = []
    for value in values:
        text = _redacted_memory_text(value, limit=limit)
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _scientist_requirement_context_packet(root: Path) -> dict[str, Any]:
    """Compact the latest requirement ledger into a safe self-evolution context.

    The packet deliberately separates code-agent-fixable gaps from external
    resource gates.  It is evidence for the next work order, not permission to
    train, download data, or submit to Kaggle.
    """
    root = Path(root)
    xsci = root / ".xsci"
    ledger, ledger_source = _load_latest_requirement_ledger(root)
    progress = _read_json_artifact(xsci / "scientist_requirement_progress.json") or {}
    memory = _read_json_artifact(xsci / "scientist_memory_consolidation.json") or {}
    terminal_turn = _read_json_artifact(xsci / "scientist_terminal_turn.json") or {}
    steps = _read_jsonl_tail(xsci / "scientist_step_trace.jsonl", limit=12)
    ledger_fingerprint = _requirement_ledger_fingerprint(ledger)
    recent_attempts = _recent_requirement_attempts(root)

    requirements = [
        item for item in (ledger.get("requirements") or [])
        if isinstance(item, dict)
    ] if isinstance(ledger, dict) else []
    open_ids = _safe_string_list(ledger.get("open_requirements") if isinstance(ledger, dict) else [], max_items=20)
    blocked_ids = _safe_string_list(ledger.get("blocked_requirements") if isinstance(ledger, dict) else [], max_items=20)

    external_resource_blockers: list[dict[str, Any]] = []
    code_agent_fixable_requirements: list[dict[str, Any]] = []
    training_gated_requirements: list[dict[str, Any]] = []
    read_only_next_actions: list[dict[str, Any]] = []

    for requirement in requirements:
        req_id = _redacted_memory_text(requirement.get("id") or "", limit=120)
        if not req_id:
            continue
        status = _redacted_memory_text(requirement.get("status") or "pending", limit=40)
        if status == "satisfied":
            continue
        gate = _redacted_memory_text(requirement.get("gate") or "", limit=120)
        reason = _redacted_memory_text(requirement.get("reason") or requirement.get("description") or "", limit=280)
        evidence = requirement.get("execution_evidence") if isinstance(requirement.get("execution_evidence"), dict) else {}
        mapped_hits = _safe_string_list(evidence.get("mapped_tool_hits"), max_items=8)
        artifact_hits = _safe_string_list(evidence.get("artifact_hits"), max_items=8)
        attempted = bool(mapped_hits) or (ledger_fingerprint, req_id) in recent_attempts or ("", req_id) in recent_attempts
        low = f"{req_id} {gate} {reason}".lower()
        external_gate = (
            req_id == "setup_gate_clearance"
            or any(token in low for token in ("gpu", "hpc", "ssh", "kaggle", "credential", "external", "data_missing"))
        )
        training_gate = (
            req_id in {"data_and_validation_contract", "execution_contract", "hypothesis_review"}
            or any(token in low for token in ("training", "submission", "rank", "medal", "leaderboard"))
        )
        summary = {
            "requirement_id": req_id,
            "status": status,
            "gate": gate,
            "reason": reason,
            "attempted_by_safe_tool": attempted,
            "mapped_tool_hits": mapped_hits,
            "artifact_hits": artifact_hits,
        }
        if external_gate:
            external_resource_blockers.append(summary)
        elif attempted:
            code_agent_fixable_requirements.append(summary)
        elif training_gate:
            training_gated_requirements.append(summary)
        else:
            spec = _requirement_action_spec(req_id, status)
            if spec:
                read_only_next_actions.append({
                    **summary,
                    "next_safe_command": spec["command"],
                    "autonomy": spec["autonomy"],
                })
            else:
                code_agent_fixable_requirements.append(summary)

    latest_progress = {
        "present": bool(progress),
        "requirement_id": _redacted_memory_text(progress.get("requirement_id") or "", limit=120) if isinstance(progress, dict) else "",
        "before_status": _redacted_memory_text(progress.get("before_status") or "", limit=40) if isinstance(progress, dict) else "",
        "after_status": _redacted_memory_text(progress.get("after_status") or "", limit=40) if isinstance(progress, dict) else "",
        "safe_tool": _redacted_memory_text(progress.get("safe_tool") or "", limit=120) if isinstance(progress, dict) else "",
        "tool_ok": bool(progress.get("tool_ok", False)) if isinstance(progress, dict) else False,
        "artifact_path": _redacted_memory_text(progress.get("artifact_path") or "", limit=220) if isinstance(progress, dict) else "",
        "open_requirements": _safe_string_list(progress.get("open_requirements") if isinstance(progress, dict) else [], max_items=20),
        "blocked_requirements": _safe_string_list(progress.get("blocked_requirements") if isinstance(progress, dict) else [], max_items=20),
    }
    latest_memory = {
        "present": bool(memory),
        "records_added": memory.get("records_added") if isinstance(memory, dict) else None,
        "records_total": memory.get("records_total") if isinstance(memory, dict) else None,
        "artifact_path": _redacted_memory_text(memory.get("artifact_path") or "", limit=220) if isinstance(memory, dict) else "",
    }
    recent_trace = [
        {
            "phase": _redacted_memory_text(item.get("phase") or "", limit=80),
            "status": _redacted_memory_text(item.get("status") or "", limit=80),
            "tool": _redacted_memory_text(item.get("tool") or "", limit=120),
            "message": _redacted_memory_text(item.get("message") or "", limit=220),
        }
        for item in steps[-5:]
        if isinstance(item, dict)
    ]
    return {
        "schema": "evomind.ai_scientist.self_evolution_context.v1",
        "goal": _redacted_memory_text(
            ledger.get("goal") if isinstance(ledger, dict) else terminal_turn.get("goal") or terminal_turn.get("user") or "",
            limit=320,
        ),
        "ledger_present": bool(ledger),
        "ledger_artifact_path": _redacted_memory_text(ledger_source, limit=220),
        "ledger_fingerprint": ledger_fingerprint,
        "open_requirements": open_ids,
        "blocked_requirements": blocked_ids,
        "latest_requirement_progress": latest_progress,
        "latest_memory_consolidation": latest_memory,
        "execution_partition": {
            "code_agent_fixable_requirements": code_agent_fixable_requirements[:8],
            "external_resource_blockers": external_resource_blockers[:8],
            "training_gated_requirements": training_gated_requirements[:8],
            "read_only_next_actions": read_only_next_actions[:8],
        },
        "recent_trace": recent_trace,
        "policy": {
            "code_edits": "allowed_only_after_code_agent_or_human_review",
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }


def _write_scientist_context_packet_markdown(payload: dict[str, Any], path: Path) -> None:
    """Write a compact human-readable briefing for the terminal and UI."""
    lines = [
        "# EvoMind AI Scientist Context Packet",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- selected_task: {payload.get('selected_task') or '(none)'}",
        f"- context_quality_score: {payload.get('context_quality', {}).get('score')}",
        f"- next_safe_command: {payload.get('next_safe_command')}",
        f"- no_training_started: {payload.get('no_training_started', True)}",
        f"- official_submit: {payload.get('official_submit')}",
        "",
        "## Readiness",
    ]
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    lines.extend([
        f"- llm_ready: {readiness.get('llm_ready')}",
        f"- kaggle_ready: {readiness.get('kaggle_ready')}",
        f"- compute_backend: {readiness.get('compute_backend')}",
        f"- can_execute: {readiness.get('can_execute')}",
    ])
    blockers = readiness.get("blocking_gates") if isinstance(readiness.get("blocking_gates"), list) else []
    if blockers:
        lines.append("- blocking_gates:")
        for item in blockers[:8]:
            lines.append(f"  - {item}")

    lines.extend(["", "## Active Strategy"])
    strategy = payload.get("active_strategy") if isinstance(payload.get("active_strategy"), dict) else {}
    if strategy:
        lines.extend([
            f"- posture: {strategy.get('strategy_posture')}",
            f"- selected_action: {strategy.get('selected_action')}",
            f"- selected_command: {strategy.get('selected_command')}",
            f"- gate_status: {strategy.get('gate_status')}",
        ])
    else:
        lines.append("- no strategy optimizer artifact yet")

    memory = payload.get("memory_digest") if isinstance(payload.get("memory_digest"), dict) else {}
    lines.extend([
        "",
        "## Memory",
        f"- retrospective_records: {memory.get('retrospective_records', 0)}",
        f"- scientist_memory_records_added: {memory.get('scientist_memory_records_added')}",
    ])
    lessons = memory.get("recent_lessons") if isinstance(memory.get("recent_lessons"), list) else []
    if lessons:
        lines.append("- recent_lessons:")
        for item in lessons[:5]:
            lines.append(f"  - {item}")

    requirements = payload.get("requirement_context") if isinstance(payload.get("requirement_context"), dict) else {}
    partition = requirements.get("execution_partition") if isinstance(requirements.get("execution_partition"), dict) else {}
    lines.extend([
        "",
        "## Requirement Context",
        f"- open_requirements: {len(requirements.get('open_requirements') or [])}",
        f"- blocked_requirements: {len(requirements.get('blocked_requirements') or [])}",
        f"- code_agent_fixable: {len(partition.get('code_agent_fixable_requirements') or [])}",
        f"- external_resource_blockers: {len(partition.get('external_resource_blockers') or [])}",
        f"- training_gated: {len(partition.get('training_gated_requirements') or [])}",
    ])
    lines.extend([
        "",
        "## Guardrails",
        "- Training remains behind EvoMind/workstation execution gates.",
        "- Official Kaggle submission remains blocked until explicit human approval.",
        "- Rank, top30, and medal claims remain blocked without Kaggle response artifacts.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_scientist_context_packet(session: SessionState, root: Path) -> dict[str, Any]:
    """Build the per-turn context packet that makes EvoMind feel stateful.

    Strong terminal agents carry a compact state packet into every turn: task,
    gates, memory, last strategy, requirement ledger, and claim boundaries. This
    tool materializes that packet for the terminal, the dashboard, and future
    LLM context injection. It is read-only with respect to training and
    submission: it may write context artifacts, but it never downloads data,
    starts model training, edits source, or submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_context_packet.json"
    markdown_path = xsci / "scientist_context_packet.md"

    task_profile = _selected_task_profile(session, root)
    readiness = {
        "llm_ready": bool(session.llm_ready),
        "kaggle_ready": bool(session.kaggle_ready),
        "compute_backend": session.compute_backend,
        "gpu_ready": bool(session.gpu_ready),
        "gpu_blocked": bool(session.gpu_blocked),
        "can_execute": bool(session.can_execute()),
        "blocking_gates": session.blocking_setup(),
        "advisory_gaps": session.missing_setup(),
    }

    strategy_artifact = _read_json_artifact(xsci / "scientist_strategy_optimizer.json") or {}
    next_decision = strategy_artifact.get("next_decision") if isinstance(strategy_artifact.get("next_decision"), dict) else {}
    selected_strategy = strategy_artifact.get("selected_strategy") if isinstance(strategy_artifact.get("selected_strategy"), dict) else {}
    active_strategy = {
        "present": bool(strategy_artifact),
        "strategy_posture": strategy_artifact.get("strategy_posture") or "",
        "selected_action": next_decision.get("selected_action") or selected_strategy.get("id") or "",
        "selected_command": next_decision.get("selected_command") or strategy_artifact.get("next_safe_command") or "",
        "gate_status": next_decision.get("gate_status") or selected_strategy.get("gate_status") or "",
        "why": _redacted_memory_text(next_decision.get("why") or selected_strategy.get("rationale") or "", limit=500),
        "artifact_path": _redacted_memory_text(strategy_artifact.get("artifact_path") or str(xsci / "scientist_strategy_optimizer.json"), limit=220),
    }

    requirement_context = _scientist_requirement_context_packet(root)
    memory_path = root / "experiments" / "evolution" / "retrospective_memory.json"
    memory_records = _memory_records_from_payload(_read_json_payload(memory_path))
    recent_lessons: list[str] = []
    ranked_memory = sorted(
        (
            (*_memory_relevance(record, task_profile), index, record)
            for index, record in enumerate(memory_records)
        ),
        key=lambda row: (-row[0], -row[2]),
    )
    relevant_memory_records = [
        record
        for score, _, _, record in ranked_memory
        if score >= 20
    ][:12]
    for record in relevant_memory_records:
        lesson = _redacted_memory_text(
            record.get("reusable_strategy") or record.get("what_worked") or record.get("failure_pattern") or "",
            limit=220,
        )
        if lesson and lesson not in recent_lessons:
            recent_lessons.append(lesson)
    memory_consolidation = _read_json_artifact(xsci / "scientist_memory_consolidation.json") or {}
    memory_digest = {
        "retrospective_records": len(memory_records),
        "retrospective_memory_path": str(memory_path),
        "scientist_memory_records_added": memory_consolidation.get("records_added"),
        "scientist_memory_records_total": memory_consolidation.get("records_total"),
        "scientist_memory_artifact_path": memory_consolidation.get("artifact_path") or "",
        "recent_lessons": recent_lessons[:5],
        "task_relevant_records": len(relevant_memory_records),
    }

    artifact_names = [
        "scientist_turn_plan.json",
        "scientist_terminal_turn.json",
        "scientist_reasoning_synthesis.json",
        "scientist_situation_model.json",
        "scientist_strategy_optimizer.json",
        "scientist_action_queue.json",
        "scientist_execution_contract.json",
        "scientist_hypothesis_review.json",
        "scientist_experiment_blueprint.json",
        "scientist_self_audit.json",
        "scientist_memory_consolidation.json",
    ]
    artifact_inventory = []
    for name in artifact_names:
        path = xsci / name
        artifact_inventory.append({
            "name": name,
            "path": str(path),
            "present": path.exists(),
        })
    missing_sources = [item["name"] for item in artifact_inventory if not item["present"]]
    present_count = len(artifact_inventory) - len(missing_sources)
    score = 40 + min(30, present_count * 3)
    if session.selected_task:
        score += 10
    if active_strategy["present"]:
        score += 10
    if memory_records:
        score += 10
    if readiness["blocking_gates"]:
        score -= 15
    score = max(0, min(100, score))

    if active_strategy.get("selected_command"):
        next_safe_command = str(active_strategy["selected_command"])
    elif readiness["blocking_gates"]:
        next_safe_command = "evomind repair"
    elif not session.selected_task:
        next_safe_command = "evomind tasks"
    else:
        next_safe_command = "evomind strategy"

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_context_packet",
        "schema": "evomind.ai_scientist.context_packet.v1",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "task_profile": task_profile,
        "readiness": readiness,
        "active_strategy": active_strategy,
        "requirement_context": requirement_context,
        "memory_digest": memory_digest,
        "artifact_inventory": artifact_inventory,
        "context_quality": {
            "score": score,
            "present_artifacts": present_count,
            "missing_sources": missing_sources,
            "interpretation": (
                "rich_context"
                if score >= 75 else
                "usable_but_gated"
                if score >= 55 else
                "thin_context_needs_observation"
            ),
        },
        "next_safe_command": next_safe_command,
        "response_contract": {
            "must_use_context_packet": True,
            "must_name_current_gates": True,
            "must_reference_artifacts_when_claiming_progress": True,
            "must_not_claim_training_success_without_metrics_artifact": True,
            "must_not_claim_rank_or_medal_without_kaggle_response": True,
        },
        "artifact_path": str(artifact_path),
        "markdown_artifact_path": str(markdown_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_scientist_context_packet_markdown(payload, markdown_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write context packet: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"context_packet_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_context_packet",
            "task": session.selected_task or "",
            "phase": "context_packet",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_context_packet",
            "message": f"context_quality={score}; next={next_safe_command}",
            "artifact_path": str(artifact_path),
            "details": {
                "score": score,
                "missing_sources": missing_sources[:8],
                "next_safe_command": next_safe_command,
            },
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def get_scientist_upgrade_plan(session: SessionState, root: Path) -> dict[str, Any]:
    """Turn the self-audit upgrade backlog into a concrete engineering plan.

    This is the planner side of self-evolution. It reads the latest capability
    audit/backlog, expands each open item into files, tests, gates, and rollback
    checks, then stops. It never edits code, starts training, downloads data, or
    submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_upgrade_plan.json"
    backlog_path = xsci / "scientist_upgrade_backlog.json"
    audit_path = xsci / "scientist_self_audit.json"

    backlog_payload = _read_json_artifact(backlog_path)
    audit_payload = _read_json_artifact(audit_path)
    if not isinstance(backlog_payload, dict) and not isinstance(audit_payload, dict):
        audit_payload = get_scientist_self_audit(session, root)
        backlog_payload = _read_json_artifact(backlog_path)

    items: list[dict[str, Any]] = []
    raw_items = []
    if isinstance(backlog_payload, dict) and isinstance(backlog_payload.get("items"), list):
        raw_items = backlog_payload.get("items") or []
    elif isinstance(audit_payload, dict) and isinstance(audit_payload.get("upgrade_backlog"), list):
        raw_items = audit_payload.get("upgrade_backlog") or []
    for item in raw_items:
        if isinstance(item, dict):
            status = str(item.get("status") or "proposed").lower()
            if status not in {"done", "closed", "resolved", "complete", "completed"}:
                items.append(item)

    priority_order = {"P0": 0, "CRITICAL": 0, "HIGH": 0, "P1": 1, "MEDIUM": 1, "P2": 2, "LOW": 2}
    items.sort(key=lambda item: (
        priority_order.get(str(item.get("priority") or "").upper(), 9),
        str(item.get("id") or item.get("title") or ""),
    ))

    plan_steps: list[dict[str, Any]] = []
    for idx, item in enumerate(items[:8], start=1):
        item_id = _redacted_memory_text(item.get("id") or f"upgrade_{idx}", limit=100)
        expected = [
            _redacted_memory_text(path, limit=180)
            for path in (item.get("expected_artifacts") or [])
            if _redacted_memory_text(path, limit=180)
        ][:8]
        plan_steps.append({
            "step_id": f"upgrade_step_{idx:02d}",
            "backlog_id": item_id,
            "priority": _redacted_memory_text(item.get("priority") or "P?", limit=20),
            "title": _redacted_memory_text(item.get("title") or item_id, limit=180),
            "why": _redacted_memory_text(item.get("why") or "", limit=260),
            "files_to_inspect": _upgrade_plan_files_for(item_id, expected),
            "expected_artifacts": expected,
            "acceptance_checks": _upgrade_plan_acceptance_for(item_id),
            "closure_gate": [
                "all acceptance checks pass",
                "no plaintext secret appears in artifacts or terminal output",
                "no training/download/official Kaggle submit is started by this upgrade tool",
                "claim audit remains blocked for rank/medal without Kaggle response artifact",
            ],
            "safe_next_command": _redacted_memory_text(item.get("safe_next_command") or "evomind self-audit", limit=120),
            "status": "planned",
            "no_training_started": True,
        })

    readiness = "ready_for_engineering_review" if plan_steps else "no_open_upgrade_backlog"
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_upgrade_plan",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "readiness": readiness,
        "source_backlog_path": str(backlog_path),
        "source_self_audit_path": str(audit_path),
        "overall_score": (audit_payload or backlog_payload or {}).get("overall_score") if isinstance((audit_payload or backlog_payload), dict) else None,
        "open_backlog_count": len(items),
        "planned_steps": plan_steps,
        "execution_policy": {
            "mode": "engineering_plan_only",
            "requires_human_or_code_agent_review": True,
            "training": "blocked",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
        },
        "next_safe_commands": [
            "evomind self-audit",
            "evomind upgrade-plan",
            "evomind decide",
        ],
        "artifact_path": str(artifact_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write upgrade plan artifact: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"upgrade_plan_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_upgrade_plan",
            "task": session.selected_task or "",
            "phase": "upgrade_plan",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_upgrade_plan",
            "message": f"planned_steps={len(plan_steps)}; readiness={readiness}",
            "artifact_path": str(artifact_path),
            "details": {"planned_steps": len(plan_steps), "readiness": readiness},
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_upgrade_plan",
            "user": "scientist_upgrade_plan",
            "forced_tools": ["scientist_self_audit", "research_decision"],
            "executed_tools": [{"tool": "scientist_upgrade_plan", "ok": payload.get("ok", True)}],
            "mode": readiness,
            "decision": {
                "selected_action": "plan_agent_upgrade_backlog",
                "selected_branch": "scientist_capability_upgrade",
                "planned_steps": len(plan_steps),
            },
            "blockers": [],
            "next_actions": payload["next_safe_commands"],
            "artifacts": [str(artifact_path), str(backlog_path), str(audit_path)],
            "answer_preview": f"upgrade plan generated; steps={len(plan_steps)}; no_training_started=True",
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _upgrade_work_order_files(step: dict[str, Any]) -> list[str]:
    files = []
    for item in step.get("files_to_inspect") or []:
        text = _redacted_memory_text(item, limit=220)
        if not text:
            continue
        if text.startswith(("src/", "web/", "scripts/", "tests/")):
            files.append(text)
    return list(dict.fromkeys(files))[:8]


def get_scientist_self_upgrade_loop(session: SessionState, root: Path) -> dict[str, Any]:
    """Select the next P0 agent-capability upgrade and create a work order.

    This is EvoMind's safe self-evolution bridge: observe the latest self-audit,
    plan from the upgrade backlog, create a concrete code-agent work order, write
    trace/turn evidence, and stop. It never edits source code, starts training,
    downloads data, or submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_self_upgrade_loop.json"
    work_order_path = xsci / "scientist_self_upgrade_work_order.json"
    trials_path = xsci / "scientist_self_upgrade_trials.jsonl"

    audit = get_scientist_self_audit(session, root)
    plan = get_scientist_upgrade_plan(session, root)
    self_evolution_context = _scientist_requirement_context_packet(root)
    steps = plan.get("planned_steps") if isinstance(plan, dict) else []
    if not isinstance(steps, list):
        steps = []
    selected_step = next((item for item in steps if isinstance(item, dict) and str(item.get("priority") or "").upper() == "P0"), None)
    if selected_step is None:
        selected_step = next((item for item in steps if isinstance(item, dict)), None)

    selected_backlog_id = _redacted_memory_text(
        (selected_step or {}).get("backlog_id") or "",
        limit=100,
    ) if isinstance(selected_step, dict) else ""
    selected_title = _redacted_memory_text(
        (selected_step or {}).get("title") or selected_backlog_id or "No open self-upgrade backlog",
        limit=220,
    ) if isinstance(selected_step, dict) else "No open self-upgrade backlog"
    files_to_edit = _upgrade_work_order_files(selected_step or {}) if isinstance(selected_step, dict) else []
    acceptance_checks = [
        _redacted_memory_text(item, limit=260)
        for item in ((selected_step or {}).get("acceptance_checks") or [])
        if _redacted_memory_text(item, limit=260)
    ][:10] if isinstance(selected_step, dict) else []
    expected_artifacts = [
        _redacted_memory_text(item, limit=220)
        for item in ((selected_step or {}).get("expected_artifacts") or [])
        if _redacted_memory_text(item, limit=220)
    ][:8] if isinstance(selected_step, dict) else []

    status = "ready_for_code_agent" if selected_step else "no_open_upgrade_backlog"
    work_order = {
        "work_order_id": f"self_upgrade_{generated_at.replace(':', '').replace('+', 'Z')}",
        "selected_backlog_id": selected_backlog_id,
        "title": selected_title,
        "priority": _redacted_memory_text((selected_step or {}).get("priority") or "P?", limit=20) if isinstance(selected_step, dict) else "",
        "objective": (
            "Implement the selected EvoMind AI Scientist capability upgrade without starting training, "
            "without official Kaggle submission, and without writing secrets to artifacts."
        ),
        "files_to_edit": files_to_edit,
        "files_to_inspect": (selected_step or {}).get("files_to_inspect") if isinstance(selected_step, dict) else [],
        "expected_artifacts": expected_artifacts,
        "self_evolution_context": self_evolution_context,
        "execution_partition": self_evolution_context.get("execution_partition", {}),
        "acceptance_checks": acceptance_checks,
        "rollback_condition": "Revert only the self-upgrade patch if py_compile, focused pytest, frontend build, or secret scan fails.",
        "claim_boundary": "This work order is not a Kaggle score, rank, medal, or top30 claim.",
        "code_agent_prompt": (
            f"Implement EvoMind self-upgrade backlog `{selected_backlog_id}`: {selected_title}. "
            "Use the self_evolution_context to separate code-agent gaps from external gates. "
            f"Edit only the listed files, preserve no-training/no-submit gates, then run acceptance checks."
        ) if selected_step else "No open self-upgrade backlog is available.",
        "human_gate": "review_patch_before_merge",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    action_queue = {
        "tool": "scientist_self_upgrade_action_queue",
        "source": "scientist_self_upgrade_loop",
        "generated_at": generated_at,
        "actions": [
            {
                "id": selected_backlog_id or "no_open_upgrade_backlog",
                "title": selected_title,
                "status": status,
                "command": "evomind self-upgrade",
                "work_order_path": str(work_order_path),
                "gate": "engineering_review_required",
                "no_training_started": True,
            }
        ] if selected_step else [],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    lesson = {
        "ts": generated_at,
        "source": "scientist_self_upgrade_loop",
        "selected_backlog_id": selected_backlog_id,
        "status": status,
        "lesson": (
            f"Selected self-upgrade backlog {selected_backlog_id}; work order created for engineering review."
            if selected_step else
            "No open self-upgrade backlog; run self-audit before claiming parity."
        ),
        "no_training_started": True,
    }
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_self_upgrade_loop",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "status": status,
        "selected_backlog_id": selected_backlog_id,
        "selected_step_id": (selected_step or {}).get("step_id") if isinstance(selected_step, dict) else None,
        "selected_title": selected_title,
        "overall_score_before": audit.get("overall_score") if isinstance(audit, dict) else None,
        "open_backlog_count": plan.get("open_backlog_count") if isinstance(plan, dict) else 0,
        "work_order": work_order,
        "self_evolution_context": self_evolution_context,
        "action_queue": action_queue,
        "loop_phases": [
            {"phase": "observe", "artifact": audit.get("artifact_path") if isinstance(audit, dict) else str(xsci / "scientist_self_audit.json")},
            {"phase": "plan", "artifact": plan.get("artifact_path") if isinstance(plan, dict) else str(xsci / "scientist_upgrade_plan.json")},
            {"phase": "act", "artifact": str(work_order_path), "status": status},
            {"phase": "reflect", "artifact": str(trials_path), "status": "lesson_recorded"},
        ],
        "next_safe_commands": [
            "evomind self-upgrade",
            "evomind upgrade-plan",
            "evomind self-audit",
        ],
        "artifact_path": str(artifact_path),
        "work_order_path": str(work_order_path),
        "trials_path": str(trials_path),
        "source_upgrade_plan_path": plan.get("artifact_path") if isinstance(plan, dict) else str(xsci / "scientist_upgrade_plan.json"),
        "source_self_audit_path": audit.get("artifact_path") if isinstance(audit, dict) else str(xsci / "scientist_self_audit.json"),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "source_code_changes": "blocked_until_code_agent_or_human_review_applies_patch",
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    try:
        xsci.mkdir(parents=True, exist_ok=True)
        work_order_path.write_text(json.dumps(work_order, ensure_ascii=False, indent=2), encoding="utf-8")
        (xsci / "scientist_self_upgrade_action_queue.json").write_text(
            json.dumps(action_queue, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with trials_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson, ensure_ascii=False) + "\n")
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write self-upgrade loop artifacts: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"self_upgrade_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_self_upgrade_loop",
            "task": session.selected_task or "",
            "phase": "self_upgrade_work_order",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_self_upgrade_loop",
            "message": f"selected={selected_backlog_id or 'none'}; status={status}",
            "artifact_path": str(artifact_path),
            "details": {
                "selected_backlog_id": selected_backlog_id,
                "files_to_edit": files_to_edit,
                "acceptance_checks": acceptance_checks[:5],
                "open_requirements": self_evolution_context.get("open_requirements", []),
                "blocked_requirements": self_evolution_context.get("blocked_requirements", []),
            },
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_self_upgrade_loop",
            "user": "scientist_self_upgrade_loop",
            "forced_tools": ["scientist_self_audit", "scientist_upgrade_plan"],
            "executed_tools": [
                {"tool": "scientist_self_audit", "ok": audit.get("ok", True) if isinstance(audit, dict) else True},
                {"tool": "scientist_upgrade_plan", "ok": plan.get("ok", True) if isinstance(plan, dict) else True},
                {"tool": "scientist_self_upgrade_loop", "ok": payload.get("ok", True)},
            ],
            "mode": status,
            "decision": {
                "selected_action": "create_self_upgrade_work_order",
                "selected_branch": "scientist_capability_upgrade",
                "selected_backlog_id": selected_backlog_id,
                "code_generation_mode": "work_order_only",
            },
            "blockers": [] if selected_step else ["no_open_upgrade_backlog"],
            "next_actions": payload["next_safe_commands"],
            "artifacts": [str(artifact_path), str(work_order_path), str(trials_path)],
            "answer_preview": f"self-upgrade work order selected={selected_backlog_id or 'none'}; no_training_started=True",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        })
    except Exception:
        pass
    return payload


def _patch_work_order_files_for(issue_id: str, fallback: list[str] | None = None) -> list[str]:
    mapping = {
        "scientist_turn_budget_exhausted": [
            "src/xsci/terminal_agent.py",
            "src/xsci/scientist_turn_planner.py",
            "src/xsci/terminal_tools.py",
            "tests/test_autokaggle_cli.py",
        ],
        "scientist_parity_lifecycle_incomplete": [
            "src/xsci/terminal_agent.py",
            "src/xsci/scientist_turns.py",
            "src/xsci/scientist_turn_planner.py",
            "web/research-agent-workstation/src/components/workstation/AiControlConsole.tsx",
            "tests/test_autokaggle_cli.py",
        ],
        "self_audit_backlog_patch": fallback or [
            "src/xsci/terminal_tools.py",
            "src/xsci/terminal_agent.py",
            "tests/test_autokaggle_cli.py",
        ],
    }
    return list(dict.fromkeys(mapping.get(issue_id, fallback or ["src/xsci/terminal_tools.py", "tests/test_autokaggle_cli.py"])))[:10]


def _patch_work_order_acceptance_for(issue_id: str) -> list[str]:
    checks = [
        "python -m py_compile src\\xsci\\kaggle.py src\\xsci\\terminal_agent.py src\\xsci\\terminal_tools.py src\\xsci\\scientist_turn_planner.py src\\xsci\\scientist_turns.py",
        "python -m pytest tests\\test_autokaggle_cli.py -q --tb=short -k \"scientist_patch_work_order or scientist_turn_plan or scientist_terminal_turn or self_audit\"",
        "python scripts\\verify_no_plaintext_secrets.py",
    ]
    if issue_id == "scientist_parity_lifecycle_incomplete":
        checks.extend([
            "cd web\\research-agent-workstation && npm run typecheck",
            "cd web\\research-agent-workstation && npm run build",
        ])
    return checks


def get_scientist_patch_work_order(session: SessionState, root: Path) -> dict[str, Any]:
    """Create a code-agent patch work order from the latest Scientist evidence.

    This is the missing bridge between "diagnose the system" and "repair the
    system like Codex/Claude Code".  It reads only local artifacts, classifies
    whether the issue is code-editable or an external gate, writes a concrete
    patch work order, and stops.  It never edits files, trains, downloads data,
    or submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    xsci = root / ".xsci"
    artifact_path = xsci / "scientist_patch_work_order.json"
    action_queue_path = xsci / "scientist_patch_action_queue.json"
    trials_path = xsci / "scientist_patch_trials.jsonl"

    terminal_turn = _read_json_artifact(xsci / "scientist_terminal_turn.json") or {}
    latest_parity = _read_json_artifact(xsci / "scientist_latest_parity_loop.json") or {}
    repair_plan = _read_json_artifact(xsci / "scientist_repair_plan.json") or {}
    self_audit = _read_json_artifact(xsci / "scientist_self_audit.json") or {}
    upgrade_plan = _read_json_artifact(xsci / "scientist_upgrade_plan.json") or {}
    steps = _read_jsonl_tail(xsci / "scientist_step_trace.jsonl", limit=30)
    requirement_context = _scientist_requirement_context_packet(root)

    parity_lifecycle = terminal_turn.get("parity_lifecycle") if isinstance(terminal_turn.get("parity_lifecycle"), dict) else {}
    phase_status = parity_lifecycle.get("phase_status") if isinstance(parity_lifecycle.get("phase_status"), dict) else {}
    if not phase_status and isinstance(latest_parity.get("phase_status"), dict):
        phase_status = latest_parity.get("phase_status") or {}
    root_causes = repair_plan.get("root_causes") if isinstance(repair_plan.get("root_causes"), list) else []
    blockers = terminal_turn.get("blocking_gates") if isinstance(terminal_turn.get("blocking_gates"), list) else []
    planned_steps = upgrade_plan.get("planned_steps") if isinstance(upgrade_plan.get("planned_steps"), list) else []

    issue_id = "self_audit_backlog_patch"
    status = "ready_for_code_agent"
    title = "Convert the highest-priority EvoMind capability gap into a patch"
    rationale = "Latest self-audit/upgrade plan can be transformed into a code-agent patch task."
    safe_next_command = "evomind self-upgrade"
    selected_step = next((item for item in planned_steps if isinstance(item, dict) and str(item.get("priority") or "").upper() == "P0"), None)
    if selected_step is None:
        selected_step = next((item for item in planned_steps if isinstance(item, dict)), None)
    fallback_files = _upgrade_work_order_files(selected_step or {}) if isinstance(selected_step, dict) else []

    budget_exhausted = bool(terminal_turn.get("budget_exhausted")) or str(phase_status.get("improve") or "") == "needs_more_tools"
    required_phases = {"observe", "plan", "act", "reflect", "improve"}
    phase_names = {str(key) for key in phase_status.keys()}
    lifecycle_incomplete = bool(phase_status) and not required_phases.issubset(phase_names)
    external_gate = any(str(cause) in {"gpu_blocked", "data_missing", "kaggle_not_configured"} for cause in root_causes)
    execution_partition = requirement_context.get("execution_partition") if isinstance(requirement_context.get("execution_partition"), dict) else {}
    external_requirement_blocked = bool(execution_partition.get("external_resource_blockers"))
    external_gate = external_gate or external_requirement_blocked

    if budget_exhausted:
        issue_id = "scientist_turn_budget_exhausted"
        title = "Repair Scientist Turn budget handling so required tools are not skipped silently"
        rationale = (
            "The latest AI Scientist turn recorded deferred or must-run tools; a Claude-Code-like agent should "
            "either expand the safe budget or mark the turn incomplete with an explicit continuation command."
        )
        safe_next_command = "evomind ask --json --max-tools 8 \"复核当前系统智能体闭环，不启动训练\""
    elif lifecycle_incomplete:
        issue_id = "scientist_parity_lifecycle_incomplete"
        title = "Repair incomplete observe-plan-act-reflect-improve lifecycle evidence"
        rationale = "The latest parity ledger does not cover all required lifecycle phases."
        safe_next_command = "evomind ask --json \"复核当前系统智能体闭环，不启动训练\""
    elif external_gate:
        issue_id = "external_gate_not_code_patch"
        status = "blocked_external_gate"
        title = "External resource/data gate blocks execution; code patch is not the next action"
        rationale = (
            "Current repair evidence points at GPU/Kaggle/data readiness rather than a code-editable defect. "
            "EvoMind should keep training blocked and refresh resource proof instead of inventing a patch."
        )
        safe_next_command = str(repair_plan.get("safe_next_command") or "evomind ready")
    elif isinstance(selected_step, dict):
        title = _redacted_memory_text(selected_step.get("title") or title, limit=220)
        rationale = _redacted_memory_text(selected_step.get("why") or rationale, limit=420)
        safe_next_command = _redacted_memory_text(selected_step.get("safe_next_command") or safe_next_command, limit=160)

    files_to_edit = [] if status == "blocked_external_gate" else _patch_work_order_files_for(issue_id, fallback_files)
    acceptance_checks = [] if status == "blocked_external_gate" else _patch_work_order_acceptance_for(issue_id)
    expected_artifacts = [
        ".xsci/scientist_patch_work_order.json",
        ".xsci/scientist_patch_action_queue.json",
        ".xsci/scientist_step_trace.jsonl",
    ]
    if status != "blocked_external_gate":
        expected_artifacts.extend([
            ".xsci/scientist_terminal_turn.json",
            ".xsci/scientist_parity_loop.jsonl",
        ])

    evidence = {
        "terminal_turn": {
            "present": bool(terminal_turn),
            "budget_exhausted": bool(terminal_turn.get("budget_exhausted")),
            "deferred_tools": terminal_turn.get("deferred_tools") if isinstance(terminal_turn.get("deferred_tools"), list) else [],
            "must_run_deferred_tools": terminal_turn.get("must_run_deferred_tools") if isinstance(terminal_turn.get("must_run_deferred_tools"), list) else [],
            "blocking_gates": [_redacted_memory_text(item, limit=260) for item in blockers[:5]],
        },
        "parity": {
            "present": bool(latest_parity or parity_lifecycle),
            "phase_status": {str(k): _redacted_memory_text(v, limit=80) for k, v in dict(phase_status).items()},
            "schema": _redacted_memory_text(parity_lifecycle.get("schema") or (latest_parity.get("lifecycle") or {}).get("schema"), limit=120)
            if isinstance((latest_parity.get("lifecycle") or {}), dict) or parity_lifecycle else "",
        },
        "repair_plan": {
            "present": bool(repair_plan),
            "root_causes": [_redacted_memory_text(item, limit=120) for item in root_causes[:6]],
            "safe_next_command": _redacted_memory_text(repair_plan.get("safe_next_command") or "", limit=160),
        },
        "self_audit": {
            "present": bool(self_audit),
            "overall_score": self_audit.get("overall_score"),
            "launch_readiness": _redacted_memory_text(self_audit.get("launch_readiness") or "", limit=120),
        },
        "requirement_context": {
            "present": bool(requirement_context.get("ledger_present")),
            "ledger_artifact_path": requirement_context.get("ledger_artifact_path", ""),
            "open_requirements": requirement_context.get("open_requirements", []),
            "blocked_requirements": requirement_context.get("blocked_requirements", []),
            "latest_requirement_progress": requirement_context.get("latest_requirement_progress", {}),
            "execution_partition": execution_partition,
        },
        "recent_steps": [
            {
                "phase": _redacted_memory_text(item.get("phase") or "", limit=80),
                "status": _redacted_memory_text(item.get("status") or "", limit=80),
                "tool": _redacted_memory_text(item.get("tool") or "", limit=120),
                "message": _redacted_memory_text(item.get("message") or "", limit=260),
            }
            for item in steps[-6:]
            if isinstance(item, dict)
        ],
    }

    work_order = {
        "work_order_id": f"patch_{generated_at.replace(':', '').replace('+', 'Z')}",
        "issue_id": issue_id,
        "status": status,
        "title": title,
        "objective": (
            "Improve EvoMind's AI Scientist behavior from the latest evidence without starting training, "
            "without downloading data, without official Kaggle submission, and without exposing secrets."
        ),
        "rationale": _redacted_memory_text(rationale, limit=700),
        "files_to_edit": files_to_edit,
        "files_to_inspect": list(dict.fromkeys(files_to_edit + [
            ".xsci/scientist_terminal_turn.json",
            ".xsci/scientist_parity_loop.jsonl",
            ".xsci/scientist_repair_plan.json",
            ".xsci/scientist_self_audit.json",
            ".xsci/scientist_requirement_progress.json",
        ]))[:14],
        "self_evolution_context": requirement_context,
        "execution_partition": execution_partition,
        "acceptance_checks": acceptance_checks,
        "expected_artifacts": expected_artifacts,
        "rollback_condition": "Revert only the patch-work-order changes if compile, focused pytest, frontend build, or secret scan fails.",
        "claim_boundary": "This is an engineering repair work order, not a Kaggle score/rank/medal/top30 claim.",
        "safe_next_command": safe_next_command,
        "code_agent_prompt": (
            f"Implement EvoMind patch work order `{issue_id}`: {title}. "
            "Use the evidence in .xsci/scientist_patch_work_order.json. "
            "Inspect self_evolution_context before editing so external gates are not treated as code defects. "
            "Edit only listed source/test/frontend files, preserve no-training/no-submit gates, "
            "do not print secrets, then run the acceptance checks."
        ) if status == "ready_for_code_agent" else (
            "Do not patch source yet. Clear the external gate first: " + safe_next_command
        ),
        "human_gate": "review_patch_before_merge" if status == "ready_for_code_agent" else "clear_external_gate_before_patch",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    action_queue = {
        "tool": "scientist_patch_action_queue",
        "source": "scientist_patch_work_order",
        "generated_at": generated_at,
        "actions": [
            {
                "id": issue_id,
                "title": title,
                "status": status,
                "command": "evomind patch-order" if status == "ready_for_code_agent" else safe_next_command,
                "work_order_path": str(artifact_path),
                "gate": work_order["human_gate"],
                "risk": "source patch requires review" if status == "ready_for_code_agent" else "external gate; source patch would be premature",
                "no_training_started": True,
            }
        ],
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_patch_work_order",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "status": status,
        "selected_issue_id": issue_id,
        "selected_title": title,
        "evidence": evidence,
        "self_evolution_context": requirement_context,
        "work_order": work_order,
        "action_queue": action_queue,
        "next_safe_commands": [
            "evomind patch-order",
            "evomind ask --json \"复核当前系统智能体闭环，不启动训练\"",
            "evomind self-audit",
        ],
        "artifact_path": str(artifact_path),
        "action_queue_path": str(action_queue_path),
        "trials_path": str(trials_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "source_code_changes": "blocked_until_code_agent_or_human_review_applies_patch",
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    lesson = {
        "ts": generated_at,
        "source": "scientist_patch_work_order",
        "selected_issue_id": issue_id,
        "status": status,
        "lesson": (
            f"Patch work order created for {issue_id}; files={len(files_to_edit)}; checks={len(acceptance_checks)}."
            if status == "ready_for_code_agent"
            else f"Patch work order blocked by external gate; safe_next={safe_next_command}."
        ),
        "no_training_started": True,
    }
    try:
        xsci.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        action_queue_path.write_text(json.dumps(action_queue, ensure_ascii=False, indent=2), encoding="utf-8")
        with trials_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson, ensure_ascii=False) + "\n")
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write patch work order artifacts: {exc}"

    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"patch_order_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_patch_work_order",
            "task": session.selected_task or "",
            "phase": "patch_work_order",
            "status": "passed" if payload.get("ok", True) else "blocked",
            "tool": "scientist_patch_work_order",
            "message": f"issue={issue_id}; status={status}; files={len(files_to_edit)}",
            "artifact_path": str(artifact_path),
            "details": {
                "selected_issue_id": issue_id,
                "files_to_edit": files_to_edit,
                "acceptance_checks": acceptance_checks[:5],
                "open_requirements": requirement_context.get("open_requirements", []),
                "blocked_requirements": requirement_context.get("blocked_requirements", []),
            },
            "no_training_started": True,
        })
    except Exception:
        pass
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_patch_work_order",
            "user": "scientist_patch_work_order",
            "forced_tools": ["scientist_repair_plan", "scientist_self_audit", "scientist_turn_plan"],
            "executed_tools": [{"tool": "scientist_patch_work_order", "ok": payload.get("ok", True)}],
            "mode": status,
            "decision": {
                "selected_action": "create_patch_work_order",
                "selected_issue_id": issue_id,
                "code_generation_mode": "work_order_only",
            },
            "blockers": [] if status == "ready_for_code_agent" else ["external_gate_not_code_patch"],
            "next_actions": payload["next_safe_commands"],
            "artifacts": [str(artifact_path), str(action_queue_path), str(trials_path)],
            "answer_preview": f"patch work order issue={issue_id}; status={status}; no_training_started=True",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        })
    except Exception:
        pass
    return payload


def get_scientist_recovery_snapshot(session: SessionState, root: Path) -> dict[str, Any]:
    """Build a durable recovery snapshot for long-running Scientist work.

    This is the terminal-facing recovery bridge: it gathers the persistent
    guard, recent scientist turns, recent step trace, current action queue, and
    latest planning artifacts into one bounded JSON file.  It never starts
    training and never reads secrets.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_path = root / ".xsci" / "scientist_recovery_snapshot.json"
    guard_path = root / ".xsci" / "recovery_guard.md"

    try:
        from .recovery_guard import RecoveryGuard, build_compaction_recovery_block
        guard = RecoveryGuard(guard_path)
        emitted_guard = guard.emit(session, event="ScientistRecoverySnapshot")
        recovery_block = build_compaction_recovery_block(emitted_guard or guard_path)
    except Exception as exc:  # pragma: no cover - defensive only
        recovery_block = ""
        emitted_guard = guard_path
        guard_error = type(exc).__name__
    else:
        guard_error = ""

    try:
        from .scientist_turns import load_recent_scientist_turns
        recent_turns = load_recent_scientist_turns(root, limit=8)
    except Exception:
        recent_turns = []

    try:
        from .scientist_trace import load_recent_scientist_step_events, record_scientist_step_event
        recent_steps = load_recent_scientist_step_events(root, limit=16)
    except Exception:
        recent_steps = []
        record_scientist_step_event = None  # type: ignore[assignment]

    def read_json_artifact(name: str) -> dict[str, Any] | None:
        path = root / ".xsci" / name
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    latest_loop = read_json_artifact("scientist_loop.json")
    latest_next_action = read_json_artifact("scientist_next_action.json")
    latest_workplan = read_json_artifact("scientist_workplan.json")
    latest_repair_plan = read_json_artifact("scientist_repair_plan.json")
    latest_contract = read_json_artifact("scientist_execution_contract.json")
    action_queue = get_scientist_action_queue(session, root)

    blockers: list[str] = []
    for source in (latest_loop, latest_contract, latest_repair_plan, action_queue):
        if not isinstance(source, dict):
            continue
        if isinstance(source.get("blockers"), list):
            blockers.extend(str(item) for item in source.get("blockers")[:8])
        if source.get("go_no_go") == "no_go":
            blockers.append("execution_contract=no_go")
        if source.get("current_blocker"):
            blockers.append(str(source.get("current_blocker")))
    if session.gpu_blocked and session.gpu_blocker:
        blockers.append(f"gpu_blocked: {session.gpu_blocker}")
    blockers = list(dict.fromkeys(item for item in blockers if item))

    ready_actions = [
        action for action in (action_queue.get("actions") or [])
        if isinstance(action, dict) and str(action.get("status") or "") == "ready"
    ]
    selected_action = ready_actions[0] if ready_actions else None
    resume_commands = [
        "evomind recovery",
        "evomind loop",
        "evomind next",
        "evomind live",
    ]
    if selected_action and selected_action.get("command"):
        resume_commands.insert(2, str(selected_action.get("command")))

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "scientist_recovery",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "last_goal": session.last_goal or "",
        "workspace_root": str(root),
        "guard_path": str(emitted_guard or guard_path),
        "guard_error": guard_error,
        "recovery_block_preview": recovery_block[:4000],
        "recent_turn_count": len(recent_turns),
        "recent_step_count": len(recent_steps),
        "recent_turns": recent_turns,
        "recent_steps": recent_steps,
        "latest_loop": {
            "mode": latest_loop.get("mode") if isinstance(latest_loop, dict) else "",
            "stop_reason": latest_loop.get("stop_reason") if isinstance(latest_loop, dict) else "",
            "trace_run_id": latest_loop.get("trace_run_id") if isinstance(latest_loop, dict) else "",
            "artifact_path": latest_loop.get("artifact_path") if isinstance(latest_loop, dict) else "",
        },
        "latest_next_action": latest_next_action,
        "latest_workplan_artifact": latest_workplan.get("artifact_path") if isinstance(latest_workplan, dict) else str(root / ".xsci" / "scientist_workplan.json"),
        "latest_repair_artifact": latest_repair_plan.get("artifact_path") if isinstance(latest_repair_plan, dict) else str(root / ".xsci" / "scientist_repair_plan.json"),
        "latest_contract_artifact": latest_contract.get("artifact_path") if isinstance(latest_contract, dict) else str(root / ".xsci" / "scientist_execution_contract.json"),
        "action_queue_artifact": action_queue.get("artifact_path"),
        "selected_resume_action": selected_action,
        "blockers": blockers,
        "resume_commands": list(dict.fromkeys(resume_commands)),
        "recovery_decision": (
            "blocked_clear_gates" if blockers else
            "resume_from_selected_action" if selected_action else
            "refresh_scientist_loop"
        ),
        "artifact_path": str(artifact_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(artifact_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write recovery snapshot: {exc}"

    if record_scientist_step_event is not None:
        try:
            record_scientist_step_event(root, {
                "trace_run_id": f"recovery_{generated_at.replace(':', '').replace('+', 'Z')}",
                "source": "scientist_recovery",
                "task": session.selected_task or "",
                "phase": "recovery_snapshot",
                "status": "passed" if payload.get("ok", True) else "blocked",
                "tool": "scientist_recovery",
                "message": f"Recovery snapshot ready; decision={payload.get('recovery_decision')}",
                "artifact_path": str(artifact_path),
                "details": {
                    "recent_turn_count": len(recent_turns),
                    "recent_step_count": len(recent_steps),
                    "selected_resume_action": selected_action,
                },
                "no_training_started": True,
            })
        except Exception:
            pass
    return payload


def get_scientist_repair_plan(session: SessionState, root: Path) -> dict[str, Any]:
    """Return and persist the read-only self-repair plan."""
    from .scientist_state import build_scientist_repair_plan

    return build_scientist_repair_plan(session, root, persist=True)


def get_scientist_execution_contract(session: SessionState, root: Path) -> dict[str, Any]:
    """Return and persist the read-only pre-execution contract."""
    from .scientist_state import build_scientist_execution_contract

    return build_scientist_execution_contract(session, root, persist=True)


def _build_scientist_action_queue(
    *,
    session: SessionState,
    root: Path,
    decision: dict[str, Any],
    workplan_result: dict[str, Any],
    repair_result: dict[str, Any],
    contract_result: dict[str, Any],
    hypothesis_review: dict[str, Any] | None,
    experiment_blueprint: dict[str, Any] | None,
    blockers: list[str],
    data_ready: bool,
    can_execute: bool,
) -> list[dict[str, Any]]:
    """Build a Claude-Code-like action queue from the Scientist artifacts.

    The queue is still advisory and gated.  It gives the terminal/frontend a
    precise next-command surface without starting training or submitting.
    """
    task = session.selected_task or ""
    go_no_go = str(contract_result.get("go_no_go") or "")
    rollback = str(contract_result.get("rollback_condition") or decision.get("rollback_condition") or "hold if gates fail")
    required_artifacts = contract_result.get("required_artifacts") or []
    if not isinstance(required_artifacts, list):
        required_artifacts = []
    hypothesis_review = hypothesis_review if isinstance(hypothesis_review, dict) else {}
    selected_hypothesis = hypothesis_review.get("selected_hypothesis")
    if not isinstance(selected_hypothesis, dict):
        selected_hypothesis = {}
    hypothesis_strategy = str(selected_hypothesis.get("strategy_name") or selected_hypothesis.get("hypothesis_id") or "")
    hypothesis_branch = str(selected_hypothesis.get("branch_type") or "")
    hypothesis_mode = str(selected_hypothesis.get("code_generation_mode") or "")
    hypothesis_score = selected_hypothesis.get("score")
    hypothesis_status = str(selected_hypothesis.get("status") or "")
    hypothesis_review_artifact = str(hypothesis_review.get("artifact_path") or root / ".xsci" / "scientist_hypothesis_review.json")
    experiment_blueprint = experiment_blueprint if isinstance(experiment_blueprint, dict) else {}
    blueprint = experiment_blueprint.get("experiment_blueprint") if isinstance(experiment_blueprint.get("experiment_blueprint"), dict) else {}
    blueprint_status = str(experiment_blueprint.get("blueprint_status") or "")
    blueprint_artifact = str(experiment_blueprint.get("artifact_path") or root / ".xsci" / "scientist_experiment_blueprint.json")
    memory_reuse_plan = (
        blueprint.get("memory_reuse_plan")
        if isinstance(blueprint.get("memory_reuse_plan"), dict)
        else experiment_blueprint.get("memory_reuse_plan") if isinstance(experiment_blueprint.get("memory_reuse_plan"), dict)
        else selected_hypothesis.get("memory_reuse_plan") if isinstance(selected_hypothesis.get("memory_reuse_plan"), dict)
        else {}
    )
    queue: list[dict[str, Any]] = []

    def item(
        action_id: str,
        title: str,
        command: str,
        *,
        status: str,
        gate: str,
        why: str,
        evidence: list[str],
        risk: str,
        rollback_condition: str = "",
        expected_artifacts: list[str] | None = None,
        autonomy: str = "requires_user_command",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        queue.append({
            "id": action_id,
            "title": title,
            "status": status,
            "command": command,
            "gate": gate,
            "why": why,
            "risk": risk,
            "rollback_condition": rollback_condition or rollback,
            "expected_artifacts": expected_artifacts or [],
            "evidence": evidence,
            "autonomy": autonomy,
            "metadata": metadata or {},
        })

    if not task:
        item(
            "select_task",
            "Select or register a Kaggle/MLE-Bench task",
            "evomind task add <kaggle-url>",
            status="ready",
            gate="task_required",
            why="No task is selected, so EvoMind cannot reason about metric, data, or branch choice.",
            risk="none; this is metadata setup only",
            rollback_condition="no execution without selected_task",
            evidence=["task_list", "task config"],
            autonomy="read_only_until_user_supplies_task",
        )
        item(
            "inspect_system",
            "Inspect installation and resource readiness",
            "evomind ready",
            status="ready",
            gate="setup_gate",
            why="Model, Kaggle, and compute state determine which research actions are allowed.",
            risk="none; read-only readiness check",
            rollback_condition="stay in planner mode while setup blockers remain",
            evidence=["system_status", "kaggle_status", "gpu_status"],
            autonomy="read_only",
        )
        return queue

    if not selected_hypothesis:
        item(
            "review_hypotheses",
            "Review proposed hypotheses before execution",
            "evomind review-hypotheses",
            status="ready",
            gate="hypothesis_review_gate",
            why="EvoMind has not ranked proposal branches against evidence, readiness, impact, and risk yet.",
            risk="running without a reviewed hypothesis may waste compute or repeat weak branches",
            rollback_condition="stay in proposal mode until scientist_hypothesis_review exists",
            expected_artifacts=[hypothesis_review_artifact, str(root / ".xsci" / "scientist_innovation_backlog.json")],
            evidence=["scientist_innovation_backlog", "scientist_hypothesis_review"],
            autonomy="read_only",
        )
    elif not blueprint:
        item(
            "prepare_experiment_blueprint",
            "Prepare gated experiment blueprint from reviewed hypothesis",
            "evomind blueprint",
            status="ready",
            gate="experiment_blueprint_gate",
            why=f"The reviewed hypothesis {hypothesis_strategy} must be translated into branch/code/resource/artifact gates before any run.",
            risk="without a blueprint, the next run lacks explicit artifacts, rollback, and memory writeback criteria",
            rollback_condition="stay read-only until scientist_experiment_blueprint exists",
            expected_artifacts=[blueprint_artifact, hypothesis_review_artifact],
            evidence=["scientist_hypothesis_review", "scientist_experiment_blueprint"],
            autonomy="read_only",
            metadata={"selected_hypothesis": selected_hypothesis},
        )

    if blockers:
        repair_safe_next = str(repair_result.get("safe_next_command") or "evomind ready")
        safe_next = repair_safe_next
        if " ".join(repair_safe_next.strip().split()).lower() in {"evomind ready", "evomind status"}:
            safe_next = "evomind repair"
        item(
            "clear_blockers",
            "Diagnose and clear setup/data gates before execution",
            safe_next,
            status="ready",
            gate="setup_or_data_gate",
            why="Blocking gates were found: " + "; ".join(blockers[:3]),
            risk="training would be non-reproducible or fail before artifacts are produced",
            rollback_condition="do not call AgentSession until blockers disappear",
            expected_artifacts=[str(root / ".xsci" / "scientist_repair_plan.json")],
            evidence=["scientist_repair_plan", "scientist_execution_contract"],
            autonomy="read_only_repair_guidance",
            metadata={
                **({"selected_hypothesis": selected_hypothesis} if selected_hypothesis else {}),
                "repair_safe_next_command": repair_safe_next,
                "diagnostic_escalation": safe_next != repair_safe_next,
            },
        )
    elif not data_ready:
        item(
            "prepare_data_contract",
            "Register/download train/test data and schema",
            f"evomind download {task}",
            status="ready",
            gate="data_gate",
            why="Core setup is usable, but the task does not have a trustworthy data contract yet.",
            risk="starting training without train/test/schema evidence can produce invalid claims",
            rollback_condition="stay in planner mode until data_check and validation_contract pass",
            expected_artifacts=["data_check", "validation_contract"],
            evidence=["data_check", "scientist_checkpoint"],
            autonomy="requires_user_command",
            metadata={"selected_hypothesis": selected_hypothesis} if selected_hypothesis else {},
        )
    elif go_no_go == "no_go":
        safe_next = str(repair_result.get("safe_next_command") or "evomind repair")
        item(
            "repair_no_go_contract",
            "Repair no-go execution contract",
            safe_next,
            status="ready",
            gate="execution_contract_gate",
            why="The execution contract returned no_go; training is blocked by policy.",
            risk="bypassing the contract would break auditability and best-so-far protection",
            rollback_condition="do not execute until go_no_go becomes go or conditional_go",
            expected_artifacts=[str(root / ".xsci" / "scientist_execution_contract.json")],
            evidence=["scientist_execution_contract", "scientist_repair_plan"],
            autonomy="read_only_repair_guidance",
            metadata={"selected_hypothesis": selected_hypothesis} if selected_hypothesis else {},
        )
    elif can_execute:
        action = str(decision.get("selected_action") or "run_audited_baseline")
        branch = hypothesis_branch or str(decision.get("selected_branch") or "baseline")
        code_mode = hypothesis_mode or str(decision.get("code_generation_mode") or "Base")
        title = "Run the reviewed workstation candidate" if selected_hypothesis else "Run the next audited workstation candidate"
        why = (
            f"Hypothesis review selected {hypothesis_strategy} "
            f"(score={hypothesis_score}, status={hypothesis_status}, branch={branch}, code_generation_mode={code_mode}); "
            f"blueprint_status={blueprint_status or 'not_generated'}."
            if selected_hypothesis else
            f"Decision layer selected action={action}, branch={branch}, code_generation_mode={code_mode}."
        )
        if isinstance(memory_reuse_plan, dict) and (
            memory_reuse_plan.get("reuse_rules") or memory_reuse_plan.get("avoid_patterns")
        ):
            rule_count = len(memory_reuse_plan.get("reuse_rules") or [])
            avoid_count = len(memory_reuse_plan.get("avoid_patterns") or [])
            item(
                "apply_memory_reuse_plan",
                "Apply retrospective memory before candidate execution",
                "evomind blueprint",
                status="applied",
                gate="memory_reuse_gate",
                why=(
                    "Experiment blueprint already embeds memory_reuse_plan "
                    f"with {rule_count} reusable rules and {avoid_count} avoid patterns."
                ),
                risk="candidate may repeat a known failure pattern if this plan is removed",
                rollback_condition="hold candidate if validation or claim audit shows a remembered failure recurred",
                expected_artifacts=[
                    blueprint_artifact,
                    str(root / "experiments" / "evolution" / "retrospective_memory.json"),
                ],
                evidence=[
                    str(root / "experiments" / "evolution" / "retrospective_memory.json"),
                    str(root / ".xsci" / "scientist_innovation_backlog.json"),
                    hypothesis_review_artifact,
                    blueprint_artifact,
                ],
                autonomy="read_only",
                metadata={
                    "memory_reuse_plan": memory_reuse_plan,
                    "selected_hypothesis": selected_hypothesis,
                },
            )
        item(
            "run_gated_candidate",
            title,
            f"evomind run {task}",
            status="ready",
            gate="human_run_command_required",
            why=why,
            risk="candidate may hold/fail; best-so-far must remain protected",
            rollback_condition=rollback,
            expected_artifacts=[str(x) for x in required_artifacts] + ([hypothesis_review_artifact, blueprint_artifact] if selected_hypothesis else []),
            evidence=[
                str(root / ".xsci" / "scientist_decision.json"),
                str(root / "experiments" / "evolution" / "retrospective_memory.json"),
                hypothesis_review_artifact,
                blueprint_artifact,
                str(root / ".xsci" / "scientist_execution_contract.json"),
                str(root / ".xsci" / "scientist_workplan.json"),
            ],
            autonomy="requires_user_run_command",
            metadata={
                "selected_hypothesis": selected_hypothesis,
                "experiment_blueprint": blueprint,
                "memory_reuse_plan": memory_reuse_plan,
            } if selected_hypothesis else {},
        )
    else:
        safe_next = str(repair_result.get("safe_next_command") or "evomind workplan")
        item(
            "refresh_plan",
            "Refresh Scientist plan before execution",
            safe_next,
            status="ready",
            gate="planner_gate",
            why="The current state is inconclusive; refresh the workplan and contract before spending compute.",
            risk="unclear readiness can waste GPU/Kaggle budget",
            rollback_condition="stay read-only until plan and contract agree",
            expected_artifacts=[str(root / ".xsci" / "scientist_workplan.json")],
            evidence=["scientist_workplan", "scientist_execution_contract"],
            autonomy="read_only",
            metadata={"selected_hypothesis": selected_hypothesis} if selected_hypothesis else {},
        )

    if selected_hypothesis and blueprint:
        feedback_gate_status = (
            "ready_for_gated_execution"
            if blueprint_status == "ready_for_gated_execution" and go_no_go != "no_go" and not blockers
            else "blocked_by_gate"
        )
        feedback_trial_id = _stable_memory_id(
            "innovation_trial",
            task,
            selected_hypothesis.get("hypothesis_id") or blueprint.get("hypothesis_id") or selected_hypothesis.get("id") or "",
            blueprint.get("blueprint_id") or "",
            feedback_gate_status,
        )
        innovation_log = _read_json_artifact(root / ".xsci" / "innovation_log.json") or {}
        tried_items = innovation_log.get("tried") if isinstance(innovation_log.get("tried"), list) else []
        feedback_exists = any(
            isinstance(entry, dict) and str(entry.get("trial_id") or "") == feedback_trial_id
            for entry in tried_items
        )
        item(
            "record_innovation_trial_feedback",
            "Record innovation gate feedback for memory reuse",
            "evomind innovation-feedback",
            status="applied" if feedback_exists else "ready",
            gate="innovation_feedback_gate",
            why=(
                "The reviewed hypothesis and experiment blueprint can be written back into innovation "
                "memory as a gate outcome after the immediate setup/data/execution gate is visible."
            ),
            risk="without this feedback, EvoMind proposes ideas but cannot prove it learned from gate outcomes",
            rollback_condition="rerun innovation-feedback after regenerating hypothesis review or blueprint",
            expected_artifacts=[
                str(root / ".xsci" / "scientist_innovation_trial_feedback.json"),
                str(root / ".xsci" / "innovation_log.json"),
            ],
            evidence=["scientist_hypothesis_review", "scientist_experiment_blueprint", "scientist_execution_contract"],
            autonomy="read_only",
            metadata={
                "trial_id": feedback_trial_id,
                "feedback_exists": feedback_exists,
                "selected_hypothesis": selected_hypothesis,
                "experiment_blueprint": blueprint,
            },
        )

    item(
        "watch_evidence",
        "Watch live evidence and step trace",
        "evomind live",
        status="ready",
        gate="observability_gate",
        why="The user and frontend need proof of every tool call, gate, artifact, and claim boundary.",
        risk="without trace evidence, a run cannot be audited or safely claimed",
        rollback_condition="if trace is missing, regenerate autopilot/workplan before execution",
        expected_artifacts=[str(root / ".xsci" / "scientist_step_trace.jsonl")],
        evidence=["scientist_step_trace", "terminal_events.jsonl"],
        autonomy="read_only",
    )
    item(
        "preserve_claim_boundary",
        "Keep official submit and medal/rank claims blocked",
        "evomind report",
        status="blocked_until_human_gate",
        gate="claim_audit_and_human_submit_gate",
        why="Official Kaggle score/rank/medal claims require a Kaggle response artifact.",
        risk="overclaiming would invalidate the benchmark report",
        rollback_condition="clear any rank/medal field unless official response artifact exists",
        expected_artifacts=["claim_audit", "kaggle_submission_response.json only after approval"],
        evidence=["claim_audit", "submission_audit", "human_gate"],
        autonomy="human_gate_only",
    )
    return queue


def _load_latest_requirement_ledger(root: Path) -> tuple[dict[str, Any], str]:
    """Return the newest AI Scientist requirement ledger, if one exists."""
    xsci = Path(root) / ".xsci"
    for artifact_name in ("scientist_terminal_turn.json", "scientist_turn_plan.json"):
        artifact_path = xsci / artifact_name
        payload = _read_json_artifact(artifact_path)
        if not isinstance(payload, dict):
            continue
        ledger = payload.get("requirement_ledger")
        if not isinstance(ledger, dict):
            continue
        if str(ledger.get("schema") or "") != "evomind.ai_scientist.requirement_ledger.v1":
            continue
        return ledger, str(artifact_path)
    return {}, ""


def _requirement_ledger_fingerprint(ledger: dict[str, Any]) -> str:
    if not isinstance(ledger, dict):
        return ""
    compact = {
        "goal": str(ledger.get("goal") or "")[:300],
        "open": [str(item) for item in (ledger.get("open_requirements") or [])],
        "blocked": [str(item) for item in (ledger.get("blocked_requirements") or [])],
        "requirements": [
            {
                "id": str(item.get("id") or ""),
                "status": str(item.get("status") or ""),
                "gate": str(item.get("gate") or ""),
                "reason": str(item.get("reason") or "")[:300],
            }
            for item in (ledger.get("requirements") or [])
            if isinstance(item, dict)
        ],
    }
    return hashlib.sha256(
        json.dumps(compact, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="replace")
    ).hexdigest()[:16]


def _requirement_action_spec(req_id: str, status: str) -> dict[str, str] | None:
    """Map an unsatisfied requirement to a safe next-action command."""
    table: dict[str, dict[str, str]] = {
        "setup_gate_clearance": {
            "id": "clear_blockers",
            "title": "Resolve blocked setup gates from the requirement ledger",
            "command": "evomind repair",
            "gate": "setup_gate",
            "autonomy": "read_only_repair_guidance",
        },
        "advisory_setup_review": {
            "id": "review_advisory_setup_gaps",
            "title": "Inspect advisory setup gaps before continuing",
            "command": "evomind ready",
            "gate": "advisory_gate",
            "autonomy": "read_only",
        },
        "situation_model": {
            "id": "refresh_situation_model",
            "title": "Refresh the Scientist situation model",
            "command": "evomind situation",
            "gate": "observe_orient_gate",
            "autonomy": "read_only",
        },
        "recoverable_workplan": {
            "id": "refresh_workplan",
            "title": "Write a recoverable Scientist workplan",
            "command": "evomind workplan",
            "gate": "workplan_gate",
            "autonomy": "read_only",
        },
        "execution_contract": {
            "id": "refresh_execution_contract",
            "title": "Refresh the pre-execution contract",
            "command": "evomind contract",
            "gate": "execution_contract_gate",
            "autonomy": "read_only",
        },
        "data_and_validation_contract": {
            "id": "check_data_validation_contract",
            "title": "Check data and validation contract readiness",
            "command": "evomind data-check",
            "gate": "data_validation_gate",
            "autonomy": "read_only",
        },
        "memory_guided_hypotheses": {
            "id": "generate_memory_guided_hypotheses",
            "title": "Generate memory-guided research hypotheses",
            "command": "evomind innovation-backlog",
            "gate": "memory_reuse_gate",
            "autonomy": "read_only",
        },
        "hypothesis_review": {
            "id": "review_hypotheses",
            "title": "Review hypotheses against evidence and risk",
            "command": "evomind review-hypotheses",
            "gate": "hypothesis_review_gate",
            "autonomy": "read_only",
        },
        "agent_self_audit": {
            "id": "satisfy_agent_self_audit",
            "title": "Run capability self-audit before claiming Scientist parity",
            "command": "evomind self-audit",
            "gate": "capability_audit_gate",
            "autonomy": "read_only",
        },
        "memory_consolidation": {
            "id": "satisfy_memory_consolidation",
            "title": "Consolidate recent lessons into durable memory",
            "command": "evomind memory-consolidation",
            "gate": "memory_writeback_gate",
            "autonomy": "read_only",
        },
        "parity_lifecycle": {
            "id": "refresh_parity_lifecycle",
            "title": "Refresh observe-plan-act-reflect-improve lifecycle evidence",
            "command": "evomind turn-plan",
            "gate": "scientist_parity_gate",
            "autonomy": "read_only",
        },
    }
    spec = table.get(req_id)
    if not spec:
        return None
    if req_id == "setup_gate_clearance" and status != "blocked":
        return None
    return spec


def _requirement_action_priority(req_id: str, status: str) -> tuple[int, int, str]:
    order = {
        "setup_gate_clearance": 0,
        "advisory_setup_review": 1,
        "situation_model": 2,
        "agent_self_audit": 3,
        "memory_consolidation": 4,
        "recoverable_workplan": 5,
        "parity_lifecycle": 6,
        "execution_contract": 7,
        "data_and_validation_contract": 8,
        "memory_guided_hypotheses": 9,
        "hypothesis_review": 10,
    }
    blocked_bonus = 0 if status == "blocked" else 1
    return (order.get(req_id, 99), blocked_bonus, req_id)


def _requirement_driven_actions(
    root: Path,
    ledger: dict[str, Any],
    source_artifact: str,
    ledger_fingerprint: str,
) -> list[dict[str, Any]]:
    """Create safe queue items for requirements that remain unsatisfied."""
    if not isinstance(ledger, dict):
        return []
    requirements = [
        item for item in (ledger.get("requirements") or [])
        if isinstance(item, dict) and str(item.get("status") or "") != "satisfied"
    ]
    requirements.sort(key=lambda item: _requirement_action_priority(
        str(item.get("id") or ""),
        str(item.get("status") or ""),
    ))
    actions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for requirement in requirements:
        req_id = str(requirement.get("id") or "")
        status = str(requirement.get("status") or "pending")
        spec = _requirement_action_spec(req_id, status)
        if not spec:
            continue
        action_id = spec["id"]
        if action_id in seen_ids:
            continue
        seen_ids.add(action_id)
        evidence_needed = [str(item) for item in (requirement.get("evidence_needed") or []) if str(item)]
        expected_artifacts = [
            item for item in evidence_needed
            if item.startswith(".xsci") or "/" in item or "\\" in item
        ]
        actions.append({
            "id": action_id,
            "title": spec["title"],
            "status": "ready",
            "command": spec["command"],
            "gate": spec["gate"],
            "why": (
                f"Requirement `{req_id}` is {status}: "
                f"{str(requirement.get('reason') or requirement.get('description') or 'needs evidence')[:260]}"
            ),
            "risk": "read-only requirement closure; no training or official submission is started",
            "rollback_condition": "if the tool cannot satisfy the requirement, keep the gate blocked and record the artifact",
            "expected_artifacts": expected_artifacts,
            "evidence": evidence_needed,
            "autonomy": spec["autonomy"],
            "metadata": {
                "source": "requirement_ledger",
                "requirement_id": req_id,
                "requirement_status": status,
                "requirement_gate": str(requirement.get("gate") or ""),
                "ledger_artifact_path": source_artifact,
                "ledger_fingerprint": ledger_fingerprint,
                "mapped_tools": [str(tool) for tool in (requirement.get("mapped_tools") or []) if str(tool)],
            },
        })
    return actions


def _apply_requirement_ledger_to_action_queue(
    payload: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Prepend requirement-driven safe actions to the queue.

    Old requirement-driven actions are removed first, so the queue always
    reflects the newest ledger rather than stale unresolved items.
    """
    if not isinstance(payload, dict):
        return payload
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    base_actions = [
        action for action in actions
        if not (
            isinstance(action, dict)
            and isinstance(action.get("metadata"), dict)
            and action["metadata"].get("source") == "requirement_ledger"
        )
    ]
    ledger, source_artifact = _load_latest_requirement_ledger(root)
    ledger_fingerprint = _requirement_ledger_fingerprint(ledger)
    requirement_actions = _requirement_driven_actions(root, ledger, source_artifact, ledger_fingerprint)
    if not requirement_actions:
        payload["actions"] = base_actions
        payload.pop("requirement_ledger_summary", None)
        return payload

    requirement_action_ids = {str(action.get("id") or "") for action in requirement_actions}
    merged_actions = requirement_actions + [
        action for action in base_actions
        if not (isinstance(action, dict) and str(action.get("id") or "") in requirement_action_ids)
    ]
    payload["actions"] = merged_actions
    payload["requirement_ledger_summary"] = {
        "source_artifact": source_artifact,
        "ledger_fingerprint": ledger_fingerprint,
        "open_requirements": [str(item) for item in (ledger.get("open_requirements") or [])],
        "blocked_requirements": [str(item) for item in (ledger.get("blocked_requirements") or [])],
        "requirement_driven_action_ids": [str(item.get("id") or "") for item in requirement_actions],
    }
    return payload


def get_scientist_action_queue(session: SessionState, root: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Return the current action queue, generating it read-only if missing."""
    root = Path(root)
    artifact_path = root / ".xsci" / "scientist_action_queue.json"
    payload: dict[str, Any] | None = None
    if artifact_path.exists() and not refresh:
        try:
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                loaded_task = str(loaded.get("selected_task") or "")
                current_task = session.selected_task or ""
                if loaded_task == current_task:
                    payload = loaded
        except Exception:
            payload = None
    if payload is None:
        autopilot = run_scientist_autopilot(session, root)
        payload = {
            "ok": True,
            "tool": "scientist_action_queue",
            "trace_run_id": autopilot.get("trace_run_id", ""),
            "selected_task": session.selected_task or "",
            "actions": autopilot.get("action_queue") or [],
            "artifact_path": str(artifact_path),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
            "generated_from": "scientist_autopilot",
        }
    payload = _apply_requirement_ledger_to_action_queue(payload, root=root)
    payload = _apply_scientist_continuation_to_action_queue(payload, root=root)
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(artifact_path)
    except OSError:
        pass
    return {
        "present": True,
        "artifact_path": str(artifact_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        **payload,
    }


def _recent_requirement_attempts(root: Path, *, limit: int = 30) -> set[tuple[str, str]]:
    """Return recently executed requirement actions as (fingerprint, req_id)."""
    attempts: set[tuple[str, str]] = set()

    def inspect_payload(payload: dict[str, Any]) -> None:
        status = str(payload.get("status") or "")
        if status != "executed_read_only_tool":
            return
        selected = payload.get("selected_action")
        if not isinstance(selected, dict):
            return
        metadata = selected.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("source") != "requirement_ledger":
            return
        req_id = str(metadata.get("requirement_id") or "")
        fingerprint = str(metadata.get("ledger_fingerprint") or "")
        if req_id:
            attempts.add((fingerprint, req_id))

    latest_next = _read_json_artifact(Path(root) / ".xsci" / "scientist_next_action.json")
    if isinstance(latest_next, dict):
        inspect_payload(latest_next)
    for event in _read_jsonl_tail(Path(root) / ".xsci" / "scientist_step_trace.jsonl", limit=limit):
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        selected = details.get("selected_action")
        if not isinstance(selected, dict):
            continue
        inspect_payload({
            "status": event.get("status"),
            "selected_action": selected,
        })
    return attempts


def _is_recently_attempted_requirement_action(action: dict[str, Any], attempts: set[tuple[str, str]]) -> bool:
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    if metadata.get("source") != "requirement_ledger":
        return False
    req_id = str(metadata.get("requirement_id") or "")
    fingerprint = str(metadata.get("ledger_fingerprint") or "")
    return bool(req_id and ((fingerprint, req_id) in attempts or ("", req_id) in attempts))


def _recompute_requirement_ledger_indexes(ledger: dict[str, Any]) -> dict[str, Any]:
    requirements = [
        item for item in (ledger.get("requirements") or [])
        if isinstance(item, dict)
    ]
    ledger["requirements"] = requirements
    ledger["satisfied_requirements"] = [
        str(item.get("id") or "")
        for item in requirements
        if item.get("status") == "satisfied" and str(item.get("id") or "")
    ]
    ledger["open_requirements"] = [
        str(item.get("id") or "")
        for item in requirements
        if item.get("status") != "satisfied" and str(item.get("id") or "")
    ]
    ledger["blocked_requirements"] = [
        str(item.get("id") or "")
        for item in requirements
        if item.get("status") == "blocked" and str(item.get("id") or "")
    ]
    ledger["next_evidence_to_collect"] = [
        evidence
        for item in requirements
        if item.get("status") != "satisfied"
        for evidence in (item.get("evidence_needed") or [])[:2]
    ][:10]
    return ledger


def _requirement_progress_status(
    *,
    req_id: str,
    current_status: str,
    safe_tool: str,
    tool_result: dict[str, Any],
) -> tuple[str, str]:
    ok = bool(tool_result.get("ok", True))
    artifact_path = str(tool_result.get("artifact_path") or tool_result.get("path") or "")
    if not ok:
        return ("blocked" if current_status == "blocked" else "pending", str(tool_result.get("message") or "Tool did not complete."))
    if req_id == "setup_gate_clearance":
        blockers = tool_result.get("blockers") if isinstance(tool_result.get("blockers"), list) else []
        root_causes = tool_result.get("root_causes") if isinstance(tool_result.get("root_causes"), list) else []
        gpu_blocked = bool(tool_result.get("gpu_blocked")) or "gpu_blocked" in {str(item) for item in root_causes}
        if blockers or gpu_blocked or str(tool_result.get("mode") or "") == "blocked_repair":
            return (
                "blocked",
                "Requirement was diagnosed by a safe next action, but the external setup gate is still blocked.",
            )
        return ("satisfied", "Setup gate has no remaining blockers in the latest safe tool result.")
    if req_id == "data_and_validation_contract":
        if tool_result.get("train_csv") or tool_result.get("data_contract_status") == "ready":
            return ("satisfied", f"Closed by {safe_tool}; data/validation evidence is present.")
        return ("pending", f"{safe_tool} ran, but data/validation evidence is not complete yet.")
    if artifact_path or safe_tool:
        return ("satisfied", f"Closed by safe next action via {safe_tool}.")
    return ("pending", f"{safe_tool} ran, but no closing evidence was produced.")


def _persist_requirement_progress_after_next_action(
    root: Path,
    *,
    selected: dict[str, Any],
    safe_tool: str,
    tool_result: dict[str, Any],
    generated_at: str,
    selected_task: str = "",
) -> dict[str, Any]:
    metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
    if metadata.get("source") != "requirement_ledger":
        return {}
    req_id = str(metadata.get("requirement_id") or "")
    if not req_id:
        return {}
    ledger, source_artifact = _load_latest_requirement_ledger(root)
    if not ledger:
        return {}
    before_fingerprint = _requirement_ledger_fingerprint(ledger)
    artifact_path = str(tool_result.get("artifact_path") or tool_result.get("path") or "")
    updated_ledger = json.loads(json.dumps(ledger, ensure_ascii=False))
    updated_requirement: dict[str, Any] | None = None
    before_status = ""
    for requirement in updated_ledger.get("requirements") or []:
        if not isinstance(requirement, dict) or str(requirement.get("id") or "") != req_id:
            continue
        before_status = str(requirement.get("status") or "")
        next_status, reason = _requirement_progress_status(
            req_id=req_id,
            current_status=before_status,
            safe_tool=safe_tool,
            tool_result=tool_result,
        )
        evidence = requirement.get("execution_evidence") if isinstance(requirement.get("execution_evidence"), dict) else {}
        mapped_hits = [str(item) for item in (evidence.get("mapped_tool_hits") or []) if str(item)]
        artifact_hits = [str(item) for item in (evidence.get("artifact_hits") or []) if str(item)]
        if safe_tool and safe_tool not in mapped_hits:
            mapped_hits.append(safe_tool)
        if artifact_path and artifact_path not in artifact_hits:
            artifact_hits.append(artifact_path)
        requirement["status"] = next_status
        requirement["reason"] = reason
        requirement["execution_evidence"] = {
            **evidence,
            "mapped_tool_hits": mapped_hits,
            "artifact_hits": artifact_hits,
            "last_next_action": {
                "ts": generated_at,
                "tool": safe_tool,
                "artifact_path": artifact_path,
                "status": next_status,
            },
        }
        updated_requirement = requirement
        break
    if updated_requirement is None:
        return {}
    updated_ledger = _recompute_requirement_ledger_indexes(updated_ledger)
    progress = {
        "schema": "evomind.ai_scientist.requirement_progress.v1",
        "updated_at": generated_at,
        "selected_task": selected_task,
        "requirement_id": req_id,
        "before_status": before_status,
        "after_status": str(updated_requirement.get("status") or ""),
        "safe_tool": safe_tool,
        "tool_ok": bool(tool_result.get("ok", True)),
        "tool_artifact_path": artifact_path,
        "source_artifact": source_artifact,
        "before_fingerprint": before_fingerprint,
        "after_fingerprint": _requirement_ledger_fingerprint(updated_ledger),
        "open_requirements": updated_ledger.get("open_requirements", []),
        "blocked_requirements": updated_ledger.get("blocked_requirements", []),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    source_path = Path(source_artifact) if source_artifact else None
    if source_path and source_path.exists():
        source_payload = _read_json_artifact(source_path)
        if isinstance(source_payload, dict):
            source_payload["requirement_ledger"] = updated_ledger
            source_payload["requirement_progress"] = progress
            try:
                tmp = source_path.with_suffix(source_path.suffix + ".tmp")
                tmp.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(source_path)
            except OSError:
                pass
    progress_path = Path(root) / ".xsci" / "scientist_requirement_progress.json"
    progress["artifact_path"] = str(progress_path)
    progress["ledger"] = updated_ledger
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(progress_path)
    except OSError:
        pass
    queue_path = Path(root) / ".xsci" / "scientist_action_queue.json"
    queue_payload = _read_json_artifact(queue_path)
    if isinstance(queue_payload, dict):
        refreshed_queue = _apply_requirement_ledger_to_action_queue(queue_payload, root=root)
        try:
            tmp = queue_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(refreshed_queue, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(queue_path)
        except OSError:
            pass
    return progress


def _select_ready_scientist_action(actions: list[Any], *, root: Path | None = None) -> dict[str, Any] | None:
    ready_actions = [
        action for action in actions
        if isinstance(action, dict) and str(action.get("status") or "") == "ready"
    ]
    preferred_actions = [
        action for action in ready_actions
        if str(action.get("id") or "") != "record_innovation_trial_feedback"
    ] or ready_actions
    if root is not None:
        attempts = _recent_requirement_attempts(root)
        not_recently_attempted = [
            action for action in preferred_actions
            if not _is_recently_attempted_requirement_action(action, attempts)
        ]
        if not_recently_attempted:
            return not_recently_attempted[0]
        non_requirement_actions = [
            action for action in preferred_actions
            if not (
                isinstance(action.get("metadata"), dict)
                and action["metadata"].get("source") == "requirement_ledger"
            )
        ]
        if non_requirement_actions:
            return non_requirement_actions[0]
    return next(
        (
            action for action in preferred_actions
        ),
        ready_actions[0] if ready_actions else None,
    )


def _should_refresh_scientist_action_queue(selected: dict[str, Any] | None) -> bool:
    if not isinstance(selected, dict):
        return False
    action_id = str(selected.get("id") or "")
    command = " ".join(str(selected.get("command") or "").strip().split()).lower()
    metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
    return (
        action_id == "clear_blockers"
        and command in {"evomind ready", "evomind status"}
        and not bool(metadata.get("diagnostic_escalation"))
    )


def _safe_read_only_action_tool(command: str) -> str | None:
    """Map a queue command to a read-only terminal tool, if safe."""
    normalized = " ".join((command or "").strip().split()).lower()
    if normalized in {"evomind ready", "evomind status"}:
        return "system_status"
    if normalized in {
        "evomind continuation",
        "evomind continuation-status",
        "evomind continue-status",
        "evomind turn-status",
        "evomind scientist-continuation-status",
    }:
        return "scientist_continuation_status"
    if normalized in {"evomind trace", "evomind steptrace", "evomind steps", "evomind live", "evomind stream", "evomind evidence-stream"}:
        return "scientist_step_trace"
    if normalized in {"evomind workplan", "evomind roadmap", "evomind agenda"}:
        return "scientist_workplan"
    if normalized in {"evomind repair", "evomind fixplan", "evomind self-repair"}:
        return "scientist_repair_plan"
    if normalized in {"evomind contract", "evomind execution-contract", "evomind run-contract", "evomind preflight-contract"}:
        return "scientist_execution_contract"
    if normalized in {"evomind data-check", "evomind data check", "evomind check-data"}:
        return "data_check"
    if normalized in {"evomind self-audit", "evomind scientist-self-audit", "evomind audit-agent"}:
        return "scientist_self_audit"
    if normalized in {"evomind readiness-report", "evomind launch-readiness", "evomind scientist-readiness"}:
        return "scientist_readiness_report"
    if normalized in {"evomind causal-diagnosis", "evomind cause-map", "evomind root-cause-map", "evomind causal-graph"}:
        return "scientist_causal_diagnosis"
    if normalized in {"evomind strategy", "evomind strategy-optimizer", "evomind priority-plan", "evomind intervention-plan", "evomind decision-matrix"}:
        return "scientist_strategy_optimizer"
    if normalized in {"evomind memory-consolidation", "evomind consolidate-memory", "evomind memory", "evomind learn"}:
        return "scientist_memory_consolidation"
    if normalized in {"evomind innovation-backlog", "evomind brainstorm", "evomind propose-hypotheses"}:
        return "scientist_innovation_backlog"
    if normalized in {"evomind turn-plan", "evomind scientist-turn-plan", "evomind parity-lifecycle"}:
        return "scientist_turn_plan"
    if normalized in {"evomind review-hypotheses", "evomind hypothesis-review", "evomind rank-hypotheses", "evomind critique"}:
        return "scientist_hypothesis_review"
    if normalized in {"evomind blueprint", "evomind experiment-blueprint", "evomind plan-experiment", "evomind candidate-blueprint"}:
        return "scientist_experiment_blueprint"
    if normalized in {"evomind innovation-feedback", "evomind trial-feedback", "evomind feedback-innovation", "evomind scientist-feedback"}:
        return "scientist_innovation_trial_feedback"
    if normalized in {"evomind situation", "evomind situation-model", "evomind state-model", "evomind scientist-state", "evomind orient"}:
        return "scientist_situation_model"
    if normalized in {"evomind autopilot", "evomind diagnose"}:
        return "scientist_autopilot"
    if normalized in {"evomind report"}:
        return "scientist_step_trace"
    return None


def _sanitize_continuation_progress(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "updated_at": _redacted_memory_text(item.get("updated_at") or "", limit=80),
        "safe_tool": _redacted_memory_text(item.get("safe_tool") or "", limit=80),
        "tool_ok": bool(item.get("tool_ok")),
        "status": _redacted_memory_text(item.get("status") or "", limit=80),
        "tool_artifact_path": _redacted_memory_text(item.get("tool_artifact_path") or "", limit=260),
        "before_remaining_safe_tools": _safe_string_list(item.get("before_remaining_safe_tools"), max_items=8),
        "after_remaining_safe_tools": _safe_string_list(item.get("after_remaining_safe_tools"), max_items=8),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }


def get_scientist_continuation_status(session: SessionState, root: Path) -> dict[str, Any]:
    """Summarize the latest incomplete AI Scientist turn.

    The status tool is intentionally read-only with respect to experiments:
    it may write a small display artifact, but it never starts training,
    downloads data, or submits to Kaggle.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    continuation_path = root / ".xsci" / "scientist_continuation.json"
    status_path = root / ".xsci" / "scientist_continuation_status.json"
    continuation = _read_json_artifact(continuation_path)

    if not isinstance(continuation, dict):
        payload: dict[str, Any] = {
            "ok": True,
            "tool": "scientist_continuation_status",
            "generated_at": generated_at,
            "selected_task": session.selected_task or "",
            "status": "no_continuation",
            "completion_ratio": 1.0,
            "total_required_tools": 0,
            "completed_required_tools": 0,
            "remaining_count": 0,
            "remaining_safe_tools": [],
            "executed_or_completed_tools": [],
            "progress_history": [],
            "safe_next_command": "",
            "next_safe_action_command": "",
            "message": "No Scientist continuation artifact exists. Start with `evomind ask \"<goal>\"` or refresh with `evomind autopilot`.",
            "continuation_artifact_path": str(continuation_path),
            "artifact_path": str(status_path),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
    elif str(continuation.get("schema") or "") != "evomind.ai_scientist.continuation.v1":
        payload = {
            "ok": False,
            "tool": "scientist_continuation_status",
            "generated_at": generated_at,
            "selected_task": session.selected_task or "",
            "status": "invalid_continuation_artifact",
            "completion_ratio": 0.0,
            "total_required_tools": 0,
            "completed_required_tools": 0,
            "remaining_count": 0,
            "remaining_safe_tools": [],
            "executed_or_completed_tools": [],
            "progress_history": [],
            "safe_next_command": "",
            "next_safe_action_command": "",
            "message": "Latest continuation artifact has an unexpected schema; run `evomind ask \"<goal>\"` to rebuild it.",
            "continuation_artifact_path": str(continuation_path),
            "artifact_path": str(status_path),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
    else:
        raw_history = continuation.get("progress_history") if isinstance(continuation.get("progress_history"), list) else []
        progress_history = [
            sanitized for sanitized in (_sanitize_continuation_progress(item) for item in raw_history[-20:])
            if sanitized
        ]
        remaining = _safe_string_list(continuation.get("remaining_safe_tools"), max_items=20)
        must_run = _safe_string_list(continuation.get("must_run_deferred_tools"), max_items=20)
        completed_from_history = []
        for item in progress_history:
            tool = str(item.get("safe_tool") or "")
            if tool and item.get("tool_ok") and tool not in completed_from_history:
                completed_from_history.append(tool)
        if must_run:
            completed = [tool for tool in must_run if tool not in remaining]
            for tool in completed_from_history:
                if tool not in completed:
                    completed.append(tool)
            total = len(must_run)
        else:
            completed = completed_from_history
            total = len(completed) + len(remaining)
        completed_count = min(total, max(0, total - len(remaining))) if total else 0
        if completed and completed_count < len(completed):
            completed_count = min(total or len(completed), len(completed))
        ratio = 1.0 if total == 0 else round(completed_count / total, 4)
        actions, _continuation = _scientist_continuation_actions(root)
        next_action_command = str(actions[0].get("command") or "") if actions else ""
        status = str(continuation.get("status") or "unknown")
        message = (
            "Continuation is complete; no required read-only tools remain."
            if status == "closed" or not remaining
            else f"Continuation needs {len(remaining)} more read-only tool(s). Run `evomind next` to consume the next safe action."
        )
        payload = {
            "ok": True,
            "tool": "scientist_continuation_status",
            "generated_at": generated_at,
            "selected_task": session.selected_task or "",
            "status": status,
            "completion_ratio": ratio,
            "total_required_tools": total,
            "completed_required_tools": completed_count,
            "remaining_count": len(remaining),
            "remaining_safe_tools": remaining,
            "executed_or_completed_tools": completed,
            "progress_history": progress_history,
            "safe_next_command": _redacted_memory_text(continuation.get("safe_next_command") or "", limit=260),
            "next_safe_action_command": _redacted_memory_text(next_action_command or "evomind next", limit=260) if remaining else "",
            "explicit_user_budget_cap": bool(continuation.get("explicit_user_budget_cap")),
            "continuation_artifact_path": str(continuation_path),
            "artifact_path": str(status_path),
            "message": message,
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }

    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write continuation status artifact: {exc}"
    return payload


def _scientist_continuation_actions(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    continuation_path = Path(root) / ".xsci" / "scientist_continuation.json"
    continuation = _read_json_artifact(continuation_path) or {}
    if str(continuation.get("schema") or "") != "evomind.ai_scientist.continuation.v1":
        return [], {}
    if str(continuation.get("status") or "") != "needs_more_tools":
        return [], continuation
    remaining = [str(tool) for tool in (continuation.get("remaining_safe_tools") or []) if str(tool)]
    if not remaining:
        return [], continuation
    hints = continuation.get("action_queue_hint") if isinstance(continuation.get("action_queue_hint"), list) else []
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, tool in enumerate(remaining):
        hint = next((item for item in hints if isinstance(item, dict) and str(item.get("safe_tool") or "") == tool), {})
        command = str((hint or {}).get("command") or f"evomind {tool.replace('scientist_', '').replace('_', '-')}")
        safe_tool = _safe_read_only_action_tool(command)
        if safe_tool != tool or tool in seen:
            continue
        seen.add(tool)
        actions.append({
            "id": f"continue_required_tool_{index + 1}",
            "title": f"Continue required read-only Scientist tool: {tool}",
            "status": "ready",
            "command": command,
            "gate": "continuation_read_only_gate",
            "why": (
                "Previous Scientist turn stopped before this must-run read-only tool completed; "
                "continue without training or official submit."
            ),
            "risk": "read-only continuation; no training or official submission is started",
            "expected_artifacts": _safe_string_list(continuation.get("must_run_deferred_tools"), max_items=8),
            "autonomy": "read_only",
            "metadata": {
                "source": "scientist_continuation",
                "safe_tool": tool,
                "continuation_artifact_path": str(continuation_path),
                "continuation_status": str(continuation.get("status") or ""),
            },
            "no_training_started": True,
        })
    return actions, continuation


def _apply_scientist_continuation_to_action_queue(
    payload: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    base_actions = [
        action for action in actions
        if not (
            isinstance(action, dict)
            and isinstance(action.get("metadata"), dict)
            and action["metadata"].get("source") == "scientist_continuation"
        )
    ]
    continuation_actions, continuation = _scientist_continuation_actions(root)
    if not continuation_actions:
        payload["actions"] = base_actions
        payload.pop("scientist_continuation_summary", None)
        return payload
    continuation_ids = {str(action.get("id") or "") for action in continuation_actions}
    payload["actions"] = continuation_actions + [
        action for action in base_actions
        if not (isinstance(action, dict) and str(action.get("id") or "") in continuation_ids)
    ]
    payload["scientist_continuation_summary"] = {
        "source_artifact": str(Path(root) / ".xsci" / "scientist_continuation.json"),
        "status": str(continuation.get("status") or ""),
        "remaining_safe_tools": [str(tool) for tool in (continuation.get("remaining_safe_tools") or []) if str(tool)],
        "action_ids": [str(action.get("id") or "") for action in continuation_actions],
        "safe_next_command": _redacted_memory_text(continuation.get("safe_next_command") or "", limit=220),
    }
    return payload


def _persist_continuation_progress_after_next_action(
    root: Path,
    *,
    selected: dict[str, Any],
    safe_tool: str,
    tool_result: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
    if metadata.get("source") != "scientist_continuation":
        return {}
    continuation_path = Path(metadata.get("continuation_artifact_path") or Path(root) / ".xsci" / "scientist_continuation.json")
    continuation = _read_json_artifact(continuation_path)
    if not isinstance(continuation, dict):
        return {}
    remaining_before = [str(tool) for tool in (continuation.get("remaining_safe_tools") or []) if str(tool)]
    ok = bool(tool_result.get("ok", True))
    remaining_after = [tool for tool in remaining_before if not (ok and tool == safe_tool)]
    artifact_path = str(tool_result.get("artifact_path") or tool_result.get("path") or "")
    progress = {
        "schema": "evomind.ai_scientist.continuation_progress.v1",
        "updated_at": generated_at,
        "safe_tool": safe_tool,
        "tool_ok": ok,
        "tool_artifact_path": artifact_path,
        "before_remaining_safe_tools": remaining_before,
        "after_remaining_safe_tools": remaining_after,
        "status": "closed" if ok and not remaining_after else "needs_more_tools",
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    history = continuation.get("progress_history") if isinstance(continuation.get("progress_history"), list) else []
    history.append(progress)
    continuation["progress_history"] = history[-20:]
    continuation["remaining_safe_tools"] = remaining_after
    continuation["status"] = progress["status"]
    continuation["last_progress"] = progress
    continuation["no_training_started"] = True
    continuation["official_submit"] = "blocked_until_explicit_human_approval"
    try:
        tmp = continuation_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(continuation, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(continuation_path)
    except OSError:
        pass
    queue_path = Path(root) / ".xsci" / "scientist_action_queue.json"
    queue_payload = _read_json_artifact(queue_path)
    if isinstance(queue_payload, dict):
        refreshed_queue = _apply_requirement_ledger_to_action_queue(queue_payload, root=Path(root))
        refreshed_queue = _apply_scientist_continuation_to_action_queue(refreshed_queue, root=Path(root))
        try:
            tmp = queue_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(refreshed_queue, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(queue_path)
        except OSError:
            pass
    return progress


def run_scientist_next_action(session: SessionState, root: Path) -> dict[str, Any]:
    """Choose the next action from the queue and execute it only if read-only.

    This is the closest EvoMind terminal equivalent to a Codex/Claude Code
    "continue" step.  It advances safe diagnostics automatically and stops at
    training/submission/user-input gates with a clear artifact.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    queue_payload = get_scientist_action_queue(session, root)
    actions = queue_payload.get("actions") if isinstance(queue_payload, dict) else []
    if not isinstance(actions, list):
        actions = []
    selected = _select_ready_scientist_action(actions, root=root)
    if _should_refresh_scientist_action_queue(selected):
        queue_payload = get_scientist_action_queue(session, root, refresh=True)
        actions = queue_payload.get("actions") if isinstance(queue_payload, dict) else []
        if not isinstance(actions, list):
            actions = []
        selected = _select_ready_scientist_action(actions, root=root)
    artifact_path = root / ".xsci" / "scientist_next_action.json"

    if selected is None:
        payload: dict[str, Any] = {
            "ok": True,
            "tool": "scientist_next_action",
            "generated_at": generated_at,
            "selected_task": session.selected_task or "",
            "status": "no_ready_action",
            "selected_action": None,
            "message": "No ready action is available. Run `evomind autopilot` to refresh the queue.",
            "artifact_path": str(artifact_path),
            "action_queue_artifact_path": queue_payload.get("artifact_path"),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }
    else:
        autonomy = str(selected.get("autonomy") or "")
        command = str(selected.get("command") or "")
        safe_tool = _safe_read_only_action_tool(command)
        if safe_tool and autonomy in {"read_only", "read_only_repair_guidance"}:
            result = run_scientist_autopilot(session, root) if safe_tool == "scientist_autopilot" else TerminalTools.dispatch(safe_tool, session, root)
            requirement_progress = _persist_requirement_progress_after_next_action(
                root,
                selected=selected,
                safe_tool=safe_tool,
                tool_result=result if isinstance(result, dict) else {},
                generated_at=generated_at,
                selected_task=session.selected_task or "",
            )
            continuation_progress = _persist_continuation_progress_after_next_action(
                root,
                selected=selected,
                safe_tool=safe_tool,
                tool_result=result if isinstance(result, dict) else {},
                generated_at=generated_at,
            )
            payload = {
                "ok": True,
                "tool": "scientist_next_action",
                "generated_at": generated_at,
                "selected_task": session.selected_task or "",
                "status": "executed_read_only_tool",
                "selected_action": selected,
                "executed_tool": safe_tool,
                "tool_result": result,
                "requirement_progress": requirement_progress,
                "continuation_progress": continuation_progress,
                "message": f"Executed safe read-only next action via {safe_tool}.",
                "artifact_path": str(artifact_path),
                "action_queue_artifact_path": queue_payload.get("artifact_path"),
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            }
        else:
            gate = str(selected.get("gate") or "unknown_gate")
            payload = {
                "ok": True,
                "tool": "scientist_next_action",
                "generated_at": generated_at,
                "selected_task": session.selected_task or "",
                "status": "blocked_by_gate",
                "selected_action": selected,
                "executed_tool": None,
                "message": (
                    "Next action requires explicit user/workstation approval; EvoMind did not execute it. "
                    f"command={command}; gate={gate}; autonomy={autonomy}."
                ),
                "blocked_reason": str(selected.get("why") or "action requires a non-read-only command"),
                "artifact_path": str(artifact_path),
                "action_queue_artifact_path": queue_payload.get("artifact_path"),
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            }
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write next-action artifact: {exc}"
    try:
        from .scientist_trace import record_scientist_step_event

        record_scientist_step_event(root, {
            "trace_run_id": f"next_action_{generated_at.replace(':', '').replace('+', 'Z')}",
            "source": "scientist_next_action",
            "task": session.selected_task or "",
            "phase": "next_action",
            "status": payload.get("status"),
            "tool": "scientist_next_action",
            "message": payload.get("message", ""),
            "artifact_path": str(artifact_path),
            "gate": (payload.get("selected_action") or {}).get("gate") if isinstance(payload.get("selected_action"), dict) else "",
            "details": {
                "executed_tool": payload.get("executed_tool"),
                "selected_action": payload.get("selected_action"),
            },
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def run_scientist_continuation_resume(
    session: SessionState,
    root: Path,
    observer: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Run remaining continuation tools until closed or a gate/stall is reached.

    This is a bounded convenience wrapper around ``scientist_next_action``.  It
    only consumes actions that the existing next-action gate already classifies
    as read-only, and it stops as soon as no progress is observed, a non
    read-only gate appears, or the step budget is exhausted.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_path = root / ".xsci" / "scientist_continuation_resume.json"
    trace_run_id = f"continuation_resume_{generated_at.replace(':', '').replace('+', 'Z')}"

    try:
        from .scientist_trace import record_scientist_step_event
    except Exception:
        record_scientist_step_event = None  # type: ignore[assignment]

    def publish_event(payload: dict[str, Any]) -> None:
        event = {
            "trace_run_id": trace_run_id,
            "source": "scientist_continuation_resume",
            "task": session.selected_task or "",
            "tool": payload.get("tool") or "scientist_continuation_resume",
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
            **payload,
        }
        if record_scientist_step_event is not None:
            try:
                record_scientist_step_event(root, event)
            except Exception:
                pass
        if observer is not None:
            try:
                observer(event)
            except Exception:
                pass

    try:
        max_steps = int(os.environ.get("EVOMIND_CONTINUATION_RESUME_MAX_STEPS", "8"))
    except (TypeError, ValueError):
        max_steps = 8
    max_steps = max(1, min(12, max_steps))

    initial_status = get_scientist_continuation_status(session, root)
    initial_remaining = [
        str(tool)
        for tool in (initial_status.get("remaining_safe_tools") or [])
        if str(tool)
    ]
    publish_event({
        "phase": "continuation_resume_start",
        "status": "running",
        "message": f"Starting bounded continuation resume; remaining={len(initial_remaining)}.",
        "details": {
            "initial_status": initial_status.get("status"),
            "remaining_safe_tools": initial_remaining,
            "max_steps": max_steps,
        },
    })
    steps: list[dict[str, Any]] = []
    stop_reason = ""

    if str(initial_status.get("status") or "") == "no_continuation":
        stop_reason = "no_continuation"
    elif not initial_remaining or str(initial_status.get("status") or "") == "closed":
        stop_reason = "already_closed"
    else:
        seen_remaining: set[tuple[str, ...]] = {tuple(initial_remaining)}
        for index in range(1, max_steps + 1):
            before_status = get_scientist_continuation_status(session, root)
            before_remaining = [
                str(tool)
                for tool in (before_status.get("remaining_safe_tools") or [])
                if str(tool)
            ]
            if not before_remaining or str(before_status.get("status") or "") == "closed":
                stop_reason = "closed"
                break

            publish_event({
                "phase": "continuation_resume_step_started",
                "status": "running",
                "message": f"Executing continuation safe step {index}; remaining_before={len(before_remaining)}.",
                "details": {
                    "index": index,
                    "before_remaining_safe_tools": before_remaining,
                    "next_safe_action_command": before_status.get("next_safe_action_command") or before_status.get("safe_next_command") or "evomind next",
                },
            })
            result = run_scientist_next_action(session, root)
            after_status = get_scientist_continuation_status(session, root)
            after_remaining = [
                str(tool)
                for tool in (after_status.get("remaining_safe_tools") or [])
                if str(tool)
            ]
            selected_action = result.get("selected_action") if isinstance(result.get("selected_action"), dict) else {}
            step = {
                "index": index,
                "status": str(result.get("status") or ""),
                "executed_tool": str(result.get("executed_tool") or ""),
                "selected_action_id": str(selected_action.get("id") or ""),
                "selected_command": _redacted_memory_text(selected_action.get("command") or "", limit=260),
                "before_remaining_safe_tools": before_remaining,
                "after_remaining_safe_tools": after_remaining,
                "tool_artifact_path": str((result.get("tool_result") or {}).get("artifact_path") or "") if isinstance(result.get("tool_result"), dict) else "",
                "next_action_artifact_path": str(result.get("artifact_path") or ""),
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            }
            steps.append(step)
            publish_event({
                "phase": "continuation_resume_step_completed",
                "status": "passed" if str(result.get("status") or "") == "executed_read_only_tool" else "blocked",
                "message": (
                    f"Step {index} {step['status']}; executed_tool={step['executed_tool'] or '(none)'}; "
                    f"remaining_after={len(after_remaining)}."
                ),
                "artifact_path": step.get("next_action_artifact_path"),
                "details": step,
            })

            if str(result.get("status") or "") == "blocked_by_gate":
                stop_reason = "blocked_by_gate"
                break
            if str(result.get("status") or "") != "executed_read_only_tool":
                stop_reason = "stalled_no_read_only_action"
                break
            if tuple(after_remaining) == tuple(before_remaining):
                stop_reason = "stalled_no_progress"
                break
            if not after_remaining or str(after_status.get("status") or "") == "closed":
                stop_reason = "closed"
                break
            remaining_key = tuple(after_remaining)
            if remaining_key in seen_remaining:
                stop_reason = "stalled_repeated_remaining_set"
                break
            seen_remaining.add(remaining_key)
        else:
            stop_reason = "max_steps_reached"

    final_status = get_scientist_continuation_status(session, root)
    final_remaining = [
        str(tool)
        for tool in (final_status.get("remaining_safe_tools") or [])
        if str(tool)
    ]
    if not final_remaining and stop_reason not in {"no_continuation", "already_closed"}:
        status = "closed"
    elif stop_reason == "no_continuation":
        status = "no_continuation"
    elif stop_reason == "blocked_by_gate":
        status = "blocked_by_gate"
    elif stop_reason.startswith("stalled"):
        status = "stalled"
    elif stop_reason == "already_closed":
        status = "closed"
    else:
        status = "needs_more_tools" if final_remaining else "closed"

    payload: dict[str, Any] = {
        "ok": status not in {"blocked_by_gate", "stalled"},
        "tool": "scientist_continuation_resume",
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "status": status,
        "stop_reason": stop_reason or status,
        "max_steps": max_steps,
        "steps_executed": len(steps),
        "steps": steps,
        "initial_status": {
            "status": initial_status.get("status"),
            "remaining_safe_tools": initial_remaining,
            "completion_ratio": initial_status.get("completion_ratio"),
        },
        "final_status": {
            "status": final_status.get("status"),
            "remaining_safe_tools": final_remaining,
            "completion_ratio": final_status.get("completion_ratio"),
            "completed_required_tools": final_status.get("completed_required_tools"),
            "total_required_tools": final_status.get("total_required_tools"),
        },
        "remaining_safe_tools": final_remaining,
        "executed_tools": [
            str(step.get("executed_tool") or "")
            for step in steps
            if step.get("executed_tool")
        ],
        "message": (
            "Continuation closed; all remaining read-only Scientist tools completed."
            if status == "closed"
            else "No continuation artifact exists."
            if status == "no_continuation"
            else "Continuation resume stopped before closure; inspect stop_reason and remaining_safe_tools."
        ),
        "artifact_path": str(artifact_path),
        "continuation_status_artifact_path": str(final_status.get("artifact_path") or ""),
        "continuation_artifact_path": str(final_status.get("continuation_artifact_path") or root / ".xsci" / "scientist_continuation.json"),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(artifact_path)
    except OSError as exc:
        payload["ok"] = False
        payload["message"] = f"Could not write continuation resume artifact: {exc}"
    publish_event({
        "phase": "continuation_resume_complete",
        "status": status,
        "message": str(payload.get("message") or ""),
        "artifact_path": str(artifact_path),
        "gate": "continuation_read_only_gate",
        "details": {
            "stop_reason": payload.get("stop_reason"),
            "steps_executed": payload.get("steps_executed"),
            "remaining_safe_tools": final_remaining,
        },
    })
    return payload


def run_scientist_autopilot(
    session: SessionState,
    root: Path,
    observer: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Run a bounded multi-tool diagnosis without starting training.

    This is the terminal equivalent of a Claude Code-style "think with tools"
    turn. It observes system readiness, task/data state, recent results,
    evolution memory, and the next research decision, then persists one
    sanitized artifact for the dashboard/LLM context. It never reads secrets and
    never submits to Kaggle.
    """
    root = Path(root)
    trace_run_id = f"autopilot_{datetime.now(timezone.utc).isoformat(timespec='seconds').replace(':', '').replace('+', 'Z')}"
    try:
        from .scientist_trace import record_scientist_step_event
    except Exception:
        record_scientist_step_event = None  # type: ignore[assignment]

    def trace_event(payload: dict[str, Any]) -> None:
        if record_scientist_step_event is None:
            return
        try:
            record_scientist_step_event(root, {
                "trace_run_id": trace_run_id,
                "source": "scientist_autopilot",
                "task": session.selected_task or "",
                "no_training_started": True,
                **payload,
            })
        except Exception:
            pass

    def publish_event(payload: dict[str, Any]) -> None:
        event = {
            "trace_run_id": trace_run_id,
            "source": "scientist_autopilot",
            "task": session.selected_task or "",
            "no_training_started": True,
            **payload,
        }
        trace_event(payload)
        if observer is not None:
            try:
                observer(event)
            except Exception:
                pass

    def tool_choice_metadata(name: str) -> dict[str, Any]:
        """Explain why this bounded Scientist turn selected a tool.

        The explanation is intentionally deterministic and non-secret so it can
        be persisted into traces, terminal streams, and dashboard context.
        """
        selected = bool(session.selected_task)
        table: dict[str, tuple[str, float, str]] = {
            "system_status": (
                "Verify model, Kaggle, compute, and dashboard readiness before any research action.",
                0.98,
                "setup_gate",
            ),
            "inspect_task": (
                "Ground the diagnosis in the selected competition, metric direction, and task metadata.",
                0.96 if selected else 0.72,
                "task_context" if selected else "missing_task",
            ),
            "data_check": (
                "Confirm train/test data or declared remote data paths before proposing execution.",
                0.95,
                "data_contract",
            ),
            "recent_run": (
                "Read the latest run evidence so new decisions do not overwrite best-so-far blindly.",
                0.91,
                "latest_artifact",
            ),
            "evolution_status": (
                "Reuse retrospective memory and prior lessons to choose the next branch intelligently.",
                0.93,
                "memory_layer",
            ),
            "scientist_checkpoint": (
                "Combine readiness, data, and gate checks into one go/no-go checkpoint.",
                0.97,
                "execution_gate",
            ),
            "research_decision": (
                "Select the safest next research branch, code mode, and rollback condition.",
                0.94,
                "search_controller",
            ),
            "scientist_hypothesis_review": (
                "Critique and rank proposal branches before training so the queue follows the strongest evidence-backed hypothesis.",
                0.93,
                "hypothesis_review_gate",
            ),
            "scientist_experiment_blueprint": (
                "Translate the reviewed hypothesis into an auditable branch, resource, artifact, rollback, and memory-writeback blueprint.",
                0.92,
                "experiment_blueprint",
            ),
            "scientist_workplan": (
                "Materialize a stepwise workplan that the terminal and UI can audit before execution.",
                0.92,
                "workplan_artifact",
            ),
            "scientist_repair_plan": (
                "Identify blockers and safe repair commands instead of silently failing or guessing.",
                0.90,
                "repair_artifact",
            ),
            "scientist_execution_contract": (
                "Enforce no-training, no-submit, artifact, and human-gate boundaries for this turn.",
                0.99,
                "safety_contract",
            ),
        }
        rationale, confidence, evidence_signal = table.get(
            name,
            ("Inspect a bounded, read-only research signal for the current Scientist turn.", 0.80, "tool_context"),
        )
        return {
            "rationale": rationale,
            "confidence": round(float(confidence), 2),
            "evidence_signal": evidence_signal,
        }

    publish_event({
        "phase": "autopilot_start",
        "status": "running",
        "tool": "scientist_autopilot",
        "message": "Starting bounded multi-tool AI Scientist diagnosis.",
    })
    calls = [
        ("system_status", get_system_status),
        ("inspect_task", inspect_selected_task),
        ("data_check", inspect_data_availability),
        ("recent_run", inspect_recent_run),
        ("evolution_status", inspect_evolution_status),
        ("scientist_checkpoint", get_scientist_checkpoint),
        ("research_decision", get_research_decision),
        ("scientist_hypothesis_review", get_scientist_hypothesis_review),
        ("scientist_experiment_blueprint", get_scientist_experiment_blueprint),
        ("scientist_workplan", get_scientist_workplan),
        ("scientist_repair_plan", get_scientist_repair_plan),
        ("scientist_execution_contract", get_scientist_execution_contract),
    ]
    tool_results: dict[str, Any] = {}
    tool_trace: list[dict[str, Any]] = []
    for name, fn in calls:
        choice = tool_choice_metadata(name)
        publish_event({
            "phase": "tool_started",
            "status": "running",
            "tool": name,
            "message": f"Calling {name}. {choice['rationale']}",
            "details": choice,
        })
        try:
            result = fn(session, root)
        except Exception as exc:
            result = {"ok": False, "tool": name, "message": f"{type(exc).__name__}: {exc}"}
        status = "ok" if result.get("ok", True) else "blocked"
        tool_results[name] = result
        tool_trace.append({
            "tool": name,
            "ok": bool(result.get("ok", True)),
            "status": status,
            "message": str(result.get("message") or result.get("mode") or "")[:220],
            "rationale": choice["rationale"],
            "confidence": choice["confidence"],
            "evidence_signal": choice["evidence_signal"],
        })
        publish_event({
            "phase": "tool_completed" if result.get("ok", True) else "tool_blocked",
            "status": status,
            "tool": name,
            "message": str(result.get("message") or result.get("mode") or f"{name} finished.")[:500],
            "artifact_path": str(result.get("artifact_path") or ""),
            "details": {
                "mode": result.get("mode"),
                "selected_task": result.get("selected_task") or session.selected_task or "",
                "summary": result.get("summary") if isinstance(result.get("summary"), dict) else {},
                "rationale": choice["rationale"],
                "confidence": choice["confidence"],
                "evidence_signal": choice["evidence_signal"],
            },
        })

    checkpoint = tool_results.get("scientist_checkpoint", {})
    decision_result = tool_results.get("research_decision", {})
    decision = decision_result.get("decision", {}) if isinstance(decision_result, dict) else {}
    hypothesis_review = tool_results.get("scientist_hypothesis_review", {})
    selected_hypothesis = hypothesis_review.get("selected_hypothesis") if isinstance(hypothesis_review, dict) else {}
    if not isinstance(selected_hypothesis, dict):
        selected_hypothesis = {}
    experiment_blueprint = tool_results.get("scientist_experiment_blueprint", {})
    blueprint = experiment_blueprint.get("experiment_blueprint") if isinstance(experiment_blueprint, dict) else {}
    if not isinstance(blueprint, dict):
        blueprint = {}
    memory_reuse_plan = (
        blueprint.get("memory_reuse_plan")
        if isinstance(blueprint.get("memory_reuse_plan"), dict)
        else experiment_blueprint.get("memory_reuse_plan") if isinstance(experiment_blueprint, dict) and isinstance(experiment_blueprint.get("memory_reuse_plan"), dict)
        else selected_hypothesis.get("memory_reuse_plan") if isinstance(selected_hypothesis.get("memory_reuse_plan"), dict)
        else {}
    )
    workplan_result = tool_results.get("scientist_workplan", {})
    repair_result = tool_results.get("scientist_repair_plan", {})
    contract_result = tool_results.get("scientist_execution_contract", {})
    system = tool_results.get("system_status", {})
    data = tool_results.get("data_check", {})
    evolution = tool_results.get("evolution_status", {})

    blockers: list[str] = []
    if isinstance(system, dict):
        blockers.extend(str(x) for x in (system.get("blockers") or []))
    if isinstance(checkpoint, dict):
        gate = checkpoint.get("gate", {}) if isinstance(checkpoint.get("gate"), dict) else {}
        blockers.extend(str(x) for x in (gate.get("blockers") or []))
        blockers.extend(str(x) for x in (gate.get("warnings") or []) if "missing" in str(x).lower())
    blockers = list(dict.fromkeys(b for b in blockers if b))

    data_ready = bool(data.get("train_csv")) or bool(
        isinstance(checkpoint, dict)
        and checkpoint.get("mode") == "ready_to_execute"
        and "Remote GPU training data is declared." in " ".join(checkpoint.get("analyze", []))
    )
    can_execute = bool(
        isinstance(checkpoint, dict)
        and checkpoint.get("gate", {}).get("can_execute")
        and not blockers
    )

    next_actions: list[str] = []
    if not session.selected_task:
        next_actions.append("Select/register a task before any experiment.")
    elif blockers:
        next_actions.append("Clear setup gates before training: " + "; ".join(blockers[:3]))
    elif not data_ready:
        next_actions.append("Register/download train/test data or declare a GPU data path.")
    elif selected_hypothesis:
        next_actions.append(
            "Ready for reviewed hypothesis: "
            f"strategy={selected_hypothesis.get('strategy_name')}, "
            f"score={selected_hypothesis.get('score')}, "
            f"branch={selected_hypothesis.get('branch_type')}, "
            f"mode={selected_hypothesis.get('code_generation_mode')}, "
            f"blueprint={blueprint.get('blueprint_id', 'not_generated')}."
        )
    elif decision:
        next_actions.append(
            "Ready for audited run: "
            f"action={decision.get('selected_action')}, "
            f"branch={decision.get('selected_branch')}, "
            f"mode={decision.get('code_generation_mode')}."
        )
    else:
        next_actions.append("Rebuild research decision before execution.")
    next_actions.append("Official Kaggle submit remains blocked until explicit human approval and response artifact.")

    tracker = evolution.get("tracker", {}) if isinstance(evolution, dict) else {}
    memory = evolution.get("retrospective_memory", {}) if isinstance(evolution, dict) else {}
    summary_lines = [
        f"selected_task={session.selected_task or '(none)'}",
        f"can_execute={can_execute}",
        f"data_ready={data_ready}",
        f"decision={decision.get('selected_action', 'none') if isinstance(decision, dict) else 'none'}",
        (
            "evolution="
            f"runs={tracker.get('total_runs', 0)}, "
            f"lessons={tracker.get('lessons_recorded', 0)}, "
            f"memory_records={memory.get('records', 0)}"
        ),
        (
            "memory_reuse="
            f"rules={len(memory_reuse_plan.get('reuse_rules') or [])}, "
            f"avoid={len(memory_reuse_plan.get('avoid_patterns') or [])}"
        ),
    ]

    artifact_path = root / ".xsci" / "scientist_autopilot.json"
    action_queue_artifact_path = root / ".xsci" / "scientist_action_queue.json"
    action_queue = _build_scientist_action_queue(
        session=session,
        root=root,
        decision=decision if isinstance(decision, dict) else {},
        workplan_result=workplan_result if isinstance(workplan_result, dict) else {},
        repair_result=repair_result if isinstance(repair_result, dict) else {},
        contract_result=contract_result if isinstance(contract_result, dict) else {},
        hypothesis_review=hypothesis_review if isinstance(hypothesis_review, dict) else {},
        experiment_blueprint=experiment_blueprint if isinstance(experiment_blueprint, dict) else {},
        blockers=blockers,
        data_ready=data_ready,
        can_execute=can_execute,
    )
    try:
        action_queue_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_queue = action_queue_artifact_path.with_suffix(".json.tmp")
        tmp_queue.write_text(json.dumps({
            "ok": True,
            "tool": "scientist_action_queue",
            "trace_run_id": trace_run_id,
            "selected_task": session.selected_task or "",
            "actions": action_queue,
            "selected_hypothesis": selected_hypothesis,
            "hypothesis_review_artifact_path": str((hypothesis_review or {}).get("artifact_path") or root / ".xsci" / "scientist_hypothesis_review.json") if isinstance(hypothesis_review, dict) else str(root / ".xsci" / "scientist_hypothesis_review.json"),
            "experiment_blueprint": blueprint,
            "memory_reuse_plan": memory_reuse_plan,
            "experiment_blueprint_artifact_path": str((experiment_blueprint or {}).get("artifact_path") or root / ".xsci" / "scientist_experiment_blueprint.json") if isinstance(experiment_blueprint, dict) else str(root / ".xsci" / "scientist_experiment_blueprint.json"),
            "no_training_started": True,
            "official_submit": "blocked_until_explicit_human_approval",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_queue.replace(action_queue_artifact_path)
    except OSError:
        pass
    payload = {
        "ok": True,
        "tool": "scientist_autopilot",
        "trace_run_id": trace_run_id,
        "selected_task": session.selected_task or "",
        "mode": "ready_to_execute" if can_execute else "needs_attention",
        "summary_lines": summary_lines,
        "tool_trace": tool_trace,
        "next_actions": next_actions,
        "action_queue": action_queue,
        "action_queue_artifact_path": str(action_queue_artifact_path),
        "selected_hypothesis": selected_hypothesis,
        "hypothesis_review_artifact_path": str((hypothesis_review or {}).get("artifact_path") or root / ".xsci" / "scientist_hypothesis_review.json") if isinstance(hypothesis_review, dict) else str(root / ".xsci" / "scientist_hypothesis_review.json"),
        "experiment_blueprint": blueprint,
        "memory_reuse_plan": memory_reuse_plan,
        "experiment_blueprint_artifact_path": str((experiment_blueprint or {}).get("artifact_path") or root / ".xsci" / "scientist_experiment_blueprint.json") if isinstance(experiment_blueprint, dict) else str(root / ".xsci" / "scientist_experiment_blueprint.json"),
        "blockers": blockers,
        "decision": decision,
        "workplan": {
            "mode": workplan_result.get("mode") if isinstance(workplan_result, dict) else "",
            "current_focus": workplan_result.get("current_focus") if isinstance(workplan_result, dict) else {},
            "summary": workplan_result.get("summary") if isinstance(workplan_result, dict) else {},
            "artifact_path": workplan_result.get("artifact_path") if isinstance(workplan_result, dict) else str(root / ".xsci" / "scientist_workplan.json"),
        },
        "repair_plan": {
            "mode": repair_result.get("mode") if isinstance(repair_result, dict) else "",
            "root_causes": repair_result.get("root_causes") if isinstance(repair_result, dict) else [],
            "safe_next_command": repair_result.get("safe_next_command") if isinstance(repair_result, dict) else "",
            "artifact_path": repair_result.get("artifact_path") if isinstance(repair_result, dict) else str(root / ".xsci" / "scientist_repair_plan.json"),
        },
        "execution_contract": {
            "go_no_go": contract_result.get("go_no_go") if isinstance(contract_result, dict) else "",
            "agent_session_ready": contract_result.get("agent_session_ready") if isinstance(contract_result, dict) else False,
            "model_training_ready": contract_result.get("model_training_ready") if isinstance(contract_result, dict) else False,
            "data_contract_status": contract_result.get("data_contract_status") if isinstance(contract_result, dict) else "",
            "artifact_path": contract_result.get("artifact_path") if isinstance(contract_result, dict) else str(root / ".xsci" / "scientist_execution_contract.json"),
        },
        "artifact_path": str(artifact_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    publish_event({
        "phase": "autopilot_complete",
        "status": "completed" if payload["mode"] == "ready_to_execute" else "blocked",
        "tool": "scientist_autopilot",
        "message": "; ".join(summary_lines + next_actions)[:900],
        "artifact_path": str(artifact_path),
        "details": {
            "mode": payload["mode"],
            "next_actions": next_actions,
            "action_queue": action_queue[:3],
            "blockers": blockers,
            "decision": decision,
        },
    })
    try:
        from .scientist_trace import load_recent_scientist_step_events, scientist_step_trace_path

        recent_step_events = load_recent_scientist_step_events(root, limit=50)
        payload["step_trace"] = {
            "artifact_path": str(scientist_step_trace_path(root)),
            "count": len(recent_step_events),
            "recent": recent_step_events,
        }
    except Exception:
        payload["step_trace"] = {
            "artifact_path": str(root / ".xsci" / "scientist_step_trace.jsonl"),
            "count": 0,
            "recent": [],
        }
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        payload["artifact_error"] = "failed_to_write_artifact"
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_autopilot",
            "user": "scientist_autopilot",
            "forced_tools": [name for name, _ in calls],
            "executed_tools": tool_trace,
            "mode": payload.get("mode"),
            "decision": decision,
            "blockers": blockers,
            "next_actions": next_actions,
            "artifacts": [
                str(artifact_path),
                str(action_queue_artifact_path),
                str(root / ".xsci" / "scientist_workplan.json"),
                str(root / ".xsci" / "scientist_repair_plan.json"),
                str(root / ".xsci" / "scientist_execution_contract.json"),
                str(root / ".xsci" / "scientist_step_trace.jsonl"),
            ],
            "answer_preview": "; ".join(summary_lines + next_actions),
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


def _scientist_loop_lesson(next_action: dict[str, Any] | None,
                           autopilot: dict[str, Any],
                           steps: list[dict[str, Any]]) -> str:
    if any(str(step.get("step") or "") == "repetition_escalation" for step in steps):
        return (
            "The loop detected that the same read-only action was repeating, so it escalated into "
            "repair-plan, execution-contract, and workplan artifacts instead of spinning in place. "
            "Use those artifacts to clear the next gate before spending compute."
        )
    if not next_action:
        return "No safe next action was available; refresh the AI Scientist diagnosis before changing code or spending compute."
    selected = next_action.get("selected_action") if isinstance(next_action, dict) else {}
    if not isinstance(selected, dict):
        selected = {}
    action_id = str(selected.get("id") or "")
    status = str(next_action.get("status") or "")
    if action_id == "run_gated_candidate" and status == "blocked_by_gate":
        return (
            "The autonomous loop reached the audited training boundary. The next improvement must be launched "
            "through `evomind run <task>` / AgentSession so metrics, OOF/submission, gates, and claim audit are produced."
        )
    if action_id in {"clear_blockers", "repair_no_go_contract", "refresh_plan"}:
        return (
            "Readiness or contract blockers dominate the next step. Reuse the repair plan before model search, "
            "then rebuild the execution contract."
        )
    if status == "executed_read_only_tool":
        return (
            "A safe read-only tool advanced the evidence state. Refresh the action queue before any compute action "
            "so stale commands do not drive the next run."
        )
    blockers = autopilot.get("blockers") if isinstance(autopilot, dict) else []
    if blockers:
        return "The loop learned that setup/data gates are still the bottleneck: " + "; ".join(str(x) for x in blockers[:3])
    return "The loop preserved the safety boundary and produced a recoverable next-action trace for the following scientist turn."


def _scientist_loop_escalation_tools(action_id: str, executed_tool: str) -> list[str]:
    """Choose complementary read-only tools when the loop would otherwise repeat.

    This is the loop's anti-stagnation path: when the same safe action appears
    twice, generate higher-level planning/contract artifacts instead of calling
    the same diagnostic again.
    """
    if action_id in {"clear_blockers", "repair_no_go_contract"} or executed_tool in {"system_status", "gpu_status", "kaggle_status"}:
        return ["scientist_repair_plan", "scientist_execution_contract", "scientist_workplan"]
    if action_id in {"refresh_plan", "watch_evidence"} or executed_tool in {"scientist_step_trace", "scientist_workplan"}:
        return ["scientist_workplan", "scientist_execution_contract", "scientist_repair_plan"]
    return ["scientist_workplan", "scientist_execution_contract"]


def run_scientist_loop(
    session: SessionState,
    root: Path,
    observer: Optional[Callable[[dict[str, Any]], None]] = None,
    *,
    max_steps: int = 3,
) -> dict[str, Any]:
    """Run a bounded autonomous scientist loop without training.

    This is stronger than a one-shot diagnosis: EvoMind observes the current
    state, builds the action queue, executes only safe read-only next actions,
    reflects on the stopping gate, and persists a reusable lesson artifact.  It
    intentionally stops at training/download/submit/user-input gates.
    """
    root = Path(root)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trace_run_id = f"scientist_loop_{generated_at.replace(':', '').replace('+', 'Z')}"
    artifact_path = root / ".xsci" / "scientist_loop.json"
    lessons_path = root / ".xsci" / "scientist_loop_lessons.jsonl"
    steps: list[dict[str, Any]] = []

    try:
        from .scientist_trace import record_scientist_step_event
    except Exception:
        record_scientist_step_event = None  # type: ignore[assignment]

    def emit(phase: str, status: str, tool: str, message: str,
             *, artifact: str = "", details: dict[str, Any] | None = None) -> None:
        event = {
            "trace_run_id": trace_run_id,
            "source": "scientist_loop",
            "task": session.selected_task or "",
            "phase": phase,
            "status": status,
            "tool": tool,
            "message": message,
            "artifact_path": artifact,
            "details": details or {},
            "no_training_started": True,
        }
        if observer is not None:
            try:
                observer(event)
            except Exception:
                pass
        if record_scientist_step_event is not None:
            try:
                record_scientist_step_event(root, event)
            except Exception:
                pass

    emit("loop_start", "running", "scientist_loop",
         "Starting bounded autonomous AI Scientist loop; read-only tools only.")

    autopilot = run_scientist_autopilot(session, root)
    steps.append({
        "step": "observe_and_decide",
        "tool": "scientist_autopilot",
        "status": "ok" if autopilot.get("ok", True) else "blocked",
        "mode": autopilot.get("mode"),
        "artifact_path": autopilot.get("artifact_path"),
    })
    emit(
        "loop_observe",
        "passed" if autopilot.get("ok", True) else "blocked",
        "scientist_autopilot",
        f"mode={autopilot.get('mode')}; actions={len(autopilot.get('action_queue') or [])}",
        artifact=str(autopilot.get("artifact_path") or ""),
    )

    final_next_action: dict[str, Any] | None = None
    repeated_keys: set[tuple[str, str, str]] = set()
    stop_reason = "step_budget_exhausted"

    def escalate_repeated_action(action_id: str, executed_tool: str, reason: str) -> list[dict[str, Any]]:
        escalation_results: list[dict[str, Any]] = []
        for tool_name in _scientist_loop_escalation_tools(action_id, executed_tool):
            try:
                result = TerminalTools.dispatch(tool_name, session, root)
            except Exception as exc:  # pragma: no cover - defensive only
                result = {"ok": False, "tool": tool_name, "message": str(exc)}
            artifact = str(result.get("artifact_path") or result.get("path") or "")
            escalation_results.append({
                "tool": tool_name,
                "status": "ok" if result.get("ok", True) else "blocked",
                "artifact_path": artifact,
            })
            steps.append({
                "step": "repetition_escalation",
                "tool": tool_name,
                "status": "ok" if result.get("ok", True) else "blocked",
                "trigger_action": action_id,
                "trigger_tool": executed_tool,
                "reason": reason,
                "artifact_path": artifact,
            })
            emit(
                "loop_repetition_escalation",
                "passed" if result.get("ok", True) else "blocked",
                tool_name,
                f"{reason}; generated {tool_name} before spending another loop step.",
                artifact=artifact,
                details={"trigger_action": action_id, "trigger_tool": executed_tool, "reason": reason},
            )
        return escalation_results

    for index in range(max(1, max_steps)):
        next_action = run_scientist_next_action(session, root)
        final_next_action = next_action
        selected = next_action.get("selected_action") if isinstance(next_action, dict) else {}
        if not isinstance(selected, dict):
            selected = {}
        status = str(next_action.get("status") or "")
        action_id = str(selected.get("id") or "")
        executed_tool = str(next_action.get("executed_tool") or "")
        steps.append({
            "step": f"safe_next_{index + 1}",
            "tool": "scientist_next_action",
            "status": status,
            "selected_action": action_id,
            "executed_tool": executed_tool,
            "gate": selected.get("gate") or "",
            "artifact_path": next_action.get("artifact_path"),
        })
        emit(
            "loop_next_action",
            "passed" if status == "executed_read_only_tool" else "blocked",
            "scientist_next_action",
            str(next_action.get("message") or status)[:700],
            artifact=str(next_action.get("artifact_path") or ""),
            details={"selected_action": action_id, "executed_tool": executed_tool},
        )

        if status != "executed_read_only_tool":
            stop_reason = status or "stopped_at_gate"
            break
        key = (action_id, status, executed_tool)
        if key in repeated_keys:
            escalate_repeated_action(
                action_id,
                executed_tool,
                f"Repeated read-only action {action_id or executed_tool}",
            )
            stop_reason = "repetition_escalated_to_planning_artifacts"
            break
        repeated_keys.add(key)

        autopilot = run_scientist_autopilot(session, root)
        steps.append({
            "step": f"refresh_after_safe_next_{index + 1}",
            "tool": "scientist_autopilot",
            "status": "ok" if autopilot.get("ok", True) else "blocked",
            "mode": autopilot.get("mode"),
            "artifact_path": autopilot.get("artifact_path"),
        })
        emit(
            "loop_refresh",
            "passed" if autopilot.get("ok", True) else "blocked",
            "scientist_autopilot",
            f"refreshed mode={autopilot.get('mode')}",
            artifact=str(autopilot.get("artifact_path") or ""),
        )

        predicted_actions = autopilot.get("action_queue") if isinstance(autopilot, dict) else []
        if not isinstance(predicted_actions, list):
            predicted_actions = []
        predicted = _select_ready_scientist_action(predicted_actions, root=root)
        if isinstance(predicted, dict):
            predicted_tool = _safe_read_only_action_tool(str(predicted.get("command") or "")) or ""
            predicted_id = str(predicted.get("id") or "")
            predicted_key = (predicted_id, "executed_read_only_tool", predicted_tool)
            if predicted_tool and predicted_key in repeated_keys:
                final_next_action = {
                    "ok": True,
                    "tool": "scientist_next_action",
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "selected_task": session.selected_task or "",
                    "status": "predicted_repeated_read_only_action",
                    "selected_action": predicted,
                    "executed_tool": None,
                    "predicted_tool": predicted_tool,
                    "message": (
                        f"Skipped duplicate safe next action {predicted_id or predicted_tool}; "
                        "escalating to planning artifacts instead."
                    ),
                    "no_training_started": True,
                    "official_submit": "blocked_until_explicit_human_approval",
                }
                steps.append({
                    "step": "predicted_repetition",
                    "tool": "scientist_next_action",
                    "status": "skipped_duplicate",
                    "selected_action": predicted_id,
                    "predicted_tool": predicted_tool,
                    "gate": predicted.get("gate") or "",
                })
                emit(
                    "loop_predicted_repetition",
                    "passed",
                    "scientist_next_action",
                    str(final_next_action["message"]),
                    details={"selected_action": predicted_id, "predicted_tool": predicted_tool},
                )
                escalate_repeated_action(
                    predicted_id,
                    predicted_tool,
                    f"Predicted repeated read-only action {predicted_id or predicted_tool}",
                )
                stop_reason = "repetition_escalated_to_planning_artifacts"
                break
    else:
        stop_reason = "step_budget_exhausted"

    lesson = {
        "ts": generated_at,
        "trace_run_id": trace_run_id,
        "task": session.selected_task or "",
        "stop_reason": stop_reason,
        "lesson": _scientist_loop_lesson(final_next_action, autopilot, steps),
        "selected_action": (
            final_next_action.get("selected_action") if isinstance(final_next_action, dict) else None
        ),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
    }
    payload = {
        "ok": True,
        "tool": "scientist_loop",
        "trace_run_id": trace_run_id,
        "generated_at": generated_at,
        "selected_task": session.selected_task or "",
        "mode": stop_reason if stop_reason.startswith("repetition_escalated") else (
            "stopped_at_gate" if stop_reason not in {"step_budget_exhausted", "repeated_read_only_action"} else stop_reason
        ),
        "stop_reason": stop_reason,
        "steps": steps,
        "final_autopilot": {
            "mode": autopilot.get("mode"),
            "decision": autopilot.get("decision"),
            "blockers": autopilot.get("blockers") or [],
            "next_actions": autopilot.get("next_actions") or [],
            "artifact_path": autopilot.get("artifact_path"),
            "action_queue_artifact_path": autopilot.get("action_queue_artifact_path"),
        },
        "final_next_action": final_next_action,
        "lesson": lesson,
        "next_safe_commands": [
            "evomind memory-consolidate",
            "evomind innovate-plan",
            "evomind self-audit",
        ],
        "artifact_path": str(artifact_path),
        "lessons_path": str(lessons_path),
        "no_training_started": True,
        "official_submit": "blocked_until_explicit_human_approval",
        "human_gate": {
            "training": "blocked_until_explicit_evomind_run_or_workstation_approval",
            "official_kaggle_submit": "blocked_until_explicit_user_approval",
            "rank_or_medal_claims": "blocked_without_kaggle_response_artifact",
        },
    }
    artifact_write_ok = True
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(artifact_path)
        with lessons_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson, ensure_ascii=False) + "\n")
    except OSError as exc:
        artifact_write_ok = False
        payload["ok"] = False
        payload["message"] = f"Could not write scientist loop artifact: {exc}"

    memory_consolidation: dict[str, Any] | None = None
    if artifact_write_ok:
        try:
            memory_consolidation = get_scientist_memory_consolidation(session, root)
        except Exception as exc:  # pragma: no cover - defensive only
            memory_consolidation = {
                "ok": False,
                "tool": "scientist_memory_consolidation",
                "message": f"Could not consolidate scientist memory: {exc}",
                "records_added": 0,
                "records_total": 0,
                "artifact_path": str(root / ".xsci" / "scientist_memory_consolidation.json"),
                "memory_path": str(root / "experiments" / "evolution" / "retrospective_memory.json"),
                "no_training_started": True,
                "official_submit": "blocked_until_explicit_human_approval",
            }
        memory_step = {
            "step": "memory_writeback",
            "tool": "scientist_memory_consolidation",
            "status": "ok" if memory_consolidation.get("ok", True) else "blocked",
            "records_added": memory_consolidation.get("records_added", 0),
            "records_total": memory_consolidation.get("records_total", 0),
            "artifact_path": memory_consolidation.get("artifact_path"),
            "memory_path": memory_consolidation.get("memory_path"),
        }
        steps.append(memory_step)
        payload["steps"] = steps
        payload["memory_consolidation"] = memory_consolidation
        payload["memory_consolidation_artifact_path"] = memory_consolidation.get("artifact_path")
        payload["memory_path"] = memory_consolidation.get("memory_path")
        payload["memory_records_added"] = memory_consolidation.get("records_added", 0)
        payload["memory_records_total"] = memory_consolidation.get("records_total", 0)
        payload["next_safe_commands"] = [
            "evomind innovate-plan",
            "evomind self-audit",
            "evomind memory-consolidate",
        ]
        emit(
            "loop_memory_consolidation",
            "passed" if memory_consolidation.get("ok", True) else "blocked",
            "scientist_memory_consolidation",
            str(memory_consolidation.get("message") or "Scientist memory consolidation completed.")[:700],
            artifact=str(memory_consolidation.get("artifact_path") or ""),
            details={
                "records_added": memory_consolidation.get("records_added", 0),
                "records_total": memory_consolidation.get("records_total", 0),
                "memory_path": memory_consolidation.get("memory_path"),
            },
        )
        try:
            tmp = artifact_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(artifact_path)
        except OSError as exc:
            payload["ok"] = False
            payload["message"] = f"Could not update scientist loop artifact after memory writeback: {exc}"

    emit(
        "loop_complete",
        "completed",
        "scientist_loop",
        f"stop_reason={stop_reason}; lesson={lesson['lesson'][:520]}",
        artifact=str(artifact_path),
        details={"stop_reason": stop_reason, "lesson": lesson},
    )
    try:
        from .scientist_turns import record_scientist_turn

        record_scientist_turn(root, {
            "task": session.selected_task or "",
            "route": "scientist_loop",
            "user": "scientist_loop",
            "forced_tools": ["scientist_autopilot", "scientist_next_action"],
            "executed_tools": steps,
            "mode": payload.get("mode"),
            "decision": (autopilot.get("decision") if isinstance(autopilot, dict) else {}),
            "blockers": (autopilot.get("blockers") if isinstance(autopilot, dict) else []),
            "next_actions": (autopilot.get("next_actions") if isinstance(autopilot, dict) else []),
            "artifacts": [str(artifact_path), str(lessons_path)],
            "answer_preview": lesson["lesson"],
            "no_training_started": True,
        })
    except Exception:
        pass
    return payload


# ── Dispatcher ──────────────────────────────────────────────────────────

class TerminalTools:
    """Namespace / dispatcher for the terminal's lightweight tool set.

    Every tool is a plain function ``(session, root) -> dict``.  This class
    provides a single ``dispatch()`` entry point plus a ``list_tool_names()``
    helper for discoverability.
    """

    _tools = {
        "model_status": get_model_status,
        "system_status": get_system_status,
        "task_list": list_registered_tasks,
        "inspect_task": inspect_selected_task,
        "data_check": inspect_data_availability,
        "recent_run": inspect_recent_run,
        "gpu_status": inspect_gpu_status,
        "kaggle_status": inspect_kaggle_status,
        "dashboard": open_dashboard_url,
        "next_steps": explain_next_steps,
        "evolution_status": inspect_evolution_status,
        "scientist_checkpoint": get_scientist_checkpoint,
        "scientist_context_packet": get_scientist_context_packet,
        "research_decision": get_research_decision,
        "scientist_workplan": get_scientist_workplan,
        "scientist_turn_plan": get_scientist_turn_plan,
        "scientist_step_trace": get_scientist_step_trace,
        "scientist_self_audit": get_scientist_self_audit,
        "scientist_readiness_report": get_scientist_readiness_report,
        "scientist_causal_diagnosis": get_scientist_causal_diagnosis,
        "scientist_strategy_optimizer": get_scientist_strategy_optimizer,
        "scientist_upgrade_plan": get_scientist_upgrade_plan,
        "scientist_self_upgrade_loop": get_scientist_self_upgrade_loop,
        "scientist_patch_work_order": get_scientist_patch_work_order,
        "scientist_memory_consolidation": get_scientist_memory_consolidation,
        "scientist_innovation_backlog": get_scientist_innovation_backlog,
        "scientist_hypothesis_review": get_scientist_hypothesis_review,
        "scientist_experiment_blueprint": get_scientist_experiment_blueprint,
        "scientist_innovation_trial_feedback": get_scientist_innovation_trial_feedback,
        "scientist_situation_model": get_scientist_situation_model,
        "scientist_recovery": get_scientist_recovery_snapshot,
        "scientist_repair_plan": get_scientist_repair_plan,
        "scientist_execution_contract": get_scientist_execution_contract,
        "scientist_action_queue": get_scientist_action_queue,
        "scientist_continuation_status": get_scientist_continuation_status,
        "scientist_next_action": run_scientist_next_action,
        "scientist_continuation_resume": run_scientist_continuation_resume,
        "scientist_autopilot": run_scientist_autopilot,
        "scientist_loop": run_scientist_loop,
    }

    @classmethod
    def list_tool_names(cls) -> list[str]:
        return sorted(cls._tools.keys())

    @classmethod
    def dispatch(cls, name: str, session: SessionState, root: Path) -> dict[str, Any]:
        """Run the named tool and return its result dict.

        Returns ``{"ok": False, "tool": name, "message": "unknown tool"}`` for
        unknown tool names.
        """
        fn = cls._tools.get(name)
        if fn is None:
            return {"ok": False, "tool": name, "message": f"Unknown terminal tool: {name}"}
        try:
            return fn(session, root)
        except Exception as exc:
            return {"ok": False, "tool": name, "message": f"Tool '{name}' errored: {type(exc).__name__}: {exc}"}
