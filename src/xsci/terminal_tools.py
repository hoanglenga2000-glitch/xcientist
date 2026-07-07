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
from pathlib import Path
from typing import Any, Optional

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
