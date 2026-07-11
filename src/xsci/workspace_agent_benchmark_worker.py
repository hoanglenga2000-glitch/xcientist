"""Subprocess boundary for the production workspace-agent benchmark."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from xsci.workspace_agent import WorkspaceAgentLimits, run_workspace_agent


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_factory(spec: str) -> Any:
    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError("client_factory must use module:qualified_name syntax")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError("client_factory did not resolve to a callable")
    return value()


def _request_limits(value: Any) -> WorkspaceAgentLimits | None:
    if value in (None, {}):
        return None
    if not isinstance(value, Mapping):
        raise TypeError("limits must be a JSON object")
    return WorkspaceAgentLimits(**{str(key): item for key, item in value.items()})


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    workspace = Path(str(request["workspace"])).resolve()
    artifact_dir = Path(str(request["artifact_dir"])).resolve()
    try:
        artifact_dir.relative_to(workspace)
    except ValueError:
        pass
    else:
        raise ValueError("artifact_dir must be outside the benchmark workspace")
    client_factory = str(request.get("client_factory") or "").strip()
    client = _load_factory(client_factory) if client_factory else None
    return run_workspace_agent(
        workspace,
        goal=str(request["goal"]),
        client=client,
        acceptance_commands=[str(item) for item in request.get("acceptance_commands", [])],
        allowed_edit_paths=[str(item) for item in request.get("allowed_edit_paths", [])],
        required_edit_paths=[str(item) for item in request.get("required_edit_paths", [])],
        require_post_patch_read=bool(request.get("require_post_patch_read")),
        artifact_dir=artifact_dir,
        limits=_request_limits(request.get("limits")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated EvoMind workspace-agent benchmark case.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    result_path = Path(args.result).resolve()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise TypeError("request document must be a JSON object")
        payload = run_request(request)
        _write_json(result_path, payload)
        return 0
    except BaseException as exc:
        _write_json(result_path, {
            "schema": "evomind.workspace_agent_benchmark_worker_error.v1",
            "ok": False,
            "completed": False,
            "status": "blocked",
            "stop_reason": "worker_exception",
            "error_type": type(exc).__name__,
            "message": str(exc)[:1600],
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
