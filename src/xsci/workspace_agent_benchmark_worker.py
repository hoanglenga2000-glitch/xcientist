"""Subprocess boundary for the production workspace-agent benchmark."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from xsci.workspace_agent import WorkspaceAgentLimits, run_workspace_agent, write_workspace_result


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


def _benchmark_candidate_state(result: Mapping[str, Any]) -> str:
    if result.get("ok") is True and result.get("completed") is True and result.get("status") == "completed":
        return "worker_completed_candidate"
    if (
        result.get("ok") is False
        and result.get("completed") is False
        and result.get("status") == "format_validated_only"
        and result.get("stop_reason") == "behavioral_acceptance_missing"
        and result.get("epistemic_status") == "format_validated_only_not_behaviorally_validated"
    ):
        return "awaiting_parent_oracle"
    return "inadmissible"


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
    result = dict(run_workspace_agent(
        workspace,
        goal=str(request["goal"]),
        client=client,
        acceptance_commands=[str(item) for item in request.get("acceptance_commands", [])],
        allowed_edit_paths=[str(item) for item in request.get("allowed_edit_paths", [])],
        required_edit_paths=[str(item) for item in request.get("required_edit_paths", [])],
        require_post_patch_read=bool(request.get("require_post_patch_read")),
        allow_dynamic_behavioral_tests=bool(request.get("allow_dynamic_behavioral_tests", False)),
        artifact_dir=artifact_dir,
        limits=_request_limits(request.get("limits")),
    ))
    candidate_state = _benchmark_candidate_state(result)
    result["benchmark_candidate_state"] = candidate_state
    result["benchmark_parent_oracle_required"] = candidate_state == "awaiting_parent_oracle"
    return result


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
        write_workspace_result(result_path, payload)
        return 0
    except BaseException as exc:
        write_workspace_result(result_path, {
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
