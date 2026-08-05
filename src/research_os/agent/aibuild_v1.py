"""AIBuildAI-2-equivalent role graph for EvoMind Multi-Agent Core v1."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xsci.user_request import UserRequest

from .multi_agent import AgentRoleSpec, AgentTask, SupervisorRun, create_run

_POINTER_LOCK = threading.RLock()
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_id(value: str, label: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def default_roles() -> list[AgentRoleSpec]:
    """Hard role/tool/resource boundaries used by every AIBuild v1 run."""
    return [
        AgentRoleSpec(
            role="SetupAgent",
            capabilities=("environment_probe", "dataset_contract", "run_directory"),
            tool_whitelist=("inspect_workspace", "inspect_data", "write_run_artifact"),
            resource_permissions=("read_workspace", "write_run_directory"),
            output_contract=("setup.json", "data_contract.json"),
        ),
        AgentRoleSpec(
            role="ResearchLead",
            capabilities=("literature", "hypothesis", "prior_experience"),
            tool_whitelist=("literature_search", "read_memory", "write_run_artifact"),
            resource_permissions=("read_workspace", "network_read", "write_run_directory"),
            output_contract=("research_context.json",),
        ),
        AgentRoleSpec(
            role="DataAuditor",
            capabilities=("quality", "leakage", "statistics", "availability"),
            tool_whitelist=("inspect_data", "write_run_artifact"),
            resource_permissions=("read_dataset", "write_run_directory"),
            output_contract=("data_audit.json",),
        ),
        AgentRoleSpec(
            role="DesignerAgent",
            capabilities=("falsifiable_design", "metric_selection", "stop_conditions"),
            tool_whitelist=("read_memory", "read_dependency_evidence", "write_run_artifact"),
            resource_permissions=("read_run_evidence", "write_solution_repository"),
            output_contract=("design.json",),
        ),
        AgentRoleSpec(
            role="CoderAgent",
            capabilities=("implementation", "static_validation", "tests"),
            tool_whitelist=("read_dependency_evidence", "write_solution_code", "run_cpu_static_check"),
            resource_permissions=("read_run_evidence", "write_solution_repository", "local_cpu"),
            output_contract=("solution.py", "code_manifest.json"),
        ),
        AgentRoleSpec(
            role="TunerAgent",
            capabilities=("controlled_training", "telemetry", "artifact_recovery"),
            tool_whitelist=("hpc_probe", "hpc_run_manifest", "hpc_execute", "hpc_collect"),
            resource_permissions=("hpc_gateway", "write_solution_repository"),
            output_contract=("metrics.json", "oof_predictions.csv", "submission.csv", "training.log"),
            wall_time_seconds=1200,
        ),
        AgentRoleSpec(
            role="IndependentReviewer",
            capabilities=("leakage_review", "metric_provenance", "claim_audit", "gate_review"),
            tool_whitelist=("read_raw_artifact", "verify_hash", "audit_conclusion"),
            resource_permissions=("read_run_evidence",),
            output_contract=("review.json",),
        ),
        AgentRoleSpec(
            role="Aggregator",
            capabilities=("candidate_selection", "evidence_synthesis", "user_report"),
            tool_whitelist=("read_reviewed_result", "write_run_artifact"),
            resource_permissions=("read_reviewed_evidence", "write_run_directory"),
            output_contract=("artifact_manifest.json", "research_report.md"),
        ),
    ]


def should_use_multi_agent(request: UserRequest) -> bool:
    independent_actions = set(request.actions).intersection(
        {"inspect_data", "literature", "design", "compare", "train", "review", "candidate_submission", "report"}
    )
    return len(independent_actions) >= 2 or request.requests_execution or "review" in request.actions


def build_aibuild_run(request: UserRequest, *, task_id: str, run_id: str | None = None) -> SupervisorRun:
    """Build a dynamic DAG; execution is still performed by injected adapters."""
    task_id = _validate_id(task_id, "task_id")
    resolved_run_id = _validate_id(run_id or f"{task_id}_aibuild_{_utc_stamp()}_{uuid.uuid4().hex[:6]}", "run_id")
    tasks: list[AgentTask] = [
        AgentTask(
            task_id="setup",
            goal="Validate environment, dataset contract, dependencies, and run-directory boundary.",
            role="SetupAgent",
            priority=100,
            acceptance_criteria=("setup.json", "data_contract.json", "dataset_sha256"),
            payload={"dataset": request.dataset or task_id, "request": request.to_dict()},
        ),
        AgentTask(
            task_id="research_context",
            goal="Retrieve relevant prior experience and define evidence-grounded hypotheses.",
            role="ResearchLead",
            dependencies=("setup",),
            priority=80,
            acceptance_criteria=("research_context.json",),
        ),
        AgentTask(
            task_id="data_audit",
            goal="Audit data quality, missingness, leakage risk, label distribution, and availability.",
            role="DataAuditor",
            dependencies=("setup",),
            priority=90,
            acceptance_criteria=("data_audit.json",),
        ),
    ]

    solution_count = request.budget.solution_repositories if request.requests_execution else min(3, request.budget.solution_repositories)
    terminal_solution_tasks: list[str] = []
    for index in range(1, solution_count + 1):
        solution_id = f"solution_{index:02d}"
        design_id = f"{solution_id}_design"
        tasks.append(AgentTask(
            task_id=design_id,
            goal=f"Design falsifiable candidate {solution_id} with fixed CV and stop conditions.",
            role="DesignerAgent",
            dependencies=("research_context", "data_audit"),
            priority=70,
            acceptance_criteria=(f"solutions/{solution_id}/design.json",),
            solution_id=solution_id,
            payload={"solution_index": index, "solution_count": solution_count},
        ))
        if request.requests_execution:
            code_id = f"{solution_id}_code"
            tune_id = f"{solution_id}_train"
            tasks.extend([
                AgentTask(
                    task_id=code_id,
                    goal=f"Implement and statically validate {solution_id} in its isolated repository.",
                    role="CoderAgent",
                    dependencies=(design_id,),
                    priority=60,
                    acceptance_criteria=(f"solutions/{solution_id}/solution.py", f"solutions/{solution_id}/code_manifest.json"),
                    solution_id=solution_id,
                    payload={"solution_index": index},
                ),
                AgentTask(
                    task_id=tune_id,
                    goal=f"Execute {solution_id} through the controlled HPC runtime and collect raw telemetry.",
                    role="TunerAgent",
                    dependencies=(code_id,),
                    priority=50,
                    resource_type="hpc_gpu" if index == solution_count else "hpc_cpu",
                    acceptance_criteria=("metrics.json", "oof_predictions.csv", "submission.csv", "training.log"),
                    solution_id=solution_id,
                    timeout_seconds=request.budget.max_minutes * 60,
                    payload={"solution_index": index, "gpu_candidate": index == solution_count},
                ),
            ])
            terminal_solution_tasks.append(tune_id)
        else:
            terminal_solution_tasks.append(design_id)

    reviewer_dependencies = tuple(["data_audit", *terminal_solution_tasks])
    tasks.extend([
        AgentTask(
            task_id="independent_review",
            goal="Review raw logs, metrics, OOF, submissions, hashes, leakage, and claim boundaries without a parent summary.",
            role="IndependentReviewer",
            dependencies=reviewer_dependencies,
            priority=40,
            acceptance_criteria=("review.json", "all_required_artifacts_verified"),
            max_retries=0,
        ),
        AgentTask(
            task_id="aggregate",
            goal="Select only reviewed candidates and generate the user-facing report and artifact manifest.",
            role="Aggregator",
            dependencies=("independent_review",),
            priority=30,
            acceptance_criteria=("artifact_manifest.json", "research_report.md"),
            max_retries=0,
        ),
    ])
    run = create_run(
        objective=request.objective,
        tasks=tasks,
        roles=default_roles(),
        run_id=resolved_run_id,
        max_concurrency=request.budget.max_parallel,
    )
    run.gates = {
        "code_quality": "required",
        "reviewer": "required",
        "claim_audit": "required",
        "promotion": "required",
        "official_submission": request.submission_policy.official_submission,
    }
    run.resource_limits = {"cpu": 3, "hpc_cpu": 2, "hpc_gpu": request.budget.gpu_count, "gpu": request.budget.gpu_count}
    return run


def run_directory(workspace_root: str | Path, run_id: str) -> Path:
    return Path(workspace_root) / "workspace" / "evomind_runs" / _validate_id(run_id, "run_id")


def write_current_run_pointer(
    workspace_root: str | Path,
    *,
    task_id: str,
    run: SupervisorRun,
    run_dir: str | Path,
) -> Path:
    root = Path(workspace_root).resolve()
    task_id = _validate_id(task_id, "task_id")
    _validate_id(run.run_id, "run_id")
    pointer_path = root / "workspace" / "current_run.json"
    resolved_run_dir = Path(run_dir).resolve()
    expected_run_dir = run_directory(root, run.run_id).resolve()
    if resolved_run_dir != expected_run_dir:
        raise ValueError("current run directory does not match run_id")
    relative_dir = resolved_run_dir.relative_to(root)
    payload = {
        "schema": "evomind.current_run.v1",
        "task_id": task_id,
        "run_id": run.run_id,
        "run_dir": str(relative_dir).replace("\\", "/"),
        "status": run.status,
        "last_seq": run.seq,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    with _POINTER_LOCK:
        _atomic_json(pointer_path, payload)
    return pointer_path


def read_current_run_pointer(workspace_root: str | Path) -> dict[str, Any] | None:
    root = Path(workspace_root).resolve()
    path = root / "workspace" / "current_run.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != "evomind.current_run.v1":
        return None
    run_id = payload.get("run_id")
    task_id = payload.get("task_id")
    if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
        return None
    if not isinstance(task_id, str) or not _SAFE_ID.fullmatch(task_id):
        return None
    run_dir = payload.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        return None
    resolved = (root / run_dir).resolve()
    if resolved != run_directory(root, run_id).resolve():
        return None
    if not resolved.is_dir():
        return None
    try:
        run_payload = json.loads((resolved / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if run_payload.get("run_id") != run_id:
        return None
    return payload


__all__ = [
    "build_aibuild_run",
    "default_roles",
    "read_current_run_pointer",
    "run_directory",
    "should_use_multi_agent",
    "write_current_run_pointer",
]
