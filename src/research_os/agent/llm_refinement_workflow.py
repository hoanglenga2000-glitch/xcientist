"""Human-gated, version-preserving LLM refinement workflow.

The refinement run reuses the reviewed dataset and base-model contract from a
completed parent run, then continues training from the parent's Adapter.  It
creates a distinct run ledger and never mutates the parent run directory.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_os.hpc_runtime import HpcRuntime
from xsci.user_request import UserRequest

from .aibuild_v1 import run_directory, write_current_run_pointer
from .llm_finetune_workflow import (
    TASK_ID,
    LlmFinetuneExecutors,
    _artifact,
    _atomic_json,
    _atomic_text,
    _hpc_task_timeout_seconds,
    _now,
    _read_json,
    _sha256,
    _sha256_text,
    _training_source,
    llm_roles,
)
from .multi_agent import (
    AgentResult,
    AgentRoleSpec,
    AgentTask,
    HandoffEnvelope,
    MultiAgentStore,
    MultiAgentSupervisor,
    SupervisorRun,
    build_idempotency_key,
    create_run,
    validate_task_graph,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,180}$")
_PLAN_SCHEMA = "evomind.llm_refinement_request.v1"


def _validate_id(value: str, label: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


def _plans_root(root: Path, task_id: str = TASK_ID) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", task_id):
        raise ValueError("invalid task_id")
    return root / "workspace" / "tasks" / task_id / "refinements"


def _plan_path(root: Path, refinement_id: str, task_id: str = TASK_ID) -> Path:
    return _plans_root(root, task_id) / f"{_validate_id(refinement_id, 'refinement_id')}.json"


@contextmanager
def _plan_lock(root: Path, refinement_id: str, task_id: str = TASK_ID):
    """Serialize cross-process refinement state changes on Windows and POSIX."""
    plan_path = _plan_path(root, refinement_id, task_id)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = plan_path.with_suffix(".lock")
    deadline = time.monotonic() + 15.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120
                if stale:
                    lock_path.unlink()
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for the refinement state lock")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _verify_parent_manifest(parent_dir: Path) -> None:
    manifest = _read_json(parent_dir / "artifact_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("parent artifact manifest is missing")
    required = {
        "data/dataset_manifest.json",
        "llm_output/adapter/adapter_config.json",
        "llm_output/adapter/adapter_model.safetensors",
        "llm_output/adapter_reload.json",
        "llm_output/metrics.json",
        "review.json",
        "claim_audit.json",
    }
    indexed: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("path") or "").replace("\\", "/")
        if relative:
            indexed[relative] = item
    missing = sorted(required - indexed.keys())
    if missing:
        raise ValueError(f"parent artifact manifest is incomplete: {', '.join(missing)}")
    for relative in sorted(required):
        artifact_path = (parent_dir / relative).resolve()
        artifact_path.relative_to(parent_dir.resolve())
        expected = str(indexed[relative].get("sha256") or "")
        if not artifact_path.is_file() or not expected or _sha256(artifact_path) != expected:
            raise ValueError(f"parent artifact hash verification failed: {relative}")


def _load_parent(
    root: Path,
    parent_run_id: str,
    *,
    task_id: str = TASK_ID,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    parent_run_id = _validate_id(parent_run_id, "parent_run_id")
    parent_dir = run_directory(root, parent_run_id).resolve()
    parent_dir.relative_to((root / "workspace" / "evomind_runs").resolve())
    run = _read_json(parent_dir / "run.json")
    config = _read_json(parent_dir / "qlora_config.json")
    request = _read_json(parent_dir / "request.json")
    review = _read_json(parent_dir / "review.json")
    claim_audit = _read_json(parent_dir / "claim_audit.json")
    adapter_reload = _read_json(parent_dir / "llm_output" / "adapter_reload.json")
    gates = run.get("gates") if isinstance(run.get("gates"), dict) else {}
    checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
    if request.get("task_type") != "llm_finetune":
        raise ValueError("parent run is not an LLM fine-tuning run")
    if task_id != TASK_ID and request.get("task_id") != task_id and run.get("task_id") != task_id:
        raise ValueError("parent run does not belong to the requested task")
    if run.get("status") != "completed" or review.get("status") != "passed" or claim_audit.get("status") != "passed":
        raise ValueError("parent run must be completed and independently approved")
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("parent Reviewer checks are incomplete")
    if adapter_reload.get("passed") is not True or gates.get("adapter_reload") != "passed":
        raise ValueError("parent Adapter reload evidence is not approved")
    if gates.get("reviewer") != "passed" or gates.get("claim_audit") != "passed":
        raise ValueError("parent run gates are incomplete")
    if gates.get("model_publication") != "blocked":
        raise ValueError("parent model publication gate must remain blocked")
    compute_policy = request.get("compute_policy") if isinstance(request.get("compute_policy"), dict) else {}
    if compute_policy.get("local_gpu_allowed") is not False:
        raise ValueError("parent run does not prove local GPU was disabled")
    if not (parent_dir / "llm_output" / "adapter" / "adapter_model.safetensors").is_file():
        raise ValueError("parent Adapter evidence is missing")
    if not config:
        raise ValueError("parent QLoRA config is missing")
    _verify_parent_manifest(parent_dir)
    return parent_dir, run, config


def _next_version(
    root: Path,
    parent_run_id: str,
    parent_version: str,
    *,
    task_id: str = TASK_ID,
) -> str:
    match = re.fullmatch(r"V(\d+)", parent_version)
    versions = [int(match.group(1)) if match else 1]
    for path in _plans_root(root, task_id).glob("*.json"):
        payload = _read_json(path)
        if payload.get("parent_run_id") != parent_run_id:
            continue
        match = re.fullmatch(r"V(\d+)", str(payload.get("proposed_version") or ""))
        if match:
            versions.append(int(match.group(1)))
    return f"V{max(versions) + 1}"


def parse_refinement_changes(prompt: str, parent_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve a bounded natural-language refinement into explicit config changes."""
    source = prompt.strip()
    low = source.lower()
    changes: list[dict[str, Any]] = []

    learning_rate = float(parent_config.get("learning_rate") or 0.0002)
    exact_lr = re.search(r"(?:学习率|learning\s*rate)\s*(?:调整为|改为|设为|to|=|:)\s*(\d+(?:\.\d+)?(?:e-?\d+)?)", low)
    if exact_lr:
        new_lr = float(exact_lr.group(1))
        if not 0 < new_lr <= 0.01:
            raise ValueError("requested learning rate is outside the bounded refinement range")
        changes.append({"field": "learning_rate", "operation": "set", "old_value": learning_rate, "value": new_lr, "source": exact_lr.group(0)})
    elif any(term in low for term in ("降低学习率", "调低学习率", "减小学习率", "lower learning rate", "reduce learning rate")):
        changes.append({"field": "learning_rate", "operation": "multiply", "old_value": learning_rate, "value": learning_rate * 0.5, "factor": 0.5, "source": "natural_language_lower_learning_rate"})

    exact_epochs = re.search(r"(?:继续训练|训练|run|continue)(?:[^\d]{0,12})(\d+(?:\.\d+)?)\s*(?:个)?(?:周期|轮|epoch)", low)
    if exact_epochs:
        epochs = float(exact_epochs.group(1))
        if not 0 < epochs <= 3:
            raise ValueError("requested refinement epochs exceed the bounded range")
        changes.append({"field": "epochs", "operation": "set", "old_value": float(parent_config.get("epochs") or 2), "value": epochs, "source": exact_epochs.group(0)})
    elif any(term in low for term in ("短周期", "短训练", "short cycle", "short epoch")):
        changes.append({"field": "epochs", "operation": "set", "old_value": float(parent_config.get("epochs") or 2), "value": 1.0, "source": "natural_language_short_cycle"})

    deduplicated: dict[str, dict[str, Any]] = {str(item["field"]): item for item in changes}
    if not deduplicated:
        raise ValueError("no supported training-parameter change was found in the refinement request")
    return list(deduplicated.values())


def create_refinement_plan(
    workspace_root: str | Path,
    *,
    parent_run_id: str,
    prompt: str,
    refinement_id: str | None = None,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    parent_dir, _parent_run, parent_config = _load_parent(root, parent_run_id, task_id=task_id)
    if not prompt.strip():
        raise ValueError("refinement prompt is required")
    resolved_id = _validate_id(refinement_id or f"refine_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}", "refinement_id")
    changes = parse_refinement_changes(prompt, parent_config)
    parent_version_path = parent_dir / "version.json"
    parent_version_record = _read_json(parent_version_path) if parent_version_path.is_file() else {}
    parent_version = str(parent_version_record.get("version") or "V1")
    payload = {
        "schema": _PLAN_SCHEMA,
        "refinement_id": resolved_id,
        "task_id": task_id,
        "parent_run_id": parent_run_id,
        "parent_version": parent_version,
        "proposed_version": _next_version(
            root,
            parent_run_id,
            parent_version,
            task_id=task_id,
        ),
        "prompt": prompt.strip(),
        "status": "awaiting_human_gate",
        "requested_changes": changes,
        "preserved": ["base_model", "source_isolated_dataset", "fixed_test_set", "data_audit", "parent_adapter", "review_contract"],
        "affected_steps": ["refinement_design", "hpc_train", "evaluate", "version_compare", "independent_review", "claim_audit", "scientific_report"],
        "rerun_policy": "only_affected_steps",
        "estimated_runtime_minutes": None,
        "gate": {"decision": "pending", "decided_at": None},
        "child_run_id": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    _atomic_json(_plan_path(root, resolved_id, task_id), payload)
    return payload


def read_refinement_plan(workspace_root: str | Path, refinement_id: str, *, task_id: str = TASK_ID) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    path = _plan_path(root, refinement_id, task_id)
    payload = _read_json(path)
    if payload.get("schema") != _PLAN_SCHEMA:
        raise FileNotFoundError(refinement_id)
    child_run_id = str(payload.get("child_run_id") or "")
    if child_run_id:
        child_dir = run_directory(root, _validate_id(child_run_id, "run_id"))
        child_run_path = child_dir / "run.json"
        if child_run_path.is_file():
            child_run = _read_json(child_run_path)
            child_status = str(child_run.get("status") or "")
            if child_status in {"completed", "needs_continuation", "cancelled", "failed"}:
                reconciled_status = _plan_status_for_run(child_status)
                if payload.get("status") != reconciled_status:
                    payload["status"] = reconciled_status
                    comparison = child_dir / "version_comparison.json"
                    if comparison.is_file():
                        payload["comparison_path"] = str(comparison.relative_to(root)).replace("\\", "/")
                    payload["updated_at"] = _now()
                    _atomic_json(path, payload)
                # The child copy is a read model used by the API/UI. Reconcile
                # it even when the durable plan already has the terminal state.
                _sync_child_refinement(child_dir, payload)
    return payload


def latest_refinement_plan(workspace_root: str | Path, *, task_id: str = TASK_ID) -> dict[str, Any] | None:
    root = Path(workspace_root).resolve()
    paths = sorted(_plans_root(root, task_id).glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths:
        payload = _read_json(path)
        if payload.get("schema") == _PLAN_SCHEMA:
            return payload
    return None


def decide_refinement_plan(
    workspace_root: str | Path,
    refinement_id: str,
    decision: str,
    *,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    resolved = "approved" if decision == "approve" else "rejected"
    with _plan_lock(root, refinement_id, task_id):
        payload = read_refinement_plan(root, refinement_id, task_id=task_id)
        current = str(payload.get("status") or "")
        current_decision = str((payload.get("gate") or {}).get("decision") or "")
        if current_decision == resolved and current in {resolved, "starting", "running", "completed", "needs_continuation"}:
            return payload
        if current != "awaiting_human_gate":
            raise ValueError("refinement is no longer awaiting a Human Gate decision")
        payload["status"] = resolved
        payload["gate"] = {"decision": resolved, "decided_at": _now()}
        payload["updated_at"] = _now()
        _atomic_json(_plan_path(root, refinement_id, task_id), payload)
        return payload


def reserve_refinement_run(
    workspace_root: str | Path,
    refinement_id: str,
    proposed_run_id: str,
    *,
    task_id: str = TASK_ID,
    reclaim_after_seconds: int = 120,
) -> tuple[dict[str, Any], bool]:
    """Reserve one durable child run id and report whether this caller should launch it."""
    root = Path(workspace_root).resolve()
    proposed_run_id = _validate_id(proposed_run_id, "run_id")
    with _plan_lock(root, refinement_id, task_id):
        payload = read_refinement_plan(root, refinement_id, task_id=task_id)
        if (payload.get("gate") or {}).get("decision") != "approved":
            raise ValueError("Human Gate approval is required before run reservation")
        current_child = str(payload.get("child_run_id") or "")
        should_start = False
        if not current_child:
            current_child = proposed_run_id
            payload["child_run_id"] = current_child
            payload["status"] = "starting"
            payload["launch_reserved_at"] = _now()
            payload["launch_attempts"] = int(payload.get("launch_attempts") or 0) + 1
            should_start = True
        elif payload.get("status") == "starting":
            child_dir = run_directory(root, current_child)
            reserved_at = str(payload.get("launch_reserved_at") or "")
            try:
                reserved_time = datetime.fromisoformat(reserved_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                reserved_time = 0.0
            if not child_dir.exists() and time.time() - reserved_time >= max(1, reclaim_after_seconds):
                payload["launch_reserved_at"] = _now()
                payload["launch_attempts"] = int(payload.get("launch_attempts") or 0) + 1
                should_start = True
        payload["updated_at"] = _now()
        _atomic_json(_plan_path(root, refinement_id, task_id), payload)
        return payload, should_start


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _prepare_child_files(parent_dir: Path, child_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json"):
        _link_or_copy(parent_dir / "data" / name, child_dir / "data" / name)
    _link_or_copy(parent_dir / "data_audit.json", child_dir / "data_audit.json")
    parent_request = _read_json(parent_dir / "request.json")
    parent_request["objective"] = str(plan["prompt"])
    parent_request["source_text"] = str(plan["prompt"])
    parent_request["parent_run_id"] = str(plan["parent_run_id"])
    _atomic_json(child_dir / "request.json", parent_request)

    source_path = child_dir / "code" / "train_qlora.py"
    _atomic_text(source_path, _training_source())
    compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
    requirements = parent_dir / "code" / "requirements.lock"
    if requirements.is_file():
        _link_or_copy(requirements, child_dir / "code" / "requirements.lock")
    _atomic_json(
        child_dir / "code_manifest.json",
        {
            "schema": "evomind.llm_refinement_code_manifest.v1",
            "source_sha256": _sha256(source_path),
            "static_check": "py_compile_passed",
            "local_gpu_used": False,
            "parent_run_id": plan["parent_run_id"],
            "generated_at": _now(),
        },
    )
    config = _read_json(parent_dir / "qlora_config.json")
    for change in plan["requested_changes"]:
        config[str(change["field"])] = change["value"]
    config.update(
        {
            "schema": "evomind.qlora_refinement_config.v1",
            "parent_run_id": plan["parent_run_id"],
            "parent_adapter_subdir": "parent_adapter",
            "refinement_id": plan["refinement_id"],
            "version": plan["proposed_version"],
            "acceptance": {
                **dict(config.get("acceptance") or {}),
                "domain_composite_improvement_pp": -1.0,
                "comparison_baseline": "parent_adapter",
                "max_material_regression_pp": 1.0,
            },
        }
    )
    return config


def _sync_child_refinement(child_dir: Path, plan: dict[str, Any]) -> None:
    """Keep the child-visible refinement contract aligned with the durable plan."""
    if child_dir.is_dir():
        refinement_path = child_dir / "refinement.json"
        _atomic_json(refinement_path, plan)
        manifest_path = child_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = _read_json(manifest_path)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            return
        refreshed = False
        for index, item in enumerate(artifacts):
            if not isinstance(item, dict) or str(item.get("path") or "").replace("\\", "/") != "refinement.json":
                continue
            artifacts[index] = _artifact(
                refinement_path,
                child_dir,
                kind=str(item.get("kind") or refinement_path.name),
            )
            refreshed = True
            break
        if refreshed:
            manifest["artifacts"] = artifacts
            manifest["generated_at"] = _now()
            _atomic_json(manifest_path, manifest)


def _build_refinement_run(plan: dict[str, Any], run_id: str, request: UserRequest) -> SupervisorRun:
    tasks = [
        AgentTask("refinement_design", "Resolve the approved change set while preserving the parent evidence contract.", "TrainingDesigner", priority=100, acceptance_criteria=("refinement.json", "version.json", "qlora_config.json")),
        AgentTask("hpc_train", "Continue the reviewed parent Adapter on the remote HPC GPU.", "HpcRuntimeAgent", dependencies=("refinement_design",), priority=80, resource_type="hpc_gpu", timeout_seconds=_hpc_task_timeout_seconds(request), acceptance_criteria=("llm_output/adapter", "llm_output/metrics.json", "llm_output/telemetry.jsonl")),
        AgentTask("evaluate", f"Compare {plan['proposed_version']} with the preserved {plan['parent_version']} Adapter on the unchanged fixed test set.", "EvaluatorAgent", dependencies=("hpc_train",), priority=60, acceptance_criteria=("evaluation_summary.json",)),
        AgentTask("version_compare", f"Materialize the {plan['parent_version']}-to-{plan['proposed_version']} comparison directly from both fixed-test metric ledgers.", "VersionComparatorAgent", dependencies=("evaluate",), priority=55, max_retries=0, acceptance_criteria=("version_comparison.json",)),
        AgentTask("independent_review", f"Independently recompute and review raw {plan['proposed_version']} logs, metrics, hashes, telemetry and parent comparison.", "IndependentReviewer", dependencies=("version_compare",), priority=50, max_retries=0, acceptance_criteria=("review.json",)),
        AgentTask("claim_audit", f"Audit the bounded {plan['parent_version']}-to-{plan['proposed_version']} comparison and publication claims.", "ClaimAuditAgent", dependencies=("independent_review",), priority=40, max_retries=0, acceptance_criteria=("claim_audit.json",)),
        AgentTask("synthesis", f"Deliver the reviewed {plan['proposed_version']} Adapter and updated evidence package.", "SynthesisAgent", dependencies=("claim_audit",), priority=30, max_retries=0, acceptance_criteria=("artifact_manifest.json", "research_report.md")),
    ]
    roles = [
        *llm_roles().values(),
        AgentRoleSpec(
            "VersionComparatorAgent",
            ("fixed_test_comparison", "lineage_validation"),
            ("read_run_evidence",),
            ("read_run_evidence", "write_run_artifacts"),
            output_contract=("version_comparison.json",),
        ),
    ]
    run = create_run(objective=str(plan["prompt"]), tasks=tasks, roles=roles, run_id=run_id, max_concurrency=1)
    run.gates = {"refinement_human_gate": "passed", "hpc_execution": "required", "reviewer": "required", "claim_audit": "required", "parent_run_preserved": "required", "model_publication": "blocked"}
    run.resource_limits = {"cpu": 1, "hpc_gpu": 1, "gpu": 1}
    return run


def _ensure_version_comparison_stage(run: SupervisorRun, plan: dict[str, Any]) -> None:
    """Migrate resumable pre-comparison runs without creating another child run."""
    role = AgentRoleSpec(
        "VersionComparatorAgent",
        ("fixed_test_comparison", "lineage_validation"),
        ("read_run_evidence",),
        ("read_run_evidence", "write_run_artifacts"),
        output_contract=("version_comparison.json",),
    )
    run.roles.setdefault(role.role, role)
    if "version_compare" not in run.tasks:
        task = AgentTask(
            "version_compare",
            f"Materialize the {plan['parent_version']}-to-{plan['proposed_version']} comparison directly from both fixed-test metric ledgers.",
            "VersionComparatorAgent",
            dependencies=("evaluate",),
            priority=55,
            max_retries=0,
            acceptance_criteria=("version_comparison.json",),
        )
        task.payload["config_hash"] = _sha256_text(
            f"{plan.get('refinement_id', '')}\0{task.task_id}\0{task.goal}"
        )
        task.idempotency_key = build_idempotency_key(
            task_id=task.task_id,
            run_id=run.run_id,
            config_hash=str(task.payload["config_hash"]),
        )
        run.tasks[task.task_id] = task
    reviewer = run.tasks.get("independent_review")
    if reviewer is not None:
        reviewer.dependencies = ("version_compare",)
    validate_task_graph(run.tasks, run.roles)


class RefinementExecutors(LlmFinetuneExecutors):
    def __init__(self, *, plan: dict[str, Any], resolved_config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.plan = plan
        self.resolved_config = resolved_config

    def refinement_design(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        config_path = self.run_dir / "qlora_config.json"
        plan_path = self.run_dir / "refinement.json"
        version_path = self.run_dir / "version.json"
        gate_path = self.run_dir / "human_gate.json"
        _atomic_json(config_path, self.resolved_config)
        _atomic_json(plan_path, self.plan)
        _atomic_json(version_path, {"schema": "evomind.run_version.v1", "version": self.plan["proposed_version"], "parent_run_id": self.plan["parent_run_id"], "refinement_id": self.plan["refinement_id"], "parent_preserved": True, "created_at": _now()})
        _atomic_json(gate_path, {"schema": "evomind.human_gate.v1", "gate": "llm_refinement", "decision": "approved", "refinement_id": self.plan["refinement_id"], "decided_at": self.plan["gate"]["decided_at"]})
        artifacts = [_artifact(path, self.run_dir, kind=path.name) for path in (config_path, plan_path, version_path, gate_path)]
        self.store.emit(run, "llm.refinement.approved", task_id=task.task_id, agent=task.role, status="completed", parent_run_id=self.plan["parent_run_id"], version=self.plan["proposed_version"], requested_changes=self.plan["requested_changes"])
        return AgentResult(task.task_id, "Applied the approved bounded refinement without changing the parent run", [item["path"] for item in artifacts], artifacts=artifacts, confidence=1.0)

    def version_compare(self, task: AgentTask, _handoff: HandoffEnvelope, run: SupervisorRun) -> AgentResult:
        parent_dir = run_directory(self.workspace_root, str(self.plan["parent_run_id"]))
        comparison_path = self.run_dir / "version_comparison.json"
        comparison = _comparison(parent_dir, self.run_dir, self.plan, "awaiting_independent_review")
        _atomic_json(comparison_path, comparison)
        artifact = _artifact(comparison_path, self.run_dir, kind="version_comparison")
        self.store.emit(
            run,
            "llm.refinement.version_comparison.created",
            task_id=task.task_id,
            agent=task.role,
            status="completed",
            comparison_sha256=artifact["sha256"],
            parent_run_id=self.plan["parent_run_id"],
            child_run_id=run.run_id,
        )
        return AgentResult(
            task.task_id,
            f"Created the immutable {self.plan['parent_version']}-to-{self.plan['proposed_version']} comparison from both fixed-test metric ledgers",
            [artifact["path"]],
            metrics={"outcome": comparison["outcome"], "delta_pp": comparison["delta_pp"]},
            artifacts=[artifact],
            confidence=1.0,
        )

    def additional_review_evidence(
        self, run: SupervisorRun
    ) -> tuple[dict[str, bool], list[dict[str, Any]], str]:
        parent_dir = run_directory(self.workspace_root, str(self.plan["parent_run_id"]))
        comparison_path = self.run_dir / "version_comparison.json"
        comparison = _read_json(comparison_path)
        expected = _comparison(parent_dir, self.run_dir, self.plan, "awaiting_independent_review")
        checked_fields = (
            "schema",
            "parent_run_id",
            "child_run_id",
            "parent_version",
            "child_version",
            "outcome",
            "metric",
            "v1",
            "v2",
            "delta_pp",
            "requested_changes",
            "parent_preserved",
            "parent_metrics_sha256",
            "child_metrics_sha256",
        )
        values_match = comparison_path.is_file() and all(
            comparison.get(field) == expected.get(field) for field in checked_fields
        )
        artifact = _artifact(comparison_path, self.run_dir, kind="version_comparison") if comparison_path.is_file() else {}
        reviewed_artifacts = [artifact] if artifact else []
        return (
            {
                "version_comparison_recomputed": values_match,
                "parent_run_preserved": parent_dir.is_dir() and comparison.get("parent_preserved") is True,
            },
            reviewed_artifacts,
            "version_comparison",
        )

    def mapping(self):
        mapping = super().mapping()
        mapping["TrainingDesigner"] = self.refinement_design
        mapping["VersionComparatorAgent"] = self.version_compare
        return mapping


def _comparison(parent_dir: Path, child_dir: Path, plan: dict[str, Any], child_status: str) -> dict[str, Any]:
    parent_metrics = _read_json(parent_dir / "llm_output" / "metrics.json")
    child_metrics = _read_json(child_dir / "llm_output" / "metrics.json")
    parent_score = (parent_metrics.get("after") or {}).get("domain_composite")
    child_score = (child_metrics.get("after") or {}).get("domain_composite")
    delta = float(child_score) - float(parent_score) if isinstance(parent_score, (int, float)) and isinstance(child_score, (int, float)) else None
    if delta is None:
        outcome = "not_available"
    elif delta > 0.25:
        outcome = "improved"
    elif delta < -1.0:
        outcome = "trade_off_detected"
    else:
        outcome = "no_material_change"
    return {
        "schema": "evomind.llm_version_comparison.v1",
        "parent_run_id": plan["parent_run_id"],
        "child_run_id": child_dir.name,
        "parent_version": plan["parent_version"],
        "child_version": plan["proposed_version"],
        "status": child_status,
        "outcome": outcome,
        "metric": "fixed_test_domain_composite",
        "v1": parent_score,
        "v2": child_score,
        "delta_pp": delta,
        "requested_changes": plan["requested_changes"],
        "parent_preserved": parent_dir.is_dir(),
        "parent_metrics_sha256": _sha256(parent_dir / "llm_output" / "metrics.json"),
        "child_metrics_sha256": _sha256(child_dir / "llm_output" / "metrics.json") if (child_dir / "llm_output" / "metrics.json").is_file() else None,
        "generated_by": "VersionComparatorAgent",
        "generated_at": _now(),
    }


def run_llm_refinement(
    workspace_root: str | Path,
    refinement_id: str,
    *,
    run_id: str | None = None,
    task_id: str = TASK_ID,
) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    plan = read_refinement_plan(root, refinement_id, task_id=task_id)
    if plan.get("status") not in {"approved", "starting"} or (plan.get("gate") or {}).get("decision") != "approved":
        raise ValueError("Human Gate approval is required before refinement execution")
    parent_dir, _parent_run, _parent_config = _load_parent(root, str(plan["parent_run_id"]), task_id=task_id)
    resolved_run_id = _validate_id(run_id or f"qwen7b_refine_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}", "run_id")
    reserved_run_id = str(plan.get("child_run_id") or "")
    if reserved_run_id and reserved_run_id != resolved_run_id:
        raise ValueError("run_id does not match the reserved refinement child")
    child_dir = run_directory(root, resolved_run_id)
    if child_dir.exists():
        raise FileExistsError(resolved_run_id)
    child_dir.mkdir(parents=True)
    config = _prepare_child_files(parent_dir, child_dir, plan)
    request = UserRequest.from_dict(_read_json(child_dir / "request.json"))
    run = _build_refinement_run(plan, resolved_run_id, request)
    request_hash = _sha256_text(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    for agent_task in run.tasks.values():
        agent_task.payload["config_hash"] = _sha256_text(f"{request_hash}\0{agent_task.task_id}\0{agent_task.goal}")
    store = MultiAgentStore(child_dir, on_save=lambda saved: write_current_run_pointer(root, task_id=task_id, run=saved, run_dir=child_dir))
    runtime = HpcRuntime(run_id=run.run_id, local_run_dir=child_dir, timeout_seconds=request.budget.max_minutes * 60)
    executors = RefinementExecutors(plan=plan, resolved_config=config, workspace_root=root, run_dir=child_dir, request=request, runtime=runtime, store=store)
    plan["status"] = "running"
    plan["child_run_id"] = run.run_id
    plan["updated_at"] = _now()
    _atomic_json(_plan_path(root, refinement_id, task_id), plan)
    _sync_child_refinement(child_dir, plan)
    store.append_message(run, sender="user", receiver="ExecutiveSupervisor", content=str(plan["prompt"]))
    write_current_run_pointer(root, task_id=task_id, run=run, run_dir=child_dir)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    try:
        result = supervisor.run_until_blocked()
    finally:
        write_current_run_pointer(root, task_id=task_id, run=run, run_dir=child_dir)
    plan["status"] = _plan_status_for_run(result.status)
    comparison_path = child_dir / "version_comparison.json"
    if comparison_path.is_file() and run.tasks["version_compare"].status == "completed":
        plan["comparison_path"] = str(comparison_path.relative_to(root)).replace("\\", "/")
    plan["updated_at"] = _now()
    _atomic_json(_plan_path(root, refinement_id, task_id), plan)
    _sync_child_refinement(child_dir, plan)
    return result


def resume_llm_refinement(workspace_root: str | Path, run_id: str) -> SupervisorRun:
    root = Path(workspace_root).resolve()
    child_dir = run_directory(root, _validate_id(run_id, "run_id"))
    version = _read_json(child_dir / "version.json")
    refinement_id = str(version.get("refinement_id") or "")
    refinement = _read_json(child_dir / "refinement.json")
    task_id = str(refinement.get("task_id") or TASK_ID)
    plan = read_refinement_plan(root, refinement_id, task_id=task_id)
    parent_dir, _parent_run, _parent_config = _load_parent(root, str(plan["parent_run_id"]), task_id=task_id)
    config = _read_json(child_dir / "qlora_config.json")
    request = UserRequest.from_dict(_read_json(child_dir / "request.json"))
    store = MultiAgentStore(child_dir, on_save=lambda saved: write_current_run_pointer(root, task_id=task_id, run=saved, run_dir=child_dir))
    run = store.load()
    _ensure_version_comparison_stage(run, plan)
    store.save(run)
    runtime = HpcRuntime(run_id=run.run_id, local_run_dir=child_dir, timeout_seconds=request.budget.max_minutes * 60)
    executors = RefinementExecutors(plan=plan, resolved_config=config, workspace_root=root, run_dir=child_dir, request=request, runtime=runtime, store=store)
    supervisor = MultiAgentSupervisor(run, store, executors.mapping())
    plan["status"] = "running"
    plan["updated_at"] = _now()
    _atomic_json(_plan_path(root, refinement_id, task_id), plan)
    _sync_child_refinement(child_dir, plan)
    supervisor.resume(retry_failed=run.status in {"needs_continuation", "running"})
    result = supervisor.run_until_blocked()
    plan["status"] = _plan_status_for_run(result.status)
    comparison_path = child_dir / "version_comparison.json"
    if comparison_path.is_file() and run.tasks.get("version_compare") and run.tasks["version_compare"].status == "completed":
        plan["comparison_path"] = str(comparison_path.relative_to(root)).replace("\\", "/")
    plan["updated_at"] = _now()
    _atomic_json(_plan_path(root, refinement_id, task_id), plan)
    _sync_child_refinement(child_dir, plan)
    return result


def _plan_status_for_run(run_status: str) -> str:
    if run_status in {"completed", "needs_continuation", "cancelled"}:
        return run_status
    return "failed"


__all__ = [
    "create_refinement_plan",
    "decide_refinement_plan",
    "latest_refinement_plan",
    "parse_refinement_changes",
    "read_refinement_plan",
    "reserve_refinement_run",
    "resume_llm_refinement",
    "run_llm_refinement",
]
