"""Persistent Multi-Agent Core v1.

This module owns protocol and scheduling state only. Role executors are
injected adapters (normally restricted ``AgentSession`` instances or the HPC
runtime), which keeps one canonical model/tool loop and makes the scheduler
testable without providers or GPUs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SCHEMA_PREFIX = "evomind.multi_agent"
TERMINAL_TASK_STATES = {"completed", "failed", "cancelled", "skipped"}
TASK_TRANSITIONS = {
    "pending": {"ready", "cancelled", "skipped"},
    "ready": {"running", "cancelled", "skipped"},
    "running": {"completed", "failed", "retry_wait", "cancelled"},
    "retry_wait": {"ready", "cancelled"},
    "failed": {"ready"},
    "completed": set(),
    "cancelled": set(),
    "skipped": {"pending"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_jsonable)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        if value and not value.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def build_idempotency_key(
    *,
    task_id: str,
    run_id: str,
    solution_id: str = "",
    data_hash: str = "",
    code_hash: str = "",
    config_hash: str = "",
) -> str:
    raw = "\0".join((task_id, run_id, solution_id, data_hash, code_hash, config_hash))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentRoleSpec:
    role: str
    capabilities: tuple[str, ...]
    tool_whitelist: tuple[str, ...] = ()
    resource_permissions: tuple[str, ...] = ("read_workspace",)
    max_turns: int = 8
    token_budget: int = 0
    wall_time_seconds: int = 1200
    output_contract: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


@dataclass
class AgentTask:
    task_id: str
    goal: str
    role: str
    dependencies: tuple[str, ...] = ()
    priority: int = 0
    resource_type: str = "cpu"
    acceptance_criteria: tuple[str, ...] = ()
    status: str = "pending"
    attempts: int = 0
    max_retries: int = 1
    timeout_seconds: int = 1200
    idempotency_key: str = ""
    solution_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result_ref: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dependencies"] = list(self.dependencies)
        payload["acceptance_criteria"] = list(self.acceptance_criteria)
        return payload


@dataclass(frozen=True)
class HandoffEnvelope:
    handoff_id: str
    run_id: str
    task_id: str
    sender: str
    receiver: str
    input_evidence_refs: tuple[str, ...]
    expected_output: tuple[str, ...]
    budget: dict[str, Any]
    deadline: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_evidence_refs"] = list(self.input_evidence_refs)
        payload["expected_output"] = list(self.expected_output)
        return payload


@dataclass
class AgentResult:
    task_id: str
    conclusion: str
    evidence_refs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    confidence: float = 0.0
    failure_type: str = ""
    accepted: bool = True
    followup_tasks: list[AgentTask] = field(default_factory=list)
    dependency_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TaskExecutionFailure(RuntimeError):
    def __init__(self, failure_type: str) -> None:
        super().__init__(failure_type)
        self.failure_type = failure_type


def _classify_task_failure(exc: Exception) -> str:
    explicit = getattr(exc, "failure_type", "")
    if isinstance(explicit, str) and explicit:
        return explicit
    message = f"{type(exc).__name__}: {exc}".lower()
    hpc_match = next(
        (
            failure
            for failure in (
                "connection_auth",
                "connection",
                "dependency",
                "timeout",
                "oom",
                "resource",
                "evidence",
                "code",
            )
            if f"hpc_{failure}" in message
        ),
        "",
    )
    if hpc_match:
        return hpc_match
    if isinstance(exc, (EOFError, ConnectionError)):
        return "connection"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, MemoryError):
        return "oom"
    if any(term in message for term in ("socks", "ssh", "banner", "socket", "network unreachable")):
        return "connection"
    if "provider" in message:
        return "provider"
    return type(exc).__name__


@dataclass
class SupervisorRun:
    run_id: str
    objective: str
    tasks: dict[str, AgentTask]
    roles: dict[str, AgentRoleSpec]
    status: str = "created"
    max_concurrency: int = 3
    max_spawn_depth: int = 2
    parent_max_turns: int = 40
    child_max_turns: int = 8
    seq: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    gates: dict[str, Any] = field(default_factory=dict)
    resource_limits: dict[str, int] = field(default_factory=lambda: {"gpu": 1, "hpc_gpu": 1})
    open_requirements: list[str] = field(default_factory=list)
    next_action: str = "validate_task_graph"
    idempotency_results: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_PREFIX}.run.v1",
            "run_id": self.run_id,
            "objective": self.objective,
            "status": self.status,
            "max_concurrency": self.max_concurrency,
            "max_spawn_depth": self.max_spawn_depth,
            "parent_max_turns": self.parent_max_turns,
            "child_max_turns": self.child_max_turns,
            "seq": self.seq,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "gates": self.gates,
            "resource_limits": self.resource_limits,
            "open_requirements": self.open_requirements,
            "next_action": self.next_action,
            "idempotency_results": self.idempotency_results,
            "roles": {key: value.to_dict() for key, value in self.roles.items()},
            "tasks": {key: value.to_dict() for key, value in self.tasks.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorRun":
        roles = {
            key: AgentRoleSpec(
                role=str(value["role"]),
                capabilities=tuple(value.get("capabilities") or ()),
                tool_whitelist=tuple(value.get("tool_whitelist") or ()),
                resource_permissions=tuple(value.get("resource_permissions") or ()),
                max_turns=int(value.get("max_turns", 8)),
                token_budget=int(value.get("token_budget", 0)),
                wall_time_seconds=int(value.get("wall_time_seconds", 1200)),
                output_contract=tuple(value.get("output_contract") or ()),
            )
            for key, value in dict(payload.get("roles") or {}).items()
        }
        tasks = {
            key: AgentTask(
                **{
                    **dict(value),
                    "dependencies": tuple(value.get("dependencies") or ()),
                    "acceptance_criteria": tuple(value.get("acceptance_criteria") or ()),
                }
            )
            for key, value in dict(payload.get("tasks") or {}).items()
        }
        return cls(
            run_id=str(payload["run_id"]),
            objective=str(payload.get("objective") or ""),
            tasks=tasks,
            roles=roles,
            status=str(payload.get("status") or "created"),
            max_concurrency=int(payload.get("max_concurrency", 3)),
            max_spawn_depth=int(payload.get("max_spawn_depth", 2)),
            parent_max_turns=int(payload.get("parent_max_turns", 40)),
            child_max_turns=int(payload.get("child_max_turns", 8)),
            seq=int(payload.get("seq", 0)),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
            gates=dict(payload.get("gates") or {}),
            resource_limits={key: int(value) for key, value in dict(payload.get("resource_limits") or {"gpu": 1, "hpc_gpu": 1}).items()},
            open_requirements=list(payload.get("open_requirements") or []),
            next_action=str(payload.get("next_action") or ""),
            idempotency_results=dict(payload.get("idempotency_results") or {}),
        )


class MultiAgentStore:
    """Crash-survivable run state and append-only ledgers."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        on_save: Callable[[SupervisorRun], None] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._on_save = on_save

    def save(self, run: SupervisorRun) -> None:
        with self._lock:
            run.updated_at = _now()
            _atomic_json(self.run_dir / "run.json", run.to_dict())
            graph = {
                "schema": f"{SCHEMA_PREFIX}.task_graph.v1",
                "run_id": run.run_id,
                "nodes": [task.to_dict() for task in run.tasks.values()],
                "edges": [
                    {"from": dependency, "to": task.task_id}
                    for task in run.tasks.values()
                    for dependency in task.dependencies
                ],
            }
            _atomic_json(self.run_dir / "task_graph.json", graph)
            if self._on_save is not None:
                self._on_save(run)

    def load(self) -> SupervisorRun:
        payload = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        return SupervisorRun.from_dict(payload)

    def _append(self, filename: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            path = self.run_dir / filename
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(dict(payload), ensure_ascii=False, default=_jsonable) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def emit(self, run: SupervisorRun, kind: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            run.seq += 1
            event = {
                "schema": f"{SCHEMA_PREFIX}.{kind}.v1",
                "seq": run.seq,
                "ts": _now(),
                "run_id": run.run_id,
                "task_id": "supervisor",
                "agent": "ExecutiveSupervisor",
                "status": run.status,
                "evidence_refs": [],
                "artifact_hashes": [],
                **payload,
            }
            self._append("events.jsonl", event)
            self.save(run)
            return event

    def append_message(self, run: SupervisorRun, *, sender: str, receiver: str, content: str, task_id: str = "") -> None:
        self._append("messages.jsonl", {
            "schema": f"{SCHEMA_PREFIX}.message.v1",
            "ts": _now(),
            "run_id": run.run_id,
            "task_id": task_id,
            "sender": sender,
            "receiver": receiver,
            "content": content,
        })

    def append_handoff(self, handoff: HandoffEnvelope) -> None:
        self._append("handoffs.jsonl", {"schema": f"{SCHEMA_PREFIX}.handoff.v1", **handoff.to_dict()})

    def write_result(self, result: AgentResult) -> str:
        path = self.run_dir / "results" / f"{result.task_id}.json"
        _atomic_json(path, {"schema": f"{SCHEMA_PREFIX}.result.v1", **result.to_dict()})
        return str(path.relative_to(self.run_dir)).replace("\\", "/")

    def write_failure_bundle(
        self,
        run: SupervisorRun,
        task: AgentTask,
        exc: Exception,
        *,
        failure_type: str,
        retry_scheduled: bool,
    ) -> str:
        """Persist the complete, non-secret recovery contract for one failed node."""

        safe_task_id = re.sub(r"[^A-Za-z0-9._-]+", "_", task.task_id).strip("._") or "unknown_task"
        failure_root = self.run_dir / "failure" / safe_task_id
        detected_at = _now()
        error_message = f"{type(exc).__name__}: {exc}"[:4000]
        error_payload = {
            "schema": "evomind.failure_error.v1",
            "run_id": run.run_id,
            "task_id": task.task_id,
            "agent": task.role,
            "failure_type": failure_type,
            "error_type": type(exc).__name__,
            "message": str(exc)[:4000],
            "attempt": task.attempts,
            "detected_at": detected_at,
        }
        environment_payload = {
            "schema": "evomind.failure_environment.v1",
            "run_id": run.run_id,
            "task_id": task.task_id,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_name": os.name,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "captured_at": detected_at,
            "environment_policy": "non_secret_runtime_facts_only",
        }
        node_payload = {
            "schema": "evomind.failure_node_status.v1",
            "run_id": run.run_id,
            "task_id": task.task_id,
            "role": task.role,
            "goal": task.goal,
            "status": task.status,
            "dependencies": list(task.dependencies),
            "attempts": task.attempts,
            "max_retries": task.max_retries,
            "error": error_message,
            "result_ref": task.result_ref or None,
            "captured_at": detected_at,
        }
        repair_strategy = {
            "connection": "Revalidate the bound connector/profile and retry the same idempotent node after readiness is restored.",
            "connection_auth": "Revalidate the named credential profile and allocation binding before retry.",
            "timeout": "Inspect the execution log and resource state, then retry with the same bounded contract or revise the timeout through Gate review.",
            "oom": "Reduce the resource request through a reviewed plan change; do not silently change the experiment contract.",
            "dependency": "Repair the declared runtime dependency and rerun the same node with unchanged evidence inputs.",
            "provider": "Restore the configured provider route, preserve the request, and retry through the same idempotency key.",
        }.get(failure_type, "Analyze the captured traceback, prepare a reviewed repair, and retry the same idempotent node.")
        recovery_payload = {
            "schema": "evomind.failure_recovery_plan.v1",
            "run_id": run.run_id,
            "task_id": task.task_id,
            "recovery_agent": "RecoveryAgent",
            "failure_type": failure_type,
            "repair_strategy": repair_strategy,
            "retry_scheduled": retry_scheduled,
            "retry_policy": "bounded_same_node" if retry_scheduled else "manual_repair_then_resume",
            "idempotency_key": task.idempotency_key,
            "actions": ["detect", "analyze", "repair", "retry"],
            "created_at": detected_at,
        }
        _atomic_json(failure_root / "error.json", error_payload)
        _atomic_text(failure_root / "traceback.txt", traceback.format_exc())
        _atomic_json(failure_root / "environment.json", environment_payload)
        _atomic_json(failure_root / "node_status.json", node_payload)
        _atomic_json(failure_root / "recovery_plan.json", recovery_payload)
        relative_root = str(failure_root.relative_to(self.run_dir)).replace("\\", "/")
        action_records = (
            ("detect", "completed", "Failure captured at the scheduler exception boundary."),
            ("analyze", "completed", f"Failure classified as {failure_type}."),
            ("repair", "plan_generated", repair_strategy),
            ("retry", "scheduled" if retry_scheduled else "manual_required", "Bounded retry decision recorded."),
        )
        for action, status, message in action_records:
            self._append("action_log.jsonl", {
                "schema": "evomind.recovery_action.v1",
                "run_id": run.run_id,
                "task_id": task.task_id,
                "agent": "RecoveryAgent",
                "action": action,
                "status": status,
                "message": message,
                "artifact_path": relative_root,
                "created_at": _now(),
            })
        return relative_root

    def consume_control(self) -> str:
        path = self.run_dir / "control.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        try:
            path.unlink()
        except OSError:
            pass
        action = str(payload.get("action") or "").lower()
        return action if action in {"pause", "resume", "cancel"} else ""


def validate_task_graph(tasks: Mapping[str, AgentTask], roles: Mapping[str, AgentRoleSpec]) -> None:
    missing_roles = sorted({task.role for task in tasks.values()} - set(roles))
    if missing_roles:
        raise ValueError(f"unknown roles: {', '.join(missing_roles)}")
    missing_dependencies = sorted({dep for task in tasks.values() for dep in task.dependencies if dep not in tasks})
    if missing_dependencies:
        raise ValueError(f"missing dependencies: {', '.join(missing_dependencies)}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"task graph contains a cycle at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in tasks[task_id].dependencies:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)


RoleExecutor = Callable[[AgentTask, HandoffEnvelope, SupervisorRun], AgentResult]


class MultiAgentSupervisor:
    """Bounded DAG scheduler with durable state and injected role executors."""

    def __init__(
        self,
        run: SupervisorRun,
        store: MultiAgentStore,
        executors: Mapping[str, RoleExecutor],
    ) -> None:
        self.run = run
        self.store = store
        self.executors = dict(executors)
        self._control_lock = threading.RLock()

    def initialize(self) -> None:
        validate_task_graph(self.run.tasks, self.run.roles)
        for task in self.run.tasks.values():
            if not task.idempotency_key:
                task.idempotency_key = build_idempotency_key(
                    task_id=task.task_id,
                    run_id=self.run.run_id,
                    solution_id=task.solution_id,
                    data_hash=str(task.payload.get("data_hash") or ""),
                    code_hash=str(task.payload.get("code_hash") or ""),
                    config_hash=str(task.payload.get("config_hash") or ""),
                )
        self.run.status = "ready"
        self.run.next_action = "dispatch_ready_tasks"
        self.store.save(self.run)
        self.store.emit(self.run, "run.created", status=self.run.status, task_count=len(self.run.tasks))

    def pause(self) -> None:
        with self._control_lock:
            if self.run.status not in {"completed", "cancelled"}:
                self.run.status = "paused"
                self.run.next_action = "resume"
                self.store.emit(self.run, "run.paused", status="paused")

    def resume(self, *, retry_failed: bool = False) -> None:
        with self._control_lock:
            if self.run.status in {"paused", "needs_continuation", "failed", "running"}:
                interrupted: set[str] = set()
                for task in self.run.tasks.values():
                    if task.status == "running":
                        task.error = "interrupted before durable completion"
                        self._transition(task, "failed")
                        interrupted.add(task.task_id)
                        self.store.emit(
                            self.run,
                            "task.interrupted",
                            task_id=task.task_id,
                            agent=task.role,
                            status="failed",
                            recovery="retry_on_resume",
                        )
                if retry_failed or interrupted:
                    for task in self.run.tasks.values():
                        if task.status == "failed" and (retry_failed or task.task_id in interrupted):
                            task.attempts = 0
                            task.error = ""
                            task.started_at = ""
                            task.finished_at = ""
                            self._transition(task, "ready")
                        elif task.status == "skipped":
                            task.error = ""
                            task.started_at = ""
                            task.finished_at = ""
                            self._transition(task, "pending")
                self.run.status = "ready"
                self.run.next_action = "dispatch_ready_tasks"
                self.store.emit(self.run, "run.resumed", status="ready")

    def cancel(self) -> None:
        with self._control_lock:
            self.run.status = "cancelling"
            for task in self.run.tasks.values():
                if task.status in {"pending", "ready", "retry_wait"}:
                    self._transition(task, "cancelled")
            self.run.status = "cancelled"
            self.run.next_action = "none"
            self.store.emit(self.run, "run.cancelled", status="cancelled")

    def _transition(self, task: AgentTask, target: str) -> None:
        if target not in TASK_TRANSITIONS.get(task.status, set()):
            raise ValueError(f"invalid task transition {task.task_id}: {task.status} -> {target}")
        previous = task.status
        task.status = target
        if target == "running":
            task.started_at = _now()
        if target in TERMINAL_TASK_STATES:
            task.finished_at = _now()
        self.store.emit(
            self.run,
            "task.state",
            task_id=task.task_id,
            agent=task.role,
            previous=previous,
            status=target,
            evidence_refs=[task.result_ref] if task.result_ref else [],
        )

    def _mark_ready(self) -> list[AgentTask]:
        ready: list[AgentTask] = []
        for task in self.run.tasks.values():
            if task.status == "ready":
                ready.append(task)
                continue
            if task.status not in {"pending", "retry_wait"}:
                continue
            dependency_states = [self.run.tasks[dep].status for dep in task.dependencies]
            if any(state in {"failed", "cancelled", "skipped"} for state in dependency_states):
                if task.status == "pending":
                    self._transition(task, "skipped")
                continue
            if all(state == "completed" for state in dependency_states):
                self._transition(task, "ready")
                ready.append(task)
        return sorted(ready, key=lambda item: (-item.priority, item.task_id))

    def _handoff(self, task: AgentTask) -> HandoffEnvelope:
        evidence = tuple(
            self.run.tasks[dependency].result_ref
            for dependency in task.dependencies
            if self.run.tasks[dependency].result_ref
        )
        handoff = HandoffEnvelope(
            handoff_id=f"handoff-{uuid.uuid4().hex[:12]}",
            run_id=self.run.run_id,
            task_id=task.task_id,
            sender="ExecutiveSupervisor",
            receiver=task.role,
            input_evidence_refs=evidence,
            expected_output=task.acceptance_criteria,
            budget={"timeout_seconds": task.timeout_seconds, "max_turns": self.run.child_max_turns},
            provenance={"dependency_task_ids": list(task.dependencies)},
        )
        self.store.append_handoff(handoff)
        self.store.emit(
            self.run,
            "handoff.created",
            task_id=task.task_id,
            agent=task.role,
            handoff_id=handoff.handoff_id,
            evidence_refs=list(evidence),
            status="created",
        )
        return handoff

    def _execute(self, task: AgentTask, handoff: HandoffEnvelope) -> AgentResult:
        executor = self.executors.get(task.role)
        if executor is None:
            raise RuntimeError(f"no executor registered for role {task.role}")
        result = executor(task, handoff, self.run)
        if not isinstance(result, AgentResult):
            raise TypeError(f"executor {task.role} returned {type(result).__name__}, expected AgentResult")
        if result.task_id != task.task_id:
            raise ValueError(f"executor result task mismatch: {result.task_id} != {task.task_id}")
        return result

    def apply_followups(self, source: AgentTask, result: AgentResult) -> None:
        if not result.followup_tasks and not result.dependency_overrides:
            return

        candidate_tasks = copy.deepcopy(self.run.tasks)
        added: list[str] = []
        for followup in result.followup_tasks:
            existing = candidate_tasks.get(followup.task_id)
            if existing is not None:
                if existing.to_dict() != followup.to_dict():
                    raise ValueError(f"conflicting follow-up task: {followup.task_id}")
                continue
            if followup.status != "pending":
                raise ValueError(f"follow-up task must start pending: {followup.task_id}")
            if not followup.idempotency_key:
                followup.idempotency_key = build_idempotency_key(
                    task_id=followup.task_id,
                    run_id=self.run.run_id,
                    solution_id=followup.solution_id,
                    data_hash=str(followup.payload.get("data_hash") or ""),
                    code_hash=str(followup.payload.get("code_hash") or ""),
                    config_hash=str(followup.payload.get("config_hash") or ""),
                )
            candidate_tasks[followup.task_id] = copy.deepcopy(followup)
            added.append(followup.task_id)

        overrides: dict[str, list[str]] = {}
        for target_id, dependencies in result.dependency_overrides.items():
            if target_id not in candidate_tasks:
                raise ValueError(f"follow-up dependency override targets unknown task: {target_id}")
            candidate_tasks[target_id].dependencies = tuple(dependencies)
            overrides[target_id] = list(dependencies)

        validate_task_graph(candidate_tasks, self.run.roles)
        for task_id in added:
            self.run.tasks[task_id] = candidate_tasks[task_id]
        for target_id in result.dependency_overrides:
            self.run.tasks[target_id].dependencies = candidate_tasks[target_id].dependencies
        self.store.emit(
            self.run,
            "task_graph.expanded",
            task_id=source.task_id,
            agent=source.role,
            status="planned",
            added_task_ids=added,
            dependency_overrides=overrides,
        )

    def run_until_blocked(self) -> SupervisorRun:
        if self.run.status == "created":
            self.initialize()
        if self.run.status == "paused":
            return self.run
        self.run.status = "running"
        self.run.next_action = "dispatch_ready_tasks"
        self.store.emit(self.run, "run.started", status="running")
        active: dict[Future[AgentResult], tuple[AgentTask, HandoffEnvelope, float]] = {}
        overdue: set[Future[AgentResult]] = set()
        pause_requested = False

        with ThreadPoolExecutor(max_workers=max(1, min(self.run.max_concurrency, 3))) as pool:
            while True:
                control = self.store.consume_control()
                if control == "pause":
                    pause_requested = True
                    self.run.next_action = "finish_active_then_pause"
                    self.store.emit(self.run, "run.pause_requested", status="pausing")
                elif control == "cancel":
                    for future, (task, _handoff, _started) in list(active.items()):
                        future.cancel()
                        if task.status == "running":
                            self._transition(task, "cancelled")
                    self.cancel()
                    break
                if self.run.status in {"paused", "cancelling", "cancelled"}:
                    break
                if pause_requested and not active:
                    self.pause()
                    return self.run
                ready = [] if pause_requested else self._mark_ready()
                dispatch_batch: list[tuple[AgentTask, HandoffEnvelope]] = []
                for task in ready:
                    if len(active) + len(dispatch_batch) >= min(self.run.max_concurrency, 3):
                        break
                    resource_active = sum(
                        1 for active_task, _active_handoff, _started in active.values()
                        if active_task.resource_type == task.resource_type
                    ) + sum(
                        1 for pending_task, _pending_handoff in dispatch_batch
                        if pending_task.resource_type == task.resource_type
                    )
                    resource_limit = self.run.resource_limits.get(task.resource_type, self.run.max_concurrency)
                    if resource_active >= max(1, resource_limit):
                        continue
                    completed_ref = self.run.idempotency_results.get(task.idempotency_key)
                    if completed_ref:
                        task.result_ref = completed_ref
                        task.error = ""
                        self._transition(task, "running")
                        self._transition(task, "completed")
                        self.store.emit(
                            self.run,
                            "task.idempotent_reuse",
                            task_id=task.task_id,
                            agent=task.role,
                            status="completed",
                            evidence_refs=[completed_ref],
                        )
                        continue
                    handoff = self._handoff(task)
                    task.attempts += 1
                    self._transition(task, "running")
                    dispatch_batch.append((task, handoff))

                # Prepare the complete fan-out batch before workers start. This keeps
                # ledger I/O for one task from serializing short sibling tasks.
                for task, handoff in dispatch_batch:
                    future = pool.submit(self._execute, task, handoff)
                    active[future] = (task, handoff, time.monotonic())

                if not active:
                    unfinished = [task for task in self.run.tasks.values() if task.status not in TERMINAL_TASK_STATES]
                    if not unfinished:
                        break
                    if any(task.status == "ready" for task in unfinished):
                        continue
                    self.run.status = "needs_continuation"
                    self.run.open_requirements = [f"blocked task: {task.task_id} ({task.status})" for task in unfinished]
                    self.run.next_action = "repair_or_resume"
                    self.store.emit(
                        self.run,
                        "run.blocked",
                        status=self.run.status,
                        open_requirements=self.run.open_requirements,
                    )
                    return self.run

                done, _ = wait(active, timeout=0.1, return_when=FIRST_COMPLETED)
                now = time.monotonic()
                for future, (task, _handoff, started) in list(active.items()):
                    if not future.done() and now - started > task.timeout_seconds:
                        if future.cancel():
                            done.add(future)
                            task.error = f"timeout after {task.timeout_seconds}s"
                        elif future not in overdue:
                            overdue.add(future)
                            self.store.emit(
                                self.run,
                                "task.deadline_exceeded",
                                task_id=task.task_id,
                                agent=task.role,
                                status="running",
                                elapsed_seconds=round(now - started, 3),
                                timeout_seconds=task.timeout_seconds,
                                action="await_cooperative_completion",
                            )

                for future in done:
                    overdue.discard(future)
                    task, _handoff, _started = active.pop(future)
                    try:
                        if task.error.startswith("timeout after"):
                            raise TimeoutError(task.error)
                        result = future.result()
                        if not result.accepted:
                            raise _TaskExecutionFailure(result.failure_type or "review_rejected")
                        self.apply_followups(task, result)
                        task.result_ref = self.store.write_result(result)
                        self.run.idempotency_results[task.idempotency_key] = task.result_ref
                        task.error = ""
                        self._transition(task, "completed")
                        self.store.emit(
                            self.run,
                            "task.result",
                            task_id=task.task_id,
                            agent=task.role,
                            status="completed",
                            confidence=result.confidence,
                            evidence_refs=result.evidence_refs + [task.result_ref],
                            artifact_hashes=[item.get("sha256") for item in result.artifacts if item.get("sha256")],
                        )
                    except Exception as exc:
                        task.error = f"{type(exc).__name__}: {exc}"
                        failure_type = _classify_task_failure(exc)
                        if task.attempts <= task.max_retries:
                            self._transition(task, "retry_wait")
                            failure_ref = self.store.write_failure_bundle(
                                self.run,
                                task,
                                exc,
                                failure_type=failure_type,
                                retry_scheduled=True,
                            )
                            self.store.emit(
                                self.run,
                                "task.retry",
                                task_id=task.task_id,
                                agent=task.role,
                                status="retry_wait",
                                attempt=task.attempts,
                                error=task.error,
                                evidence_refs=[failure_ref],
                            )
                        else:
                            self._transition(task, "failed")
                            failure_ref = self.store.write_failure_bundle(
                                self.run,
                                task,
                                exc,
                                failure_type=failure_type,
                                retry_scheduled=False,
                            )
                            self.store.emit(
                                self.run,
                                "task.failed",
                                task_id=task.task_id,
                                agent=task.role,
                                status="failed",
                                failure_type=failure_type,
                                error=task.error,
                                evidence_refs=[failure_ref],
                            )

        failed = [task.task_id for task in self.run.tasks.values() if task.status == "failed"]
        if self.run.status == "cancelled":
            return self.run
        if failed:
            self.run.status = "needs_continuation"
            self.run.open_requirements = [f"repair failed task: {task_id}" for task_id in failed]
            self.run.next_action = "repair_or_resume"
        else:
            self.run.status = "completed"
            self.run.open_requirements = []
            self.run.next_action = "review_final_report"
        self.store.emit(
            self.run,
            "run.finished",
            status=self.run.status,
            open_requirements=self.run.open_requirements,
        )
        return self.run


def create_run(
    *,
    objective: str,
    tasks: Iterable[AgentTask],
    roles: Iterable[AgentRoleSpec],
    run_id: str | None = None,
    max_concurrency: int = 3,
) -> SupervisorRun:
    task_list = list(tasks)
    role_list = list(roles)
    task_map = {task.task_id: task for task in task_list}
    role_map = {role.role: role for role in role_list}
    if len(task_map) != len(task_list):
        raise ValueError("duplicate task id")
    if len(role_map) != len(role_list):
        raise ValueError("duplicate role id")
    return SupervisorRun(
        run_id=run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        objective=objective,
        tasks=task_map,
        roles=role_map,
        max_concurrency=max(1, min(max_concurrency, 3)),
    )


__all__ = [
    "AgentResult",
    "AgentRoleSpec",
    "AgentTask",
    "HandoffEnvelope",
    "MultiAgentStore",
    "MultiAgentSupervisor",
    "SupervisorRun",
    "build_idempotency_key",
    "create_run",
    "validate_task_graph",
]
