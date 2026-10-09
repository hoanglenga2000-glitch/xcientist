from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .recovery import (
    ExactGate,
    FailureLedger,
    RecoveryPolicy,
    RetrySuppressed,
    RunStateRepository,
    canonical_sha256,
    failure_signature,
    precondition_fingerprint,
    utc_now,
)


class KernelState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    CANARY_VERIFYING = "canary_verifying"
    WAITING_EXACT_GATE = "waiting_exact_gate"
    COMPLETED = "completed"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    CANARY_VERIFYING = "canary_verifying"
    WAITING_EXACT_GATE = "waiting_exact_gate"
    COMPLETED = "completed"


class AttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RepairStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    CANARY_VERIFYING = "canary_verifying"
    ACTIVATED = "activated"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


_TRANSITIONS: dict[KernelState, frozenset[KernelState]] = {
    KernelState.PLANNED: frozenset({KernelState.RUNNING, KernelState.WAITING_EXACT_GATE}),
    KernelState.RUNNING: frozenset(
        {KernelState.VERIFYING, KernelState.REPAIRING, KernelState.WAITING_EXACT_GATE}
    ),
    KernelState.VERIFYING: frozenset(
        {KernelState.RUNNING, KernelState.REPAIRING, KernelState.WAITING_EXACT_GATE, KernelState.COMPLETED}
    ),
    KernelState.REPAIRING: frozenset({KernelState.CANARY_VERIFYING, KernelState.WAITING_EXACT_GATE}),
    KernelState.CANARY_VERIFYING: frozenset(
        {KernelState.RUNNING, KernelState.REPAIRING, KernelState.WAITING_EXACT_GATE}
    ),
    KernelState.WAITING_EXACT_GATE: frozenset({KernelState.RUNNING}),
    KernelState.COMPLETED: frozenset(),
}


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


_SECRET_KEY = re.compile(
    r"^(?:password|passwd|pwd|access_token|refresh_token|id_token|api_key|authorization|cookie|secret|private_key|密码|口令|凭据|密钥)$",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"(?:\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?:password|passwd|pwd|token|secret|api[_ -]?key|authorization|bearer|密码|口令|凭据|密钥)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


def _assert_sanitized(value: Any, *, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY.fullmatch(str(key)) and item is not None and item not in ("", "[REDACTED]"):
                raise ValueError(f"{field_name} contains raw secret material")
            _assert_sanitized(item, field_name=field_name)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _assert_sanitized(item, field_name=field_name)
        return
    if isinstance(value, str) and _SECRET_TEXT.search(value):
        raise ValueError(f"{field_name} contains raw secret material")


@dataclass
class TaskNode:
    id: str
    action: str
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    capability_ids: list[str] = field(default_factory=list)
    connector_id: str = ""
    status: str = NodeStatus.PENDING.value
    budget: dict[str, Any] = field(default_factory=dict)
    receipt_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.action.strip():
            raise ValueError("task node requires id and action")
        NodeStatus(self.status)
        self.dependencies = list(dict.fromkeys(self.dependencies))
        self.acceptance_criteria = [str(item) for item in self.acceptance_criteria]
        self.capability_ids = list(dict.fromkeys(self.capability_ids))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskNode":
        return cls(**dict(payload))


@dataclass
class TaskGraph:
    objective: str
    nodes: list[TaskNode]
    id: str = field(default_factory=lambda: _id("graph"))
    tenant_id: str = ""
    project_id: str = ""
    status: str = KernelState.PLANNED.value
    budget: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("task graph requires an objective")
        if not self.nodes:
            raise ValueError("task graph requires at least one node")
        KernelState(self.status)
        normalized: list[TaskNode] = []
        for item in self.nodes:
            normalized.append(item if isinstance(item, TaskNode) else TaskNode.from_dict(item))
        self.nodes = normalized
        self.validate()

    def node(self, node_id: str) -> TaskNode:
        item = next((candidate for candidate in self.nodes if candidate.id == node_id), None)
        if item is None:
            raise KeyError(node_id)
        return item

    def validate(self) -> None:
        ids = [item.id for item in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("task graph node ids must be unique")
        known = set(ids)
        for item in self.nodes:
            if item.id in item.dependencies:
                raise ValueError(f"task node {item.id} cannot depend on itself")
            missing = set(item.dependencies) - known
            if missing:
                raise ValueError(f"task node {item.id} has missing dependencies: {sorted(missing)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("task graph contains a dependency cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in self.node(node_id).dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in ids:
            visit(node_id)

    def ready_nodes(self) -> list[TaskNode]:
        completed = {item.id for item in self.nodes if item.status == NodeStatus.COMPLETED.value}
        return [
            item
            for item in self.nodes
            if item.status == NodeStatus.PENDING.value and set(item.dependencies) <= completed
        ]

    @property
    def complete(self) -> bool:
        return all(item.status == NodeStatus.COMPLETED.value for item in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["nodes"] = [item.to_dict() for item in self.nodes]
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskGraph":
        value = dict(payload)
        value["nodes"] = [TaskNode.from_dict(item) for item in value.get("nodes", [])]
        return cls(**value)


@dataclass
class ExecutionAttempt:
    graph_id: str
    node_id: str
    idempotency_key: str
    connector_id: str
    tool_version: str
    precondition_fingerprint: str
    input_evidence: dict[str, Any] = field(default_factory=dict)
    output_evidence: dict[str, Any] = field(default_factory=dict)
    status: str = AttemptStatus.RUNNING.value
    failure_signature: str = ""
    id: str = field(default_factory=lambda: _id("attempt"))
    started_at: str = field(default_factory=utc_now)
    completed_at: str = ""
    request_fingerprint: str = ""

    def __post_init__(self) -> None:
        AttemptStatus(self.status)
        _assert_sanitized(self.input_evidence, field_name="attempt input evidence")
        _assert_sanitized(self.output_evidence, field_name="attempt output evidence")
        expected = canonical_sha256(
            {
                "graph_id": self.graph_id,
                "node_id": self.node_id,
                "connector_id": self.connector_id,
                "tool_version": self.tool_version,
                "input_evidence": self.input_evidence,
            }
        )
        if self.request_fingerprint and self.request_fingerprint != expected:
            raise ValueError("attempt request fingerprint mismatch")
        self.request_fingerprint = expected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionAttempt":
        return cls(**dict(payload))


@dataclass
class FailureEnvelope:
    graph_id: str
    node_id: str
    attempt_id: str
    stage: str
    error_class: str
    sanitized_error: str
    retryable: bool
    secrets_touched: bool
    required_preconditions: dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    precondition_fingerprint: str = ""
    id: str = field(default_factory=lambda: _id("failure"))
    observed_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.stage.strip() or not self.error_class.strip():
            raise ValueError("failure envelope requires stage and error_class")
        _assert_sanitized(self.sanitized_error, field_name="sanitized error")
        _assert_sanitized(self.required_preconditions, field_name="failure preconditions")
        expected_signature = failure_signature(
            stage=self.stage,
            error_class=self.error_class,
            sanitized_error=self.sanitized_error,
        )
        expected_preconditions = precondition_fingerprint(self.required_preconditions)
        if self.signature and self.signature != expected_signature:
            raise ValueError("failure signature does not match the sanitized failure")
        if self.precondition_fingerprint and self.precondition_fingerprint != expected_preconditions:
            raise ValueError("failure precondition fingerprint mismatch")
        self.signature = expected_signature
        self.precondition_fingerprint = expected_preconditions
        self.sanitized_error = " ".join(self.sanitized_error.strip().split())[:512]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureEnvelope":
        return cls(**dict(payload))


@dataclass
class RepairAction:
    graph_id: str
    node_id: str
    failure_id: str
    description: str
    test_contract: list[str]
    rollback_contract: list[str]
    status: str = RepairStatus.PROPOSED.value
    reversible: bool = True
    evidence_refs: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: _id("repair"))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        RepairStatus(self.status)
        if not self.description.strip() or not self.test_contract or not self.rollback_contract:
            raise ValueError("repair requires a description, tests, and rollback contract")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairAction":
        return cls(**dict(payload))


@dataclass
class OperationReceipt:
    graph_id: str
    node_id: str
    attempt_id: str
    operation: str
    connector_id: str
    scope: dict[str, Any]
    bytes_processed: int
    sha256: str
    exit_status: str
    evidence_refs: list[str]
    verified: bool = False
    id: str = field(default_factory=lambda: _id("receipt"))
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.bytes_processed < 0:
            raise ValueError("receipt byte count cannot be negative")
        _assert_sanitized(self.scope, field_name="operation receipt scope")
        _assert_sanitized(self.evidence_refs, field_name="operation receipt evidence")
        if self.sha256 and (len(self.sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in self.sha256)):
            raise ValueError("receipt sha256 must be a hexadecimal digest")
        if self.verified and (not self.exit_status or not self.evidence_refs):
            raise ValueError("verified receipt requires exit status and evidence references")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_store_dict(self) -> dict[str, Any]:
        """Return the additive SuperAgentStore wire shape without importing it."""

        return {
            **self.to_dict(),
            "receipt_id": self.id,
            "run_id": self.graph_id,
            "ok": self.verified and self.exit_status in {"completed", "succeeded", "ok", "0"},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationReceipt":
        return cls(**dict(payload))


class AgentKernel:
    """Evidence-first state machine for one durable task graph.

    The kernel records intent and outcomes only. Connectors execute outside this
    class, which keeps shadow-mode integration free from legacy tool behavior.
    """

    def __init__(
        self,
        graph: TaskGraph,
        *,
        repository: RunStateRepository | None = None,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.repository = repository
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self.attempts: list[ExecutionAttempt] = []
        self.failures: list[FailureEnvelope] = []
        self.repairs: list[RepairAction] = []
        self.receipts: list[OperationReceipt] = []
        self.failure_ledger = FailureLedger()
        self.exact_gate: ExactGate | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> KernelState:
        return KernelState(self.graph.status)

    def _transition(self, target: KernelState) -> None:
        current = self.state
        if target == current:
            return
        if target not in _TRANSITIONS[current]:
            raise ValueError(f"invalid kernel transition: {current.value} -> {target.value}")
        self.graph.status = target.value
        self.graph.updated_at = utc_now()

    def checkpoint(self) -> None:
        if self.repository is not None:
            self.repository.save(self.graph.id, self.snapshot())

    def start(self) -> None:
        with self._lock:
            self._transition(KernelState.RUNNING)
            self.checkpoint()

    def begin_attempt(
        self,
        node_id: str,
        *,
        idempotency_key: str,
        connector_id: str,
        tool_version: str,
        preconditions: Mapping[str, Any] | None = None,
        input_evidence: Mapping[str, Any] | None = None,
    ) -> ExecutionAttempt:
        if not idempotency_key.strip():
            raise ValueError("execution attempt requires an idempotency key")
        with self._lock:
            if self.state is KernelState.COMPLETED:
                raise ValueError("completed task graph cannot execute another attempt")
            if self.state is KernelState.WAITING_EXACT_GATE:
                raise ValueError("exact human gate must be resolved before execution")
            node = self.graph.node(node_id)
            completed = {item.id for item in self.graph.nodes if item.status == NodeStatus.COMPLETED.value}
            if not set(node.dependencies) <= completed:
                raise ValueError("task node dependencies are not complete")

            fingerprint = precondition_fingerprint(preconditions)
            existing = next((item for item in self.attempts if item.idempotency_key == idempotency_key), None)
            proposed_request = canonical_sha256(
                {
                    "graph_id": self.graph.id,
                    "node_id": node_id,
                    "connector_id": connector_id,
                    "tool_version": tool_version,
                    "input_evidence": dict(input_evidence or {}),
                }
            )
            if existing is not None:
                if existing.request_fingerprint != proposed_request:
                    raise ValueError("idempotency key is bound to a different execution request")
                if existing.status in {AttemptStatus.RUNNING.value, AttemptStatus.SUCCEEDED.value}:
                    return existing

            latest_failure = next((item for item in reversed(self.failures) if item.node_id == node_id), None)
            if latest_failure is not None:
                self.failure_ledger.assert_retry_allowed(
                    node_id=node_id,
                    signature=latest_failure.signature,
                    preconditions=fingerprint,
                )

            if self.state is KernelState.PLANNED:
                self._transition(KernelState.RUNNING)
            elif self.state in {KernelState.REPAIRING, KernelState.CANARY_VERIFYING}:
                self._transition(KernelState.RUNNING)
            elif self.state is KernelState.VERIFYING:
                self._transition(KernelState.RUNNING)

            attempt = ExecutionAttempt(
                graph_id=self.graph.id,
                node_id=node_id,
                idempotency_key=idempotency_key,
                connector_id=connector_id,
                tool_version=tool_version,
                precondition_fingerprint=fingerprint,
                input_evidence=dict(input_evidence or {}),
            )
            self.attempts.append(attempt)
            node.status = NodeStatus.RUNNING.value
            node.updated_at = utc_now()
            self.checkpoint()
            return attempt

    def record_failure(self, attempt_id: str, failure: FailureEnvelope) -> ExactGate | None:
        with self._lock:
            attempt = self._attempt(attempt_id)
            if attempt.status != AttemptStatus.RUNNING.value:
                raise ValueError("only a running attempt may fail")
            if failure.attempt_id != attempt_id or failure.node_id != attempt.node_id:
                raise ValueError("failure envelope does not match the attempt")
            if failure.precondition_fingerprint != attempt.precondition_fingerprint:
                raise ValueError("failure preconditions do not match the attempt")

            repeated = self.failure_ledger.has_seen(
                node_id=failure.node_id,
                signature=failure.signature,
                preconditions=failure.precondition_fingerprint,
            )
            attempt.status = AttemptStatus.FAILED.value
            attempt.failure_signature = failure.signature
            attempt.completed_at = utc_now()
            self.failures.append(failure)
            self.failure_ledger.record(
                node_id=failure.node_id,
                signature=failure.signature,
                preconditions=failure.precondition_fingerprint,
                attempt_id=attempt_id,
            )
            if repeated or not failure.retryable:
                gate = self.request_exact_gate(
                    node_id=failure.node_id,
                    reason=failure.sanitized_error or failure.error_class,
                    required_action="change the failed precondition and provide its verification evidence",
                    resume_point=failure.node_id,
                    failure=failure,
                )
                return gate

            node = self.graph.node(failure.node_id)
            node.status = NodeStatus.REPAIRING.value
            node.updated_at = utc_now()
            if self.state is KernelState.VERIFYING:
                self._transition(KernelState.REPAIRING)
            elif self.state is KernelState.RUNNING:
                self._transition(KernelState.REPAIRING)
            self.checkpoint()
            return None

    def propose_repair(
        self,
        failure_id: str,
        *,
        description: str,
        test_contract: Sequence[str],
        rollback_contract: Sequence[str],
        reversible: bool = True,
    ) -> RepairAction | ExactGate:
        with self._lock:
            failure = next((item for item in self.failures if item.id == failure_id), None)
            if failure is None:
                raise KeyError(failure_id)
            if not reversible:
                return self._gate_for_irreversible_repair(failure)
            repair = RepairAction(
                graph_id=self.graph.id,
                node_id=failure.node_id,
                failure_id=failure.id,
                description=description,
                test_contract=list(test_contract),
                rollback_contract=list(rollback_contract),
                reversible=True,
                status=RepairStatus.TESTING.value,
            )
            self.repairs.append(repair)
            self.checkpoint()
            return repair

    def _gate_for_irreversible_repair(self, failure: FailureEnvelope) -> ExactGate:
        return self.request_exact_gate(
            node_id=failure.node_id,
            reason="the proposed repair is irreversible",
            required_action="approve the exact irreversible repair scope",
            resume_point=failure.node_id,
            failure=failure,
        )

    def begin_canary(self, repair_id: str, *, evidence_refs: Sequence[str]) -> RepairAction:
        with self._lock:
            repair = self._repair(repair_id)
            if repair.status not in {RepairStatus.PROPOSED.value, RepairStatus.TESTING.value}:
                raise ValueError("repair is not ready for canary verification")
            if not evidence_refs:
                raise ValueError("canary verification requires test evidence")
            repair.status = RepairStatus.CANARY_VERIFYING.value
            repair.evidence_refs = list(evidence_refs)
            repair.updated_at = utc_now()
            self.graph.node(repair.node_id).status = NodeStatus.CANARY_VERIFYING.value
            if self.state is KernelState.REPAIRING:
                self._transition(KernelState.CANARY_VERIFYING)
            self.checkpoint()
            return repair

    def activate_repair(self, repair_id: str, *, canary_passed: bool) -> RepairAction | ExactGate:
        with self._lock:
            repair = self._repair(repair_id)
            if repair.status != RepairStatus.CANARY_VERIFYING.value:
                raise ValueError("repair has not entered canary verification")
            if not canary_passed:
                repair.status = RepairStatus.REJECTED.value
                repair.updated_at = utc_now()
                failure = next(item for item in self.failures if item.id == repair.failure_id)
                return self.request_exact_gate(
                    node_id=repair.node_id,
                    reason="repair canary failed",
                    required_action="provide one changed precondition or an alternative reversible repair",
                    resume_point=repair.node_id,
                    failure=failure,
                )
            repair.status = RepairStatus.ACTIVATED.value
            repair.updated_at = utc_now()
            node = self.graph.node(repair.node_id)
            node.status = NodeStatus.PENDING.value
            node.updated_at = utc_now()
            self._transition(KernelState.RUNNING)
            self.checkpoint()
            return repair

    def record_success(self, attempt_id: str, receipt: OperationReceipt | Mapping[str, Any] | Any) -> OperationReceipt:
        with self._lock:
            attempt = self._attempt(attempt_id)
            if attempt.status != AttemptStatus.RUNNING.value:
                raise ValueError("only a running attempt may succeed")
            receipt = self._coerce_receipt(attempt, receipt)
            if (
                receipt.graph_id != self.graph.id
                or receipt.attempt_id != attempt_id
                or receipt.node_id != attempt.node_id
            ):
                raise ValueError("operation receipt does not match the attempt")
            attempt.status = AttemptStatus.SUCCEEDED.value
            attempt.output_evidence = {"receipt_id": receipt.id, "receipt_sha256": canonical_sha256(receipt.to_dict())}
            attempt.completed_at = utc_now()
            self.receipts.append(receipt)
            node = self.graph.node(attempt.node_id)
            node.status = NodeStatus.VERIFYING.value
            node.receipt_ids = list(dict.fromkeys([*node.receipt_ids, receipt.id]))
            node.updated_at = utc_now()
            if self.state is KernelState.RUNNING:
                self._transition(KernelState.VERIFYING)
            self.checkpoint()
            return receipt

    def verify_receipt(self, receipt_id: str, *, independent_evidence_refs: Sequence[str]) -> OperationReceipt:
        with self._lock:
            receipt = next((item for item in self.receipts if item.id == receipt_id), None)
            if receipt is None:
                raise KeyError(receipt_id)
            if not independent_evidence_refs:
                raise ValueError("receipt verification requires independent evidence")
            if receipt.exit_status not in {"completed", "succeeded", "ok", "0"}:
                raise ValueError("failed operation receipt cannot be verified")
            receipt.evidence_refs = list(dict.fromkeys([*receipt.evidence_refs, *independent_evidence_refs]))
            receipt.verified = True
            self.checkpoint()
            return receipt

    def verify_node(self, node_id: str, *, receipt_ids: Sequence[str] | None = None) -> TaskNode:
        with self._lock:
            node = self.graph.node(node_id)
            selected = list(receipt_ids or node.receipt_ids)
            if not selected:
                raise ValueError("node completion requires at least one operation receipt")
            receipts = [item for item in self.receipts if item.id in selected and item.node_id == node_id]
            if len(receipts) != len(set(selected)) or not all(item.verified for item in receipts):
                raise ValueError("node completion requires all selected receipts to be independently verified")
            node.status = NodeStatus.COMPLETED.value
            node.receipt_ids = selected
            node.updated_at = utc_now()
            if self.graph.complete:
                self._transition(KernelState.COMPLETED)
            elif self.state is KernelState.VERIFYING:
                self._transition(KernelState.RUNNING)
            self.checkpoint()
            return node

    def request_exact_gate(
        self,
        *,
        node_id: str,
        reason: str,
        required_action: str,
        resume_point: str,
        failure: FailureEnvelope | None = None,
    ) -> ExactGate:
        with self._lock:
            node = self.graph.node(node_id)
            gate = ExactGate(
                run_id=self.graph.id,
                node_id=node_id,
                reason=reason,
                required_action=required_action,
                resume_point=resume_point,
                failure_signature=failure.signature if failure else "",
                precondition_fingerprint=failure.precondition_fingerprint if failure else "",
            )
            self.exact_gate = gate
            node.status = NodeStatus.WAITING_EXACT_GATE.value
            node.updated_at = utc_now()
            if self.state is not KernelState.WAITING_EXACT_GATE:
                self._transition(KernelState.WAITING_EXACT_GATE)
            self.checkpoint()
            return gate

    def resume_after_gate(self, *, changed_preconditions: Mapping[str, Any]) -> None:
        with self._lock:
            if self.state is not KernelState.WAITING_EXACT_GATE or self.exact_gate is None:
                raise ValueError("run is not waiting on an exact gate")
            changed = precondition_fingerprint(changed_preconditions)
            if self.exact_gate.precondition_fingerprint and changed == self.exact_gate.precondition_fingerprint:
                raise RetrySuppressed(self.exact_gate.failure_signature, changed)
            node = self.graph.node(self.exact_gate.node_id)
            node.status = NodeStatus.PENDING.value
            node.updated_at = utc_now()
            self.exact_gate = None
            self._transition(KernelState.RUNNING)
            self.checkpoint()

    def _attempt(self, attempt_id: str) -> ExecutionAttempt:
        item = next((candidate for candidate in self.attempts if candidate.id == attempt_id), None)
        if item is None:
            raise KeyError(attempt_id)
        return item

    def _coerce_receipt(self, attempt: ExecutionAttempt, value: Any) -> OperationReceipt:
        if isinstance(value, OperationReceipt):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
        elif callable(getattr(value, "to_dict", None)):
            payload = dict(value.to_dict())
        else:
            raise TypeError("operation receipt must be mapping-compatible")
        if payload.get("ok") is False or payload.get("exit_code") not in {None, 0}:
            raise ValueError("failed connector receipt cannot complete an attempt")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        scope = {
            key: payload[key]
            for key in (
                "directory_id",
                "relative_path",
                "source_directory_id",
                "source_relative_path",
                "destination_directory_id",
                "destination_relative_path",
            )
            if payload.get(key) is not None and payload.get(key) != ""
        }
        return OperationReceipt(
            id=str(payload.get("receipt_id") or payload.get("id") or _id("receipt")),
            graph_id=str(payload.get("graph_id") or payload.get("run_id") or self.graph.id),
            node_id=str(payload.get("node_id") or attempt.node_id),
            attempt_id=str(payload.get("attempt_id") or attempt.id),
            operation=str(payload.get("operation") or "connector_operation"),
            connector_id=str(payload.get("connector_id") or attempt.connector_id),
            scope=scope,
            bytes_processed=int(payload.get("bytes_processed") or 0)
            or int(payload.get("bytes_read") or 0) + int(payload.get("bytes_written") or 0),
            sha256=str(payload.get("sha256") or ""),
            exit_status=str(payload.get("exit_status") or payload.get("status") or "completed"),
            evidence_refs=[str(item) for item in payload.get("evidence_refs") or []],
            verified=bool(payload.get("verified") or metadata.get("independently_verified")),
            created_at=str(payload.get("created_at") or payload.get("completed_at") or utc_now()),
        )

    def _repair(self, repair_id: str) -> RepairAction:
        item = next((candidate for candidate in self.repairs if candidate.id == repair_id), None)
        if item is None:
            raise KeyError(repair_id)
        return item

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "evomind.agent_kernel.v2",
            "graph": self.graph.to_dict(),
            "attempts": [item.to_dict() for item in self.attempts],
            "failures": [item.to_dict() for item in self.failures],
            "repairs": [item.to_dict() for item in self.repairs],
            "receipts": [item.to_dict() for item in self.receipts],
            "failure_ledger": self.failure_ledger.to_list(),
            "exact_gate": self.exact_gate.to_dict() if self.exact_gate else None,
            "checkpointed_at": utc_now(),
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        repository: RunStateRepository | None = None,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> "AgentKernel":
        if snapshot.get("schema") != "evomind.agent_kernel.v2":
            raise ValueError("unsupported agent kernel snapshot schema")
        kernel = cls(
            TaskGraph.from_dict(snapshot["graph"]),
            repository=repository,
            recovery_policy=recovery_policy,
        )
        kernel.attempts = [ExecutionAttempt.from_dict(item) for item in snapshot.get("attempts", [])]
        kernel.failures = [FailureEnvelope.from_dict(item) for item in snapshot.get("failures", [])]
        kernel.repairs = [RepairAction.from_dict(item) for item in snapshot.get("repairs", [])]
        kernel.receipts = [OperationReceipt.from_dict(item) for item in snapshot.get("receipts", [])]
        kernel.failure_ledger = FailureLedger.from_list(snapshot.get("failure_ledger", []))
        if snapshot.get("exact_gate"):
            kernel.exact_gate = ExactGate.from_dict(snapshot["exact_gate"])
        return kernel


@dataclass(frozen=True)
class ResumeResult:
    kernel: AgentKernel
    reattach_attempt_ids: tuple[str, ...]
    ready_node_ids: tuple[str, ...]
    exact_gate: ExactGate | None


class RunSupervisor:
    """Own durable kernels independently from any browser or chat connection."""

    def __init__(self, repository: RunStateRepository) -> None:
        self.repository = repository
        self._kernels: dict[str, AgentKernel] = {}
        self._lock = threading.RLock()

    def register(self, kernel: AgentKernel) -> AgentKernel:
        with self._lock:
            if kernel.graph.id in self._kernels:
                raise ValueError("run is already registered")
            kernel.repository = self.repository
            self._kernels[kernel.graph.id] = kernel
            kernel.checkpoint()
            return kernel

    def get(self, run_id: str) -> AgentKernel:
        with self._lock:
            current = self._kernels.get(run_id)
            if current is not None:
                return current
            snapshot = self.repository.load(run_id)
            if snapshot is None:
                raise KeyError(run_id)
            kernel = AgentKernel.from_snapshot(snapshot, repository=self.repository)
            self._kernels[run_id] = kernel
            return kernel

    def checkpoint(self, run_id: str) -> None:
        self.get(run_id).checkpoint()

    def resume(
        self,
        run_id: str,
        *,
        changed_preconditions: Mapping[str, Any] | None = None,
    ) -> ResumeResult:
        """Restore without replaying an in-flight side effect.

        Running attempts are returned as reattach targets. A connector watcher
        must reconcile them by idempotency key before any new execution.
        """

        kernel = self.get(run_id)
        if kernel.state is KernelState.PLANNED:
            kernel.start()
        elif kernel.state is KernelState.WAITING_EXACT_GATE and changed_preconditions is not None:
            kernel.resume_after_gate(changed_preconditions=changed_preconditions)
        kernel.checkpoint()
        return ResumeResult(
            kernel=kernel,
            reattach_attempt_ids=tuple(
                item.id for item in kernel.attempts if item.status == AttemptStatus.RUNNING.value
            ),
            ready_node_ids=tuple(item.id for item in kernel.graph.ready_nodes()),
            exact_gate=kernel.exact_gate,
        )

    def active_run_ids(self) -> list[str]:
        result: list[str] = []
        for run_id in self.repository.list_run_ids():
            snapshot = self.repository.load(run_id)
            if snapshot and snapshot.get("graph", {}).get("status") != KernelState.COMPLETED.value:
                result.append(run_id)
        return sorted(result)


# Compatibility names for callers that prefer explicit V2 nomenclature.
AgentKernelV2 = AgentKernel
TaskGraphNode = TaskNode
