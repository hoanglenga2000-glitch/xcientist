from __future__ import annotations

from pathlib import Path

import pytest

from evomind_runtime.agent_kernel_v2 import (
    AgentKernel,
    AttemptStatus,
    FailureEnvelope,
    KernelState,
    OperationReceipt,
    RunSupervisor,
    TaskGraph,
    TaskNode,
)
from evomind_runtime.evidence_memory import (
    EvidenceMemory,
    EvidenceRef,
    EvidenceScope,
    EvidenceStatus,
    JsonEvidenceBackend,
    SensitiveEvidenceError,
)
from evomind_runtime.recovery import ExactGate, JsonRunStateRepository, RetrySuppressed


def _graph() -> TaskGraph:
    return TaskGraph(
        id="run-v2-test",
        objective="download, verify, and prepare data",
        tenant_id="tenant-a",
        project_id="project-a",
        nodes=[
            TaskNode("fetch", "fetch directly into authorized mount", acceptance_criteria=["hash verified"]),
            TaskNode("smoke", "run loader smoke", dependencies=["fetch"], acceptance_criteria=["exit zero"]),
        ],
    )


def _verified_receipt(graph_id: str, node_id: str, attempt_id: str) -> OperationReceipt:
    return OperationReceipt(
        graph_id=graph_id,
        node_id=node_id,
        attempt_id=attempt_id,
        operation="transfer_fetch",
        connector_id="fixture",
        scope={"directory_id": "test-data", "relative_path": "source"},
        bytes_processed=4,
        sha256="0" * 64,
        exit_status="completed",
        evidence_refs=["artifact:" + "1" * 64],
        verified=True,
    )


def test_task_graph_rejects_missing_dependencies_and_cycles() -> None:
    with pytest.raises(ValueError, match="missing dependencies"):
        TaskGraph(objective="bad", nodes=[TaskNode("a", "a", dependencies=["missing"])])

    with pytest.raises(ValueError, match="dependency cycle"):
        TaskGraph(
            objective="bad cycle",
            nodes=[TaskNode("a", "a", dependencies=["b"]), TaskNode("b", "b", dependencies=["a"])],
        )


def test_kernel_completes_only_after_verified_receipts() -> None:
    graph = _graph()
    kernel = AgentKernel(graph)
    kernel.start()

    fetch = kernel.begin_attempt(
        "fetch",
        idempotency_key="fetch-v1",
        connector_id="fixture",
        tool_version="1.0.0",
        preconditions={"connector_health_sha256": "a" * 64},
    )
    kernel.record_success(fetch.id, _verified_receipt(graph.id, "fetch", fetch.id))
    assert kernel.state is KernelState.VERIFYING
    kernel.verify_node("fetch")
    assert kernel.state is KernelState.RUNNING
    assert [item.id for item in graph.ready_nodes()] == ["smoke"]

    smoke = kernel.begin_attempt(
        "smoke",
        idempotency_key="smoke-v1",
        connector_id="fixture",
        tool_version="1.0.0",
        preconditions={"manifest_sha256": "b" * 64},
    )
    unverified = _verified_receipt(graph.id, "smoke", smoke.id)
    unverified.verified = False
    kernel.record_success(smoke.id, unverified)
    with pytest.raises(ValueError, match="independently verified"):
        kernel.verify_node("smoke")

    unverified.verified = True
    kernel.verify_node("smoke")
    assert kernel.state is KernelState.COMPLETED


def test_same_failure_and_preconditions_are_never_blindly_retried() -> None:
    graph = TaskGraph(objective="recover", nodes=[TaskNode("download", "download")])
    kernel = AgentKernel(graph)
    attempt = kernel.begin_attempt(
        "download",
        idempotency_key="download-1",
        connector_id="sftp",
        tool_version="1.0.0",
        preconditions={"adapter_version": "r1", "route": "same"},
    )
    failure = FailureEnvelope(
        graph_id=graph.id,
        node_id="download",
        attempt_id=attempt.id,
        stage="adapter_parse",
        error_class="ParserError",
        sanitized_error="adapter parser failed",
        retryable=True,
        secrets_touched=False,
        required_preconditions={"adapter_version": "r1", "route": "same"},
    )
    assert kernel.record_failure(attempt.id, failure) is None
    repair = kernel.propose_repair(
        failure.id,
        description="replace malformed parser expression",
        test_contract=["parser-only check"],
        rollback_contract=["restore preimage hash"],
    )
    kernel.begin_canary(repair.id, evidence_refs=["test:" + "2" * 64])
    kernel.activate_repair(repair.id, canary_passed=True)

    with pytest.raises(RetrySuppressed, match="unchanged preconditions"):
        kernel.begin_attempt(
            "download",
            idempotency_key="download-2",
            connector_id="sftp",
            tool_version="1.0.0",
            preconditions={"adapter_version": "r1", "route": "same"},
        )

    changed = kernel.begin_attempt(
        "download",
        idempotency_key="download-3",
        connector_id="sftp",
        tool_version="1.0.1",
        preconditions={"adapter_version": "r2", "route": "same"},
    )
    assert changed.status == AttemptStatus.RUNNING.value


def test_kernel_snapshot_contract_rejects_raw_secret_material() -> None:
    graph = TaskGraph(objective="safe snapshot", nodes=[TaskNode("fetch", "fetch")])
    kernel = AgentKernel(graph)
    with pytest.raises(ValueError, match="raw secret material"):
        kernel.begin_attempt(
            "fetch",
            idempotency_key="unsafe",
            connector_id="sftp",
            tool_version="1",
            preconditions={},
            input_evidence={"password": "plaintext"},
        )


def test_kernel_accepts_connector_receipt_protocol_without_shared_type_import() -> None:
    graph = TaskGraph(objective="connector protocol", nodes=[TaskNode("fetch", "fetch")])
    kernel = AgentKernel(graph)
    attempt = kernel.begin_attempt(
        "fetch",
        idempotency_key="connector-receipt",
        connector_id="local",
        tool_version="1",
        preconditions={"mount_health": "verified"},
    )
    normalized = kernel.record_success(
        attempt.id,
        {
            "receipt_id": "op_connector",
            "operation": "write",
            "connector_id": "local",
            "directory_id": "workspace",
            "relative_path": "probe.bin",
            "ok": True,
            "status": "completed",
            "exit_code": 0,
            "bytes_written": 4,
            "sha256": "3" * 64,
            "evidence_refs": ["write:" + "4" * 64],
        },
    )
    assert normalized.id == "op_connector"
    assert normalized.bytes_processed == 4
    assert normalized.verified is False
    kernel.verify_receipt(normalized.id, independent_evidence_refs=["readback:" + "5" * 64])
    kernel.verify_node("fetch")
    assert kernel.state is KernelState.COMPLETED
    assert normalized.to_store_dict()["receipt_id"] == normalized.id


def test_nonretryable_failure_yields_one_exact_gate_and_requires_changed_fact() -> None:
    graph = TaskGraph(objective="gate", nodes=[TaskNode("join", "external terms action")])
    kernel = AgentKernel(graph)
    attempt = kernel.begin_attempt(
        "join",
        idempotency_key="join-1",
        connector_id="browser",
        tool_version="1",
        preconditions={"terms_accepted": False},
    )
    failure = FailureEnvelope(
        graph_id=graph.id,
        node_id="join",
        attempt_id=attempt.id,
        stage="authorization",
        error_class="HumanGate",
        sanitized_error="the user must accept the exact terms",
        retryable=False,
        secrets_touched=False,
        required_preconditions={"terms_accepted": False},
    )
    gate = kernel.record_failure(attempt.id, failure)
    assert gate is not None
    assert kernel.state is KernelState.WAITING_EXACT_GATE
    assert gate.resume_point == "join"

    with pytest.raises(RetrySuppressed):
        kernel.resume_after_gate(changed_preconditions={"terms_accepted": False})
    kernel.resume_after_gate(changed_preconditions={"terms_accepted": True, "receipt": "verified"})
    assert kernel.state is KernelState.RUNNING

    with pytest.raises(ValueError, match="single action"):
        ExactGate("run", "join", "reason", "first\nsecond", "join")


def test_supervisor_restart_restores_and_reattaches_without_replay(tmp_path: Path) -> None:
    repository = JsonRunStateRepository(tmp_path / "runs")
    first = RunSupervisor(repository)
    kernel = first.register(AgentKernel(_graph()))
    attempt = kernel.begin_attempt(
        "fetch",
        idempotency_key="durable-fetch",
        connector_id="sftp",
        tool_version="1.2.3",
        preconditions={"route_receipt": "verified"},
    )

    restarted = RunSupervisor(JsonRunStateRepository(tmp_path / "runs"))
    result = restarted.resume(kernel.graph.id)

    assert result.kernel is not kernel
    assert result.reattach_attempt_ids == (attempt.id,)
    assert len(result.kernel.attempts) == 1
    assert result.ready_node_ids == ()
    assert result.exact_gate is None
    assert restarted.active_run_ids() == [kernel.graph.id]


def test_evidence_memory_scopes_status_and_secret_redaction(tmp_path: Path) -> None:
    backend = JsonEvidenceBackend(tmp_path / "evidence" / "memory.json")
    memory = EvidenceMemory(backend)
    source = EvidenceRef("a" * 64, opaque_ref="receipt-a")
    project = memory.remember(
        claim="adapter parser fix passed",
        status=EvidenceStatus.PROVEN.value,
        payload={"password": "must-never-persist", "test": "parser-only", "path": r"C:\\private\\run"},
        source_evidence=[source],
        tenant_id="tenant-a",
        project_id="project-a",
        category="recovery",
        failure_signature="failure-a",
    )
    assert project.payload["password"] == "[REDACTED]"
    assert project.payload["path"].startswith("[REDACTED_PATH:")
    raw = (tmp_path / "evidence" / "memory.json").read_text(encoding="utf-8")
    assert "must-never-persist" not in raw
    assert r"C:\\private\\run" not in raw
    assert memory.query(tenant_id="tenant-b", project_id="project-a") == []
    assert memory.query(tenant_id="tenant-a", project_id="project-a") == [project]

    global_record = memory.promote_anonymous_global(project.id)
    assert global_record.scope == EvidenceScope.ANONYMOUS_GLOBAL.value
    assert global_record.anonymous is True
    assert global_record.tenant_id == global_record.project_id == ""
    assert global_record.payload["path"] == "[ANONYMIZED_PATH]"

    reloaded = EvidenceMemory(JsonEvidenceBackend(tmp_path / "evidence" / "memory.json"))
    assert [item.id for item in reloaded.query(tenant_id="tenant-b", project_id="project-b")] == [global_record.id]

    with pytest.raises(ValueError, match="only proven"):
        memory.remember(
            claim="unverified global claim",
            status=EvidenceStatus.OBSERVED.value,
            payload={},
            source_evidence=[source],
            scope=EvidenceScope.ANONYMOUS_GLOBAL.value,
        )
    with pytest.raises(SensitiveEvidenceError):
        memory.remember(
            claim="contains raw key",
            status=EvidenceStatus.OBSERVED.value,
            payload={"body": "-----BEGIN PRIVATE KEY-----"},
            source_evidence=[],
            tenant_id="tenant-a",
            project_id="project-a",
        )
