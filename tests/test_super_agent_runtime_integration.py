from __future__ import annotations

from evomind_runtime import AgentRuntime
from evomind_runtime.capabilities import CapabilityDescriptor, DirectoryCapability, OperationReceipt
from evomind_runtime.remote_connectors import HpcExecutionConnector
from evomind_runtime.policy import argument_fingerprint


def test_agent_runtime_registers_capability_scoped_directory_tools(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="write a verified file")
    tool_names = {item["name"] for item in runtime.tools()}
    assert {
        "super_agent_status",
        "capability_discover",
        "connector_health",
        "directory_read",
        "directory_write_atomic",
        "directory_delete",
    } <= tool_names

    discovered = runtime.invoke_tool(
        session["id"],
        "capability_discover",
        {"objective": "write a file", "required_operations": ["write"]},
    )
    assert discovered["result"]["ok"] is True
    written = runtime.invoke_tool(
        session["id"],
        "directory_write_atomic",
        {"directory_id": "workspace", "relative_path": "outputs/v2.txt", "content": "v2"},
    )
    assert written["result"]["ok"] is True
    read = runtime.invoke_tool(
        session["id"],
        "directory_read",
        {"directory_id": "workspace", "relative_path": "outputs/v2.txt"},
    )
    assert read["result"]["content"]["text"] == "v2"
    runtime.close()


def test_directory_delete_is_always_exact_approval(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="delete only after approval", permission_level="full-auto")
    runtime.invoke_tool(
        session["id"],
        "directory_write_atomic",
        {"directory_id": "workspace", "relative_path": "temporary.txt", "content": "temporary"},
    )
    waiting = runtime.invoke_tool(
        session["id"],
        "directory_delete",
        {"directory_id": "workspace", "relative_path": "temporary.txt"},
        idempotency_key="delete-temporary",
    )
    assert waiting["status"] == "waiting_approval"
    approved = runtime.decide_approval(waiting["approval"]["id"], True, "approved exact path")
    assert approved["result"]["ok"] is True
    assert not (tmp_path / "temporary.txt").exists()
    runtime.close()


def test_forged_approval_fingerprint_cannot_bypass_durable_receipt(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="fail closed", permission_level="full-auto")
    runtime.invoke_tool(
        session["id"],
        "directory_write_atomic",
        {"directory_id": "workspace", "relative_path": "forged.txt", "content": "keep"},
    )
    arguments = {"directory_id": "workspace", "relative_path": "forged.txt"}
    forged = runtime.invoke_tool(
        session["id"],
        "directory_delete",
        arguments,
        tool_call_id="call_forged_approval",
        approved_fingerprint=argument_fingerprint("directory_delete", arguments),
    )
    assert forged["status"] == "waiting_approval"
    assert (tmp_path / "forged.txt").exists()
    terminal = runtime.invoke_tool(
        session["id"],
        "tool_reject",
        {"package_id": "candidate"},
    )
    assert terminal["status"] == "waiting_approval"
    runtime.close()


def test_sync_deletion_requires_durable_approval(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="sync exactly", permission_level="full-auto")
    for path, value in (("source/keep.txt", "keep"), ("destination/remove.txt", "remove")):
        runtime.invoke_tool(
            session["id"],
            "directory_write_atomic",
            {"directory_id": "workspace", "relative_path": path, "content": value},
        )
    waiting = runtime.invoke_tool(
        session["id"],
        "directory_sync",
        {
            "source_directory_id": "workspace",
            "source_relative_path": "source",
            "destination_directory_id": "workspace",
            "destination_relative_path": "destination",
            "delete_extraneous": True,
        },
    )
    assert waiting["status"] == "waiting_approval"
    assert (tmp_path / "destination" / "remove.txt").exists()
    approved = runtime.decide_approval(waiting["approval"]["id"], True, "approve exact sync deletion")
    assert approved["result"]["ok"] is True
    assert not (tmp_path / "destination" / "remove.txt").exists()
    assert (tmp_path / "destination" / "keep.txt").read_text() == "keep"
    runtime.close()


def test_catalog_expansion_can_add_specialized_tool_after_discovery(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="prepare data in my authorized directory")
    initial = {item.name for item in runtime._message_tool_specs("prepare data", session)}
    assert "capability_discover" in initial
    expanded = {
        item.name
        for item in runtime._message_tool_specs(
            "prepare data",
            session,
            {"competition_data_prepare"},
        )
    }
    assert "competition_data_prepare" in expanded
    runtime.close()


class _SourceConnector:
    connector_id = "fixture-http"
    version = "1.0.0"

    def descriptor(self):
        return CapabilityDescriptor(
            self.connector_id,
            "http",
            self.version,
            ("fetch",),
            {"type": "object"},
        )

    def fetch(self, _url, destination, relative_path, **_kwargs):
        return OperationReceipt(
            "fetch",
            self.connector_id,
            destination.directory_id,
            relative_path,
            bytes_written=7,
            sha256="a" * 64,
        )


def _identity():
    names = (
        "job_container_verified",
        "designated_proxy_path_verified",
        "pinned_gateway_host_key_verified",
        "allocation_role_authenticated",
        "expected_host_uuid_match",
        "expected_gpu_uuid_match",
        "expected_gpu_model_and_memory_match",
        "allowed_remote_root_match",
        "identity_consistent",
        "job_profile_match",
    )
    return {
        "ok": True,
        **{name: True for name in names},
        "samples_requested": 5,
        "samples_passed": 5,
        "samples": [{"complete": True} for _ in range(5)],
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def test_transfer_and_execution_connectors_are_injected_and_policy_gated(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="fetch and execute", permission_level="full-auto")
    runtime.super_agent.register_source_connector(_SourceConnector())
    fetched = runtime.invoke_tool(
        session["id"],
        "transfer_fetch",
        {
            "source_connector_id": "fixture-http",
            "source_url": "https://example.invalid/data.bin",
            "directory_id": "workspace",
            "relative_path": "data/data.bin",
        },
    )
    assert fetched["result"]["ok"] is True

    executor = HpcExecutionConnector(
        lambda _capability: _identity(),
        lambda **_kwargs: {"ok": True, "exit_code": 0, "sha256": "b" * 64, "artifacts_count": 1},
        connector_id="fixture-execution",
    )
    capability = DirectoryCapability(
        "fixture-execution-mount",
        "fixture-execution",
        "/bounded/root",
        ("execute",),
        tenant_id=runtime.super_agent.tenant_id,
        project_id=runtime.super_agent.project_id,
    )
    runtime.super_agent.register_execution_connector(executor, capability)
    executed = runtime.invoke_tool(
        session["id"],
        "job_execute",
        {"directory_id": capability.directory_id, "relative_path": "jobs/a", "job_spec": {"kind": "smoke"}},
    )
    assert executed["result"]["ok"] is True
    waiting = runtime.invoke_tool(
        session["id"],
        "job_cancel",
        {"directory_id": capability.directory_id, "job_ref": "job-a"},
    )
    assert waiting["status"] == "waiting_approval"
    runtime.close()


def test_unconfigured_transfer_and_job_interfaces_return_exact_gate(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="exact gate", permission_level="full-auto")
    transfer = runtime.invoke_tool(
        session["id"],
        "transfer_fetch",
        {
            "source_connector_id": "missing-source",
            "source_url": "https://example.invalid/data",
            "directory_id": "workspace",
            "relative_path": "data.bin",
        },
    )
    assert transfer["result"]["error"] == "source_connector_not_configured"
    assert transfer["result"]["content"]["exact_gate"]["resume_point"] == "transfer_fetch"
    job = runtime.invoke_tool(
        session["id"],
        "job_execute",
        {"directory_id": "workspace", "job_spec": {"kind": "smoke"}},
    )
    assert job["result"]["error"] == "execution_connector_not_configured"
    assert job["result"]["content"]["exact_gate"]["resume_point"] == "job_execute"
    runtime.close()


def test_default_public_https_source_is_discoverable_and_rejects_unlisted_hosts(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="public HTTPS source", permission_level="full-auto")
    discovered = runtime.super_agent.discover(
        "public source transfer fetch",
        required_operations=("transfer_fetch",),
        run_id=session["id"],
        workspace_root=runtime.workspace_root,
        metadata=session["metadata"],
    )
    assert any(item["descriptor"]["capability_id"] == "public-https-source" for item in discovered)
    directory_id = session["metadata"]["super_agent_directory_ids"][0]
    transfer = runtime.invoke_tool(
        session["id"],
        "transfer_fetch",
        {
            "source_connector_id": "public-https-source",
            "source_url": "https://example.invalid/data",
            "directory_id": directory_id,
            "relative_path": "data.bin",
        },
    )
    assert transfer["result"]["ok"] is False
    assert "not allowlisted" in transfer["result"]["error"]
    assert transfer["result"]["error"] != "source_connector_not_configured"
    runtime.close()


def test_run_scoped_tool_package_requires_published_test_and_canary_evidence(tmp_path):
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(objective="stage a reversible run tool")
    manifest = {
        "schema": "evomind.tool_package.v1",
        "package_id": "auditor",
        "run_id": session["id"],
        "version": "1.0.0",
        "provider_id": "run-sandbox",
        "capabilities": [
            {
                "name": "audit_mount",
                "operations": ["audit_mount"],
                "input_schema": {"type": "object"},
                "risk_class": "observe",
                "idempotency": "idempotent",
            }
        ],
        "permissions": {"directories": ["declared-only"]},
        "tests": [{"id": "contract", "expected": "pass"}],
        "canary": {"scope": "isolated"},
        "rollback": {"action": "discard-run-package", "reversible": True},
    }
    staged = runtime.invoke_tool(session["id"], "tool_synthesize", {"manifest": manifest})
    assert staged["result"]["ok"] is True
    runtime.invoke_tool(
        session["id"],
        "directory_write_atomic",
        {"directory_id": "workspace", "relative_path": "outputs/evidence.json", "content": "{}"},
    )
    published = runtime.invoke_tool(
        session["id"],
        "artifact_publish",
        {"path": "outputs/evidence.json", "name": "evidence.json"},
    )
    artifact = published["result"]["content"]["artifact"]
    for tool_name in ("tool_test", "tool_canary"):
        recorded = runtime.invoke_tool(
            session["id"],
            tool_name,
            {
                "package_id": "auditor",
                "passed": True,
                "evidence_artifact_id": artifact["id"],
                "evidence_sha256": artifact["sha256"],
            },
        )
        assert recorded["result"]["ok"] is True
    promoted = runtime.invoke_tool(session["id"], "tool_promote", {"package_id": "auditor"})
    assert promoted["result"]["ok"] is True
    assert promoted["result"]["content"]["package"]["promotion_status"] == "promoted"
    runtime.close()


def test_session_workspace_capabilities_are_run_and_tenant_scoped(tmp_path):
    runtime = AgentRuntime(tmp_path / "service", runtime_root=tmp_path / "runtime")
    first = runtime.create_session(
        objective="first",
        workspace_root=str(tmp_path / "run-first"),
        metadata={"tenant_id": "tenant_first", "project_id": "project_shared"},
    )
    second = runtime.create_session(
        objective="second",
        workspace_root=str(tmp_path / "run-second"),
        metadata={"tenant_id": "tenant_second", "project_id": "project_shared"},
    )
    first_directory = first["metadata"]["super_agent_directory_ids"][0]
    second_directory = second["metadata"]["super_agent_directory_ids"][0]
    assert first_directory != second_directory
    written = runtime.invoke_tool(
        first["id"],
        "directory_write_atomic",
        {"directory_id": first_directory, "relative_path": "private.txt", "content": "first"},
    )
    assert written["result"]["ok"] is True
    cross_run = runtime.invoke_tool(
        second["id"],
        "directory_read",
        {"directory_id": first_directory, "relative_path": "private.txt"},
    )
    assert cross_run["result"]["ok"] is False
    assert "CapabilityPermissionError" in cross_run["result"]["error"]
    global_workspace = runtime.invoke_tool(
        first["id"],
        "directory_list",
        {"directory_id": "workspace", "relative_path": ""},
    )
    assert global_workspace["result"]["ok"] is False
    own = runtime.invoke_tool(
        second["id"],
        "directory_list",
        {"directory_id": second_directory, "relative_path": ""},
    )
    assert own["result"]["ok"] is True
    runtime.close()


def test_same_physical_workspace_gets_session_capability_for_tenant_identity(tmp_path):
    service_root = tmp_path / "service"
    runtime = AgentRuntime(service_root, runtime_root=tmp_path / "runtime")
    session = runtime.create_session(
        objective="read the authenticated tenant workspace",
        workspace_root=str(service_root),
        metadata={"tenant_id": "tenant_current", "project_id": "project_current"},
    )
    directory_id = session["metadata"]["super_agent_directory_ids"][0]
    assert directory_id != "workspace"

    capability = runtime.super_agent.broker.capability(directory_id)
    assert capability.tenant_id == "tenant_current"
    assert capability.project_id == "project_current"
    assert capability.run_id == session["id"]

    own = runtime.invoke_tool(
        session["id"],
        "directory_list",
        {"directory_id": directory_id, "relative_path": ""},
    )
    assert own["result"]["ok"] is True

    global_workspace = runtime.invoke_tool(
        session["id"],
        "directory_list",
        {"directory_id": "workspace", "relative_path": ""},
    )
    assert global_workspace["result"]["ok"] is False
    assert "CapabilityPermissionError" in global_workspace["result"]["error"]
    runtime.close()


def test_persisted_same_root_tenant_capability_is_remounted_after_restart(tmp_path):
    service_root = tmp_path / "service"
    runtime_root = tmp_path / "runtime"
    runtime = AgentRuntime(service_root, runtime_root=runtime_root)
    session = runtime.create_session(
        objective="persist tenant capability",
        workspace_root=str(service_root),
        metadata={
            "managed_hpc_identity": {
                "tenant_id": "tenant_current",
                "project_id": "project_current",
            }
        },
    )
    directory_id = session["metadata"]["super_agent_directory_ids"][0]
    runtime.close()

    reopened = AgentRuntime(service_root, runtime_root=runtime_root)
    restored = reopened.get_session(session["id"])
    assert restored["metadata"]["super_agent_directory_ids"] == [directory_id]
    capability = reopened.super_agent.broker.capability(directory_id)
    assert capability.tenant_id == "tenant_current"
    assert capability.project_id == "project_current"
    assert capability.run_id == session["id"]

    own = reopened.invoke_tool(
        session["id"],
        "directory_list",
        {"directory_id": directory_id, "relative_path": ""},
    )
    assert own["result"]["ok"] is True
    discovered = reopened.super_agent.discover(
        "read the authenticated tenant workspace",
        required_operations=("read",),
        run_id=session["id"],
        workspace_root=restored["workspace_root"],
        metadata=restored["metadata"],
    )
    assert directory_id in {
        item["descriptor"]["capability_id"] for item in discovered
    }
    global_workspace = reopened.invoke_tool(
        session["id"],
        "directory_list",
        {"directory_id": "workspace", "relative_path": ""},
    )
    assert global_workspace["result"]["ok"] is False
    assert "CapabilityPermissionError" in global_workspace["result"]["error"]
    reopened.close()
