from __future__ import annotations

from evomind_runtime.super_agent_store import MIGRATION_ID, SuperAgentStore


def test_additive_store_persists_v2_entities_and_migration(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    store = SuperAgentStore(path)
    assert store.migration_applied() is True
    store.put_capability({"capability_id": "cap_local", "provider_id": "local", "version": "1.0.0"})
    store.put_connector({"connector_id": "local_a", "connector_type": "local"})
    store.put_directory({
        "directory_id": "workspace",
        "connector_id": "local_a",
        "tenant_id": "tenant_a",
        "project_id": "project_a",
    })
    assert store.list_directories(tenant_id="tenant_a", project_id="project_a")[0]["directory_id"] == "workspace"
    store.put_task_graph("run_a", "planned", {"nodes": []})
    assert store.get_task_graph("run_a")["status"] == "planned"
    store.close()
    reopened = SuperAgentStore(path)
    assert reopened.migration_applied() is True
    assert MIGRATION_ID == "super_agent_v1_001"
    reopened.close()


def test_failure_loop_and_memory_lookup_are_scoped(tmp_path):
    store = SuperAgentStore(tmp_path / "runtime.sqlite3")
    failure = {
        "failure_id": "failure_a",
        "run_id": "run_a",
        "node_id": "node_a",
        "signature": "same_failure",
        "category": "runtime",
    }
    store.append_failure(failure)
    assert store.failure_count(run_id="run_a", node_id="node_a", signature="same_failure") == 1
    store.put_memory({
        "record_id": "memory_project",
        "tenant_id": "tenant_a",
        "project_id": "project_a",
        "memory_scope": "tenant_project",
        "evidence_status": "proven",
        "failure_signature": "same_failure",
        "evidence_sha256": "a" * 64,
    })
    store.put_memory({
        "record_id": "memory_global",
        "tenant_id": "anonymous",
        "project_id": "anonymous",
        "memory_scope": "anonymous_global",
        "evidence_status": "proven",
        "failure_signature": "same_failure",
        "evidence_sha256": "b" * 64,
    })
    found = store.search_memory(
        tenant_id="tenant_a",
        project_id="project_a",
        failure_signature="same_failure",
    )
    assert {item["record_id"] for item in found} == {"memory_project", "memory_global"}
    other = store.search_memory(
        tenant_id="tenant_b",
        project_id="project_b",
        failure_signature="same_failure",
    )
    assert {item["record_id"] for item in other} == {"memory_global"}
    store.close()
