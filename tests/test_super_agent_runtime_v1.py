from __future__ import annotations

from evomind_runtime.super_agent_runtime import SuperAgentRuntime
from evomind_runtime.models import ToolSpec


def test_runtime_mounts_workspace_and_builds_durable_shadow_graph(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime = SuperAgentRuntime(
        workspace_root=workspace,
        runtime_root=runtime_root,
        tenant_id="tenant_a",
        project_id="project_a",
    )
    status = runtime.status()
    assert status["mode"] == "shadow"
    assert status["migration_applied"] is True
    assert status["directory_count"] == 1
    assert "opaque_root_ref" not in status["directories"][0]
    discovered = runtime.discover("write a report in the authorized directory", required_operations=("write",))
    assert discovered[0]["descriptor"]["capability_id"] == "workspace"
    graph = runtime.shadow_plan(run_id="run_shadow_a", objective="write a report")
    assert graph.graph.id == "run_shadow_a"
    status = runtime.status()
    assert status["runs"][0]["nodes"][0]["id"] == "discover"
    assert status["runs"][0]["exact_gate"] is None
    runtime.close()

    reopened = SuperAgentRuntime(
        workspace_root=workspace,
        runtime_root=runtime_root,
        tenant_id="tenant_a",
        project_id="project_a",
    )
    restored = reopened.shadow_plan(run_id="run_shadow_a", objective="ignored because durable")
    assert restored.graph.objective == "write a report"
    assert reopened.status()["active_run_ids"] == ["run_shadow_a"]
    reopened.close()


def test_runtime_broker_roundtrip_is_capability_scoped(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = SuperAgentRuntime(workspace_root=workspace, runtime_root=tmp_path / "runtime")
    receipt = runtime.broker.verify_roundtrip("workspace", "evidence/probe.bin", b"super-agent-v1")
    assert receipt.ok is True
    assert receipt.metadata["write_read_hash_match"] is True
    assert (workspace / "evidence" / "probe.bin").read_bytes() == b"super-agent-v1"
    runtime.close()


def test_runtime_bounded_skill_refresh_adds_declarative_capability(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: bounded demo capability\n---\nignored body\n",
        encoding="utf-8",
    )
    runtime = SuperAgentRuntime(workspace_root=workspace, runtime_root=tmp_path / "runtime")
    report = runtime.refresh_installed_skills([tmp_path / "skills"])
    assert report["scanned_files"] == 1
    assert any("demo-skill" in item.capability_id for item in runtime.catalog.all())
    runtime.close()


def test_runtime_ingests_tool_specs_for_descriptor_driven_discovery(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = SuperAgentRuntime(workspace_root=workspace, runtime_root=tmp_path / "runtime")
    spec = ToolSpec(
        "weather_lookup",
        "Look up a weather forecast",
        {"type": "object", "properties": {"city": {"type": "string"}}},
        "weather.read",
    )
    assert runtime.ingest_runtime_tools([spec]) == 1
    matches = runtime.discover("weather forecast")
    assert any(item["descriptor"]["capability_id"] == "weather_lookup" for item in matches)
    runtime.close()


def test_verified_experience_is_project_scoped_and_anonymously_promotable(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = SuperAgentRuntime(
        workspace_root=workspace,
        runtime_root=tmp_path / "runtime",
        tenant_id="tenant_a",
        project_id="project_a",
    )
    rows = runtime.remember_verified_experience(
        claim="SFTP roundtrip succeeded after connector preflight",
        evidence_sha256="d" * 64,
        payload={"write_read_hash_match": True},
        category="connector-recovery",
        promote_anonymous_global=True,
    )
    assert rows[0]["tenant_id"] == "tenant_a"
    assert rows[1]["anonymous"] is True
    assert rows[1]["tenant_id"] == ""
    found = runtime.store.search_memory(
        tenant_id="tenant_a", project_id="project_a", limit=10
    )
    assert len(found) == 2
    runtime.close()
