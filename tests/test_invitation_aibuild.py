import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evomind_runtime.aibuild_engine import AIBuildEngine, use_aibuild
from evomind_runtime.model_evidence import verify_linear_bundle
from evomind_runtime.runtime import AgentRuntime


def test_new_training_runs_use_invited_orchestrator_only(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOMIND_AIBUILD_MODE", "enabled")
    monkeypatch.setenv("EVOMIND_AIBUILD_TENANTS", "tenant-a")
    assert use_aibuild("训练模型并评测", {"tenant_id": "tenant-a"})
    assert not use_aibuild("你好", {"tenant_id": "tenant-a"})
    assert not use_aibuild("训练模型并评测", {"tenant_id": "tenant-b"})
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective="训练模型并评测", metadata={"tenant_id": "tenant-a"})
        assert session["metadata"]["research_orchestrator"] == "aibuildai2"
    finally:
        runtime.close()


def test_manager_expands_real_scheduler_graph_with_isolated_roles(tmp_path):
    runtime = AgentRuntime(tmp_path)
    seen = []
    class Client:
        def send(self, messages, *, system, tools):
            seen.append(system)
            manager_calls = sum("research Manager" in text for text in seen)
            calls = []
            if "research Manager" in system:
                choice = {"role": "DesignerAgent" if manager_calls == 1 else "stop", "candidate": "solution_01", "reason": "Create a frozen design" if manager_calls == 1 else "Input protocol requires confirmation"}
                calls = [SimpleNamespace(id=f"manager-{manager_calls}", name="aibuild_dispatch", input=choice)]
            return SimpleNamespace(text="role complete", raw_content=[{"type": "text", "text": "role complete"}], tool_calls=calls, provider="fixture", model="fixture", input_tokens=10, output_tokens=10)
    try:
        session = runtime.create_session(objective="训练模型", workspace_root=str(tmp_path / "task"))
        result = AIBuildEngine(runtime, session, client_factory=Client).execute()
        graph = json.loads((runtime.runtime_root / "aibuild_runs" / session["id"] / "task_graph.json").read_text())
        assert not (tmp_path / "task/work/aibuild/run.json").exists()
        roles = [node["role"] for node in graph["nodes"]]
        assert "SetupAgent" in roles and "Manager" in roles and "DesignerAgent" in roles
        assert any("DesignerAgent" in system for system in seen)
        assert result["status"] == "blocked"
        assert not runtime.store.list_tool_calls(session["id"])
    finally:
        runtime.close()


def test_reviewer_reloads_checkpoint_and_recomputes_predictions(tmp_path):
    np.savez(tmp_path / "model.npz", coef=np.array([[2.0]]), intercept=np.array([1.0]))
    np.savez(tmp_path / "evaluation.npz", x=np.array([[1.0], [2.0]]), y=np.array([[3.0], [5.0]]), train_ids=np.array([1,2]), evaluation_ids=np.array([3,4]))
    np.savez(tmp_path / "predictions.npz", predictions=np.array([[3.0], [5.0]]))
    hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ("model.npz", "evaluation.npz", "predictions.npz")}
    (tmp_path / "evaluation-contract.json").write_text(json.dumps({"schema": "evomind.linear_evaluation.v1", "scope": "public_internal", "metric": "rmse", "sha256": hashes}))
    result = verify_linear_bundle(tmp_path)
    assert result["verified"] is True and result["independent"] is True
    assert result["value"] == 0.0
    assert result["paper_baseline_beat"] is False
    (tmp_path / "predictions.npz").write_bytes(b"tampered")
    assert verify_linear_bundle(tmp_path)["verified"] is False


def test_complete_role_chain_exports_only_projections_and_publishes_verification(tmp_path, monkeypatch):
    import evomind_runtime.aibuild_engine as module
    runtime = AgentRuntime(tmp_path)
    roles = iter(["DesignerAgent", "CoderAgent", "TunerAgent", "IndependentReviewer", "Aggregator"])
    seen = []
    class Client:
        def send(self, messages, *, system, tools):
            seen.append(system)
            calls = []
            if "research Manager" in system:
                calls = [SimpleNamespace(id="dispatch-"+str(len(seen)), name="aibuild_dispatch", input={"role": next(roles), "candidate": "solution_01", "reason": "Follow the fixture evidence"})]
            return SimpleNamespace(text="Fixture role completed", raw_content=[{"type": "text", "text": "Fixture role completed"}], tool_calls=calls, provider="fixture", model="fixture", input_tokens=10, output_tokens=10)
    monkeypatch.setattr(module, "verify_managed_tensor_bundle", lambda *_: {"verified": True, "independent": True, "metric": "rmse", "value": 0.1, "protocol_id": "a"*64, "scope": "engineering", "source_hashes": {"model.safetensors": "b"*64}})
    try:
        run = runtime.assistant.create_run(prompt="Orchestration fixture", start=False)
        session = runtime.get_session(run["id"])
        engine = AIBuildEngine(runtime, session, client_factory=Client)
        result = engine.execute()
        assert result["status"] == "completed"
        for role in ("SetupAgent", "DesignerAgent", "CoderAgent", "TunerAgent", "IndependentReviewer", "Aggregator"):
            assert any(role in system for system in seen)
        assert (engine.path / "run.json").is_file()
        assert not (Path(session["workspace_root"]) / "work/aibuild/run.json").exists()
        published = runtime.store.list_deliverables(run["id"])
        assert any("independent-verification" in row["name"] for row in published)
        assert any("task_graph" in row["name"] for row in published)
    finally:
        runtime.close()


def test_reconciled_setup_reuses_existing_manager_node(tmp_path):
    from research_os.agent.multi_agent import MultiAgentStore
    runtime = AgentRuntime(tmp_path)
    seen = []
    class Client:
        def send(self, messages, *, system, tools):
            seen.append(system)
            calls = [SimpleNamespace(id="stop", name="aibuild_dispatch", input={"role": "stop", "candidate": "solution_01", "reason": "Fixture ends at this boundary"})] if "research Manager" in system else []
            return SimpleNamespace(text="Setup complete", raw_content=[], tool_calls=calls, provider="fixture", model="fixture", input_tokens=1, output_tokens=1)
    try:
        session = runtime.create_session(objective="reconciliation fixture", workspace_root=str(tmp_path / "task"))
        engine = AIBuildEngine(runtime, session, client_factory=Client)
        assert engine.execute()["status"] == "blocked"
        store = MultiAgentStore(engine.path)
        graph = store.load()
        cached_ref = graph.idempotency_results.pop(graph.tasks["setup"].idempotency_key)
        assert cached_ref == graph.tasks["setup"].result_ref
        cached = Path(cached_ref)
        if not cached.is_absolute():
            cached = engine.path / cached
        assert cached.is_file()
        cached.rename(engine.path / "setup-before-reconciliation.json")
        graph.tasks["setup"].status = "ready"
        graph.tasks["setup"].result_ref = ""
        graph.tasks["setup"].finished_at = ""
        graph.tasks["manager_0"].status = "pending"
        graph.tasks["manager_0"].error = ""
        graph.status = "ready"
        store.save(graph)
        assert AIBuildEngine(runtime, session, client_factory=Client).execute()["status"] == "blocked"
        after = store.load()
        assert len([task for task in after.tasks.values() if task.role == "Manager"]) == 1
        assert after.tasks["setup"].attempts == 2
        assert sum("You are EvoMind's SetupAgent," in system for system in seen) == 2
        assert sum("research Manager" in system for system in seen) == 2
    finally:
        runtime.close()


def test_execution_capabilities_preserve_role_and_session_boundaries(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective="train a model using managed_tensor_train")
        engine = AIBuildEngine(runtime, session)
        _, contract = engine._execution_capabilities(session["objective"])
        roles = contract["tools_by_role"]
        assert "file_write" not in roles["SetupAgent"]
        assert "managed_tensor_train" not in roles["SetupAgent"]
        assert "file_write" in roles["CoderAgent"]
        assert "managed_tensor_train" in roles["TunerAgent"]
        assert "managed_tensor_train" in contract["exact_approval_tools"]
        assert contract["availability_is_permission"] is False
        assert contract["catalog_search_is_complete_inventory"] is False
        verifier = contract["coordinator_actions"]["IndependentReviewer"]
        assert verifier["entrypoint"] == "verify_managed_tensor_bundle"
        assert verifier["model_may_read_evaluation_labels"] is False
        session["metadata"] = {**session.get("metadata", {}), "run_allowed_tool_names": ["model_protocols"]}
        _, restricted = AIBuildEngine(runtime, session)._execution_capabilities(session["objective"])
        assert all(set(names) <= {"model_protocols"} for names in restricted["tools_by_role"].values())
        assert "managed_tensor_train" not in restricted["exact_approval_tools"]
    finally:
        runtime.close()


def test_manager_receives_core_role_contract_despite_empty_shadow_catalog(tmp_path):
    runtime = AgentRuntime(tmp_path)
    seen = []
    class Client:
        def send(self, messages, *, system, tools):
            payload = json.loads(messages[0]["content"])
            contract = payload["execution_capabilities"]
            seen.append(system)
            assert contract["catalog_search_is_complete_inventory"] is False
            calls = []
            if "research Manager" in system:
                assert "file_write" in contract["tools_by_role"]["CoderAgent"]
                assert "managed_tensor_train" in contract["tools_by_role"]["TunerAgent"]
                first = sum("research Manager" in item for item in seen) == 1
                calls = [SimpleNamespace(id=str(len(seen)), name="aibuild_dispatch", input={"role": "DesignerAgent" if first else "stop", "candidate": "solution_01", "reason": "Use registered role tools" if first else "End fixture before actual training"})]
            elif "SetupAgent" in system:
                assert "file_write" not in {item.name for item in tools}
                assert "file_write" in contract["tools_by_role"]["CoderAgent"]
            return SimpleNamespace(text="Optional shadow catalog has zero matches; use the role contract", raw_content=[], tool_calls=calls, provider="fixture", model="fixture", input_tokens=1, output_tokens=1)
    try:
        session = runtime.create_session(objective="train a model using managed_tensor_train")
        engine = AIBuildEngine(runtime, session, client_factory=Client)
        assert engine.execute()["status"] == "blocked"
        assert any("DesignerAgent" in system for system in seen)
        assert not runtime.store.list_tool_calls(session["id"])
        assert (engine.path / "capability-contract.json").is_file()
    finally:
        runtime.close()
