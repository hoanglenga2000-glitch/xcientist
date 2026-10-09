from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from evomind_runtime.aibuild_engine import AIBuildEngine, use_aibuild
from evomind_runtime.execution_progress import queued_resource
from evomind_runtime.model_evidence import verify_managed_tensor_bundle
from evomind_runtime.research_budget import ENGINEERING_LIMIT_SECONDS, GpuExecutionBudget, read_execution_budget
from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.tensor_inference import predict, read_weights
from evomind_runtime.training_control import digest, list_protocols, load_protocol, training_config


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    control = tmp_path / "control"
    registry = control / "protocols"
    registry.mkdir(parents=True)
    metadata = {"tenant_id": "tenant_" + "a" * 24, "owner_principal_id": "alice", "project_id": "project_" + "a" * 32}
    train_x = np.array([[0, 1], [1, 2], [2, 3], [3, 1]], dtype=np.float32)
    eval_x = np.array([[4, 1], [5, 2]], dtype=np.float32)
    np.savez(registry / "train.npz", x=train_x, y=2*train_x[:, 0]-train_x[:, 1], ids=np.arange(4))
    np.savez(registry / "eval.npz", x=eval_x, y=2*eval_x[:, 0]-eval_x[:, 1], ids=np.arange(4, 6))
    value = {"schema": "evomind.tensor_protocol.v1", "tenant_id": metadata["tenant_id"], "owner_id": "alice", "project_id": metadata["project_id"], "scope": "engineering", "task": "regression", "metric": "rmse", "training": {"path": "train.npz", "sha256": digest(registry / "train.npz")}, "evaluation": {"path": "eval.npz", "sha256": digest(registry / "eval.npz")}, "source": {"uri": "urn:fixture:synthetic", "version": "1", "license": "CC0-1.0"}, "acceptance": {"max_value": 0.1}}
    temporary = registry / "temporary.json"
    json_file(temporary, value)
    protocol_id = digest(temporary)
    temporary.rename(registry / (protocol_id + ".json"))
    policy = {"schema": "evomind.training_control.v1", "runtime_root": str(root), "enabled": True, "tenants": [metadata["tenant_id"]], "owners": ["alice"], "projects": [metadata["project_id"]], "study_id": "test-study"}
    json_file(control / "policy.json", policy)
    monkeypatch.setenv("EVOMIND_AIBUILD_POLICY_FILE", str(control / "policy.json"))
    monkeypatch.delenv("EVOMIND_AIBUILD_MODE", raising=False)
    return root, metadata, protocol_id, registry, value


def write_safetensors(path, values):
    offset, header, parts = 0, {}, []
    for name, value in sorted(values.items()):
        array = np.asarray(value, dtype="<f4")
        data = array.tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset+len(data)]}
        parts.append(data); offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(parts))


def model_fixture(tmp_path, protocol):
    runtime_root, metadata, protocol_id, _, _ = protocol
    workspace = tmp_path / "task"
    output = workspace / "outputs/hpc/solution_01"
    output.mkdir(parents=True)
    config = {"schema": "evomind.tensor_training.v1", "protocol_id": protocol_id, "task": "regression", "input_features": 2, "outputs": 1, "hidden_sizes": []}
    json_file(output / "model-config.json", config)
    write_safetensors(output / "model.safetensors", {"normalizer_mean": [0, 0], "normalizer_scale": [1, 1], "layers.0.weight": [[2, -1]], "layers.0.bias": [0]})
    np.savez(output / "predictions.npz", predictions=np.array([[7], [8]], dtype=np.float32))
    for name in ("training-receipt.json", "environment.lock.json", "data-protocol.json"):
        json_file(output / name, {"fixture": True})
    (output / "predict.py").write_text("# fixture inference entry\n")
    (output / "training-log.jsonl").write_text('{"steps":1}\n')
    files = {path.name: digest(path) for path in output.iterdir()}
    evidence = {"schema": "evomind.managed_training_evidence.v1", "candidate": "solution_01", "protocol_id": protocol_id, "relative_root": "outputs/hpc/solution_01", "files": files, "fit_steps": 1, "adapter_sha256": "a" * 64}
    calls = [{"tool_name": "managed_tensor_train", "status": "completed", "result": {"ok": True, "content": {"managed_training": evidence}}}]
    runtime = SimpleNamespace(runtime_root=runtime_root, store=SimpleNamespace(list_tool_calls=lambda _: calls))
    session = {"id": "fixture-run", "workspace_root": str(workspace), "metadata": metadata}
    return output, runtime, session, calls


def test_protocols_are_immutable_private_and_do_not_expose_labels(protocol):
    root, metadata, protocol_id, registry, _ = protocol
    result = list_protocols(root, metadata)
    assert result[0]["protocol_id"] == protocol_id and result[0]["scope"] == "engineering"
    assert "evaluation" not in result[0] and "evaluation_path" not in result[0]
    with pytest.raises(ValueError):
        load_protocol(root, {**metadata, "owner_principal_id": "bob"}, protocol_id)
    assert not use_aibuild("hello", metadata, root)
    assert use_aibuild("train a model", metadata, root)
    assert not use_aibuild("train a model", {**metadata, "project_id": "uninvited"}, root)
    (registry / "eval.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_protocol(root, metadata, protocol_id)


def test_safe_model_reload_uses_trusted_labels_and_execution_journal(tmp_path, protocol):
    output, runtime, session, calls = model_fixture(tmp_path, protocol)
    verified = verify_managed_tensor_bundle(runtime, session, "solution_01")
    assert verified["verified"] is True and verified["value"] == 0
    assert verified["official_score"] is False and verified["paper_baseline_beat"] is False
    calls.clear()
    json_file(output / "independent-verification.json", {"passed": True})
    assert verify_managed_tensor_bundle(runtime, session, "solution_01")["verified"] is False


def test_modified_checkpoint_or_prediction_is_rejected(tmp_path, protocol):
    output, runtime, session, _ = model_fixture(tmp_path, protocol)
    (output / "model.safetensors").write_bytes(b"tampered")
    result = verify_managed_tensor_bundle(runtime, session, "solution_01")
    assert result["verified"] is False and result["reason"] == "model_execution_artifact_hash_mismatch"


def test_checkpoint_header_and_nonfinite_values_fail_closed(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 2**62) + b"{}")
    with pytest.raises(ValueError, match="header"):
        read_weights(path)
    write_safetensors(path, {"x": [float("nan")]})
    with pytest.raises(ValueError, match="nonfinite"):
        read_weights(path)


def test_training_configuration_is_bounded_and_no_arbitrary_code(protocol):
    root, metadata, protocol_id, _, _ = protocol
    loaded = load_protocol(root, metadata, protocol_id)
    assert training_config({}, loaded)["epochs"] == 15
    for config in ({"code": "anything"}, {"epochs": 100000}, {"learning_rate": float("nan")}, {"hidden_sizes": [100000]}):
        with pytest.raises(ValueError):
            training_config(config, loaded)


def test_gpu_budget_charges_failures_and_retains_uncertainty(tmp_path):
    budget = GpuExecutionBudget(str(tmp_path / "gpu.sqlite"), "study")
    operation = budget.reserve("run1", 60)
    budget.settle(operation["id"], 10, success=False)
    assert budget.summary()["failed_operations"] == 1
    other = budget.reserve("run2", 60)
    budget.settle(other["id"], 5, uncertain=True)
    assert budget.summary()["reconciliation_required"] is True
    with pytest.raises(RuntimeError, match="reconciliation"):
        budget.reserve("run3", 60)


def test_gpu_budget_enforces_engineering_cap_and_detects_old_controller(tmp_path):
    path = tmp_path / "gpu.sqlite"
    budget = GpuExecutionBudget(str(path), "study")
    for index in range(ENGINEERING_LIMIT_SECONDS // 1200):
        budget.reserve(str(index), 1200)
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.reserve("too-many", 60)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE gpu_operations SET boot_id='old-controller'")
    with pytest.raises(RuntimeError, match="reconciliation"):
        budget.reserve("restart", 60)
    assert budget.summary()['reconciliation_required'] is True


def test_budget_diagnostic_is_read_only_and_explains_engineering_cap(tmp_path):
    path = tmp_path / 'budget.sqlite3'
    assert read_execution_budget(path, 'study')['operations'] == 0
    assert not path.exists()
    budget = GpuExecutionBudget(str(path), 'study')
    for index in range(ENGINEERING_LIMIT_SECONDS // 1200):
        op = budget.reserve(str(index), 1200)
        budget.settle(op['id'], 1200, success=True)
    before = path.read_bytes()
    summary = read_execution_budget(path, 'study')
    assert path.read_bytes() == before
    assert summary['engineering_remaining_seconds'] == 0
    assert summary['gpu_remaining_seconds'] == 4 * 3600
    assert summary['rebind_will_restore_budget'] is False
    assert summary['siim_calibration_budget_is_separate'] is True
    assert read_execution_budget(path, 'another-study')['operations'] == 0


def test_exhausted_execution_returns_actionable_budget_without_dispatch(protocol, monkeypatch):
    from evomind_runtime import tools
    root, metadata, _, _, _ = protocol
    context = tools.ToolContext('blocked_run', root, root, root, root, None, metadata)
    monkeypatch.setattr(tools, '_validate_hpc_solution_inputs', lambda *_: (root / 'script.py', root, ''))
    monkeypatch.setattr(tools, '_hpc_execute_solution_impl', lambda *_: pytest.fail('must not dispatch'))
    ledger = GpuExecutionBudget(str(root / 'gpu_budget.sqlite3'), 'test-study')
    for index in range(ENGINEERING_LIMIT_SECONDS // 1200):
        op = ledger.reserve(str(index), 1200)
        ledger.settle(op['id'], 1200, success=True)
    result = tools._hpc_execute_solution({'timeout_seconds': 60}, context)
    assert result.error == 'gpu_execution_budget_exhausted'
    assert result.content['remote_dispatched'] is False
    assert result.content['gpu_budget']['engineering_remaining_seconds'] == 0
    assert result.content['next_action'] == 'administrator_budget_review'
    status = tools._hpc_execution_budget_status({}, context)
    assert status.ok and status.content['operations'] == ENGINEERING_LIMIT_SECONDS // 1200
    context.metadata = {**metadata, 'owner_principal_id': 'other'}
    denied = tools._hpc_execution_budget_status({}, context)
    assert not denied.ok and not denied.content


def test_uncertain_remote_execution_does_not_release_the_queue(tmp_path):
    with pytest.raises(RuntimeError, match="remote_settlement_unconfirmed"):
        with queued_resource(tmp_path, "gpu", "one", wait_seconds=1):
            raise RuntimeError("remote_settlement_unconfirmed")
    with sqlite3.connect(tmp_path / "queue.sqlite3") as connection:
        assert connection.execute("SELECT state FROM resource_queue").fetchone()[0] == "uncertain"
    with pytest.raises(TimeoutError):
        with queued_resource(tmp_path, "gpu", "two", wait_seconds=0):
            pytest.fail("uncertain worker cannot be replaced")


def test_aibuild_empty_role_response_retries_without_tool_execution(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective="bounded role", workspace_root=str(tmp_path / "task"))
        engine = AIBuildEngine(runtime, session)
        responses = ["", "meaningful result"]
        class Client:
            def send(self, *args, **kwargs):
                text = responses.pop(0)
                return SimpleNamespace(text=text, tool_calls=[], raw_content=[], provider="fixture", model="fixture", input_tokens=1, output_tokens=1)
        assert engine._send(Client(), [], "role", []).text == "meaningful result"
        assert engine.calls == 2 and not runtime.store.list_tool_calls(session["id"])
    finally:
        runtime.close()


def test_controlled_hpc_entrypoint_reserves_and_charges_budget(protocol, monkeypatch):
    from evomind_runtime import tools
    from evomind_runtime.models import ToolResult
    root, metadata, _, _, _ = protocol
    updates, observed = [], []
    context = tools.ToolContext("budget_run", root, root, root, root, None, metadata, progress=lambda **values: updates.append(values))
    monkeypatch.setattr(tools, "_validate_hpc_solution_inputs", lambda *_: (root / "script.py", root, ""))
    def execute(args, scope):
        observed.append(args["timeout_seconds"])
        assert scope.execution_deadline is not None
        return ToolResult("", True, {}, "fixture")
    monkeypatch.setattr(tools, "_hpc_execute_solution_impl", execute)
    result = tools._hpc_execute_solution({"timeout_seconds": 86400}, context)
    assert result.ok and observed == [1200]
    assert result.content["gpu_budget"]["operations"] == 1
    assert context.execution_deadline is None
    assert updates[-1]["budget"]["gpu_limit_seconds"] == 43200


def test_coder_cannot_write_the_coordinator_state(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        session = runtime.create_session(objective="control isolation", workspace_root=str(tmp_path / "task"))
        engine = AIBuildEngine(runtime, session)
        class Client:
            def send(self, *args, **kwargs):
                call = SimpleNamespace(id="tamper", name="file_write", input={"path": str(engine.path / "budget.json"), "content": "{}"})
                return SimpleNamespace(text="", raw_content=[], tool_calls=[call], provider="fixture", model="fixture", input_tokens=1, output_tokens=1)
        engine.client_factory = Client
        task = SimpleNamespace(role="CoderAgent", task_id="coder", goal="attempt forbidden control write", solution_id="solution_01", dependencies=(), payload={})
        with pytest.raises(ValueError):
            engine._role(task, SimpleNamespace(input_evidence_refs=[]), SimpleNamespace(objective="control isolation", tasks={}))
        assert not runtime.store.list_tool_calls(session["id"])
    finally:
        runtime.close()


def test_registered_training_permission_requires_exact_configuration_approval(tmp_path):
    from evomind_runtime.models import PermissionLevel
    from evomind_runtime.policy import PolicyEngine, argument_fingerprint
    engine = PolicyEngine()
    read = engine.evaluate(tool_name="model_protocols", arguments={}, permission_level=PermissionLevel.OBSERVE.value, workspace_root=tmp_path)
    assert read.allowed and not read.requires_approval
    arguments = {"protocol_id": "a"*64, "solution_id": "solution_01", "configuration_path": "work/config.json", "configuration_sha256": "b"*64}
    for level in (PermissionLevel.WORKSPACE_WRITE.value, PermissionLevel.FULL_AUTO.value):
        pending = engine.evaluate(tool_name="managed_tensor_train", arguments=arguments, permission_level=level, workspace_root=tmp_path)
        assert not pending.allowed and pending.requires_approval
        fingerprint = argument_fingerprint("managed_tensor_train", pending.normalized_arguments)
        approved = engine.evaluate(tool_name="managed_tensor_train", arguments=arguments, permission_level=level, workspace_root=tmp_path, approved_fingerprint=fingerprint)
        assert approved.allowed and not approved.requires_approval
        changed = engine.evaluate(tool_name="managed_tensor_train", arguments={**arguments, "configuration_sha256": "c"*64}, permission_level=level, workspace_root=tmp_path, approved_fingerprint=fingerprint)
        assert not changed.allowed and changed.requires_approval
    observe = engine.evaluate(tool_name="managed_tensor_train", arguments=arguments, permission_level=PermissionLevel.OBSERVE.value, workspace_root=tmp_path)
    assert not observe.allowed
