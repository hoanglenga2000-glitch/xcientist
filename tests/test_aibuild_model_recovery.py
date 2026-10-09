import hashlib
import json
from types import SimpleNamespace

import pytest

from evomind_runtime.aibuild_engine import AIBuildEngine
from evomind_runtime.aibuild_model_recovery import reconcile_model_failure
from evomind_runtime.model_transport import ModelTransportError
from evomind_runtime.runtime import AgentRuntime
from research_os.agent.multi_agent import MultiAgentStore


def failed_setup(runtime, tmp_path):
    class Broken:
        def send(self, *_args, **_kwargs):
            raise ModelTransportError("http_401", retryable=False, status=401)
    run = runtime.assistant.create_run(prompt="recovery fixture", start=False)
    session = runtime.get_session(run["id"])
    runtime.store.update_session(run["id"], metadata_json={**session["metadata"], "research_orchestrator": "aibuildai2"})
    session = runtime.get_session(run["id"])
    engine = AIBuildEngine(runtime, session, client_factory=Broken)
    assert engine.execute()["status"] == "blocked"
    return run, engine


def test_explicit_recovery_preserves_run_failure_and_budget(tmp_path, monkeypatch):
    runtime = AgentRuntime(tmp_path)
    try:
        run, engine = failed_setup(runtime, tmp_path)
        before = (engine.path / "run.json").read_bytes()
        budget = (engine.path / "budget.json").read_bytes()
        seen = []
        monkeypatch.setattr(runtime.assistant, "start", lambda identifier, **kw: seen.append((identifier, kw)) or True)
        snapshot = runtime.assistant.action(run["id"], "resume", "Retry only the failed model call", idempotency_key="recovery-fixture")
        assert snapshot["id"] == run["id"]
        assert len(runtime.store.list_assistant_runs()) == 1
        assert seen == [(run["id"], {"resume": True})]
        graph = MultiAgentStore(engine.path).load()
        assert graph.tasks["setup"].status == "ready"
        assert graph.tasks["setup"].attempts == 1
        assert graph.tasks["setup"].payload["user_continuation"] == "Retry only the failed model call"
        assert (engine.path / "budget.json").read_bytes() == budget
        full_request_id = hashlib.sha256(b"recovery-fixture").hexdigest()
        journal = engine.path / "model-reconciliations" / full_request_id[:24]
        assert len(str(journal / "before-budget.json")) < 260
        assert (journal / "before-run.json").read_bytes() == before
        receipt = json.loads((journal / "receipt.json").read_text())
        assert receipt["training_started"] is False
        assert receipt["request_id"] == full_request_id
        assert not runtime.store.list_tool_calls(run["id"])
    finally:
        runtime.close()


def test_recovery_refuses_pending_operation_and_wrong_role(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, engine = failed_setup(runtime, tmp_path)
        pending = engine.path / "setup.pending.json"
        pending.write_text("{}")
        with pytest.raises(ValueError, match="requires_settlement"):
            reconcile_model_failure(runtime, run["id"], "retry")
        pending.rename(engine.path / "pending-preserved.json")
        store = MultiAgentStore(engine.path)
        graph = store.load()
        graph.tasks["setup"].role = "TunerAgent"
        store.save(graph)
        with pytest.raises(ValueError, match="requires_role_review"):
            reconcile_model_failure(runtime, run["id"], "retry")
    finally:
        runtime.close()


def test_role_transport_retries_are_charged_as_model_calls(tmp_path, monkeypatch):
    import evomind_runtime.aibuild_engine as module
    runtime = AgentRuntime(tmp_path)
    class Flaky:
        count = 0
        def send(self, *_args, **_kwargs):
            self.count += 1
            if self.count < 3:
                raise ModelTransportError("upstream_error", retryable=True)
            return SimpleNamespace(text="ready", tool_calls=[], provider="openai", model="gpt-6-astra", input_tokens=5, output_tokens=2)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    try:
        session = runtime.create_session(objective="fixture")
        engine = AIBuildEngine(runtime, session)
        assert engine._send(Flaky(), [], "fixture", []).text == "ready"
        assert engine.calls == 3
        assert engine.tokens == 7
    finally:
        runtime.close()


def test_short_journal_name_still_checks_complete_request_identity(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, engine = failed_setup(runtime, tmp_path)
        before = (engine.path / "run.json").read_bytes()
        request_id = hashlib.sha256(b"collision-fixture").hexdigest()
        fingerprint = hashlib.sha256(json.dumps({"run": run["id"], "task": "setup",
                                                "graph_sha256": hashlib.sha256(before).hexdigest(),
                                                "instruction": "retry"}, sort_keys=True).encode()).hexdigest()
        journal = engine.path / "model-reconciliations" / request_id[:24]
        journal.mkdir(parents=True)
        different_id = request_id[:-1] + ("0" if request_id[-1] != "0" else "1")
        (journal / "receipt.json").write_text(json.dumps({"fingerprint": fingerprint, "request_id": different_id}))
        with pytest.raises(ValueError, match="research_resume_request_conflict"):
            reconcile_model_failure(runtime, run["id"], "retry", "collision-fixture")
        assert (engine.path / "run.json").read_bytes() == before
    finally:
        runtime.close()
