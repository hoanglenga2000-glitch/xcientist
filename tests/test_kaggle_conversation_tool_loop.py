"""Provider-neutral native tool-loop integration tests for EvoMind."""
from __future__ import annotations

import hashlib
import json
import os

from research_os.agent import messaging
from xsci import kaggle_conversation
from xsci.kaggle_conversation import ConversationAgent
from xsci.kaggle_session import SessionState
from xsci.scientist_turns import record_scientist_turn
from xsci import terminal_tools


def _openai_only(monkeypatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_FILE",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY_FILE",
        "OPENAI_API_KEY_FILE",
        "OPENAI_SERVICE_TIER",
        "OPENAI_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "gateway-test-key-not-logged")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("EVOLUTION_PRIMARY_PROVIDER", "openai")
    monkeypatch.setenv("EVOLUTION_PROVIDER_STRICT", "1")


def test_openai_native_tool_call_result_final_answer_loop(monkeypatch, tmp_path):
    """The OpenAI path must execute tool_call -> tool result -> final answer."""

    _openai_only(monkeypatch)
    payloads: list[dict] = []

    def fake_post(url, headers, payload, timeout):
        assert url == "http://127.0.0.1:65068/v1/chat/completions"
        assert headers["Authorization"] == "Bearer gateway-test-key-not-logged"
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "model": "gpt-5.6-sol",
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "I will inspect the live status.",
                        "tool_calls": [{
                            "id": "call_status_1",
                            "type": "function",
                            "function": {"name": "system_status", "arguments": "{}"},
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 40, "completion_tokens": 8},
            }

        messages = payload["messages"]
        assistant = next(message for message in messages if message["role"] == "assistant")
        tool_result = next(message for message in messages if message["role"] == "tool")
        assert assistant["tool_calls"][0]["function"]["name"] == "system_status"
        assert tool_result["tool_call_id"] == "call_status_1"
        assert tool_result["content"] == "GPU idle and data ready"
        return {
            "model": "gpt-5.6-sol",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "Inspection complete: GPU idle and data ready."},
            }],
            "usage": {"prompt_tokens": 70, "completion_tokens": 12},
        }

    monkeypatch.setattr(messaging, "_post_json", fake_post)
    monkeypatch.setattr(
        kaggle_conversation,
        "_execute_agent_tool_call",
        lambda name, tool_input, session, **_kwargs: ("GPU idle and data ready", True),
    )

    session = SessionState(workspace_root=str(tmp_path), llm_ready=True, llm_provider="openai")
    agent = ConversationAgent()
    answer = agent._real_tool_loop(session, "Inspect current system status")

    assert answer == "Inspection complete: GPU idle and data ready."
    assert len(payloads) == 2
    assert agent._last_llm_execution == {
        "native_tool_loop": True,
        "requested_provider": "openai",
        "strict_provider": True,
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "native_tool_calls": 1,
        "tool_names": ["system_status"],
        "tool_rounds": 2,
        "input_tokens": 110,
        "output_tokens": 20,
        "stop_reason": "stop",
        "status": "completed",
    }


def test_hpc_connection_question_forces_connection_tool_and_counts_it(monkeypatch, tmp_path):
    _openai_only(monkeypatch)
    calls: list[str] = []
    payloads: list[dict] = []

    def fake_post(url, headers, payload, timeout):
        payloads.append(payload)
        return {
            "model": "gpt-5.6-sol",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "结论：job90673 的 HPC 连接状态已经检查，profile ready，未启动训练。"},
            }],
            "usage": {"prompt_tokens": 90, "completion_tokens": 20},
        }

    def fake_execute(name, session):
        calls.append(name)
        assert name == "hpc_connection_status"
        return "[hpc_connection_status] status=OK\nprofile: job90673\nreadiness_status: ready\ntraining_started: False"

    monkeypatch.setattr(messaging, "_post_json", fake_post)
    monkeypatch.setattr(kaggle_conversation, "_execute_terminal_tool", fake_execute)

    events: list[tuple[str, str, bool]] = []
    session = SessionState(workspace_root=str(tmp_path), llm_ready=True, llm_provider="openai")
    agent = ConversationAgent()
    answer = agent._real_tool_loop(
        session,
        "帮我解决这个连接服务器问题，作业号90673，像 Codex 一样先自己检查网关和 profile。",
        web_safe_context=True,
        on_tool_event=lambda phase, tool, ok: events.append((phase, tool, ok)),
    )

    assert "job90673" in answer
    assert calls == ["hpc_connection_status"]
    assert ("started", "hpc_connection_status", True) in events
    assert ("completed", "hpc_connection_status", True) in events
    assert agent._last_llm_execution["tool_names"] == ["hpc_connection_status"]
    assert agent._last_llm_execution["orchestrated_tool_calls"] == 1
    assert agent._last_llm_execution["tool_calls_total"] == 1
    assert "hpc_connection_status" in json.dumps(payloads[0], ensure_ascii=False)


def test_hpc_connection_transport_error_uses_precheck_fallback(monkeypatch, tmp_path):
    _openai_only(monkeypatch)
    calls: list[str] = []
    events: list[tuple[str, str, bool]] = []

    def fake_post(_url, _headers, _payload, _timeout):
        raise ConnectionError("gateway unavailable")

    def fake_execute(name, session):
        calls.append(name)
        assert name == "hpc_connection_status"
        return "\n".join([
            "[hpc_connection_status] status=OK",
            "profile: job90673",
            "job_id: 90673",
            "profile_state: active",
            "readiness_status: ready",
            "live_status: job_container_verified",
            "job_container_verified: True",
            "samples_passed: 5",
            "failed_checks: []",
            "gateway_banner_ok=True",
            "training_started=False",
            "kaggle_submissions: 0",
            "safe_next_action: profile is ready; execution still must pass task-specific gates before training",
            "readiness_artifact: workspace/hpc/job90673_profile_readiness_current.json",
        ])

    monkeypatch.setattr(messaging, "_post_json", fake_post)
    monkeypatch.setattr(kaggle_conversation, "_execute_terminal_tool", fake_execute)

    session = SessionState(workspace_root=str(tmp_path), llm_ready=True, llm_provider="openai")
    agent = ConversationAgent()
    answer = agent._real_tool_loop(
        session,
        "我是小白，帮我检查现在能不能连接服务器，作业号90673；不要训练，也不要提交 Kaggle。",
        web_safe_context=True,
        on_tool_event=lambda phase, tool, ok: events.append((phase, tool, ok)),
    )

    assert "job90673" in answer
    assert "已实时进入目标容器" in answer
    assert "未启动训练" in answer
    assert "未提交 Kaggle" in answer
    assert calls == ["hpc_connection_status"]
    assert ("started", "hpc_connection_status", True) in events
    assert ("completed", "hpc_connection_status", True) in events
    assert agent._last_llm_execution["tool_names"] == ["hpc_connection_status"]
    assert agent._last_llm_execution["tool_calls_total"] == 1
    assert agent._last_llm_execution["status"] == "transport_error"


def test_hpc_connection_status_tool_reports_active_profile_without_secret(monkeypatch, tmp_path):
    appdata = tmp_path / "AppData"
    profile = appdata / "ResearchAgentWorkstation" / "profiles" / "job90673"
    profile.mkdir(parents=True)
    metadata = {
        "schema": "evomind.hpc.dpapi_profile.v2",
        "credential_profile": "job90673",
        "job_id": 90673,
        "profile_state": "active",
        "allocation_generation": 1,
        "host": "100.85.169.63",
        "port": 1235,
        "socks_host": "127.0.0.1",
        "socks_port": 7890,
        "remote_workspace": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        "expected_host_uuid": "host-uuid",
        "expected_gpu_uuid": "GPU-12d4a1a8-3637-ad40-5599-fa3293cdecb6",
        "container_binding_sha256": "a" * 64,
    }
    (profile / "hpc_ssh_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    readiness_dir = tmp_path / "workspace" / "hpc"
    readiness_dir.mkdir(parents=True)
    (readiness_dir / "job90673_profile_readiness_current.json").write_text(json.dumps({
        "status": "ready",
        "checks": {"profile_state_active": True},
        "details": {"socks_host": "127.0.0.1", "socks_port": 7890},
        "failed_checks": [],
    }), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(terminal_tools, "_loopback_port_listening", lambda host, port: True)
    monkeypatch.setattr(terminal_tools, "_socks_gateway_banner", lambda *args, **kwargs: (True, "SSH-2.0-SSHPiper"))
    monkeypatch.setattr(terminal_tools, "_live_hpc_connection_probe", lambda profile, job_id, sample_count=5: {
        "ok": True,
        "status": "job_container_verified",
        "job_container_verified": True,
        "samples_requested": sample_count,
        "samples_passed": sample_count,
        "dpapi_decrypted": True,
        "ssh_connected": True,
        "remote_commands_attempted": sample_count,
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
    })

    session = SessionState(workspace_root=str(tmp_path))
    result = terminal_tools.inspect_hpc_connection_status(session, tmp_path)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["ok"] is True
    assert result["profile"] == "job90673"
    assert result["job_id"] == 90673
    assert result["socks_bridge"]["gateway_banner_ok"] is True
    assert result["job_container_verified"] is True
    assert result["samples_passed"] == 5
    assert result["boundaries"]["remote_commands"] == 5
    assert result["boundaries"]["training_started"] is False
    assert result["boundaries"]["secrets_returned"] is False
    assert "password" not in serialized.lower()
    assert "BdVq" not in serialized


def test_hpc_connection_status_blocks_gateway_only_false_positive(monkeypatch, tmp_path):
    appdata = tmp_path / "AppData"
    profile = appdata / "ResearchAgentWorkstation" / "profiles" / "job90673"
    profile.mkdir(parents=True)
    metadata = {
        "schema": "evomind.hpc.dpapi_profile.v2",
        "credential_profile": "job90673",
        "job_id": 90673,
        "profile_state": "active",
        "allocation_generation": 1,
        "host": "100.85.169.63",
        "port": 1235,
        "socks_host": "127.0.0.1",
        "socks_port": 7890,
        "remote_workspace": "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
        "expected_host_uuid": "host-uuid",
        "expected_gpu_uuid": "GPU-fixture",
        "container_binding_sha256": "a" * 64,
    }
    (profile / "hpc_ssh_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    readiness_dir = tmp_path / "workspace" / "hpc"
    readiness_dir.mkdir(parents=True)
    (readiness_dir / "job90673_profile_readiness_current.json").write_text(json.dumps({
        "status": "ready",
        "checks": {"profile_state_active": True},
        "failed_checks": [],
    }), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(terminal_tools, "_loopback_port_listening", lambda host, port: True)
    monkeypatch.setattr(terminal_tools, "_socks_gateway_banner", lambda *args, **kwargs: (True, "SSH-2.0-SSHPiper"))
    monkeypatch.setattr(terminal_tools, "_live_hpc_connection_probe", lambda profile, job_id, sample_count=5: {
        "ok": False,
        "status": "job_container_channel_closed",
        "job_container_verified": False,
        "samples_requested": sample_count,
        "samples_passed": 0,
        "dpapi_decrypted": True,
        "ssh_connected": True,
        "remote_commands_attempted": 1,
        "read_only": True,
        "signals_sent": 0,
        "other_processes_modified": False,
        "error_type": "EOFError",
        "reason": "SSH authentication completed, but the allocation container closed the session channel",
    })

    result = terminal_tools.inspect_hpc_connection_status(
        SessionState(workspace_root=str(tmp_path)),
        tmp_path,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["readiness_status"] == "ready"
    assert result["socks_bridge"]["gateway_banner_ok"] is True
    assert result["job_container_verified"] is False
    assert result["live_status"] == "job_container_channel_closed"
    assert "passed five" not in result["message"]
    assert result["boundaries"]["training_started"] is False
    assert result["boundaries"]["signals_sent"] == 0


def test_live_hpc_probe_isolates_named_profile_from_legacy_env(monkeypatch):
    from research_agent_workstation.server.core import gpu_credentials

    observed = {}

    class FakeClient:
        def close(self):
            observed["closed"] = True

    def fake_load(*, strict_named_profile=False):
        observed["strict"] = strict_named_profile
        observed["profile"] = os.environ.get("EVOMIND_HPC_CREDENTIAL_PROFILE")
        observed["legacy_present"] = any(
            name in os.environ for name in gpu_credentials.STRICT_NAMED_PROFILE_OVERRIDE_KEYS
        )
        return object()

    monkeypatch.setattr(gpu_credentials, "load_gpu_ssh_config", fake_load)
    monkeypatch.setattr(gpu_credentials, "connect_ssh", lambda config, timeout=30: FakeClient())
    monkeypatch.setattr(gpu_credentials, "verify_job_container_identity", lambda client, config, expected_job_id: {
        "job_container_verified": True,
        "gpu_name": "fixture-gpu",
        "gpu_memory_total_mib": 81920,
        "read_only": True,
    })
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "previous-profile")
    monkeypatch.setenv("GPU_SSH_HOST", "legacy.example")
    monkeypatch.setenv("GPU_SSH_PASSWORD", "legacy-secret")

    result = terminal_tools._live_hpc_connection_probe("job90948", 90948, sample_count=2)

    assert result["ok"] is True
    assert result["samples_passed"] == 2
    assert observed == {
        "strict": True,
        "profile": "job90948",
        "legacy_present": False,
        "closed": True,
    }
    assert os.environ["EVOMIND_HPC_CREDENTIAL_PROFILE"] == "previous-profile"
    assert os.environ["GPU_SSH_HOST"] == "legacy.example"
    assert os.environ["GPU_SSH_PASSWORD"] == "legacy-secret"


def test_live_hpc_probe_reconnects_after_transient_eof(monkeypatch):
    from research_agent_workstation.server.core import gpu_credentials

    connect_calls = []
    closed = []
    sleeps = []

    class FakeClient:
        def __init__(self, attempt):
            self.attempt = attempt

        def close(self):
            closed.append(self.attempt)

    def fake_connect(config, timeout=30):
        attempt = len(connect_calls) + 1
        connect_calls.append(attempt)
        return FakeClient(attempt)

    def fake_verify(client, config, expected_job_id):
        if client.attempt == 1:
            raise EOFError()
        return {
            "job_container_verified": True,
            "gpu_name": "fixture-gpu",
            "gpu_memory_total_mib": 81920,
            "read_only": True,
        }

    monkeypatch.setattr(gpu_credentials, "load_gpu_ssh_config", lambda **kwargs: object())
    monkeypatch.setattr(gpu_credentials, "connect_ssh", fake_connect)
    monkeypatch.setattr(gpu_credentials, "verify_job_container_identity", fake_verify)
    monkeypatch.setattr(terminal_tools.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = terminal_tools._live_hpc_connection_probe("job90948", 90948, sample_count=2)

    assert result["ok"] is True
    assert result["samples_passed"] == 2
    assert result["connection_attempts"] == 2
    assert result["transport_retries"] == 1
    assert result["remote_commands_attempted"] == 3
    assert connect_calls == [1, 2]
    assert closed == [1, 2]
    assert sleeps == [1.0]


def test_live_hpc_probe_does_not_retry_identity_failure(monkeypatch):
    from research_agent_workstation.server.core import gpu_credentials

    connect_calls = []
    sleeps = []

    class FakeClient:
        def close(self):
            pass

    def fake_connect(config, timeout=30):
        connect_calls.append(True)
        return FakeClient()

    def fail_identity(client, config, expected_job_id):
        raise gpu_credentials.CredentialError("job-container GPU UUID mismatch")

    monkeypatch.setattr(gpu_credentials, "load_gpu_ssh_config", lambda **kwargs: object())
    monkeypatch.setattr(gpu_credentials, "connect_ssh", fake_connect)
    monkeypatch.setattr(gpu_credentials, "verify_job_container_identity", fail_identity)
    monkeypatch.setattr(terminal_tools.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = terminal_tools._live_hpc_connection_probe("job90948", 90948, sample_count=1)

    assert result["ok"] is False
    assert result["connection_attempts"] == 1
    assert result["transport_retries"] == 0
    assert result["error_type"] == "CredentialError"
    assert connect_calls == [True]
    assert sleeps == []


def test_gpu_status_prefers_matching_strict_release_evidence(monkeypatch, tmp_path):
    appdata = tmp_path / "AppData"
    profile_dir = appdata / "ResearchAgentWorkstation" / "profiles" / "job90948"
    profile_dir.mkdir(parents=True)
    (profile_dir / "hpc_ssh_metadata.json").write_text(json.dumps({
        "credential_profile": "job90948",
        "job_id": 90948,
        "profile_state": "active",
        "socks_host": "127.0.0.1",
        "socks_port": 7890,
    }), encoding="utf-8")
    report_dir = tmp_path / "docs"
    report_dir.mkdir()
    (report_dir / "launch_resource_readiness.json").write_text(json.dumps({
        "strict_hpc_runtime_status": "ready",
        "external_resources": {
            "hpc_gpu_strict_runtime": {
                "state": "strict_hpc_runtime_verified",
                "profile": {"profile": "job90948", "job_id": 90948},
                "live_probe": {
                    "job_container_verified": True,
                    "samples_passed": 5,
                    "samples": [{"gpu_name": "fixture-gpu"}],
                },
                "bounded_smoke": {"status": "passed"},
            }
        },
    }), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(terminal_tools, "_loopback_port_listening", lambda host, port: True)
    session = SessionState(
        workspace_root=str(tmp_path),
        gpu_ready=True,
        gpu_status="configured_channels_closed",
        gpu_blocker="legacy blocker",
        gpu_blocked=True,
    )

    result = terminal_tools.inspect_gpu_status(session, tmp_path)

    assert result["blocked"] is False
    assert result["can_execute_gpu"] is True
    assert result["manifest_status"] == "strict_hpc_runtime_verified"
    assert result["strict_hpc_evidence"]["profile"] == "job90948"
    assert result["strict_hpc_evidence"]["samples_passed"] == 5


def test_verified_context_tool_returns_hashed_current_run_deliverables(tmp_path):
    run_id = "evomind_siim_isic_a800_job90353_20260730_095826"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    delivery = run_dir / "delivery"
    delivery.mkdir(parents=True)
    report = delivery / "report.pdf"
    report.write_bytes(b"verified-report")
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    (tmp_path / "workspace" / "current_run.json").write_text(json.dumps({
        "schema": "evomind.current_run.v1",
        "task_id": "siim-isic-melanoma-classification",
        "run_id": run_id,
        "run_dir": f"workspace/evomind_runs/{run_id}",
        "status": "completed",
    }), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id,
        "status": "completed",
        "tasks": {},
        "gates": {},
    }), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "roc_auc": 0.9225,
        "pr_auc": 0.2397,
        "brier": 0.2952,
        "metric_scope": "independent_offline_patient_content_grouped_oof",
    }), encoding="utf-8")
    (run_dir / "review.json").write_text(json.dumps({
        "status": "passed",
        "checks": {"patient_group_overlap_zero": True},
    }), encoding="utf-8")
    (run_dir / "private_grader.json").write_text(json.dumps({
        "status": "failed_closed",
        "score": None,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
    }), encoding="utf-8")
    (run_dir / "private_grader_ledger.json").write_text(json.dumps({
        "execution_count": 1,
        "outcome": "failed_closed",
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
    }), encoding="utf-8")
    (run_dir / "deliverables.json").write_text(json.dumps({
        "files": [{
            "name": "report.pdf",
            "path": "report.pdf",
            "bytes": report.stat().st_size,
            "sha256": digest,
            "download_url": "/api/download/report.pdf",
        }],
    }), encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text(json.dumps({
        "review_status": "passed",
        "claim_audit_status": "passed",
        "artifacts": [{
            "path": "delivery/report.pdf",
            "bytes": report.stat().st_size,
            "sha256": digest,
        }],
    }), encoding="utf-8")
    session = SessionState(workspace_root=str(tmp_path), selected_task="siim-isic-melanoma-classification")

    result, ok = kaggle_conversation._execute_agent_tool_call(
        "verified_context",
        {"section": "artifacts"},
        session,
    )
    payload = json.loads(result)

    assert ok is True
    assert payload["run_id"] == run_id
    assert payload["data"]["artifacts"]["deliverables"][0]["absolute_path"] == str(report.resolve())
    assert payload["data"]["artifacts"]["deliverables"][0]["sha256"] == digest
    assert payload["data"]["metrics"]["roc_auc"] == 0.9225
    assert payload["data"]["review"]["status"] == "passed"
    assert payload["data"]["private_grader"] == {
        "status": "failed_closed",
        "score": None,
        "execution_count": 1,
        "feedback_used_for_tuning": False,
        "official_submission_executed": False,
    }
    assert payload["data"]["governance"]["official_kaggle_submission"] == "human_gate"

    web_result, web_ok = kaggle_conversation._execute_agent_tool_call(
        "verified_context",
        {"section": "artifacts"},
        session,
        web_safe=True,
    )
    web_payload = json.loads(web_result)
    projected = web_payload["data"]["artifacts"]
    assert web_ok is True
    assert "hash_prefixes" not in projected
    assert "absolute_path" not in json.dumps(projected)
    assert projected["deliverables"][0]["download_url"] == "/api/download/report.pdf"
    assert projected["deliverables"][0]["sha256"] == digest


def test_verified_context_does_not_mix_current_run_from_another_task(tmp_path):
    run_id = "siim_run"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    (tmp_path / "workspace" / "current_run.json").write_text(json.dumps({
        "schema": "evomind.current_run.v1",
        "task_id": "siim-isic-melanoma-classification",
        "run_id": run_id,
        "run_dir": f"workspace/evomind_runs/{run_id}",
        "status": "completed",
    }), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id,
        "task_id": "siim-isic-melanoma-classification",
        "status": "completed",
        "tasks": {},
        "gates": {},
    }), encoding="utf-8")
    session = SessionState(
        workspace_root=str(tmp_path),
        selected_task="tabular-playground-series-dec-2021",
    )

    summary_result, summary_ok = kaggle_conversation._execute_agent_tool_call(
        "verified_context", {"section": "summary"}, session, web_safe=True,
    )
    summary = json.loads(summary_result)
    metrics_result, metrics_ok = kaggle_conversation._execute_agent_tool_call(
        "verified_context", {"section": "metrics"}, session, web_safe=True,
    )
    metrics = json.loads(metrics_result)

    assert summary_ok is True
    assert summary["selected_task"] == "tabular-playground-series-dec-2021"
    assert summary["run_id"] is None
    assert summary["data"]["task_label"] == "tabular-playground-series-dec-2021"
    assert summary["data"]["current_run_matches_selected_task"] is False
    assert metrics_ok is True
    assert metrics["run_id"] is None
    assert metrics["data"]["available"] is False
    assert metrics["data"]["reason"] == "no_metrics_for_selected_task_in_current_run"


def test_verified_context_resolves_task_bound_historical_run_and_literature(tmp_path):
    current_id = "wr_titanic_current"
    current_dir = tmp_path / "workspace" / "evomind_runs" / current_id
    current_dir.mkdir(parents=True)
    (tmp_path / "workspace" / "current_run.json").write_text(json.dumps({
        "schema": "evomind.current_run.v1",
        "task_id": "titanic",
        "run_id": current_id,
        "run_dir": f"workspace/evomind_runs/{current_id}",
        "status": "COMPLETED",
    }), encoding="utf-8")
    (current_dir / "run.json").write_text(json.dumps({
        "run_id": current_id,
        "task_id": "titanic",
        "status": "COMPLETED",
        "tasks": {},
        "gates": {},
    }), encoding="utf-8")

    task_id = "siim-isic-melanoma-classification"
    historical_id = "evomind_siim_isic_a800_job90353_20260730_095826"
    historical_dir = tmp_path / "workspace" / "evomind_runs" / historical_id
    historical_dir.mkdir(parents=True)
    (historical_dir / "run.json").write_text(json.dumps({
        "run_id": historical_id,
        "status": "completed",
        "tasks": {},
        "gates": {},
    }), encoding="utf-8")
    (historical_dir / "artifact_manifest.json").write_text(json.dumps({
        "schema": "evomind.siim.artifact_manifest.v1",
        "run_id": historical_id,
        "task_id": task_id,
        "status": "verified",
        "artifacts": [],
    }), encoding="utf-8")
    (historical_dir / "metrics.json").write_text(json.dumps({
        "roc_auc": 0.9225357247684676,
        "pr_auc": 0.23970933285011597,
        "brier": 0.29524735217259324,
    }), encoding="utf-8")
    rag_dir = tmp_path / "workspace" / "tasks" / task_id / "rag"
    rag_dir.mkdir(parents=True)
    (rag_dir / "context_2026-08-06T05-43-56-445Z.json").write_text(json.dumps({
        "task_id": task_id,
        "query": "SIIM ISIC melanoma",
        "integrity": {"external_verified": 2, "fabricated": 0},
        "papers": [
            {
                "title": "Patient-contextual melanoma diagnosis",
                "year": "2024",
                "source": "crossref",
                "doi": "10.1111/jdv.20479",
                "url": "https://doi.org/10.1111/jdv.20479",
            },
            {
                "title": "Deep learning in medical image analysis",
                "year": "2021",
                "source": "crossref",
                "doi": "10.1016/j.media.2021.102305",
                "url": "https://doi.org/10.1016/j.media.2021.102305",
            },
        ],
    }), encoding="utf-8")
    session = SessionState(workspace_root=str(tmp_path), selected_task=task_id)

    metrics_result, metrics_ok = kaggle_conversation._execute_agent_tool_call(
        "verified_context", {"section": "metrics"}, session, web_safe=True,
    )
    literature_result, literature_ok = kaggle_conversation._execute_agent_tool_call(
        "verified_context", {"section": "literature"}, session, web_safe=True,
    )
    metrics = json.loads(metrics_result)
    literature = json.loads(literature_result)

    assert metrics_ok is True
    assert metrics["run_id"] == historical_id
    assert metrics["run_status"] == "completed"
    assert metrics["data"]["metrics"]["roc_auc"] == 0.9225357247684676
    assert metrics["data"]["metrics"]["pr_auc"] == 0.23970933285011597
    assert metrics["data"]["metrics"]["brier"] == 0.29524735217259324
    assert literature_ok is True
    assert literature["run_id"] == historical_id
    assert [item["doi"] for item in literature["data"]["literature"]["papers"]] == [
        "10.1111/jdv.20479",
        "10.1016/j.media.2021.102305",
    ]


def test_runtime_tool_paths_resolve_from_active_project_inside_wider_root(tmp_path):
    agent_root = tmp_path / "desktop"
    project_root = agent_root / "codex" / "科研港科技"
    report = project_root / "video-production" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")

    resolved = kaggle_conversation._project_relative_runtime_tool_input(
        {
            "path": "video-production/report.json",
            "source": "video-production/report.json",
            "destination": "video-production/report-copy.json",
            "cwd": "video-production",
        },
        agent_root=agent_root,
        project_root=project_root,
    )

    assert resolved == {
        "path": "codex/科研港科技/video-production/report.json",
        "source": "codex/科研港科技/video-production/report.json",
        "destination": "codex/科研港科技/video-production/report-copy.json",
        "cwd": "codex/科研港科技/video-production",
    }


def test_scientist_turn_artifact_persists_native_llm_evidence(tmp_path):
    evidence = {
        "native_tool_loop": True,
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "native_tool_calls": 2,
        "status": "completed",
    }

    entry = record_scientist_turn(tmp_path, {
        "route": "llm_tool_loop",
        "user": "inspect",
        "llm_execution": evidence,
    })
    latest = json.loads((tmp_path / ".xsci" / "scientist_latest_turn.json").read_text(encoding="utf-8"))

    assert entry["llm_execution"] == evidence
    assert latest["llm_execution"] == evidence


def test_web_agent_system_prompt_uses_novice_codex_style_protocol(monkeypatch, tmp_path):
    """The web assistant must be guided as an LLM-first UX orchestrator, not a template bot."""

    _openai_only(monkeypatch)
    from xsci import recovery_guard

    monkeypatch.setattr(
        recovery_guard,
        "build_compaction_recovery_block",
        lambda _path: "OLD_RECOVERY allocation-legacy-123 house_prices SalePrice",
    )
    payloads: list[dict] = []

    def fake_post(url, headers, payload, timeout):
        payloads.append(payload)
        system = payload["messages"][0]["content"]
        assert "完全小白" in system
        assert "像 Codex/Claude Code 一样做任务编排" in system
        assert "先回答用户真正关心的结论" in system
        assert "你可以直接发这句话" in system
        assert "不要把配置 Kaggle、提交榜单" in system
        assert "必须逐字复制为 Markdown 链接" in system
        assert "使用渐进式披露" in system
        assert "多轮追问只回答本轮新增问题" in system
        assert "不要求小白先学会专业 prompt" in system
        assert "静默完成度检查" in system
        assert "现在安全可做" in system
        assert "只有用户明确询问文件、交付物、报告、下载链接或完整证据包" in system
        assert payload["max_tokens"] == 900
        wire = json.dumps(payload, ensure_ascii=False)
        request_wire = payload["messages"][-1]["content"]
        assert "allocation-legacy-123" not in wire
        assert "house_prices" not in wire
        assert "SalePrice" not in wire
        if len(payloads) == 1:
            assert {tool["function"]["name"] for tool in payload["tools"]} == (
                set(kaggle_conversation._RUNTIME_AGENT_TOOL_NAMES) | {"verified_context"}
            )
            assert "[RESPONSE COMPLETION CONTRACT]" in wire
            assert "[WEB REQUEST COMPILER" in wire
            assert "[DISTILLED INTERACTION BEHAVIOURS" in wire
            assert '"audience":"novice"' in request_wire
            assert '"side_effect_mode":"read_only_reasoning"' in request_wire
            assert "逐项回答用户在本轮明确提出的所有问题" in wire
            content = "结论：我会先理解目标，再检查证据并给出下一步。"
        else:
            assert payload["tools"] == []
            assert "[VISIBLE RESPONSE AUDIT]" in request_wire
            assert "[VERIFIED DRAFT TO REPAIR" in request_wire
            assert "结论：我会先理解目标，再检查证据并给出下一步。" in request_wire
            content = (
                "结论：我会先理解你的实验目标。下一步计划是先做只读检查，"
                "确认数据、指标和资源门禁后再进入受控执行。"
                "你可以直接发这句话：帮我开始只读检查。"
            )
        return {
            "model": "gpt-5.6-sol",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": content},
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 16},
        }

    monkeypatch.setattr(messaging, "_post_json", fake_post)
    session = SessionState(
        workspace_root=str(tmp_path),
        selected_task="siim-isic-melanoma-classification",
        task_brief="name=house_prices | target=SalePrice",
        gpu_blocker="allocation-legacy-123",
        llm_ready=True,
        llm_provider="openai",
    )
    agent = ConversationAgent()

    answer = agent.agent_reply("我是小白，怎么让系统帮我做一次实验？", session, history=[])

    assert answer.startswith("结论：")
    assert len(payloads) == 2
    assert agent._last_llm_execution["status"] == "completed"
    assert agent._last_llm_execution["repair_rounds"] == 1
    assert agent._last_llm_execution["response_audit"]["after"]["passed"] is True


def test_web_agent_orchestrator_prefetches_required_evidence_before_llm(monkeypatch, tmp_path):
    _openai_only(monkeypatch)
    payloads: list[dict] = []
    events: list[tuple[str, str, bool]] = []

    def fake_execute(name, tool_input, session, **_kwargs):
        del session
        assert name == "verified_context"
        assert tool_input == {"section": "metrics"}
        return '{"run_id":"verified_run","metrics":{"roc_auc":0.9225}}', True

    def fake_post(url, headers, payload, timeout):
        del url, headers, timeout
        payloads.append(payload)
        wire = json.dumps(payload, ensure_ascii=False)
        assert "[ORCHESTRATED VERIFIED CONTEXT" in wire
        assert "verified_run" in wire
        assert {tool["function"]["name"] for tool in payload["tools"]} == set(
            kaggle_conversation._RUNTIME_AGENT_TOOL_NAMES
        )
        return {
            "model": "gpt-5.6-sol",
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "结论：简单说，这次结果指标 ROC-AUC=0.9225。"
                        "证据来自 verified_run。下一步保持同一分组做单变量比较。"
                        "你可以直接复制下一句继续。"
                    )
                },
            }],
            "usage": {"prompt_tokens": 80, "completion_tokens": 30},
        }

    monkeypatch.setattr(kaggle_conversation, "_execute_agent_tool_call", fake_execute)
    monkeypatch.setattr(messaging, "_post_json", fake_post)
    session = SessionState(
        workspace_root=str(tmp_path),
        selected_task="siim-isic-melanoma-classification",
        llm_ready=True,
        llm_provider="openai",
    )
    agent = ConversationAgent()

    answer = agent.agent_reply(
        "我是小白，先不要训练，帮我解释上次结果、证据和下一步。",
        session,
        history=[],
        on_tool_event=lambda phase, name, ok: events.append((phase, name, ok)),
    )

    assert "ROC-AUC=0.9225" in answer
    assert "本轮未启动训练" in answer
    assert len(payloads) == 1
    assert events == [
        ("started", "verified_context", True),
        ("completed", "verified_context", True),
    ]
    assert agent._last_llm_execution["native_tool_calls"] == 0
    assert agent._last_llm_execution["orchestrated_tool_calls"] == 1
    assert agent._last_llm_execution["tool_calls_total"] == 1
    assert agent._last_llm_execution["tool_names"] == ["verified_context"]
    assert agent._last_llm_execution["response_audit"]["after"]["passed"] is True


def test_web_output_budget_uses_progressive_disclosure():
    budget = kaggle_conversation._web_output_budget

    assert budget("我是小白，帮我解释上次实验结果", []) == 900
    assert budget("这些实验参考了哪些论文和 DOI？", []) == 700
    assert budget("那我下一步做什么？", [{"role": "assistant", "content": "上一轮"}]) == 700
    assert budget("把全部交付物、下载链接和 SHA-256 逐项给我", []) == 2200
    assert budget("给我一份完整详细的实验报告", []) == 2200


def test_web_request_compiler_preserves_novice_facets_and_hard_constraints():
    compiled = kaggle_conversation._web_request_compiler_context(
        "我是第一次用的小白。帮我看上次 SIIM 结果、真实证据和论文，再给三个下一步；不要训练，不要提交 Kaggle。",
        [],
    )
    payload = json.loads(compiled.splitlines()[-1])

    assert payload["schema"] == "evomind.web_request_compiler.v3"
    assert payload["conversation_mode"] == "first_turn"
    assert payload["audience"] == "novice"
    assert payload["task"]["dataset"] == "siim-isic-melanoma-classification"
    assert payload["side_effect_mode"] == "read_only_reasoning"
    assert payload["preferred_evidence_section"] == "literature"
    assert set(payload["intent"]["facets"]) >= {
        "result_explanation",
        "evidence",
        "literature",
        "planning",
    }
    assert set(payload["hard_constraints"]) >= {"no_training", "no_official_submission"}
    assert payload["requested_outputs"] == []
    assert payload["evidence_required"] is True
    assert payload["interaction_contract"]["audit_visible_answer_before_return"] is True
    assert payload["user_experience_contract"]["schema"] == "evomind.user_experience_contract.v1"
    assert payload["user_experience_contract"]["copyable_followup_required"] is True
    assert "goal_compile_for_novice_v1" in payload["behavior_card_ids"]
    assert "evidence_before_claim_v1" in payload["behavior_card_ids"]
    assert "不要训练" not in compiled


def test_web_request_compiler_marks_followup_and_only_expands_explicit_artifacts():
    history = [{"role": "assistant", "content": "上一轮已解释结果。"}]
    followup = kaggle_conversation._web_request_compiler_context("那下一步呢？", history)
    artifacts = kaggle_conversation._web_request_compiler_context(
        "把 SIIM 的全部交付物、下载链接和 SHA-256 逐项给我。",
        history,
    )
    followup_payload = json.loads(followup.splitlines()[-1])
    artifact_payload = json.loads(artifacts.splitlines()[-1])

    assert followup_payload["conversation_mode"] == "follow_up"
    assert followup_payload["answer_depth"] == "progressive"
    assert followup_payload["requested_outputs"] == []
    assert artifact_payload["answer_depth"] == "detailed"
    assert artifact_payload["preferred_evidence_section"] == "artifacts"
    assert "evomind-siim-isic-report.pdf" in artifact_payload["requested_outputs"]


def test_web_tool_specs_keep_core_agent_tools_and_focus_research_tools():
    specs = kaggle_conversation._terminal_tool_specs()
    select = kaggle_conversation._web_tool_specs_for_user
    core = set(kaggle_conversation._RUNTIME_AGENT_TOOL_NAMES) | {"verified_context"}

    assert {item.name for item in select("解释上次 SIIM 结果", specs)} == core
    assert {item.name for item in select("这些结果参考了哪些论文 DOI？", specs)} == core | {"literature_search"}
    assert {item.name for item in select("检查 GPU 和系统状态", specs)} == core | {"gpu_status", "system_status"}
    assert {item.name for item in select("job90948 链接了吗？", specs)} == core | {"hpc_connection_status"}
    assert {item.name for item in select("给我创新假设和下一步", specs)} == core | {
        "scientist_innovation_backlog", "scientist_hypothesis_review", "scientist_experiment_blueprint",
    }
    assert {item.name for item in select("系统当前阻塞和下一安全动作是什么？", specs)} == core | {"next_steps"}
    preferred = kaggle_conversation._preferred_verified_context_section
    assert preferred("解释上次实验结果和 ROC-AUC") == "metrics"
    assert preferred("上次 SIIM 实验能说明什么，有哪些真实证据？") == "metrics"
    assert preferred("给我全部文件、下载链接和 SHA-256") == "artifacts"
    assert preferred("这些结论参考了哪些论文 DOI") == "literature"
    assert preferred("当前 Run 和 grader 状态") == "current_run"


def test_failed_live_literature_search_falls_back_to_reviewed_context(monkeypatch, tmp_path):
    _openai_only(monkeypatch)
    payloads: list[dict] = []

    def fake_post(url, headers, payload, timeout):
        del url, headers, timeout
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "model": "gpt-5.6-sol",
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "call_literature_1",
                            "type": "function",
                            "function": {
                                "name": "literature_search",
                                "arguments": '{"query":"patient grouped validation"}',
                            },
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
        tool_result = next(message for message in payload["messages"] if message["role"] == "tool")
        assert "REVIEWED LITERATURE FALLBACK" in tool_result["content"]
        assert "10.1111/jdv.20479" in tool_result["content"]
        return {
            "model": "gpt-5.6-sol",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "已审核文献 DOI：10.1111/jdv.20479"},
            }],
            "usage": {"prompt_tokens": 70, "completion_tokens": 12},
        }

    def fake_execute(name, tool_input, session, **_kwargs):
        del tool_input, session
        if name == "literature_search":
            return '{"ok":false,"message":"live unavailable"}', False
        if name == "verified_context":
            return '{"section":"literature","doi":"10.1111/jdv.20479"}', True
        raise AssertionError(name)

    monkeypatch.setattr(messaging, "_post_json", fake_post)
    monkeypatch.setattr(kaggle_conversation, "_execute_agent_tool_call", fake_execute)
    session = SessionState(
        workspace_root=str(tmp_path),
        selected_task="siim-isic-melanoma-classification",
        llm_ready=True,
        llm_provider="openai",
    )

    answer = ConversationAgent()._real_tool_loop(
        session,
        "请结合参考文献解释患者分组验证。",
        web_safe_context=True,
    )

    assert answer == "已审核文献 DOI：10.1111/jdv.20479"
    assert len(payloads) == 2
