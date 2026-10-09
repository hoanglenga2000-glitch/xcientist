from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_os import llm_client
from research_os.llm_client import LLMClient, LLMError, LLMStreamEvent, ProviderConfig
from xsci import assistant_stream, kaggle_conversation, kaggle_intent
from xsci import kaggle as kaggle_cli
from xsci.kaggle_conversation import ConversationAgent


class _SwitchSession:
    def __init__(self, root: Path) -> None:
        self.workspace_root = str(root)
        self.selected_task = "siim-isic-melanoma-classification"
        self.task_brief = ""
        self.recent_run_id = ""
        self.recent_events_path = ""
        self.recent_best_cv = None

    def refresh_task_brief(self, root: Path) -> None:
        self.task_brief = "registered task brief"

    def refresh_recent_run(self, root: Path) -> None:
        self.recent_run_id = ""

    def persist(self, root: Path | None = None) -> None:
        self.persisted = True


def test_public_context_uses_request_scoped_selected_task(tmp_path):
    session = _SwitchSession(tmp_path)
    session.selected_task = "tabular-playground-series-dec-2021"
    stale_context = SimpleNamespace(public_status=lambda: {
        "current_task": True,
        "task_label": "siim-isic-melanoma-classification",
        "memory_available": True,
        "tools_available": True,
    })

    status = assistant_stream._public_context_for_session(stale_context, session)

    assert status["task_label"] == "tabular-playground-series-dec-2021"
    assert status["current_task"] is True
    assert status["memory_available"] is True
    assert status["tools_available"] is True


def test_training_coverage_report_request_stays_read_only():
    prompt = (
        "请读取 reports/mlebench_lite22_training_coverage.json，"
        "总结 22 个比赛的训练覆盖、private grader 和达标情况。"
    )

    assert assistant_stream._prompt_requests_training_execution(prompt) is False
    assert assistant_stream._prompt_requests_training_execution(
        "我是第一次用 EvoMind，请帮我训练这个比赛，最后展示报告。"
    ) is True


def test_web_agent_natural_language_switches_to_evolution_config(tmp_path):
    config_dir = tmp_path / "configs" / "evolution"
    config_dir.mkdir(parents=True)
    (config_dir / "evomind_demo_customer_churn.json").write_text(
        json.dumps(
            {
                "task_name": "evomind_demo_customer_churn",
                "display_name": "客户流失多轮进化演示",
                "modality": "tabular",
                "task_type": "binary_classification",
                "metric": "roc_auc",
                "metric_direction": "maximize",
                "local_data_dir": "workspace/demo_campaigns/evomind_demo_customer_churn/data",
                "data_schema": "Deterministic synthetic customer churn dataset.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    session = _SwitchSession(tmp_path)

    target = kaggle_conversation._infer_switch_task_target(
        "请切换到 evomind_demo_customer_churn 客户流失预测任务",
        tmp_path,
    )
    out, ok = kaggle_conversation._switch_session_task(session, tmp_path, target)

    assert ok is True
    assert target == "evomind_demo_customer_churn"
    assert session.selected_task == "evomind_demo_customer_churn"
    assert "source=evolution_config" in out
    assert "metric=roc_auc" in session.task_brief


def test_experiment_results_tool_aggregates_local_summaries(tmp_path):
    base = tmp_path / "experiments" / "evolution"
    for index, score in enumerate((0.81, 0.92), 1):
        run = base / f"churn_run_{index}"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "task": "evomind_demo_customer_churn",
                    "best_exp_id": f"EXP00{index}",
                    "best_cv_score": score,
                    "metric": "roc_auc",
                    "metric_direction": "maximize",
                    "n_iterations": 8,
                    "n_promotions": index,
                }
            ),
            encoding="utf-8",
        )

    payload = kaggle_conversation._experiment_results_payload(
        tmp_path,
        "evomind_demo_customer_churn",
    )

    assert payload["run_count"] == 2
    assert payload["best_cv_score"] == 0.92
    assert payload["total_iterations"] == 16
    assert payload["total_promotions"] == 3


def test_anthropic_stream_parser_emits_thinking_text_and_done():
    records = iter([
        ("message_start", json.dumps({"type": "message_start", "message": {"model": "opus", "usage": {"input_tokens": 7}}})),
        ("content_block_delta", json.dumps({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hidden"}})),
        ("content_block_delta", json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}})),
        ("message_delta", json.dumps({"type": "message_delta", "usage": {"output_tokens": 3}})),
        ("message_stop", json.dumps({"type": "message_stop"})),
    ])
    events = list(llm_client._anthropic_stream_events(records, ProviderConfig("anthropic", "http://test", "opus", "secret")))
    assert [(event.kind, event.text) for event in events] == [
        ("start", ""), ("thinking_delta", "hidden"), ("text_delta", "hello"), ("done", ""),
    ]
    assert events[-1].input_tokens == 7
    assert events[-1].output_tokens == 3


def test_openai_stream_parser_emits_reasoning_and_text():
    records = iter([
        ("message", json.dumps({"model": "deepseek", "choices": [{"delta": {"reasoning_content": "hidden", "content": "visible"}, "finish_reason": None}]})),
        ("message", json.dumps({"model": "deepseek", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})),
    ])
    events = list(llm_client._openai_stream_events(records, ProviderConfig("deepseek", "http://test", "deepseek", "secret")))
    assert [(event.kind, event.text) for event in events] == [
        ("thinking_delta", "hidden"), ("text_delta", "visible"), ("done", ""),
    ]
    assert events[-1].input_tokens == 5
    assert events[-1].output_tokens == 2


def test_llm_client_resolves_generic_openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "gateway-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:65068/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_SERVICE_TIER", "priority")

    config = LLMClient(primary="openai", fallback="openai")._resolve("openai")

    assert config is not None
    assert config.name == "openai"
    assert config.model == "gpt-5.6-sol"
    assert config.reasoning_effort == "low"
    assert config.service_tier == "priority"
    assert llm_client._openai_chat_url(config.base_url) == (
        "http://127.0.0.1:65068/v1/chat/completions"
    )


def test_openai_stream_request_carries_validated_performance_profile(monkeypatch):
    captured = {}

    def fake_post_sse(url, headers, payload, timeout):
        captured.update(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
        )
        return iter([("message", "[DONE]")])

    monkeypatch.setattr(llm_client, "_post_sse", fake_post_sse)
    config = ProviderConfig(
        "openai",
        "http://127.0.0.1:65068/v1",
        "gpt-5.6-sol",
        "secret",
        reasoning_effort="low",
        service_tier="priority",
    )

    events = list(
        llm_client._stream_provider(
            config,
            system="system",
            user="probe",
            max_tokens=32,
            temperature=0,
            timeout=10,
        )
    )

    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["service_tier"] == "priority"
    assert events[-1].request_profile == {
        "reasoning_effort": "low",
        "service_tier": "priority",
    }


def test_openai_request_profile_rejects_unsupported_values(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "turbo")
    with pytest.raises(ValueError, match="OPENAI_REASONING_EFFORT"):
        llm_client.openai_request_profile_from_env()


def test_stream_failover_only_before_visible_output(monkeypatch):
    monkeypatch.delenv("EVOLUTION_PROVIDER_STRICT", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    calls: list[str] = []

    def fake_stream(config, **_kwargs):
        calls.append(config.name)
        if config.name == "anthropic":
            raise TimeoutError("before first token")
        yield LLMStreamEvent("text_delta", text="ok", provider="deepseek")
        yield LLMStreamEvent("done", provider="deepseek")

    monkeypatch.setattr(llm_client, "_stream_provider", fake_stream)
    events = list(LLMClient(primary="anthropic", fallback="deepseek", max_retries=0).generate_stream("hello"))
    assert calls == ["anthropic", "deepseek"]
    assert "".join(event.text for event in events if event.kind == "text_delta") == "ok"


def test_stream_does_not_replay_after_visible_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    calls: list[str] = []

    def interrupted(config, **_kwargs):
        calls.append(config.name)
        yield LLMStreamEvent("text_delta", text="partial", provider=config.name)
        raise ConnectionError("after first token")

    monkeypatch.setattr(llm_client, "_stream_provider", interrupted)
    stream = LLMClient(primary="anthropic", fallback="deepseek", max_retries=0).generate_stream("hello")
    assert next(stream).text == "partial"
    with pytest.raises(LLMError, match="after output started"):
        next(stream)
    assert calls == ["anthropic"]


def test_capability_fallback_is_three_sentences_without_task_dump():
    session = SimpleNamespace(llm_ready=False)
    events = list(ConversationAgent(client=None).stream_chat("你好，你是谁？请用三句话回答。", session, history=[]))
    answer = "".join(event.text for event in events if event.kind == "text_delta")
    assert answer.count("。") == 3
    assert "当前任务" not in answer
    assert "tools_executed" not in answer


def test_natural_language_artifact_question_routes_to_llm_chat():
    intent = kaggle_intent.classify("当前完成的实际结果储存在哪里")

    assert intent.kind == kaggle_intent.CHAT


def test_llm_agent_bridge_exposes_sanitized_tool_activity(monkeypatch, capsys):
    def fake_reply(self, text, session, *, history=None, context=None, on_tool_event=None):
        assert text == "当前完成的实际结果储存在哪里"
        assert history == []
        assert on_tool_event is not None
        on_tool_event("started", "verified_context", True)
        on_tool_event("completed", "verified_context", True)
        self._last_llm_execution = {
            "status": "completed",
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "input_tokens": 20,
            "output_tokens": 10,
            "native_tool_calls": 1,
            "tool_names": ["verified_context"],
        }
        return "真实 Run 的四项交付物、路径、下载链接、大小和 SHA-256"

    monkeypatch.setattr(ConversationAgent, "agent_reply", fake_reply)
    writer = assistant_stream.EventWriter("session_test")
    rc = assistant_stream._direct_chat(
        writer,
        "当前完成的实际结果储存在哪里",
        SimpleNamespace(selected_task=None),
        [],
    )
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert payloads[-1]["type"] == "answer_completed"
    assert "四项交付物" in payloads[-1]["answer"]
    assert payloads[-1]["route"] == "agent"
    assert "stop_reason" in payloads[-1]
    assert any(item["type"] == "tool_started" and item["tool"] == "verified_context" for item in payloads)
    assert any(item["type"] == "tool_completed" and item["tool"] == "verified_context" for item in payloads)
    assert any(
        item["type"] == "model"
        and item["model"] == "gpt-5.6-sol"
        and item["label"] == "gpt-5.6-sol via openai"
        for item in payloads
    )
    assert "hidden" not in json.dumps(payloads)


def test_direct_chat_uses_hpc_precheck_when_llm_returns_empty(monkeypatch, capsys):
    def fake_reply(self, text, session, *, history=None, context=None, on_tool_event=None):
        assert "作业号90673" in text
        self._last_llm_execution = {
            "status": "transport_error",
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "input_tokens": 0,
            "output_tokens": 0,
            "native_tool_calls": 0,
            "tool_names": [],
        }
        return ""

    def fake_execute(name, session):
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

    monkeypatch.setattr(ConversationAgent, "agent_reply", fake_reply)
    monkeypatch.setattr(kaggle_conversation, "_execute_terminal_tool", fake_execute)

    rc = assistant_stream._direct_chat(
        assistant_stream.EventWriter("hpc_session"),
        "我是小白，帮我检查现在能不能连接服务器，作业号90673；不要训练，也不要提交 Kaggle。",
        SimpleNamespace(selected_task="siim-isic-melanoma-classification"),
        [],
    )

    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert any(item["type"] == "tool_started" and item["tool"] == "hpc_connection_status" for item in payloads)
    assert any(item["type"] == "tool_completed" and item["tool"] == "hpc_connection_status" for item in payloads)
    final = payloads[-1]
    assert final["type"] == "answer_completed"
    assert final["llm_status"] == "completed_with_precheck_fallback"
    assert final["tool_calls_total"] == 1
    assert final["tool_names"] == ["hpc_connection_status"]
    assert "job90673" in final["answer"]
    assert "已实时进入目标容器" in final["answer"]
    assert "未启动训练" in final["answer"]
    assert "未提交 Kaggle" in final["answer"]


def test_workflow_bridge_uses_captured_stdout_when_summary_is_empty(monkeypatch, tmp_path, capsys):
    class FakeTerminalAgent:
        def __init__(self, *, colour=False):
            assert colour is False

        def handle(self, prompt, session, root):
            print("真实工具结果：workspace/report.pdf")
            return SimpleNamespace(summary="", blocked=False, rc=0)

    monkeypatch.setattr(assistant_stream, "TerminalAgent", FakeTerminalAgent)
    rc = assistant_stream._workflow_turn(
        assistant_stream.EventWriter("workflow_session"),
        "报告在哪里",
        SimpleNamespace(selected_task=None),
        tmp_path,
        assistant_stream.TOOL_QUERY,
    )
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rc == 0
    assert payloads[-1]["type"] == "answer_completed"
    assert payloads[-1]["answer"] == "真实工具结果：workspace/report.pdf"
    assert any(item.get("type") == "answer_delta" and "workspace/report.pdf" in item.get("delta", "") for item in payloads)


def test_siim_assistant_reuses_current_governed_run(tmp_path, capsys):
    from xsci.user_request import parse_user_request

    run_id = "evomind_siim_isic_hpc_fixture"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    (tmp_path / "workspace" / "current_run.json").write_text(json.dumps({
        "schema": "evomind.current_run.v1",
        "task_id": assistant_stream.SIIM_TASK_ID,
        "run_id": run_id,
        "run_dir": f"workspace/evomind_runs/{run_id}",
        "status": "needs_continuation",
    }), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "run_id": run_id,
        "status": "needs_continuation",
        "tasks": {
            "request_setup": {"status": "completed"},
            "hpc_data_preflight": {"status": "failed"},
        },
    }), encoding="utf-8")
    request = parse_user_request(
        "请分析 SIIM-ISIC 皮肤镜图像，检查患者泄漏，在 A800 上训练并交付报告，"
        "不要提交公开榜单。"
    )

    rc = assistant_stream._siim_research_turn(
        assistant_stream.EventWriter("siim_session"),
        request.objective,
        tmp_path,
        request,
    )
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert rc == 0
    assert payloads[-1]["type"] == "answer_completed"
    assert payloads[-1]["run_id"] == run_id
    assert payloads[-1]["reused"] is True
    assert any(item["type"] == "research_run" and item["run_id"] == run_id for item in payloads)
    assert list((tmp_path / "workspace" / "evomind_runs").iterdir()) == [run_dir]


def test_direct_chat_fallback_uses_real_project_and_run_context(tmp_path):
    from xsci.assistant_context import build_assistant_context

    run_id = "context_run"
    run_dir = tmp_path / "workspace" / "evomind_runs" / run_id
    run_dir.mkdir(parents=True)
    (tmp_path / "workspace" / "current_run.json").write_text(json.dumps({
        "schema": "evomind.current_run.v1",
        "task_id": "evomind-qwen7b-finetune",
        "run_id": run_id,
        "run_dir": f"workspace/evomind_runs/{run_id}",
        "status": "completed",
    }), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "status": "completed", "objective": "domain refinement", "tasks": {}, "gates": {},
    }), encoding="utf-8")
    (run_dir / "request.json").write_text(json.dumps({
        "task_type": "llm_finetune", "base_model": "Qwen/Qwen2.5-7B-Instruct",
    }), encoding="utf-8")
    packet = build_assistant_context(tmp_path)

    architecture = ConversationAgent._direct_chat_fallback("解释当前项目的核心架构", context=packet)
    status = ConversationAgent._direct_chat_fallback("上次模型微调完成到哪了", context=packet)
    assert "Executive Supervisor" in architecture
    assert "Qwen/Qwen2.5-7B-Instruct" in status
    assert "状态为 completed" in status


def test_interactive_terminal_chat_streams_visible_text_only(monkeypatch, capsys):
    class FakeConversation:
        def stream_chat(self, text, session):
            yield LLMStreamEvent("start", provider="test", model="model")
            yield LLMStreamEvent("thinking_status", text="hidden chain", provider="test", model="model")
            yield LLMStreamEvent("text_delta", text="逐", provider="test", model="model")
            yield LLMStreamEvent("text_delta", text="字回答", provider="test", model="model")
            yield LLMStreamEvent("done", provider="test", model="model")

    monkeypatch.setattr(kaggle_cli, "_conversation", lambda: FakeConversation())
    session = SimpleNamespace(last_action="", selected_task=None)
    rc, should_exit = kaggle_cli._dispatch_intent("你好", SimpleNamespace(), session)
    output = capsys.readouterr().out

    assert rc == 0
    assert should_exit is False
    assert "逐字回答" in output
    assert "正在理解问题" in output
    assert "hidden chain" not in output
    assert session.last_action == "greeting"


def test_literature_search_forwards_scoped_browser_auth_to_loopback_only(monkeypatch, tmp_path):
    from xsci import terminal_tools

    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "ok": True,
                "papers": [],
                "source_counts": {},
                "source_errors": [],
                "integrity": {
                    "external_verified": 0,
                    "imported": 0,
                    "internal_context": 0,
                    "fabricated": 0,
                },
            }).encode("utf-8")

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("EVOMIND_LITERATURE_API_URL", "http://127.0.0.1:8088/api/literature/search")
    monkeypatch.setenv("EVOMIND_INTERNAL_SESSION_COOKIE", "evomind_local_session=session-value")
    monkeypatch.setenv("EVOMIND_INTERNAL_CSRF", "csrf-value")
    monkeypatch.setenv("EVOMIND_INTERNAL_ORIGIN", "http://127.0.0.1:8088")
    monkeypatch.setattr(terminal_tools.urllib.request, "urlopen", fake_urlopen)

    result = terminal_tools.search_literature(
        SimpleNamespace(selected_task="siim-isic-melanoma-classification", task_brief="melanoma"),
        tmp_path,
        query="patient-grouped cross validation",
    )

    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["cookie"] == "evomind_local_session=session-value"
    assert headers["x-evomind-csrf"] == "csrf-value"
    assert headers["origin"] == "http://127.0.0.1:8088"
    assert result["ok"] is True
    assert "session-value" not in json.dumps(result)
    assert "EVOMIND_INTERNAL_SESSION_COOKIE" not in terminal_tools.os.environ
    assert "EVOMIND_INTERNAL_CSRF" not in terminal_tools.os.environ
    assert "EVOMIND_INTERNAL_ORIGIN" not in terminal_tools.os.environ


def test_literature_search_never_forwards_local_auth_to_external_endpoint(monkeypatch, tmp_path):
    from xsci import terminal_tools

    captured: dict[str, str] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok":true,"papers":[],"source_errors":[],"integrity":{}}'

    def fake_urlopen(request, *, timeout):
        del timeout
        captured.update({key.lower(): value for key, value in request.header_items()})
        return FakeResponse()

    monkeypatch.setenv("EVOMIND_LITERATURE_API_URL", "https://example.test/api/literature/search")
    monkeypatch.setenv("EVOMIND_INTERNAL_SESSION_COOKIE", "evomind_local_session=session-value")
    monkeypatch.setenv("EVOMIND_INTERNAL_CSRF", "csrf-value")
    monkeypatch.setenv("EVOMIND_INTERNAL_ORIGIN", "http://127.0.0.1:8088")
    monkeypatch.setattr(terminal_tools.urllib.request, "urlopen", fake_urlopen)

    terminal_tools.search_literature(
        SimpleNamespace(selected_task="siim-isic-melanoma-classification", task_brief="melanoma"),
        tmp_path,
        query="patient-grouped cross validation",
    )

    assert "cookie" not in captured
    assert "x-evomind-csrf" not in captured
    assert "origin" not in captured


def test_assistant_stream_reads_utf8_stdin_bytes_without_surrogates(monkeypatch):
    payload = {"prompt": "我是小白，帮我解释上次实验结果", "session_id": "utf8_session"}

    class FakeBuffer:
        def read(self):
            return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(assistant_stream.sys, "stdin", SimpleNamespace(buffer=FakeBuffer()))

    parsed = assistant_stream._read_payload_from_stdin()

    assert parsed == payload
    assert r"\udc" not in repr(parsed)


def test_web_selected_task_clears_stale_terminal_task_metadata(tmp_path):
    from xsci.kaggle_session import SessionState

    session = SessionState(
        workspace_root=str(tmp_path),
        selected_task="house-prices",
        task_brief="name=house_prices | target=SalePrice | metric=RMSE",
        recent_run_id="house_run",
    )

    assistant_stream._bind_requested_task(
        session,
        "siim-isic-melanoma-classification",
        tmp_path,
    )

    assert session.selected_task == "siim-isic-melanoma-classification"
    assert session.task_brief == ""
    assert session.recent_run_id == ""


def test_siim_planning_and_no_training_never_enter_execution_template():
    read_only = assistant_stream.classify(
        "我是小白，先不要训练，请解释 SIIM 训练集和验证集为什么按患者分组，并给下一步计划。"
    )

    assert read_only.kind != assistant_stream.EXECUTION
    assert assistant_stream._should_start_siim_execution(read_only) is False


def test_explicit_siim_a800_training_enters_governed_execution():
    execution = assistant_stream.classify(
        "请在 A800 上训练 SIIM-ISIC 图像分类模型并生成报告，不提交 Kaggle。"
    )

    assert execution.kind == assistant_stream.EXECUTION
    assert assistant_stream._should_start_siim_execution(execution) is True

