from __future__ import annotations

import hashlib
import sqlite3
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomind_runtime.runtime import AgentRuntime
from evomind_runtime.assistant_runs import (
    _artifact_requirements_from_prompt,
    _failure_policy,
    _managed_hpc_identity_fingerprint,
)
from evomind_runtime.competition_goal import FIXED_RUN_ID
from evomind_runtime.models import ApprovalRequest, ToolCall, ToolResult, utc_now
from evomind_runtime.tools import _required_managed_hpc_identity, build_default_registry


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_failure_taxonomy_distinguishes_transient_recoverable_and_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOMIND_ASSISTANT_AUTO_RETRY", raising=False)
    assert _failure_policy(RuntimeError("gateway returned 429")) == ("transient", False)
    assert _failure_policy(ConnectionError("model stream disconnected")) == ("transient", False)
    assert _failure_policy(OSError("No space left on device")) == ("recoverable", False)
    assert _failure_policy(ValueError("artifact integrity verification failed")) == ("terminal", False)
    monkeypatch.setenv("EVOMIND_ASSISTANT_AUTO_RETRY", "1")
    assert _failure_policy(RuntimeError("gateway returned 429")) == ("transient", True)


def test_controlled_secret_consumer_failure_is_blocked_without_automatic_replay() -> None:
    for message in (
        "controlled_secret_required",
        "consumer_process_exit:Stage_envelope_json",
        "consumer_host_exit:Unknown",
        "consumer_invalid_base64",
    ):
        assert _failure_policy(RuntimeError(message)) == ("controlled_secret_required", False)


def test_controlled_secret_failure_does_not_replay_the_model_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AgentRuntime(tmp_path)
    calls = 0

    def failed_consumer(_session_id: str, _content: str, *, max_steps: int = 24) -> dict:
        nonlocal calls
        del max_steps
        calls += 1
        return {"status": "paused", "text": "consumer_process_exit:Stage_envelope_json", "model": {"provider": "fixture", "model": "route"}}

    monkeypatch.setattr("evomind_runtime.assistant_runs.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime, "message", failed_consumer)
    run = runtime.assistant.create_run(prompt="下载 Weather4cast 官方数据", conversation_id="secret_failure_no_replay")
    blocked = runtime.assistant.wait(run["id"], timeout=5)

    assert blocked["status"] == "blocked"
    assert blocked["error_class"] == "controlled_secret_required"
    assert blocked["retry_count"] == 1
    assert calls == 1
    assert not any(item["event_type"] == "retry_scheduled" for item in runtime.assistant.events(run["id"]))
    runtime.close()


def test_explicit_final_artifact_requirements_exclude_input_filenames() -> None:
    requirements = _artifact_requirements_from_prompt(
        "先下载 train.csv、test.csv。最终产物至少包括 metrics.json、fold_metrics.csv、"
        "training.log、complete-results.zip、最佳模型及加载环境、Experience Board/Cards，"
        "以及专业 Markdown/PDF 报告。"
    )

    assert requirements["names"] == [
        "metrics.json",
        "fold_metrics.csv",
        "training.log",
        "complete-results.zip",
    ]
    assert set(requirements["groups"]) == {
        "model",
        "environment",
        "experience_board",
        "complete_bundle",
        "markdown_report",
        "pdf_report",
    }


def test_publish_clause_extracts_named_outputs_and_structured_evidence_groups() -> None:
    requirements = _artifact_requirements_from_prompt(
        "先读取 train.csv 和 test.csv。完成后发布 metrics.json、submission.csv、training.log、"
        "模型、search_graph、retrospective_memory、Experience Board、validation_contract、"
        "claim_audit、完整 ZIP 和专业 Markdown/PDF 报告及哈希清单。"
    )

    assert requirements["names"] == ["metrics.json", "submission.csv", "training.log"]
    assert set(requirements["groups"]) == {
        "model",
        "experience_board",
        "search_graph",
        "retrospective_memory",
        "validation_contract",
        "claim_audit",
        "manifest",
        "complete_bundle",
        "markdown_report",
        "pdf_report",
    }


def test_strategy_or_model_artifact_is_an_alternative_group_not_a_mandatory_model() -> None:
    requirements = _artifact_requirements_from_prompt(
        "至少发布 data-audit.json、strategy.json 或 model artifact、run-manifest.json；\n"
        "若环境不可执行，不生成模型；不得伪造训练或模型。"
    )

    assert requirements["names"] == ["data-audit.json", "run-manifest.json"]
    assert requirements["groups"] == ["model_or_strategy", "manifest"]


def test_strategy_artifact_satisfies_model_or_strategy_group(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终交付文件：strategy.json 或 model artifact。",
        conversation_id="strategy_or_model_artifact_group",
        start=False,
    )
    strategy = Path(run["task_root"]) / "outputs" / "strategy.json"
    strategy.write_text('{"kind":"deterministic_strategy"}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], strategy, source_tool_call="fixture")

    assert runtime.assistant._missing_expected_artifacts(run) == []
    runtime.close()


def test_negative_model_wording_does_not_create_a_model_artifact_requirement() -> None:
    requirements = _artifact_requirements_from_prompt(
        "最终交付文件：metrics.json。\n不得伪造训练、胜率或模型。"
    )

    assert requirements == {"names": ["metrics.json"], "groups": []}


def test_named_frozen_model_or_strategy_is_an_alternative_group() -> None:
    requirements = _artifact_requirements_from_prompt(
        "最终交付文件：metrics.json、frozen-model.joblib 或 strategy.json、run-manifest.json。"
    )

    assert requirements["names"] == ["metrics.json", "run-manifest.json"]
    assert requirements["groups"] == ["model_or_strategy", "manifest"]


def test_missing_explicit_final_files_prevent_false_completion(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终产物至少包括 metrics.json、submission.csv、complete-results.zip。",
        conversation_id="explicit_artifact_gate",
        start=False,
    )
    metrics = Path(run["task_root"]) / "outputs" / "metrics.json"
    metrics.write_text('{"score": 0.8}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], metrics, source_tool_call="fixture")

    with pytest.raises(RuntimeError, match="submission.csv, complete-results.zip"):
        runtime.assistant._finalize_outputs(run["id"], "partial result")

    runtime.store.update_assistant_run(run["id"], status="completed", completed_at="2026-08-18T00:00:00Z")
    assert runtime.assistant._recoverable_false_completion(run["id"]) is True
    snapshot = runtime.assistant.snapshot(run["id"])
    assert snapshot["status"] == "blocked"
    assert snapshot["terminal"] is True
    assert snapshot["completed_at"] == ""
    assert snapshot["completion_repair_required"] is True
    assert snapshot["error_class"] == "incomplete_deliverables"
    assert snapshot["missing_expected_artifacts"] == ["submission.csv", "complete-results.zip"]
    assert snapshot["unresolved_tool_names"] == []
    assert "submission.csv, complete-results.zip" in snapshot["error_message"]
    persisted = runtime.store.get_assistant_run(run["id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    resume_prompt = runtime.assistant._execution_prompt(persisted, resume=True)
    assert "submission.csv, complete-results.zip" in resume_prompt
    runtime.close()


def test_execution_prompt_reuses_managed_kaggle_connector_and_classifies_six_source_status_mechanically(
    tmp_path: Path,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="把六个比赛数据下载到 HPC 并验收。",
        conversation_id="managed_kaggle_six_source_contract",
        start=False,
    )
    prompt = runtime.assistant._execution_prompt(run, resume=False)

    assert "already-configured server-managed Kaggle connector" in prompt
    assert "Weather4cast SFTP still requires its tenant-bound one-time controlled secret" in prompt
    assert "PARTIAL for 1-5/6 ready" in prompt
    assert "never authorizes Join, rule acceptance, account creation, or submission" in prompt
    assert "Workbench > Data / Kaggle > AIMSLAB 60-second quick replacement" in prompt
    assert "Do not claim the enrollment surface is unavailable" in prompt
    runtime.close()


def test_completed_snapshot_projects_latest_final_files_and_retains_failure_evidence(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="生成训练结果文件", conversation_id="artifact_projection", start=False)
    outputs = Path(run["task_root"]) / "outputs"
    attempt1 = outputs / "attempt1" / "training.log"
    attempt2 = outputs / "attempt2" / "training.log"
    blocker = outputs / "attempt1" / "execution-blocker-report.md"
    metrics = outputs / "final" / "metrics.json"
    for path, content in (
        (attempt1, "failed attempt\n"),
        (blocker, "# blocker\n"),
        (attempt2, "successful training\n"),
        (metrics, '{"score": 0.8}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        runtime.assistant.publish_path(
            run["id"],
            path,
            name=path.name,
            source_tool_call="artifact_publish",
        )
    runtime.store.update_assistant_run(run["id"], status="completed", completed_at="2026-08-18T00:00:00Z")

    snapshot = runtime.assistant.snapshot(run["id"])
    assert [item["name"] for item in snapshot["artifacts"]] == ["training.log", "metrics.json"]
    assert snapshot["artifacts"][0]["sha256"] == _digest(attempt2.read_bytes())
    assert snapshot["artifact_evidence_count"] == 4
    assert {item["name"] for item in snapshot["artifact_evidence"]} == {
        "training.log",
        "execution-blocker-report.md",
    }
    runtime.close()


def _upload(runtime: AgentRuntime, name: str, data: bytes, media_type: str = "") -> dict:
    upload = runtime.assistant.create_upload(
        name=name,
        total_bytes=len(data),
        media_type=media_type,
        sha256=_digest(data),
    )
    if data:
        runtime.assistant.put_chunk(upload["id"], 0, data, _digest(data))
    return runtime.assistant.complete_upload(upload["id"])


def _completed_message(runtime: AgentRuntime, session_id: str, _content: str, *, max_steps: int = 24) -> dict:
    del max_steps
    workspace = Path(runtime.get_session(session_id)["workspace_root"])
    output = workspace / "outputs" / "result.md"
    output.write_text("# Verified result\n\nassistant-run-ok\n", encoding="utf-8")
    runtime.store.add_turn(session_id, "assistant", "Completed with a verified artifact.")
    return {
        "status": "completed",
        "text": "Completed with a verified artifact.",
        "model_execution": {"provider": "fixture", "model": "served-model-v1", "native_tool_calls": 2},
    }


def test_chunk_upload_is_hashed_idempotent_and_name_bounded(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    data = b"name,value\nalpha,1\n"
    upload = runtime.assistant.create_upload(
        name="../../unsafe.csv",
        total_bytes=len(data),
        media_type="text/csv",
        sha256=_digest(data),
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        runtime.assistant.put_chunk(upload["id"], 0, data, "0" * 64)

    first = runtime.assistant.put_chunk(upload["id"], 0, data, _digest(data))
    replay = runtime.assistant.put_chunk(upload["id"], 0, data, _digest(data))
    assert first["replayed"] is False
    assert replay["replayed"] is True

    completed = runtime.assistant.complete_upload(upload["id"])
    replayed_complete = runtime.assistant.complete_upload(upload["id"])
    attachment = completed["attachment"]
    assert attachment["name"] == "unsafe.csv"
    assert attachment["bytes"] == len(data)
    assert attachment["sha256"] == _digest(data)
    assert Path(attachment["path"]).read_bytes() == data
    assert replayed_complete["replayed"] is True
    assert replayed_complete["attachment"]["id"] == attachment["id"]
    runtime.close()


def test_upload_rejects_conflicts_and_sniffs_mime_from_bytes(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    data = b"plain text, not a png\n"
    upload = runtime.assistant.create_upload(name="misleading.png", total_bytes=len(data), media_type="image/png")
    runtime.assistant.put_chunk(upload["id"], 0, data, _digest(data))
    with pytest.raises(ValueError, match="idempotency conflict"):
        changed = b"x" * len(data)
        runtime.assistant.put_chunk(upload["id"], 0, changed, _digest(changed))
    completed = runtime.assistant.complete_upload(upload["id"])
    assert completed["attachment"]["media_type"] == "text/plain"
    runtime.close()


def test_run_binds_attachment_to_isolated_workspace_and_publishes_verified_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    monkeypatch.setattr(runtime, "message", lambda session_id, content, max_steps=24: _completed_message(runtime, session_id, content, max_steps=max_steps))
    uploaded = _upload(runtime, "数据.csv", b"x,y\n1,2\n", "text/csv")

    run = runtime.assistant.create_run(
        prompt="分析附件并生成报告",
        conversation_id="conversation_fixture",
        attachment_ids=[uploaded["attachment"]["id"]],
    )
    completed = runtime.assistant.wait(run["id"], timeout=5)

    assert completed["status"] == "completed"
    assert completed["model_provider"] == "fixture"
    assert completed["model"] == "served-model-v1"
    assert len(completed["attachments"]) == 1
    assert len(completed["artifacts"]) == 1
    artifact = completed["artifacts"][0]
    artifact_path = Path(artifact["path"])
    assert artifact_path.is_relative_to(Path(completed["task_root"]) / "outputs")
    assert artifact["sha256"] == _digest(artifact_path.read_bytes())
    assert runtime.assistant.artifact(artifact["id"])[1].read_bytes() == artifact_path.read_bytes()
    events = runtime.assistant.events(run["id"])
    event_types = [item["event_type"] for item in events]
    assert "model_observed" in event_types
    assert "verification_completed" in event_types
    assert "artifact_published" in event_types
    assert event_types[-1] == "run_completed"
    completed_event = events[-1]
    assert completed_event["payload"]["answer"].startswith(
        "**本轮实际模型**：`fixture / served-model-v1`（来自受管 model_observed 事件）"
    )
    runtime.close()


def test_followup_run_requires_explicit_attachment_reuse_in_same_conversation(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    data = b"feature,target\n1,0\n2,1\n"
    uploaded = _upload(runtime, "training.csv", data, "text/csv")
    first = runtime.assistant.create_run(
        prompt="分析这个训练数据",
        conversation_id="attachment_context_fixture",
        attachment_ids=[uploaded["attachment"]["id"]],
        start=False,
    )

    followup = runtime.assistant.create_run(
        prompt="继续使用刚才的数据训练",
        conversation_id="attachment_context_fixture",
        attachment_ids=[],
        start=False,
    )

    assert len(first["attachments"]) == 1
    assert followup["attachments"] == []
    assert followup["attachment_context"] == {"mode": "none", "source_run_id": ""}
    runtime.close()


def test_attachment_context_does_not_cross_conversations_or_override_explicit_uploads(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    old_data = b"x,y\n1,0\n"
    new_data = b"x,y\n2,1\n"
    old_upload = _upload(runtime, "old.csv", old_data, "text/csv")
    new_upload = _upload(runtime, "new.csv", new_data, "text/csv")
    runtime.assistant.create_run(
        prompt="读取旧文件",
        conversation_id="attachment_scope_fixture",
        attachment_ids=[old_upload["attachment"]["id"]],
        start=False,
    )

    other_conversation = runtime.assistant.create_run(
        prompt="新对话不能看到旧文件",
        conversation_id="attachment_scope_other",
        attachment_ids=[],
        start=False,
    )
    explicit_replacement = runtime.assistant.create_run(
        prompt="改用新文件",
        conversation_id="attachment_scope_fixture",
        attachment_ids=[new_upload["attachment"]["id"]],
        start=False,
    )

    assert other_conversation["attachments"] == []
    assert other_conversation["attachment_context"]["mode"] == "none"
    assert [item["name"] for item in explicit_replacement["attachments"]] == ["new.csv"]
    assert explicit_replacement["attachment_context"]["mode"] == "explicit"
    runtime.close()


def test_repair_attachments_action_restores_a_legacy_empty_run_without_starting_it(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    data = b"feature,target\n1,0\n"
    uploaded = _upload(runtime, "training.csv", data, "text/csv")
    source = runtime.assistant.create_run(
        prompt="读取训练数据",
        conversation_id="attachment_repair_fixture",
        attachment_ids=[uploaded["attachment"]["id"]],
        start=False,
    )
    legacy = runtime.assistant.create_run(
        prompt="继续训练",
        conversation_id="attachment_repair_fixture",
        attachment_ids=[],
        start=False,
    )
    for item in legacy["attachments"]:
        Path(item["path"]).unlink()
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute("DELETE FROM attachments WHERE run_id=?", (legacy["id"],))
    runtime.store.update_assistant_run(legacy["id"], attachment_ids=[])

    repaired = runtime.assistant.action(legacy["id"], "repair_attachments")

    assert repaired["status"] == "queued"
    assert len(repaired["attachments"]) == 1
    assert repaired["attachments"][0]["name"] == "training.csv"
    assert Path(repaired["attachments"][0]["path"]).read_bytes() == data
    assert repaired["attachment_context"] == {"mode": "repaired", "source_run_id": source["id"]}
    assert any(item["event_type"] == "attachments_inherited" for item in runtime.assistant.events(legacy["id"]))
    runtime.close()


def test_artifact_publish_snapshots_outputs_and_detects_snapshot_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="生成文件", conversation_id="artifact_boundary", start=False)
    task_root = Path(run["task_root"])
    outside = task_root / "work" / "draft.txt"
    outside.write_text("draft", encoding="utf-8")
    rejected = runtime.invoke_tool(run["id"], "artifact_publish", {"path": str(outside)})
    assert rejected["status"] == "failed"
    assert rejected["result"]["error"] == "artifact_outside_outputs"

    output = task_root / "outputs" / "final.txt"
    output.write_text("final", encoding="utf-8")
    published = runtime.invoke_tool(run["id"], "artifact_publish", {"path": str(output)})
    assert published["status"] == "completed"
    artifact_id = published["result"]["content"]["artifact"]["id"]
    output.write_text("tampered", encoding="utf-8")
    _, published_path = runtime.assistant.artifact(artifact_id)
    assert published_path.read_text(encoding="utf-8") == "final"
    assert ".a" in published_path.parts
    published_path.write_text("tampered snapshot", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        runtime.assistant.artifact(artifact_id)
    runtime.close()


def test_finalize_prunes_legacy_superseded_same_path_rows(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="生成文件", conversation_id="legacy_artifact_retry", start=False)
    output = Path(run["task_root"]) / "outputs" / "training.log"
    output.write_text("first failure", encoding="utf-8")
    first = runtime.assistant.publish_path(run["id"], output, source_tool_call="fixture")
    first_snapshot = Path(first["path"])

    # Recreate two legacy mutable-path rows and leave the live path at the
    # newest bytes, matching retry behavior before immutable snapshots.
    runtime.store.remove_deliverables_by_ids(run["id"], [first["id"]])
    first_snapshot.unlink()
    first_snapshot.parent.rmdir()
    output.write_text("first failure", encoding="utf-8")
    first_payload = {
        **first,
        "path": str(output),
        "id": "artifact_legacy_first",
        "verified_at": "2026-01-01T00:00:00Z",
    }
    runtime.store.publish_deliverable(first_payload)
    output.write_bytes(b"")
    second_payload = {
        **first_payload,
        "id": "artifact_legacy_second",
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "verified_at": "2026-01-01T00:00:01Z",
    }
    runtime.store.publish_deliverable(second_payload)

    finalized = runtime.assistant._finalize_outputs(run["id"], "done")
    assert [item["id"] for item in finalized] == ["artifact_legacy_second"]
    assert runtime.store.get_deliverable("artifact_legacy_first") is None
    runtime.close()


def test_blocked_run_retry_resets_budget_and_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AgentRuntime(tmp_path)
    monkeypatch.setenv("EVOMIND_ASSISTANT_AUTO_RETRY", "1")
    attempts = 0

    def flaky(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            return {"status": "paused", "text": "temporary gateway outage", "model": {"provider": "fixture", "model": "route"}}
        return _completed_message(runtime, session_id, content, max_steps=max_steps)

    monkeypatch.setattr("evomind_runtime.assistant_runs.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime, "message", flaky)
    run = runtime.assistant.create_run(prompt="生成恢复测试文件", conversation_id="retry_fixture")
    blocked = runtime.assistant.wait(run["id"], timeout=5)
    assert blocked["status"] == "blocked"
    assert blocked["retry_count"] == 3

    resumed = runtime.assistant.action(run["id"], "retry")
    assert resumed["retry_count"] == 0
    completed = runtime.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    assert attempts == 4
    runtime.close()


def test_terminal_verification_failure_does_not_retry_or_fake_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    calls = 0

    def completed_without_valid_artifact(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "completed", "text": "claim", "model_execution": {"provider": "fixture", "model": "served", "native_tool_calls": 0}}

    monkeypatch.setattr(runtime, "message", completed_without_valid_artifact)
    monkeypatch.setattr(runtime.assistant, "_finalize_outputs", lambda _run_id, _answer: (_ for _ in ()).throw(ValueError("artifact integrity verification failed")))
    run = runtime.assistant.create_run(prompt="生成必须校验的报告", conversation_id="terminal_fixture")
    failed = runtime.assistant.wait(run["id"], timeout=5)

    assert failed["status"] == "failed"
    assert failed["error_class"] == "terminal"
    assert failed["retry_count"] == 1
    assert calls == 1
    events = runtime.assistant.events(run["id"])
    assert events[-1]["event_type"] == "run_failed"
    assert events[-1]["payload"]["recoverable"] is False
    assert not any(item["event_type"] == "run_completed" for item in events)
    runtime.close()


def test_failed_tools_cannot_complete_with_only_verified_answer_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)

    def completed_after_failed_tool(session_id: str, _content: str, *, max_steps: int = 24) -> dict:
        del max_steps
        failed = runtime.invoke_tool(session_id, "file_read", {"path": "inputs/missing.csv"})
        assert failed["status"] == "failed"
        return {
            "status": "completed",
            "text": "The required input could not be processed.",
            "model_execution": {"provider": "fixture", "model": "served", "native_tool_calls": 1},
        }

    monkeypatch.setattr("evomind_runtime.assistant_runs.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime, "message", completed_after_failed_tool)
    run = runtime.assistant.create_run(prompt="生成必须包含真实结果的报告", conversation_id="failed_tool_fallback")
    blocked = runtime.assistant.wait(run["id"], timeout=5)

    assert blocked["status"] == "blocked"
    assert blocked["retry_count"] == 1
    assert blocked["artifacts"] == []
    events = runtime.assistant.events(run["id"])
    assert events[-1]["event_type"] == "run_blocked"
    assert not any(item["event_type"] == "run_completed" for item in events)
    runtime.close()


def test_failed_tools_cannot_complete_with_published_blocker_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    published = False

    def completed_with_blocker_report(session_id: str, _content: str, *, max_steps: int = 24) -> dict:
        nonlocal published
        del max_steps
        failed = runtime.invoke_tool(session_id, "file_read", {"path": "inputs/missing.csv"})
        assert failed["status"] == "failed"
        if not published:
            root = Path(runtime.get_session(session_id)["workspace_root"])
            report = root / "outputs" / "execution-blocker-report.md"
            report.write_text("# Execution blocker\n\nRequired result was not produced.\n", encoding="utf-8")
            runtime.assistant.publish_path(session_id, report, source_tool_call="artifact_publish")
            published = True
        return {
            "status": "completed",
            "text": "The required input could not be processed.",
            "model_execution": {"provider": "fixture", "model": "served", "native_tool_calls": 2},
        }

    monkeypatch.setattr("evomind_runtime.assistant_runs.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime, "message", completed_with_blocker_report)
    run = runtime.assistant.create_run(prompt="训练并交付真实模型", conversation_id="failed_tool_blocker_report")
    blocked = runtime.assistant.wait(run["id"], timeout=5)

    assert blocked["status"] == "blocked"
    assert blocked["retry_count"] == 1
    assert [item["name"] for item in blocked["artifacts"]] == ["execution-blocker-report.md"]
    events = runtime.assistant.events(run["id"])
    assert events[-1]["event_type"] == "run_blocked"
    assert not any(item["event_type"] == "run_completed" for item in events)
    runtime.close()


def test_successful_kaggle_download_supersedes_optional_list_failure(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="下载并训练 Kaggle 数据", conversation_id="kaggle_supersession", start=False)
    runtime.store.put_tool_call(
        ToolCall(
            id="call_kaggle_list_failed",
            session_id=run["id"],
            tool_name="kaggle_list",
            arguments={"query": "fixture"},
            status="failed",
            created_at="2026-08-18T00:00:00Z",
            started_at="2026-08-18T00:00:00Z",
            completed_at="2026-08-18T00:00:01Z",
        ),
        ToolResult("call_kaggle_list_failed", False, {}, "listing failed", error="transient"),
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_kaggle_download_completed",
            session_id=run["id"],
            tool_name="kaggle_download",
            arguments={"competition": "fixture"},
            status="completed",
            created_at="2026-08-18T00:00:02Z",
            started_at="2026-08-18T00:00:02Z",
            completed_at="2026-08-18T00:00:03Z",
        ),
        ToolResult("call_kaggle_download_completed", True, {"status": "completed"}, "downloaded"),
    )

    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    runtime.close()


@pytest.mark.parametrize("error", ["permission_denied", "profile_job_binding_invalid", "DATA_UNAVAILABLE"])
def test_settled_competition_status_error_is_evidence_not_an_unresolved_call(
    tmp_path: Path, error: str
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="检查六个比赛数据状态并发布 data-source-manifest.json。",
        conversation_id="settled_competition_status",
        start=False,
    )
    runtime.store.put_tool_call(
        ToolCall(
            id=f"call_competition_status_{error}",
            session_id=run["id"],
            tool_name="competition_data_status",
            arguments={"competition": "open_polymer"},
            status="failed",
            created_at="2026-08-24T00:00:00Z",
            started_at="2026-08-24T00:00:00Z",
            completed_at="2026-08-24T00:00:01Z",
        ),
        ToolResult(
            f"call_competition_status_{error}",
            False,
            {"competition": "open_polymer", "status": "DATA_UNAVAILABLE"},
            "competition status returned a structured negative observation",
            error=error,
        ),
    )

    assert runtime.assistant._effective_active_tool_calls(run["id"]) == []
    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    stored = runtime.store.list_tool_calls(run["id"])[0]
    assert stored["status"] == "failed"
    assert stored["result"]["error"] == error
    runtime.close()


def test_settled_competition_prepare_error_is_evidence_not_an_unresolved_call(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="准备六个比赛数据并发布 data-source-manifest.json。",
        conversation_id="settled_competition_prepare",
        start=False,
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_competition_prepare_value_error",
            session_id=run["id"],
            tool_name="competition_data_prepare",
            arguments={"competition": "mindgames"},
            status="failed",
            created_at="2026-08-24T00:00:00Z",
            started_at="2026-08-24T00:00:00Z",
            completed_at="2026-08-24T00:00:01Z",
        ),
        ToolResult(
            "call_competition_prepare_value_error",
            False,
            {"competition": "mindgames", "status": "DATA_UNAVAILABLE"},
            "competition prepare returned a settled structured error",
            error="ValueError",
        ),
    )

    assert runtime.assistant._effective_active_tool_calls(run["id"]) == []
    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    stored = runtime.store.list_tool_calls(run["id"])[0]
    assert stored["status"] == "failed"
    assert stored["result"]["error"] == "ValueError"
    runtime.close()


def test_missing_result_artifact_failure_requires_changed_precondition_not_blind_retry() -> None:
    failure = RuntimeError(
        "required result artifacts unavailable; unresolved tool calls: kaggle_download"
    )

    assert _failure_policy(failure) == ("recoverable", False)


def test_later_settled_success_supersedes_overlapping_duplicate_failure(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="训练并交付真实模型",
        conversation_id="overlapping_tool_completion",
        start=False,
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_primary_training",
            session_id=run["id"],
            tool_name="hpc_execute_solution",
            arguments={"solution_id": "primary"},
            status="completed",
            created_at="2026-08-18T00:00:00Z",
            started_at="2026-08-18T00:00:00Z",
            completed_at="2026-08-18T00:10:00Z",
        ),
        ToolResult("call_primary_training", True, {"status": "completed"}, "training completed"),
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_duplicate_training",
            session_id=run["id"],
            tool_name="hpc_execute_solution",
            arguments={"solution_id": "duplicate"},
            status="failed",
            created_at="2026-08-18T00:05:00Z",
            started_at="2026-08-18T00:05:00Z",
            completed_at="2026-08-18T00:09:00Z",
        ),
        ToolResult("call_duplicate_training", False, {}, "duplicate failed", error="transient"),
    )

    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    runtime.close()


def test_active_duplicate_remains_blocking_after_completed_call(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="训练并交付真实模型",
        conversation_id="active_duplicate_tool",
        start=False,
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_completed_training",
            session_id=run["id"],
            tool_name="hpc_execute_solution",
            arguments={"solution_id": "completed"},
            status="completed",
            created_at="2026-08-18T00:00:00Z",
            started_at="2026-08-18T00:00:00Z",
            completed_at="2026-08-18T00:10:00Z",
        ),
        ToolResult("call_completed_training", True, {"status": "completed"}, "training completed"),
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_active_duplicate",
            session_id=run["id"],
            tool_name="hpc_execute_solution",
            arguments={"solution_id": "active"},
            status="running",
            created_at="2026-08-18T00:11:00Z",
            started_at="2026-08-18T00:11:00Z",
        )
    )

    blockers = runtime.assistant._latest_tool_blockers(run["id"])
    assert [item["id"] for item in blockers] == ["call_active_duplicate"]
    runtime.close()


def test_success_backed_supersession_ignores_orphaned_running_rows(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="下载并训练 Kaggle 数据",
        conversation_id="orphaned_running_rows",
        start=False,
    )
    for call_id, tool_name in (
        ("call_orphaned_download", "kaggle_download"),
        ("call_orphaned_list", "kaggle_list"),
    ):
        runtime.store.put_tool_call(
            ToolCall(
                id=call_id,
                session_id=run["id"],
                tool_name=tool_name,
                arguments={"competition": "fixture"},
                status="running",
                created_at="2026-08-18T00:00:00Z",
                started_at="2026-08-18T00:00:00Z",
            )
        )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_completed_download_after_restart",
            session_id=run["id"],
            tool_name="kaggle_download",
            arguments={"competition": "fixture"},
            status="completed",
            created_at="2026-08-18T00:01:00Z",
            started_at="2026-08-18T00:01:00Z",
            completed_at="2026-08-18T00:02:00Z",
        ),
        ToolResult("call_completed_download_after_restart", True, {"status": "completed"}, "downloaded"),
    )

    assert runtime.assistant._effective_active_tool_calls(run["id"]) == []
    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    runtime.store.update_assistant_run(run["id"], status="blocked", error_class="llm_error", error_message="fixture")
    snapshot = runtime.assistant.snapshot(run["id"])
    assert snapshot["status"] == "blocked"
    assert snapshot["active_tool_calls"] == []
    runtime.close()


def test_later_assistant_turn_supersedes_orphaned_running_tool_call(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="检索并发布已验证结果",
        conversation_id="assistant_turn_tool_supersession",
        start=False,
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_orphaned_file_search",
            session_id=run["id"],
            tool_name="file_search",
            arguments={"path": ".", "query": "proof"},
            status="running",
            created_at="2026-08-18T00:00:00Z",
            started_at="2026-08-18T00:00:00Z",
        )
    )
    runtime.store.add_turn(run["id"], "assistant", "The verified result is ready.")

    assert runtime.assistant._effective_active_tool_calls(run["id"]) == []
    assert runtime.assistant._latest_tool_blockers(run["id"]) == []

    runtime.store.put_tool_call(
        ToolCall(
            id="call_new_recovery_file_search",
            session_id=run["id"],
            tool_name="file_search",
            arguments={"path": ".", "query": "manifest"},
            status="running",
            created_at="2099-01-01T00:00:00Z",
            started_at="2099-01-01T00:00:00Z",
        )
    )
    blockers = runtime.assistant._latest_tool_blockers(run["id"])
    assert [item["id"] for item in blockers] == ["call_new_recovery_file_search"]
    runtime.close()


def test_final_artifact_inventory_supersedes_same_millisecond_orphaned_shell_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终发布 proof.json。",
        conversation_id="same_millisecond_shell_exec_orphan",
        start=False,
    )
    proof = Path(run["task_root"]) / "outputs" / "proof.json"
    proof.write_text('{"status":"passed"}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], proof, source_tool_call="artifact_publish")
    timestamp = "2026-08-22T00:00:00.000+00:00"
    for call_id in ("call_shell_completed_1", "call_shell_completed_2"):
        runtime.store.put_tool_call(
            ToolCall(
                id=call_id,
                session_id=run["id"],
                tool_name="shell_exec",
                arguments={"command": "fixture"},
                status="completed",
                created_at=timestamp,
                started_at=timestamp,
                completed_at=timestamp,
            ),
            ToolResult(call_id, True, {"status": "completed"}, "completed"),
        )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_shell_orphan",
            session_id=run["id"],
            tool_name="shell_exec",
            arguments={"command": "fixture"},
            status="running",
            created_at=timestamp,
            started_at=timestamp,
        )
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_artifact_inventory",
            session_id=run["id"],
            tool_name="artifact_list",
            arguments={},
            status="completed",
            created_at=timestamp,
            started_at=timestamp,
            completed_at=timestamp,
        ),
        ToolResult("call_artifact_inventory", True, {"count": 1}, "verified"),
    )
    monkeypatch.setattr("evomind_runtime.store.utc_now", lambda: timestamp)
    runtime.store.add_turn(run["id"], "assistant", "The verified result is ready.")

    assert [item["id"] for item in runtime.assistant._effective_active_tool_calls(run["id"])] == [
        "call_shell_orphan"
    ]
    assert runtime.assistant._latest_tool_blockers(run["id"], final_answer_ready=True) == []
    assert [item["name"] for item in runtime.assistant._finalize_outputs(run["id"], "verified")] == [
        "proof.json"
    ]
    runtime.close()


def test_final_answer_does_not_hide_tool_started_after_artifact_inventory(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终发布 proof.json。",
        conversation_id="newer_live_shell_exec",
        start=False,
    )
    proof = Path(run["task_root"]) / "outputs" / "proof.json"
    proof.write_text('{"status":"passed"}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], proof, source_tool_call="artifact_publish")
    runtime.store.put_tool_call(
        ToolCall(
            id="call_old_artifact_inventory",
            session_id=run["id"],
            tool_name="artifact_list",
            arguments={},
            status="completed",
            created_at="2026-08-22T00:00:00.000+00:00",
            started_at="2026-08-22T00:00:00.000+00:00",
            completed_at="2026-08-22T00:00:01.000+00:00",
        ),
        ToolResult("call_old_artifact_inventory", True, {"count": 1}, "verified"),
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_live_shell",
            session_id=run["id"],
            tool_name="shell_exec",
            arguments={"command": "long-running"},
            status="running",
            created_at="2099-01-01T00:00:00.000+00:00",
            started_at="2099-01-01T00:00:00.000+00:00",
        )
    )
    runtime.store.add_turn(run["id"], "assistant", "An earlier artifact was verified.")

    blockers = runtime.assistant._latest_tool_blockers(run["id"], final_answer_ready=True)
    assert [item["id"] for item in blockers] == ["call_live_shell"]
    with pytest.raises(RuntimeError, match="unresolved tool calls: shell_exec"):
        runtime.assistant._finalize_outputs(run["id"], "not terminal")
    runtime.close()


def test_verified_artifacts_supersede_recovered_optional_tool_failure(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终发布 proof.json。",
        conversation_id="recovered_optional_failure",
        start=False,
    )
    proof = Path(run["task_root"]) / "outputs" / "proof.json"
    proof.write_text('{"status":"passed"}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], proof, source_tool_call="artifact_publish")
    runtime.store.put_tool_call(
        ToolCall(
            id="call_optional_search_failed",
            session_id=run["id"],
            tool_name="file_search",
            arguments={"path": ".", "query": "optional"},
            status="failed",
            created_at="2026-08-18T00:00:00Z",
            started_at="2026-08-18T00:00:00Z",
            completed_at="2026-08-18T00:00:01Z",
        ),
        ToolResult("call_optional_search_failed", False, {}, "optional search failed", error="not found"),
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_proof_preview_completed",
            session_id=run["id"],
            tool_name="artifact_preview",
            arguments={"artifact_id": "fixture"},
            status="completed",
            created_at="2026-08-18T00:00:02Z",
            started_at="2026-08-18T00:00:02Z",
            completed_at="2026-08-18T00:00:03Z",
        ),
        ToolResult("call_proof_preview_completed", True, {"status": "completed"}, "proof verified"),
    )
    runtime.store.add_turn(run["id"], "assistant", "The required proof was verified.")

    assert runtime.assistant._missing_expected_artifacts(run) == []
    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    runtime.close()


def test_completed_hpc_artifacts_supersede_failed_optional_shell_preflight(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终发布 metrics.json。",
        conversation_id="hpc_supersedes_shell_preflight",
        start=False,
    )
    metrics = Path(run["task_root"]) / "outputs" / "metrics.json"
    metrics.write_text('{"status":"complete"}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], metrics, source_tool_call="hpc_execute_solution")
    runtime.store.put_tool_call(
        ToolCall(
            id="call_shell_preflight_failed",
            session_id=run["id"],
            tool_name="shell_exec",
            arguments={"argv": ["python", "-m", "py_compile", "work/train.py"]},
            status="failed",
            created_at="2026-08-25T00:00:00Z",
            started_at="2026-08-25T00:00:00Z",
            completed_at="2026-08-25T00:00:01Z",
        ),
        ToolResult(
            "call_shell_preflight_failed",
            False,
            {"exit_code": 1},
            "optional local preflight failed",
            error="shell_exit_1",
        ),
    )
    runtime.store.put_tool_call(
        ToolCall(
            id="call_hpc_training_completed",
            session_id=run["id"],
            tool_name="hpc_execute_solution",
            arguments={"solution_id": "fixture"},
            status="completed",
            created_at="2026-08-25T00:00:02Z",
            started_at="2026-08-25T00:00:02Z",
            completed_at="2026-08-25T00:00:03Z",
        ),
        ToolResult(
            "call_hpc_training_completed",
            True,
            {"status": "completed", "published_output_artifact_count": 1},
            "HPC solution completed",
        ),
    )
    runtime.store.add_turn(run["id"], "assistant", "HPC artifacts were independently verified.")

    assert runtime.assistant._missing_expected_artifacts(run) == []
    assert runtime.assistant._latest_tool_blockers(run["id"]) == []
    runtime.close()


def test_running_direct_tool_overrides_stale_blocked_snapshot_and_prevents_duplicate_resume(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="训练并交付模型",
        conversation_id="active_tool_projection",
        start=False,
    )
    runtime.store.update_assistant_run(run["id"], status="blocked", error_class="llm_error", error_message="fixture")
    runtime.store.update_session(run["id"], status="blocked")
    runtime.store.put_tool_call(
        ToolCall(
            id="call_active_managed_training",
            session_id=run["id"],
            tool_name="hpc_execute_solution",
            arguments={"solution_id": "fixture"},
            status="running",
            created_at="2026-08-18T00:00:00Z",
            started_at="2026-08-18T00:00:00Z",
        )
    )

    snapshot = runtime.assistant.snapshot(run["id"])
    assert snapshot["status"] == "running"
    assert snapshot["terminal"] is False
    assert snapshot["error_class"] == ""
    assert snapshot["error_message"] == ""
    assert snapshot["active_tool_calls"] == ["call_active_managed_training"]

    resumed = runtime.assistant.action(run["id"], "resume")
    assert resumed["status"] == "running"
    assert runtime.assistant._threads.get(run["id"]) is None
    runtime.close()


def test_resume_accepts_bounded_user_instruction_and_keeps_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    observed: list[str] = []
    run = runtime.assistant.create_run(
        prompt="下载比赛数据",
        conversation_id="resume_instruction_fixture",
        start=False,
    )
    runtime.store.update_assistant_run(run["id"], status="blocked", error_class="source_missing")

    def completed_message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        observed.append(content)
        return _completed_message(runtime, session_id, content, max_steps=max_steps)

    monkeypatch.setattr(runtime, "message", completed_message)
    resumed = runtime.assistant.action(
        run["id"],
        "resume",
        "不要读取父 Run；直接写入固定 competition_data 目录。",
    )
    assert resumed["id"] == run["id"]
    completed = runtime.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    assert len(observed) == 1
    assert "User continuation instructions" in observed[0]
    assert "直接写入固定 competition_data 目录" in observed[0]
    events = runtime.store.list_events(run["id"])
    instruction_events = [item for item in events if item["event_type"] == "run_instruction_added"]
    assert len(instruction_events) == 1
    assert instruction_events[0]["payload"]["instruction_chars"] > 0
    runtime.close()


def test_resume_rejects_oversized_user_instruction(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="下载比赛数据",
        conversation_id="resume_instruction_limit_fixture",
        start=False,
    )
    runtime.store.update_assistant_run(run["id"], status="blocked")
    with pytest.raises(ValueError, match="at most 4000"):
        runtime.assistant.action(run["id"], "resume", "x" * 4_001)
    assert runtime.assistant.snapshot(run["id"])["status"] == "blocked"
    runtime.close()


def test_resume_repairs_completed_fallback_run_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="生成可验证报告", conversation_id="repair_false_completion", start=False)
    failed = runtime.invoke_tool(run["id"], "file_read", {"path": "inputs/missing.csv"})
    assert failed["status"] == "failed"
    fallback = Path(run["task_root"]) / "outputs" / "result.md"
    fallback.write_text("# Blocked fallback\n", encoding="utf-8")
    runtime.assistant.publish_path(run["id"], fallback, source_tool_call="verified_answer_fallback")
    runtime.store.update_assistant_run(run["id"], status="completed", completed_at="2026-08-18T00:00:00Z")
    runtime.store.update_session(run["id"], status="completed")

    def completed_after_resolving_tool(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        inputs = Path(runtime.get_session(session_id)["workspace_root"]) / "inputs"
        inputs.mkdir(exist_ok=True)
        (inputs / "missing.csv").write_text("value\n1\n", encoding="utf-8")
        succeeded = runtime.invoke_tool(session_id, "file_read", {"path": "inputs/missing.csv"})
        assert succeeded["status"] == "completed"
        return _completed_message(runtime, session_id, content, max_steps=max_steps)

    monkeypatch.setattr(runtime, "message", completed_after_resolving_tool)
    resumed = runtime.assistant.action(run["id"], "resume")
    assert resumed["id"] == run["id"]
    completed = runtime.assistant.wait(run["id"], timeout=5)

    assert completed["status"] == "completed"
    assert len(completed["artifacts"]) == 1
    assert completed["artifacts"][0]["source_tool_call"] == "auto_finalize"
    assert all(item["source_tool_call"] != "verified_answer_fallback" for item in runtime.store.list_deliverables(run["id"]))
    runtime.close()


def test_incomplete_run_recovers_after_runtime_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = AgentRuntime(tmp_path)
    run = first.assistant.create_run(prompt="重启后继续并生成文件", conversation_id="restart_fixture", start=False)
    first.store.update_assistant_run(run["id"], status="running")
    first.store.update_session(run["id"], status="running")
    first.close()

    second = AgentRuntime(tmp_path)
    monkeypatch.setattr(second, "message", lambda session_id, content, max_steps=24: _completed_message(second, session_id, content, max_steps=max_steps))
    recovered = second.assistant.recover_incomplete()
    assert recovered == [run["id"]]
    completed = second.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    assert completed["id"] == run["id"]
    assert len(completed["artifacts"]) == 1
    second.close()


def test_exact_approval_continues_same_run_and_executes_side_effect_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    turns = 0
    resumed_prompts: list[str] = []

    def approval_message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        nonlocal turns
        turns += 1
        workspace = Path(runtime.get_session(session_id)["workspace_root"])
        target = workspace / "outputs" / "delete-once.txt"
        if turns == 1:
            target.write_text("delete me", encoding="utf-8")
            waiting = runtime.invoke_tool(
                session_id,
                "file_delete",
                {"path": str(target)},
                idempotency_key="delete-once",
            )
            return {
                **waiting,
                "text": "Waiting for exact deletion approval.",
                "model_execution": {"provider": "fixture", "model": "served-model-v1", "native_tool_calls": 1},
            }
        resumed_prompts.append(content)
        return _completed_message(runtime, session_id, content, max_steps=max_steps)

    monkeypatch.setattr(runtime, "message", approval_message)
    run = runtime.assistant.create_run(prompt="删除临时文件后交付结果", conversation_id="approval_fixture")
    waiting = runtime.assistant.wait(run["id"], timeout=5)
    assert waiting["status"] == "waiting_approval"
    pending = [item for item in waiting["approvals"] if item["status"] == "pending"]
    assert len(pending) == 1

    decision = runtime.assistant.decide_approval(pending[0]["id"], True, "approve one exact deletion")
    assert decision["approval"]["status"] == "approved"
    completed = runtime.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    calls = [item for item in runtime.store.list_tool_calls(run["id"]) if item["tool_name"] == "file_delete"]
    assert len(calls) == 1
    assert calls[0]["status"] == "completed"
    assert not (Path(run["task_root"]) / "outputs" / "delete-once.txt").exists()
    assert len(resumed_prompts) == 1
    assert "workspace-relative paths" in resumed_prompts[0]
    assert "Approval/tool continuation receipt" in resumed_prompts[0]
    assert calls[0]["id"] in resumed_prompts[0]
    assert '"status": "completed"' in resumed_prompts[0]
    runtime.close()


def test_attach_secret_refs_binds_one_weather_ref_without_starting_or_approving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    identity = {
        "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
        "owner_principal_id": "admin",
        "job_id": 92257,
        "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job92257_g21",
        "allocation_generation": 21,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job92257-fixture",
    }
    run = runtime.assistant.create_run(
        prompt="加速 Weather4cast",
        conversation_id="attach_secret_fixture",
        managed_hpc_identity=identity,
        start=False,
    )
    waiting = runtime.invoke_tool(
        run["id"],
        "competition_data_accelerate",
        {"competition": "weather4cast", "timeout_seconds": 86400},
        idempotency_key="attach-weather-once",
    )
    runtime.store.update_assistant_run(run["id"], status="waiting_approval")
    secret_ref = "rs_" + "A" * 43
    claimed: list[tuple[str, list[str], dict]] = []
    monkeypatch.setattr(
        "evomind_runtime.run_secrets.RunSecretStore.available_purposes_for_run",
        lambda _store, _run_id: {},
    )
    monkeypatch.setattr(
        "evomind_runtime.run_secrets.RunSecretStore.claim_refs",
        lambda _store, run_id, refs, *, managed_hpc_identity: claimed.append(
            (run_id, list(refs or []), dict(managed_hpc_identity))
        ) or list(refs or []),
    )
    started_before = len([item for item in runtime.store.list_events(run["id"]) if item["event_type"] == "tool.started"])

    attached = runtime.assistant.action(
        run["id"], "attach_secret_refs", "", [secret_ref], identity,
    )

    assert attached["id"] == run["id"]
    assert attached["status"] == "waiting_approval"
    assert [item for item in attached["approvals"] if item["status"] == "pending"]
    assert runtime.get_session(run["id"])["status"] == "waiting_approval"
    assert runtime.get_session(run["id"])["metadata"]["run_secret_refs"] == [secret_ref]
    assert claimed == [(run["id"], [secret_ref], identity)]
    assert len([item for item in runtime.store.list_events(run["id"]) if item["event_type"] == "tool.started"]) == started_before
    call = runtime.store.get_tool_call(waiting["tool_call"]["id"])
    approval = runtime.store.get_approval(waiting["approval"]["id"])
    assert call and call["status"] == "waiting_approval"
    assert approval and approval["status"] == "pending"
    runtime.close()


def test_attach_secret_refs_rejects_identity_drift_or_running_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    identity = {
        "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
        "owner_principal_id": "admin",
        "job_id": 92257,
        "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job92257_g21",
        "allocation_generation": 21,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job92257-fixture",
    }
    run = runtime.assistant.create_run(
        prompt="加速 Weather4cast",
        conversation_id="attach_secret_reject_fixture",
        managed_hpc_identity=identity,
        start=False,
    )
    runtime.invoke_tool(
        run["id"], "competition_data_accelerate",
        {"competition": "weather4cast", "timeout_seconds": 86400},
    )
    runtime.store.update_assistant_run(run["id"], status="waiting_approval")
    secret_ref = "rs_" + "B" * 43
    monkeypatch.setattr(
        "evomind_runtime.run_secrets.RunSecretStore.claim_refs",
        lambda *_args, **_kwargs: pytest.fail("secret reference moved after gate rejection"),
    )

    with pytest.raises(ValueError, match="identity_mismatch"):
        runtime.assistant.action(
            run["id"], "attach_secret_refs", "", [secret_ref], {**identity, "job_id": 1},
        )
    monkeypatch.setattr(
        runtime.assistant,
        "_effective_active_tool_calls",
        lambda _run_id: [{"status": "running", "tool_name": "competition_data_accelerate"}],
    )
    with pytest.raises(ValueError, match="requires_one_pending_approval"):
        runtime.assistant.action(run["id"], "attach_secret_refs", "", [secret_ref], identity)
    assert runtime.get_session(run["id"])["metadata"].get("run_secret_refs") in (None, [])
    runtime.close()


def test_exact_approval_defers_resume_until_approval_worker_unwinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="删除临时文件后继续交付",
        conversation_id="approval_race_fixture",
        start=False,
    )
    target = Path(run["task_root"]) / "outputs" / "delete-after-approval.txt"
    target.write_text("delete me", encoding="utf-8")
    waiting = runtime.invoke_tool(
        run["id"],
        "file_delete",
        {"path": str(target)},
        idempotency_key="approval-race-delete",
    )
    runtime.store.update_assistant_run(run["id"], status="waiting_approval")

    release = threading.Event()
    old_worker = threading.Thread(target=release.wait, daemon=True)
    with runtime.assistant._lock:
        runtime.assistant._threads[run["id"]] = old_worker
    old_worker.start()
    monkeypatch.setattr(
        runtime,
        "message",
        lambda session_id, content, max_steps=24: _completed_message(runtime, session_id, content, max_steps=max_steps),
    )

    decision = runtime.assistant.decide_approval(waiting["approval"]["id"], True, "approve exact deletion")
    assert decision["approval"]["status"] == "approved"
    assert not target.exists()
    assert runtime.assistant.snapshot(run["id"])["status"] == "recovering"

    release.set()
    completed = runtime.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    assert run["id"] not in runtime.assistant._resume_waiters
    runtime.close()


def test_runtime_restart_recovers_approved_stale_waiting_approval_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = AgentRuntime(tmp_path)
    run = first.assistant.create_run(
        prompt="审批已执行后恢复交付",
        conversation_id="approved_restart_fixture",
        start=False,
    )
    target = Path(run["task_root"]) / "outputs" / "approved-delete.txt"
    target.write_text("delete me", encoding="utf-8")
    waiting = first.invoke_tool(run["id"], "file_delete", {"path": str(target)})
    approved = first.decide_approval(waiting["approval"]["id"], True, "approved before restart")
    assert approved["status"] == "completed"
    first.store.update_assistant_run(run["id"], status="waiting_approval")
    first.store.update_session(run["id"], status="waiting_approval")
    first.close()

    second = AgentRuntime(tmp_path)
    monkeypatch.setattr(
        second,
        "message",
        lambda session_id, content, max_steps=24: _completed_message(second, session_id, content, max_steps=max_steps),
    )
    recovered = second.assistant.recover_incomplete()
    assert recovered == [run["id"]]
    completed = second.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    second.close()


def test_environment_audit_satisfies_environment_artifact_group(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终交付文件：\nenvironment-audit.json",
        conversation_id="environment_artifact_group",
        start=False,
    )
    output = Path(run["task_root"]) / "outputs" / "environment-audit.json"
    output.write_text('{"environment":"verified"}\n', encoding="utf-8")
    runtime.assistant.publish_path(run["id"], output, source_tool_call="fixture")

    assert runtime.assistant._missing_expected_artifacts(run) == []
    runtime.close()


def test_missing_artifact_group_is_not_rendered_as_at_prefixed_filename(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="最终交付文件：\n加载环境说明",
        conversation_id="environment_artifact_group_label",
        start=False,
    )

    missing = runtime.assistant._missing_expected_artifacts(run)

    assert missing == ["artifact-group:environment"]
    assert all(not item.startswith("@") for item in missing)
    runtime.close()


def test_domain_tool_schemas_are_loaded_only_when_relevant(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    coding = {item.name for item in runtime.registry.specs_for_prompt("读取代码并修复测试")}
    research = {item.name for item in runtime.registry.specs_for_prompt("检索论文 DOI，搜索记忆并生成报告")}
    gpu = {item.name for item in runtime.registry.specs_for_prompt("下载 Kaggle 数据并在 HPC GPU 训练")}

    assert {"file_read", "file_write", "shell_exec", "artifact_publish", "training_route"} <= coding
    assert "literature_search" not in coding and "kaggle_submit" not in coding and "hpc_execute_solution" not in coding
    assert {"literature_search", "citation_audit", "memory_search", "report_generate"} <= research
    assert {"kaggle_list", "kaggle_download", "kaggle_submit", "hpc_verify", "hpc_execute_solution", "hpc_cancel"} <= gpu
    runtime.close()


def test_training_route_selects_cpu_for_small_tabular_data(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="训练这个小表格", conversation_id="route_small", start=False)
    source = Path(run["task_root"]) / "inputs" / "small.csv"
    source.write_text("x1,x2,target\n" + "\n".join(f"{index},{index % 7},{index % 2}" for index in range(442)) + "\n", encoding="utf-8")

    routed = runtime.invoke_tool(run["id"], "training_route", {
        "path": str(source), "task_description": "训练一个经典表格分类模型", "target_column": "target",
    })

    content = routed["result"]["content"]
    assert routed["status"] == "completed"
    assert content["decision"] == "local_cpu"
    assert content["profile"]["row_count"] == 442
    assert content["profile"]["feature_count"] == 2
    assert content["local_gpu_allowed"] is False
    runtime.close()


def test_training_route_keeps_negated_gpu_strategy_search_on_cpu(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="MindGames 策略进化", conversation_id="route_mindgames_cpu", start=False)

    routed = runtime.invoke_tool(run["id"], "training_route", {
        "task_description": (
            "小型 CPU-only ThreePlayerIPD 有限状态策略参数搜索和离线评估，"
            "不训练大模型，不使用 GPU，不调用 HPC"
        ),
    })

    content = routed["result"]["content"]
    assert routed["status"] == "completed"
    assert content["decision"] == "local_cpu"
    assert content["local_gpu_allowed"] is False
    assert content["hpc_required"] is False
    assert content["reasons"] == [{"code": "bounded_tabular_cpu_fit", "observed": True}]
    runtime.close()


def test_training_route_keeps_separate_positive_hpc_clause(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="受管 HPC 训练", conversation_id="route_positive_hpc", start=False)

    routed = runtime.invoke_tool(run["id"], "training_route", {
        "task_description": "不得使用本地 GPU；使用受管 HPC 执行训练",
    })

    content = routed["result"]["content"]
    assert content["decision"] == "hpc_gpu"
    assert "gpu_explicitly_requested" in {item["code"] for item in content["reasons"]}
    runtime.close()


def test_training_route_accepts_matching_attachment_id_and_path(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    uploaded = _upload(runtime, "training.csv", b"x1,x2,target\n1,2,0\n2,3,1\n", "text/csv")
    run = runtime.assistant.create_run(
        prompt="训练这个表格",
        conversation_id="route_matching_attachment",
        attachment_ids=[uploaded["attachment"]["id"]],
        start=False,
    )
    attachment = run["attachments"][0]

    routed = runtime.invoke_tool(run["id"], "training_route", {
        "attachment_id": attachment["id"],
        "path": attachment["path"],
        "task_description": "训练一个经典表格分类模型",
        "target_column": "target",
    })

    content = routed["result"]["content"]
    assert routed["status"] == "completed"
    assert content["source"]["attachment_id"] == attachment["id"]
    assert content["source"]["path"] == "inputs/training.csv"
    runtime.close()


def test_training_route_selects_hpc_for_complex_data_and_never_falls_back(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="自动训练复杂数据", conversation_id="route_complex", start=False)
    source = Path(run["task_root"]) / "inputs" / "complex.csv"
    header = ",".join([*(f"f{index}" for index in range(20)), "target"])
    row = ",".join([*("0.125" for _ in range(20)), "1"])
    source.write_text(header + "\n" + (row + "\n") * 30_000, encoding="utf-8")

    routed = runtime.invoke_tool(run["id"], "training_route", {
        "path": str(source), "task_description": "自动选择算力并训练二分类模型", "target_column": "target",
    })

    content = routed["result"]["content"]
    assert routed["status"] == "completed"
    assert content["decision"] == "hpc_gpu"
    assert content["hpc_required"] is True
    assert content["hpc_fallback"] == "blocked_recoverable_no_local_fallback"
    assert content["required_steps"][:3] == ["training_route", "hpc_verify_5_of_5", "hpc_execute_solution"]
    assert "row_threshold" in {item["code"] for item in content["reasons"]}
    event_types = [item["event_type"] for item in runtime.assistant.events(run["id"])]
    assert "data_profiled" in event_types
    assert "compute_route_selected" in event_types
    runtime.close()


def test_training_route_binds_full_managed_competition_without_run_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_sha256 = "b" * 64
    persistent = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/cure_bench"
    monkeypatch.setattr(
        "evomind_runtime.tools._competition_data_status",
        lambda _args, _context: ToolResult(
            "",
            True,
            {
                "status": "FULL_DATA_READY",
                "receipt": {
                    "status": "FULL_DATA_READY",
                    "persistent_root": persistent,
                    "manifest_sha256": manifest_sha256,
                    "files": 3,
                    "bytes": 2_631_105,
                },
                "artifact": {"id": "artifact_status_fixture"},
            },
            "ready",
        ),
    )
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="训练 CURE-Bench", conversation_id="route_managed_comp", start=False)

    routed = runtime.invoke_tool(run["id"], "training_route", {
        "competition": "cure_bench",
        "task_description": "在受管 HPC GPU 上训练当前 CURE-Bench 数据",
    })

    content = routed["result"]["content"]
    assert routed["status"] == "completed"
    assert content["decision"] == "hpc_gpu"
    assert content["source"]["source_kind"] == "managed_competition_data"
    assert content["source"]["persistent_root"] == persistent
    assert content["source"]["manifest_sha256"] == manifest_sha256
    assert content["source"]["local_copy_created"] is False
    assert content["profile"]["format"] == "managed_competition_data"
    assert "managed_competition_data" in {item["code"] for item in content["reasons"]}
    assert not any((Path(run["task_root"]) / "inputs").iterdir())
    runtime.close()


def test_training_route_rejects_non_ready_managed_competition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evomind_runtime.tools._competition_data_status",
        lambda _args, _context: ToolResult(
            "",
            True,
            {"status": "PARTIAL", "receipt": {"status": "PARTIAL"}},
            "partial",
        ),
    )
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="训练 Ariel", conversation_id="route_managed_partial", start=False)

    outcome = runtime.invoke_tool(run["id"], "training_route", {
        "competition": "ariel_2025",
        "task_description": "在受管 HPC GPU 上训练 Ariel",
    })
    assert outcome["status"] == "failed"
    assert "not FULL_DATA_READY" in outcome["result"]["error"]
    runtime.close()


def test_training_route_selects_hpc_for_deep_learning_without_data(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="训练图像模型", conversation_id="route_deep", start=False)
    routed = runtime.invoke_tool(run["id"], "training_route", {
        "task_description": "对图像数据做 Transformer 微调和多折超参数搜索",
    })
    content = routed["result"]["content"]
    assert content["decision"] == "hpc_gpu"
    assert {item["code"] for item in content["reasons"]} >= {
        "accelerator_modality", "deep_learning_requested", "search_or_repeated_fit_requested",
    }
    runtime.close()


def test_training_route_selects_hpc_for_llm_checkpoint_evaluation_without_data(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="评估语言模型", conversation_id="route_llm_eval", start=False)
    routed = runtime.invoke_tool(run["id"], "training_route", {
        "task_description": (
            "Use lm_eval to evaluate tiiuae/dense-500m-arch1 checkpoint iter_0004000 "
            "on mmlu_var5shots_astronomy"
        ),
    })

    content = routed["result"]["content"]
    assert content["decision"] == "hpc_gpu"
    assert "deep_learning_requested" in {item["code"] for item in content["reasons"]}
    assert content["required_steps"][:3] == ["training_route", "hpc_verify_5_of_5", "hpc_execute_solution"]
    runtime.close()


def _rebind_identity(
    job_id: int,
    generation: int,
    *,
    tenant_id: str = "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
    owner: str = "admin",
) -> dict:
    return {
        "tenant_id": tenant_id,
        "owner_principal_id": owner,
        "job_id": job_id,
        "credential_profile": f"{tenant_id}_job{job_id}_g{generation}",
        "allocation_generation": generation,
        "profile_instance_id": (
            "35279c5f-6a99-4dd0-b53a-789b7f03b376"
            if generation == 21
            else "8f247e8a-1450-4f8b-8c77-535ea25e2f55"
        ),
        "allocation_binding_id": f"aimslab-job{job_id}-20260817T092437",
    }


def _fixed_terminal_rebind_run(
    runtime: AgentRuntime,
    monkeypatch: pytest.MonkeyPatch,
    identity: dict,
) -> dict:
    monkeypatch.setattr(
        "evomind_runtime.assistant_runs.new_id",
        lambda prefix: FIXED_RUN_ID if prefix == "run" else f"{prefix}_fixture",
    )
    run = runtime.assistant.create_run(
        prompt="固定 Run 身份迁移测试",
        conversation_id="managed_hpc_rebind_fixture",
        managed_hpc_identity=identity,
        start=False,
    )
    session = runtime.get_session(run["id"])
    metadata = dict(session["metadata"])
    metadata["run_secret_refs"] = ["rs_consumed_fixture"]
    runtime.store.update_session(run["id"], status="cancelled", metadata_json=metadata)
    runtime.store.update_assistant_run(run["id"], status="cancelled", completed_at=utc_now())
    return run


def test_fixed_run_rebinds_current_server_identity_once_without_resuming_or_hpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    old_identity = _rebind_identity(92257, 21)
    new_identity = _rebind_identity(93015, 24)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, old_identity)
    consumed = runtime.runtime_root / "run-secrets" / "consumed" / run["id"] / "fixture.json"
    consumed.parent.mkdir(parents=True, exist_ok=True)
    consumed.write_text('{"schema":"fixture"}', encoding="utf-8")
    monkeypatch.setattr(
        runtime.assistant,
        "start",
        lambda *_args, **_kwargs: pytest.fail("rebind must not resume the Run"),
    )

    first = runtime.assistant.action(
        run["id"],
        "rebind_managed_hpc_identity",
        managed_hpc_identity=new_identity,
        idempotency_key="rebind-job93015-g24-once",
    )
    second = runtime.assistant.action(
        run["id"],
        "rebind_managed_hpc_identity",
        managed_hpc_identity=new_identity,
        idempotency_key="rebind-job93015-g24-once",
    )

    session = runtime.get_session(run["id"])
    assert session["status"] == "cancelled"
    assert runtime.store.get_assistant_run(run["id"])["status"] == "cancelled"
    assert session["metadata"]["managed_hpc_identity"] == new_identity
    assert session["metadata"]["run_secret_refs"] == []
    assert consumed.is_file()
    assert first["managed_hpc_identity_rebind"]["updated"] is True
    assert second["managed_hpc_identity_rebind"]["updated"] is False
    events = [
        item
        for item in runtime.store.list_events(run["id"])
        if item["event_type"] == "managed_hpc_identity.rebound"
    ]
    assert len(events) == 1
    assert events[0]["payload"]["run_resumed"] is False
    assert events[0]["payload"]["hpc_accessed"] is False
    assert events[0]["payload"]["from_identity_sha256"] == _managed_hpc_identity_fingerprint(old_identity)
    assert events[0]["payload"]["to_identity_sha256"] == _managed_hpc_identity_fingerprint(new_identity)
    context = SimpleNamespace(metadata=session["metadata"])
    assert _required_managed_hpc_identity({}, context) == (93015, new_identity["credential_profile"])
    runtime.close()


def test_managed_hpc_rebind_concurrent_same_key_writes_one_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, _rebind_identity(92257, 21))
    target = _rebind_identity(93015, 24)
    barrier = threading.Barrier(2)

    def invoke() -> dict:
        barrier.wait(timeout=5)
        return runtime.assistant.action(
            run["id"],
            "rebind_managed_hpc_identity",
            managed_hpc_identity=target,
            idempotency_key="rebind-concurrent-same-key",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: invoke(), range(2)))

    assert sorted(result["managed_hpc_identity_rebind"]["updated"] for result in results) == [False, True]
    events = [
        item
        for item in runtime.store.list_events(run["id"], after_seq=0, limit=100)
        if item["event_type"] == "managed_hpc_identity.rebound"
    ]
    assert len(events) == 1
    assert runtime.get_session(run["id"])["metadata"]["managed_hpc_identity"] == target
    runtime.close()


def test_managed_hpc_rebind_event_insert_failure_rolls_back_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    old_identity = _rebind_identity(92257, 21)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, old_identity)
    before = runtime.get_session(run["id"])
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute(
            """
            CREATE TRIGGER reject_managed_hpc_rebind_event
            BEFORE INSERT ON events
            WHEN NEW.event_type = 'managed_hpc_identity.rebound'
            BEGIN
              SELECT RAISE(ABORT, 'fixture_rebind_event_insert_failed');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="fixture_rebind_event_insert_failed"):
        runtime.assistant.action(
            run["id"],
            "rebind_managed_hpc_identity",
            managed_hpc_identity=_rebind_identity(93015, 24),
            idempotency_key="rebind-event-failure-rollback",
        )

    after = runtime.get_session(run["id"])
    assert after["status"] == before["status"] == "cancelled"
    assert after["metadata"] == before["metadata"]
    assert not any(
        item["event_type"] == "managed_hpc_identity.rebound"
        for item in runtime.store.list_events(run["id"], after_seq=0, limit=100)
    )
    assert runtime.store.get_assistant_run(run["id"])["status"] == "cancelled"
    runtime.close()


def test_managed_hpc_rebind_rejects_idempotency_key_reuse_for_new_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, _rebind_identity(92257, 21))
    first_target = _rebind_identity(93015, 24)
    second_target = _rebind_identity(93016, 25)
    key = "rebind-key-reuse-fixture"
    runtime.assistant.action(
        run["id"], "rebind_managed_hpc_identity",
        managed_hpc_identity=first_target, idempotency_key=key,
    )
    with pytest.raises(ValueError, match="idempotency_key_reused"):
        runtime.assistant.action(
            run["id"], "rebind_managed_hpc_identity",
            managed_hpc_identity=second_target, idempotency_key=key,
        )
    assert runtime.get_session(run["id"])["metadata"]["managed_hpc_identity"] == first_target
    assert len([
        item for item in runtime.store.list_events(run["id"])
        if item["event_type"] == "managed_hpc_identity.rebound"
    ]) == 1
    runtime.close()


def test_managed_hpc_rebind_checks_idempotency_conflicts_beyond_event_projection_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, _rebind_identity(92257, 21))
    for index in range(2_005):
        runtime.store.append_event(
            run["id"],
            "fixture.history",
            {"index": index},
        )
    key = "rebind-long-history-key"
    runtime.assistant.action(
        run["id"],
        "rebind_managed_hpc_identity",
        managed_hpc_identity=_rebind_identity(93015, 24),
        idempotency_key=key,
    )

    with pytest.raises(ValueError, match="idempotency_key_reused"):
        runtime.assistant.action(
            run["id"],
            "rebind_managed_hpc_identity",
            managed_hpc_identity=_rebind_identity(93016, 25),
            idempotency_key=key,
        )

    rebound = [
        event
        for event in runtime.store.list_events(run["id"], after_seq=2_000, limit=200)
        if event["event_type"] == "managed_hpc_identity.rebound"
    ]
    assert len(rebound) == 1
    assert runtime.get_session(run["id"])["metadata"]["managed_hpc_identity"] == _rebind_identity(93015, 24)
    runtime.close()


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (_rebind_identity(93015, 24, tenant_id="tenant_bbbbbbbbbbbbbbbbbbbbbbbb"), "principal_mismatch"),
        (_rebind_identity(93015, 24, owner="another-owner"), "principal_mismatch"),
        (_rebind_identity(93015, 21), "generation_not_monotonic"),
        (_rebind_identity(93015, 20), "generation_not_monotonic"),
    ],
)
def test_fixed_run_rebind_rejects_cross_principal_or_generation_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: dict,
    message: str,
) -> None:
    runtime = AgentRuntime(tmp_path)
    old_identity = _rebind_identity(92257, 21)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, old_identity)

    with pytest.raises(ValueError, match=message):
        runtime.assistant.action(
            run["id"],
            "rebind_managed_hpc_identity",
            managed_hpc_identity=candidate,
            idempotency_key="rebind-invalid-candidate",
        )

    assert runtime.get_session(run["id"])["metadata"]["managed_hpc_identity"] == old_identity
    assert not any(
        item["event_type"] == "managed_hpc_identity.rebound"
        for item in runtime.store.list_events(run["id"])
    )
    runtime.close()


def test_managed_hpc_rebind_rejects_nonfixed_run(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="不能迁移其他 Run",
        conversation_id="managed_hpc_cross_run_fixture",
        managed_hpc_identity=_rebind_identity(92257, 21),
        start=False,
    )
    runtime.store.update_session(run["id"], status="cancelled")
    runtime.store.update_assistant_run(run["id"], status="cancelled", completed_at=utc_now())

    with pytest.raises(ValueError, match="fixed_run_required"):
        runtime.assistant.action(
            run["id"],
            "rebind_managed_hpc_identity",
            managed_hpc_identity=_rebind_identity(93015, 24),
            idempotency_key="rebind-cross-run-rejected",
        )
    runtime.close()


def test_managed_hpc_rebind_blocks_active_pending_and_live_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = ("active", "pending", "live_secret")
    for scenario in scenarios:
        root = tmp_path / scenario
        runtime = AgentRuntime(root)
        old_identity = _rebind_identity(92257, 21)
        run = _fixed_terminal_rebind_run(runtime, monkeypatch, old_identity)
        if scenario == "active":
            runtime.store.put_tool_call(ToolCall("call_active", run["id"], "hpc_verify", {}, status="running"))
            expected = "active_tool_calls"
        elif scenario == "pending":
            call = ToolCall(
                "call_pending",
                run["id"],
                "hpc_cancel",
                {"solution_id": "fixture"},
                status="completed",
                approval_id="approval_pending",
            )
            runtime.store.put_tool_call(call, ToolResult(call.id, True, {}, "fixture"))
            runtime.store.put_approval(ApprovalRequest(
                "approval_pending",
                run["id"],
                call.id,
                call.tool_name,
                "f" * 64,
                call.arguments,
                {},
                "high",
                True,
                expires_at="2099-01-01T00:00:00+00:00",
            ))
            expected = "pending_approvals"
        else:
            monkeypatch.setattr(
                "evomind_runtime.run_secrets.RunSecretStore.available_purposes_for_run",
                lambda _store, _run_id: {"weather4cast_sftp": 1},
            )
            expected = "live_secret"
        with pytest.raises(ValueError, match=expected):
            runtime.assistant.action(
                run["id"],
                "rebind_managed_hpc_identity",
                managed_hpc_identity=_rebind_identity(93015, 24),
                idempotency_key=f"rebind-gate-{scenario}",
            )
        runtime.close()


@pytest.mark.parametrize("approval_state", ["approved", "missing"])
def test_managed_hpc_rebind_blocks_unsettled_waiting_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_state: str,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, _rebind_identity(92257, 21))
    call = ToolCall(
        f"call_{approval_state}",
        run["id"],
        "hpc_cancel",
        {"solution_id": "fixture"},
        status="waiting_approval",
        approval_id=f"approval_{approval_state}",
    )
    runtime.store.put_tool_call(call)
    if approval_state == "approved":
        runtime.store.put_approval(ApprovalRequest(
            call.approval_id,
            run["id"],
            call.id,
            call.tool_name,
            "e" * 64,
            call.arguments,
            {},
            "high",
            True,
            status="approved",
            expires_at="2099-01-01T00:00:00+00:00",
        ))

    with pytest.raises(ValueError, match="active_tool_calls"):
        runtime.assistant.action(
            run["id"],
            "rebind_managed_hpc_identity",
            managed_hpc_identity=_rebind_identity(93015, 24),
            idempotency_key=f"rebind-unsettled-{approval_state}",
        )
    runtime.close()


@pytest.mark.parametrize("approval_state", ["rejected", "expired"])
def test_managed_hpc_rebind_allows_settled_stale_waiting_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_state: str,
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = _fixed_terminal_rebind_run(runtime, monkeypatch, _rebind_identity(92257, 21))
    call = ToolCall(
        f"call_{approval_state}",
        run["id"],
        "hpc_cancel",
        {"solution_id": "fixture"},
        status="waiting_approval",
        approval_id=f"approval_{approval_state}",
    )
    runtime.store.put_tool_call(call)
    runtime.store.put_approval(ApprovalRequest(
        call.approval_id,
        run["id"],
        call.id,
        call.tool_name,
        "d" * 64,
        call.arguments,
        {},
        "high",
        True,
        status=approval_state,
        expires_at="2099-01-01T00:00:00+00:00",
    ))

    result = runtime.assistant.action(
        run["id"],
        "rebind_managed_hpc_identity",
        managed_hpc_identity=_rebind_identity(93015, 24),
        idempotency_key=f"rebind-settled-{approval_state}",
    )

    assert result["managed_hpc_identity_rebind"]["updated"] is True
    assert runtime.store.get_tool_call(call.id)["status"] == "waiting_approval"
    assert runtime.store.get_approval(call.approval_id)["status"] == approval_state
    runtime.close()


def test_managed_hpc_identity_uses_current_tenant_profile_and_rejects_retarget(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = "tenant_fixture_job91555_g13"
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "91555")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", profile)

    assert _required_managed_hpc_identity({}) == (91555, profile)
    assert _required_managed_hpc_identity({"job_id": 91555, "credential_profile": profile}) == (91555, profile)
    with pytest.raises(ValueError, match="retarget"):
        _required_managed_hpc_identity({"job_id": 91493, "credential_profile": profile})
    with pytest.raises(ValueError, match="retarget"):
        _required_managed_hpc_identity({"job_id": 91555, "credential_profile": "job91555"})


def test_managed_hpc_identity_prefers_server_bound_run_identity_over_stale_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id = "tenant_aaaaaaaaaaaaaaaaaaaaaaaa"
    current_profile = f"{tenant_id}_job91699_g15"
    monkeypatch.setenv("EVOMIND_SIIM_HPC_JOB_ID", "91493")
    monkeypatch.setenv("EVOMIND_HPC_CREDENTIAL_PROFILE", "tenant_fixture_job91493_g13")
    context = SimpleNamespace(metadata={
        "managed_hpc_identity": {
            "tenant_id": tenant_id,
            "owner_principal_id": "admin",
            "job_id": 91699,
            "credential_profile": current_profile,
            "allocation_generation": 15,
            "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
            "allocation_binding_id": "aimslab-job91699-20260817T092437",
        },
    })

    assert _required_managed_hpc_identity({}, context) == (91699, current_profile)
    assert _required_managed_hpc_identity(
        {"job_id": 91699, "credential_profile": current_profile}, context,
    ) == (91699, current_profile)
    assert _required_managed_hpc_identity(
        {"job_id": 91699, "credential_profile": "G15"}, context,
    ) == (91699, current_profile)
    with pytest.raises(ValueError, match="retarget"):
        _required_managed_hpc_identity({"job_id": 91493}, context)
    with pytest.raises(ValueError, match="retarget"):
        _required_managed_hpc_identity({"job_id": 91699, "credential_profile": "G14"}, context)


def test_assistant_run_persists_server_bound_hpc_identity_in_session_metadata(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    identity = {
        "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
        "owner_principal_id": "admin",
        "job_id": 91699,
        "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job91699_g15",
        "allocation_generation": 15,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job91699-20260817T092437",
    }

    run = runtime.assistant.create_run(
        prompt="只读复测 GPU 身份",
        conversation_id="managed_hpc_fixture",
        managed_hpc_identity=identity,
        start=False,
    )

    assert runtime.get_session(run["id"])["metadata"]["managed_hpc_identity"] == identity
    runtime.close()


def test_hpc_verify_rejects_five_samples_with_incomplete_identity_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from xsci import terminal_tools

    identity = {
        "tenant_id": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa",
        "owner_principal_id": "admin",
        "job_id": 91699,
        "credential_profile": "tenant_aaaaaaaaaaaaaaaaaaaaaaaa_job91699_g15",
        "allocation_generation": 15,
        "profile_instance_id": "35279c5f-6a99-4dd0-b53a-789b7f03b376",
        "allocation_binding_id": "aimslab-job91699-20260817T092437",
    }
    monkeypatch.setattr(
        terminal_tools,
        "_live_hpc_connection_probe",
        lambda profile, job_id, sample_count=5: {
            "ok": True,
            "status": "job_container_verified",
            "job_container_verified": True,
            "samples_requested": 5,
            "samples_passed": 5,
            "samples": [
                {"sample_index": index + 1, "job_container_verified": True}
                for index in range(5)
            ],
        },
    )
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="只读复测完整 HPC 身份回执",
        conversation_id="strict_hpc_receipt_fixture",
        managed_hpc_identity=identity,
        start=False,
    )

    outcome = runtime.invoke_tool(run["id"], "hpc_verify", {})

    assert outcome["status"] == "failed"
    assert outcome["result"]["ok"] is False
    assert outcome["result"]["error"] == "hpc_identity_evidence_incomplete"
    runtime.close()


def test_hpc_tool_schemas_never_expose_server_bound_identity_to_the_model() -> None:
    registry = build_default_registry()

    verify = registry.get("hpc_verify")
    execute = registry.get("hpc_execute_solution")
    cancel = registry.get("hpc_cancel")

    assert verify is not None and verify.input_schema["properties"] == {}
    assert execute is not None and set(execute.input_schema["properties"]) == {
        "solution_id", "script_path", "data_dir", "competition", "timeout_seconds",
    }
    assert cancel is not None and set(cancel.input_schema["properties"]) == {"solution_id"}
    for spec in (verify, execute, cancel):
        assert spec.input_schema["additionalProperties"] is False
        assert "job_id" not in spec.input_schema["properties"]
        assert "credential_profile" not in spec.input_schema["properties"]


def test_external_side_effect_tools_stop_at_exact_approval(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="准备提交", conversation_id="external_gate", start=False)
    submission = Path(run["task_root"]) / "outputs" / "submission.csv"
    submission.write_text("id,target\n1,0\n", encoding="utf-8")

    waiting_submit = runtime.invoke_tool(
        run["id"],
        "kaggle_submit",
        {"competition": "house-prices-advanced-regression-techniques", "submission_path": str(submission), "message": "verified"},
        idempotency_key="official-submit-once",
    )
    waiting_cancel = runtime.invoke_tool(
        run["id"],
        "hpc_cancel",
        {"solution_id": "fixture"},
        idempotency_key="hpc-cancel-once",
    )

    assert waiting_submit["status"] == "waiting_approval"
    assert waiting_submit["approval"]["tool_name"] == "kaggle_submit"
    assert waiting_submit["approval"]["normalized_arguments"]["submission_path"] == str(submission.resolve())
    assert waiting_cancel["status"] == "waiting_approval"
    assert waiting_cancel["approval"]["tool_name"] == "hpc_cancel"
    runtime.close()


def test_expired_approval_never_executes_the_pending_side_effect(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    session = runtime.create_session(permission_level="workspace-write")
    target = tmp_path / "expires-before-delete.txt"
    target.write_text("preserve", encoding="utf-8")
    waiting = runtime.invoke_tool(session["id"], "file_delete", {"path": str(target)}, idempotency_key="expired-delete")
    approval_id = waiting["approval"]["id"]
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute(
            "UPDATE approvals SET expires_at=? WHERE id=?",
            ("2000-01-01T00:00:00+00:00", approval_id),
        )

    first = runtime.decide_approval(approval_id, True, "too late")
    replay = runtime.decide_approval(approval_id, True, "still too late")

    assert first["status"] == "expired"
    assert replay["status"] == "expired"
    assert first["approval"]["status"] == "expired"
    assert target.read_text(encoding="utf-8") == "preserve"
    call = runtime.store.get_tool_call(waiting["tool_call"]["id"])
    assert call and call["status"] == "waiting_approval"
    runtime.close()


def test_expired_assistant_approval_is_clear_and_same_run_can_request_fresh_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="删除临时文件后继续交付",
        conversation_id="expired_assistant_approval_fixture",
        start=False,
    )
    target = Path(run["task_root"]) / "outputs" / "expired-delete.txt"
    target.write_text("preserve", encoding="utf-8")
    waiting = runtime.invoke_tool(
        run["id"],
        "file_delete",
        {"path": str(target)},
        idempotency_key="expired-assistant-delete",
    )
    runtime.store.update_assistant_run(run["id"], status="waiting_approval")
    with runtime.store._lock, runtime.store._connection:
        runtime.store._connection.execute(
            "UPDATE approvals SET expires_at=? WHERE id=?",
            ("2000-01-01T00:00:00+00:00", waiting["approval"]["id"]),
        )

    decision = runtime.assistant.decide_approval(
        waiting["approval"]["id"],
        True,
        "Approved in DeepEvo assistant UI",
    )

    assert decision["approval"]["status"] == "expired"
    assert decision["run"]["status"] == "blocked"
    assert decision["run"]["error_class"] == "approval_expired"
    assert decision["run"]["error_message"] == "Approval expired before decision. Resume this Run to request a fresh approval."
    assert "Approved in DeepEvo assistant UI" not in decision["run"]["error_message"]
    assert decision["run"]["last_event_seq"] >= 1
    assert target.read_text(encoding="utf-8") == "preserve"

    monkeypatch.setattr(
        runtime,
        "message",
        lambda session_id, content, max_steps=24: _completed_message(
            runtime, session_id, content, max_steps=max_steps
        ),
    )
    resumed = runtime.assistant.action(run["id"], "resume")
    assert resumed["id"] == run["id"]
    completed = runtime.assistant.wait(run["id"], timeout=5)
    assert completed["status"] == "completed"
    assert completed["error_class"] == ""
    assert completed["error_message"] == ""
    assert target.read_text(encoding="utf-8") == "preserve"
    runtime.close()


def test_rejected_approval_is_visible_and_latest_continuation_overrides_stale_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="删除旧文件并继续原任务",
        conversation_id="rejected_continuation_fixture",
        start=False,
    )
    target = Path(run["task_root"]) / "outputs" / "preserve.txt"
    target.write_text("preserve", encoding="utf-8")
    waiting = runtime.invoke_tool(
        run["id"],
        "file_delete",
        {"path": str(target)},
        idempotency_key="rejected-continuation-delete",
    )
    runtime.store.update_assistant_run(run["id"], status="waiting_approval")
    decision = runtime.assistant.decide_approval(
        waiting["approval"]["id"],
        False,
        "Rejected in DeepEvo assistant UI",
    )
    assert decision["approval"]["status"] == "rejected"

    resumed_prompts: list[str] = []

    def completed_message(session_id: str, content: str, *, max_steps: int = 24) -> dict:
        resumed_prompts.append(content)
        return _completed_message(runtime, session_id, content, max_steps=max_steps)

    monkeypatch.setattr(runtime, "message", completed_message)
    latest = "保留旧文件；只执行新的只读状态检查，不得重新申请删除审批。"
    resumed = runtime.assistant.action(run["id"], "resume", instruction=latest)
    assert resumed["id"] == run["id"]
    completed = runtime.assistant.wait(run["id"], timeout=5)

    assert completed["status"] == "completed"
    assert target.read_text(encoding="utf-8") == "preserve"
    assert len(resumed_prompts) == 1
    prompt = resumed_prompts[0]
    assert waiting["tool_call"]["id"] in prompt
    assert '"approval_status": "rejected"' in prompt
    assert "A rejected or expired approval is settled negative evidence." in prompt
    assert "Continuation precedence:" in prompt
    assert "Controlled secret availability (counts only; values and references are never exposed):" in prompt
    assert prompt.rfind(latest) > prompt.rfind("删除旧文件并继续原任务")
    runtime.close()


def test_explicit_continuation_deliverables_replace_stale_initial_artifact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="训练模型并最终交付 model.joblib、oof-predictions.csv、training-log.jsonl。",
        conversation_id="continuation_artifact_contract_fixture",
        start=False,
    )
    runtime.store.update_assistant_run(run["id"], status="blocked")
    outputs = Path(run["task_root"]) / "outputs"
    expected = [
        "data-source-manifest.json",
        "status-receipt.json",
        "prepare-receipt.json",
        "run-manifest.json",
    ]
    for name in expected:
        path = outputs / name
        path.write_text('{"status":"verified"}\n', encoding="utf-8")
        runtime.assistant.publish_path(run["id"], path, source_tool_call="fixture")

    monkeypatch.setattr(
        runtime,
        "message",
        lambda session_id, content, max_steps=24: _completed_message(
            runtime, session_id, content, max_steps=max_steps
        ),
    )
    instruction = (
        "本轮不启动训练。最终发布新的 data-source-manifest.json、status-receipt.json、"
        "prepare-receipt.json、run-manifest.json。"
    )
    runtime.assistant.action(run["id"], "resume", instruction=instruction)
    completed = runtime.assistant.wait(run["id"], timeout=5)

    assert completed["status"] == "completed"
    assert completed["missing_expected_artifacts"] == []
    plan = runtime.store.get_assistant_run(run["id"])["plan"]
    assert plan["expected_artifacts"] == expected
    assert plan["artifact_requirement_history"][-1]["previous_names"] == [
        "model.joblib",
        "oof-predictions.csv",
        "training-log.jsonl",
    ]
    runtime.close()


def test_explicit_continuation_tool_scope_removes_stale_browser_and_kaggle_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="打开浏览器搜索 Open Polymer 并下载 Kaggle 数据。",
        conversation_id="continuation_tool_scope_fixture",
        start=False,
    )
    runtime.store.update_assistant_run(run["id"], status="blocked")
    monkeypatch.setattr(
        runtime,
        "message",
        lambda session_id, content, max_steps=24: _completed_message(
            runtime, session_id, content, max_steps=max_steps
        ),
    )
    instruction = (
        "仅允许使用 competition_data_status 和 competition_data_prepare。"
        "不得调用浏览、搜索、Kaggle listing、训练或提交工具。"
    )
    runtime.assistant.action(run["id"], "resume", instruction=instruction)
    completed = runtime.assistant.wait(run["id"], timeout=5)

    assert completed["status"] == "completed"
    session = runtime.get_session(run["id"])
    allowed = set(session["metadata"]["run_allowed_tool_names"])
    assert {"competition_data_status", "competition_data_prepare", "file_write", "artifact_publish"} <= allowed
    assert not ({"browser_open", "browser_dom", "kaggle_list", "kaggle_download", "training_route"} & allowed)
    selected = {
        item.name
        for item in runtime._message_tool_specs(
            "Open Polymer browser search Kaggle download; competition_data_status competition_data_prepare",
            session,
        )
    }
    assert {"competition_data_status", "competition_data_prepare"} <= selected
    assert not ({"browser_open", "browser_dom", "kaggle_list", "kaggle_download", "training_route"} & selected)
    runtime.close()


def test_memory_report_and_evolution_tools_share_the_run_ledger(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".xsci").mkdir(parents=True)
    (project / ".xsci" / "lessons.md").write_text("A reusable calibration lesson for tabular ensembles.\n", encoding="utf-8")
    runtime = AgentRuntime(project)
    run = runtime.assistant.create_run(prompt="搜索记忆并生成报告", conversation_id="domain_fixture", start=False)

    memory = runtime.invoke_tool(run["id"], "memory_search", {"query": "calibration lesson", "limit": 5})
    evolution = runtime.invoke_tool(
        run["id"],
        "evolution_evaluate",
        {"metrics": {"target_fixed": True, "regression_green": True, "critical_subset_no_regression": True, "success_rate_delta_pp": 2.1}},
    )
    report = runtime.invoke_tool(run["id"], "report_generate", {"title": "Domain tool report", "summary": "Verified domain tools."})

    assert memory["status"] == "completed" and memory["result"]["content"]["count"] == 1
    assert evolution["status"] == "completed"
    assert evolution["result"]["content"]["decision"]["status"] == "awaiting_human_promotion"
    assert evolution["result"]["content"]["auto_promoted"] is False
    assert report["status"] == "completed"
    artifact = report["result"]["content"]["artifact"]
    assert Path(artifact["path"]).is_file()
    assert artifact["sha256"] == _digest(Path(artifact["path"]).read_bytes())
    runtime.close()


def test_memory_writeback_persists_verified_experience_across_runtime_and_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    first = AgentRuntime(project)
    source_run = first.assistant.create_run(
        prompt="复盘本轮自进化实验并沉淀经验",
        conversation_id="memory_source",
        start=False,
    )
    source_path = Path(source_run["task_root"]) / "outputs" / "validation-contract.json"
    source_path.write_text('{"folds": 5, "seed": 20260818}\n', encoding="utf-8")
    source_artifact = first.assistant.publish_path(source_run["id"], source_path, source_tool_call="fixture")
    evidence_path = Path(source_run["task_root"]) / "evidence" / "claim-audit.json"
    evidence_path.write_text('{"claims_verified": true}\n', encoding="utf-8")
    arguments = {
        "task_id": "porto_seguro_medal_mode",
        "record_type": "experience",
        "record": {
            "title": "Pairwise calibration lesson",
            "summary": "Pairwise monotonic calibration sentinel improved the stable search order.",
            "applicable_conditions": ["binary tabular ranking with fixed folds"],
            "lessons": ["reuse identical folds before comparing candidate families"],
            "failure_modes": ["changing folds hides regressions"],
            "strategy_changes": ["evaluate calibrated rank averaging before wider search"],
            "metrics": {"folds": 5, "validated": True},
            "tags": ["calibration", "ranking"],
        },
        "source_run_id": source_run["id"],
        "source_evidence": [
            {"kind": "artifact", "id": source_artifact["id"], "sha256": source_artifact["sha256"]},
            {"kind": "evidence", "id": evidence_path.name, "sha256": _digest(evidence_path.read_bytes())},
        ],
        "idempotency_key": "pairwise-calibration-v1",
    }

    written = first.invoke_tool(source_run["id"], "memory_writeback", arguments, idempotency_key="writeback-call-a")
    replayed = first.invoke_tool(source_run["id"], "memory_writeback", arguments, idempotency_key="writeback-call-b")

    assert written["status"] == "completed"
    memory = written["result"]["content"]
    assert memory["replayed"] is False
    assert memory["path"].startswith("workspace/evolution/porto_seguro_medal_mode/memory/")
    memory_path = project / memory["path"]
    assert memory_path.is_file()
    assert memory["bytes"] == memory_path.stat().st_size
    assert memory["sha256"] == _digest(memory_path.read_bytes())
    assert replayed["status"] == "completed"
    assert replayed["result"]["content"] == {**memory, "replayed": True}
    conflicting = first.invoke_tool(
        source_run["id"],
        "memory_writeback",
        {
            **arguments,
            "record": {**arguments["record"], "summary": "Different content must not replace the frozen record."},
        },
        idempotency_key="writeback-call-c",
    )
    assert conflicting["status"] == "failed"
    assert conflicting["result"]["error"] == "idempotency_conflict"
    assert memory["sha256"] == _digest(memory_path.read_bytes())
    first.close()

    second = AgentRuntime(project, runtime_root=project / "workspace" / "runtime-second")
    new_run = second.assistant.create_run(
        prompt="新会话搜索 calibration sentinel 经验",
        conversation_id="memory_reader",
        start=False,
    )
    found = second.invoke_tool(new_run["id"], "memory_search", {"query": "pairwise monotonic calibration sentinel", "limit": 5})

    assert found["status"] == "completed"
    matches = found["result"]["content"]["matches"]
    assert len(matches) == 1
    assert matches[0]["record_id"] == memory["record_id"]
    assert matches[0]["sha256"] == memory["sha256"]
    assert matches[0]["task_id"] == "porto_seguro_medal_mode"
    assert matches[0]["source_run_id"] == source_run["id"]
    assert matches[0]["path"] == memory["path"]
    assert "memory_writeback" in {item.name for item in second.registry.specs_for_prompt("执行自进化复盘")}
    assert "memory_writeback" in {item.name for item in second.registry.specs_for_prompt("写入项目记忆")}
    second.close()


def test_memory_writeback_rejects_unsafe_identity_secret_fields_paths_and_bad_hash(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="沉淀经验", conversation_id="memory_rejection", start=False)
    source_path = Path(run["task_root"]) / "outputs" / "evidence.json"
    source_path.write_text('{"validated": true}\n', encoding="utf-8")
    artifact = runtime.assistant.publish_path(run["id"], source_path, source_tool_call="fixture")
    base = {
        "task_id": "safe_task",
        "record_type": "retrospective",
        "record": {
            "summary": "Stable validation result.",
            "lessons": ["keep fixed folds"],
            "strategy_changes": ["test the narrow candidate first"],
        },
        "source_run_id": run["id"],
        "source_evidence": [{"kind": "artifact", "id": artifact["id"], "sha256": artifact["sha256"]}],
        "idempotency_key": "safe-write-v1",
    }

    unsafe_task = runtime.invoke_tool(run["id"], "memory_writeback", {**base, "task_id": "../escape"})
    secret_field = runtime.invoke_tool(
        run["id"],
        "memory_writeback",
        {**base, "record": {**base["record"], "metrics": {"api_key": "do-not-store"}}, "idempotency_key": "secret-v1"},
    )
    absolute_path = runtime.invoke_tool(
        run["id"],
        "memory_writeback",
        {**base, "record": {**base["record"], "summary": r"read C:\\private\\evidence.json"}, "idempotency_key": "path-v1"},
    )
    bad_hash = runtime.invoke_tool(
        run["id"],
        "memory_writeback",
        {
            **base,
            "source_evidence": [{"kind": "artifact", "id": artifact["id"], "sha256": "0" * 64}],
            "idempotency_key": "hash-v1",
        },
    )

    assert unsafe_task["status"] == "failed"
    assert "does not match" in unsafe_task["result"]["error"]
    assert secret_field["status"] == "failed"
    assert "forbidden secret field" in secret_field["result"]["error"]
    assert absolute_path["status"] == "failed"
    assert "absolute filesystem paths" in absolute_path["result"]["error"]
    assert bad_hash["status"] == "failed"
    assert bad_hash["result"]["error"] == "artifact_integrity"
    assert not (tmp_path / "workspace" / "evolution" / "escape").exists()
    runtime.close()


def test_selected_task_is_bound_to_the_run_and_verified_context_is_always_available(tmp_path: Path) -> None:
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(
        prompt="查看已有实验",
        conversation_id="selected_task_fixture",
        selected_task="siim-isic-melanoma-classification",
        start=False,
    )

    session = runtime.get_session(run["id"])
    assert session is not None
    assert session["metadata"]["selected_task"] == "siim-isic-melanoma-classification"
    assert run["selected_task"] == "siim-isic-melanoma-classification"
    assert runtime.assistant.snapshot(run["id"])["selected_task"] == "siim-isic-melanoma-classification"
    assert "verified_context" in {item.name for item in runtime.registry.specs_for_prompt("普通状态问题")}
    context = runtime.invoke_tool(run["id"], "verified_context", {})
    assert context["status"] == "completed"
    assert context["result"]["content"]["schema"] == "evomind.assistant_context.v1"
    runtime.close()


def test_office_artifact_gets_verified_pdf_and_png_derivatives(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    soffice = shutil.which("soffice") or str(Path.home() / ".codex-office" / "bin" / "soffice.exe")
    if not Path(soffice).is_file():
        pytest.skip("LibreOffice preview converter is unavailable")
    runtime = AgentRuntime(tmp_path)
    run = runtime.assistant.create_run(prompt="生成 DOCX 报告", conversation_id="office_preview", start=False)
    source = Path(run["task_root"]) / "outputs" / "研究报告.docx"
    document = docx.Document()
    document.add_heading("EvoMind Office Preview", level=1)
    document.add_paragraph("Verified DOCX to PDF and PNG preview fixture.")
    document.save(source)

    published = runtime.assistant.publish_path(run["id"], source, source_tool_call="office_fixture")
    public = runtime.assistant.snapshot(run["id"])["artifacts"][0]
    pdf_row, pdf_path = runtime.assistant.artifact_file(published["id"], "pdf")
    png_row, png_path = runtime.assistant.artifact_file(published["id"], "png")

    assert published["preview_kind"] == "office"
    assert public["source_preview_kind"] == "office"
    assert public["preview_kind"] == "pdf"
    assert public["preview_url"].endswith("?derived=pdf")
    assert pdf_row["media_type"] == "application/pdf" and pdf_path.read_bytes()[:5] == b"%PDF-"
    assert png_row["media_type"] == "image/png" and png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    runtime.close()
