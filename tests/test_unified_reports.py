import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from evomind_runtime.report_document import build_document
from evomind_runtime.report_figures import roc_points
from evomind_runtime.runtime import AgentRuntime


def source_run(runtime, *, prompt="连接诊断", payload=None):
    run = runtime.assistant.create_run(prompt=prompt, start=False)
    root = Path(runtime.get_session(run["id"])["workspace_root"])
    path = root / "outputs" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"accuracy": 0.875, "rmse": 1.25}), encoding="utf-8")
    artifact = runtime.assistant.publish_path(run["id"], path, source_tool_call="fixture_only")
    runtime.store.update_assistant_run(run["id"], status="completed")
    return run, artifact, path


def test_report_document_separates_integrity_from_science_and_sanitizes(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, artifact, _path = source_run(runtime, prompt="训练模型 fixture")
        document = build_document(runtime, run["id"], {"summary": "Safe explanation\npassword=not-for-reports", "formats": ["markdown"]})
        assert document["evidence_status"] == "hash_verified"
        assert document["report_status"] == "partial"
        assert "not-for-reports" not in json.dumps(document)
        assert document["metrics"][1]["direction"] == "lower"
        assert document["metrics"][0]["source_artifact_id"] == artifact["id"]
        assert any("独立" in text for text in document["missing"])
    finally:
        runtime.close()


def test_report_unknown_foreign_and_modified_sources_fail_closed(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, artifact, _path = source_run(runtime)
        other, foreign, _other_path = source_run(runtime)
        with pytest.raises(ValueError, match="not_owned"):
            build_document(runtime, run["id"], {"artifact_ids": [foreign["id"]]})
        Path(artifact["path"]).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="integrity_failed"):
            build_document(runtime, run["id"], {"artifact_ids": [artifact["id"]]})
    finally:
        runtime.close()


def test_unrecorded_metrics_do_not_become_zero(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, _artifact, _path = source_run(runtime, payload={"accuracy": None, "loss": "0.2", "roc_auc": True})
        document = build_document(runtime, run["id"], {"report_kind": "training"})
        assert document["metrics"] == []
        assert document["report_status"] == "partial"
    finally:
        runtime.close()


def test_native_trainer_log_and_independent_metric_value_are_adapted(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, _artifact, path = source_run(runtime, prompt="训练模型", payload={"metric": "rmse", "value": 0.125})
        log = path.parent / "training-log.jsonl"
        log.write_text('\n'.join(json.dumps(row) for row in [
            {"epoch": 1, "steps": 32, "loss": 1.2}, {"epoch": 2, "steps": 64, "loss": 0.4}]), encoding="utf-8")
        runtime.assistant.publish_path(run["id"], log, source_tool_call="fixture_only")
        document = build_document(runtime, run["id"], {})
        assert any(row["name"] == "rmse" and row["value"] == 0.125 for row in document["metrics"])
        figure = next(row for row in document["figures"] if row["title"] == "Training loss")
        assert figure["x"] == [32.0, 64.0] and figure["x_label"] == "Optimizer step"
        assert not any(row["title"] == "Learning rate" for row in document["figures"])
    finally:
        runtime.close()


def test_roc_ties_are_grouped_and_both_classes_required():
    assert roc_points([0, 1], [0.5, 0.5]) == [(0.0, 0.0), (1.0, 1.0)]
    assert roc_points([0, 1], [0.1, 0.9]) == [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    with pytest.raises(ValueError): roc_points([1, 1], [0.1, 0.9])


def test_async_report_ack_and_idempotency(tmp_path, monkeypatch):
    import evomind_runtime.report_jobs as jobs
    entered, release = threading.Event(), threading.Event()
    runtime = AgentRuntime(tmp_path)
    calls = []
    def renderer(document, output):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        output.mkdir(parents=True)
        (output / "report.md").write_text("fixture report", encoding="utf-8")
        return {"schema": "evomind.report_package.v2", "document_sha256": "a"*64,
                "report_status": "ready", "files": []}
    monkeypatch.setattr(jobs, "render_document", renderer)
    try:
        run, _artifact, _path = source_run(runtime)
        args = {"formats": ["markdown"]}
        started = time.monotonic()
        first = runtime.reports.enqueue(run["id"], args)
        assert time.monotonic() - started < 1 and entered.wait(1)
        second = runtime.reports.enqueue(run["id"], args)
        assert first["id"] == second["id"] and len(calls) == 1
        release.set()
        deadline = time.monotonic() + 5
        while runtime.reports.get(run["id"], first["id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        final = runtime.reports.get(run["id"], first["id"])
        assert final["status"] == "ready", final
        assert len(final["artifacts"]) == 1
        assert runtime.reports.enqueue(run["id"], args)["id"] == first["id"]
        assert len(calls) == 1
    finally:
        release.set()
        runtime.close(timeout=6)


def test_report_ids_do_not_cross_runs(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, _artifact, _path = source_run(runtime)
        runtime.pause(run["id"])  # completed parent is not rewritten
        runtime.reports.stopping = True
        job = runtime.reports.enqueue(run["id"], {"formats": ["markdown"]})
        other, _other_artifact, _other_path = source_run(runtime)
        with pytest.raises(KeyError): runtime.reports.get(other["id"], job["id"])
    finally:
        runtime.close()


def test_report_queries_require_owner_even_with_a_known_report_id(tmp_path):
    from evomind_runtime.tenant_access import Principal, current_principal, AccessError
    runtime = AgentRuntime(tmp_path)
    owner = Principal("tenant_" + "a"*24, "owner")
    stranger = Principal("tenant_" + "b"*24, "stranger")
    token = current_principal.set(owner)
    try:
        run, _artifact, _path = source_run(runtime)
        runtime.reports.stopping = True
        job = runtime.reports.enqueue(run["id"], {"formats": ["markdown"]})
        assert runtime.reports.get(run["id"], job["id"])["run_id"] == run["id"]
        current_principal.set(stranger)
        with pytest.raises(AccessError): runtime.reports.get(run["id"], job["id"])
        with pytest.raises(AccessError): runtime.reports.list(run["id"])
        with pytest.raises(AccessError): runtime.reports.resume(run["id"], job["id"])
    finally:
        current_principal.reset(token)
        runtime.close()


def test_report_resume_cannot_change_frozen_inputs(tmp_path):
    runtime = AgentRuntime(tmp_path)
    try:
        run, _artifact, _path = source_run(runtime)
        runtime.reports.stopping = True
        result = runtime.invoke_tool(run["id"], "report_generate", {"formats": ["markdown"]})
        assert result["result"]["ok"] is True
        identifier = result["result"]["content"]["report_job"]["id"]
        changed = runtime.invoke_tool(run["id"], "report_generate", {"report_id": identifier, "summary": "change"})
        assert changed["result"]["ok"] is False
        current = runtime.invoke_tool(run["id"], "report_status", {"report_id": identifier})
        assert current["result"]["content"]["report_job"]["status"] == "queued"
    finally:
        runtime.close()


@pytest.mark.skipif(not os.environ.get("EVOMIND_REPORT_TEST_PYTHON"), reason="separate document runtime not selected")
def test_real_word_pdf_and_figures_share_document(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "tests" / "report_render_probe.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run([os.environ["EVOMIND_REPORT_TEST_PYTHON"], str(script), str(tmp_path / "rendered")],
                            env=environment, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stderr[-2000:]
    receipt = json.loads(result.stdout.strip())
    assert receipt["docx_editable_table"] and receipt["pdf_pages"] >= 1
    assert receipt["metric_in_all_formats"] and receipt["all_payload_hashes_verified"]
