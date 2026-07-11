from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import xsci.agentic_capability_benchmark as benchmark_module
from xsci.agentic_capability_benchmark import (
    _audit_workspace_agent_candidate,
    _initialize_fixture_repository,
    _snapshot,
    run_workspace_agent_benchmark,
)


class _ToolSequenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, delay_seconds: float = 0) -> None:
        super().__init__(("127.0.0.1", 0), _ToolSequenceHandler)
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.lock = threading.Lock()

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class _ToolSequenceHandler(BaseHTTPRequestHandler):
    server: _ToolSequenceServer

    def log_message(self, _format: str, *_args) -> None:
        return None

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.server.delay_seconds:
            time.sleep(self.server.delay_seconds)
        with self.server.lock:
            self.server.call_count += 1
            call_number = self.server.call_count
        name, arguments = self._action(call_number, body)
        response = {
            "model": "benchmark-local-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{call_number}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        encoded = json.dumps(response).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def _action(call_number: int, body: dict) -> tuple[str, dict]:
        if call_number == 1:
            return "workspace_search", {"query": "current_release_token", "path": "docs"}
        if call_number == 2:
            return "workspace_read", {"path": "docs/release_notes.txt"}
        if call_number == 3:
            match = re.search(r"current_release_token=(release-\d+)", json.dumps(body))
            assert match is not None
            token = match.group(1)
            patch = (
                "diff --git a/answer.txt b/answer.txt\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/answer.txt\n"
                "@@ -0,0 +1 @@\n"
                f"+{token}\n"
            )
            return "workspace_patch", {"unified_diff": patch}
        if call_number == 4:
            return "workspace_read", {"path": "answer.txt"}
        if call_number == 5:
            return "workspace_test", {"command": "git diff --check"}
        if call_number == 6:
            return "workspace_diff", {}
        if call_number == 7:
            return "workspace_finish", {
                "summary": "The candidate was inspected and validated.",
                "review": (
                    "The final diff creates only answer.txt with the release token read from the current "
                    "release notes, exactly matching the goal."
                ),
                "claims": ["review_candidate_ready"],
            }
        raise AssertionError(f"unexpected provider call {call_number}")


def _serve(server: ThreadingHTTPServer):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_workspace_agent_benchmark_without_provider_is_blocked_not_scored(tmp_path: Path):
    report = run_workspace_agent_benchmark(
        workspace_root=tmp_path,
        case_ids=["retrieval_exact_release_token"],
    )

    assert report["execution_status"] == "blocked"
    assert report["block_reason"] == "provider_required"
    assert report["cases_run"] == 0
    assert report["passed_cases"] == 0
    assert report["failed_cases"] == 0
    assert report["task_success_rate"] is None
    assert report["case_results"] == []
    assert report["run_id"].startswith("agentic_benchmark_")
    assert report["report_written"] is True
    assert Path(report["report_path"]) == (tmp_path / ".xsci" / "agentic_capability_benchmark.json").resolve()
    assert json.loads(Path(report["report_path"]).read_text(encoding="utf-8")) == report
    assert not Path(report["report_path"] + ".lock").exists()


def test_workspace_agent_benchmark_active_cross_process_lock_fails_closed_without_overwrite(tmp_path: Path):
    target = tmp_path / ".xsci" / "agentic_capability_benchmark.json"
    lock_path = target.with_name(target.name + ".lock")
    marker = tmp_path / "lock-ready"
    target.parent.mkdir(parents=True)
    target.write_text('{"sentinel": true}\n', encoding="utf-8")
    script = (
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "lock_path = Path(sys.argv[1])\n"
        "marker = Path(sys.argv[2])\n"
        "fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
        "with os.fdopen(fd, 'w', encoding='utf-8') as handle:\n"
        "    json.dump({'run_id': 'active-child-run', 'pid': os.getpid(), 'started_at': '2026-07-11T00:00:00Z'}, handle)\n"
        "marker.write_text('ready', encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script, str(lock_path), str(marker)])
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()

        report = run_workspace_agent_benchmark(
            workspace_root=tmp_path,
            case_ids=["retrieval_exact_release_token"],
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
        lock_path.unlink(missing_ok=True)

    assert report["execution_status"] == "blocked"
    assert report["block_reason"] == "benchmark_already_running"
    assert report["cases_run"] == 0
    assert report["task_success_rate"] is None
    assert report["report_written"] is False
    assert report["active_run_id"] == "active-child-run"
    assert report["run_id"] != report["active_run_id"]
    assert json.loads(target.read_text(encoding="utf-8")) == {"sentinel": True}


def test_workspace_agent_benchmark_reclaims_dead_pid_lock_and_writes_new_run(tmp_path: Path):
    target = tmp_path / ".xsci" / "agentic_capability_benchmark.json"
    lock_path = target.with_name(target.name + ".lock")
    target.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({
        "run_id": "stale-run",
        "pid": 2_147_483_647,
        "started_at": "2020-01-01T00:00:00Z",
    }), encoding="utf-8")

    report = run_workspace_agent_benchmark(
        workspace_root=tmp_path,
        case_ids=["retrieval_exact_release_token"],
    )

    assert report["execution_status"] == "blocked"
    assert report["block_reason"] == "provider_required"
    assert report["cases_run"] == 0
    assert report["run_id"] != "stale-run"
    assert report["report_written"] is True
    assert not lock_path.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["run_id"] == report["run_id"]


def test_snapshot_ignores_git_metadata_but_not_workspace_files(tmp_path: Path):
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "config").write_text("internal\n", encoding="utf-8")
    (tmp_path / ".git" / "objects" / "record").write_text("internal\n", encoding="utf-8")
    (tmp_path / "answer.txt").write_text("visible\n", encoding="utf-8")

    snapshot = _snapshot(tmp_path)
    assert set(snapshot) == {"answer.txt"}
    assert snapshot["answer.txt"].replace(b"\r\n", b"\n") == b"visible\n"


def test_workspace_agent_benchmark_runs_production_subprocess_and_scores_oracle(
    tmp_path: Path,
    monkeypatch,
):
    server = _ToolSequenceServer()
    thread = _serve(server)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", server.base_url)
    monkeypatch.setenv("DEEPSEEK_MODEL", "benchmark-local-model")
    try:
        report = run_workspace_agent_benchmark(
            workspace_root=tmp_path,
            provider="deepseek",
            case_ids=["retrieval_exact_release_token"],
            timeout_seconds=30,
            limits={
                "max_steps": 8,
                "command_timeout_seconds": 5,
                "total_timeout_seconds": 20,
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert server.call_count == 7
    assert report["execution_status"] == "completed"
    assert report["cases_run"] == 1
    assert report["passed_cases"] == 1, report
    assert report["task_success_rate"] == 1.0
    result = report["case_results"][0]
    assert result["passed"] is True
    assert result["oracle_passed"] is True
    assert result["scope_violation"] is False
    assert result["unsupported_claim"] is False
    assert result["failure_reason"] == ""


def test_workspace_agent_benchmark_parent_timeout_terminates_worker(
    tmp_path: Path,
    monkeypatch,
):
    server = _ToolSequenceServer(delay_seconds=8)
    thread = _serve(server)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", server.base_url)
    cleaned_roots: list[Path] = []
    original_cleanup = benchmark_module._cleanup_fixture_worktrees

    def recording_cleanup(workspace: Path, child_temp_root: Path) -> list[str]:
        cleaned_roots.append(child_temp_root)
        return original_cleanup(workspace, child_temp_root)

    monkeypatch.setattr(benchmark_module, "_cleanup_fixture_worktrees", recording_cleanup)
    try:
        started = time.monotonic()
        report = run_workspace_agent_benchmark(
            workspace_root=tmp_path,
            provider="deepseek",
            case_ids=["retrieval_exact_release_token"],
            timeout_seconds=0.25,
            limits={"command_timeout_seconds": 10, "total_timeout_seconds": 20},
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert elapsed < 4
    assert report["cases_run"] == 1
    assert report["timed_out_cases"] == 1
    assert report["passed_cases"] == 0
    assert report["case_results"][0]["timed_out"] is True
    assert "was terminated" in report["case_results"][0]["failure_reason"]
    assert cleaned_roots and all(not path.exists() for path in cleaned_roots)


def test_parent_audit_rejects_forged_status_scope_sha_claims_and_allowed_paths(tmp_path: Path):
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    workspace.mkdir()
    artifacts.mkdir()
    (workspace / "fixture.txt").write_text("fixture\n", encoding="utf-8")
    source_head = _initialize_fixture_repository(workspace)
    diff_text = (
        "diff --git a/README.md b/README.md\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/README.md\n"
        "@@ -0,0 +1 @@\n"
        "+out of scope\n"
    )
    candidate_path = artifacts / "candidate.diff"
    candidate_path.write_text(diff_text, encoding="utf-8")
    manifest_path = artifacts / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    forged_result = {
        "schema": "evomind.workspace_agent.v1",
        "ok": True,
        "completed": False,
        "status": "needs_continuation",
        "needs_continuation": True,
        "source_revision": source_head,
        "provider": "deepseek",
        "allowed_edit_paths": ["README.md"],
        "scope_violations": [{"path": "README.md", "reason": "outside_allowed_edit_paths"}],
        "unsupported_claims": [{"claim": "review_candidate_ready", "reason": "runtime_evidence_missing"}],
        "main_worktree_modified": False,
        "main_dirty_before": False,
        "main_dirty_after": False,
        "main_head_before": source_head,
        "main_head_after": source_head,
        "commit_created": False,
        "merged": False,
        "final_diff": diff_text,
        "candidate_diff_sha256": "forged-sha",
        "candidate_diff_path": str(candidate_path),
        "artifact_path": str(manifest_path),
    }

    audit = _audit_workspace_agent_candidate(
        forged_result,
        workspace=workspace,
        artifact_root=artifacts,
        source_head=source_head,
        provider="deepseek",
        allowed_paths=("answer.txt",),
    )

    rendered = "; ".join(audit["reasons"])
    assert audit["ok"] is False
    assert audit["scope_violation_paths"] == ["README.md"]
    assert "did not complete" in rendered
    assert "requires continuation" in rendered
    assert "allowed-edit contract" in rendered
    assert "scope violation" in rendered
    assert "unsupported claims" in rendered
    assert "SHA-256" in rendered
    assert "outside the benchmark edit scope" in rendered
    assert not (workspace / "README.md").exists()
