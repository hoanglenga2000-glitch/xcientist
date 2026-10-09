"""Real loopback HTTP acceptance for an isolated async-approval candidate."""
from __future__ import annotations

import argparse
import contextvars
import hashlib
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import secrets
import sys
import threading
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--web-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()
    stage = args.stage_root.resolve(strict=True)
    stage.relative_to(Path("C:/ProgramData/EvoMind/staging").resolve(strict=True))
    if sha(stage/"web.zip") != args.web_sha256 or sha(stage/"runtime.zip") != args.runtime_sha256:
        raise ValueError("candidate_archive_hash_mismatch")
    source = json.loads((stage/"web/runtime-build-manifest.json").read_text())
    if source["build_id"] != args.build_id:
        raise ValueError("candidate_build_id_mismatch")
    helper_spec = importlib.util.spec_from_file_location("async_acceptance_helper", stage/"verify_invitation_server_candidate.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    before = helper.production_snapshot()
    sys.path.insert(0, str(stage/"runtime"))
    import evomind_runtime
    from evomind_runtime.http_server import make_handler
    from evomind_runtime.models import ToolResult
    from evomind_runtime.runtime import AgentRuntime
    Path(evomind_runtime.__file__).resolve().relative_to(stage/"runtime")
    workspace = stage/"async-approval-fixture"
    workspace.mkdir(exist_ok=False)
    runtime = AgentRuntime(workspace)
    release, started, resumed = threading.Event(), threading.Event(), threading.Event()
    principal = contextvars.ContextVar("async_fixture_principal", default="missing")
    invocations, resumes, checks = [], [], []
    token = secrets.token_urlsafe(32)
    server = None
    report = {"schema": "evomind.async_approval_server_acceptance.v1", "build_id": args.build_id,
              "web_sha256": args.web_sha256, "runtime_sha256": args.runtime_sha256,
              "fixture_only": True, "hpc_accessed": False, "production_deployed": False,
              "real_training_verified": False}

    def check(name, passed, **evidence):
        checks.append({"name": name, "passed": bool(passed), **evidence})
        if not passed:
            raise RuntimeError(name)

    def invoke(_name, _arguments, context):
        invocations.append((principal.get(), context.approval_verified))
        started.set()
        if not release.wait(15):
            raise RuntimeError("fixture_release_timeout")
        return ToolResult("", True, {}, "isolated async fixture only")

    def resume(*_args, **_kwargs):
        resumes.append(True)
        resumed.set()

    try:
        run = runtime.assistant.create_run(prompt="Isolated async approval acceptance", start=False)
        gate = runtime.invoke_tool(run["id"], "file_delete", {"path": "fixture.txt"}, idempotency_key="async-fixture-action")
        runtime.registry.invoke = invoke
        runtime.assistant._start_or_defer = resume
        base_handler = make_handler(runtime, token)
        class Handler(base_handler):
            def do_POST(self):
                marker = principal.set("isolated-http-owner")
                try:
                    return super().do_POST()
                finally:
                    principal.reset(marker)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def request(path, payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}"+path, data=data,
                                         headers={"Authorization": "Bearer "+token, "Content-Type": "application/json"})
            with opener.open(req, timeout=2) as response:
                return response.status, json.load(response)
        endpoint = "/v1/approvals/"+gate["approval"]["id"]+"/decision"
        timestamp = time.monotonic()
        status, answer = request(endpoint, {"approved": True})
        elapsed = time.monotonic()-timestamp
        check("approval_ack_before_action_settlement", status == 200 and elapsed < 1 and answer["approval"]["status"] == "approved" and not release.is_set(), elapsed_ms=round(elapsed*1000, 3))
        check("approved_action_started", started.wait(2))
        status, state = request("/v1/runs/"+run["id"])
        check("same_run_progress_readable_during_action", status == 200 and state["id"] == run["id"] and state["status"] == "running" and state["active_tool_calls"] == [gate["tool_call"]["id"]])
        check("request_identity_context_preserved", invocations == [("isolated-http-owner", True)])
        request(endpoint, {"approved": True})
        check("inflight_replay_does_not_repeat_action", len(invocations) == 1)
        release.set()
        check("coordinator_resumes_after_settlement", resumed.wait(3) and len(resumes) == 1)
        deadline = time.monotonic()+3
        while runtime.assistant._approval_workers and time.monotonic() < deadline:
            time.sleep(0.01)
        status, repeated = request(endpoint, {"approved": True})
        check("terminal_replay_does_not_redispatch", status == 200 and repeated["tool_result"].get("execution_enqueued") is False and len(invocations) == 1 and len(resumes) == 1)
        check("one_durable_tool_execution", len([call for call in runtime.store.list_tool_calls(run["id"]) if call.get("started_at")]) == 1)
    except Exception as error:
        report["error_class"] = type(error).__name__
    finally:
        release.set()
        if server is not None:
            server.shutdown()
            server.server_close()
        checks.append({"name": "local_workers_drained", "passed": runtime.close(timeout=5)})
        checks.append({"name": "production_unchanged", "passed": helper.production_snapshot() == before})
        report["checks"] = checks
        report["status"] = "passed" if len(checks) == 10 and all(row["passed"] for row in checks) and "error_class" not in report else "failed"
        (stage/"async-approval-acceptance.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
