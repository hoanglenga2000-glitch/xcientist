from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .benchmark import parity_status, parity_suite
from .assistant_runs import CHUNK_SIZE
from .competition_goal import FIXED_GOAL_ID, FIXED_RUN_ID, validate_goal_record
from .runtime import AgentRuntime
from .tenant_access import AccessError, AccessStore, current_principal, principal_from_headers
from .run_requests import RunRequests, request_key
from .user_projects import ProjectStore
from .user_tasks import UserTasks
from .model_profiles import ModelProfiles

MAX_BODY_BYTES = 1024 * 1024
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


class RequestContractError(ValueError):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


def _request_report_export(runtime: AgentRuntime, run_id: str, body: dict) -> dict:
    """Queue a report without re-entering the Run's execution lifecycle."""
    from jsonschema import Draft202012Validator
    from .tools import _report_job_summary

    allowed = {"report_id", "title", "summary", "artifact_ids", "report_kind", "language", "formats", "idempotency_key"}
    if set(body) - allowed:
        raise RequestContractError(400, "report_request_field_not_allowed")
    key = body.get("idempotency_key", "")
    if not isinstance(key, str) or len(key) > 200:
        raise RequestContractError(400, "report_idempotency_key_invalid")
    arguments = {key: value for key, value in body.items() if key != "idempotency_key"}
    if "report_id" in arguments and set(arguments) != {"report_id"}:
        raise RequestContractError(400, "report_resume_cannot_change_source")
    if "report_id" not in arguments:
        arguments.setdefault("formats", ["markdown", "html", "docx", "pdf"])
    spec = runtime.registry.get("report_generate")
    if spec is None or not spec.available:
        raise RequestContractError(503, "report_service_unavailable")
    if not Draft202012Validator(spec.input_schema).is_valid(arguments):
        raise RequestContractError(400, "report_request_invalid")
    session = runtime.get_session(run_id)
    decision = runtime.policy.evaluate(tool_name="report_generate", arguments=arguments,
        permission_level=session["permission_level"], workspace_root=session["workspace_root"])
    if not decision.allowed or decision.requires_approval:
        raise RequestContractError(403, "report_generation_not_permitted")
    # ReportJobs binds idempotency to the immutable full source document; an
    # arbitrary HTTP request key cannot substitute a report from another Run.
    job = (runtime.reports.resume(run_id, arguments["report_id"]) if "report_id" in arguments
           else runtime.reports.enqueue(run_id, arguments))
    runtime.store.append_event(run_id, "report.export_requested", {
        "report_id": job["id"], "request_kind": "resume" if "report_id" in arguments else "create",
        "report_status": job["status"], "execution_lifecycle_unchanged": True})
    return {"status": "accepted", "result": {"ok": True, "content": {"report_job": _report_job_summary(job)}}}


_DIRECT_TOOL_SESSION_STATUS = {
    "started": "running",
    "running": "running",
    "completed": "completed",
    "waiting_approval": "waiting_approval",
    "failed": "failed",
}


def _sync_direct_tool_run(runtime: AgentRuntime, run_id: str, phase: str, outcome: dict | None = None) -> None:
    run = runtime.store.get_assistant_run(run_id)
    result = outcome.get("result") if isinstance(outcome, dict) and isinstance(outcome.get("result"), dict) else {}
    if phase == "started":
        status, error_class, error_message = "running", "", ""
    elif phase == "completed":
        status, error_class, error_message = "recovering", "", ""
    elif phase == "waiting_approval":
        status, error_class, error_message = "waiting_approval", "approval_required", "Exact approval is required."
    else:
        status = "blocked"
        error_class = str(result.get("error") or "tool_failed")[:200]
        error_message = str(result.get("summary") or result.get("error") or "Direct managed tool execution failed.")[:1000]
    if not run:
        # Raw sessions (POST /v1/sessions) have no assistant Run row, so this
        # path is their only lifecycle writer.  A settled direct tool call must
        # therefore settle the session as well; otherwise the session keeps
        # claiming "running" forever and every status report over-counts live
        # work.  Cancellation stays sticky and is never overwritten here.
        session = runtime.store.get_session(run_id)
        if session is None or str(session.get("status") or "") == "cancelled":
            return
        settled = _DIRECT_TOOL_SESSION_STATUS.get(phase, "failed")
        if str(session.get("status") or "") == settled:
            return
        runtime.store.update_session(run_id, status=settled)
        if phase not in {"started", "running"}:
            runtime.store.append_event(
                run_id,
                "session.direct_tool_settled",
                {
                    "phase": phase,
                    "status": settled,
                    "tool_outcome": str((outcome or {}).get("status") or ""),
                },
            )
        return
    runtime.store.update_assistant_run(
        run_id,
        status=status,
        error_class=error_class,
        error_message=error_message,
        completed_at="",
    )
    runtime.store.update_session(run_id, status=status)


def _terminal_direct_tool_state(
    runtime: AgentRuntime,
    run_id: str,
    tool_name: str,
    arguments: dict,
    idempotency_key: str,
) -> dict | None:
    """Return a restorable session lifecycle for the fixed cancelled probe.

    The assistant Run itself must remain byte-for-byte unchanged throughout the
    call.  Only ``invoke_tool``'s transient session status is restored.
    """

    if run_id != FIXED_RUN_ID or tool_name != "hpc_asset_probe":
        return None
    run = runtime.store.get_assistant_run(run_id)
    if run is None:
        raise RequestContractError(409, "terminal_read_only_run_missing")
    session_id = str(run.get("session_id") or "")
    session = runtime.store.get_session(session_id)
    if session is None:
        raise RequestContractError(409, "terminal_read_only_session_missing")
    spec = runtime.registry.get(tool_name)
    if not spec or not spec.available or not spec.read_only:
        raise RequestContractError(409, "terminal_read_only_policy_drift")
    if (
        str(run.get("status") or "") != "cancelled"
        or str(session.get("status") or "") != "cancelled"
        or session_id != run_id
    ):
        raise RequestContractError(409, "terminal_read_only_lifecycle_mismatch")
    if not str(idempotency_key or "").strip():
        raise RequestContractError(409, "terminal_read_only_idempotency_required")
    if runtime.assistant._effective_active_tool_calls(run_id):
        raise RequestContractError(409, "terminal_read_only_active_tool_call")
    pending = [
        item
        for item in runtime.store.list_approvals(status="pending", limit=1_000_000)
        if str(item.get("session_id") or "") == run_id
    ]
    if pending:
        raise RequestContractError(409, "terminal_read_only_pending_approval")
    decision = runtime.policy.evaluate(
        tool_name=tool_name,
        arguments=arguments,
        permission_level=str(session.get("permission_level") or ""),
        workspace_root=str(session.get("workspace_root") or ""),
        approved_fingerprint="",
    )
    if not decision.allowed or decision.requires_approval:
        raise RequestContractError(409, "terminal_read_only_policy_drift")
    return {
        "session_id": session_id,
        "session_status": str(session.get("status") or ""),
        "assistant_status": str(run.get("status") or ""),
        "error_class": str(run.get("error_class") or ""),
        "error_message": str(run.get("error_message") or ""),
        "completed_at": str(run.get("completed_at") or ""),
    }


def _invoke_direct_tool(
    runtime: AgentRuntime,
    run_id: str,
    tool_name: str,
    arguments: dict,
    idempotency_key: str,
) -> dict:
    """Invoke a direct tool, preserving an already-cancelled read-only Run."""

    with runtime._invoke_lock:
        terminal_state = _terminal_direct_tool_state(
            runtime,
            run_id,
            tool_name,
            arguments,
            idempotency_key,
        )
        if terminal_state is not None:
            try:
                # The fixed read-only probe passed every terminal-state gate
                # while holding the invocation lock. Admit only this call;
                # the assistant Run stays cancelled and finally restores the
                # session even when execution or replay fails.
                runtime.store.update_session(run_id, status="running")
                outcome = runtime.invoke_tool(
                    run_id,
                    tool_name,
                    arguments,
                    idempotency_key=idempotency_key,
                )
                if (
                    str(outcome.get("status") or "") == "waiting_approval"
                    or any(
                        str(item.get("session_id") or "") == run_id
                        for item in runtime.store.list_approvals(status="pending", limit=1_000_000)
                    )
                ):
                    raise RequestContractError(409, "terminal_read_only_approval_drift")
                return outcome
            finally:
                runtime.store.update_session(
                    terminal_state["session_id"],
                    status=terminal_state["session_status"],
                )
                restored = runtime.store.get_assistant_run(run_id) or {}
                assistant_lifecycle = (
                    str(restored.get("status") or ""),
                    str(restored.get("error_class") or ""),
                    str(restored.get("error_message") or ""),
                    str(restored.get("completed_at") or ""),
                )
                expected_lifecycle = (
                    terminal_state["assistant_status"],
                    terminal_state["error_class"],
                    terminal_state["error_message"],
                    terminal_state["completed_at"],
                )
                if assistant_lifecycle != expected_lifecycle:
                    raise RequestContractError(409, "terminal_read_only_assistant_drift")

    _sync_direct_tool_run(runtime, run_id, "started")
    try:
        outcome = runtime.invoke_tool(
            run_id,
            tool_name,
            arguments,
            idempotency_key=idempotency_key,
        )
    except Exception:
        _sync_direct_tool_run(runtime, run_id, "failed")
        raise
    outcome_status = str(outcome.get("status") or "failed")
    phase = "completed" if outcome_status == "completed" else "waiting_approval" if outcome_status == "waiting_approval" else "started" if outcome_status == "running" else "failed"
    _sync_direct_tool_run(runtime, run_id, phase, outcome)
    return outcome


def ensure_token(root: Path) -> str:
    path = root / "runtime.token"
    current = ""
    try:
        current = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        pass
    if TOKEN_PATTERN.fullmatch(current):
        return current

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(token, encoding="ascii", newline="\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return token


def make_handler(runtime: AgentRuntime, token: str):
    access = AccessStore(runtime.runtime_root)
    requests = RunRequests(runtime.runtime_root)
    projects = ProjectStore(runtime.runtime_root)
    user_tasks = UserTasks(runtime.runtime_root)
    model_profiles = ModelProfiles(runtime.runtime_root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "EvoMindRuntime/0.3"

        def _json(self, status: int, value):
            data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _binary(self, status: int, body: bytes, headers: dict[str, str]):
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            raw_size = self.headers.get("Content-Length")
            if raw_size is None:
                raise RequestContractError(411, "content_length_required")
            if not raw_size.isdigit():
                raise RequestContractError(400, "invalid_content_length")
            size = int(raw_size)
            if size > MAX_BODY_BYTES:
                raise RequestContractError(413, "body_too_large")
            data = self.rfile.read(size)
            if len(data) != size:
                raise RequestContractError(400, "incomplete_body")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise RequestContractError(415, "unsupported_content_type")
            try:
                value = json.loads(data or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RequestContractError(400, "invalid_json") from exc
            if not isinstance(value, dict):
                raise RequestContractError(400, "json_object_required")
            return value

        def _raw_body(self, *, max_bytes: int, expected_content_type: str) -> bytes:
            raw_size = self.headers.get("Content-Length")
            if raw_size is None:
                raise RequestContractError(411, "content_length_required")
            if not raw_size.isdigit():
                raise RequestContractError(400, "invalid_content_length")
            size = int(raw_size)
            if size < 0 or size > max_bytes:
                raise RequestContractError(413, "body_too_large")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != expected_content_type:
                raise RequestContractError(415, "unsupported_content_type")
            data = self.rfile.read(size)
            if len(data) != size:
                raise RequestContractError(400, "incomplete_body")
            return data

        def _contract_error(self, exc: Exception):
            if isinstance(exc, (RequestContractError, AccessError)):
                self._json(exc.status, {"error": exc.code})
            elif getattr(self, "principal", None) is not None:
                # Never leak paths, database details, or exception credentials.
                status = 404 if isinstance(exc, KeyError) else 400 if isinstance(exc, (ValueError, TypeError)) else 500
                self._json(status, {"error": "not_found" if status == 404 else "invalid_request" if status == 400 else "internal_error"})
            elif isinstance(exc, KeyError):
                self._json(404, {"error": "not_found", "id": str(exc)})
            elif isinstance(exc, (ValueError, TypeError)):
                self._json(400, {"error": "invalid_request", "message": str(exc)[:1000]})
            else:
                self._json(500, {"error": type(exc).__name__, "message": str(exc)[:1000]})

        def _authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {token}"):
                self._json(401, {"error": "unauthorized"})
                return False
            try:
                self.principal = principal_from_headers(self.headers)
            except AccessError as exc:
                self._contract_error(exc)
                return False
            if getattr(self, "_scope_token", None) is not None:
                current_principal.reset(self._scope_token)
            self._scope_token = current_principal.set(self.principal)
            return True

        def finish(self):
            try:
                super().finish()
            finally:
                if getattr(self, "_scope_token", None) is not None:
                    current_principal.reset(self._scope_token)
                    self._scope_token = None

        def do_GET(self):
            if not self._authorized():
                return
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            try:
                if len(parts) == 3 and parts[:2] == ["v1", "run-requests"]:
                    return self._json(200, requests.lookup(runtime, access, self.principal, parts[2]))
                access.guard(runtime, self.principal, "GET", parts)
                if parts[:2] == ['v1', 'user-files'] and len(parts) in {2, 3}:
                    from .user_files import catalog, open_verified
                    if len(parts) == 2:
                        return self._json(200, {'files': catalog(runtime, access, self.principal)})
                    item, handle = open_verified(runtime, access, self.principal, parts[2])
                    with handle:
                        self.send_response(200)
                        self.send_header('Content-Length', str(item['bytes']))
                        self.send_header('Content-Type', 'application/octet-stream')
                        self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(item['name'], safe=''))
                        self.send_header('Cache-Control', 'private, no-store')
                        self.send_header('X-Artifact-SHA256', item['sha256'])
                        self.send_header('X-Content-Type-Options', 'nosniff')
                        self.end_headers()
                        for block in iter(lambda: handle.read(1024 * 1024), b''):
                            self.wfile.write(block)
                    return
                if parts == ["v1", "model-profiles"]:
                    return self._json(200, model_profiles.list(self.principal))
                if len(parts) == 3 and parts[:2] == ["v1", "model-profiles"]:
                    return self._json(200, {"profile": model_profiles.get(self.principal, parts[2])})
                if parts == ["v1", "user-tasks"]:
                    return self._json(200, {"tasks": user_tasks.list(self.principal, runtime, access)})
                if len(parts) == 3 and parts[:2] == ["v1", "user-tasks"]:
                    return self._json(200, user_tasks.detail(self.principal, parts[2], runtime, access))
                if parts == ["v1", "execution-budget"]:
                    from .budget_web import status
                    return self._json(200, status(runtime.runtime_root, self.principal))
                if parts == ["v1", "projects"]:
                    return self._json(200, {"projects": projects.list_owned(self.principal), "runs": projects.runs(runtime, access, self.principal), "limit": 100})
                if len(parts) == 3 and parts[:2] == ["v1", "projects"]:
                    return self._json(200, {"project": projects.get_owned(self.principal, parts[2]), "runs": projects.runs(runtime, access, self.principal, parts[2]), "limit": 100})
                if parsed.path == "/v1/health":
                    backend_version = os.environ.get("EVOMIND_BACKEND_VERSION", "0.3.0").strip() or "0.3.0"
                    return self._json(
                        200,
                        {
                            "status": "ready",
                            "version": backend_version,
                            "backend_version": backend_version,
                            "commit_hash": os.environ.get("EVOMIND_BUILD_COMMIT_HASH", "").strip(),
                            "source_tree_sha256": os.environ.get("EVOMIND_SOURCE_TREE_SHA256", "").strip(),
                        },
                    )
                if parsed.path == "/v1/tools":
                    return self._json(200, {"tools": runtime.tools()})
                if parsed.path == "/v1/goals":
                    goal = runtime.store.get_goal_record(FIXED_GOAL_ID)
                    if goal is not None and self.principal is not None and not access.owns_run(runtime, str(goal.get("run_id") or ""), self.principal):
                        goal = None
                    if goal is None:
                        return self._json(200, {"goals": []})
                    checked = validate_goal_record(goal)
                    if not checked.valid:
                        return self._json(409, {"error": "goal_record_invalid", "issues": [item.to_dict() for item in checked.issues]})
                    return self._json(
                        200,
                        {
                            "goals": [
                                {
                                    key: goal.get(key)
                                    for key in (
                                        "id", "run_id", "goal_key", "schema", "allocation", "status",
                                        "spec_sha256", "board_sha256", "human_baseline_sha256", "created_at", "updated_at",
                                    )
                                }
                            ]
                        },
                    )
                if len(parts) == 3 and parts[:2] == ["v1", "goals"]:
                    if parts[2] != FIXED_GOAL_ID:
                        raise KeyError(parts[2])
                    goal = runtime.store.get_goal_record(parts[2])
                    if goal is None:
                        raise KeyError(parts[2])
                    checked = validate_goal_record(goal)
                    if not checked.valid:
                        return self._json(409, {"error": "goal_record_invalid", "issues": [item.to_dict() for item in checked.issues]})
                    return self._json(200, goal)
                if parsed.path == "/v1/super-agent/status":
                    return self._json(200, runtime.super_agent.status())
                if parsed.path == "/v1/super-agent/capabilities":
                    query = parse_qs(parsed.query)
                    objective = query.get("objective", [""])[0]
                    operations = [
                        item
                        for value in query.get("operation", [])
                        for item in str(value).split(",")
                        if item
                    ]
                    return self._json(
                        200,
                        {
                            "schema": "evomind.capability_discovery.v1",
                            "matches": runtime.super_agent.discover(
                                objective,
                                required_operations=operations,
                                limit=min(int(query.get("limit", ["20"])[0]), 100),
                            ),
                        },
                    )
                if len(parts) == 4 and parts[:3] == ["v1", "super-agent", "runs"]:
                    snapshot = runtime.super_agent.repository.load(parts[3])
                    if snapshot is None:
                        raise KeyError(parts[3])
                    return self._json(200, snapshot)
                if parsed.path == "/v1/sessions":
                    sessions = access.list_sessions(runtime, self.principal) if self.principal is not None else runtime.list_sessions()
                    return self._json(200, {"sessions": sessions})
                if parsed.path == "/v1/runs":
                    query = parse_qs(parsed.query)
                    conversation_id = query.get("conversation_id", [""])[0]
                    limit = int(query.get("limit", ["50"])[0])
                    runs = access.list_runs(runtime, self.principal, conversation_id, limit) if self.principal is not None else runtime.assistant.list_runs(conversation_id, limit)
                    return self._json(200, {"runs": runs})
                if parsed.path == "/v1/approvals":
                    approvals = runtime.store.list_approvals(parse_qs(parsed.query).get("status", [""])[0])
                    if self.principal is not None:
                        approvals = [item for item in approvals if access.owns("session", str(item.get("session_id") or ""), self.principal)]
                    return self._json(200, {"approvals": approvals})
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "reports":
                    runtime.get_session(parts[2])
                    return self._json(200, {"reports": runtime.reports.list(parts[2])})
                if len(parts) == 5 and parts[:2] == ["v1", "runs"] and parts[3] == "reports":
                    runtime.get_session(parts[2])
                    return self._json(200, {"report": runtime.reports.get(parts[2], parts[4])})
                if parsed.path == "/v1/benchmarks":
                    return self._json(
                        200,
                        {
                            "suite": parity_suite(),
                            "runs": runtime.store.list_benchmarks(),
                            "gate": parity_status(runtime.store.list_benchmarks()),
                        },
                    )
                if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
                    return self._json(200, runtime.get_session(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "events":
                    after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                    events = runtime.store.list_events(parts[2], after)
                    if "text/event-stream" in self.headers.get("Accept", ""):
                        data = "".join(
                            f"id: {e['seq']}\nevent: {e['event_type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                            for e in events
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    return self._json(200, {"events": events})
                if len(parts) == 3 and parts[:2] == ["v1", "runs"]:
                    return self._json(200, runtime.assistant.snapshot(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "goal":
                    if parts[2] != FIXED_RUN_ID:
                        raise KeyError(parts[2])
                    goal = runtime.get_goal(parts[2])
                    if goal is None:
                        raise KeyError(parts[2])
                    checked = validate_goal_record(goal)
                    if not checked.valid:
                        return self._json(409, {"error": "goal_record_invalid", "issues": [item.to_dict() for item in checked.issues]})
                    return self._json(200, goal)
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "events":
                    after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                    return self._json(200, {"events": runtime.assistant.events(parts[2], after)})
                if len(parts) == 4 and parts[:2] == ["v1", "artifacts"] and parts[3] == "preview":
                    return self._json(200, runtime.assistant.preview(parts[2]))
                if len(parts) == 3 and parts[:2] == ["v1", "artifacts"]:
                    query = parse_qs(parsed.query)
                    derived = query.get("derived", [""])[0]
                    artifact, artifact_path = runtime.assistant.artifact_file(parts[2], derived)
                    body = artifact_path.read_bytes()
                    download = query.get("download", ["0"])[0] == "1"
                    disposition = "attachment" if download else "inline"
                    filename = re.sub(r"[^A-Za-z0-9._()\-\u3400-\u9fff]+", "_", str(artifact["name"]))[:180]
                    fallback = re.sub(r"[^A-Za-z0-9._()-]+", "_", filename).strip("._") or "artifact"
                    encoded_filename = quote(filename, safe="")
                    headers = {
                        "Content-Type": str(artifact.get("media_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"),
                        "Content-Disposition": f'{disposition}; filename="{fallback}"; filename*=UTF-8\'\'{encoded_filename}',
                        "Cache-Control": "private, no-store",
                        "X-Artifact-Bytes": str(artifact["bytes"]),
                        "X-Artifact-SHA256": str(artifact["sha256"]),
                        "X-Content-Type-Options": "nosniff",
                    }
                    if artifact_path.suffix.lower() in {".html", ".htm"}:
                        headers["Content-Security-Policy"] = "default-src 'none'; script-src 'none'; object-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; sandbox"
                    if artifact_path.suffix.lower() == ".svg":
                        source = body[:1024 * 1024].decode("utf-8", "replace")
                        if re.search(r"<(?:script|foreignObject|iframe|object|embed)\b|\son[a-z]+\s*=|(?:href|src)\s*=\s*['\"]\s*(?:javascript:|https?:|//|file:)", source, re.I):
                            raise ValueError("SVG artifact contains active content")
                    return self._binary(200, body, headers)
                self._json(404, {"error": "not_found"})
            except Exception as exc:
                self._contract_error(exc)

        def do_POST(self):
            if not self._authorized():
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                body = self._body()
                if parts == ["v1", "runs"]:
                    return self._json(201, requests.create(runtime, access, self.principal, body, self.headers.get("Idempotency-Key", "")))
                if parts == ["v1", "run-requests", "replay"]:
                    key = request_key(body, self.headers.get("Idempotency-Key", ""))
                    return self._json(200, requests.lookup(runtime, access, self.principal, key, body))
                access.guard(runtime, self.principal, "POST", parts, body)
                if parts == ["v1", "model-profiles"]:
                    return self._json(201, model_profiles.save(self.principal, body))
                if len(parts) == 3 and parts[:2] == ["v1", "model-profiles"]:
                    return self._json(200, model_profiles.save(self.principal, body, parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "model-profiles"]:
                    return self._json(200, model_profiles.command(self.principal, parts[2], parts[3], body))
                if parts == ["v1", "user-tasks"]:
                    result = user_tasks.create(self.principal, body)
                    return self._json(200 if result["replayed"] else 201, result)
                if len(parts) == 4 and parts[:2] == ["v1", "user-tasks"] and parts[3] == "draft":
                    return self._json(200, user_tasks.update_draft(self.principal, parts[2], body))
                if len(parts) == 4 and parts[:2] == ['v1', 'user-tasks'] and parts[3] == 'files':
                    return self._json(200, user_tasks.select_files(self.principal, parts[2], body, runtime, access))
                if len(parts) == 4 and parts[:2] == ["v1", "user-tasks"] and parts[3] == "links":
                    if set(body) != {"run_id"} or not isinstance(body.get("run_id"), str):
                        raise AccessError(400, "invalid_task_link")
                    return self._json(200, user_tasks.link_run(self.principal, parts[2], body["run_id"], runtime, access))
                if parts == ["v1", "execution-budget"]:
                    from .budget_web import amend
                    return self._json(200, amend(runtime.runtime_root, self.principal, body))
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "reports":
                    return self._json(202, _request_report_export(runtime, parts[2], body))
                if parts == ["v1", "projects"]:
                    result = projects.create(self.principal, body)
                    return self._json(200 if result["replayed"] else 201, result)
                if parts == ["v1", "sessions"]:
                    return self._json(201, runtime.create_session(**body))
                if parts == ["v1", "goals"]:
                    run_id = str(body.get("run_id") or "")
                    spec = body.get("spec")
                    board = body.get("board")
                    if not isinstance(spec, dict) or not isinstance(board, dict):
                        raise RequestContractError(400, "goal_spec_and_board_objects_required")
                    result = runtime.ensure_fixed_goal(
                        run_id=run_id,
                        spec=spec,
                        board=board,
                        status=str(body.get("status") or "blocked"),
                    )
                    return self._json(201 if result.get("created") else 200, result)
                if parts == ["v1", "uploads"]:
                    result = runtime.assistant.create_upload(
                        name=str(body.get("name", "")),
                        total_bytes=int(body.get("total_bytes", -1)),
                        media_type=str(body.get("media_type", "")),
                        sha256=str(body.get("sha256", "")),
                    )
                    access.created_upload(result, self.principal)
                    return self._json(201, result)
                if len(parts) == 4 and parts[:2] == ["v1", "uploads"] and parts[3] == "complete":
                    result = runtime.assistant.complete_upload(parts[2])
                    access.completed_upload(result, self.principal)
                    return self._json(200, result)
                if parts == ["v1", "runs"]:
                    attachment_ids = body.get("attachment_ids") if isinstance(body.get("attachment_ids"), list) else []
                    secret_refs = body.get("secret_refs") if isinstance(body.get("secret_refs"), list) else []
                    result = runtime.assistant.create_run(
                            prompt=str(body.get("prompt", "")),
                            conversation_id=str(body.get("conversation_id", "")),
                            attachment_ids=[str(value) for value in attachment_ids],
                            secret_refs=[str(value) for value in secret_refs],
                            managed_hpc_identity=(
                                dict(body["managed_hpc_identity"])
                                if isinstance(body.get("managed_hpc_identity"), dict)
                                else None
                            ),
                            selected_task=str(body.get("selected_task", "")),
                    )
                    access.created_run(runtime, result, self.principal)
                    return self._json(201, result)
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "actions":
                    secret_refs = body.get("secret_refs") if isinstance(body.get("secret_refs"), list) else []
                    return self._json(
                        200,
                        runtime.assistant.action(
                            parts[2],
                            str(body.get("action", "")),
                            str(body.get("instruction", "")),
                            [str(value) for value in secret_refs],
                            (
                                dict(body["managed_hpc_identity"])
                                if isinstance(body.get("managed_hpc_identity"), dict)
                                else None
                            ),
                            idempotency_key=str(body.get("idempotency_key") or ""),
                        ),
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":
                    return self._json(
                        200,
                        runtime.message(
                            parts[2], str(body.get("content", "")), max_steps=int(body.get("max_steps", 12))
                        ),
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "tools":
                    outcome = _invoke_direct_tool(
                        runtime,
                        parts[2],
                        str(body["tool_name"]),
                        dict(body.get("arguments") or {}),
                        str(body.get("idempotency_key") or ""),
                    )
                    return self._json(200, outcome)
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "resume":
                    return self._json(200, runtime.resume(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "cancel":
                    return self._json(200, runtime.cancel(parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "approvals"] and parts[3] == "decision":
                    approved = body.get("approved")
                    if type(approved) is not bool:
                        raise RequestContractError(400, "approved_boolean_required")
                    note = body.get("note", "")
                    if not isinstance(note, str):
                        raise RequestContractError(400, "approval_note_string_required")
                    return self._json(200, runtime.assistant.decide_approval(parts[2], approved, note, background=True))
                self._json(404, {"error": "not_found"})
            except Exception as exc:
                self._contract_error(exc)

        def do_PUT(self):
            if not self._authorized():
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                access.guard(runtime, self.principal, "PUT", parts)
                if len(parts) == 3 and parts[:2] == ["v1", "goals"]:
                    if parts[2] != FIXED_GOAL_ID:
                        raise KeyError(parts[2])
                    body = self._body()
                    migration = body.get("policy_migration")
                    if migration is not None:
                        if not isinstance(migration, dict):
                            raise RequestContractError(400, "goal_policy_migration_object_required")
                        if "board" in body or "status" in body:
                            raise RequestContractError(400, "goal_policy_migration_must_be_isolated")
                        new_spec = migration.get("new_spec")
                        policy_evidence = migration.get("policy_evidence")
                        if not isinstance(new_spec, dict) or not isinstance(policy_evidence, dict):
                            raise RequestContractError(
                                400,
                                "goal_policy_migration_spec_and_evidence_objects_required",
                            )
                        result = runtime.migrate_fixed_goal_policy(
                            goal_id=parts[2],
                            run_id=str(migration.get("run_id") or ""),
                            expected_spec_sha256=str(
                                migration.get("expected_spec_sha256") or ""
                            ),
                            new_spec=new_spec,
                            policy_evidence=policy_evidence,
                            migration_id=str(migration.get("migration_id") or ""),
                        )
                        return self._json(200, result)
                    board = body.get("board")
                    if not isinstance(board, dict):
                        raise RequestContractError(400, "goal_board_object_required")
                    updated = runtime.update_fixed_goal(
                        goal_id=parts[2],
                        board=board,
                        status=str(body["status"]) if "status" in body else None,
                    )
                    return self._json(200, updated)
                if len(parts) == 5 and parts[:2] == ["v1", "uploads"] and parts[3] == "chunks":
                    data = self._raw_body(max_bytes=CHUNK_SIZE, expected_content_type="application/octet-stream")
                    result = runtime.assistant.put_chunk(
                        parts[2],
                        int(parts[4]),
                        data,
                        self.headers.get("X-Chunk-SHA256", ""),
                    )
                    return self._json(200, result)
                self._json(404, {"error": "not_found"})
            except Exception as exc:
                self._contract_error(exc)

        def log_message(self, _format, *_args):
            pass

    return Handler


def serve(workspace: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    runtime = AgentRuntime(workspace)
    runtime.assistant.recover_incomplete()
    token = ensure_token(runtime.runtime_root)
    server = ThreadingHTTPServer((host, port), make_handler(runtime, token))
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        runtime.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(Path(args.workspace).resolve(), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
