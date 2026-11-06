from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .benchmark import parity_status, parity_suite
from .assistant_runs import CHUNK_SIZE, UploadLimitError
from .competition_goal import FIXED_GOAL_ID, FIXED_RUN_ID, validate_goal_record
from .runtime import AgentRuntime
from .tenant_access import AccessError, AccessStore, current_principal, principal_from_headers
from .run_requests import RunRequests, request_key
from .user_memory import UserMemoryStore
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


def _stop_run_after_gpu_cancel(runtime, run_id: str, cancelled: dict) -> None:
    """Cancel the assistant Run once the tenant's own queue place or training was cancelled (best effort, idempotent)."""
    job = cancelled.get("job") if isinstance(cancelled, dict) else None
    if not isinstance(job, dict) or job.get("label") not in {"已取消", "正在取消…"}:
        return
    try:
        row = runtime.assistant.store.get_assistant_run(run_id)
        if row and str(row.get("status") or "") in {"created", "queued", "planning", "running", "verifying", "recovering"}:
            runtime.assistant.action(run_id, "cancel")
    except Exception:  # noqa: BLE001 - the GPU job is cancelled either way; the lease deadline still bounds the run
        logging.getLogger(__name__).exception("run cancel after gpu cancel failed for %s", run_id)


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
    memories = UserMemoryStore(runtime.runtime_root)
    user_tasks = UserTasks(runtime.runtime_root)
    model_profiles = ModelProfiles(runtime.runtime_root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "EvoMindRuntime/0.3"

        def handle_one_request(self):
            # t238 C7e O3: one api latency sample per request; SSE/ZIP streams are excluded.
            started, self._ops_stream = time.monotonic(), False
            try:
                super().handle_one_request()
            finally:
                if getattr(self, "command", None) and not self._ops_stream:
                    from .ops_metrics import METRICS
                    METRICS.observe_api(time.monotonic() - started)

        def send_header(self, keyword, value):
            if keyword.lower() == "content-type" and str(value).startswith(("text/event-stream", "application/zip")):
                self._ops_stream = True
            super().send_header(keyword, value)

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
            if isinstance(exc, UploadLimitError):
                # Fixed product copy only (no paths or exception text).
                self._json(exc.status, {"error": exc.code, "message": exc.message})
            elif isinstance(exc, (RequestContractError, AccessError)):
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

        def _export(self):
            """t238 C7b D6: stream the caller's files as ZIP64 (no Content-Length; the connection closes)."""
            from .user_data import acquire_export_slot, export_filename, export_items, write_export
            items = export_items(runtime, access, self.principal)
            slot = acquire_export_slot()
            try:
                name = export_filename()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name, safe='')}")
                self.send_header("Cache-Control", "private, no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                write_export(runtime, items, self.wfile)
            finally:
                slot.release()

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
                if len(parts) == 4 and parts[:2] == ['v1', 'user-files'] and parts[3] == 'usage':
                    from .user_data import usage
                    return self._json(200, usage(runtime, access, self.principal, user_tasks, parts[2]))
                if parts == ['v1', 'tenant', 'export']:
                    return self._export()
                if parts[:2] == ['v1', 'user-files'] and len(parts) in {2, 3}:
                    from .user_files import catalog, generated_catalog, open_verified
                    if len(parts) == 2:
                        hidden = user_tasks.hidden_run_ids(self.principal)
                        files = catalog(runtime, access, self.principal)
                        if project_filter := parse_qs(parsed.query).get('project_id', [''])[0]:
                            bound = projects.file_ids(runtime, access, self.principal, project_filter)
                            files = [item for item in files if item['id'] in bound]
                        return self._json(200, {'files': files,
                                                 'generated': [item for item in generated_catalog(runtime, access, self.principal) if item['run_id'] not in hidden]})
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
                    query = parse_qs(parsed.query)
                    return self._json(200, {"tasks": user_tasks.list(self.principal, runtime, access, archived=query.get("archived", [""])[0] == "1",
                                                                     query=query.get("q", [""])[0])})
                if len(parts) == 3 and parts[:2] == ["v1", "user-tasks"]:
                    return self._json(200, user_tasks.detail(self.principal, parts[2], runtime, access))
                if len(parts) == 4 and parts[:2] == ["v1", "user-tasks"] and parts[3] == "export":
                    if parse_qs(parsed.query).get("format", ["md"])[0] != "md":
                        raise AccessError(400, "invalid_export_format")
                    from .task_export import content_disposition, today_utc
                    name, text = user_tasks.export_markdown(self.principal, parts[2], runtime, access)
                    return self._binary(200, text.encode("utf-8"), {
                        "Content-Type": "text/markdown; charset=utf-8", "Content-Disposition": content_disposition(name, today_utc()),
                        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
                if parts == ["v1", "user-memory"]:
                    def live_title(task_id: str) -> str | None:
                        try:
                            return str(user_tasks.get(self.principal, task_id).get("title") or "") or None
                        except AccessError:
                            return None
                    query = parse_qs(parsed.query)
                    pick = lambda name: query.get(name, [""])[0] or None
                    pinned = {"1": True, "true": True, "0": False, "false": False}.get(pick("pinned") or "")
                    return self._json(200, {"memories": memories.list_owned(self.principal, title_of=live_title, kind=pick("kind"),
                                                                           scope=pick("scope"), q=pick("q"), pinned=pinned)})
                if parts == ["v1", "user-settings"]:
                    return self._json(200, memories.settings(self.principal))
                if parts == ["v1", "experience", "summary"]:
                    from .experience_entry import summary
                    from .experience_store import ExperienceStore
                    counted = summary(self.principal)
                    count = ExperienceStore(runtime.runtime_root).count_mine(self.principal)
                    # 06 §2: an empty personal store keeps the unknown (null) count so the 有经验 entry stays visible.
                    return self._json(200, {**counted, "count": count or counted.get("count")})
                if parts == ["v1", "experience"]:
                    from .experience_store import ExperienceStore
                    store = ExperienceStore(runtime.runtime_root)
                    return self._json(200, {"experiences": store.list_mine(self.principal), "settings": store.settings(self.principal)})
                if parts == ["v1", "experience", "settings"]:
                    from .experience_store import ExperienceStore
                    return self._json(200, ExperienceStore(runtime.runtime_root).settings(self.principal))
                if parts == ["v1", "execution-budget"]:
                    from .budget_web import status
                    return self._json(200, status(runtime.runtime_root, self.principal))
                if parts == ["v1", "gpu", "me"]:
                    from .gpu_web import tenant_status
                    run_id = parse_qs(parsed.query).get("run_id", [""])[0]
                    if run_id:
                        access.require_run(runtime, run_id, self.principal)
                    return self._json(200, tenant_status(runtime.runtime_root, self.principal, run_id=run_id))
                if parts == ["v1", "gpu", "pool"]:
                    from .gpu_web import operator_pool
                    return self._json(200, operator_pool(runtime.runtime_root, self.principal))
                if parts == ["v1", "ops", "supervision"]:
                    # Run supervision v1: what waits for a person, slot use and failure categories (operator only).
                    if self.principal is not None:
                        raise AccessError(403, "operator_required")
                    from .run_supervision import supervision_view
                    return self._json(200, supervision_view(runtime))
                if len(parts) == 4 and parts[:2] == ["v1", "projects"] and parts[3] == "files":
                    return self._json(200, projects.files(runtime, access, self.principal, parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "projects"] and parts[3] == "memory-settings":
                    projects.get_owned(self.principal, parts[2])
                    return self._json(200, memories.project_settings(self.principal, parts[2]))
                if parts[:2] == ["v1", "projects"] and len(parts) in {2, 3}:
                    hidden = user_tasks.hidden_run_ids(self.principal)
                    runs = [row for row in projects.runs(runtime, access, self.principal, parts[2] if len(parts) == 3 else "") if row["id"] not in hidden]
                    if len(parts) == 2:
                        return self._json(200, {"projects": projects.list_owned(self.principal, user_tasks.project_counts(self.principal)), "runs": runs, "limit": 100})
                    return self._json(200, {"project": projects.get_owned(self.principal, parts[2]), "runs": runs, "limit": 100})
                if parsed.path == "/v1/health/deep":
                    # t238 C7b O2: operator only (service token, no user principal).
                    if self.principal is not None:
                        raise AccessError(403, "operator_required")
                    from .deep_health import deep_health, default_checks, ops_checks
                    version = os.environ.get("EVOMIND_BACKEND_VERSION", "0.3.0").strip() or "0.3.0"
                    report = deep_health({**default_checks(runtime), **ops_checks(runtime)}, version=version)  # t238 C7e O2/O4: + hpc, llm, capacity
                    from .ops_metrics import collect
                    report["metrics"] = collect(runtime)  # t238 C7e O3
                    return self._json(200 if report["status"] == "ok" else 503, report)
                if parsed.path == "/v1/health/models":
                    # 2026-11-03: authenticated per-provider probe for the LLM monitor; operator only.
                    if self.principal is not None:
                        raise AccessError(403, "operator_required")
                    from .model_probe import probe_providers
                    force = parse_qs(parsed.query).get("force", ["0"])[0] == "1"
                    report = probe_providers(runtime.runtime_root / "model-provider-probe.json", force=force)
                    return self._json(200 if report["ok"] else 503, report)
                if parsed.path == "/v1/metrics":
                    # t238 C7e O3: Prometheus text, operator only (service token, no user principal).
                    if self.principal is not None:
                        raise AccessError(403, "operator_required")
                    from .ops_metrics import collect, prometheus
                    return self._binary(200, prometheus(collect(runtime)).encode("utf-8"),
                                        {"Content-Type": "text/plain; version=0.0.4; charset=utf-8", "Cache-Control": "private, no-store"})
                if parsed.path == "/v1/health":
                    backend_version = os.environ.get("EVOMIND_BACKEND_VERSION", "0.3.0").strip() or "0.3.0"
                    source_commit = os.environ.get("EVOMIND_SOURCE_COMMIT", "").strip().lower()
                    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
                        source_commit = ""
                    return self._json(
                        200,
                        {
                            "status": "ready",
                            "version": backend_version,
                            "backend_version": backend_version,
                            "commit_hash": os.environ.get("EVOMIND_BUILD_COMMIT_HASH", "").strip(),
                            "source_commit": source_commit,
                            "source_revision": source_commit,
                            "source_tree_sha256": os.environ.get("EVOMIND_SOURCE_TREE_SHA256", "").strip(),
                        },
                    )
                if parsed.path == "/v1/tools":
                    from .personal_tool_boundary import offered
                    tenant_id = self.principal.tenant_id if self.principal is not None else ""
                    return self._json(200, {"tools": [item for item in runtime.tools() if offered(tenant_id, item["name"])]})
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
                    from .personal_tool_boundary import offered
                    tenant_id = self.principal.tenant_id if self.principal is not None else ""
                    matches = runtime.super_agent.discover(
                        objective,
                        required_operations=operations,
                        limit=min(int(query.get("limit", ["20"])[0]), 100),
                    )
                    return self._json(
                        200,
                        {
                            "schema": "evomind.capability_discovery.v1",
                            "matches": [
                                item for item in matches
                                if offered(tenant_id, str(item["descriptor"]["capability_id"]))
                            ],
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
                    if self.principal is None:
                        return self._json(200, {"runs": runtime.assistant.list_runs(conversation_id, limit)})
                    hidden = user_tasks.hidden_run_ids(self.principal)
                    return self._json(200, {"runs": [run for run in access.list_runs(runtime, self.principal, conversation_id, limit) if run["id"] not in hidden]})
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
                if len(parts) == 5 and parts[:3] == ["v1", "gpu", "runs"] and parts[4] == "cancel":
                    from .gpu_web import tenant_cancel
                    access.require_run(runtime, parts[3], self.principal)
                    cancelled = tenant_cancel(runtime.runtime_root, self.principal, parts[3])
                    # t238 C7c-FIX4: stopping the training is stopping the task. Cancelling only the GPU job left the Run
                    # "正在处理" and the loop re-entered with a 继续执行此任务 turn after the cancelled queue place.
                    _stop_run_after_gpu_cancel(runtime, parts[3], cancelled)
                    return self._json(200, cancelled)
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
                if len(parts) == 4 and parts[:2] == ["v1", "user-tasks"] and parts[3] == "restore":
                    return self._json(200, user_tasks.restore(self.principal, parts[2]))
                if len(parts) == 4 and parts[:2] == ["v1", "user-tasks"] and parts[3] == "links":
                    if set(body) != {"run_id"} or not isinstance(body.get("run_id"), str):
                        raise AccessError(400, "invalid_task_link")
                    return self._json(200, user_tasks.link_run(self.principal, parts[2], body["run_id"], runtime, access))
                if parts == ["v1", "execution-budget"]:
                    from .budget_web import amend
                    return self._json(200, amend(runtime.runtime_root, self.principal, body))
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "reports":
                    return self._json(202, _request_report_export(runtime, parts[2], body))
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "plan":
                    from .experience_entry import PlanDecisions
                    return self._json(200, PlanDecisions(runtime.runtime_root).decide(runtime, parts[2], body))
                if len(parts) == 4 and parts[:2] == ["v1", "user-memory"] and parts[3] == "restore":
                    if body:
                        raise AccessError(400, "invalid_memory_request")
                    return self._json(200, memories.restore(self.principal, parts[2]))
                if parts == ["v1", "projects"]:
                    result = projects.create(self.principal, body)
                    return self._json(200 if result["replayed"] else 201, result)
                if len(parts) == 4 and parts[:2] == ["v1", "projects"] and parts[3] == "files":
                    return self._json(*projects.bind_file(runtime, access, self.principal, parts[2], body))
                if len(parts) == 4 and parts[:2] == ["v1", "projects"] and parts[3] == "restore":
                    if body:
                        raise AccessError(400, "invalid_project_request")
                    return self._json(200, projects.restore(self.principal, parts[2], user_tasks))
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
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "messages":
                    return self._json(
                        200,
                        runtime.assistant.post_message(
                            parts[2],
                            mode=str(body.get("mode") or ""),
                            text=str(body.get("text") or ""),
                            idempotency_key=str(body.get("idempotency_key") or ""),
                        ),
                    )
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

        def do_PATCH(self):
            if not self._authorized():
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                body = self._body()
                access.guard(runtime, self.principal, "PATCH", parts, body)
                if len(parts) == 3 and parts[:2] == ["v1", "user-tasks"]:
                    return self._json(200, user_tasks.patch(self.principal, parts[2], body, projects))
                if len(parts) == 3 and parts[:2] == ["v1", "projects"]:
                    return self._json(200, projects.patch(self.principal, parts[2], body))
                if len(parts) == 3 and parts[:2] == ["v1", "user-memory"]:
                    return self._json(200, memories.patch(self.principal, parts[2], body))
                if len(parts) == 4 and parts[:2] == ["v1", "projects"] and parts[3] == "memory-settings":
                    projects.get_owned(self.principal, parts[2])
                    return self._json(200, memories.patch_project_settings(self.principal, parts[2], body))
                if parts == ["v1", "user-settings"]:
                    return self._json(200, memories.patch_settings(self.principal, body))
                if parts == ["v1", "experience", "settings"]:
                    from .experience_store import ExperienceStore
                    return self._json(200, ExperienceStore(runtime.runtime_root).patch_settings(self.principal, body))
                self._json(404, {"error": "not_found"})
            except Exception as exc:
                self._contract_error(exc)

        def do_DELETE(self):
            if not self._authorized():
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                # The web proxy always sends a JSON body; a bare DELETE is accepted too.
                body = self._body() if self.headers.get("Content-Length", "0") != "0" else {}
                access.guard(runtime, self.principal, "DELETE", parts, body)
                if len(parts) == 3 and parts[:2] == ["v1", "user-tasks"] and not body:
                    return self._json(200, user_tasks.delete(self.principal, parts[2]))
                if len(parts) == 3 and parts[:2] == ["v1", "model-profiles"]:
                    return self._json(200, model_profiles.delete(self.principal, parts[2], body))
                if len(parts) == 3 and parts[:2] == ["v1", "projects"] and not body:
                    return self._json(200, projects.delete(self.principal, parts[2], user_tasks))
                if parts == ["v1", "user-memory"] and not body:
                    return self._json(200, {"cleared": memories.clear_all(self.principal)})  # no undo (LOCKED)
                if len(parts) == 3 and parts[:2] == ["v1", "user-memory"] and not body:
                    return self._json(200, memories.delete(self.principal, parts[2]))
                if len(parts) == 3 and parts[:2] == ["v1", "experience"] and parts[2] != "settings" and not body:
                    from .experience_store import ExperienceStore
                    return self._json(200, ExperienceStore(runtime.runtime_root).delete_mine(self.principal, parts[2]))
                if len(parts) == 5 and parts[:2] == ["v1", "projects"] and parts[3] == "files" and not body:
                    return self._json(200, projects.unbind_file(self.principal, parts[2], parts[4]))
                if len(parts) == 3 and parts[1] in {"uploads", "attachments"}:
                    from .user_data import delete_upload
                    key = "upload_id" if parts[1] == "uploads" else "attachment_id"
                    return self._json(200, delete_upload(runtime, access, self.principal, user_tasks, **{key: parts[2]}))
                if parts == ["v1", "tenant", "data"]:
                    from .user_data import delete_all
                    return self._json(200, delete_all(runtime, access, self.principal, user_tasks, body))
                self._json(404 if not body else 400, {"error": "not_found" if not body else "invalid_task_request"})
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
    # t238 C7c FIX1 H3: the port binds first. schedule() itself reconciles before any grant, and a
    # broker restore can take minutes.
    from .gpu_admission import broker_housekeeping_in_background, reconcile_in_background
    reconcile_in_background(runtime.runtime_root)
    broker_housekeeping_in_background(runtime.runtime_root)
    from .run_supervisor import install_job_liveness
    install_job_liveness(runtime.runtime_root)  # t238 C7e m-2: stall exemption for live C7c GPU jobs
    from .tenant_purge import retry_in_background
    retry_in_background(runtime.runtime_root, runtime.assistant.task_root)
    from .ops_alerts import monitor_for
    monitor = monitor_for(runtime)  # t238 C7e O3: local alerts; EVOMIND_OPS_MONITOR=0 disables
    if monitor is not None:
        monitor.start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        if monitor is not None:
            monitor.stop()
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
