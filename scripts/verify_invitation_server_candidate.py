"""Server-side packaged-candidate acceptance with isolated data and model fixture.

No production switch, HPC access, real provider request, or model training is
performed. Loopback services are child processes owned by this controller.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile


PRODUCTION = Path("C:/ProgramData/EvoMind")
RELEASE = Path("C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866")
COOKIE = "evomind_local_session"
CSRF = "x-evomind-csrf"


class AcceptanceError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def request(port: int, method: str, path: str, *, body=None, headers=None, timeout=10) -> tuple[int, dict, object]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    outgoing = dict(headers or {})
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
        outgoing["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=outgoing)
        response = connection.getresponse()
        data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise AcceptanceError("response_size_limit")
        response_headers = {name.lower(): value for name, value in response.getheaders()}
        if "application/json" in response_headers.get("content-type", ""):
            data = json.loads(data)
        return response.status, response_headers, data
    finally:
        connection.close()


def production_snapshot() -> dict:
    state = json.loads((PRODUCTION / "state/node-processes.json").read_text(encoding="utf-8-sig"))
    records = {row["role"]: int(row["pid"]) for row in state["records"]}
    status, _, health = request(8088, "GET", "/api/healthz", timeout=5)
    return {"roles": records, "http_status": status, "build_id": health.get("build_id"), "status": health.get("status")}


def read_sse_frames(stream) -> dict:
    frames = []
    frame = {"event": "message", "id": None, "data": []}
    keepalive = False
    total = 0
    while len(frames) < 500:
        raw = stream.readline(65537)
        total += len(raw)
        if len(raw) > 65536 or total > 1024 * 1024:
            raise AcceptanceError("sse_response_size_limit")
        if not raw:
            if frame["data"] or frame["id"] is not None:
                raise AcceptanceError("sse_truncated_frame")
            break
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if frame["data"] or frame["id"] is not None or frame["event"] != "message":
                value = {"event": frame["event"], "id": frame["id"], "data": json.loads("\n".join(frame["data"])) if frame["data"] else {}}
                frames.append(value)
                frame = {"event": "message", "id": None, "data": []}
                if value["event"] in {"run_completed", "run_failed", "run_blocked"}:
                    break
            elif keepalive:
                break
            continue
        if line.startswith(":"):
            keepalive = True
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.removeprefix(" ")
        if key == "event":
            frame["event"] = value
        elif key == "id":
            if not value.isdecimal():
                raise AcceptanceError("sse_event_id_invalid")
            frame["id"] = int(value)
        elif key == "data":
            frame["data"].append(value)
    return {"frames": frames, "keepalive_seen": keepalive}


def request_sse(port: int, path: str, headers: dict) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    try:
        connection.request("GET", path, headers={**headers, "Accept": "text/event-stream"})
        response = connection.getresponse()
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            response.read(65536)
            return response.status, {"frames": [], "is_sse": False}
        return response.status, {**read_sse_frames(response), "is_sse": True}
    finally:
        connection.close()


def process_binding(worker_pid: int, launcher_pid: int, port: int, stage: Path, role: str) -> dict:
    if min(worker_pid, launcher_pid) <= 4 or port <= 1024 or port in {7890, 8088, 8765, 65068}:
        raise AcceptanceError("fixture_process_identity_rejected")
    expected = str(stage).replace("'", "''").replace("/", "\\")
    script = fr"""
$candidatePid={worker_pid};$launcherPid={launcher_pid};$expected='{expected}'.ToLowerInvariant();$rows=@()
for($i=0;$i-lt6;$i++){{
  $p=Get-CimInstance Win32_Process -Filter "ProcessId=$candidatePid"
  if($null-eq$p){{break}}
  $command=([string]$p.CommandLine).Replace('/','\').ToLowerInvariant()
  $matches=if('{role}'-eq'backend'){{$command.Contains($expected)-and$command.Contains('verify_invitation_server_candidate.py')-and$command-match'(?:^|\s)backend(?:\s|$)'}}else{{$command.Contains('server.js')}}
  $rows+= [pscustomobject]@{{pid=[int]$p.ProcessId;parent_pid=[int]$p.ParentProcessId;image=[string]$p.Name;created_at=$p.CreationDate.ToUniversalTime().ToString('o');command_matches=[bool]$matches}}
  if($candidatePid-eq$launcherPid){{break}}
  $candidatePid=[int]$p.ParentProcessId
}}
$listeners=@(Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction SilentlyContinue|Where-Object{{$_.LocalAddress-eq'127.0.0.1'}}|ForEach-Object{{[int]$_.OwningProcess}})
[pscustomobject]@{{chain=@($rows);listener_pids=@($listeners)}}|ConvertTo-Json -Depth 5 -Compress
"""
    process = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=10)
    if process.returncode:
        raise AcceptanceError("fixture_process_binding_query_failed")
    return json.loads(process.stdout)


def binding_valid(binding: dict, worker_pid: int, launcher_pid: int, role: str) -> bool:
    chain = binding.get("chain") or []
    images = {"python.exe", "pythonw.exe"} if role == "backend" else {"node.exe"}
    return bool(chain and chain[0].get("pid") == worker_pid and chain[0].get("command_matches") is True
                and str(chain[0].get("image", "")).lower() in images
                and launcher_pid in {row.get("pid") for row in chain}
                and binding.get("listener_pids") == [worker_pid])


def extract(archive: Path, expected: str, target: Path, manifest_name: str) -> dict:
    if archive.is_symlink() or sha(archive) != expected:
        raise AcceptanceError("candidate_archive_hash_mismatch")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = []
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            if (not name or "\\" in name or path.is_absolute() or path.as_posix() != name
                    or ".." in path.parts or any(":" in part for part in path.parts)
                    or ((entry.external_attr >> 16) & 0xF000) == 0xA000):
                raise AcceptanceError("candidate_archive_path_rejected")
            names.append(name)
        if len(names) > 40000 or len(names) != len({name.casefold() for name in names}) or sum(row.file_size for row in entries) > 3 * 1024 ** 3:
            raise AcceptanceError("candidate_archive_bounds_rejected")
        manifest = json.loads(bundle.read(manifest_name))
        declared = {row["path"]: row for row in manifest["files"]}
        if len(declared) != manifest["file_count"] or set(names) != {*declared, manifest_name}:
            raise AcceptanceError("candidate_manifest_file_set_mismatch")
        target.mkdir(exist_ok=False)
        for name in names:
            payload = bundle.read(name)
            if name in declared:
                entry = declared[name]
                if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise AcceptanceError("candidate_file_hash_mismatch")
            destination = target.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(payload)
    return manifest


def backend(stage: Path, port: int, boot_id: str) -> int:
    from http.server import ThreadingHTTPServer
    import evomind_runtime
    import evomind_runtime.aibuild_engine
    from evomind_runtime.http_server import ensure_token, make_handler
    from evomind_runtime.runtime import AgentRuntime

    module = Path(evomind_runtime.__file__).resolve()
    if not module.is_relative_to(stage / "runtime"):
        raise AcceptanceError("backend_import_not_candidate")
    runtime = AgentRuntime(stage / "workspace")
    token = ensure_token(runtime.runtime_root)

    def fixture_message(session_id, content, *, max_steps=12):
        if "ISOLATED_ACCEPTANCE_ECHO" not in content:
            return {"status": "blocked", "text": "fixture_prompt_required"}
        runtime.store.append_event(session_id, "acceptance_fixture_invoked", {})
        run = runtime.store.get_assistant_run(session_id)
        output = Path(run["task_root"]) / "outputs/fixture-result.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"isolated transport fixture; not a trained model\n")
        runtime.assistant.publish_path(session_id, output, source_tool_call="acceptance_fixture")
        return {"status": "completed", "text": "Isolated transport fixture completed. No model training was performed.",
                "model_execution": {"provider": "acceptance_fixture", "model": "transport-only", "native_tool_calls": 0}}

    runtime.message = fixture_message
    base = make_handler(runtime, token)

    class Handler(base):
        def do_POST(self):
            if self.path == "/fixture/shutdown":
                if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                    return self._json(401, {"error": "unauthorized"})
                self._json(202, {"status": "stopping_fixture"})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            return super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    write_json(stage / "backend-ready.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "port": server.server_port, "module": str(module), "provider": "fixture", "boot_id": boot_id})
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        if not runtime.close(timeout=5):
            raise AcceptanceError("fixture_workers_did_not_drain")
    return 0


def fixture_environment(stage: Path, session_secret: str, password: str, build_id: str, identity: str) -> dict:
    names = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in names}
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32).hex()
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(stage / "runtime"), str(PRODUCTION / "bundle/runtime"), str(stage / "web/support/python-vendor"), str(RELEASE / "src"))),
        "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8", "PYTHON": sys.executable,
        "WORKSTATION_PYTHON": sys.executable, "WORKSTATION_ROOT": str(stage / "workspace"),
        "WORKSTATION_DATA_DIR": str(stage / "workspace"), "DATABASE_URL": "file:" + (stage / "workspace/workstation.sqlite").as_posix(),
        "WORKSTATION_SESSION_SECRET": session_secret, "WORKSTATION_ADMIN_USERNAME": "invitation_acceptance",
        "WORKSTATION_ADMIN_PASSWORD_SCRYPT": "scrypt$16384$8$1$" + salt.hex() + "$" + digest,
        "WORKSTATION_LOCAL_HTTPS": "0", "HOSTNAME": "127.0.0.1", "NODE_ENV": "production",
        "NEXT_TELEMETRY_DISABLED": "1", "EVOMIND_AIBUILD_MODE": "disabled", "NO_PROXY": "*",
        "EVOMIND_RUNTIME_BUILD_MANIFEST": str(stage / "web/runtime-build-manifest.json"),
        "EVOMIND_SOURCE_TREE_SHA256": identity, "EVOMIND_BUILD_COMMIT_HASH": "664a636ddd419a66f73cc10820c0a16429784866",
        "EVOMIND_FRONTEND_VERSION": "0.3.0", "EVOMIND_BACKEND_VERSION": "0.3.0",
    })
    return environment


class FixtureServices:
    def __init__(self, stage: Path, environment: dict, node: str):
        self.stage, self.environment, self.node = stage, environment, node
        self.backend_process = self.web_process = None
        self.backend_port = 0
        self.web_port = 0
        self.round = 0
        self.log_handles = []
        self.terminated_owned_children = 0
        self.launch_records = []

    def launch(self, argv, cwd, label):
        output = (self.stage / (label + ".out.log")).open("wb")
        error = (self.stage / (label + ".err.log")).open("wb")
        self.log_handles.extend((output, error))
        return subprocess.Popen(argv, cwd=cwd, env=self.environment, stdin=subprocess.DEVNULL, stdout=output, stderr=error,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def start(self):
        self.round += 1
        ready = self.stage / "backend-ready.json"
        boot_id = uuid.uuid4().hex
        self.backend_process = self.launch([sys.executable, str(Path(__file__).resolve()), "backend", "--stage-root", str(self.stage), "--port", str(self.backend_port), "--boot-id", boot_id], self.stage, f"backend-{self.round}")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.backend_process.poll() is not None:
                raise AcceptanceError("backend_process_exited")
            if ready.exists():
                record = json.loads(ready.read_text())
                if record.get("boot_id") == boot_id:
                    actual_pid = int(record["pid"])
                    binding = process_binding(actual_pid, self.backend_process.pid, int(record["port"]), self.stage, "backend")
                    if not binding_valid(binding, actual_pid, self.backend_process.pid, "backend"):
                        raise AcceptanceError("backend_process_ownership_mismatch")
                    self.launch_records.append({"round": self.round, "role": "backend", "launcher_pid": self.backend_process.pid, "worker_pid": actual_pid, "port": int(record["port"]), "binding": binding})
                    self.backend_port = int(record["port"])
                    break
            time.sleep(0.1)
        else:
            raise AcceptanceError("backend_start_timeout")
        self.environment["EVOMIND_RUNTIME_PORT"] = str(self.backend_port)
        if not self.web_port:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                self.web_port = listener.getsockname()[1]
        self.environment["PORT"] = str(self.web_port)
        self.web_process = self.launch([self.node, "server.js"], self.stage / "web", f"web-{self.round}")
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.web_process.poll() is not None:
                raise AcceptanceError("web_process_exited")
            try:
                status, _, _ = request(self.web_port, "GET", "/login", timeout=2)
                if status == 200:
                    binding = process_binding(self.web_process.pid, self.web_process.pid, self.web_port, self.stage, "web")
                    if not binding_valid(binding, self.web_process.pid, self.web_process.pid, "web"):
                        raise AcceptanceError("web_process_ownership_mismatch")
                    self.launch_records.append({"round": self.round, "role": "web", "launcher_pid": self.web_process.pid, "worker_pid": self.web_process.pid, "port": self.web_port, "binding": binding})
                    return
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.2)
        raise AcceptanceError("web_start_timeout")

    def stop(self):
        if self.web_process is not None and self.web_process.poll() is None:
            self.web_process.terminate()
            self.web_process.wait(timeout=10)
            self.terminated_owned_children += 1
        if self.backend_process is not None and self.backend_process.poll() is None:
            try:
                token = (self.stage / "workspace/workspace/runtime/runtime.token").read_text().strip()
                request(self.backend_port, "POST", "/fixture/shutdown", body={}, headers={"Authorization": "Bearer " + token})
                self.backend_process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired, http.client.HTTPException):
                if self.backend_process.poll() is None:
                    self.backend_process.terminate()
                    self.backend_process.wait(timeout=10)
                    self.terminated_owned_children += 1
        for handle in self.log_handles:
            handle.close()
        self.log_handles.clear()


def signed_principal(secret: str, username: str, tenant: str) -> str:
    now = int(time.time())
    payload = {"typ": "public", "sub": username, "tid": tenant, "iat": now, "exp": now + 43200, "jti": str(uuid.uuid4())}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(secret.encode(), ("evomind.public.session.v2:" + encoded).encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return "v2." + encoded + "." + signature


def run_acceptance(args) -> int:
    stage = Path(args.stage_root).resolve()
    stage.relative_to((PRODUCTION / "staging").resolve())
    if not stage.name.startswith("invitation-"):
        raise AcceptanceError("staging_root_rejected")
    result_path = stage / "acceptance-result.json"
    if result_path.exists():
        raise AcceptanceError("acceptance_result_already_exists")
    before = production_snapshot()
    checks = []
    services = None
    result = {"schema": "evomind.invitation_server_acceptance.v1", "status": "running", "build_id": args.build_id,
              "model_execution": "fixture_only", "hpc_accessed": False, "production_deployed": False,
              "real_training_verified": False, "browser_e2e": "pending", "production_before": before}

    def check(name, passed, **evidence):
        checks.append({"name": name, "passed": bool(passed), **evidence})

    try:
        web_manifest = extract(stage / "web.zip", args.web_sha256, stage / "web", "operational-overlay-manifest.json")
        runtime_manifest = extract(stage / "runtime.zip", args.runtime_sha256, stage / "runtime", "runtime-hotfix-manifest.json")
        source_hash = sha(stage / "web/release-source-manifest.json")
        check("candidate_identity", web_manifest.get("overlay_id") == args.build_id and runtime_manifest.get("source_manifest_sha256") == source_hash,
              web_sha256=args.web_sha256, runtime_sha256=args.runtime_sha256, runtime_file_count=runtime_manifest["file_count"])
        if not checks[-1]["passed"]:
            raise AcceptanceError("candidate_identity_failed")
        fixture = stage / "fixture-schema.sqlite"
        if sha(fixture) != args.fixture_sha256:
            raise AcceptanceError("fixture_schema_hash_mismatch")
        connection = sqlite3.connect(fixture.as_uri() + "?mode=ro", uri=True)
        try:
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
                name = str(row[0]).replace('"', '""')
                if connection.execute('SELECT count(*) FROM "' + name + '"').fetchone()[0]:
                    raise AcceptanceError("fixture_database_contains_rows")
        finally:
            connection.close()
        (stage / "workspace").mkdir()
        shutil.copyfile(fixture, stage / "workspace/workstation.sqlite")
        session_secret, password = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        identity = json.loads((stage / "web/runtime-build-manifest.json").read_text())["source_tree_sha256"]
        environment = fixture_environment(stage, session_secret, password, args.build_id, identity)
        services = FixtureServices(stage, environment, args.node)
        services.start()
        port = services.web_port
        origin = "http://127.0.0.1:" + str(port)
        status, _, health = request(port, "GET", "/api/healthz")
        check("web_health_identity", status == 200 and health.get("build_id") == args.build_id and health.get("status") == "ready", http_status=status)
        if not checks[-1]["passed"]:
            raise AcceptanceError("web_health_identity_failed")
        status, _, _ = request(port, "GET", "/api/assistant/runs")
        check("unauthenticated_denied", status == 401, http_status=status)
        status, headers, login = request(port, "POST", "/api/auth/login", body={"username": "invitation_acceptance", "password": password}, headers={"Origin": origin})
        check("real_login_endpoint", status == 200, http_status=status)
        if status != 200 or "set-cookie" not in headers:
            raise AcceptanceError("fixture_login_failed")
        cookie = headers["set-cookie"].split(";", 1)[0]
        auth = {"Cookie": cookie, "Origin": origin}
        status, _, principal = request(port, "GET", "/api/session/status", headers=auth)
        check("session_status", status == 200 and principal.get("authenticated") is True, http_status=status)
        auth[CSRF] = principal["csrf_token"]
        status, _, super_agent = request(port, "GET", "/api/super-agent/status", headers=auth)
        check("super_agent_scope_preserved", status == 403 and super_agent.get("error") == "scoped_status_unavailable", http_status=status)
        status, _, _ = request(port, "POST", "/api/assistant/runs", body={"prompt": "ISOLATED_ACCEPTANCE_ECHO"}, headers={"Cookie": cookie, "Origin": origin})
        check("missing_csrf_denied", status == 403, http_status=status)
        status, _, _ = request(port, "POST", "/api/runtime/sessions", body={"objective": "ISOLATED_ACCEPTANCE_ECHO", "permission_level": "full_access"}, headers=auth)
        check("generic_mutation_denied", status == 403, http_status=status)
        data = b"feature,target\n0,0\n1,1\n"
        digest = hashlib.sha256(data).hexdigest()
        status, _, upload = request(port, "POST", "/api/assistant/uploads", body={"name": "fixture.csv", "total_bytes": len(data), "media_type": "text/csv", "sha256": digest}, headers=auth)
        check("upload_created", status == 201, http_status=status)
        upload_id = upload.get("id") or upload.get("upload_id")
        if not upload_id:
            raise AcceptanceError("upload_identity_missing")
        tenant = "tenant_" + "b" * 24
        foreign_cookie = COOKIE + "=" + signed_principal(session_secret, "foreign_fixture", tenant)
        foreign = {"Cookie": foreign_cookie, "Origin": origin}
        _, _, foreign_status = request(port, "GET", "/api/session/status", headers=foreign)
        foreign[CSRF] = foreign_status["csrf_token"]
        chunk_path = f"/api/assistant/uploads/{upload_id}/chunks/0"
        binary = {**auth, "Content-Type": "application/octet-stream", "X-Chunk-SHA256": digest}
        status, _, _ = request(port, "PUT", chunk_path, body=data, headers={**binary, **foreign})
        check("foreign_upload_denied", status == 404, http_status=status)
        status, _, _ = request(port, "PUT", chunk_path, body=data, headers=binary)
        check("chunk_uploaded", status == 200, http_status=status)
        complete_path = f"/api/assistant/uploads/{upload_id}/complete"
        status, _, attachment = request(port, "POST", complete_path, body={}, headers=auth)
        check("upload_verified", status == 200, http_status=status)
        attachment_id = (attachment.get("attachment") or attachment).get("id") or attachment.get("attachment_id")
        _, _, repeated = request(port, "POST", complete_path, body={}, headers=auth)
        repeated_id = (repeated.get("attachment") or repeated).get("id") or repeated.get("attachment_id")
        check("upload_completion_idempotent", attachment_id == repeated_id)
        body = {"prompt": "ISOLATED_ACCEPTANCE_ECHO", "attachment_ids": [attachment_id], "idempotency_key": "isolated-create-once"}
        status, _, created = request(port, "POST", "/api/assistant/runs", body=body, headers={**auth, "Idempotency-Key": "isolated-create-once"})
        check("run_created", status == 201, http_status=status)
        run_id = created.get("id")
        if not run_id:
            raise AcceptanceError("run_identity_missing")
        _, _, repeat = request(port, "POST", "/api/assistant/runs", body=body, headers={**auth, "Idempotency-Key": "isolated-create-once"})
        check("run_creation_idempotent", repeat.get("id") == run_id)
        changed_status, _, _ = request(port, "POST", "/api/assistant/runs", body={**body, "prompt": "ISOLATED_ACCEPTANCE_ECHO_CHANGED"}, headers={**auth, "Idempotency-Key": "isolated-create-once"})
        check("changed_idempotent_payload_denied", changed_status == 409, http_status=changed_status)
        foreign_body = {"prompt": "ISOLATED_ACCEPTANCE_ECHO_FOREIGN", "idempotency_key": "isolated-create-once"}
        foreign_create_status, _, foreign_created = request(port, "POST", "/api/assistant/runs", body=foreign_body, headers=foreign)
        check("principal_scoped_request_keys", foreign_create_status == 201 and bool(foreign_created.get("id")) and foreign_created.get("id") != run_id)
        check("independent_default_conversations", bool(created.get("conversation_id")) and bool(foreign_created.get("conversation_id")) and created.get("conversation_id") != foreign_created.get("conversation_id"))
        snapshot = {}
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            status, _, snapshot = request(port, "GET", "/api/assistant/runs/" + run_id, headers=auth)
            if snapshot.get("terminal"):
                break
            time.sleep(0.1)
        check("fixture_run_completed", status == 200 and snapshot.get("status") == "completed")
        check("no_default_task_context", snapshot.get("selected_task", "") == "")
        status, _, _ = request(port, "GET", "/api/assistant/runs/" + run_id, headers=foreign)
        check("foreign_run_denied", status == 404, http_status=status)
        status, denied_stream = request_sse(port, "/api/assistant/runs/" + run_id + "/events", foreign)
        denied_frames = denied_stream["frames"]
        stream_denied = status in {401, 403, 404} or (
            status == 200 and bool(denied_frames) and all(frame["id"] is None for frame in denied_frames)
            and any(frame["event"] == "run_failed" and frame["data"].get("status") == "failed" for frame in denied_frames)
        )
        check("foreign_run_events_denied", stream_denied, http_status=status, persisted_events=sum(frame["id"] is not None for frame in denied_frames))
        cursor = snapshot.get("last_event_seq", 0)
        status, events = request_sse(port, f"/api/assistant/runs/{run_id}/events?after={cursor}", {**auth, "Last-Event-ID": str(cursor)})
        check("event_cursor_no_replay", status == 200 and events["is_sse"] and all(frame["id"] is not None and frame["id"] > cursor for frame in events["frames"]), frames=len(events["frames"]), keepalive_seen=events.get("keepalive_seen", False))
        artifacts = snapshot.get("artifacts") or []
        check("fixture_artifact_published", bool(artifacts))
        artifact_path = None
        if artifacts:
            artifact_path = "/api/assistant/artifacts/" + artifacts[0]["id"] + "?download=1"
            status, _, downloaded = request(port, "GET", artifact_path, headers=auth)
            check("artifact_bytes_verified", status == 200 and isinstance(downloaded, bytes) and hashlib.sha256(downloaded).hexdigest() == artifacts[0]["sha256"])
            status, _, _ = request(port, "GET", artifact_path, headers=foreign)
            check("foreign_artifact_denied", status == 404, http_status=status)
        def timed(_):
            start = time.monotonic()
            status, _, _ = request(port, "GET", "/api/assistant/runs/" + run_id, headers=auth)
            return status, (time.monotonic() - start) * 1000
        with ThreadPoolExecutor(max_workers=5) as pool:
            latency = list(pool.map(timed, range(30)))
        p95 = sorted(value for _, value in latency)[28]
        check("readonly_five_client_latency", all(status == 200 for status, _ in latency) and p95 <= 1000, p95_ms=round(p95, 3), samples=30, scope="one authenticated fixture principal, five clients")
        services.stop()
        services.start()
        status, _, restored = request(port, "GET", "/api/assistant/runs/" + run_id, headers=auth)
        check("restart_restores_same_run", status == 200 and restored.get("id") == run_id and restored.get("status") == snapshot.get("status") and restored.get("last_event_seq") == cursor)
        if artifact_path:
            status, _, downloaded = request(port, "GET", artifact_path, headers=auth)
            check("restart_restores_artifact", status == 200 and isinstance(downloaded, bytes) and hashlib.sha256(downloaded).hexdigest() == artifacts[0]["sha256"])
        status, _, _ = request(port, "GET", "/api/assistant/runs/" + run_id, headers=foreign)
        check("restart_preserves_owner_scope", status == 404, http_status=status)
        status, _, replayed = request(port, "POST", "/api/assistant/runs", body=body, headers=auth)
        check("restart_preserves_request_key", status == 200 and replayed.get("id") == run_id, http_status=status)
        connection = sqlite3.connect((stage / "workspace/workspace/runtime/runtime.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            invocations = connection.execute("SELECT count(*) FROM events WHERE session_id=? AND event_type='acceptance_fixture_invoked'", (run_id,)).fetchone()[0]
        finally:
            connection.close()
        check("single_fixture_invocation", invocations == 1, invocations=invocations)
        result["fixture_run_id"] = run_id
    except Exception as error:
        result["error_class"] = type(error).__name__
        result["error"] = str(error) if re.fullmatch(r"[a-z0-9_:-]{1,120}", str(error)) else "diagnostic_withheld"
    finally:
        if services is not None:
            try:
                services.stop()
            except Exception:
                result["cleanup_error"] = "owned_fixture_cleanup_failed"
            result["owned_fixture_process_terminations"] = services.terminated_owned_children
            result["launch_records"] = services.launch_records
        after = production_snapshot()
        check("production_unchanged", after == before)
        result["production_after"] = after
        result["checks"] = checks
        result["status"] = "passed" if checks and all(row["passed"] for row in checks) and "error" not in result and "cleanup_error" not in result else "failed"
        result["harness_sha256"] = sha(Path(__file__))
        result["logs"] = [{"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(stage.glob("*.log"))]
        write_json(result_path, result)
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="role", required=True)
    backend_parser = sub.add_parser("backend")
    backend_parser.add_argument("--stage-root", required=True)
    backend_parser.add_argument("--port", type=int, default=0)
    backend_parser.add_argument("--boot-id", required=True)
    controller = sub.add_parser("verify")
    controller.add_argument("--stage-root", required=True)
    controller.add_argument("--web-sha256", required=True)
    controller.add_argument("--runtime-sha256", required=True)
    controller.add_argument("--fixture-sha256", required=True)
    controller.add_argument("--build-id", required=True)
    controller.add_argument("--node", default="C:/Program Files/nodejs/node.exe")
    args = parser.parse_args()
    if args.role == "backend":
        return backend(Path(args.stage_root).resolve(), args.port, args.boot_id)
    return run_acceptance(args)


if __name__ == "__main__":
    raise SystemExit(main())
