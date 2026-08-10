from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "research-agent-workstation"
ARTIFACT_ROOT = ROOT / "artifacts" / "launch-fix-20260728-170231" / "functional"
PORT = 18091
ORIGIN = f"http://127.0.0.1:{PORT}"
SESSION_COOKIE = ""
CSRF_TOKEN = ""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def request_json(method: str, path: str, body: dict | None = None, expected: int = 200) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if SESSION_COOKIE:
        headers["Cookie"] = SESSION_COOKIE
    if method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        headers["Origin"] = ORIGIN
        if CSRF_TOKEN:
            headers["x-evomind-csrf"] = CSRF_TOKEN
    request = urllib.request.Request(
        f"{ORIGIN}{path}",
        data=payload,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        status = error.code
        result = json.loads(error.read().decode("utf-8"))
    require(status == expected, f"{method} {path}: expected HTTP {expected}, got {status}: {result}")
    return result


def wait_ready(log_path: Path, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(f"{ORIGIN}/api/healthz", method="GET")
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    status = response.status
                    health = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                status = error.code
                health = json.loads(error.read().decode("utf-8"))
            require(status in {200, 503}, f"Unexpected health status: {status}")
            require(health.get("service") == "evomind-workstation", f"Unexpected health payload: {health}")
            return
        except Exception as error:  # noqa: BLE001 - readiness captures transient startup failures
            last_error = str(error)
            time.sleep(0.5)
    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:] if log_path.exists() else ""
    raise RuntimeError(f"Next server did not become ready: {last_error}\n{log_tail}")


def bootstrap_session(token: str) -> None:
    global SESSION_COOKIE, CSRF_TOKEN
    payload = json.dumps({"token": token}).encode("utf-8")
    request = urllib.request.Request(
        f"{ORIGIN}/api/session/bootstrap",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Origin": ORIGIN},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
        set_cookie = response.headers.get("Set-Cookie", "")
    require(response.status == 200 and result.get("ok") is True, f"Session bootstrap failed: {result}")
    cookie_pair = set_cookie.split(";", 1)[0].strip()
    require(cookie_pair.startswith("evomind_local_session=") and len(cookie_pair) > 32, "Session bootstrap did not set the expected cookie")
    csrf_token = result.get("csrf_token")
    require(isinstance(csrf_token, str) and 24 <= len(csrf_token) <= 256, "Session bootstrap returned invalid CSRF state")
    SESSION_COOKIE = cookie_pair
    CSRF_TOKEN = csrf_token
    status = request_json("GET", "/api/session/status")
    require(status.get("ok") is True and status.get("authenticated") is True, f"Session verification failed: {status}")


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def static_contract() -> dict:
    tasks = (WEB / "src/components/workstation/screens/TasksScreen.tsx").read_text(encoding="utf-8")
    code = (WEB / "src/components/workstation/screens/CodeAgentScreen.tsx").read_text(encoding="utf-8")
    shell = (WEB / "src/components/workstation/AppShell.tsx").read_text(encoding="utf-8")
    actions = (WEB / "src/lib/server/workstation-actions.ts").read_text(encoding="utf-8")
    lifecycle = (WEB / "src/app/api/tasks/[taskId]/route.ts").read_text(encoding="utf-8")
    require('action: "create_workstation_run"' in shell, "Tasks Create Run must map to the real action")
    require('action: "dispatch_task_agents"' in shell, "Tasks Dispatch must map to real local multi-agent dispatch")
    require('data-ui-action="tasks_create_workstation_run"\n              data-ui-skip-action' not in tasks, "Create Run still bypasses AppShell")
    require('data-ui-action="tasks_dispatch_agents"\n              data-ui-skip-action' not in tasks, "Dispatch still bypasses AppShell")
    for token in (
        'fetch(`/api/tasks/${encodeURIComponent(props.selectedTask)}/code-agent-draft`',
        'executeCodeAction("code_add_new_file")',
        'executeCodeAction("run_code_smoke_test")',
        'executeCodeAction("request_code_quality_gate")',
        'executeCodeAction("apply_agent_patch")',
        'executeCodeAction("rollback_agent_patch")',
    ):
        require(token in code, f"Code screen contract missing: {token}")
    require("applied_logical_only: false" in actions, "Patch apply remains logical-only")
    require("applyPatchTransaction" in actions and "rollbackPatchTransaction" in actions, "Atomic patch lifecycle is missing")
    require("export async function PATCH" in lifecycle and "export async function DELETE" in lifecycle, "Task lifecycle methods are missing")
    return {"tasks": True, "code": True, "patch_transaction": True, "task_lifecycle": True}


def copy_test_app(destination: Path) -> None:
    ignored = shutil.ignore_patterns("node_modules", ".next", ".next-*", ".backups")
    shutil.copytree(WEB, destination, ignore=ignored)
    junction = destination / "node_modules"
    command = ["cmd", "/c", "mklink", "/J", str(junction), str(WEB / "node_modules")]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    require(result.returncode == 0 and junction.exists(), f"node_modules junction failed: {result.stdout} {result.stderr}")


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    result: dict = {"static_contract": static_contract(), "steps": []}
    runtime_root = Path(tempfile.mkdtemp(prefix="launchfix-functional-", dir=ARTIFACT_ROOT))
    app_root = runtime_root / "app"
    data_root = runtime_root / "workstation"
    data_root.mkdir(parents=True)
    copy_test_app(app_root)
    database = data_root / "workstation.db"
    bootstrap_token = secrets.token_urlsafe(32)
    session_secret = secrets.token_urlsafe(48)
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_URL": f"file:{database.as_posix()}",
            "WORKSTATION_ROOT": str(data_root),
            "WORKSTATION_PYTHON": sys.executable,
            "WORKSTATION_DISABLE_AGENT_EXECUTION": "1",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "DEEPSEEK_API_KEY": "",
            "KAGGLE_USERNAME": "",
            "KAGGLE_KEY": "",
            "NEXT_TELEMETRY_DISABLED": "1",
            "NODE_ENV": "development",
            "PORT": str(PORT),
            "WORKSTATION_SESSION_SECRET": session_secret,
            "WORKSTATION_BOOTSTRAP_TOKEN_HASH": hashlib.sha256(bootstrap_token.encode("utf-8")).hexdigest(),
        }
    )
    prisma = subprocess.run(
        ["node", str(WEB / "node_modules/prisma/build/index.js"), "db", "push", "--skip-generate", "--schema", str(app_root / "prisma/schema.prisma")],
        cwd=app_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    require(prisma.returncode == 0, f"prisma db push failed: {prisma.stdout}\n{prisma.stderr}")
    log_path = ARTIFACT_ROOT / "isolated-next.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["node", str(app_root / "node_modules/next/dist/bin/next"), "dev", "--webpack", "--hostname", "127.0.0.1", "--port", str(PORT)],
        cwd=app_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    task_id = "launchfix_functional_task"
    local_objective = "Do not use HPC. Run this local-only isolated dispatch contract verification."
    try:
        wait_ready(log_path)
        bootstrap_session(bootstrap_token)
        bootstrap_token = ""
        session_secret = ""
        created = request_json("POST", "/api/workstation-actions", {"action": "create_workstation_run", "task_id": task_id, "metadata": {"trigger": "launchfix_verifier", "objective": local_objective}})
        require(created.get("ok") is True and created.get("run_id"), f"Create Run failed: {created}")
        result["steps"].append({"step": "create_run", "run_id": created["run_id"], "ok": True})

        dispatched = request_json("POST", "/api/workstation-actions", {"action": "dispatch_task_agents", "task_id": task_id, "metadata": {"objective": local_objective}})
        require(dispatched.get("ok") is True and dispatched.get("execution_started") is False, f"Isolated dispatch failed: {dispatched}")
        result["steps"].append({"step": "dispatch_agents", "run_id": dispatched["run_id"], "status": dispatched["status"], "external_execution": False})

        draft = request_json("POST", f"/api/tasks/{task_id}/code-agent-draft", {"source_agent": "local_template"})
        patch_path = str(draft.get("patch_path", "")).replace("\\", "/")
        applied_target = str(draft.get("applied_target_path", "")).replace("\\", "/")
        require(patch_path.endswith(".diff") and applied_target.endswith(".py"), f"Draft patch contract failed: {draft}")
        result["steps"].append({"step": "ask_code_agent", "patch_path": patch_path, "target": applied_target})

        reviewed = request_json("POST", "/api/workstation-actions", {"action": "review_agent_patch", "task_id": task_id, "metadata": {"patch_path": patch_path, "source_agent": "local_template"}})
        require(reviewed.get("quality_status") == "passed", f"Quality gate failed: {reviewed}")
        smoke = request_json("POST", "/api/workstation-actions", {"action": "run_code_smoke_test", "task_id": task_id, "metadata": {"patch_path": patch_path}})
        require(smoke.get("ok") is True and smoke.get("status") == "passed", f"Code smoke failed: {smoke}")
        result["steps"].append({"step": "smoke_and_quality", "quality": "passed", "smoke": "passed"})

        applied = request_json("POST", "/api/workstation-actions", {"action": "apply_agent_patch", "task_id": task_id, "metadata": {"source_agent": "local_template"}})
        target = data_root / Path(applied_target)
        require(applied.get("ok") is True and target.is_file(), f"Patch was not applied: {applied}")
        applied_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        rolled_back = request_json("POST", "/api/workstation-actions", {"action": "rollback_agent_patch", "task_id": task_id, "metadata": {"source_agent": "launchfix_verifier"}})
        require(rolled_back.get("ok") is True and not target.exists(), f"Patch rollback failed: {rolled_back}")
        result["steps"].append({"step": "apply_and_rollback", "transaction_id": applied["transaction_id"], "applied_sha256": applied_hash, "target_removed": True})

        archived = request_json("PATCH", f"/api/tasks/{task_id}", {"action": "archive"})
        require(archived.get("archived") is True, f"Archive failed: {archived}")
        visible = request_json("GET", "/api/tasks")
        require(task_id not in {task["id"] for task in visible["tasks"]}, "Archived task remained visible")
        archived_list = request_json("GET", "/api/tasks?include_archived=1")
        require(task_id in {task["id"] for task in archived_list["tasks"]}, "Archived task was not recoverable")
        restored = request_json("PATCH", f"/api/tasks/{task_id}", {"action": "restore"})
        require(restored.get("archived") is False, f"Restore failed: {restored}")
        request_json("PATCH", f"/api/tasks/{task_id}", {"action": "archive"})
        purged = request_json("DELETE", f"/api/tasks/{task_id}", {"confirm_task_id": task_id})
        require(purged.get("purged") is True and purged.get("zero_generated_residual") is True and not purged.get("cleanup_pending"), f"Purge failed: {purged}")
        request_json("GET", f"/api/tasks/{task_id}", expected=404)
        result["steps"].append({"step": "archive_restore_purge", "tombstone": purged["tombstone"], "zero_generated_residual": True})

        traversal = request_json("GET", "/api/tasks/..%2F..%2Fescape", expected=400)
        require(traversal.get("ok") is False, "Traversal task id was not rejected")

        connection = sqlite3.connect(database)
        try:
            table_counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table} WHERE task_id = ?", (task_id,)).fetchone()[0]
                for table in ("experiment_runs", "workflows", "gates", "evidence", "reports")
            }
            task_count = connection.execute("SELECT COUNT(*) FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]
            tombstone_count = connection.execute("SELECT COUNT(*) FROM action_logs WHERE action = 'purge_task' AND metadata_json LIKE ?", (f"%{task_id}%",)).fetchone()[0]
            action_ids = [row[0] for row in connection.execute("SELECT id FROM action_logs ORDER BY created_at ASC, id ASC").fetchall()]
        finally:
            connection.close()
        require(task_count == 0 and all(value == 0 for value in table_counts.values()), f"Relational purge residual: {task_count}, {table_counts}")
        require(tombstone_count >= 1, "Purge tombstone action was not retained")
        mirror_path = data_root / "workspace" / "runtime" / "action_log.jsonl"
        mirror_records = [json.loads(line) for line in mirror_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mirror_ids = [str(record["action_id"]) for record in mirror_records]
        require(mirror_ids == action_ids, f"SQLite/JSONL audit mirror drift: sqlite={len(action_ids)} jsonl={len(mirror_ids)}")
        checkpoint = json.loads((data_root / "workspace" / "runtime" / "action_log.checkpoint.json").read_text(encoding="utf-8"))
        require(checkpoint.get("sqlite_is_canonical") is True, "Audit checkpoint did not declare SQLite canonical")
        require(checkpoint.get("action_count") == len(action_ids), "Audit checkpoint count drift")
        require(checkpoint.get("last_action_id") == (action_ids[-1] if action_ids else None), "Audit checkpoint tail drift")
        result["steps"].append({"step": "audit_mirror_consistency", "action_count": len(action_ids), "exact_id_order": True})
        result["database"] = {"task": task_count, **table_counts, "purge_tombstones": tombstone_count}
        result["ok"] = True
    finally:
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=20)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()
        deadline = time.monotonic() + 10
        while port_open(PORT) and time.monotonic() < deadline:
            time.sleep(0.2)
        result["cleanup"] = {"pid": process.pid, "port_open": port_open(PORT), "runtime_root": str(runtime_root)}
        (ARTIFACT_ROOT / "isolated_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if result.get("ok") is True and not result["cleanup"]["port_open"]:
            shutil.rmtree(runtime_root, ignore_errors=True)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
