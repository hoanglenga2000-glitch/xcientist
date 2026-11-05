"""RUN291 temporary tools (owner-approved 2026-10-22, IOAI 2026 test run 291 only; remove on rollback).

ioai_task_info / ioai_kb / ioai_status are read-only; ioai_dev_run / ioai_submit upload the Run workspace
`solution/` dir to the HPC run291 area and execute the fixed operator scripts dev291.py / submit291.py
(A split only; split B is never reachable from these tools).  Every call is gated on
operations/run291/active_runs.json (session id + task + state=active + not past deadline) and on the
tenant's active HPC binding.  GPU time is reserved in the normal GPU budget (cap 4200 s per call) inside
the shared GPU lease.  Remote work has a hard `timeout`; on overrun the process group is killed and the
call returns an error -- these tools never mark remote settlement as uncertain.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .models import ToolResult

OPS = Path(r"C:\ProgramData\EvoMind\operations\run291")
ACTIVE = OPS / "active_runs.json"
KB = OPS / "kb_frozen"
TASK_VIEWS = OPS / "tasks"
CALL_LOG = OPS / "tool_calls.jsonl"
TOOL_NAMES = frozenset({"ioai_task_info", "ioai_kb", "ioai_status", "ioai_dev_run", "ioai_submit"})
READ_ONLY = frozenset({"ioai_task_info", "ioai_kb", "ioai_status"})
TASKS = ("fto", "robot", "potato", "da", "ghost", "field")
R = "/hpc2hdd/home/aimslab/jinghw/ioai_diag_20261018/run291"
H = R + "/harness"
PY = "/hpc2hdd/home/aimslab/miniconda3/bin/python"
MAX_FILES = 64
MAX_BYTES = 50 * 1024 * 1024
SUBMIT_TIMEOUT = 3300   # submit291 itself: run <=1800 s + grading
DEV_MAX = 3600
MARGIN = 300


def _log(entry: dict) -> None:
    try:
        OPS.mkdir(parents=True, exist_ok=True)
        with open(CALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}) + "\n")
    except OSError:
        pass


def _entry(session_id: str) -> dict | None:
    try:
        data = json.loads(ACTIVE.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    item = (data.get("sessions") or {}).get(str(session_id or ""))
    if not isinstance(item, dict) or item.get("state") != "active" or item.get("task") not in TASKS:
        return None
    return item


def gate_reason(session: dict, scope: dict, tool_name: str) -> str:
    """Called from personal_tool_boundary.constrain for ioai_* tools."""
    item = _entry(str(session.get("id") or ""))
    meta = session.get("metadata") or {}
    if item is None:
        return "run291 tools are only available to registered active run291 sessions"
    if str(meta.get("run291_task") or "") != item["task"]:
        return "run291 task binding mismatch"
    if tool_name in {"ioai_dev_run", "ioai_submit"}:
        from .personal_tool_boundary import _binding_matches
        reason = _binding_matches(session, scope)
        if reason:
            return reason
        if time.time() > float(item.get("deadline_epoch") or 0):
            return "run291 task window (2 h) has ended; no new runs"
    return ""


def _fail(msg: str, code: str, **content: Any) -> ToolResult:
    return ToolResult("", False, content, msg, error=code)


def _ctx_entry(context) -> dict | None:
    item = _entry(context.session_id)
    if item is None or str((context.metadata or {}).get("run291_task") or "") != item["task"]:
        return None
    return item


def _manifest() -> dict[str, str]:
    out = {}
    for line in (KB / "MANIFEST.md5").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            out[rel.strip().removeprefix("./")] = digest
    return out


def _task_info(args: dict[str, Any], context) -> ToolResult:
    item = _ctx_entry(context)
    if item is None:
        return _fail("not a registered run291 session", "run291_gate")
    task = item["task"]
    part = str(args.get("part") or "contract")
    name = {"contract": "CONTRACT.md", "statement": "statement.md"}.get(part)
    if not name:
        return _fail("part must be contract or statement", "invalid_arguments")
    text = (TASK_VIEWS / task / name).read_text(encoding="utf-8")
    off = max(0, int(args.get("offset") or 0))
    chunk = text[off:off + 40000]
    left = max(0.0, float(item.get("deadline_epoch") or 0) - time.time())
    return ToolResult("", True, {"task": task, "part": part, "offset": off, "total_chars": len(text), "text": chunk,
                                 "seconds_left_in_task_window": round(left)}, f"{task} {part}")


def _kb(args: dict[str, Any], context) -> ToolResult:
    item = _ctx_entry(context)
    if item is None:
        return _fail("not a registered run291 session", "run291_gate")
    man = _manifest()
    rel = str(args.get("path") or "").replace("\\", "/").strip("/")
    if rel and rel not in man:  # runtime may rewrite "path" args to absolute workspace paths: map back by unique manifest suffix
        cands = [k for k in man if rel.endswith("/" + k)]
        if len(cands) == 1:
            rel = cands[0]
    if not rel:
        index = (KB / "INDEX.md").read_text(encoding="utf-8")
        return ToolResult("", True, {"index": index, "files": sorted(man)}, "run291 KB index")
    if rel not in man or ".." in PurePosixPath(rel).parts:
        return _fail("unknown KB path; call ioai_kb without path for the file list", "kb_path_unknown")
    data = (KB / Path(*PurePosixPath(rel).parts)).read_bytes()
    if hashlib.md5(data).hexdigest() != man[rel]:
        return _fail("KB file failed its frozen md5 check", "kb_integrity")
    text = data.decode("utf-8", errors="replace")
    off = max(0, int(args.get("offset") or 0))
    return ToolResult("", True, {"path": rel, "offset": off, "total_chars": len(text), "text": text[off:off + 40000]}, f"KB {rel}")


def _connect(context):
    from .tools import _load_bound_hpc_config, _required_managed_hpc_identity
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    job_id, profile = _required_managed_hpc_identity({}, context)
    config = _load_bound_hpc_config(profile, job_id, context.session_id)
    def connect():
        client = connect_ssh(config, timeout=30)
        try:
            client.get_transport().set_keepalive(30)
        except Exception:
            pass
        return client
    return config, job_id, profile, connect


def _sh(client, cmd: str, timeout: int = 60) -> tuple[int, str]:
    _i, so, _e = client.exec_command(cmd + " 2>&1", timeout=timeout)
    text = so.read().decode("utf-8", errors="replace")
    return so.channel.recv_exit_status(), text


def _collect(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise ValueError("write your code under the workspace folder solution/ first")
    files, total = [], 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("symlinks are not allowed in solution/")
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if not re.fullmatch(r"[A-Za-z0-9_.\-/]{1,200}", rel) or rel.startswith("."):
            raise ValueError(f"unsupported file name in solution/: {rel}")
        total += p.stat().st_size
        files.append((p, rel))
    if not files or len(files) > MAX_FILES or total > MAX_BYTES:
        raise ValueError(f"solution/ must hold 1..{MAX_FILES} files, <= {MAX_BYTES} bytes")
    return files


def _execute(kind: str, args: dict[str, Any], context) -> ToolResult:
    item = _ctx_entry(context)
    if item is None:
        return _fail("not a registered run291 session", "run291_gate")
    left = float(item.get("deadline_epoch") or 0) - time.time()
    if left <= 0:
        return _fail("run291 task window has ended; no new runs", "run291_window_closed")
    task = item["task"]
    if kind == "dev":
        inner = int(args.get("timeout_seconds") or 1800)
        if not 60 <= inner <= DEV_MAX:
            return _fail(f"timeout_seconds must be 60..{DEV_MAX}", "invalid_arguments")
        hard = inner + 600   # build + grading head-room
    else:
        inner = 1800
        hard = SUBMIT_TIMEOUT
    reserve = hard + MARGIN
    try:
        files = _collect(context.workspace_root / "solution")
    except ValueError as exc:
        return _fail(str(exc), "solution_invalid")
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    base = f"{R}/agent/{task}/{context.session_id[:24]}/{kind}_{stamp}"
    code_dir, out_dir = base + "/code", base + "/out"
    from .training_control import load_policy
    from .research_budget import GpuExecutionBudget
    from .execution_progress import queued_resource
    from .models import new_id
    try:
        config, job_id, profile, connect = _connect(context)
    except Exception as exc:
        return _fail("managed HPC binding unavailable: " + type(exc).__name__, "profile_job_binding_invalid")
    ledger = GpuExecutionBudget(str(context.runtime_root / "gpu_budget.sqlite3"), load_policy(context.runtime_root)["study_id"])
    try:
        reservation = ledger.reserve(context.session_id, reserve, max_seconds=4200)
    except (ValueError, RuntimeError) as exc:
        return _fail("GPU budget rejected the run: " + str(exc), str(exc), gpu_budget=ledger.summary())
    resource = "gpu:" + str(getattr(config, "expected_gpu_uuid", "") or f"{job_id}:{profile}")
    started = time.monotonic()
    result_line, status, pid = None, "unknown", ""
    _log({"session": context.session_id, "task": task, "kind": kind, "remote": base, "files": [r for _, r in files], "reserve": reserve})
    try:
        with queued_resource(context.runtime_root / "resource_leases", resource, context.session_id + ":" + new_id("lease")):
            client = connect()
            try:
                _sh(client, "mkdir -p " + shlex.quote(code_dir) + " " + shlex.quote(out_dir))
                sftp = client.open_sftp()
                try:
                    made = set()
                    for path, rel in files:
                        parent = str(PurePosixPath(code_dir, rel).parent)
                        if parent not in made:
                            _sh(client, "mkdir -p " + shlex.quote(parent)); made.add(parent)
                        sftp.put(str(path), f"{code_dir}/{rel}")
                finally:
                    sftp.close()
                if kind == "dev":
                    inner_cmd = (f"{PY} {H}/dev291.py --task {task} --code {shlex.quote(code_dir)} "
                                 f"--out {shlex.quote(out_dir)} --timeout {inner}")
                else:
                    inner_cmd = (f"{PY} {H}/submit291.py --task {task} --sub {shlex.quote(code_dir)} "
                                 f"--run-id {shlex.quote(context.session_id)} --label {shlex.quote(str(args.get('label') or '')[:80])}")
                launch = (f"cd {H} || exit 1; setsid nohup timeout -k 30 {hard} {inner_cmd} > {shlex.quote(base)}/stdout.txt 2>&1 "
                          f"< /dev/null & echo PID=$!")
                rc, text = _sh(client, launch)
                m = re.search(r"PID=(\d+)", text)
                if not m:
                    raise RuntimeError("launch failed")
                pid = m.group(1)
            finally:
                client.close()
            deadline = started + hard + 120
            client = None
            while True:
                time.sleep(15)
                try:
                    if client is None:
                        client = connect()
                    _rc, text = _sh(client, f"kill -0 {pid} 2>/dev/null && echo RUNNING || echo DONE; "
                                            f"grep -a -m1 -E '^(DEV|SUBMIT)291_RESULT ' {shlex.quote(base)}/stdout.txt || true")
                except Exception:
                    try:
                        client and client.close()
                    except Exception:
                        pass
                    client = None
                    if time.monotonic() > deadline + 300:
                        status = "lost_connection"
                        break
                    continue
                line = next((l for l in text.splitlines() if l.startswith(("DEV291_RESULT ", "SUBMIT291_RESULT "))), None)
                if line:
                    result_line, status = line, "done"
                    break
                if text.splitlines()[:1] == ["DONE"]:
                    status = "exited_without_result"
                    break
                if time.monotonic() > deadline:
                    _sh(client, f"pkill -KILL -s {pid} || kill -KILL -{pid} || true")
                    status = "killed_overrun"
                    break
            if client is not None:
                try:
                    if status != "done":
                        _rc, tail = _sh(client, f"tail -c 2000 {shlex.quote(base)}/stdout.txt")
                        result_line = result_line or ("TAIL " + tail)
                finally:
                    client.close()
    except Exception as exc:
        status = "error:" + type(exc).__name__
    finally:
        elapsed = min(time.monotonic() - started, float(reserve))
        ledger.settle(reservation["id"], elapsed, uncertain=False, success=status == "done")
    _log({"session": context.session_id, "task": task, "kind": kind, "remote": base, "status": status,
          "elapsed_s": round(time.monotonic() - started, 1)})
    if status != "done" or not result_line:
        return _fail(f"{kind} run did not return a result ({status})", "run291_" + status.split(":")[0],
                     remote_dir=base, detail=(result_line or "")[-2000:])
    payload = json.loads(result_line.split(" ", 1)[1])
    payload["remote_dir"] = base
    return ToolResult("", True, payload, f"{task} {kind}: " + ("score=" + str(payload.get('dev_score') if kind == 'dev' else payload.get('A_score'))))


def _status(args: dict[str, Any], context) -> ToolResult:
    item = _ctx_entry(context)
    if item is None:
        return _fail("not a registered run291 session", "run291_gate")
    task = item["task"]
    try:
        _cfg, _j, _p, connect = _connect(context)
        client = connect()
        try:
            _rc, text = _sh(client, f"cat {H}/ledger/{task}.jsonl 2>/dev/null || true")
        finally:
            client.close()
    except Exception as exc:
        return _fail("status unavailable: " + type(exc).__name__, "run291_status")
    rows = []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if str(r.get("sid", "")).startswith("eq"):
            continue
        rows.append({k: r.get(k) for k in ("sid", "label", "score", "wall_s", "grader_elapsed_s", "r4_pass", "counted", "error", "reason", "finished")})
    caps = {t: 50 for t in TASKS}  # run291 registered: 50 scored submissions per task
    left = max(0.0, float(item.get("deadline_epoch") or 0) - time.time())
    return ToolResult("", True, {"task": task, "submissions": rows, "counted": sum(1 for r in rows if r["counted"]),
                                 "cap": caps[task], "seconds_left_in_task_window": round(left)}, f"{task} status")


def register(add) -> None:
    add("ioai_task_info", "run291: read this task's official statement or execution contract (I/O, env vars, limits).",
        {"part": {"type": "string", "enum": ["contract", "statement"]}, "offset": {"type": "integer", "minimum": 0}},
        [], "ioai.read", _task_info)
    add("ioai_kb", "run291: read the frozen IOAI experience knowledge base (contains our own task-specific code). No path = index.",
        {"path": {"type": "string", "maxLength": 300}, "offset": {"type": "integer", "minimum": 0}},
        [], "ioai.read", _kb)
    add("ioai_status", "run291: list this task's scored submissions, remaining cap and task-window time.",
        {}, [], "ioai.read", _status)
    add("ioai_dev_run", "run291: run workspace solution/ on PUBLIC dev data on the A800 (R=4 CPU emulation) and score it. Not counted.",
        {"timeout_seconds": {"type": "integer", "minimum": 60, "maximum": DEV_MAX}},
        [], "ioai.execute", lambda a, c: _execute("dev", a, c), read_only=False)
    add("ioai_submit", "run291: scored submission of workspace solution/ on the hidden A split. Returns A score, wall time, R4 pass, remaining cap.",
        {"label": {"type": "string", "maxLength": 80}},
        [], "ioai.execute", lambda a, c: _execute("submit", a, c), read_only=False)
