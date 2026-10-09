from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
OVERLAY_SCHEMA = "evomind.g21_goal_runtime_overlay.clean.v1"
BASELINE_MANIFEST = "runtime-hotfix-manifest.json"
GOAL_MANIFEST = "goal-runtime-overlay-manifest.json"
BOOTSTRAP_SCHEMA = "evomind.g21_goal_runtime_bootstrap.v1"
EXPECTED_R115_ZIP_SHA256 = "fc65d7134f9f53fa6eda21ee2fcd804a330d9184a83d5206a6b7469a217d89ba"
EXPECTED_R115_SOURCE_MANIFEST_SHA256 = "3a42107ba57f0ecfd04247e21561555cd5c1abc7f73e9b78b6ff7fdf13d4ed4f"
EXPECTED_R115_PRODUCTION_TREE_SHA256 = "b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a"
EXPECTED_GOAL_SPEC_FILE_SHA256 = "380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f"
EXPECTED_HUMAN_EVIDENCE_FILE_SHA256 = "5b6ed3d912ad1cd4655c96f832629cf9603a4b0ad58ab6e27c01861fadd69c5f"
EXPECTED_GOAL_BOARD_FILE_SHA256 = "7d5dcb469fb8debfd7f615957294ce46d24af2834532b7c15946737b51bde999"
EXPECTED_GOAL_BOARD_SHA256 = "85288de01557f7801804c4a5e0c86d3dbb07a4d93acba1ec9c3277835f9a4cc6"
REJECTED_CANDIDATE_ZIP_SHA256 = frozenset(
    {"37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d"}
)
DEPENDENT_FILES = (
    "evomind_runtime/__init__.py",
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
)
NEW_FILES = ("evomind_runtime/competition_goal.py", "evomind_runtime/goal_board.py")
BASELINE_RUNTIME_FILE_COUNT = 30
EXPECTED_UNCHANGED_BASELINE_FILE_COUNT = 25
EXPECTED_CHANGED_FILES = frozenset((*DEPENDENT_FILES, *NEW_FILES))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_zip_names(path: Path) -> tuple[list[str], list[str]]:
    unsafe: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if name in seen:
                duplicates.append(name)
            seen.add(name)
            parts = Path(name).parts
            if name.startswith("/") or Path(name).drive or ".." in parts:
                unsafe.append(name)
            mode = (info.external_attr >> 16) & 0xF000
            if mode == 0xA000:
                unsafe.append(name)
    return sorted(set(unsafe)), sorted(set(duplicates))


def regular_files(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def canonical_json_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def runtime_tree_sha256(files: list[dict[str, object]]) -> str:
    """Match Deploy Tree(): paths are relative to bundle/runtime/evomind_runtime."""

    prefix = "evomind_runtime/"
    rows: list[str] = []
    for item in files:
        path = str(item["path"]).replace("\\", "/")
        if not path.startswith(prefix) or path == prefix:
            raise RuntimeError(f"GOAL_CLEAN_RUNTIME_TREE_PATH_REJECTED:{path}")
        relative = path[len(prefix):]
        rows.append(f'{relative}|{int(item["bytes"])}|{item["sha256"]}')
    rows.sort()
    return sha256_bytes("\n".join(rows).encode("utf-8"))


def load_bootstrap_artifact(source_root: Path, goal_board_path: Path) -> dict[str, object]:
    """Freeze and validate the three non-secret Goal bootstrap inputs."""

    spec_path = source_root / "configs" / "g21_five_competition_goal.json"
    evidence_path = source_root / "configs" / "g21_five_competition_human_baseline_gate.json"
    paths = {
        "goal_spec": spec_path,
        "human_baseline_evidence": evidence_path,
        "initial_goal_board": goal_board_path,
    }
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"GOAL_CLEAN_BOOTSTRAP_INPUT_REJECTED:{label}")
    raw_hashes = {label: sha256_file(path) for label, path in paths.items()}
    expected_hashes = {
        "goal_spec": EXPECTED_GOAL_SPEC_FILE_SHA256,
        "human_baseline_evidence": EXPECTED_HUMAN_EVIDENCE_FILE_SHA256,
        "initial_goal_board": EXPECTED_GOAL_BOARD_FILE_SHA256,
    }
    if raw_hashes != expected_hashes:
        bad = sorted(label for label in raw_hashes if raw_hashes[label] != expected_hashes[label])
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_FILE_SHA_REJECTED:" + ",".join(bad))
    try:
        values = {label: json.loads(path.read_text(encoding="utf-8-sig")) for label, path in paths.items()}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_JSON_INVALID") from exc
    spec = values["goal_spec"]
    evidence = values["human_baseline_evidence"]
    board = values["initial_goal_board"]
    if not all(isinstance(value, dict) for value in values.values()):
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_ROOT_INVALID")
    if (
        spec.get("run_id") != FIXED_RUN_ID
        or spec.get("allocation") != FIXED_ALLOCATION
        or spec.get("human_baseline_evidence_sha256") != raw_hashes["human_baseline_evidence"]
    ):
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_SPEC_BINDING_REJECTED")
    if (
        evidence.get("run_id") != FIXED_RUN_ID
        or evidence.get("allocation") != FIXED_ALLOCATION
        or evidence.get("classification") != "HUMAN_BASELINE_UNDEFINED"
    ):
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_EVIDENCE_BINDING_REJECTED")
    competitions = board.get("competitions") if isinstance(board, dict) else None
    names = [item.get("competition") for item in competitions or () if isinstance(item, dict)]
    if (
        board.get("schema") != "evomind.goal-board.v2"
        or board.get("run_id") != FIXED_RUN_ID
        or board.get("allocation") != FIXED_ALLOCATION
        or board.get("goal_record_status") != "blocked"
        or board.get("completion_count") != 0
        or board.get("weather_actions") != 0
        or set(names) != {"cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025"}
        or len(names) != 5
        or any(item.get("goal_status") != "WAITING_EXACT_GATE" for item in competitions or ())
    ):
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_BOARD_BINDING_REJECTED")
    canonical_hashes = {label: canonical_json_sha256(value) for label, value in values.items()}
    if canonical_hashes["initial_goal_board"] != EXPECTED_GOAL_BOARD_SHA256:
        raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_BOARD_CANONICAL_SHA_REJECTED")
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "goal_record_status": "blocked",
        "goal_spec": spec,
        "goal_spec_file_sha256": raw_hashes["goal_spec"],
        "goal_spec_sha256": canonical_hashes["goal_spec"],
        "human_baseline_evidence": evidence,
        "human_baseline_evidence_file_sha256": raw_hashes["human_baseline_evidence"],
        "human_baseline_evidence_sha256": canonical_hashes["human_baseline_evidence"],
        "initial_goal_board": board,
        "goal_board_file_sha256": raw_hashes["initial_goal_board"],
        "goal_board_sha256": canonical_hashes["initial_goal_board"],
        "production_deployable": False,
    }


def smoke_candidate(root: Path, bootstrap_path: Path, round_number: int) -> dict[str, object]:
    """Exercise the unpacked candidate through its real loopback Goal HTTP API."""

    smoke = root / f"goal-overlay-smoke-{round_number}.py"
    smoke.write_text(
        """
import http.client
import json
import pathlib
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer

import evomind_runtime
from evomind_runtime.competition_goal import sha256_json, validate_goal_board_identity, validate_goal_spec
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import utc_now
from evomind_runtime.runtime import AgentRuntime

candidate_root = pathlib.Path(sys.argv[1]).resolve()
bootstrap_path = pathlib.Path(sys.argv[2]).resolve()
module_path = pathlib.Path(evomind_runtime.__file__).resolve()
assert candidate_root == module_path.parent.parent, (candidate_root, module_path)
bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8-sig"))
assert bootstrap["schema"] == "evomind.g21_goal_runtime_bootstrap.v1"
assert bootstrap["run_id"] == "run_7b1efb878afb40f396db431e91f093a5"
assert bootstrap["allocation"] == "G21"
spec = bootstrap["goal_spec"]
board = bootstrap["initial_goal_board"]
spec_valid, spec_issues = validate_goal_spec(spec)
board_valid, board_issues = validate_goal_board_identity(board)
assert spec_valid, [item.code for item in spec_issues]
assert board_valid, [item.code for item in board_issues]
assert sha256_json(spec) == bootstrap["goal_spec_sha256"]
assert sha256_json(bootstrap["human_baseline_evidence"]) == bootstrap["human_baseline_evidence_sha256"]
assert sha256_json(board) == bootstrap["goal_board_sha256"]
assert bootstrap["human_baseline_evidence"]["classification"] == "HUMAN_BASELINE_UNDEFINED"
assert board["goal_record_status"] == "blocked"


def request(port, token, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": "Bearer " + token}
    if raw is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
    try:
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload
    finally:
        connection.close()


workspace = pathlib.Path(tempfile.mkdtemp(prefix="goal-overlay-http-smoke-"))
runtime = AgentRuntime(workspace)
runtime.create_session(
    session_id=board["run_id"],
    workspace_root=str(workspace),
    objective="fixed G21 Goal candidate HTTP smoke",
)
now = utc_now()
runtime.store.create_assistant_run({
    "id": board["run_id"], "session_id": board["run_id"], "conversation_id": "goal-overlay-http-smoke",
    "prompt": "fixed G21 Goal candidate HTTP smoke", "task_root": str(workspace), "status": "recovering", "plan": {},
    "attachment_ids": [], "retry_count": 0, "error_class": "", "error_message": "",
    "model_provider": "", "model": "", "created_at": now, "updated_at": now, "completed_at": "",
})
token = ensure_token(runtime.runtime_root)
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
port = int(server.server_address[1])
payload = {"run_id": board["run_id"], "spec": spec, "board": board, "status": "blocked"}
try:
    post_status, created = request(port, token, "POST", "/v1/goals", payload)
    assert post_status == 201 and created["created"] is True
    replay_status, replayed = request(port, token, "POST", "/v1/goals", payload)
    assert replay_status == 200 and replayed["created"] is False
    assert replayed["record"] == created["record"]

    goal_id = created["record"]["id"]
    get_status, fetched = request(port, token, "GET", "/v1/goals/" + goal_id)
    assert get_status == 200 and fetched == created["record"]
    list_status, listed = request(port, token, "GET", "/v1/goals")
    assert list_status == 200 and [item["id"] for item in listed["goals"]] == [goal_id]
    run_status, run_goal = request(port, token, "GET", "/v1/runs/" + board["run_id"] + "/goal")
    assert run_status == 200 and run_goal["id"] == goal_id
    session_status, session = request(port, token, "GET", "/v1/sessions/" + board["run_id"])
    assert session_status == 200
    metadata = session["metadata"]
    assert metadata["goal_id"] == goal_id
    assert metadata["goal_key"] == spec["goal_key"]
    assert metadata["goal_spec_sha256"] == bootstrap["goal_spec_sha256"]
    assert metadata["goal_board_sha256"] == bootstrap["goal_board_sha256"]
    assert metadata["goal_human_baseline_sha256"] == bootstrap["human_baseline_evidence_file_sha256"]

    updated_board = json.loads(json.dumps(board))
    updated_board["competitions"][0]["exact_gate"] = "SMOKE_HTTP_PUT_UPDATE"
    put_status, updated = request(
        port,
        token,
        "PUT",
        "/v1/goals/" + goal_id,
        {"board": updated_board, "status": "blocked"},
    )
    assert put_status == 200 and updated["status"] == "blocked"
    assert updated["board"]["competitions"][0]["exact_gate"] == "SMOKE_HTTP_PUT_UPDATE"
    final_status, final_goal = request(port, token, "GET", "/v1/goals/" + goal_id)
    assert final_status == 200 and final_goal == updated
    final_session_status, final_session = request(port, token, "GET", "/v1/sessions/" + board["run_id"])
    assert final_session_status == 200
    assert final_session["metadata"]["goal_board_sha256"] == sha256_json(updated_board)
    assert final_session["metadata"]["goal_human_baseline_sha256"] == bootstrap["human_baseline_evidence_file_sha256"]

    created_events = [
        item for item in runtime.store.list_events(board["run_id"])
        if item["event_type"] == "goal.created"
    ]
    updated_events = [
        item for item in runtime.store.list_events(board["run_id"])
        if item["event_type"] == "goal.updated"
    ]
    assert len(created_events) == 1 and len(updated_events) == 1
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    runtime.close()

assert not thread.is_alive()
print(json.dumps({
    "status": "passed",
    "candidate_import": str(module_path.relative_to(candidate_root)),
    "bind_host": "127.0.0.1",
    "port_ephemeral": True,
    "methods": ["POST", "GET", "PUT"],
    "post_status": post_status,
    "replay_status": replay_status,
    "get_status": get_status,
    "put_status": put_status,
    "session_status": session_status,
    "session_metadata_fields": sorted(metadata),
    "created_events": len(created_events),
    "updated_events": len(updated_events),
    "goal_id": goal_id,
}))
""",
        encoding="utf-8",
        newline="\n",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [os.environ.get("PYTHON") or sys.executable, str(smoke), str(root), str(bootstrap_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    smoke.unlink(missing_ok=True)
    if completed.returncode != 0:
        stderr_sha = sha256_bytes(completed.stderr.encode("utf-8", errors="replace"))
        raise RuntimeError(
            f"GOAL_CLEAN_SMOKE_FAILED:{round_number}:exit={completed.returncode}:stderr_sha256={stderr_sha}"
        )
    try:
        details = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GOAL_CLEAN_SMOKE_OUTPUT_INVALID:{round_number}") from exc
    if not isinstance(details, dict) or details.get("status") != "passed":
        raise RuntimeError(f"GOAL_CLEAN_SMOKE_OUTPUT_REJECTED:{round_number}")
    return {"round": round_number, "status": "passed", "http": details}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_source(source_root: Path, relative: str) -> str:
    return (source_root / "src" / relative).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _replace_once(text: str, anchor: str, replacement: str, code: str) -> str:
    """Apply one literal patch only when its anchor is unique."""

    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"{code}:count={count}")
    return text.replace(anchor, replacement, 1)


def _insert_once(text: str, anchor: str, insertion: str, code: str) -> str:
    """Insert a Goal fragment after one explicit baseline anchor."""

    return _replace_once(text, anchor, anchor + insertion, code)


def _require_absent(text: str, marker: str, code: str) -> None:
    if marker in text:
        raise RuntimeError(f"{code}:already_present")


def _validate_baseline_manifest(
    baseline_root: Path, baseline_files: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Prove the frozen R115 ZIP is the exact 30-file runtime baseline."""

    manifest_path = baseline_root / BASELINE_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_MISSING")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_INVALID") from exc
    if manifest.get("schema") != "evomind.super_agent_runtime_hotfix.v1":
        raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_SCHEMA_REJECTED")
    if str(manifest.get("release") or "") != "r115-progress-parser-r114-base":
        raise RuntimeError("GOAL_CLEAN_BASELINE_RELEASE_REJECTED")
    if str(manifest.get("source_manifest_sha256") or "").lower() != EXPECTED_R115_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("GOAL_CLEAN_BASELINE_SOURCE_MANIFEST_SHA_REJECTED")
    entries = manifest.get("files")
    if not isinstance(entries, list) or manifest.get("file_count") != BASELINE_RUNTIME_FILE_COUNT:
        raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_COUNT_REJECTED")
    listed: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_ENTRY_REJECTED")
        path = str(entry["path"]).replace("\\", "/")
        if path in listed or not path.endswith(".py"):
            raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_ENTRY_REJECTED")
        listed[path] = {
            "path": path,
            "bytes": int(entry.get("bytes", -1)),
            "sha256": str(entry.get("sha256", "")).lower(),
        }
    actual = {path: info for path, info in baseline_files.items() if path != BASELINE_MANIFEST}
    if set(listed) != set(actual) or len(actual) != BASELINE_RUNTIME_FILE_COUNT:
        raise RuntimeError("GOAL_CLEAN_BASELINE_MANIFEST_FILESET_REJECTED")
    for path, expected in listed.items():
        observed = actual[path]
        if expected != observed:
            raise RuntimeError(f"GOAL_CLEAN_BASELINE_MANIFEST_HASH_REJECTED:{path}")
    return manifest


def patch_diff_evidence(
    baseline_bytes: dict[str, bytes], candidate_root: Path
) -> tuple[list[dict[str, object]], str]:
    """Return per-file and combined SHA evidence for the seven actual unified diffs."""

    records: list[dict[str, object]] = []
    combined = bytearray()
    for relative in sorted(EXPECTED_CHANGED_FILES):
        before_raw = baseline_bytes.get(relative, b"")
        before = before_raw.decode("utf-8-sig").replace("\r\n", "\n")
        candidate_path = candidate_root / relative
        after_raw = candidate_path.read_bytes()
        after = after_raw.decode("utf-8-sig").replace("\r\n", "\n")
        lines = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"r115/{relative}",
                tofile=f"r116/{relative}",
                lineterm="",
            )
        )
        diff_bytes = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        if not diff_bytes:
            raise RuntimeError(f"GOAL_CLEAN_PATCH_DIFF_EMPTY:{relative}")
        combined.extend(relative.encode("utf-8"))
        combined.extend(b"\0")
        combined.extend(diff_bytes)
        combined.extend(b"\0")
        records.append(
            {
                "path": relative,
                "new_file": relative in NEW_FILES,
                "baseline_bytes": len(before_raw),
                "baseline_sha256": sha256_bytes(before_raw) if before_raw else None,
                "candidate_bytes": len(after_raw),
                "candidate_sha256": sha256_bytes(after_raw),
                "added_lines": sum(1 for line in lines if line.startswith("+") and not line.startswith("+++")),
                "deleted_lines": sum(1 for line in lines if line.startswith("-") and not line.startswith("---")),
                "diff_bytes": len(diff_bytes),
                "diff_sha256": sha256_bytes(diff_bytes),
            }
        )
    if len(records) != len(EXPECTED_CHANGED_FILES):
        raise RuntimeError("GOAL_CLEAN_PATCH_DIFF_COUNT_REJECTED")
    return records, sha256_bytes(bytes(combined))


def extract_block(text: str, start: str, end: str, code: str, expected_methods: tuple[str, ...]) -> str:
    """Extract a bounded source block and reject anchor ambiguity or drift."""

    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"{code}:anchor_count")
    begin = text.index(start)
    finish = text.index(end, begin)
    if finish <= begin:
        raise RuntimeError(f"{code}:anchor_order")
    block = text[begin:finish]
    # The extracted text is indented as a class body.  Parsing the wrapped
    # block lets us prove that only the intended Goal methods crossed the
    # baseline boundary; a broad dirty-worktree slice cannot pass this gate.
    wrapped = "class _GoalPatch:\n" + "".join(
        (" " + line if line.strip() else line) for line in block.splitlines(True)
    )
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as exc:
        raise RuntimeError(f"{code}:syntax") from exc
    methods = tuple(node.name for node in tree.body[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    if methods != expected_methods:
        raise RuntimeError(f"{code}:methods={','.join(methods)}")
    return block


def apply_goal_patches(source_root: Path, baseline_root: Path) -> tuple[set[str], dict[str, str]]:
    current = {relative: read_source(source_root, relative) for relative in (*DEPENDENT_FILES, *NEW_FILES)}
    baseline = {
        relative: (baseline_root / relative).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        for relative in DEPENDENT_FILES
    }
    patched: dict[str, str] = {}

    init = baseline["evomind_runtime/__init__.py"]
    _require_absent(init, "from .goal_board import", "GOAL_CLEAN_INIT_IMPORT")
    _require_absent(init, '"GoalRecordError"', "GOAL_CLEAN_INIT_EXPORT")
    init = _replace_once(
        init,
        "from .runtime import AgentRuntime\n",
        "from .runtime import AgentRuntime\nfrom .goal_board import GoalRecordError, ensure_fixed_goal, update_fixed_goal_board\n",
        "GOAL_CLEAN_INIT_IMPORT_ANCHOR",
    )
    init = _replace_once(
        init,
        '    "AgentRuntime",\n',
        '    "AgentRuntime",\n    "GoalRecordError",\n',
        "GOAL_CLEAN_INIT_AGENT_EXPORT_ANCHOR",
    )
    init = _replace_once(
        init,
        '    "SuperAgentRuntime",\n',
        '    "SuperAgentRuntime",\n    "ensure_fixed_goal",\n    "update_fixed_goal_board",\n',
        "GOAL_CLEAN_INIT_SUPER_EXPORT_ANCHOR",
    )
    patched["evomind_runtime/__init__.py"] = init

    runtime = baseline["evomind_runtime/runtime.py"]
    runtime_methods = extract_block(
        current["evomind_runtime/runtime.py"],
        "    def ensure_fixed_goal(",
        "    def tools(",
        "GOAL_CLEAN_RUNTIME_METHOD_BLOCK",
        ("ensure_fixed_goal", "get_goal", "update_fixed_goal"),
    )
    runtime_anchor = "    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:\n        return self.store.list_sessions(limit)\n\n"
    runtime = _insert_once(
        runtime,
        runtime_anchor,
        runtime_methods,
        "GOAL_CLEAN_RUNTIME_ANCHOR",
    )
    patched["evomind_runtime/runtime.py"] = runtime

    store = baseline["evomind_runtime/store.py"]
    schema_anchor = "CREATE TABLE IF NOT EXISTS uploads (\n"
    goal_schema = (
        "CREATE TABLE IF NOT EXISTS goal_records (\n"
        "  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, goal_key TEXT NOT NULL,\n"
        "  schema TEXT NOT NULL, allocation TEXT NOT NULL,\n"
        "  spec_sha256 TEXT NOT NULL, spec_json TEXT NOT NULL,\n"
        "  board_sha256 TEXT NOT NULL, human_baseline_sha256 TEXT NOT NULL, board_json TEXT NOT NULL,\n"
        "  status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,\n"
        "  UNIQUE(goal_key),\n"
        "  FOREIGN KEY(run_id) REFERENCES assistant_runs(id)\n"
        ");\n"
    )
    store = _replace_once(
        store,
        schema_anchor,
        goal_schema + schema_anchor,
        "GOAL_CLEAN_STORE_SCHEMA_ANCHOR",
    )
    index_anchor = "CREATE INDEX IF NOT EXISTS idx_attachments_run ON attachments(run_id, created_at);\n"
    goal_index = "CREATE INDEX IF NOT EXISTS idx_goal_records_run ON goal_records(run_id, updated_at);\n"
    store = _replace_once(
        store,
        index_anchor,
        goal_index + index_anchor,
        "GOAL_CLEAN_STORE_INDEX_ANCHOR",
    )
    store_methods = extract_block(
        current["evomind_runtime/store.py"],
        "    @staticmethod\n    def _decode_goal_row",
        "    def add_turn(",
        "GOAL_CLEAN_STORE_METHOD_BLOCK",
        (
            "_decode_goal_row",
            "get_goal_record",
            "get_goal_for_run",
            "list_goal_records",
            "goal_records_schema_issues",
            "ensure_goal_record",
            "update_goal_record",
        ),
    )
    store_anchor = "    def add_turn(self, session_id: str, role: str, content: Any) -> str:\n"
    store = _replace_once(
        store,
        store_anchor,
        store_methods + store_anchor,
        "GOAL_CLEAN_STORE_METHOD_ANCHOR",
    )
    patched["evomind_runtime/store.py"] = store

    assistant = baseline["evomind_runtime/assistant_runs.py"]
    _require_absent(assistant, '    "goal_created",\n', "GOAL_CLEAN_ASSISTANT_EVENT_TYPES")
    assistant = _replace_once(
        assistant,
        '    "run_failed",\n',
        '    "run_failed",\n    "goal_created",\n    "goal_updated",\n',
        "GOAL_CLEAN_ASSISTANT_EVENT_TYPES_ANCHOR",
    )
    snapshot_anchor = (
        '        if not isinstance(attachment_context, dict):\n'
        '            attachment_context = {"mode": "explicit" if attachments else "none", "source_run_id": ""}\n'
    )
    snapshot_patch = (
        snapshot_anchor
        + "        persisted_goal = self.store.get_goal_for_run(run_id)\n"
        + "        goal_projection = None\n"
        + "        if persisted_goal is not None:\n"
        + "            goal_projection = {\n"
        + "                key: persisted_goal.get(key)\n"
        + "                for key in (\n"
        '                    "id", "run_id", "goal_key", "schema", "allocation", "status",\n'
        '                    "spec_sha256", "board_sha256", "human_baseline_sha256", "created_at", "updated_at",\n'
        + "                )\n"
        + "            }\n"
    )
    if snapshot_anchor not in assistant:
        raise RuntimeError("GOAL_CLEAN_ASSISTANT_SNAPSHOT_ANCHOR_MISSING")
    assistant = _replace_once(assistant, snapshot_anchor, snapshot_patch, "GOAL_CLEAN_ASSISTANT_SNAPSHOT_ANCHOR")
    approvals_anchor = '            "approvals": approvals,\n'
    if approvals_anchor not in assistant:
        raise RuntimeError("GOAL_CLEAN_ASSISTANT_RETURN_ANCHOR_MISSING")
    assistant = _insert_once(
        assistant,
        approvals_anchor,
        '            "goal": goal_projection,\n',
        "GOAL_CLEAN_ASSISTANT_RETURN_ANCHOR",
    )
    aliases_anchor = '            "model.response": "model_observed",\n'
    if aliases_anchor not in assistant:
        raise RuntimeError("GOAL_CLEAN_ASSISTANT_ALIAS_ANCHOR_MISSING")
    assistant = _replace_once(
        assistant,
        aliases_anchor,
        aliases_anchor + '            "goal.created": "goal_created",\n            "goal.updated": "goal_updated",\n',
        "GOAL_CLEAN_ASSISTANT_ALIAS_ANCHOR",
    )
    patched["evomind_runtime/assistant_runs.py"] = assistant

    http = baseline["evomind_runtime/http_server.py"]
    http = _replace_once(
        http,
        "from .assistant_runs import CHUNK_SIZE\n",
        "from .assistant_runs import CHUNK_SIZE\nfrom .competition_goal import FIXED_GOAL_ID, FIXED_RUN_ID, validate_goal_record\n",
        "GOAL_CLEAN_HTTP_IMPORT_ANCHOR",
    )
    get_tools_anchor = '                    return self._json(200, {"tools": runtime.tools()})\n'
    get_tools_patch = get_tools_anchor + '''                if parsed.path == "/v1/goals":
                    goal = runtime.store.get_goal_record(FIXED_GOAL_ID)
                    if goal is None:
                        return self._json(200, {"goals": []})
                    checked = validate_goal_record(goal)
                    if not checked.valid:
                        return self._json(409, {"error": "goal_record_invalid", "issues": [item.to_dict() for item in checked.issues]})
                    return self._json(200, {"goals": [{
                        key: goal.get(key)
                        for key in ("id", "run_id", "goal_key", "schema", "allocation", "status", "spec_sha256", "board_sha256", "human_baseline_sha256", "created_at", "updated_at")
                    }]})
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
'''
    http = _replace_once(http, get_tools_anchor, get_tools_patch, "GOAL_CLEAN_HTTP_GET_ANCHOR")
    run_snapshot_anchor = '                    return self._json(200, runtime.assistant.snapshot(parts[2]))\n'
    run_goal_patch = run_snapshot_anchor + '''                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "goal":
                    if parts[2] != FIXED_RUN_ID:
                        raise KeyError(parts[2])
                    goal = runtime.get_goal(parts[2])
                    if goal is None:
                        raise KeyError(parts[2])
                    checked = validate_goal_record(goal)
                    if not checked.valid:
                        return self._json(409, {"error": "goal_record_invalid", "issues": [item.to_dict() for item in checked.issues]})
                    return self._json(200, goal)
'''
    http = _replace_once(http, run_snapshot_anchor, run_goal_patch, "GOAL_CLEAN_HTTP_RUN_ANCHOR")
    post_session_anchor = '                    return self._json(201, runtime.create_session(**body))\n'
    post_goal_patch = post_session_anchor + '''                if parts == ["v1", "goals"]:
                    spec = body.get("spec")
                    board = body.get("board")
                    if not isinstance(spec, dict) or not isinstance(board, dict):
                        raise RequestContractError(400, "goal_spec_and_board_objects_required")
                    result = runtime.ensure_fixed_goal(
                        run_id=str(body.get("run_id") or ""),
                        spec=spec,
                        board=board,
                        status=str(body.get("status") or "blocked"),
                    )
                    return self._json(201 if result.get("created") else 200, result)
'''
    http = _replace_once(http, post_session_anchor, post_goal_patch, "GOAL_CLEAN_HTTP_POST_ANCHOR")
    put_try_anchor = "            try:\n                if len(parts) == 5 and parts[:2] == [\"v1\", \"uploads\"] and parts[3] == \"chunks\":\n"
    put_goal_patch = '''            try:
                if len(parts) == 3 and parts[:2] == ["v1", "goals"]:
                    if parts[2] != FIXED_GOAL_ID:
                        raise KeyError(parts[2])
                    body = self._body()
                    board = body.get("board")
                    if not isinstance(board, dict):
                        raise RequestContractError(400, "goal_board_object_required")
                    return self._json(200, runtime.update_fixed_goal(
                        goal_id=parts[2],
                        board=board,
                        status=str(body["status"]) if "status" in body else None,
                    ))
                if len(parts) == 5 and parts[:2] == ["v1", "uploads"] and parts[3] == "chunks":
'''
    http = _replace_once(http, put_try_anchor, put_goal_patch, "GOAL_CLEAN_HTTP_PUT_ANCHOR")
    patched["evomind_runtime/http_server.py"] = http

    for relative in NEW_FILES:
        patched[relative] = current[relative]
    changed = set()
    for relative, text in patched.items():
        target = baseline_root / relative
        if not target.exists() or target.read_text(encoding="utf-8-sig").replace("\r\n", "\n") != text:
            changed.add(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    return changed, {relative: sha256_bytes(text.encode("utf-8")) for relative, text in patched.items()}


def build(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root).resolve(strict=True)
    baseline_zip = Path(args.baseline_zip).resolve(strict=True)
    goal_board_path = Path(args.goal_board)
    if not goal_board_path.is_absolute():
        goal_board_path = source_root / goal_board_path
    goal_board_path = goal_board_path.resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    zip_path = output_root / args.zip_name
    manifest_path = output_root / args.manifest_name
    result_path = output_root / args.result_name
    bootstrap_output_path = output_root / args.bootstrap_name
    if any(path.exists() for path in (zip_path, manifest_path, result_path, bootstrap_output_path)):
        raise RuntimeError("GOAL_CLEAN_OUTPUT_EXISTS")
    baseline_sha = sha256_file(baseline_zip)
    if (
        baseline_sha != args.expected_baseline_sha256.lower()
        or baseline_sha != EXPECTED_R115_ZIP_SHA256
    ):
        raise RuntimeError("GOAL_CLEAN_BASELINE_SHA_MISMATCH")
    unsafe, duplicates = safe_zip_names(baseline_zip)
    if unsafe or duplicates:
        raise RuntimeError("GOAL_CLEAN_BASELINE_ZIP_UNSAFE")
    bootstrap = load_bootstrap_artifact(source_root, goal_board_path)
    bootstrap_bytes = (json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    bootstrap_sha = sha256_bytes(bootstrap_bytes)
    with tempfile.TemporaryDirectory(prefix="g21-goal-clean-") as temporary:
        temporary_root = Path(temporary)
        frozen_bootstrap_path = temporary_root / args.bootstrap_name
        frozen_bootstrap_path.write_bytes(bootstrap_bytes)
        if sha256_file(frozen_bootstrap_path) != bootstrap_sha:
            raise RuntimeError("GOAL_CLEAN_BOOTSTRAP_FREEZE_REJECTED")
        package_root = Path(temporary) / "package"
        package_root.mkdir()
        with zipfile.ZipFile(baseline_zip) as archive:
            archive.extractall(package_root)
        baseline_files_with_manifest = regular_files(package_root)
        baseline_manifest = _validate_baseline_manifest(package_root, baseline_files_with_manifest)
        baseline_patch_bytes = {
            relative: (package_root / relative).read_bytes()
            for relative in DEPENDENT_FILES
        }
        (package_root / BASELINE_MANIFEST).unlink(missing_ok=True)
        baseline_files = {
            path: info for path, info in baseline_files_with_manifest.items() if path != BASELINE_MANIFEST
        }
        baseline_py = {key: value for key, value in baseline_files.items() if key.endswith(".py")}
        if len(baseline_py) != BASELINE_RUNTIME_FILE_COUNT:
            raise RuntimeError("GOAL_CLEAN_BASELINE_RUNTIME_COUNT_MISMATCH")
        changed, patch_hashes = apply_goal_patches(source_root, package_root)
        if changed != EXPECTED_CHANGED_FILES:
            raise RuntimeError("GOAL_CLEAN_CHANGESET_REJECTED:" + ",".join(sorted(changed ^ EXPECTED_CHANGED_FILES)))
        patch_diffs, patch_diff_sha = patch_diff_evidence(baseline_patch_bytes, package_root)
        candidate_before_manifest = regular_files(package_root)
        expected_paths = set(baseline_files) | set(NEW_FILES)
        if set(candidate_before_manifest) != expected_paths:
            extras = sorted(set(candidate_before_manifest) - expected_paths)
            missing = sorted(expected_paths - set(candidate_before_manifest))
            raise RuntimeError("GOAL_CLEAN_FILESET_REJECTED:extra=" + ",".join(extras) + ":missing=" + ",".join(missing))
        unchanged_files: list[dict[str, object]] = []
        for relative, baseline_info in sorted(baseline_files.items()):
            observed = candidate_before_manifest[relative]
            if relative not in changed and observed != baseline_info:
                raise RuntimeError(f"GOAL_CLEAN_UNCHANGED_HASH_DRIFT:{relative}")
            if relative not in changed:
                unchanged_files.append(
                    {
                        "path": relative,
                        "bytes": observed["bytes"],
                        "sha256": observed["sha256"],
                        "byte_identical_to_r115": True,
                    }
                )
        if len(unchanged_files) != EXPECTED_UNCHANGED_BASELINE_FILE_COUNT:
            raise RuntimeError("GOAL_CLEAN_UNCHANGED_FILE_COUNT_REJECTED")
        candidate_files = [value for key, value in sorted(candidate_before_manifest.items())]
        if len(candidate_files) != BASELINE_RUNTIME_FILE_COUNT + len(NEW_FILES):
            raise RuntimeError("GOAL_CLEAN_RUNTIME_CLOSURE_COUNT_REJECTED")
        if any(not str(item["path"]).endswith(".py") for item in candidate_files):
            raise RuntimeError("GOAL_CLEAN_RUNTIME_CLOSURE_NON_PYTHON_REJECTED")
        runtime_tree_sha = runtime_tree_sha256(candidate_files)
        manifest = {
            "schema": OVERLAY_SCHEMA,
            "run_id": FIXED_RUN_ID,
            "allocation": FIXED_ALLOCATION,
            "release": "r116-g21-goal-r115-base",
            "target": "bundle/runtime/evomind_runtime",
            "baseline_zip_sha256": baseline_sha,
            "baseline_source_manifest_sha256": str(baseline_manifest["source_manifest_sha256"]),
            "expected_production_baseline_runtime_tree_sha256": EXPECTED_R115_PRODUCTION_TREE_SHA256,
            "baseline_runtime_file_count": len(baseline_py),
            "candidate_runtime_file_count": len(candidate_files),
            "candidate_runtime_tree_sha256": runtime_tree_sha,
            "unchanged_baseline_file_count": len(unchanged_files),
            "unchanged_baseline_files": unchanged_files,
            "patched_existing_file_count": len(DEPENDENT_FILES),
            "new_goal_file_count": len(NEW_FILES),
            "changed_files": sorted(changed),
            "changed_file_count": len(changed),
            "file_count": len(candidate_files),
            "files": candidate_files,
            "patch_source_sha256": patch_hashes,
            "patch_diff_file_count": len(patch_diffs),
            "patch_diffs": patch_diffs,
            "patch_diff_sha256": patch_diff_sha,
            "manifest_file_count": 1,
            "archive_regular_file_count": len(candidate_files) + 1,
            "baseline_runtime_manifest_present": False,
            "goal_manifest_name": GOAL_MANIFEST,
            "goal_spec_sha256": bootstrap["goal_spec_file_sha256"],
            "goal_spec_canonical_sha256": bootstrap["goal_spec_sha256"],
            "human_baseline_evidence_sha256": bootstrap["human_baseline_evidence_file_sha256"],
            "human_baseline_evidence_canonical_sha256": bootstrap["human_baseline_evidence_sha256"],
            "goal_board_file_sha256": bootstrap["goal_board_file_sha256"],
            "goal_board_sha256": bootstrap["goal_board_sha256"],
            "bootstrap_artifact_name": args.bootstrap_name,
            "bootstrap_artifact_sha256": bootstrap_sha,
            "production_deployed": False,
            "production_deployable": False,
            "reviewed_goal_only": True,
            "requires_frozen_patch_review": False,
            "hpc_accessed": False,
            "gpu_touched": False,
            "remote_writes": 0,
        }
        write_json(package_root / GOAL_MANIFEST, manifest)
        staged_zip = temporary_root / "candidate.zip"
        with zipfile.ZipFile(staged_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(package_root).as_posix())
        zip_unsafe, zip_duplicates = safe_zip_names(staged_zip)
        if zip_unsafe or zip_duplicates:
            raise RuntimeError("GOAL_CLEAN_ZIP_SAFETY_REJECTED")
        rounds: list[dict[str, object]] = []
        for round_number in (1, 2):
            verify_root = Path(temporary) / f"verify-{round_number}"
            verify_root.mkdir()
            with zipfile.ZipFile(staged_zip) as archive:
                archive.extractall(verify_root)
            actual = regular_files(verify_root)
            if GOAL_MANIFEST not in actual or BASELINE_MANIFEST in actual:
                raise RuntimeError("GOAL_CLEAN_MANIFEST_LAYOUT_REJECTED")
            manifest_names = sorted(
                path for path in actual if Path(path).name.casefold().endswith("manifest.json")
            )
            if manifest_names != [GOAL_MANIFEST]:
                raise RuntimeError("GOAL_CLEAN_UNIQUE_MANIFEST_REJECTED:" + ",".join(manifest_names))
            manifest_value = json.loads((verify_root / GOAL_MANIFEST).read_text(encoding="utf-8"))
            listed = {
                str(item.get("path")): item
                for item in manifest_value.get("files", [])
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            }
            actual_payload = {key: value for key, value in actual.items() if key != GOAL_MANIFEST}
            if manifest_value.get("schema") != OVERLAY_SCHEMA or set(listed) != set(actual_payload):
                raise RuntimeError(f"GOAL_CLEAN_MANIFEST_FILESET_REJECTED:{round_number}")
            if (
                manifest_value.get("candidate_runtime_tree_sha256") != runtime_tree_sha256(
                    [actual_payload[key] for key in sorted(actual_payload)]
                )
                or manifest_value.get("patch_diff_sha256") != patch_diff_sha
                or manifest_value.get("unchanged_baseline_file_count") != EXPECTED_UNCHANGED_BASELINE_FILE_COUNT
                or manifest_value.get("bootstrap_artifact_sha256") != bootstrap_sha
                or manifest_value.get("goal_board_sha256") != bootstrap["goal_board_sha256"]
            ):
                raise RuntimeError(f"GOAL_CLEAN_MANIFEST_BINDING_REJECTED:{round_number}")
            for relative, expected in listed.items():
                observed = actual_payload[relative]
                if (
                    int(expected.get("bytes", -1)) != int(observed["bytes"])
                    or str(expected.get("sha256", "")).lower() != str(observed["sha256"]).lower()
                ):
                    raise RuntimeError(f"GOAL_CLEAN_MANIFEST_HASH_REJECTED:{round_number}:{relative}")
            py_files = [key for key in actual_payload if key.endswith(".py")]
            if len(py_files) != BASELINE_RUNTIME_FILE_COUNT + len(NEW_FILES):
                raise RuntimeError("GOAL_CLEAN_RUNTIME_COUNT_REJECTED")
            for relative in py_files:
                completed = subprocess.run(
                    [os.environ.get("PYTHON") or sys.executable, "-m", "py_compile", str(verify_root / relative)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"GOAL_CLEAN_PYCOMPILE_FAILED:{round_number}:{relative}")
            rounds.append(smoke_candidate(verify_root, frozen_bootstrap_path, round_number))
        staged_zip_sha = sha256_file(staged_zip)
        staged_zip_bytes = staged_zip.stat().st_size
        if staged_zip_sha in REJECTED_CANDIDATE_ZIP_SHA256:
            raise RuntimeError("GOAL_CLEAN_REJECTED_CANDIDATE_SHA")
        manifest_sha = sha256_file(package_root / GOAL_MANIFEST)
        result = {
            "schema": OVERLAY_SCHEMA,
            "status": "built_and_verified",
            "release": "r116-g21-goal-r115-base",
            "zip_path": str(zip_path),
            "zip_bytes": staged_zip_bytes,
            "zip_sha256": staged_zip_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "bootstrap_artifact_path": str(bootstrap_output_path),
            "bootstrap_artifact_sha256": bootstrap_sha,
            "baseline_zip_sha256": baseline_sha,
            "baseline_source_manifest_sha256": str(baseline_manifest["source_manifest_sha256"]),
            "expected_production_baseline_runtime_tree_sha256": EXPECTED_R115_PRODUCTION_TREE_SHA256,
            "changed_file_count": len(changed),
            "patch_diff_file_count": len(patch_diffs),
            "patch_diff_sha256": patch_diff_sha,
            "unchanged_baseline_file_count": len(unchanged_files),
            "unchanged_baseline_files": unchanged_files,
            "candidate_runtime_file_count": len(candidate_files),
            "candidate_runtime_tree_sha256": runtime_tree_sha,
            "goal_spec_sha256": bootstrap["goal_spec_file_sha256"],
            "human_baseline_evidence_sha256": bootstrap["human_baseline_evidence_file_sha256"],
            "goal_board_file_sha256": bootstrap["goal_board_file_sha256"],
            "goal_board_sha256": bootstrap["goal_board_sha256"],
            "verification_rounds": rounds,
            "test_receipts": {
                "independent_extraction_rounds": 2,
                "py_compile_per_round": len(candidate_files),
                "goal_loopback_http_rounds": 2,
                "goal_loopback_http_methods": ["POST", "GET", "PUT"],
            },
            "reviewed_goal_only": True,
            "production_deployed": False,
            "production_deployable": False,
            "hpc_accessed": False,
            "gpu_touched": False,
            "remote_writes": 0,
        }
        zip_path.write_bytes(staged_zip.read_bytes())
        bootstrap_output_path.write_bytes(bootstrap_bytes)
        write_json(manifest_path, manifest)
        write_json(result_path, result)
        if (
            sha256_file(zip_path) != staged_zip_sha
            or sha256_file(bootstrap_output_path) != bootstrap_sha
            or sha256_file(manifest_path) != manifest_sha
        ):
            raise RuntimeError("GOAL_CLEAN_FINAL_OUTPUT_HASH_REJECTED")
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--baseline-zip", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--goal-board", default="configs/g21_five_competition_goal_board_bootstrap.json")
    parser.add_argument("--zip-name", default="evomind-runtime-r116-g21-goal-r115-base.zip")
    parser.add_argument("--manifest-name", default="evomind-runtime-r116-g21-goal-r115-base-source-manifest.json")
    parser.add_argument("--result-name", default="evomind-runtime-r116-g21-goal-r115-base-build-result.json")
    parser.add_argument("--bootstrap-name", default="evomind-runtime-r116-g21-goal-r115-base-bootstrap.json")
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
