from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
RELEASE = "r120-g21-terminal-preserving-probe-r119-base"
SCHEMA = "evomind.g21_terminal_preserving_probe_runtime.r120.v1"
MANIFEST_NAME = "terminal-preserving-probe-runtime-manifest.json"
BASELINE_MANIFEST_NAME = "hpc-asset-probe-runtime-manifest.json"
BASELINE_ZIP_SHA256 = "9368c7962632fb976e7a8ba8ede0d3108989e96c00c4dd75bca154008d438206"
BASELINE_TREE_SHA256 = "906868d22a841f591fb84c7a505b75acee3c8df58f179e83d3d4585601d03975"
BASELINE_MANIFEST_SHA256 = "12f80d02d6fd87e413c57bc606d09a27f90c3edac3b4b1d7462826374666b9ac"
BASELINE_FILE_COUNT = 32
CHANGED_FILE = "evomind_runtime/http_server.py"
EXPECTED_SOURCE_RAW_SHA256 = "519820314f63d2e5ae6ba354e044ea7df55c544a6c0ab85de43102279af0af4d"
EXPECTED_PACKAGED_SHA256 = "6d2d17f1257736683c4bc590f346c183c9feb25e06cce7a77d63d35843e23a9d"
REQUIRED_MARKERS = (
    "def _terminal_direct_tool_state(",
    "def _invoke_direct_tool(",
    'tool_name != "hpc_asset_probe"',
    "runtime._invoke_lock",
    "terminal_read_only_idempotency_required",
    "terminal_read_only_pending_approval",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_member(info: zipfile.ZipInfo) -> str:
    raw = info.filename.replace("\\", "/")
    pure = PurePosixPath(raw)
    mode = (info.external_attr >> 16) & 0xFFFF
    if (
        not raw
        or raw.startswith("/")
        or "\\" in info.filename
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise ValueError(f"ZIP_UNSAFE_PATH:{raw}")
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP_SYMLINK_REJECTED:{raw}")
    return raw


def runtime_tree_sha256(files: Mapping[str, bytes]) -> str:
    prefix = "evomind_runtime/"
    rows = []
    for name, raw in files.items():
        if not name.startswith(prefix) or name == prefix:
            raise ValueError(f"RUNTIME_TREE_PATH_REJECTED:{name}")
        rows.append(f"{name[len(prefix):]}|{len(raw)}|{sha256_bytes(raw)}")
    return sha256_bytes("\n".join(sorted(rows)).encode("utf-8"))


def inspect_baseline(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("R120_BASELINE_REGULAR_FILE_REQUIRED")
    if sha256_file(path) != BASELINE_ZIP_SHA256:
        raise ValueError("R120_BASELINE_ZIP_SHA_MISMATCH")
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"ZIP_DUPLICATE_PATH:{name}")
            entries[name] = archive.read(info)
    if set(name for name in entries if name.endswith("manifest.json")) != {BASELINE_MANIFEST_NAME}:
        raise ValueError("R120_BASELINE_MANIFEST_SET_REJECTED")
    manifest_raw = entries[BASELINE_MANIFEST_NAME]
    if sha256_bytes(manifest_raw) != BASELINE_MANIFEST_SHA256:
        raise ValueError("R120_BASELINE_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(runtime) != BASELINE_FILE_COUNT or any(not name.endswith(".py") for name in runtime):
        raise ValueError("R120_BASELINE_RUNTIME_FILESET_REJECTED")
    declared = {str(item.get("path")): item for item in manifest.get("files", []) if isinstance(item, Mapping)}
    if set(declared) != set(runtime) or len(declared) != BASELINE_FILE_COUNT:
        raise ValueError("R120_BASELINE_MANIFEST_CLOSURE_REJECTED")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R120_BASELINE_FILE_HASH_REJECTED:{name}")
    tree = runtime_tree_sha256(runtime)
    if tree != BASELINE_TREE_SHA256 or str(manifest.get("candidate_runtime_tree_sha256")) != tree:
        raise ValueError("R120_BASELINE_TREE_REJECTED")
    return runtime, manifest


def definition_map(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def source_overlay(source_root: Path, baseline: Mapping[str, bytes]) -> bytes:
    path = source_root / CHANGED_FILE
    if not path.is_file() or path.is_symlink():
        raise ValueError("R120_HTTP_SERVER_REGULAR_FILE_REQUIRED")
    raw = path.read_bytes()
    if sha256_bytes(raw) != EXPECTED_SOURCE_RAW_SHA256:
        raise ValueError("R120_HTTP_SERVER_SOURCE_SHA_MISMATCH")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    old_text = baseline[CHANGED_FILE].decode("utf-8-sig").replace("\r\n", "\n")
    for marker in REQUIRED_MARKERS:
        if marker not in text:
            raise ValueError(f"R120_HTTP_SERVER_MARKER_MISSING:{marker}")
    if "/actions" in text or "action=cancel" in text:
        raise ValueError("R120_HTTP_SERVER_CANCEL_WORKAROUND_REJECTED")
    old_defs = definition_map(old_text)
    new_defs = definition_map(text)
    added_defs = set(new_defs) - set(old_defs)
    removed_defs = set(old_defs) - set(new_defs)
    changed_defs = {name for name in old_defs if name in new_defs and old_defs[name] != new_defs[name]}
    if added_defs != {"_terminal_direct_tool_state", "_invoke_direct_tool"} or removed_defs:
        raise ValueError("R120_HTTP_SERVER_DEFINITION_SET_REJECTED")
    if changed_defs != {"make_handler"}:
        raise ValueError("R120_HTTP_SERVER_AST_SCOPE_REJECTED")
    ast.parse(text, filename=CHANGED_FILE)
    packaged = text.encode("utf-8")
    if sha256_bytes(packaged) != EXPECTED_PACKAGED_SHA256:
        raise ValueError("R120_HTTP_SERVER_PACKAGED_SHA_MISMATCH")
    return packaged


def diff_evidence(old: bytes, new: bytes) -> tuple[dict[str, Any], str]:
    diff = "".join(
        difflib.unified_diff(
            old.decode("utf-8-sig").splitlines(keepends=True),
            new.decode("utf-8-sig").splitlines(keepends=True),
            fromfile=f"r119/{CHANGED_FILE}",
            tofile=f"r120/{CHANGED_FILE}",
        )
    ).encode("utf-8")
    row = {
        "path": CHANGED_FILE,
        "baseline_bytes": len(old),
        "baseline_sha256": sha256_bytes(old),
        "candidate_bytes": len(new),
        "candidate_sha256": sha256_bytes(new),
        "diff_bytes": len(diff),
        "diff_sha256": sha256_bytes(diff),
    }
    return row, canonical_sha256([row])


def extract_candidate(path: Path, destination: Path) -> None:
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in seen:
                raise ValueError(f"R120_CANDIDATE_DUPLICATE:{name}")
            seen.add(name)
            target = destination / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))


def compile_candidate(root: Path) -> int:
    files = sorted((root / "evomind_runtime").glob("*.py"))
    if len(files) != BASELINE_FILE_COUNT:
        raise ValueError("R120_PY_COMPILE_COUNT_REJECTED")
    cache = root / ".pycompile"
    cache.mkdir()
    for index, path in enumerate(files):
        py_compile.compile(str(path), cfile=str(cache / f"{index}.pyc"), doraise=True)
    return len(files)


SMOKE = r'''
import http.client
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer

candidate, runtime_root, expected_ok = sys.argv[1:]
sys.path.insert(0, candidate)
from evomind_runtime.competition_goal import FIXED_RUN_ID
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import ToolResult, utc_now
from evomind_runtime.runtime import AgentRuntime

runtime = AgentRuntime(Path(runtime_root))
runtime.create_session(session_id=FIXED_RUN_ID, workspace_root=runtime_root, objective="r120 terminal preserve smoke")
now = utc_now()
runtime.store.create_assistant_run({
    "id": FIXED_RUN_ID, "session_id": FIXED_RUN_ID, "conversation_id": "r120-smoke",
    "prompt": "fixture", "task_root": str(Path(runtime_root) / "assistant_tasks" / FIXED_RUN_ID),
    "status": "cancelled", "error_class": "fixture_gate", "error_message": "preserve me",
    "created_at": now, "updated_at": now, "completed_at": now, "retry_count": 0,
    "plan": {}, "attachment_ids": [], "model_provider": "", "model": "",
})
runtime.store.update_session(FIXED_RUN_ID, status="cancelled")
ok = expected_ok == "1"
content = {
    "schema": "evomind.g21_hpc_asset_probe.v1", "scope": "cure_mindgames",
    "read_only": True, "signals_sent": 0, "remote_writes": 0,
    "other_processes_modified": False, "training_started": False, "worker_control": 0,
}
runtime.registry._handlers["hpc_asset_probe"] = lambda args, context: ToolResult("", ok, content, "fixture", error="" if ok else "fixture_gate")
token = ensure_token(Path(runtime_root))
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
try:
    body = json.dumps({"tool_name":"hpc_asset_probe","arguments":{"scope":"cure_mindgames"},"idempotency_key":"r120-smoke-key-" + expected_ok}).encode()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    connection.request("POST", f"/v1/sessions/{FIXED_RUN_ID}/tools", body=body, headers={"Authorization":"Bearer " + token,"Content-Type":"application/json"})
    response = connection.getresponse(); payload = json.loads(response.read()); connection.close()
    assert response.status == 200
    assert payload["status"] == ("completed" if ok else "failed")
    snapshot = runtime.assistant.snapshot(FIXED_RUN_ID)
    session = runtime.store.get_session(FIXED_RUN_ID)
    assert snapshot["status"] == "cancelled" and snapshot["terminal"] is True
    assert snapshot["error_class"] == "fixture_gate" and snapshot["error_message"] == "preserve me"
    assert snapshot["completed_at"] == now and snapshot["active_tool_calls"] == []
    assert session["status"] == "cancelled"
    events = runtime.store.list_events(FIXED_RUN_ID, 0)
    assert len([e for e in events if e["event_type"] == "tool.started"]) == 1
    assert len([e for e in events if e["event_type"] == ("tool.completed" if ok else "tool.failed")]) == 1
finally:
    server.shutdown(); server.server_close(); thread.join(timeout=5); runtime.close()
print(json.dumps({"preserve_terminal":True,"tool_ok":ok,"cancel_calls":0}))
'''


def run_smoke(root: Path, round_number: int) -> dict[str, Any]:
    evidence = []
    for ok in (True, False):
        with tempfile.TemporaryDirectory(prefix=f"r120-smoke-{round_number}-") as runtime_root:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", SMOKE, str(root), runtime_root, "1" if ok else "0"],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if completed.returncode != 0:
                raise ValueError(f"R120_SMOKE_FAILED:{round_number}:{ok}:{completed.stdout[-1000:]}:{completed.stderr[-1000:]}")
            evidence.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    return {"success_and_failure_preserved": True, "cancel_calls": 0, "cases": evidence}


def verify_extraction(root: Path, expected: Mapping[str, bytes], manifest_sha: str, round_number: int) -> dict[str, Any]:
    actual = {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted((root / "evomind_runtime").glob("*.py"))}
    regular = [path for path in root.rglob("*") if path.is_file() and ".pycompile" not in path.parts]
    if set(actual) != set(expected) or any(actual[name] != expected[name] for name in actual):
        raise ValueError(f"R120_EXTRACTED_RUNTIME_REJECTED:{round_number}")
    manifests = [path for path in regular if path.name.endswith("manifest.json")]
    if len(manifests) != 1 or manifests[0].name != MANIFEST_NAME or sha256_file(manifests[0]) != manifest_sha:
        raise ValueError(f"R120_EXTRACTED_MANIFEST_REJECTED:{round_number}")
    if len(regular) != BASELINE_FILE_COUNT + 1:
        raise ValueError(f"R120_EXTRACTED_FILE_COUNT_REJECTED:{round_number}")
    return {"round": round_number, "regular_files": len(regular), "runtime_files": len(actual), "tree_sha256": runtime_tree_sha256(actual)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    baseline_zip = Path(args.baseline_zip).resolve()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    baseline, _baseline_manifest = inspect_baseline(baseline_zip)
    overlay = source_overlay(source_root, baseline)
    if overlay == baseline[CHANGED_FILE]:
        raise ValueError("R120_OVERLAY_DID_NOT_CHANGE_BASELINE")
    candidate = dict(baseline)
    candidate[CHANGED_FILE] = overlay
    changed = sorted(name for name in candidate if candidate[name] != baseline[name])
    unchanged = sorted(name for name in candidate if candidate[name] == baseline[name])
    if changed != [CHANGED_FILE] or len(unchanged) != 31:
        raise ValueError("R120_PATCH_SCOPE_REJECTED")
    patch, patch_sha = diff_evidence(baseline[CHANGED_FILE], overlay)
    tree_sha = runtime_tree_sha256(candidate)
    files = [{"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)} for name, raw in sorted(candidate.items())]
    manifest = {
        "schema": SCHEMA,
        "release": RELEASE,
        "target": "bundle/runtime/evomind_runtime",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_release": "r119-g21-hpc-asset-probe-r118-base",
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree_sha,
        "unchanged_file_count": 31,
        "unchanged_files": [{"path": name, "bytes": len(candidate[name]), "sha256": sha256_bytes(candidate[name])} for name in unchanged],
        "changed_file_count": 1,
        "changed_files": [CHANGED_FILE],
        "files": files,
        "source_http_server_raw_sha256": EXPECTED_SOURCE_RAW_SHA256,
        "source_http_server_packaged_sha256": EXPECTED_PACKAGED_SHA256,
        "patch_diff_sha256": patch_sha,
        "patch_diffs": [patch],
        "old_manifest_removed": True,
        "manifest_file_count": 1,
        "terminal_preserving_direct_tool": "hpc_asset_probe",
        "terminal_preserving_fixed_run": FIXED_RUN_ID,
        "cancel_workaround_forbidden": True,
        "production_deployable": False,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
    }
    if args.dry_run:
        return {
            "schema": SCHEMA, "status": "dry_run_verified", "release": RELEASE,
            "baseline_zip_sha256": BASELINE_ZIP_SHA256, "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "candidate_runtime_file_count": 32, "changed_files": [CHANGED_FILE], "unchanged_file_count": 31,
            "candidate_runtime_tree_sha256": tree_sha, "production_deployable": False,
        }
    if output_dir.exists():
        raise ValueError(f"OUTPUT_DIRECTORY_EXISTS:{output_dir}")
    output_dir.mkdir(parents=True)
    prefix = "evomind-runtime-r120-g21-terminal-preserving-probe-r119-base"
    candidate_zip = output_dir / f"{prefix}.zip"
    manifest_path = output_dir / f"{prefix}-source-manifest.json"
    build_result_path = output_dir / f"{prefix}-build-result.json"
    write_json(manifest_path, manifest)
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_raw)
    with zipfile.ZipFile(candidate_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(candidate.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
        info = zipfile.ZipInfo(MANIFEST_NAME, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_raw)
    rounds = []
    for number in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"r120-round-{number}-") as temporary:
            extracted = Path(temporary) / "candidate"; extracted.mkdir()
            extract_candidate(candidate_zip, extracted)
            checked = verify_extraction(extracted, candidate, manifest_sha, number)
            checked["py_compile"] = compile_candidate(extracted)
            checked["terminal_preserving_smoke"] = run_smoke(extracted, number)
            rounds.append(checked)
    result = {
        "schema": SCHEMA, "status": "built_and_verified", "release": RELEASE,
        "run_id": FIXED_RUN_ID, "allocation": FIXED_ALLOCATION,
        "zip_filename": candidate_zip.name, "zip_bytes": candidate_zip.stat().st_size, "zip_sha256": sha256_file(candidate_zip),
        "source_manifest_filename": manifest_path.name, "source_manifest_sha256": manifest_sha,
        "candidate_runtime_file_count": 32, "candidate_runtime_tree_sha256": tree_sha,
        "unchanged_file_count": 31, "changed_file_count": 1, "changed_files": [CHANGED_FILE],
        "patch_diff_sha256": patch_sha, "verification_rounds": rounds,
        "production_deployable": False, "production_deployed": False,
        "hpc_accessed": False, "gpu_touched": False, "remote_writes": 0,
    }
    write_json(build_result_path, result)
    return {**result, "build_result_filename": build_result_path.name}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build the fixed R120 terminal-preserving probe runtime from R119")
    value.add_argument("--baseline-zip", required=True)
    value.add_argument("--source-root", default=str(ROOT / "src"))
    value.add_argument("--output-dir", required=True)
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> int:
    try:
        result = build(parser().parse_args())
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
