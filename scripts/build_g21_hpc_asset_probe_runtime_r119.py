from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
RELEASE = "r119-g21-hpc-asset-probe-r118-base"
SCHEMA = "evomind.g21_hpc_asset_probe_runtime.r119.v1"
MANIFEST_NAME = "hpc-asset-probe-runtime-manifest.json"
OLD_MANIFEST_NAME = "open-polymer-legacy-adoption-runtime-manifest.json"
STALE_MANIFEST_NAME = "runtime-hotfix-manifest.json"
BASELINE_ZIP_SHA256 = "6e0c5c67a9cb69c68f5f2f88777e41a0db86c4340cfb2b0d04b981e84259908c"
BASELINE_TREE_SHA256 = "b9d60506be3d01e4bf8dab9111915cc532acb74c2aa8632adc6a8d15312c2592"
BASELINE_MANIFEST_SHA256 = "5cf108c09395e236987bdf7924e84a15ffd6437c603c0ac34d722f068feda76d"
OLD_SPEC_FILE_SHA256 = "380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f"
OLD_SPEC_CANONICAL_SHA256 = "a569bae731723fc7046817df737876d6dc7674eadd9f7b812fe7b4a87e521655"
NEW_SPEC_FILE_SHA256 = "da8c3fbe8b14107dfadf96ad906b4cf982bc714ebf6420225a1e3722cd41a752"
NEW_SPEC_CANONICAL_SHA256 = "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
POLICY_EVIDENCE_FILE_SHA256 = "47df860192502afcaacd7b1b6bc7fc4d86110c2e665278c8b7b3ea421893b7a7"
POLICY_EVIDENCE_CANONICAL_SHA256 = "f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0"
MIGRATION_ID = "g21-conditional-strong-baseline-v2"
BASELINE_FILE_COUNT = 32
CHANGED_FILES = (
    "evomind_runtime/tools.py",
)
EXPECTED_SOURCE_SHA256 = {
    "evomind_runtime/tools.py": "7b8cdd65a22c66c4bbfd9058a1d8b35db9f969ead1b9f9c583d46c55055355da",
}
REQUIRED_MARKERS = {
    "evomind_runtime/tools.py": (
        "HPC_ASSET_PROBE_SCOPE",
        "def _hpc_asset_probe_remote_source()",
        "def _parse_hpc_asset_probe_payload(",
        "def _hpc_asset_probe(",
        '"hpc_asset_probe"',
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(raw.encode("utf-8"))


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_member(info: zipfile.ZipInfo) -> str:
    raw = info.filename.replace("\\", "/")
    pure = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or "\\" in info.filename
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise ValueError(f"ZIP_UNSAFE_PATH:{raw}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP_SYMLINK_REJECTED:{raw}")
    return raw


def inspect_zip(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if sha256_file(path) != BASELINE_ZIP_SHA256:
        raise ValueError("R116_BASELINE_ZIP_SHA_MISMATCH")
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = _safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"ZIP_DUPLICATE_PATH:{name}")
            entries[name] = archive.read(info)
    if OLD_MANIFEST_NAME not in entries:
        raise ValueError("R116_GOAL_MANIFEST_MISSING")
    if STALE_MANIFEST_NAME in entries or MANIFEST_NAME in entries:
        raise ValueError("R116_MANIFEST_SET_REJECTED")
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(runtime) != BASELINE_FILE_COUNT or any(not name.endswith(".py") for name in runtime):
        raise ValueError("R116_RUNTIME_FILESET_REJECTED")
    if sha256_bytes(entries[OLD_MANIFEST_NAME]) != BASELINE_MANIFEST_SHA256:
        raise ValueError("R116_MANIFEST_RAW_SHA_MISMATCH")
    manifest = json.loads(entries[OLD_MANIFEST_NAME].decode("utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("R116_MANIFEST_OBJECT_REQUIRED")
    declared = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    if set(declared) != set(runtime) or len(declared) != BASELINE_FILE_COUNT:
        raise ValueError("R116_MANIFEST_FILESET_MISMATCH")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R116_MANIFEST_HASH_MISMATCH:{name}")
    tree = runtime_tree_sha256(runtime)
    if tree != BASELINE_TREE_SHA256 or str(manifest.get("candidate_runtime_tree_sha256")) != tree:
        raise ValueError("R116_RUNTIME_TREE_MISMATCH")
    if int(manifest.get("candidate_runtime_file_count", -1)) != BASELINE_FILE_COUNT:
        raise ValueError("R116_MANIFEST_COUNT_MISMATCH")
    return runtime, manifest


def runtime_tree_sha256(files: Mapping[str, bytes]) -> str:
    prefix = "evomind_runtime/"
    rows: list[str] = []
    for name, raw in files.items():
        if not name.startswith(prefix) or name == prefix:
            raise ValueError(f"RUNTIME_TREE_PATH_REJECTED:{name}")
        rows.append(f"{name[len(prefix):]}|{len(raw)}|{sha256_bytes(raw)}")
    return sha256_bytes("\n".join(sorted(rows)).encode("utf-8"))


def verify_policy_inputs(old_spec_path: Path, new_spec_path: Path, evidence_path: Path) -> dict[str, Any]:
    expected = (
        (old_spec_path, OLD_SPEC_FILE_SHA256, OLD_SPEC_CANONICAL_SHA256, "OLD_SPEC"),
        (new_spec_path, NEW_SPEC_FILE_SHA256, NEW_SPEC_CANONICAL_SHA256, "NEW_SPEC"),
        (
            evidence_path,
            POLICY_EVIDENCE_FILE_SHA256,
            POLICY_EVIDENCE_CANONICAL_SHA256,
            "POLICY_EVIDENCE",
        ),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for path, raw_sha, canonical_sha, label in expected:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label}_REGULAR_FILE_REQUIRED")
        if sha256_file(path) != raw_sha:
            raise ValueError(f"{label}_RAW_SHA_MISMATCH")
        value = load_object(path)
        if canonical_sha256(value) != canonical_sha:
            raise ValueError(f"{label}_CANONICAL_SHA_MISMATCH")
        loaded[label] = value
    old_spec = loaded["OLD_SPEC"]
    new_spec = loaded["NEW_SPEC"]
    evidence = loaded["POLICY_EVIDENCE"]
    for value, label in ((old_spec, "OLD_SPEC"), (new_spec, "NEW_SPEC"), (evidence, "EVIDENCE")):
        if str(value.get("run_id")) != FIXED_RUN_ID or str(value.get("allocation")) != FIXED_ALLOCATION:
            raise ValueError(f"{label}_IDENTITY_MISMATCH")
    if str(new_spec.get("previous_goal_spec_file_sha256")) != OLD_SPEC_FILE_SHA256:
        raise ValueError("NEW_SPEC_OLD_RAW_BINDING_MISMATCH")
    if str(new_spec.get("previous_goal_spec_canonical_sha256")) != OLD_SPEC_CANONICAL_SHA256:
        raise ValueError("NEW_SPEC_OLD_CANONICAL_BINDING_MISMATCH")
    if str(new_spec.get("policy_evidence_sha256")) != POLICY_EVIDENCE_CANONICAL_SHA256:
        raise ValueError("NEW_SPEC_EVIDENCE_CANONICAL_BINDING_MISMATCH")
    if str(new_spec.get("policy_evidence_file_sha256")) != POLICY_EVIDENCE_FILE_SHA256:
        raise ValueError("NEW_SPEC_EVIDENCE_RAW_BINDING_MISMATCH")
    if str(new_spec.get("policy_migration_id")) != MIGRATION_ID:
        raise ValueError("NEW_SPEC_MIGRATION_ID_MISMATCH")
    if str(evidence.get("previous_goal_spec_file_sha256")) != OLD_SPEC_FILE_SHA256:
        raise ValueError("EVIDENCE_OLD_RAW_BINDING_MISMATCH")
    if str(evidence.get("previous_goal_spec_canonical_sha256")) != OLD_SPEC_CANONICAL_SHA256:
        raise ValueError("EVIDENCE_OLD_CANONICAL_BINDING_MISMATCH")
    if str(evidence.get("migration_id")) != MIGRATION_ID:
        raise ValueError("EVIDENCE_MIGRATION_ID_MISMATCH")
    return loaded


def _extract(text: str, start: str, end: str, code: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"{code}_ANCHOR_REJECTED")
    left = text.index(start)
    right = text.index(end, left)
    return text[left:right]


def _replace_once(text: str, anchor: str, replacement: str, code: str) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"{code}_ANCHOR_REJECTED")
    return text.replace(anchor, replacement, 1)


def _ast_scope(baseline: str, candidate: str) -> None:
    def definitions(value: str) -> dict[str, str]:
        tree = ast.parse(value)
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    before = definitions(baseline)
    after = definitions(candidate)
    added = set(after) - set(before)
    removed = set(before) - set(after)
    changed = {name for name in set(before) & set(after) if before[name] != after[name]}
    if added != {"_hpc_asset_probe_remote_source", "_parse_hpc_asset_probe_payload", "_hpc_asset_probe"}:
        raise ValueError("R119_AST_ADDED_FUNCTIONS_REJECTED")
    if removed or changed != {"build_default_registry"}:
        raise ValueError("R119_AST_EXISTING_FUNCTION_SCOPE_REJECTED")


def source_overlays(source_root: Path, baseline: Mapping[str, bytes]) -> dict[str, bytes]:
    relative = "evomind_runtime/tools.py"
    path = source_root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("R119_TOOLS_SOURCE_REGULAR_FILE_REQUIRED")
    raw = path.read_bytes()
    if sha256_bytes(raw) != EXPECTED_SOURCE_SHA256[relative]:
        raise ValueError("R119_TOOLS_SOURCE_SHA_MISMATCH")
    current = raw.decode("utf-8-sig").replace("\r\n", "\n")
    original = baseline[relative].decode("utf-8-sig").replace("\r\n", "\n")
    for marker in REQUIRED_MARKERS[relative]:
        if marker not in current:
            raise ValueError(f"R119_TOOLS_MARKER_MISSING:{marker}")

    constants = _extract(
        current,
        'HPC_ASSET_PROBE_SCOPE = "cure_mindgames"',
        "\n\n@dataclass\nclass ToolContext",
        "R119_CONSTANTS",
    ).rstrip() + "\n"
    functions = _extract(
        current,
        "def _hpc_asset_probe_remote_source()",
        "def _profile_environment(",
        "R119_FUNCTIONS",
    ).rstrip() + "\n\n"
    registration = _extract(
        current,
        '    add(\n        "hpc_asset_probe",',
        '    add(\n        "hpc_execute_solution",',
        "R119_TOOL_SPEC",
    ).rstrip() + "\n"

    candidate = _replace_once(
        original,
        "\n\n@dataclass\nclass ToolContext",
        "\n" + constants + "\n@dataclass\nclass ToolContext",
        "R119_CONSTANTS_INSERT",
    )
    candidate = _replace_once(
        candidate,
        "def _profile_environment(",
        functions + "def _profile_environment(",
        "R119_FUNCTIONS_INSERT",
    )
    candidate = _replace_once(
        candidate,
        '    READ_ONLY_TOOLS.add("competition_data_progress")',
        '    READ_ONLY_TOOLS.update({"competition_data_progress", "hpc_asset_probe"})',
        "R119_READ_ONLY_REGISTRATION",
    )
    hpc_verify_line = '    add("hpc_verify", "Verify the server-bound designated proxy, allocation role route, host/GPU UUIDs, remote root and five read-only GPU samples. The model cannot select or retarget the job/profile.", {}, [], "hpc.read", _hpc_verify)'
    candidate = _replace_once(
        candidate,
        hpc_verify_line + "\n",
        hpc_verify_line + "\n" + registration,
        "R119_TOOL_SPEC_INSERT",
    )
    ast.parse(candidate, filename=relative)
    _ast_scope(original, candidate)
    return {relative: candidate.encode("utf-8")}


def diff_evidence(baseline: Mapping[str, bytes], candidate: Mapping[str, bytes]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for relative in CHANGED_FILES:
        old = baseline[relative]
        new = candidate[relative]
        diff = "".join(
            difflib.unified_diff(
                old.decode("utf-8-sig").splitlines(keepends=True),
                new.decode("utf-8-sig").splitlines(keepends=True),
                fromfile=f"r116/{relative}",
                tofile=f"r119/{relative}",
            )
        ).encode("utf-8")
        rows.append(
            {
                "path": relative,
                "baseline_bytes": len(old),
                "baseline_sha256": sha256_bytes(old),
                "candidate_bytes": len(new),
                "candidate_sha256": sha256_bytes(new),
                "diff_bytes": len(diff),
                "diff_sha256": sha256_bytes(diff),
            }
        )
    return rows, canonical_sha256(rows)


SMOKE = r'''
import http.client
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer

candidate, runtime_root, old_path, new_path, evidence_path, round_text = sys.argv[1:]
sys.path.insert(0, candidate)
from evomind_runtime.competition_goal import (
    CONDITIONAL_BASELINE_MIGRATION_ID, FIXED_ALLOCATION, FIXED_GOAL_ID, FIXED_RUN_ID,
    INITIAL_GOAL_SPEC_CANONICAL_SHA256, load_goal_spec, sha256_json,
)
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import utc_now
from evomind_runtime.runtime import AgentRuntime

def request(port, token, method, path, body=None):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": "Bearer " + token}
    if raw is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(raw))})
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=raw, headers=headers)
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, data

old_spec = load_goal_spec(old_path)
new_spec = load_goal_spec(new_path)
evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8-sig"))
board = {
    "schema": "evomind.goal-board.v2", "run_id": FIXED_RUN_ID,
    "allocation": FIXED_ALLOCATION, "target_count": 5, "weather_actions": 0,
    "competitions": [
        {"competition": name, "goal_status": "WAITING_EXACT_GATE", "exact_gate": "pending"}
        for name in ("cure_bench", "e2lmc", "mindgames", "open_polymer", "ariel_2025")
    ],
}
runtime = AgentRuntime(Path(runtime_root))
runtime.create_session(session_id=FIXED_RUN_ID, workspace_root=runtime_root, objective="r119 policy smoke")
now = utc_now()
runtime.store.create_assistant_run({
    "id": FIXED_RUN_ID, "session_id": FIXED_RUN_ID,
    "conversation_id": "r119-policy-smoke-" + round_text, "prompt": "r119 policy smoke",
    "task_root": runtime_root, "status": "blocked", "plan": {}, "attachment_ids": [],
    "retry_count": 0, "error_class": "", "error_message": "", "model_provider": "",
    "model": "", "created_at": now, "updated_at": now, "completed_at": now,
})
created = runtime.ensure_fixed_goal(run_id=FIXED_RUN_ID, spec=old_spec, board=board)
assert created["record"]["spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
token = ensure_token(runtime.runtime_root)
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
port = int(server.server_address[1])
body = {"policy_migration": {
    "run_id": FIXED_RUN_ID, "expected_spec_sha256": INITIAL_GOAL_SPEC_CANONICAL_SHA256,
    "new_spec": new_spec, "policy_evidence": evidence,
    "migration_id": CONDITIONAL_BASELINE_MIGRATION_ID,
}}
try:
    status, migrated = request(port, token, "PUT", "/v1/goals/" + FIXED_GOAL_ID, body)
    assert status == 200 and migrated["migrated"] is True
    status, goal = request(port, token, "GET", "/v1/goals/" + FIXED_GOAL_ID)
    assert status == 200 and goal["spec_sha256"] == sha256_json(new_spec) and goal["board"] == board
    status, replay = request(port, token, "PUT", "/v1/goals/" + FIXED_GOAL_ID, body)
    assert status == 200 and replay["migrated"] is False
    status, rejected = request(port, token, "PUT", "/v1/goals/" + FIXED_GOAL_ID, {**body, "status": "active"})
    assert status == 400 and rejected["error"] == "goal_policy_migration_must_be_isolated"
    events = [event for event in runtime.store.list_events(FIXED_RUN_ID) if event["event_type"] == "goal.policy_migrated"]
    assert len(events) == 1
    assert events[0]["payload"]["old_spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
    metadata = runtime.store.get_session(FIXED_RUN_ID)["metadata"]
    assert metadata["goal_spec_sha256"] == sha256_json(new_spec)
    assert metadata["goal_previous_spec_sha256"] == INITIAL_GOAL_SPEC_CANONICAL_SHA256
    assert metadata["goal_policy_evidence_sha256"] == sha256_json(evidence)
finally:
    server.shutdown(); server.server_close(); thread.join(timeout=5); runtime.close()
print(json.dumps({"round": int(round_text), "migration": True, "replay": True, "events": 1, "metadata": True}))
'''


def compile_candidate(root: Path) -> int:
    files = sorted((root / "evomind_runtime").glob("*.py"))
    if len(files) != BASELINE_FILE_COUNT:
        raise ValueError("PY_COMPILE_FILE_COUNT_REJECTED")
    cache = root / ".pycompile"
    cache.mkdir()
    for index, path in enumerate(files):
        py_compile.compile(str(path), cfile=str(cache / f"{index}.pyc"), doraise=True)
    return len(files)


def run_smoke(root: Path, old_spec: Path, new_spec: Path, evidence: Path, round_number: int) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_hpc_asset_probe.py"),
            str(ROOT / "tests/test_competition_data_progress.py"),
            str(ROOT / "tests/test_assistant_run_service.py::test_managed_hpc_identity_prefers_server_bound_run_identity_over_stale_process_environment"),
            str(ROOT / "tests/test_assistant_run_service.py::test_hpc_verify_rejects_five_samples_with_incomplete_identity_evidence"),
            "--rootdir",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        raise ValueError(f"R119_HPC_ASSET_PROBE_SMOKE_FAILED:{round_number}:{completed.stdout[-1000:]}:{completed.stderr[-1000:]}")
    return {
        "round": round_number,
        "hpc_asset_probe_tests": True,
        "progress_identity_regression": True,
        "real_hpc_accessed": False,
        "pytest_summary": completed.stdout.strip().splitlines()[-1],
    }


def extract_candidate(path: Path, destination: Path) -> None:
    names: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = _safe_member(info)
            if info.is_dir():
                continue
            if name in names:
                raise ValueError(f"CANDIDATE_DUPLICATE_PATH:{name}")
            names.add(name)
            target = destination / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))


def verify_extraction(root: Path, expected_runtime: Mapping[str, bytes], manifest_sha: str) -> dict[str, Any]:
    actual: dict[str, bytes] = {}
    for path in sorted((root / "evomind_runtime").glob("*.py")):
        actual[path.relative_to(root).as_posix()] = path.read_bytes()
    regular = [path for path in root.rglob("*") if path.is_file() and ".pycompile" not in path.parts]
    if set(actual) != set(expected_runtime) or len(actual) != BASELINE_FILE_COUNT:
        raise ValueError("EXTRACTED_RUNTIME_FILESET_MISMATCH")
    if any(actual[name] != expected_runtime[name] for name in actual):
        raise ValueError("EXTRACTED_RUNTIME_BYTES_MISMATCH")
    manifests = [path for path in regular if path.name.endswith("manifest.json")]
    if len(manifests) != 1 or manifests[0].name != MANIFEST_NAME:
        raise ValueError("EXTRACTED_MANIFEST_SET_REJECTED")
    if sha256_file(manifests[0]) != manifest_sha:
        raise ValueError("EXTRACTED_MANIFEST_SHA_MISMATCH")
    if len(regular) != BASELINE_FILE_COUNT + 1:
        raise ValueError("EXTRACTED_FILE_COUNT_REJECTED")
    return {"regular_files": len(regular), "runtime_files": len(actual), "tree_sha256": runtime_tree_sha256(actual)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    baseline_zip = Path(args.baseline_zip).resolve()
    source_root = Path(args.source_root).resolve()
    old_spec = Path(args.old_spec).resolve()
    new_spec = Path(args.new_spec).resolve()
    policy_evidence = Path(args.policy_evidence).resolve()
    output_dir = Path(args.output_dir).resolve()
    baseline, baseline_manifest = inspect_zip(baseline_zip)
    verify_policy_inputs(old_spec, new_spec, policy_evidence)
    overlays = source_overlays(source_root, baseline)
    candidate = dict(baseline)
    for relative, raw in overlays.items():
        if raw == baseline[relative]:
            raise ValueError(f"OVERLAY_DID_NOT_CHANGE_BASELINE:{relative}")
        candidate[relative] = raw
    changed = sorted(name for name in candidate if candidate[name] != baseline[name])
    unchanged = sorted(name for name in candidate if candidate[name] == baseline[name])
    if changed != sorted(CHANGED_FILES) or len(unchanged) != 31:
        raise ValueError("R119_PATCH_SCOPE_REJECTED")
    diffs, diff_sha = diff_evidence(baseline, candidate)
    files = [
        {"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)}
        for name, raw in sorted(candidate.items())
    ]
    tree_sha = runtime_tree_sha256(candidate)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "release": RELEASE,
        "target": "bundle/runtime/evomind_runtime",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_release": "r118-g21-open-polymer-legacy-adoption-r117-base",
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree_sha,
        "unchanged_file_count": len(unchanged),
        "unchanged_files": [
            {"path": name, "bytes": len(candidate[name]), "sha256": sha256_bytes(candidate[name])}
            for name in unchanged
        ],
        "changed_file_count": len(changed),
        "changed_files": changed,
        "files": files,
        "patch_diff_sha256": diff_sha,
        "patch_diffs": diffs,
        "old_goal_spec_file_sha256": OLD_SPEC_FILE_SHA256,
        "old_goal_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
        "new_goal_spec_file_sha256": NEW_SPEC_FILE_SHA256,
        "new_goal_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
        "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
        "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
        "policy_migration_id": MIGRATION_ID,
        "old_manifest_removed": True,
        "manifest_file_count": 1,
        "production_deployable": False,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
    }
    if args.dry_run:
        return {
            "schema": SCHEMA,
            "status": "dry_run_verified",
            "release": RELEASE,
            "baseline_zip_sha256": BASELINE_ZIP_SHA256,
            "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
            "candidate_runtime_file_count": len(candidate),
            "unchanged_file_count": len(unchanged),
            "changed_file_count": len(changed),
            "changed_files": changed,
            "old_manifest_removed": True,
            "new_manifest_count": 1,
            "old_goal_spec_file_sha256": OLD_SPEC_FILE_SHA256,
            "old_goal_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
            "new_goal_spec_file_sha256": NEW_SPEC_FILE_SHA256,
            "new_goal_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
            "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
            "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
            "production_deployable": False,
        }
    if output_dir.exists():
        raise ValueError(f"OUTPUT_DIRECTORY_EXISTS:{output_dir}")
    output_dir.mkdir(parents=True)
    prefix = "evomind-runtime-r119-g21-hpc-asset-probe-r118-base"
    candidate_zip = output_dir / f"{prefix}.zip"
    manifest_path = output_dir / f"{prefix}-source-manifest.json"
    build_result_path = output_dir / f"{prefix}-build-result.json"
    write_json(manifest_path, manifest)
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_raw)
    with zipfile.ZipFile(candidate_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(candidate.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
        info = zipfile.ZipInfo(MANIFEST_NAME, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_raw)
    zip_sha = sha256_file(candidate_zip)
    rounds: list[dict[str, Any]] = []
    for round_number in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"r119-round-{round_number}-") as temporary:
            extracted = Path(temporary) / "candidate"
            extracted.mkdir()
            extract_candidate(candidate_zip, extracted)
            checked = verify_extraction(extracted, candidate, manifest_sha)
            checked["py_compile"] = compile_candidate(extracted)
            checked["policy_smoke"] = run_smoke(
                extracted, old_spec, new_spec, policy_evidence, round_number
            )
            checked["round"] = round_number
            rounds.append(checked)
    build_result = {
        "schema": SCHEMA,
        "status": "built_and_verified",
        "release": RELEASE,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "zip_filename": candidate_zip.name,
        "zip_bytes": candidate_zip.stat().st_size,
        "zip_sha256": zip_sha,
        "source_manifest_filename": manifest_path.name,
        "source_manifest_sha256": manifest_sha,
        "candidate_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree_sha,
        "unchanged_file_count": len(unchanged),
        "changed_file_count": len(changed),
        "changed_files": changed,
        "patch_diff_sha256": diff_sha,
        "old_goal_spec_file_sha256": OLD_SPEC_FILE_SHA256,
        "old_goal_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
        "new_goal_spec_file_sha256": NEW_SPEC_FILE_SHA256,
        "new_goal_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
        "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
        "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
        "verification_rounds": rounds,
        "production_deployable": False,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
    }
    write_json(build_result_path, build_result)
    return {**build_result, "build_result_filename": build_result_path.name}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build the fixed R119 conditional-policy runtime")
    value.add_argument("--baseline-zip", required=True)
    value.add_argument("--source-root", default=str(ROOT / "src"))
    value.add_argument("--old-spec", default=str(ROOT / "configs/g21_five_competition_goal_v1_frozen.json"))
    value.add_argument("--new-spec", default=str(ROOT / "configs/g21_five_competition_goal.json"))
    value.add_argument(
        "--policy-evidence",
        default=str(ROOT / "configs/g21_conditional_strong_baseline_policy_evidence.json"),
    )
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
