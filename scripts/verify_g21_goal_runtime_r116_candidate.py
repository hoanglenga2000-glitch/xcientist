"""Independently verify an R116 Goal-only candidate and emit its test receipt."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
MANIFEST_SCHEMA = "evomind.g21_goal_runtime_overlay.clean.v1"
RECEIPT_SCHEMA = "evomind.g21_goal_runtime_test_receipt.v1"
R115_ZIP_SHA256 = "fc65d7134f9f53fa6eda21ee2fcd804a330d9184a83d5206a6b7469a217d89ba"
R115_SOURCE_MANIFEST_SHA256 = "3a42107ba57f0ecfd04247e21561555cd5c1abc7f73e9b78b6ff7fdf13d4ed4f"
R115_TREE_SHA256 = "b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a"
REJECTED_CANDIDATE_ZIP_SHA256 = "37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d"
EXPECTED_CHANGED = {
    "evomind_runtime/__init__.py",
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/store.py",
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
}
NEW_FILES = {"evomind_runtime/competition_goal.py", "evomind_runtime/goal_board.py"}
MANIFEST_NAME = "goal-runtime-overlay-manifest.json"
BASELINE_MANIFEST_NAME = "runtime-hotfix-manifest.json"


SMOKE = r'''
import http.client, json, pathlib, sys, tempfile, threading
from http.server import ThreadingHTTPServer
candidate = pathlib.Path(sys.argv[1]).resolve()
bootstrap = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
spec = bootstrap["goal_spec"]
board = bootstrap["initial_goal_board"]
sys.path.insert(0, str(candidate))
from evomind_runtime.http_server import ensure_token, make_handler
from evomind_runtime.models import utc_now
from evomind_runtime.runtime import AgentRuntime
workspace = pathlib.Path(tempfile.mkdtemp(prefix="r116-independent-http-"))
runtime = AgentRuntime(workspace)
run_id = board["run_id"]
runtime.create_session(session_id=run_id, workspace_root=str(workspace), objective="R116 independent Goal HTTP smoke")
now = utc_now()
runtime.store.create_assistant_run({
  "id":run_id,"session_id":run_id,"conversation_id":"r116-independent","prompt":"fixture","task_root":str(workspace),
  "status":"blocked","plan":{},"attachment_ids":[],"retry_count":0,"error_class":"","error_message":"",
  "model_provider":"","model":"","created_at":now,"updated_at":now,"completed_at":""
})
token = ensure_token(runtime.runtime_root)
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, token))
thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start(); port = int(server.server_address[1])
def request(method, path, body=None):
  connection=http.client.HTTPConnection("127.0.0.1",port,timeout=5)
  raw=json.dumps(body).encode() if body is not None else None
  headers={"Authorization":"Bearer "+token}
  if raw is not None: headers.update({"Content-Type":"application/json","Content-Length":str(len(raw))})
  try:
    connection.request(method,path,body=raw,headers=headers);response=connection.getresponse()
    return response.status,json.loads(response.read().decode())
  finally: connection.close()
try:
  payload={"run_id":run_id,"spec":spec,"board":board,"status":"blocked"}
  post,created=request("POST","/v1/goals",payload); replay,reused=request("POST","/v1/goals",payload)
  get_status,goal=request("GET","/v1/goals/goal_g21_five_competition")
  list_status,listed=request("GET","/v1/goals")
  run_status,run_goal=request("GET",f"/v1/runs/{run_id}/goal")
  put_status,updated=request("PUT","/v1/goals/goal_g21_five_competition",{"board":board,"status":"blocked"})
  event_types=[item["event_type"] for item in runtime.store.list_events(run_id)]
  assert (post,replay,get_status,list_status,run_status,put_status)==(201,200,200,200,200,200)
  assert created["created"] is True and reused["created"] is False
  assert goal["id"]=="goal_g21_five_competition" and goal["status"]=="blocked" and run_goal["id"]==goal["id"]
  assert len(listed["goals"])==1 and updated["board_sha256"]==goal["board_sha256"]
  assert event_types.count("goal.created")==1 and event_types.count("goal.updated")==0
  metadata=runtime.store.get_session(run_id)["metadata"]
  assert metadata["goal_id"]==goal["id"] and metadata["goal_spec_sha256"]==goal["spec_sha256"] and metadata["goal_board_sha256"]==goal["board_sha256"]
  assert metadata["goal_human_baseline_sha256"]==goal["human_baseline_sha256"]
  print(json.dumps({"status":"passed","post":post,"replay":replay,"get":get_status,"put":put_status,"goal_created":1,"goal_updated":0,"metadata_bound":True},separators=(",",":")))
finally:
  server.shutdown();server.server_close();thread.join(timeout=5);runtime.close()
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def inspect_zip(path: Path) -> tuple[list[zipfile.ZipInfo], dict[str, bytes]]:
    names: set[str] = set()
    regular: dict[str, bytes] = {}
    entries: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            entries.append(entry)
            name = entry.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if (
                not name or name.startswith("/") or "\\" in entry.filename or pure.is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts) or re.match(r"^[A-Za-z]:", name)
            ):
                raise ValueError(f"ZIP_UNSAFE:{name}")
            key = name.casefold()
            if key in names:
                raise ValueError(f"ZIP_DUPLICATE:{name}")
            names.add(key)
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"ZIP_SYMLINK:{name}")
            if not entry.is_dir():
                regular[name] = archive.read(entry)
    return entries, regular


def runtime_tree_sha(files: dict[str, bytes]) -> str:
    rows = []
    for name, raw in sorted(files.items()):
        if not name.startswith("evomind_runtime/"):
            raise ValueError(f"RUNTIME_SCOPE_REJECTED:{name}")
        relative = name.removeprefix("evomind_runtime/")
        rows.append(f"{relative}|{len(raw)}|{hashlib.sha256(raw).hexdigest()}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def patch_diff_evidence(baseline: dict[str, bytes], candidate: dict[str, bytes]) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    combined = bytearray()
    for relative in sorted(EXPECTED_CHANGED):
        before_raw = baseline.get(relative, b"")
        after_raw = candidate[relative]
        before = before_raw.decode("utf-8-sig").replace("\r\n", "\n")
        after = after_raw.decode("utf-8-sig").replace("\r\n", "\n")
        lines = list(
            difflib.unified_diff(
                before.splitlines(), after.splitlines(),
                fromfile=f"r115/{relative}", tofile=f"r116/{relative}", lineterm="",
            )
        )
        diff_raw = (("\n".join(lines) + "\n") if lines else "").encode()
        if not diff_raw:
            raise ValueError(f"R116_PATCH_DIFF_EMPTY:{relative}")
        combined.extend(relative.encode()); combined.extend(b"\0"); combined.extend(diff_raw); combined.extend(b"\0")
        records.append({
            "path": relative, "new_file": relative in NEW_FILES,
            "baseline_bytes": len(before_raw), "baseline_sha256": hashlib.sha256(before_raw).hexdigest() if before_raw else None,
            "candidate_bytes": len(after_raw), "candidate_sha256": hashlib.sha256(after_raw).hexdigest(),
            "added_lines": sum(1 for line in lines if line.startswith("+") and not line.startswith("+++")),
            "deleted_lines": sum(1 for line in lines if line.startswith("-") and not line.startswith("---")),
            "diff_bytes": len(diff_raw), "diff_sha256": hashlib.sha256(diff_raw).hexdigest(),
        })
    return records, hashlib.sha256(bytes(combined)).hexdigest()


def verify_repository_checks() -> dict[str, str]:
    secret = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_no_plaintext_secrets.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if secret.returncode != 0:
        raise RuntimeError("SECRET_SCAN_FAILED")
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    if diff.returncode != 0:
        raise RuntimeError("GIT_DIFF_CHECK_FAILED")
    return {"secret_scan": "passed", "git_diff_check": "passed"}


def verify_junit(path: Path, *, minimum_tests: int, required_classnames: set[str]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise ValueError(f"JUNIT_SUITE_MISSING:{path.name}")
    tests = sum(int(item.attrib.get("tests", "0")) for item in suites)
    failures = sum(int(item.attrib.get("failures", "0")) for item in suites)
    errors = sum(int(item.attrib.get("errors", "0")) for item in suites)
    classnames = {str(item.attrib.get("classname") or "") for item in root.findall(".//testcase")}
    if tests < minimum_tests or failures != 0 or errors != 0 or not required_classnames <= classnames:
        raise ValueError(f"JUNIT_RESULT_REJECTED:{path.name}")
    return {
        "filename": path.name, "sha256": sha256_file(path), "tests": tests,
        "failures": failures, "errors": errors,
        "required_classnames": sorted(required_classnames),
    }


def verify_candidate(
    *, candidate_zip: Path, source_manifest_path: Path, build_result_path: Path,
    baseline_zip: Path, baseline_source_manifest_path: Path, bootstrap_artifact: Path, targeted_junit: Path,
    runtime_junit: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"RECEIPT_OUTPUT_EXISTS:{output}")
    if sha256_file(candidate_zip) == REJECTED_CANDIDATE_ZIP_SHA256:
        raise ValueError("REJECTED_CANDIDATE_ZIP_SHA256")
    if sha256_file(baseline_zip) != R115_ZIP_SHA256 or sha256_file(baseline_source_manifest_path) != R115_SOURCE_MANIFEST_SHA256:
        raise ValueError("R115_BASELINE_HASH_REJECTED")
    _, baseline_regular = inspect_zip(baseline_zip)
    if BASELINE_MANIFEST_NAME not in baseline_regular:
        raise ValueError("R115_MANIFEST_MISSING")
    baseline_embedded_manifest = json.loads(baseline_regular[BASELINE_MANIFEST_NAME].decode("utf-8-sig"))
    baseline_payload = {name: raw for name, raw in baseline_regular.items() if name != BASELINE_MANIFEST_NAME}
    if len(baseline_payload) != 30 or len([name for name in baseline_payload if name.endswith(".py")]) != 30:
        raise ValueError("R115_FILESET_REJECTED")
    baseline_external_manifest = load_object(baseline_source_manifest_path)
    if (
        baseline_embedded_manifest.get("schema") != "evomind.super_agent_runtime_hotfix.v1"
        or baseline_embedded_manifest.get("file_count") != 30
        or baseline_external_manifest.get("schema") != "evomind.super_agent_source_set.v1"
        or baseline_external_manifest.get("file_count") != 30
        or baseline_external_manifest.get("frozen") is not True
    ):
        raise ValueError("R115_MANIFEST_CONTRACT_REJECTED")
    baseline_listed = {
        str(item["path"]).removeprefix("src/"): item
        for item in baseline_external_manifest.get("files") or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    embedded_listed = {
        str(item["path"]): item
        for item in baseline_embedded_manifest.get("files") or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if set(baseline_listed) != set(baseline_payload) or set(embedded_listed) != set(baseline_payload):
        raise ValueError("R115_MANIFEST_FILESET_REJECTED")
    for name, raw in baseline_payload.items():
        observed = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if any(
            int(source[name].get("bytes", -1)) != observed["bytes"] or source[name].get("sha256") != observed["sha256"]
            for source in (baseline_listed, embedded_listed)
        ):
            raise ValueError(f"R115_MANIFEST_HASH_REJECTED:{name}")
    _, candidate_regular = inspect_zip(candidate_zip)
    if MANIFEST_NAME not in candidate_regular or BASELINE_MANIFEST_NAME in candidate_regular:
        raise ValueError("R116_MANIFEST_LAYOUT_REJECTED")
    manifest_raw = candidate_regular.pop(MANIFEST_NAME)
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    external_manifest = load_object(source_manifest_path)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest != external_manifest or sha256_file(source_manifest_path) != manifest_sha:
        raise ValueError("R116_SOURCE_MANIFEST_MISMATCH")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("run_id") != FIXED_RUN_ID
        or manifest.get("allocation") != FIXED_ALLOCATION or manifest.get("baseline_zip_sha256") != R115_ZIP_SHA256
        or manifest.get("baseline_source_manifest_sha256") != R115_SOURCE_MANIFEST_SHA256
        or manifest.get("expected_production_baseline_runtime_tree_sha256") != R115_TREE_SHA256
        or manifest.get("candidate_runtime_file_count") != 32 or manifest.get("changed_file_count") != 7
        or set(manifest.get("changed_files") or []) != EXPECTED_CHANGED
        or manifest.get("unchanged_baseline_file_count") != 25
        or manifest.get("production_deployable") is not False or manifest.get("production_deployed") is not False
    ):
        raise ValueError("R116_MANIFEST_CONTRACT_REJECTED")
    manifest_files = manifest.get("files") or []
    listed: dict[str, dict[str, Any]] = {}
    for item in manifest_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or item["path"] in listed:
            raise ValueError("R116_MANIFEST_ENTRY_DUPLICATE_OR_INVALID")
        listed[item["path"]] = item
    if manifest.get("file_count") != 32 or len(manifest_files) != 32 or set(listed) != set(candidate_regular) or len(candidate_regular) != 32:
        raise ValueError("R116_CANDIDATE_FILESET_REJECTED")
    for name, raw in candidate_regular.items():
        item = listed[name]
        if item.get("bytes") != len(raw) or item.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"R116_CANDIDATE_HASH_REJECTED:{name}")
    unchanged = {name for name in baseline_payload if name in candidate_regular and baseline_payload[name] == candidate_regular[name]}
    changed_existing = {name for name in baseline_payload if name in candidate_regular and baseline_payload[name] != candidate_regular[name]}
    new_files = set(candidate_regular) - set(baseline_payload)
    if len(unchanged) != 25 or changed_existing != EXPECTED_CHANGED - NEW_FILES or new_files != NEW_FILES:
        raise ValueError("R116_R115_DIFF_SCOPE_REJECTED")
    patch_records, patch_aggregate_sha = patch_diff_evidence(baseline_payload, candidate_regular)
    if (
        manifest.get("patch_diff_file_count") != 7
        or manifest.get("patch_diff_sha256") != patch_aggregate_sha
        or manifest.get("patch_diffs") != patch_records
    ):
        raise ValueError("R116_PATCH_DIFF_EVIDENCE_REJECTED")
    tree_sha = runtime_tree_sha(candidate_regular)
    if manifest.get("candidate_runtime_tree_sha256") != tree_sha:
        raise ValueError("R116_RUNTIME_TREE_HASH_REJECTED")
    build_result = load_object(build_result_path)
    bootstrap = load_object(bootstrap_artifact)
    if (
        bootstrap.get("schema") != "evomind.g21_goal_runtime_bootstrap.v1"
        or bootstrap.get("run_id") != FIXED_RUN_ID
        or bootstrap.get("allocation") != FIXED_ALLOCATION
        or bootstrap.get("goal_record_status") != "blocked"
        or bootstrap.get("goal_spec_sha256") != canonical_sha256(bootstrap.get("goal_spec"))
        or bootstrap.get("human_baseline_evidence_sha256") != canonical_sha256(bootstrap.get("human_baseline_evidence"))
        or bootstrap.get("goal_board_sha256") != canonical_sha256(bootstrap.get("initial_goal_board"))
        or bootstrap.get("goal_spec_file_sha256") != manifest.get("goal_spec_sha256")
        or bootstrap.get("human_baseline_evidence_file_sha256") != manifest.get("human_baseline_evidence_sha256")
        or bootstrap.get("goal_board_file_sha256") != manifest.get("goal_board_file_sha256")
        or bootstrap.get("goal_board_sha256") != manifest.get("goal_board_sha256")
        or (bootstrap.get("goal_spec") or {}).get("run_id") != FIXED_RUN_ID
        or (bootstrap.get("goal_spec") or {}).get("allocation") != FIXED_ALLOCATION
        or (bootstrap.get("human_baseline_evidence") or {}).get("classification") != "HUMAN_BASELINE_UNDEFINED"
        or (bootstrap.get("human_baseline_evidence") or {}).get("completion", {}).get("training_authorized_by_this_artifact") is not False
        or (bootstrap.get("initial_goal_board") or {}).get("goal_record_status") != "blocked"
        or (bootstrap.get("initial_goal_board") or {}).get("completion_count") != 0
        or bootstrap.get("production_deployable") is not False
        or build_result.get("bootstrap_artifact_sha256") != sha256_file(bootstrap_artifact)
        or manifest.get("bootstrap_artifact_sha256") != sha256_file(bootstrap_artifact)
    ):
        raise ValueError("R116_BOOTSTRAP_ARTIFACT_REJECTED")
    if (
        build_result.get("schema") != MANIFEST_SCHEMA or build_result.get("status") != "built_and_verified"
        or build_result.get("zip_sha256") != sha256_file(candidate_zip)
        or build_result.get("manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != tree_sha
        or build_result.get("patch_diff_file_count") != 7
        or build_result.get("patch_diff_sha256") != patch_aggregate_sha
        or build_result.get("production_deployable") is not False or build_result.get("production_deployed") is not False
        or build_result.get("hpc_accessed") is not False or build_result.get("gpu_touched") is not False
        or build_result.get("remote_writes") != 0
    ):
        raise ValueError("R116_BUILD_RESULT_REJECTED")
    rounds: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="r116-independent-") as temporary:
        temporary_root = Path(temporary)
        for round_number in (1, 2):
            extract_root = temporary_root / f"round-{round_number}"
            extract_root.mkdir()
            with zipfile.ZipFile(candidate_zip) as archive:
                archive.extractall(extract_root)
            python_files = sorted((extract_root / "evomind_runtime").glob("*.py"))
            if len(python_files) != 32:
                raise ValueError(f"R116_PYTHON_COUNT_REJECTED:{round_number}")
            for path in python_files:
                compiled = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(path)],
                    cwd=extract_root, capture_output=True, text=True, check=False,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if compiled.returncode != 0:
                    raise RuntimeError(f"R116_PYCOMPILE_FAILED:{round_number}:{path.name}")
            smoke = subprocess.run(
                [sys.executable, "-c", SMOKE, str(extract_root), str(bootstrap_artifact)],
                cwd=ROOT, capture_output=True, text=True, check=False,
                env={**os.environ, "PYTHONPATH": str(extract_root), "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if smoke.returncode != 0:
                raise RuntimeError(f"R116_HTTP_SMOKE_FAILED:{round_number}:{smoke.stderr[-500:]}")
            rounds.append({"round": round_number, "py_compile": 32, "http": json.loads(smoke.stdout.strip().splitlines()[-1])})
    repo_checks = verify_repository_checks()
    test_artifacts = {
        "targeted_goal_builder_junit": verify_junit(
            targeted_junit, minimum_tests=55,
            required_classnames={
                "tests.test_g21_goal_runtime_overlay_clean", "tests.test_goal_board_http",
                "tests.test_goal_board_persistence", "tests.test_competition_goal",
                "tests.test_goal_release_scope", "tests.test_g21_goal_patch_scope",
            },
        ),
        "runtime_regression_junit": verify_junit(
            runtime_junit, minimum_tests=112,
            required_classnames={
                "tests.test_evomind_runtime", "tests.test_evomind_runtime_http_server",
                "tests.test_assistant_run_service", "tests.test_large_tool_result_artifacts",
            },
        ),
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": {
            "filename": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "candidate_zip_sha256": sha256_file(candidate_zip),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "build_result_sha256": sha256_file(build_result_path),
        "baseline_zip_sha256": R115_ZIP_SHA256,
        "baseline_source_manifest_sha256": R115_SOURCE_MANIFEST_SHA256,
        "candidate_runtime_tree_sha256": tree_sha,
        "bootstrap_artifact_sha256": sha256_file(bootstrap_artifact),
        "test_artifacts": test_artifacts,
        "verification_rounds": rounds,
        "checks": {
            "dual_extract_rounds": 2, "python_files_compiled": 32, "goal_http_smoke_rounds": 2,
            "baseline_unchanged_files": 25, "changed_files": 7,
            "unsafe_entries": 0, "duplicate_entries": 0, "symlinks": 0,
            **repo_checks, "hpc_accessed": False, "gpu_touched": False, "remote_writes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-zip", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--build-result", required=True)
    parser.add_argument("--baseline-zip", default=r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r115-progress-parser-r114-base-20260829.zip")
    parser.add_argument("--baseline-source-manifest", default=r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\evomind-runtime-r115-progress-parser-r114-base-source-manifest-20260829.json")
    parser.add_argument("--bootstrap-artifact", required=True)
    parser.add_argument("--targeted-junit", required=True)
    parser.add_argument("--runtime-junit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = verify_candidate(
        candidate_zip=Path(args.candidate_zip).resolve(strict=True),
        source_manifest_path=Path(args.source_manifest).resolve(strict=True),
        build_result_path=Path(args.build_result).resolve(strict=True),
        baseline_zip=Path(args.baseline_zip).resolve(strict=True),
        baseline_source_manifest_path=Path(args.baseline_source_manifest).resolve(strict=True),
        bootstrap_artifact=Path(args.bootstrap_artifact).resolve(strict=True),
        targeted_junit=Path(args.targeted_junit).resolve(strict=True), runtime_junit=Path(args.runtime_junit).resolve(strict=True),
        output=Path(args.output).resolve(),
    )
    print(json.dumps({"status": receipt["status"], "schema": receipt["schema"], "output": str(Path(args.output).resolve())}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
