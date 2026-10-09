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
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
TARGET_ALLOCATION = "G24"
TARGET_JOB_ID = 93015
RELEASE = "r123-managed-hpc-identity-rebind-r121-base"
SCHEMA = "evomind.managed_hpc_identity_rebind_runtime.r123.v1"
MANIFEST_NAME = "managed-hpc-identity-rebind-runtime-r123-manifest.json"
BASELINE_MANIFEST_NAME = "hpc-asset-probe-runtime-r121-manifest.json"
BASELINE_ZIP_SHA256 = "f927d3538b6b1f1a02ed4fb57312d2663296e98baa56940660c321058440f4d2"
BASELINE_TREE_SHA256 = "a9595ab4ed9a86786ca65f4456893d64741b449ec84cf8c9ddb42e9bf4110dc7"
BASELINE_MANIFEST_SHA256 = "d644dc05a01cc4fec0201716bbfdc7a108a7f412c9cc98edfe95ef7a3bac8da8"
RUNTIME_FILE_COUNT = 32
CHANGED_FILES = (
    "evomind_runtime/assistant_runs.py",
    "evomind_runtime/http_server.py",
    "evomind_runtime/store.py",
)
SOURCE_RAW_SHA256 = {
    "evomind_runtime/assistant_runs.py": "91cf9c71f5ad2aae0b60a3263d9d42d27cbe36b7c4c53fd127bd90b51e95a370",
    "evomind_runtime/http_server.py": "0a891a386b66abf3bb74ddf8811bb2efbce64f62c2716d83796360e1c8447df7",
    "evomind_runtime/store.py": "9ac49cbea4824b141fa72b5184baf2cce4b880069941249ff5703922e60a0f4a",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError(f"R123_ZIP_UNSAFE_PATH:{raw}")
    if stat.S_ISLNK(mode):
        raise ValueError(f"R123_ZIP_SYMLINK_REJECTED:{raw}")
    return raw


def runtime_tree_sha256(files: Mapping[str, bytes]) -> str:
    prefix = "evomind_runtime/"
    rows: list[str] = []
    for name, raw in files.items():
        if not name.startswith(prefix) or name == prefix:
            raise ValueError(f"R123_RUNTIME_TREE_PATH_REJECTED:{name}")
        rows.append(f"{name[len(prefix):]}|{len(raw)}|{sha256_bytes(raw)}")
    return sha256_bytes("\n".join(sorted(rows)).encode("utf-8"))


def inspect_baseline(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("R123_BASELINE_REGULAR_FILE_REQUIRED")
    if sha256_file(path) != BASELINE_ZIP_SHA256:
        raise ValueError("R123_BASELINE_ZIP_SHA_MISMATCH")
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"R123_BASELINE_DUPLICATE_PATH:{name}")
            entries[name] = archive.read(info)
    manifests = {name for name in entries if name.endswith("manifest.json")}
    if manifests != {BASELINE_MANIFEST_NAME}:
        raise ValueError("R123_BASELINE_MANIFEST_SET_REJECTED")
    manifest_raw = entries[BASELINE_MANIFEST_NAME]
    if sha256_bytes(manifest_raw) != BASELINE_MANIFEST_SHA256:
        raise ValueError("R123_BASELINE_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(runtime) != RUNTIME_FILE_COUNT or any(not name.endswith(".py") for name in runtime):
        raise ValueError("R123_BASELINE_RUNTIME_FILESET_REJECTED")
    declared = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    if set(declared) != set(runtime) or len(declared) != RUNTIME_FILE_COUNT:
        raise ValueError("R123_BASELINE_MANIFEST_CLOSURE_REJECTED")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R123_BASELINE_FILE_HASH_REJECTED:{name}")
    if runtime_tree_sha256(runtime) != BASELINE_TREE_SHA256:
        raise ValueError("R123_BASELINE_TREE_REJECTED")
    return runtime, manifest


def _module_function_node(text: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"R123_MODULE_FUNCTION_ANCHOR_REJECTED:{name}")
    return matches[0]


def _class_method_node(text: str, class_name: str, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(text)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"R123_CLASS_ANCHOR_REJECTED:{class_name}")
    matches = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    if len(matches) != 1:
        raise ValueError(f"R123_CLASS_METHOD_ANCHOR_REJECTED:{class_name}.{method_name}")
    return matches[0]


def _node_source(text: str, node: ast.AST) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno]).rstrip("\n") + "\n"


def module_function_source(text: str, name: str) -> str:
    return _node_source(text, _module_function_node(text, name))


def class_method_source(text: str, class_name: str, method_name: str) -> str:
    return _node_source(text, _class_method_node(text, class_name, method_name))


def replace_class_method(text: str, class_name: str, method_name: str, replacement: str) -> str:
    node = _class_method_node(text, class_name, method_name)
    lines = text.splitlines(keepends=True)
    newline = newline_style(text)
    value = with_newline(replacement, newline).rstrip("\r\n") + newline
    return "".join(lines[: node.lineno - 1] + [value] + lines[node.end_lineno :])


def insert_before_class_method(text: str, class_name: str, method_name: str, addition: str) -> str:
    node = _class_method_node(text, class_name, method_name)
    lines = text.splitlines(keepends=True)
    newline = newline_style(text)
    value = with_newline(addition, newline).rstrip("\r\n") + newline + newline
    return "".join(lines[: node.lineno - 1] + [value] + lines[node.lineno - 1 :])


def insert_before_module_function(text: str, name: str, addition: str) -> str:
    node = _module_function_node(text, name)
    lines = text.splitlines(keepends=True)
    newline = newline_style(text)
    value = with_newline(addition, newline).rstrip("\r\n") + newline + newline
    return "".join(lines[: node.lineno - 1] + [value] + lines[node.lineno - 1 :])


def replace_once(text: str, old: str, new: str, code: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"R123_TEXT_ANCHOR_REJECTED:{code}")
    return text.replace(old, new, 1)


def newline_style(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def with_newline(text: str, newline: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", newline)


def load_source(source_root: Path, name: str) -> str:
    path = source_root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"R123_SOURCE_REGULAR_FILE_REQUIRED:{name}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != SOURCE_RAW_SHA256[name]:
        raise ValueError(f"R123_SOURCE_RAW_SHA_MISMATCH:{name}")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    ast.parse(text, filename=name)
    return text


def assistant_overlay(source_root: Path, baseline: bytes) -> bytes:
    name = "evomind_runtime/assistant_runs.py"
    source = load_source(source_root, name)
    candidate = baseline.decode("utf-8-sig")
    newline = newline_style(candidate)
    candidate = replace_once(
        candidate,
        with_newline('    "goal_updated",\n}', newline),
        with_newline('    "goal_updated",\n    "managed_hpc_identity.rebound",\n}', newline),
        "assistant_public_event",
    )
    constants = """_MANAGED_HPC_TENANT = re.compile(r"^tenant_[a-f0-9]{24}$")
_MANAGED_HPC_OWNER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MANAGED_HPC_PROFILE_INSTANCE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_MANAGED_HPC_BINDING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_MANAGED_HPC_REBIND_KEY = re.compile(r"^[A-Za-z0-9_.:-]{8,180}$")
_MANAGED_HPC_IDENTITY_FIELDS = (
    "tenant_id",
    "owner_principal_id",
    "job_id",
    "credential_profile",
    "allocation_generation",
    "profile_instance_id",
    "allocation_binding_id",
)"""
    candidate = replace_once(
        candidate,
        with_newline('_SHA256 = re.compile(r"^[a-f0-9]{64}$")\n_ARTIFACT_FILE', newline),
        with_newline('_SHA256 = re.compile(r"^[a-f0-9]{64}$")\n' + constants + "\n_ARTIFACT_FILE", newline),
        "assistant_constants",
    )
    helpers = "\n\n".join(
        module_function_source(source, function).rstrip("\n")
        for function in ("_validated_managed_hpc_identity", "_managed_hpc_identity_fingerprint")
    )
    candidate = insert_before_module_function(
        candidate, "_artifact_requirements_from_prompt", with_newline(helpers, newline)
    )
    rebind_method = class_method_source(source, "AssistantRunService", "rebind_managed_hpc_identity")
    candidate = insert_before_class_method(
        candidate, "AssistantRunService", "action", with_newline(rebind_method, newline)
    )
    candidate = replace_class_method(
        candidate,
        "AssistantRunService",
        "action",
        with_newline(class_method_source(source, "AssistantRunService", "action"), newline),
    )
    ast.parse(candidate, filename=name)
    return candidate.encode("utf-8")


def store_overlay(source_root: Path, baseline: bytes) -> bytes:
    name = "evomind_runtime/store.py"
    source = load_source(source_root, name)
    candidate = baseline.decode("utf-8-sig")
    newline = newline_style(candidate)
    method = class_method_source(source, "RuntimeStore", "rebind_session_managed_hpc_identity")
    candidate = insert_before_class_method(
        candidate, "RuntimeStore", "get_session", with_newline(method, newline)
    )
    ast.parse(candidate, filename=name)
    return candidate.encode("utf-8")


def http_overlay(source_root: Path, baseline: bytes) -> bytes:
    name = "evomind_runtime/http_server.py"
    load_source(source_root, name)
    candidate = baseline.decode("utf-8-sig")
    newline = newline_style(candidate)
    old = """                            ),
                        ),
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":"""
    new = """                            ),
                            idempotency_key=str(body.get("idempotency_key") or ""),
                        ),
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"] and parts[3] == "messages":"""
    candidate = replace_once(
        candidate,
        with_newline(old, newline),
        with_newline(new, newline),
        "http_action_idempotency",
    )
    ast.parse(candidate, filename=name)
    return candidate.encode("utf-8")


def build_overlays(source_root: Path, baseline: Mapping[str, bytes]) -> dict[str, bytes]:
    return {
        "evomind_runtime/assistant_runs.py": assistant_overlay(
            source_root, baseline["evomind_runtime/assistant_runs.py"]
        ),
        "evomind_runtime/http_server.py": http_overlay(
            source_root, baseline["evomind_runtime/http_server.py"]
        ),
        "evomind_runtime/store.py": store_overlay(
            source_root, baseline["evomind_runtime/store.py"]
        ),
    }


def extract_candidate(path: Path, destination: Path) -> None:
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in seen:
                raise ValueError(f"R123_CANDIDATE_DUPLICATE:{name}")
            seen.add(name)
            target = destination / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))


def verify_extraction(
    root: Path,
    expected: Mapping[str, bytes],
    manifest_sha: str,
    round_number: int,
) -> dict[str, Any]:
    actual = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "evomind_runtime").glob("*.py"))
    }
    regular = [path for path in root.rglob("*") if path.is_file() and ".pycompile" not in path.parts]
    if set(actual) != set(expected) or any(actual[name] != expected[name] for name in actual):
        raise ValueError(f"R123_EXTRACTED_RUNTIME_REJECTED:{round_number}")
    manifests = [path for path in regular if path.name.endswith("manifest.json")]
    if len(manifests) != 1 or manifests[0].name != MANIFEST_NAME or sha256_file(manifests[0]) != manifest_sha:
        raise ValueError(f"R123_EXTRACTED_MANIFEST_REJECTED:{round_number}")
    if len(regular) != RUNTIME_FILE_COUNT + 1:
        raise ValueError(f"R123_EXTRACTED_FILE_COUNT_REJECTED:{round_number}")
    return {
        "round": round_number,
        "regular_files": len(regular),
        "runtime_files": len(actual),
        "tree_sha256": runtime_tree_sha256(actual),
    }


def compile_candidate(root: Path) -> int:
    files = sorted((root / "evomind_runtime").glob("*.py"))
    if len(files) != RUNTIME_FILE_COUNT:
        raise ValueError("R123_PY_COMPILE_COUNT_REJECTED")
    cache = root / ".pycompile"
    cache.mkdir()
    for index, path in enumerate(files):
        py_compile.compile(str(path), cfile=str(cache / f"{index}.pyc"), doraise=True)
    return len(files)


def parse_junit_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("R123_JUNIT_REGULAR_FILE_REQUIRED")
    document = ET.parse(path).getroot()
    suites = [document] if document.tag.rsplit("}", 1)[-1] == "testsuite" else [
        item for item in document if item.tag.rsplit("}", 1)[-1] == "testsuite"
    ]
    if not suites:
        raise ValueError("R123_JUNIT_TESTSUITE_REQUIRED")
    totals = {
        key: sum(int(str(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] < 1:
        raise ValueError("R123_JUNIT_ZERO_TESTS_REJECTED")
    if totals["failures"] != 0 or totals["errors"] != 0:
        raise ValueError("R123_JUNIT_FAILURES_REJECTED")
    return {**totals, "sha256": sha256_file(path)}


def run_smoke(root: Path, round_number: int) -> dict[str, Any]:
    smoke_tests = root / ".r123-tests"
    smoke_tests.mkdir()
    selected = (
        ROOT / "tests/test_assistant_run_service.py",
        ROOT / "tests/test_evomind_runtime_http_server.py",
    )
    for source in selected:
        shutil.copy2(source, smoke_tests / source.name)
    junit_path = smoke_tests / "pytest-junit.xml"
    expected_hashes = {
        name: sha256_file(root / "evomind_runtime" / f"{name}.py")
        for name in ("assistant_runs", "store", "http_server")
    }
    pytest_args = [
        "-q",
        str(smoke_tests / "test_assistant_run_service.py"),
        str(smoke_tests / "test_evomind_runtime_http_server.py"),
        "-k",
        "managed_hpc_rebind or fixed_run_rebind or assistant_action_route_propagates_rebind_idempotency_key",
        "--confcutdir",
        str(smoke_tests),
        "--maxfail=1",
        f"--junitxml={junit_path}",
    ]
    isolated_program = "\n".join(
        (
            "import hashlib",
            "import sys",
            "from pathlib import Path",
            f"root = Path({str(root)!r}).resolve()",
            "sys.path.insert(0, str(root))",
            "import evomind_runtime.assistant_runs as assistant_runs",
            "import evomind_runtime.store as store",
            "import evomind_runtime.http_server as http_server",
            f"expected = {expected_hashes!r}",
            "modules = {'assistant_runs': assistant_runs, 'store': store, 'http_server': http_server}",
            "for name, module in modules.items():",
            "    path = Path(module.__file__).resolve()",
            "    assert path == root / 'evomind_runtime' / f'{name}.py', (name, path, root)",
            "    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name]",
            "import pytest",
            f"raise SystemExit(pytest.main({pytest_args!r}))",
        )
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-I", "-c", isolated_program],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"R123_SMOKE_FAILED:{round_number}:{completed.stdout[-2000:]}:{completed.stderr[-2000:]}"
        )
    junit = parse_junit_receipt(junit_path)
    return {
        "round": round_number,
        "candidate_import_origin": True,
        "candidate_import_hashes": expected_hashes,
        "isolated_python": True,
        "managed_hpc_rebind_tests": True,
        "http_idempotency_route_test": True,
        "real_hpc_accessed": False,
        "junit": junit,
        "pytest_summary": completed.stdout.strip().splitlines()[-1],
    }


def diff_record(name: str, before: bytes, after: bytes) -> dict[str, Any]:
    patch = "".join(
        difflib.unified_diff(
            before.decode("utf-8-sig").splitlines(keepends=True),
            after.decode("utf-8-sig").splitlines(keepends=True),
            fromfile=f"r121/{name}",
            tofile=f"r123/{name}",
        )
    ).encode("utf-8")
    return {
        "path": name,
        "baseline_bytes": len(before),
        "baseline_sha256": sha256_bytes(before),
        "candidate_bytes": len(after),
        "candidate_sha256": sha256_bytes(after),
        "source_raw_sha256": SOURCE_RAW_SHA256[name],
        "diff_bytes": len(patch),
        "diff_sha256": sha256_bytes(patch),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    baseline, baseline_manifest = inspect_baseline(Path(args.baseline_zip).resolve())
    overlays = build_overlays(Path(args.source_root).resolve(), baseline)
    candidate = dict(baseline)
    candidate.update(overlays)
    changed = sorted(name for name in candidate if candidate[name] != baseline[name])
    unchanged = sorted(name for name in candidate if candidate[name] == baseline[name])
    if changed != sorted(CHANGED_FILES) or len(unchanged) != 29:
        raise ValueError("R123_PATCH_SCOPE_REJECTED")
    if candidate["evomind_runtime/tools.py"] != baseline["evomind_runtime/tools.py"]:
        raise ValueError("R123_TOOLS_DRIFT_REJECTED")
    if candidate["evomind_runtime/__init__.py"] != baseline["evomind_runtime/__init__.py"]:
        raise ValueError("R123_INIT_DRIFT_REJECTED")
    tree = runtime_tree_sha256(candidate)
    patch_details = [diff_record(name, baseline[name], candidate[name]) for name in changed]
    files = [
        {"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)}
        for name, raw in sorted(candidate.items())
    ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "release": RELEASE,
        "target": "bundle/runtime/evomind_runtime",
        "run_id": FIXED_RUN_ID,
        "target_allocation": TARGET_ALLOCATION,
        "target_job_id": TARGET_JOB_ID,
        "baseline_release": str(baseline_manifest.get("release") or "r121-g21-hpc-asset-probe-r120-base"),
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_runtime_file_count": RUNTIME_FILE_COUNT,
        "candidate_runtime_file_count": RUNTIME_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree,
        "unchanged_file_count": 29,
        "unchanged_files": [
            {"path": name, "bytes": len(candidate[name]), "sha256": sha256_bytes(candidate[name])}
            for name in unchanged
        ],
        "changed_file_count": 3,
        "changed_files": changed,
        "files": files,
        "source_raw_sha256": dict(SOURCE_RAW_SHA256),
        "patch_details": patch_details,
        "rebind_contract": {
            "fixed_run_only": True,
            "terminal_run_required": True,
            "effective_active_required": 0,
            "pending_approvals_required": 0,
            "live_controlled_secrets_required": 0,
            "tenant_and_principal_immutable": True,
            "allocation_generation_must_increase": True,
            "event_exactly_once": "managed_hpc_identity.rebound",
            "run_resumed": False,
            "hpc_accessed": False,
        },
        "goal_identity_handling": "preserve_legacy_g21_goal_label_without_using_it_as_g24_identity_evidence",
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
            "candidate_runtime_file_count": RUNTIME_FILE_COUNT,
            "candidate_runtime_tree_sha256": tree,
            "changed_files": changed,
            "unchanged_file_count": len(unchanged),
            "source_raw_sha256": dict(SOURCE_RAW_SHA256),
            "patch_details": patch_details,
            "production_deployable": False,
            "hpc_accessed": False,
            "gpu_touched": False,
        }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"R123_OUTPUT_DIRECTORY_EXISTS:{output_dir}")
    output_dir.mkdir(parents=True)
    prefix = "evomind-runtime-r123-managed-hpc-identity-rebind-r121-base"
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
    rounds = []
    for number in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"r123-build-round-{number}-") as temporary:
            extracted = Path(temporary) / "candidate"
            extracted.mkdir()
            extract_candidate(candidate_zip, extracted)
            checked = verify_extraction(extracted, candidate, manifest_sha, number)
            checked["py_compile"] = compile_candidate(extracted)
            checked["contract_smoke"] = run_smoke(extracted, number)
            rounds.append(checked)
    result = {
        "schema": SCHEMA,
        "status": "built_and_verified",
        "release": RELEASE,
        "run_id": FIXED_RUN_ID,
        "target_allocation": TARGET_ALLOCATION,
        "target_job_id": TARGET_JOB_ID,
        "zip_filename": candidate_zip.name,
        "zip_bytes": candidate_zip.stat().st_size,
        "zip_sha256": sha256_file(candidate_zip),
        "source_manifest_filename": manifest_path.name,
        "source_manifest_sha256": manifest_sha,
        "candidate_runtime_file_count": RUNTIME_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree,
        "changed_file_count": 3,
        "unchanged_file_count": 29,
        "changed_files": changed,
        "patch_details": patch_details,
        "verification_rounds": rounds,
        "production_deployable": False,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
    }
    write_json(build_result_path, result)
    return {**result, "build_result_filename": build_result_path.name}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build the fixed R123 managed HPC identity rebind runtime from frozen R121"
    )
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
