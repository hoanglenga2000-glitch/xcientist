from __future__ import annotations

import argparse
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


RECEIPT_SCHEMA = "evomind.managed_hpc_identity_rebind_runtime_test_receipt.r123.v1"


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
        or stat.S_ISLNK(mode)
    ):
        raise ValueError(f"R123_UNSAFE_ZIP_MEMBER:{info.filename}")
    return pure.as_posix()


def _read_zip(path: Path, code: str) -> dict[str, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"R123_{code}_REGULAR_FILE_REQUIRED")
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"R123_{code}_DUPLICATE:{name}")
            entries[name] = archive.read(info)
    return entries


def runtime_tree_sha256(files: Mapping[str, bytes]) -> str:
    prefix = "evomind_runtime/"
    rows: list[str] = []
    for name, raw in files.items():
        if not name.startswith(prefix) or name == prefix:
            raise ValueError(f"R123_RUNTIME_TREE_PATH_REJECTED:{name}")
        rows.append(f"{name[len(prefix):]}|{len(raw)}|{sha256_bytes(raw)}")
    return sha256_bytes("\n".join(sorted(rows)).encode("utf-8"))


def inspect_baseline(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    if sha256_file(path) != BASELINE_ZIP_SHA256:
        raise ValueError("R123_BASELINE_ZIP_SHA_MISMATCH")
    entries = _read_zip(path, "BASELINE")
    manifests = {name for name in entries if name.endswith("manifest.json")}
    if manifests != {BASELINE_MANIFEST_NAME}:
        raise ValueError("R123_BASELINE_MANIFEST_SET_REJECTED")
    manifest_raw = entries[BASELINE_MANIFEST_NAME]
    if sha256_bytes(manifest_raw) != BASELINE_MANIFEST_SHA256:
        raise ValueError("R123_BASELINE_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    declared = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    if (
        len(runtime) != RUNTIME_FILE_COUNT
        or any(not name.endswith(".py") for name in runtime)
        or set(declared) != set(runtime)
        or len(declared) != RUNTIME_FILE_COUNT
    ):
        raise ValueError("R123_BASELINE_MANIFEST_CLOSURE_REJECTED")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R123_BASELINE_FILE_HASH_REJECTED:{name}")
    if runtime_tree_sha256(runtime) != BASELINE_TREE_SHA256:
        raise ValueError("R123_BASELINE_TREE_REJECTED")
    return runtime, manifest


def extract_candidate(path: Path, destination: Path) -> None:
    entries = _read_zip(path, "CANDIDATE")
    for name, raw in entries.items():
        target = destination / PurePosixPath(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def verify_extraction(
    root: Path,
    expected: Mapping[str, bytes],
    manifest_sha: str,
    round_number: int,
) -> dict[str, Any]:
    actual = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "evomind_runtime").glob("*.py"))
        if path.is_file() and not path.is_symlink()
    }
    regular = [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and ".pycompile" not in path.parts
    ]
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
    document = ET.parse(path).getroot()
    suites = [document] if document.tag.rsplit("}", 1)[-1] == "testsuite" else [
        item for item in document if item.tag.rsplit("}", 1)[-1] == "testsuite"
    ]
    totals = {
        key: sum(int(str(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if not suites or totals["tests"] < 1:
        raise ValueError("R123_JUNIT_ZERO_TESTS_REJECTED")
    if totals["failures"] or totals["errors"]:
        raise ValueError("R123_JUNIT_FAILURES_REJECTED")
    return {**totals, "sha256": sha256_file(path)}


def run_smoke(root: Path, round_number: int) -> dict[str, Any]:
    smoke_tests = root / ".r123-tests"
    smoke_tests.mkdir()
    for source in (
        ROOT / "tests/test_assistant_run_service.py",
        ROOT / "tests/test_evomind_runtime_http_server.py",
    ):
        shutil.copy2(source, smoke_tests / source.name)
    junit_path = smoke_tests / "pytest-junit.xml"
    expected_hashes = {
        name: sha256_file(root / "evomind_runtime" / f"{name}.py")
        for name in ("assistant_runs", "store", "http_server")
    }
    arguments = [
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
    program = "\n".join(
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
            "    assert path == root / 'evomind_runtime' / f'{name}.py'",
            "    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name]",
            "import pytest",
            f"raise SystemExit(pytest.main({arguments!r}))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"R123_INDEPENDENT_SMOKE_FAILED:{round_number}:{completed.stdout[-2000:]}:{completed.stderr[-2000:]}"
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


def verified_rounds(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    for item in value:
        if not isinstance(item, dict) or int(item.get("py_compile", 0)) != RUNTIME_FILE_COUNT:
            return False
        smoke = item.get("contract_smoke")
        if not isinstance(smoke, dict) or smoke.get("isolated_python") is not True:
            return False
        if smoke.get("candidate_import_origin") is not True:
            return False
        hashes = smoke.get("candidate_import_hashes")
        if not isinstance(hashes, dict) or set(hashes) != {"assistant_runs", "store", "http_server"}:
            return False
        junit = smoke.get("junit")
        if (
            not isinstance(junit, dict)
            or int(junit.get("tests", 0)) < 1
            or int(junit.get("failures", -1)) != 0
            or int(junit.get("errors", -1)) != 0
        ):
            return False
    return True


def load_object(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"R123_{code}_REGULAR_FILE_REQUIRED")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"R123_{code}_OBJECT_REQUIRED")
    return value


def inspect_candidate(path: Path) -> dict[str, bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("R123_CANDIDATE_REGULAR_FILE_REQUIRED")
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"R123_CANDIDATE_DUPLICATE:{name}")
            entries[name] = archive.read(info)
    manifests = {name for name in entries if name.endswith("manifest.json")}
    if manifests != {MANIFEST_NAME}:
        raise ValueError("R123_CANDIDATE_MANIFEST_SET_REJECTED")
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(runtime) != RUNTIME_FILE_COUNT or len(entries) != RUNTIME_FILE_COUNT + 1:
        raise ValueError("R123_CANDIDATE_FILESET_REJECTED")
    return entries


def verify(args: argparse.Namespace) -> dict[str, Any]:
    candidate_zip = Path(args.candidate_zip).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    build_result_path = Path(args.build_result).resolve()
    baseline_zip = Path(args.baseline_zip).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise ValueError(f"R123_RECEIPT_EXISTS:{output_path}")

    baseline, _ = inspect_baseline(baseline_zip)
    entries = inspect_candidate(candidate_zip)
    embedded_manifest = entries[MANIFEST_NAME]
    if embedded_manifest != source_manifest_path.read_bytes():
        raise ValueError("R123_EMBEDDED_MANIFEST_BINDING_REJECTED")
    manifest = load_object(source_manifest_path, "SOURCE_MANIFEST")
    build_result = load_object(build_result_path, "BUILD_RESULT")
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    tree = runtime_tree_sha256(runtime)
    changed = sorted(name for name in runtime if runtime[name] != baseline[name])
    unchanged = sorted(name for name in runtime if runtime[name] == baseline[name])
    if changed != sorted(CHANGED_FILES) or len(unchanged) != 29:
        raise ValueError("R123_PATCH_SCOPE_REJECTED")
    for name in unchanged:
        if runtime[name] != baseline[name]:
            raise ValueError(f"R123_UNCHANGED_FILE_DRIFT:{name}")
    if runtime["evomind_runtime/tools.py"] != baseline["evomind_runtime/tools.py"]:
        raise ValueError("R123_TOOLS_DRIFT_REJECTED")
    if runtime["evomind_runtime/__init__.py"] != baseline["evomind_runtime/__init__.py"]:
        raise ValueError("R123_INIT_DRIFT_REJECTED")

    declared = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    if set(declared) != set(runtime) or len(declared) != RUNTIME_FILE_COUNT:
        raise ValueError("R123_MANIFEST_FILE_CLOSURE_REJECTED")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R123_MANIFEST_FILE_HASH_REJECTED:{name}")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("release") != RELEASE
        or manifest.get("target") != "bundle/runtime/evomind_runtime"
        or manifest.get("run_id") != FIXED_RUN_ID
        or manifest.get("target_allocation") != TARGET_ALLOCATION
        or manifest.get("target_job_id") != TARGET_JOB_ID
        or manifest.get("baseline_zip_sha256") != BASELINE_ZIP_SHA256
        or manifest.get("baseline_runtime_tree_sha256") != BASELINE_TREE_SHA256
        or manifest.get("baseline_manifest_sha256") != BASELINE_MANIFEST_SHA256
        or manifest.get("candidate_runtime_file_count") != RUNTIME_FILE_COUNT
        or manifest.get("candidate_runtime_tree_sha256") != tree
        or manifest.get("changed_file_count") != 3
        or manifest.get("unchanged_file_count") != 29
        or sorted(manifest.get("changed_files") or []) != sorted(CHANGED_FILES)
        or manifest.get("source_raw_sha256") != SOURCE_RAW_SHA256
        or manifest.get("goal_identity_handling")
        != "preserve_legacy_g21_goal_label_without_using_it_as_g24_identity_evidence"
        or manifest.get("production_deployable") is not False
        or manifest.get("production_deployed") is not False
        or manifest.get("hpc_accessed") is not False
        or manifest.get("gpu_touched") is not False
        or manifest.get("remote_writes") != 0
    ):
        raise ValueError("R123_MANIFEST_BINDING_REJECTED")
    contract = manifest.get("rebind_contract") if isinstance(manifest.get("rebind_contract"), Mapping) else {}
    if (
        contract.get("fixed_run_only") is not True
        or contract.get("terminal_run_required") is not True
        or contract.get("effective_active_required") != 0
        or contract.get("pending_approvals_required") != 0
        or contract.get("live_controlled_secrets_required") != 0
        or contract.get("tenant_and_principal_immutable") is not True
        or contract.get("allocation_generation_must_increase") is not True
        or contract.get("event_exactly_once") != "managed_hpc_identity.rebound"
        or contract.get("run_resumed") is not False
        or contract.get("hpc_accessed") is not False
    ):
        raise ValueError("R123_REBIND_CONTRACT_REJECTED")
    manifest_sha = sha256_file(source_manifest_path)
    if (
        build_result.get("schema") != SCHEMA
        or build_result.get("status") != "built_and_verified"
        or build_result.get("run_id") != FIXED_RUN_ID
        or build_result.get("target_allocation") != TARGET_ALLOCATION
        or build_result.get("target_job_id") != TARGET_JOB_ID
        or build_result.get("zip_sha256") != sha256_file(candidate_zip)
        or build_result.get("source_manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != tree
        or build_result.get("changed_file_count") != 3
        or build_result.get("unchanged_file_count") != 29
        or sorted(build_result.get("changed_files") or []) != sorted(CHANGED_FILES)
        or not verified_rounds(build_result.get("verification_rounds"))
        or build_result.get("production_deployable") is not False
        or build_result.get("production_deployed") is not False
        or build_result.get("hpc_accessed") is not False
        or build_result.get("gpu_touched") is not False
        or build_result.get("remote_writes") != 0
    ):
        raise ValueError("R123_BUILD_RESULT_REJECTED")

    rounds = []
    for number in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"r123-independent-round-{number}-") as temporary:
            extracted = Path(temporary) / "candidate"
            extracted.mkdir()
            extract_candidate(candidate_zip, extracted)
            checked = verify_extraction(extracted, runtime, manifest_sha, number)
            checked["py_compile"] = compile_candidate(extracted)
            checked["contract_smoke"] = run_smoke(extracted, number)
            rounds.append(checked)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "verified",
        "run_id": FIXED_RUN_ID,
        "target_allocation": TARGET_ALLOCATION,
        "target_job_id": TARGET_JOB_ID,
        "candidate_filename": candidate_zip.name,
        "candidate_zip_bytes": candidate_zip.stat().st_size,
        "candidate_zip_sha256": sha256_file(candidate_zip),
        "source_manifest_filename": source_manifest_path.name,
        "source_manifest_sha256": manifest_sha,
        "build_result_filename": build_result_path.name,
        "build_result_sha256": sha256_file(build_result_path),
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "candidate_runtime_file_count": RUNTIME_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree,
        "changed_file_count": 3,
        "unchanged_file_count": 29,
        "changed_files": changed,
        "source_raw_sha256": dict(SOURCE_RAW_SHA256),
        "verification_rounds": rounds,
        "production_deployable": False,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, receipt)
    return {**receipt, "test_receipt_filename": output_path.name}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Independently verify the R123 rebind runtime")
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
    value.add_argument("--baseline-zip", required=True)
    value.add_argument("--output", required=True)
    return value


def main() -> int:
    try:
        result = verify(parser().parse_args())
    except Exception as exc:
        print(json.dumps({"schema": RECEIPT_SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
