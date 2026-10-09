from __future__ import annotations

import argparse
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
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from build_g21_conditional_policy_runtime_r117 import SMOKE  # noqa: E402


SCHEMA = "evomind.g21_conditional_policy_runtime.r117.v1"
RECEIPT_SCHEMA = "evomind.g21_conditional_policy_runtime_test_receipt.r117.v1"
FIXED_RUN_ID = "run_7b1efb878afb40f396db431e91f093a5"
FIXED_ALLOCATION = "G21"
MANIFEST_NAME = "conditional-policy-runtime-manifest.json"
BASELINE_MANIFEST_NAME = "goal-runtime-overlay-manifest.json"
STALE_MANIFEST_NAME = "runtime-hotfix-manifest.json"
BASELINE_ZIP_SHA256 = "2aa31006f33bb53de56afe6888c9eb03bc8597fa5334e3831e6fd788fda877c2"
BASELINE_TREE_SHA256 = "7e7b0babec6146640dc56cb42c05bbe37765bca2aca6348f42c5bf569ab69b38"
OLD_SPEC_FILE_SHA256 = "380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f"
OLD_SPEC_CANONICAL_SHA256 = "a569bae731723fc7046817df737876d6dc7674eadd9f7b812fe7b4a87e521655"
NEW_SPEC_FILE_SHA256 = "da8c3fbe8b14107dfadf96ad906b4cf982bc714ebf6420225a1e3722cd41a752"
NEW_SPEC_CANONICAL_SHA256 = "fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036"
POLICY_EVIDENCE_FILE_SHA256 = "47df860192502afcaacd7b1b6bc7fc4d86110c2e665278c8b7b3ea421893b7a7"
POLICY_EVIDENCE_CANONICAL_SHA256 = "f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0"
CHANGED_FILES = {
    "evomind_runtime/competition_goal.py",
    "evomind_runtime/goal_board.py",
    "evomind_runtime/store.py",
    "evomind_runtime/runtime.py",
    "evomind_runtime/http_server.py",
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
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def safe_entries(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
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
                raise ValueError(f"ZIP_ENTRY_REJECTED:{raw}")
            if info.is_dir():
                continue
            if raw in result:
                raise ValueError(f"ZIP_DUPLICATE:{raw}")
            result[raw] = archive.read(info)
    return result


def runtime_files(entries: Mapping[str, bytes]) -> dict[str, bytes]:
    result = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(result) != 32 or any(not name.endswith(".py") for name in result):
        raise ValueError("RUNTIME_FILESET_REJECTED")
    return result


def tree_sha(files: Mapping[str, bytes]) -> str:
    prefix = "evomind_runtime/"
    rows = [
        f"{name[len(prefix):]}|{len(raw)}|{sha256_bytes(raw)}"
        for name, raw in files.items()
        if name.startswith(prefix) and name != prefix
    ]
    if len(rows) != len(files):
        raise ValueError("TREE_PATH_REJECTED")
    return sha256_bytes("\n".join(sorted(rows)).encode("utf-8"))


def verify_evidence(path: Path, raw_sha: str, canonical_sha: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label}_REGULAR_FILE_REQUIRED")
    if sha256_file(path) != raw_sha:
        raise ValueError(f"{label}_RAW_SHA_MISMATCH")
    value = load_object(path)
    if canonical_sha256(value) != canonical_sha:
        raise ValueError(f"{label}_CANONICAL_SHA_MISMATCH")
    return value


def verify_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    if tests < 1 or failures or errors:
        raise ValueError(f"JUNIT_REJECTED:{path}")
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def extract(entries: Mapping[str, bytes], destination: Path) -> None:
    for name, raw in entries.items():
        target = destination / PurePosixPath(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def compile_round(root: Path) -> int:
    paths = sorted((root / "evomind_runtime").glob("*.py"))
    if len(paths) != 32:
        raise ValueError("PY_COMPILE_COUNT_REJECTED")
    cache = root / ".verify-pycompile"
    cache.mkdir()
    for index, path in enumerate(paths):
        py_compile.compile(str(path), cfile=str(cache / f"{index}.pyc"), doraise=True)
    return len(paths)


def smoke_round(
    root: Path,
    old_spec: Path,
    new_spec: Path,
    evidence: Path,
    round_number: int,
) -> dict[str, Any]:
    runtime_root = root.parent / f"verifier-runtime-{round_number}"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            SMOKE,
            str(root),
            str(runtime_root),
            str(old_spec),
            str(new_spec),
            str(evidence),
            str(round_number),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )
    if completed.returncode:
        raise ValueError(f"LOOPBACK_POLICY_SMOKE_FAILED:{round_number}:{completed.stderr[-1000:]}")
    rows = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not rows:
        raise ValueError(f"LOOPBACK_POLICY_SMOKE_RECEIPT_MISSING:{round_number}")
    value = json.loads(rows[-1])
    expected = {"round": round_number, "migration": True, "replay": True, "events": 1, "metadata": True}
    if value != expected:
        raise ValueError(f"LOOPBACK_POLICY_SMOKE_RECEIPT_REJECTED:{round_number}")
    return value


def verify(args: argparse.Namespace) -> dict[str, Any]:
    baseline_zip = Path(args.baseline_zip).resolve()
    candidate_zip = Path(args.candidate_zip).resolve()
    manifest_path = Path(args.source_manifest).resolve()
    build_result_path = Path(args.build_result).resolve()
    old_spec_path = Path(args.old_spec).resolve()
    new_spec_path = Path(args.new_spec).resolve()
    evidence_path = Path(args.policy_evidence).resolve()
    if sha256_file(baseline_zip) != BASELINE_ZIP_SHA256:
        raise ValueError("BASELINE_ZIP_SHA_MISMATCH")
    baseline_entries = safe_entries(baseline_zip)
    if set(baseline_entries) - {BASELINE_MANIFEST_NAME} != set(runtime_files(baseline_entries)):
        raise ValueError("BASELINE_ARCHIVE_SCOPE_REJECTED")
    baseline = runtime_files(baseline_entries)
    if tree_sha(baseline) != BASELINE_TREE_SHA256:
        raise ValueError("BASELINE_TREE_SHA_MISMATCH")
    candidate_entries = safe_entries(candidate_zip)
    if BASELINE_MANIFEST_NAME in candidate_entries or STALE_MANIFEST_NAME in candidate_entries:
        raise ValueError("STALE_MANIFEST_PRESENT")
    manifests = [name for name in candidate_entries if name.endswith("manifest.json")]
    if manifests != [MANIFEST_NAME]:
        raise ValueError("CANDIDATE_MANIFEST_SET_REJECTED")
    candidate = runtime_files(candidate_entries)
    if set(candidate_entries) != set(candidate) | {MANIFEST_NAME}:
        raise ValueError("CANDIDATE_ARCHIVE_SCOPE_REJECTED")
    changed = {name for name in candidate if candidate[name] != baseline[name]}
    unchanged = {name for name in candidate if candidate[name] == baseline[name]}
    if changed != CHANGED_FILES or len(unchanged) != 27:
        raise ValueError("CANDIDATE_PATCH_SCOPE_REJECTED")
    manifest_raw = candidate_entries[MANIFEST_NAME]
    if manifest_raw != manifest_path.read_bytes():
        raise ValueError("EMBEDDED_MANIFEST_BYTES_MISMATCH")
    manifest = load_object(manifest_path)
    build_result = load_object(build_result_path)
    candidate_sha = sha256_file(candidate_zip)
    manifest_sha = sha256_file(manifest_path)
    candidate_tree = tree_sha(candidate)
    required_manifest = {
        "schema": SCHEMA,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "candidate_runtime_tree_sha256": candidate_tree,
        "old_goal_spec_file_sha256": OLD_SPEC_FILE_SHA256,
        "old_goal_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
        "new_goal_spec_file_sha256": NEW_SPEC_FILE_SHA256,
        "new_goal_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
        "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
        "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
    }
    for key, value in required_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"MANIFEST_BINDING_MISMATCH:{key}")
    if (
        int(manifest.get("candidate_runtime_file_count", -1)) != 32
        or int(manifest.get("changed_file_count", -1)) != 5
        or int(manifest.get("unchanged_file_count", -1)) != 27
        or set(manifest.get("changed_files", [])) != CHANGED_FILES
        or bool(manifest.get("production_deployable"))
        or bool(manifest.get("production_deployed"))
    ):
        raise ValueError("MANIFEST_SCOPE_REJECTED")
    declared = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    if set(declared) != set(candidate) or len(declared) != 32:
        raise ValueError("MANIFEST_FILE_CLOSURE_REJECTED")
    for name, raw in candidate.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"MANIFEST_FILE_HASH_REJECTED:{name}")
    if (
        build_result.get("schema") != SCHEMA
        or build_result.get("status") != "built_and_verified"
        or build_result.get("zip_sha256") != candidate_sha
        or build_result.get("source_manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != candidate_tree
        or bool(build_result.get("production_deployable"))
        or bool(build_result.get("production_deployed"))
        or bool(build_result.get("hpc_accessed"))
        or bool(build_result.get("gpu_touched"))
        or int(build_result.get("remote_writes", -1)) != 0
    ):
        raise ValueError("BUILD_RESULT_REJECTED")
    verify_evidence(old_spec_path, OLD_SPEC_FILE_SHA256, OLD_SPEC_CANONICAL_SHA256, "OLD_SPEC")
    verify_evidence(new_spec_path, NEW_SPEC_FILE_SHA256, NEW_SPEC_CANONICAL_SHA256, "NEW_SPEC")
    verify_evidence(
        evidence_path,
        POLICY_EVIDENCE_FILE_SHA256,
        POLICY_EVIDENCE_CANONICAL_SHA256,
        "POLICY_EVIDENCE",
    )
    rounds: list[dict[str, Any]] = []
    for number in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"r117-independent-{number}-") as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            extract(candidate_entries, root)
            if tree_sha({p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "evomind_runtime").glob("*.py")}) != candidate_tree:
                raise ValueError(f"INDEPENDENT_TREE_MISMATCH:{number}")
            compiled = compile_round(root)
            smoke = smoke_round(root, old_spec_path, new_spec_path, evidence_path, number)
            rounds.append({"round": number, "py_compile": compiled, "policy_smoke": smoke})
    junit = [verify_junit(Path(path).resolve()) for path in args.junit]
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "verified",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "candidate_zip_filename": candidate_zip.name,
        "candidate_zip_sha256": candidate_sha,
        "source_manifest_filename": manifest_path.name,
        "source_manifest_sha256": manifest_sha,
        "build_result_filename": build_result_path.name,
        "build_result_sha256": sha256_file(build_result_path),
        "candidate_runtime_tree_sha256": candidate_tree,
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "candidate_runtime_file_count": 32,
        "changed_file_count": 5,
        "unchanged_file_count": 27,
        "changed_files": sorted(CHANGED_FILES),
        "old_goal_spec_file_sha256": OLD_SPEC_FILE_SHA256,
        "old_goal_spec_canonical_sha256": OLD_SPEC_CANONICAL_SHA256,
        "new_goal_spec_file_sha256": NEW_SPEC_FILE_SHA256,
        "new_goal_spec_canonical_sha256": NEW_SPEC_CANONICAL_SHA256,
        "policy_evidence_file_sha256": POLICY_EVIDENCE_FILE_SHA256,
        "policy_evidence_canonical_sha256": POLICY_EVIDENCE_CANONICAL_SHA256,
        "verifier": {"filename": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        "verification_rounds": rounds,
        "junit_receipts": junit,
        "production_deployable": False,
        "production_deployed": False,
        "hpc_accessed": False,
        "gpu_touched": False,
        "remote_writes": 0,
        "signals_sent": 0,
        "other_processes_modified": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Independently verify the R117 conditional-policy runtime")
    value.add_argument("--baseline-zip", required=True)
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
    value.add_argument("--old-spec", required=True)
    value.add_argument("--new-spec", required=True)
    value.add_argument("--policy-evidence", required=True)
    value.add_argument("--junit", action="append", default=[])
    value.add_argument("--output", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"schema": RECEIPT_SCHEMA, "status": "failed", "error": "OUTPUT_EXISTS"}))
        return 1
    try:
        receipt = verify(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    except Exception as exc:
        print(json.dumps({"schema": RECEIPT_SCHEMA, "status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({**receipt, "receipt_filename": output.name, "receipt_sha256": sha256_file(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
