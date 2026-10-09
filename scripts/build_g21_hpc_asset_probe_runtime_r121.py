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
RELEASE = "r121-g21-hpc-asset-probe-r120-base"
SCHEMA = "evomind.g21_hpc_asset_probe_runtime.r121.v1"
MANIFEST_NAME = "hpc-asset-probe-runtime-r121-manifest.json"
BASELINE_MANIFEST_NAME = "terminal-preserving-probe-runtime-manifest.json"
BASELINE_ZIP_SHA256 = "979224d6baf2d779357d2d2a7ffacd57637289dce11a716857c15fe913eafa3c"
BASELINE_TREE_SHA256 = "5998ea21823c121c23a421bb145d8e2266bb79ab59eccc91a16122c3626b61ce"
BASELINE_MANIFEST_SHA256 = "ba78b0f3c0759e86d8af5e89f10a1adc07a8c547f7afa11aedf684fa477125e4"
BASELINE_FILE_COUNT = 32
CHANGED_FILE = "evomind_runtime/tools.py"
RAW_SOURCE_SHA256 = "c5d3f63fa448c2097a0fb07c139a364f6c922170fb32818ed0fe38a0a5d17674"
NORMALIZED_SOURCE_SHA256 = "1f484f27ba89ab1f7d3e2215004d7286617fdfac377181a8bc878d3a5deda85b"
EXPECTED_HPC_OUTPUT_MAX_BYTES = 262_144
EXPECTED_MODEL_FILE_DETAILS_INCLUDED = False
EXPECTED_MODEL_CLOSURE_COMPLETE = True


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
        raise ValueError("R121_BASELINE_REGULAR_FILE_REQUIRED")
    if sha256_file(path) != BASELINE_ZIP_SHA256:
        raise ValueError("R121_BASELINE_ZIP_SHA_MISMATCH")
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"R121_BASELINE_DUPLICATE_PATH:{name}")
            entries[name] = archive.read(info)
    manifests = {name for name in entries if name.endswith("manifest.json")}
    if manifests != {BASELINE_MANIFEST_NAME}:
        raise ValueError("R121_BASELINE_MANIFEST_SET_REJECTED")
    manifest_raw = entries[BASELINE_MANIFEST_NAME]
    if sha256_bytes(manifest_raw) != BASELINE_MANIFEST_SHA256:
        raise ValueError("R121_BASELINE_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(runtime) != BASELINE_FILE_COUNT or any(not name.endswith(".py") for name in runtime):
        raise ValueError("R121_BASELINE_RUNTIME_FILESET_REJECTED")
    declared = {str(item.get("path")): item for item in manifest.get("files", []) if isinstance(item, Mapping)}
    if set(declared) != set(runtime) or len(declared) != BASELINE_FILE_COUNT:
        raise ValueError("R121_BASELINE_MANIFEST_CLOSURE_REJECTED")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R121_BASELINE_FILE_HASH_REJECTED:{name}")
    if runtime_tree_sha256(runtime) != BASELINE_TREE_SHA256:
        raise ValueError("R121_BASELINE_TREE_REJECTED")
    return runtime, manifest


def function_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"R121_FUNCTION_ANCHOR_REJECTED:{name}")
    node = matches[0]
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def replace_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"R121_BASELINE_FUNCTION_ANCHOR_REJECTED:{name}")
    node = matches[0]
    replacement = replacement.rstrip("\n") + "\n"
    return "".join(lines[: node.lineno - 1] + [replacement] + lines[node.end_lineno :])


def source_overlay(source_root: Path, baseline: Mapping[str, bytes]) -> tuple[bytes, dict[str, Any]]:
    path = source_root / CHANGED_FILE
    if not path.is_file() or path.is_symlink():
        raise ValueError("R121_TOOLS_SOURCE_REGULAR_FILE_REQUIRED")
    raw = path.read_bytes()
    if sha256_bytes(raw) != RAW_SOURCE_SHA256:
        raise ValueError("R121_TOOLS_SOURCE_RAW_SHA_MISMATCH")
    normalized = raw.decode("utf-8-sig").replace("\r\n", "\n")
    if sha256_bytes(normalized.encode("utf-8")) != NORMALIZED_SOURCE_SHA256:
        raise ValueError("R121_TOOLS_SOURCE_NORMALIZED_SHA_MISMATCH")
    baseline_text = baseline[CHANGED_FILE].decode("utf-8-sig").replace("\r\n", "\n")
    candidate = baseline_text
    patch_rows = []
    for name in ("_hpc_asset_probe_remote_source", "_parse_hpc_asset_probe_payload"):
        before = function_source(baseline_text, name)
        after = function_source(normalized, name)
        if before == after:
            raise ValueError(f"R121_FUNCTION_NOT_CHANGED:{name}")
        candidate = replace_function(candidate, name, after)
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"r120/{CHANGED_FILE}:{name}",
                tofile=f"r121/{CHANGED_FILE}:{name}",
            )
        ).encode("utf-8")
        patch_rows.append({"function": name, "baseline_bytes": len(before.encode()), "candidate_bytes": len(after.encode()), "diff_bytes": len(patch), "diff_sha256": sha256_bytes(patch)})
    ast.parse(candidate, filename=CHANGED_FILE)
    candidate_bytes = candidate.encode("utf-8")
    details = {
        "raw_source_sha256": RAW_SOURCE_SHA256,
        "normalized_source_sha256": NORMALIZED_SOURCE_SHA256,
        "packaged_tools_sha256": sha256_bytes(candidate_bytes),
        "functions": patch_rows,
        "output_bound": {
            "max_probe_stdout_bytes": EXPECTED_HPC_OUTPUT_MAX_BYTES,
            "model_file_details_included": EXPECTED_MODEL_FILE_DETAILS_INCLUDED,
            "model_metadata_closure_complete": EXPECTED_MODEL_CLOSURE_COMPLETE,
        },
    }
    return candidate_bytes, details


def extract_candidate(path: Path, destination: Path) -> None:
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in seen:
                raise ValueError(f"R121_CANDIDATE_DUPLICATE:{name}")
            seen.add(name)
            target = destination / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))


def verify_extraction(root: Path, expected: Mapping[str, bytes], manifest_sha: str, round_number: int) -> dict[str, Any]:
    actual = {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted((root / "evomind_runtime").glob("*.py"))}
    regular = [path for path in root.rglob("*") if path.is_file() and ".pycompile" not in path.parts]
    if set(actual) != set(expected) or any(actual[name] != expected[name] for name in actual):
        raise ValueError(f"R121_EXTRACTED_RUNTIME_REJECTED:{round_number}")
    manifests = [path for path in regular if path.name.endswith("manifest.json")]
    if len(manifests) != 1 or manifests[0].name != MANIFEST_NAME or sha256_file(manifests[0]) != manifest_sha:
        raise ValueError(f"R121_EXTRACTED_MANIFEST_REJECTED:{round_number}")
    if len(regular) != BASELINE_FILE_COUNT + 1:
        raise ValueError(f"R121_EXTRACTED_FILE_COUNT_REJECTED:{round_number}")
    return {"round": round_number, "regular_files": len(regular), "runtime_files": len(actual), "tree_sha256": runtime_tree_sha256(actual)}


def compile_candidate(root: Path) -> int:
    files = sorted((root / "evomind_runtime").glob("*.py"))
    if len(files) != BASELINE_FILE_COUNT:
        raise ValueError("R121_PY_COMPILE_COUNT_REJECTED")
    cache = root / ".pycompile"
    cache.mkdir()
    for index, path in enumerate(files):
        py_compile.compile(str(path), cfile=str(cache / f"{index}.pyc"), doraise=True)
    return len(files)


def run_smoke(root: Path, round_number: int) -> dict[str, Any]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root)}
    tests = [
        str(ROOT / "tests/test_hpc_asset_probe.py"),
        str(ROOT / "tests/test_large_tool_result_artifacts.py"),
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests, "--rootdir", str(ROOT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"R121_SMOKE_FAILED:{round_number}:{completed.stdout[-1200:]}:{completed.stderr[-1200:]}")
    return {"round": round_number, "hpc_asset_probe_tests": True, "output_bound_tests": True, "real_hpc_accessed": False, "pytest_summary": completed.stdout.strip().splitlines()[-1]}


def build(args: argparse.Namespace) -> dict[str, Any]:
    baseline, baseline_manifest = inspect_baseline(Path(args.baseline_zip).resolve())
    overlay, overlay_details = source_overlay(Path(args.source_root).resolve(), baseline)
    candidate = dict(baseline)
    candidate[CHANGED_FILE] = overlay
    changed = sorted(name for name in candidate if candidate[name] != baseline[name])
    unchanged = sorted(name for name in candidate if candidate[name] == baseline[name])
    if changed != [CHANGED_FILE] or len(unchanged) != 31:
        raise ValueError("R121_PATCH_SCOPE_REJECTED")
    tree = runtime_tree_sha256(candidate)
    patch_diff = "".join(
        difflib.unified_diff(
            baseline[CHANGED_FILE].decode("utf-8-sig").splitlines(keepends=True),
            overlay.decode("utf-8").splitlines(keepends=True),
            fromfile=f"r120/{CHANGED_FILE}",
            tofile=f"r121/{CHANGED_FILE}",
        )
    ).encode("utf-8")
    files = [{"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)} for name, raw in sorted(candidate.items())]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "release": RELEASE,
        "target": "bundle/runtime/evomind_runtime",
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_release": "r120-g21-terminal-preserving-probe-r119-base",
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree,
        "unchanged_file_count": 31,
        "unchanged_files": [{"path": name, "bytes": len(candidate[name]), "sha256": sha256_bytes(candidate[name])} for name in unchanged],
        "changed_file_count": 1,
        "changed_files": [CHANGED_FILE],
        "files": files,
        "source_tools_raw_sha256": RAW_SOURCE_SHA256,
        "source_tools_normalized_sha256": NORMALIZED_SOURCE_SHA256,
        "packaged_tools_sha256": sha256_bytes(overlay),
        "patch_diff_sha256": sha256_bytes(patch_diff),
        "patch_details": overlay_details,
        "output_bound": {
            "max_probe_stdout_bytes": EXPECTED_HPC_OUTPUT_MAX_BYTES,
            "oversized_payload_strategy": "bounded_model_summary_without_file_details",
            "model_file_details_included": EXPECTED_MODEL_FILE_DETAILS_INCLUDED,
            "model_metadata_closure_complete": EXPECTED_MODEL_CLOSURE_COMPLETE,
        },
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
            "candidate_runtime_file_count": BASELINE_FILE_COUNT,
            "candidate_runtime_tree_sha256": tree,
            "changed_files": changed,
            "unchanged_file_count": len(unchanged),
            "source_tools_raw_sha256": RAW_SOURCE_SHA256,
            "source_tools_normalized_sha256": NORMALIZED_SOURCE_SHA256,
            "packaged_tools_sha256": sha256_bytes(overlay),
            "output_bound": manifest["output_bound"],
            "production_deployable": False,
        }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"OUTPUT_DIRECTORY_EXISTS:{output_dir}")
    output_dir.mkdir(parents=True)
    prefix = "evomind-runtime-r121-g21-hpc-asset-probe-r120-base"
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
        with tempfile.TemporaryDirectory(prefix=f"r121-round-{number}-") as temporary:
            extracted = Path(temporary) / "candidate"; extracted.mkdir()
            extract_candidate(candidate_zip, extracted)
            checked = verify_extraction(extracted, candidate, manifest_sha, number)
            checked["py_compile"] = compile_candidate(extracted)
            checked["policy_smoke"] = run_smoke(extracted, number)
            rounds.append(checked)
    result = {
        "schema": SCHEMA,
        "status": "built_and_verified",
        "release": RELEASE,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "zip_filename": candidate_zip.name,
        "zip_bytes": candidate_zip.stat().st_size,
        "zip_sha256": sha256_file(candidate_zip),
        "source_manifest_filename": manifest_path.name,
        "source_manifest_sha256": manifest_sha,
        "candidate_runtime_file_count": BASELINE_FILE_COUNT,
        "candidate_runtime_tree_sha256": tree,
        "changed_file_count": 1,
        "unchanged_file_count": 31,
        "changed_files": changed,
        "source_tools_raw_sha256": RAW_SOURCE_SHA256,
        "source_tools_normalized_sha256": NORMALIZED_SOURCE_SHA256,
        "packaged_tools_sha256": sha256_bytes(overlay),
        "patch_diff_sha256": sha256_bytes(patch_diff),
        "output_bound": manifest["output_bound"],
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
    value = argparse.ArgumentParser(description="Build the fixed R121 output-bound asset-probe runtime from R120")
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
