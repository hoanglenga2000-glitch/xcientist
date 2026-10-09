from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

from build_g21_hpc_asset_probe_runtime_r121 import (
    BASELINE_FILE_COUNT,
    BASELINE_MANIFEST_SHA256,
    BASELINE_TREE_SHA256,
    BASELINE_ZIP_SHA256,
    CHANGED_FILE,
    EXPECTED_HPC_OUTPUT_MAX_BYTES,
    EXPECTED_MODEL_CLOSURE_COMPLETE,
    EXPECTED_MODEL_FILE_DETAILS_INCLUDED,
    FIXED_ALLOCATION,
    FIXED_RUN_ID,
    MANIFEST_NAME,
    NORMALIZED_SOURCE_SHA256,
    RAW_SOURCE_SHA256,
    SCHEMA,
    compile_candidate,
    extract_candidate,
    inspect_baseline,
    run_smoke,
    runtime_tree_sha256,
    safe_member,
    sha256_bytes,
    sha256_file,
    verify_extraction,
)


RECEIPT_SCHEMA = "evomind.g21_hpc_asset_probe_runtime_test_receipt.r121.v1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def inspect_candidate(path: Path) -> tuple[dict[str, bytes], bytes]:
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = safe_member(info)
            if info.is_dir():
                continue
            if name in entries:
                raise ValueError(f"R121_CANDIDATE_DUPLICATE:{name}")
            entries[name] = archive.read(info)
    manifests = {name for name in entries if name.endswith("manifest.json")}
    if manifests != {MANIFEST_NAME}:
        raise ValueError("R121_CANDIDATE_MANIFEST_SET_REJECTED")
    runtime = {name: raw for name, raw in entries.items() if name.startswith("evomind_runtime/")}
    if len(runtime) != BASELINE_FILE_COUNT or len(entries) != BASELINE_FILE_COUNT + 1:
        raise ValueError("R121_CANDIDATE_FILESET_REJECTED")
    return runtime, entries[MANIFEST_NAME]


def verify_junit(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("R121_JUNIT_REGULAR_FILE_REQUIRED")
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    if "failures=\"0\"" not in text or "errors=\"0\"" not in text:
        raise ValueError(f"R121_JUNIT_NOT_GREEN:{path.name}")
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(args: argparse.Namespace) -> dict[str, Any]:
    baseline_zip = Path(args.baseline_zip).resolve()
    candidate_zip = Path(args.candidate_zip).resolve()
    manifest_path = Path(args.source_manifest).resolve()
    build_result_path = Path(args.build_result).resolve()
    baseline, _ = inspect_baseline(baseline_zip)
    runtime, embedded_raw = inspect_candidate(candidate_zip)
    manifest = load_object(manifest_path)
    build_result = load_object(build_result_path)
    if sha256_bytes(embedded_raw) != sha256_file(manifest_path):
        raise ValueError("R121_EMBEDDED_MANIFEST_BINDING_REJECTED")
    if json.loads(embedded_raw.decode("utf-8-sig")) != manifest:
        raise ValueError("R121_EMBEDDED_MANIFEST_CONTENT_REJECTED")
    changed = sorted(name for name in runtime if runtime[name] != baseline[name])
    unchanged = sorted(name for name in runtime if runtime[name] == baseline[name])
    if changed != [CHANGED_FILE] or len(unchanged) != 31:
        raise ValueError("R121_PATCH_SCOPE_REJECTED")
    candidate_tools_sha = sha256_bytes(runtime[CHANGED_FILE])
    if candidate_tools_sha != manifest.get("packaged_tools_sha256"):
        raise ValueError("R121_PACKAGED_TOOLS_SHA_REJECTED")
    tree = runtime_tree_sha256(runtime)
    declared = {str(item.get("path")): item for item in manifest.get("files", []) if isinstance(item, Mapping)}
    if len(declared) != 32 or set(declared) != set(runtime):
        raise ValueError("R121_MANIFEST_FILE_CLOSURE_REJECTED")
    for name, raw in runtime.items():
        item = declared[name]
        if int(item.get("bytes", -1)) != len(raw) or str(item.get("sha256")) != sha256_bytes(raw):
            raise ValueError(f"R121_MANIFEST_FILE_HASH_REJECTED:{name}")
    required = {
        "schema": SCHEMA,
        "run_id": FIXED_RUN_ID,
        "allocation": FIXED_ALLOCATION,
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "candidate_runtime_tree_sha256": tree,
        "source_tools_raw_sha256": RAW_SOURCE_SHA256,
        "source_tools_normalized_sha256": NORMALIZED_SOURCE_SHA256,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"R121_MANIFEST_BINDING_REJECTED:{key}")
    output_bound = manifest.get("output_bound")
    if not isinstance(output_bound, dict) or (
        output_bound.get("max_probe_stdout_bytes") != EXPECTED_HPC_OUTPUT_MAX_BYTES
        or output_bound.get("model_file_details_included") is not EXPECTED_MODEL_FILE_DETAILS_INCLUDED
        or output_bound.get("model_metadata_closure_complete") is not EXPECTED_MODEL_CLOSURE_COMPLETE
    ):
        raise ValueError("R121_OUTPUT_BOUND_CONTRACT_REJECTED")
    if (
        int(manifest.get("candidate_runtime_file_count", -1)) != 32
        or int(manifest.get("changed_file_count", -1)) != 1
        or int(manifest.get("unchanged_file_count", -1)) != 31
        or bool(manifest.get("production_deployable"))
        or bool(manifest.get("production_deployed"))
    ):
        raise ValueError("R121_MANIFEST_SCOPE_REJECTED")
    candidate_sha = sha256_file(candidate_zip)
    manifest_sha = sha256_file(manifest_path)
    if (
        build_result.get("schema") != SCHEMA
        or build_result.get("status") != "built_and_verified"
        or build_result.get("zip_sha256") != candidate_sha
        or build_result.get("source_manifest_sha256") != manifest_sha
        or build_result.get("candidate_runtime_tree_sha256") != tree
        or build_result.get("packaged_tools_sha256") != candidate_tools_sha
        or bool(build_result.get("production_deployed"))
        or bool(build_result.get("hpc_accessed"))
        or bool(build_result.get("gpu_touched"))
        or int(build_result.get("remote_writes", -1)) != 0
    ):
        raise ValueError("R121_BUILD_RESULT_REJECTED")
    rounds = []
    for number in (1, 2):
        with tempfile.TemporaryDirectory(prefix=f"r121-independent-{number}-") as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            extract_candidate(candidate_zip, root)
            checked = verify_extraction(root, runtime, manifest_sha, number)
            checked["py_compile"] = compile_candidate(root)
            checked["policy_smoke"] = run_smoke(root, number)
            rounds.append(checked)
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
        "candidate_runtime_tree_sha256": tree,
        "baseline_zip_sha256": BASELINE_ZIP_SHA256,
        "baseline_runtime_tree_sha256": BASELINE_TREE_SHA256,
        "candidate_runtime_file_count": 32,
        "changed_file_count": 1,
        "unchanged_file_count": 31,
        "changed_files": [CHANGED_FILE],
        "source_tools_raw_sha256": RAW_SOURCE_SHA256,
        "source_tools_normalized_sha256": NORMALIZED_SOURCE_SHA256,
        "packaged_tools_sha256": candidate_tools_sha,
        "output_bound": output_bound,
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
    value = argparse.ArgumentParser(description="Independently verify the R121 output-bound asset-probe runtime")
    value.add_argument("--baseline-zip", required=True)
    value.add_argument("--candidate-zip", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--build-result", required=True)
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
