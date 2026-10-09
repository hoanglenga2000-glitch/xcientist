#!/usr/bin/env python3
"""Build and verify the pinned MLE-Bench unified remote bundle.

The upstream MLE-Bench grader tree is reused from a previously verified stage;
all first-party runners and plans are refreshed from the live worktree.  The
result is independently hashed, compiled, imported and archive-verified before
it can be pinned by ``mlebench_remote_ops.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    PROJECT_ROOT
    / "workspace"
    / "deploy"
    / "mlebench_unified_20260725_055130"
    / "stage"
)
SOURCE_MAP = {
    "scripts/build_mlebench_candidate_gates.py": PROJECT_ROOT / "scripts" / "build_mlebench_candidate_gates.py",
    "scripts/generate_mlebench_recovery_deep_review.py": PROJECT_ROOT / "scripts" / "generate_mlebench_recovery_deep_review.py",
    "scripts/generate_mlebench_unscored_deep_review.py": PROJECT_ROOT / "scripts" / "generate_mlebench_unscored_deep_review.py",
    "scripts/mlebench_medal_recovery_adapters.py": PROJECT_ROOT / "scripts" / "mlebench_medal_recovery_adapters.py",
    "scripts/mlebench_wave2_adapters.py": PROJECT_ROOT / "scripts" / "mlebench_wave2_adapters.py",
    "scripts/russian_transliteration.py": PROJECT_ROOT / "scripts" / "russian_transliteration.py",
    "scripts/regrade_mlebench_lite_run.py": PROJECT_ROOT / "scripts" / "regrade_mlebench_lite_run.py",
    "scripts/run_mlebench_lite_full.py": PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py",
    "scripts/run_mlebench_lite_wave0.py": PROJECT_ROOT / "scripts" / "run_mlebench_lite_wave0.py",
    "src/research_os/mlebench_phase_a.py": PROJECT_ROOT / "src" / "research_os" / "mlebench_phase_a.py",
    "plans/may2022_gpt56_review_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "may2022_gpt56_review_current.json",
    "plans/may2022_nested_selection_repair_s42_frozen_plan_20260727.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "may2022_nested_selection_repair_s42_frozen_plan_20260727.json",
    "plans/may2022_nested_selection_execution_s42_hpc88240_v2_20260728.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "may2022_nested_selection_execution_s42_hpc88240_v2_20260728.json",
    "plans/may2022_nested_selection_execution_s42_hpc88240_v3_cache_20260728.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "may2022_nested_selection_execution_s42_hpc88240_v3_cache_20260728.json",
    "plans/may2022_nested_selection_execution_s42_hpc88240_v4_cache_20260728.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "may2022_nested_selection_execution_s42_hpc88240_v4_cache_20260728.json",
    "plans/may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json",
    "plans/medal_recovery_gpt56_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_gpt56_current.json",
    "plans/medal_recovery_deep_gpt56_review_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_recovery_deep_gpt56_review_current.json",
    "plans/medal_candidate_gates_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "medal_candidate_gates_current.json",
    "plans/unscored8_deep_gpt56_review_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "unscored8_deep_gpt56_review_current.json",
    "plans/wave1_gpt56_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "wave1_gpt56_current.json",
    "plans/wave2_gpt56_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "wave2_gpt56_current.json",
    "plans/wave2_deep_gpt56_review_current.json": PROJECT_ROOT / "workspace" / "mlebench_plans" / "wave2_deep_gpt56_review_current.json",
}
COMPILE_PATHS = (
    "scripts/build_mlebench_candidate_gates.py",
    "scripts/generate_mlebench_recovery_deep_review.py",
    "scripts/generate_mlebench_unscored_deep_review.py",
    "scripts/mlebench_medal_recovery_adapters.py",
    "scripts/mlebench_wave2_adapters.py",
    "scripts/russian_transliteration.py",
    "scripts/regrade_mlebench_lite_run.py",
    "scripts/run_mlebench_lite_full.py",
    "scripts/run_mlebench_lite_wave0.py",
    "src/research_os/__init__.py",
    "src/research_os/mlebench_phase_a.py",
)
SECRET_PATTERNS = (
    re.compile(rb"agt_codex_[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
MINIMAL_RESEARCH_OS_INIT = '"""Minimal MLE-Bench deployment namespace."""\n'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Bundle stage contains a symbolic link: {path}")
        if path.is_file():
            yield path


def scan_secrets(stage: Path) -> int:
    scanned = 0
    for path in iter_regular_files(stage):
        if path.stat().st_size > 8 * 1024 * 1024:
            continue
        data = path.read_bytes()
        scanned += 1
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"Secret-like material found in bundle member: {path.relative_to(stage)}")
    return scanned


def build(args: argparse.Namespace) -> dict[str, Any]:
    template = args.template_stage.resolve()
    if not (template / "upstream_mlebench" / "mlebench").is_dir():
        raise FileNotFoundError("Verified upstream MLE-Bench template stage is missing")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / f"mlebench_unified_{timestamp}").resolve()
    stage = output_dir / "stage"
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(template, stage)
    (stage / "bundle_manifest.json").unlink(missing_ok=True)
    for relative, source in SOURCE_MAP.items():
        if not source.is_file():
            raise FileNotFoundError(f"Bundle source is missing: {source}")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    (stage / "src" / "research_os" / "__init__.py").write_text(
        MINIMAL_RESEARCH_OS_INIT,
        encoding="utf-8",
    )

    for relative in COMPILE_PATHS:
        py_compile.compile(str(stage / relative), doraise=True)
    for cache in stage.rglob("__pycache__"):
        shutil.rmtree(cache)
    scanned_files = scan_secrets(stage)

    manifest_files = {
        path.relative_to(stage).as_posix(): sha256_file(path)
        for path in iter_regular_files(stage)
    }
    deep_review = json.loads((stage / "plans" / "wave2_deep_gpt56_review_current.json").read_text(encoding="utf-8"))
    recovery_review = json.loads(
        (stage / "plans" / "medal_recovery_deep_gpt56_review_current.json").read_text(encoding="utf-8")
    )
    unscored_review = json.loads(
        (stage / "plans" / "unscored8_deep_gpt56_review_current.json").read_text(encoding="utf-8")
    )
    wave1_plan = json.loads(
        (stage / "plans" / "wave1_gpt56_current.json").read_text(encoding="utf-8")
    )
    recovery_plan = json.loads(
        (stage / "plans" / "medal_recovery_gpt56_current.json").read_text(encoding="utf-8")
    )
    may_nested_plan = json.loads(
        (
            stage
            / "plans"
            / "may2022_nested_selection_repair_s42_frozen_plan_20260727.json"
        ).read_text(encoding="utf-8")
    )
    may_execution_plan = json.loads(
        (
            stage
            / "plans"
            / "may2022_nested_selection_execution_s42_hpc88240_v2_20260728.json"
        ).read_text(encoding="utf-8")
    )
    may_cache_execution_plan = json.loads(
        (
            stage
            / "plans"
            / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
        ).read_text(encoding="utf-8")
    )
    if recovery_plan.get("schema") != "evomind.mlebench_lite.medal_recovery_plan.v1":
        raise ValueError("Medal recovery plan schema mismatch")
    if (recovery_plan.get("planner") or {}).get("model") != "gpt-5.6-sol":
        raise ValueError("Medal recovery plan must carry gpt-5.6-sol planner evidence")
    if may_nested_plan.get("schema") != "evomind.mlebench.may2022_nested_selection_repair_plan.v1":
        raise ValueError("May-2022 nested-selection repair plan schema mismatch")
    if (
        may_execution_plan.get("schema")
        != "evomind.mlebench.may2022_nested_selection_execution_plan.v1"
        or may_execution_plan.get("parent_plan_sha256")
        != manifest_files[
            "plans/may2022_nested_selection_repair_s42_frozen_plan_20260727.json"
        ]
        or (may_execution_plan.get("authorization") or {}).get("full_training_approved")
        is not True
    ):
        raise ValueError("May-2022 execution plan is not bound and approved")
    may_cache = may_cache_execution_plan.get("public_precomputed_cache") or {}
    may_cache_contract = may_cache_execution_plan.get("execution_contract") or {}
    may_cache_argv = list(may_cache_execution_plan.get("launch_argv_template") or [])
    if (
        may_cache_execution_plan.get("schema")
        != "evomind.mlebench.may2022_nested_selection_execution_plan.v1"
        or may_cache_execution_plan.get("parent_plan_sha256")
        != manifest_files[
            "plans/may2022_nested_selection_repair_s42_frozen_plan_20260727.json"
        ]
        or may_cache.get("required") is not True
        or may_cache.get("status") != "completed"
        or may_cache.get("visibility_mode") != "PUBLIC_ONLY"
        or may_cache.get("feature_builder_sha256")
        != manifest_files["scripts/mlebench_medal_recovery_adapters.py"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(may_cache.get("manifest_sha256") or ""))
        or may_cache_contract.get("verified_precomputed_cache_required") is not True
        or may_cache_contract.get("gpu_feature_rebuild_forbidden") is not True
        or may_cache_argv.count("--may-precomputed-cache-dir") != 1
        or may_cache_argv.count("--may-require-precomputed-cache") != 1
    ):
        raise ValueError("May-2022 cache execution plan is not fully bound")
    manifest = {
        "schema": "evomind.mlebench_lite.unified_bundle.v1",
        "created_at": utc_now(),
        "waves": ["Wave0", "Wave1", "Wave2"],
        "wave1_competition_count": len(wave1_plan.get("competition_order") or []),
        "wave1_planner_model": (wave1_plan.get("planner") or {}).get("model"),
        "wave1_plan_sha256": manifest_files["plans/wave1_gpt56_current.json"],
        "wave2_competition_count": 11,
        "medal_recovery_targets": 9,
        "medal_recovery_plan_schema": recovery_plan.get("schema"),
        "medal_recovery_plan_sha256": manifest_files[
            "plans/medal_recovery_gpt56_current.json"
        ],
        "planner_model": deep_review.get("planner", {}).get("model"),
        "deep_review_ok": bool(deep_review.get("ok")),
        "medal_recovery_planner_model": recovery_review.get("planner", {}).get("model"),
        "medal_recovery_deep_review_parse_ok": bool(recovery_review.get("parse_ok")),
        "medal_recovery_deep_review_sha256": manifest_files[
            "plans/medal_recovery_deep_gpt56_review_current.json"
        ],
        "unscored8_planner_model": unscored_review.get("planner", {}).get("model"),
        "unscored8_deep_review_parse_ok": bool(unscored_review.get("parse_ok")),
        "unscored8_deep_review_sha256": manifest_files[
            "plans/unscored8_deep_gpt56_review_current.json"
        ],
        "medal_candidate_gates_sha256": manifest_files[
            "plans/medal_candidate_gates_current.json"
        ],
        "medal_recovery_adapter_sha256": manifest_files[
            "scripts/mlebench_medal_recovery_adapters.py"
        ],
        "may2022_nested_selection_plan_sha256": manifest_files[
            "plans/may2022_nested_selection_repair_s42_frozen_plan_20260727.json"
        ],
        "may2022_nested_selection_execution_plan_sha256": manifest_files[
            "plans/may2022_nested_selection_execution_s42_hpc88240_v2_20260728.json"
        ],
        "may2022_cached_execution_plan_sha256": manifest_files[
            "plans/may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
        ],
        "may2022_public_cache_manifest_sha256": may_cache["manifest_sha256"],
        "kaggle_submission_enabled": False,
        "human_gate_preserved": True,
        "minimal_research_os_namespace": True,
        "files": manifest_files,
    }
    write_json(stage / "bundle_manifest.json", manifest)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        str(stage / "src"), str(stage / "scripts"), str(stage / "upstream_mlebench")
    ])
    import_probe = subprocess.run(
        [sys.executable, "-c", "import mlebench_wave2_adapters, run_mlebench_lite_full; print('IMPORT_OK')"],
        cwd=stage,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if import_probe.returncode != 0 or "IMPORT_OK" not in import_probe.stdout:
        raise RuntimeError(
            "Standalone bundle import failed\n"
            f"stdout:\n{import_probe.stdout}\n"
            f"stderr:\n{import_probe.stderr}"
        )
    for cache in stage.rglob("__pycache__"):
        shutil.rmtree(cache)

    bundle = output_dir / "mlebench_unified_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=9) as archive:
        for path in iter_regular_files(stage):
            archive.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)

    temporary_root = PROJECT_ROOT / "workspace" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlebench_bundle_verify_", dir=temporary_root) as raw:
        verify_root = Path(raw)
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                    raise RuntimeError(f"Unsafe bundle member: {member.name}")
            archive_names = {member.name for member in members}
            expected_archive_names = set(manifest_files) | {"bundle_manifest.json"}
            if archive_names != expected_archive_names:
                missing = sorted(expected_archive_names - archive_names)
                unexpected = sorted(archive_names - expected_archive_names)
                raise RuntimeError(
                    "Archive/manifest member mismatch: "
                    f"missing={missing[:10]} unexpected={unexpected[:10]}"
                )
            archive.extractall(verify_root, filter="data")
        extracted_manifest = json.loads((verify_root / "bundle_manifest.json").read_text(encoding="utf-8"))
        verified = 0
        for relative, expected in extracted_manifest["files"].items():
            if sha256_file(verify_root / relative) != expected:
                raise RuntimeError(f"Extracted manifest hash mismatch: {relative}")
            verified += 1

    report = {
        "schema": "evomind.mlebench_lite.unified_bundle_build.v1",
        "created_at": utc_now(),
        "stage": str(stage),
        "bundle": str(bundle),
        "sha256": sha256_file(bundle),
        "file_count": len(manifest_files) + 1,
        "manifest_hash_count": len(manifest_files),
        "size_bytes": bundle.stat().st_size,
        "deep_review_sha256": manifest_files["plans/wave2_deep_gpt56_review_current.json"],
        "wave1_plan_sha256": manifest_files["plans/wave1_gpt56_current.json"],
        "medal_recovery_deep_review_sha256": manifest_files[
            "plans/medal_recovery_deep_gpt56_review_current.json"
        ],
        "unscored8_deep_review_sha256": manifest_files[
            "plans/unscored8_deep_gpt56_review_current.json"
        ],
        "medal_candidate_gates_sha256": manifest_files[
            "plans/medal_candidate_gates_current.json"
        ],
        "medal_recovery_adapter_sha256": manifest_files[
            "scripts/mlebench_medal_recovery_adapters.py"
        ],
        "wave2_adapter_sha256": manifest_files["scripts/mlebench_wave2_adapters.py"],
        "full_runner_sha256": manifest_files["scripts/run_mlebench_lite_full.py"],
        "may2022_cached_execution_plan_sha256": manifest_files[
            "plans/may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
        ],
        "may2022_public_cache_manifest_sha256": may_cache["manifest_sha256"],
        "verification": {
            "archive_file_members": len(members),
            "expected_archive_file_members": len(manifest_files) + 1,
            "archive_manifest_member_set_match": True,
            "manifest_hashes_verified": verified,
            "compiled_entry_files": len(COMPILE_PATHS),
            "standalone_import": "IMPORT_OK",
            "secret_scan_files": scanned_files,
            "first_party_secret_scan": "passed",
        },
        "passed": True,
    }
    write_json(output_dir / "build_report.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-stage", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "workspace" / "deploy")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    report = build(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


