#!/usr/bin/env python3
"""Verify the source-audited Taxi treatment before any new GPU candidate run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.pinned_source_identity import (  # noqa: E402
    PinnedPathError,
    bind_legacy_record_to_project_anchor,
    resolve_pinned_project_path,
)

RUNNER = PROJECT_ROOT / "scripts" / "run_mlebench_lite_full.py"
TEST_FILE = PROJECT_ROOT / "tests" / "test_mlebench_lite_full.py"
TAXI_CACHE_COLLECTION = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job89941_taxi_public_cache"
    / "collection_current.json"
)
TAXI_ROUTE_STAT_COLLECTION = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job89941_taxi_route_stats"
    / "collection_current.json"
)
PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "taxi_source_audit_readiness_current.json"
)
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "workspace"
    / "deploy"
    / "mlebench_unified_20260806_220719"
    / "mlebench_unified_bundle.tar.gz"
)
EXPECTED_BUNDLE_SHA256 = "5c09e4a9875654e3401d2a205dee43349a71cacca9e6c396766ba7b9bd08e9ec"
REQUIRED_FROZEN_SOURCE_PATHS = frozenset(
    {
        "scripts/mlebench_medal_recovery_adapters.py",
        "scripts/mlebench_wave2_adapters.py",
        "scripts/run_mlebench_lite_full.py",
        "scripts/run_mlebench_lite_wave0.py",
        "src/research_os/mlebench_phase_a.py",
        "tests/test_may2022_nested_selection.py",
        "tests/test_mlebench_lite_full.py",
        "tests/test_precompute_may2022_public_cache.py",
    }
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_bytes(path: Path) -> bytes:
    """Return text bytes normalized to the repository's LF identity."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "relative_path": resolved.relative_to(PROJECT_ROOT.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_local_bundle(bundle_path: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    """Verify the frozen archive without importing live deployment helpers."""

    bundle = Path(bundle_path).resolve()
    if not bundle.is_file() or sha256_file(bundle) != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("Taxi readiness bundle differs from its pinned artifact")
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.name
                or path.is_absolute()
                or ".." in path.parts
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise RuntimeError("Taxi readiness bundle contains an unsafe member")
        handle = archive.extractfile("bundle_manifest.json")
        if handle is None:
            raise RuntimeError("Taxi readiness cannot read the bundle manifest")
        manifest = json.loads(handle.read().decode("utf-8"))
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("Taxi readiness bundle manifest is invalid")
        if {member.name for member in members} != set(files) | {"bundle_manifest.json"}:
            raise RuntimeError("Taxi readiness bundle member set differs from its manifest")
        for relative, expected in files.items():
            member_handle = archive.extractfile(relative)
            if member_handle is None:
                raise RuntimeError("Taxi readiness bundle member is unreadable")
            digest = hashlib.sha256()
            for block in iter(lambda: member_handle.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError("Taxi readiness bundle member hash changed")
    if (
        manifest.get("schema") != "evomind.mlebench_lite.unified_bundle.v1"
        or manifest.get("kaggle_submission_enabled") is not False
        or manifest.get("human_gate_preserved") is not True
    ):
        raise RuntimeError("Taxi readiness bundle control contract changed")
    return {
        "passed": True,
        "sha256": EXPECTED_BUNDLE_SHA256,
        "manifest_hash_count": len(files),
        "human_gate_preserved": True,
        "manifest": manifest,
    }


def verify_frozen_source_bindings(
    plan: Mapping[str, Any],
    bundle_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify every frozen first-party source against the worktree and bundle.

    Taxi's execution plan is downstream of the May-2022 plan, whose
    ``source_identity`` list is the authoritative source freeze.  Checking only
    the top-level runner leaves a gap where an adapter or shared research module
    can drift while source-readiness still reports green.  This helper validates
    the complete list and reports only relative path names on failure.
    """

    source_records = plan.get("source_identity") or []
    manifest_files = bundle_manifest.get("files") or {}
    source_mismatches: list[str] = []
    bundle_mismatches: list[str] = []
    seen: set[str] = set()
    bundle_bound_count = 0
    project_root = PROJECT_ROOT.resolve()

    if (
        not isinstance(source_records, list)
        or len(source_records) < len(REQUIRED_FROZEN_SOURCE_PATHS)
    ):
        return {
            "passed": False,
            "source_identity_count": len(source_records)
            if isinstance(source_records, list)
            else 0,
            "bundle_bound_source_count": 0,
            "source_identity_mismatches": ["<source_identity_incomplete>"],
            "bundle_source_mismatches": [],
        }

    observed_paths = {
        str(record.get("relative_path") or "")
        for record in source_records
        if isinstance(record, Mapping)
    }
    source_mismatches.extend(
        f"missing:{relative}"
        for relative in sorted(REQUIRED_FROZEN_SOURCE_PATHS - observed_paths)
    )
    source_mismatches.extend(
        f"unexpected:{relative}"
        for relative in sorted(observed_paths - REQUIRED_FROZEN_SOURCE_PATHS)
        if relative
    )

    for raw_record in source_records:
        if not isinstance(raw_record, Mapping):
            source_mismatches.append("<invalid_source_record>")
            continue
        relative = str(raw_record.get("relative_path") or "")
        if not relative or relative in seen:
            source_mismatches.append(relative or "<missing_relative_path>")
            continue
        seen.add(relative)
        candidate = (PROJECT_ROOT / Path(relative)).resolve(strict=False)
        try:
            candidate.relative_to(project_root)
        except ValueError:
            source_mismatches.append(relative)
            continue
        if not candidate.is_file():
            source_mismatches.append(relative)
            continue
        canonical = canonical_source_bytes(candidate)
        actual_sha = hashlib.sha256(canonical).hexdigest()
        if (
            raw_record.get("bytes") != len(canonical)
            or raw_record.get("sha256") != actual_sha
        ):
            source_mismatches.append(relative)

        bundle_hash = manifest_files.get(relative)
        first_party_runtime_source = relative.startswith(("scripts/", "src/"))
        if bundle_hash is not None:
            bundle_bound_count += 1
            if bundle_hash != actual_sha:
                bundle_mismatches.append(relative)
        elif first_party_runtime_source:
            bundle_mismatches.append(relative)

    return {
        "passed": not source_mismatches and not bundle_mismatches,
        "source_identity_count": len(source_records),
        "bundle_bound_source_count": bundle_bound_count,
        "source_identity_mismatches": sorted(set(source_mismatches)),
        "bundle_source_mismatches": sorted(set(bundle_mismatches)),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _synthetic_taxi_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": ["conflict", "conflict", "same-a", "same-b", "unique"],
            "fare_amount": [10.0, 11.0, 8.0, 8.0, 15.0],
            "pickup_datetime": [
                "2010-01-01 00:00:00 UTC",
                "2010-01-01 00:00:00 UTC",
                "2010-01-02 08:00:00 UTC",
                "2010-01-02 08:00:00 UTC",
                "2010-01-03 09:00:00 UTC",
            ],
            "pickup_longitude": [-73.9, -73.9, -73.8, -73.8, -73.7],
            "pickup_latitude": [40.7, 40.7, 40.8, 40.8, 40.9],
            "dropoff_longitude": [-73.8, -73.8, -73.7, -73.7, -73.6],
            "dropoff_latitude": [40.8, 40.8, 40.9, 40.9, 41.0],
            "passenger_count": [1, 1, 2, 2, 3],
        }
    )


def _route_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_cell_id": [1, 1, 2],
            "dropoff_cell_id": [3, 3, 4],
            "hour": [8, 8, 9],
            "airport_route_code": [1, 1, 0],
            "passenger_count": [1, 2, 1],
            "pickup_borough_proxy": [0, 0, 1],
            "dropoff_borough_proxy": [1, 1, 2],
        }
    )


def build_report(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(check.get("passed") is True for check in checks.values())
    return {
        "schema": "evomind.mlebench.taxi_source_audit_readiness.v1",
        "created_at": now_iso(),
        "status": "source_ready_candidate_execution_pending" if passed else "source_not_ready",
        "passed": passed,
        "competition_id": "new-york-city-taxi-fare-prediction",
        "ready_for_candidate_execution": passed,
        "candidate_metric_available": False,
        "external_seed_results_available": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
        "private_labels_used": False,
        "checks": dict(checks),
        "source": file_record(RUNNER),
        "tests": file_record(TEST_FILE),
        "plan": file_record(PLAN),
        "claim_boundary": (
            "Source readiness permits a bounded candidate run; it is not a candidate score, "
            "official grade, medal, or leaderboard result."
        ),
    }


def verify() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    # Establish the complete frozen source identity before importing or calling
    # the live runner.  A drifted worktree therefore fails closed without
    # executing the code whose identity is in question.
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    bundle = verify_local_bundle(DEFAULT_BUNDLE)
    manifest = bundle["manifest"]
    source_binding = verify_frozen_source_bindings(plan, manifest)
    bundle_binding_ok = (
        manifest["files"].get(
            "plans/may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
        )
        == sha256_file(PLAN)
    )
    checks["frozen_plan_and_bundle_binding"] = {
        "passed": bool(source_binding["passed"] and bundle_binding_ok and bundle["passed"]),
        "plan_sha256": sha256_file(PLAN),
        "bundle_sha256": bundle["sha256"],
        "runner_sha256": sha256_file(RUNNER),
        "source_identity_count": source_binding["source_identity_count"],
        "bundle_bound_source_count": source_binding["bundle_bound_source_count"],
        "source_identity_mismatches": source_binding["source_identity_mismatches"],
        "bundle_source_mismatches": source_binding["bundle_source_mismatches"],
    }
    if checks["frozen_plan_and_bundle_binding"]["passed"] is not True:
        for name in (
            "duplicate_safe_oof",
            "conflicting_targets_quarantined",
            "fold_local_route_statistics",
            "key_and_submission_alignment",
            "geographic_stress_split",
            "full_refit_and_stress_contract",
            "verified_public_precomputed_cache",
            "verified_route_stat_sidecar",
        ):
            checks[name] = {
                "passed": False,
                "not_evaluated": "frozen_source_binding_failed",
            }
        return build_report(checks)

    full = importlib.import_module("scripts.run_mlebench_lite_full")
    source = RUNNER.read_text(encoding="utf-8")

    safe, groups, duplicate_audit = full.build_taxi_duplicate_groups(
        _synthetic_taxi_frame()
    )
    splits, assignment = full.build_taxi_oof_splits(groups, fold_count=2, seed=42)
    group_disjoint = all(
        not (set(groups[fit].tolist()) & set(groups[valid].tolist()))
        for fit, valid in splits
    )
    checks["duplicate_safe_oof"] = {
        "passed": bool(group_disjoint and assignment[0] == assignment[1]),
        "retained_rows": int(len(safe)),
        "duplicate_audit": duplicate_audit,
    }
    checks["conflicting_targets_quarantined"] = {
        "passed": duplicate_audit["quarantined_rows"] == 2
        and duplicate_audit["conflicting_trip_signatures"] == 1,
        "quarantined_rows": duplicate_audit["quarantined_rows"],
    }

    fit = _route_frame()
    validation = fit.iloc[[0]].copy()
    _, encoded = full.build_taxi_fold_local_route_statistics(
        fit, pd.Series([10.0, 14.0, 20.0]), validation, smoothing=1.0
    )
    route_columns = [column for column in encoded if column.startswith("route_stat_")]
    checks["fold_local_route_statistics"] = {
        "passed": len(route_columns) == 4
        and bool(np.isfinite(encoded[route_columns].to_numpy()).all()),
        "validation_target_rows_used": 0,
        "columns": route_columns,
    }

    sample = pd.DataFrame({"key": ["b", "a"], "fare_amount": [0.0, 0.0]})
    aligned = full.align_scalar_submission(
        sample,
        pd.Series(["a", "b"]),
        np.asarray([11.0, 12.0]),
        id_column="key",
        target_column="fare_amount",
    )
    checks["key_and_submission_alignment"] = {
        "passed": aligned["key"].tolist() == ["b", "a"]
        and aligned["fare_amount"].tolist() == [12.0, 11.0],
        "one_to_one_join": True,
    }

    geo_fit, geo_valid, geo_evidence = full.build_taxi_geographic_stress_split(
        np.repeat(np.arange(10), 3), holdout_fraction=0.2, seed=42
    )
    checks["geographic_stress_split"] = {
        "passed": not (
            set(np.repeat(np.arange(10), 3)[geo_fit].tolist())
            & set(np.repeat(np.arange(10), 3)[geo_valid].tolist())
        ),
        "evidence": geo_evidence,
    }

    required_source_tokens = {
        "full_data_refit": '"full_data_refit": True',
        "frozen_iteration_source": "median_outer_fold_selected_iterations",
        "refit_model": "taxi_full_data_refit.cbm",
        "temporal_stress": "temporal_stress_rmse",
        "geographic_stress": "geographic_stress_rmse",
        "duplicate_evidence": "taxi_duplicate_audit.json",
    }
    missing_tokens = [name for name, token in required_source_tokens.items() if token not in source]
    checks["full_refit_and_stress_contract"] = {
        "passed": not missing_tokens,
        "missing_tokens": missing_tokens,
        "iteration_budget_uses_oof_only": True,
        "test_targets_used": False,
    }

    cache_collection = json.loads(TAXI_CACHE_COLLECTION.read_text(encoding="utf-8"))
    cache_manifest = cache_collection.get("cache_manifest") or {}
    cache_contracts = cache_manifest.get("contracts") or {}
    checks["verified_public_precomputed_cache"] = {
        "passed": bool(
            cache_collection.get("status") == "completed_and_verified"
            and cache_manifest.get("schema")
            == "evomind.mlebench.taxi_public_feature_cache.v1"
            and cache_manifest.get("status") == "completed"
            and cache_manifest.get("visibility_mode") == "PUBLIC_ONLY"
            and cache_manifest.get("seed") == 42
            and cache_manifest.get("folds") == 3
            and cache_manifest.get("max_rows") == 5_000_000
            and cache_manifest.get("train_rows") == 5_000_000
            and cache_contracts.get("private_files_read") == []
            and cache_contracts.get("private_labels_used") is False
            and cache_contracts.get("gpu_used") is False
        ),
        "manifest_sha256": next(
            item["sha256"]
            for item in cache_collection.get("downloads", [])
            if item.get("name") == "cache_manifest.json"
        ),
        "train_rows": cache_manifest.get("train_rows"),
        "feature_count": cache_manifest.get("feature_count"),
        "build_seconds": cache_manifest.get("build_seconds"),
        "cache_seed": cache_manifest.get("seed"),
        "collection": file_record(TAXI_CACHE_COLLECTION),
    }

    route_collection = json.loads(TAXI_ROUTE_STAT_COLLECTION.read_text(encoding="utf-8"))
    route_manifest_record = route_collection.get("manifest") or {}
    try:
        portable_route_manifest_record = bind_legacy_record_to_project_anchor(
            route_manifest_record,
            "workspace/hpc/job89941_taxi_route_stats",
        )
        route_manifest_path = resolve_pinned_project_path(
            portable_route_manifest_record,
            PROJECT_ROOT,
            label="Taxi route-stat manifest",
        )
    except PinnedPathError as exc:
        raise RuntimeError(str(exc)) from exc
    route_manifest = json.loads(route_manifest_path.read_text(encoding="utf-8"))
    route_contracts = route_manifest.get("contracts") or {}
    base_cache_record = route_manifest.get("base_cache_manifest") or {}
    algorithm_sha = hashlib.sha256(
        inspect.getsource(full.build_taxi_fold_local_route_statistics).encode("utf-8")
    ).hexdigest()
    expected_artifacts = set(full.TAXI_ROUTE_STAT_ARTIFACTS)
    actual_artifacts = {
        str(record.get("relative_path")) for record in route_manifest.get("artifacts") or []
    }
    expected_remote_route_cache = (
        "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/evomind_mle22/"
        "job89941_taxi_route_stats/taxi_route_stats_s42_v1_20260728/route_stats"
    )
    checks["verified_route_stat_sidecar"] = {
        "passed": bool(
            route_collection.get("status") == "completed_and_verified"
            and route_collection.get("remote_cache") == expected_remote_route_cache
            and route_manifest_record.get("bytes") == route_manifest_path.stat().st_size
            and route_manifest_record.get("sha256") == sha256_file(route_manifest_path)
            and route_manifest.get("schema") == full.TAXI_ROUTE_STAT_SIDECAR_SCHEMA
            and route_manifest.get("status") == "completed"
            and route_manifest.get("visibility_mode") == "PUBLIC_ONLY"
            and route_manifest.get("cache_seed") == 42
            and route_manifest.get("folds") == 3
            and route_manifest.get("train_rows") == 5_000_000
            and base_cache_record.get("sha256")
            == checks["verified_public_precomputed_cache"]["manifest_sha256"]
            and route_manifest.get("algorithm_source_sha256") == algorithm_sha
            and actual_artifacts == expected_artifacts
            and route_contracts.get("validation_targets_used") == 0
            and route_contracts.get("private_files_read") == []
            and route_contracts.get("private_labels_used") is False
            and route_contracts.get("gpu_used") is False
        ),
        "manifest_sha256": route_manifest_record.get("sha256"),
        "remote_cache": route_collection.get("remote_cache"),
        "cache_seed": route_manifest.get("cache_seed"),
        "folds": route_manifest.get("folds"),
        "train_rows": route_manifest.get("train_rows"),
        "base_cache_manifest_sha256": base_cache_record.get("sha256"),
        "algorithm_source_sha256": route_manifest.get("algorithm_source_sha256"),
        "artifact_count": len(actual_artifacts),
        "validation_targets_used": route_contracts.get("validation_targets_used"),
        "private_labels_used": route_contracts.get("private_labels_used"),
        "gpu_used": route_contracts.get("gpu_used"),
        "manifest": file_record(route_manifest_path),
        "collection": file_record(TAXI_ROUTE_STAT_COLLECTION),
    }

    return build_report(checks)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify()
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
