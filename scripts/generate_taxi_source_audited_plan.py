#!/usr/bin/env python3
"""Freeze the source-audited Taxi three-seed candidate execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import verify_taxi_source_audit_readiness as readiness_verifier  # noqa: E402
from scripts.pinned_source_identity import (  # noqa: E402
    PinnedPathError,
    bind_legacy_record_to_project_anchor,
    resolve_pinned_project_path,
)

READINESS = (
    PROJECT_ROOT / "workspace" / "mlebench_plans" / "taxi_source_audit_readiness_current.json"
)
MAY_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
)
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
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "taxi_source_audited_multiseed_plan_current.json"
)
SEEDS = (43, 44, 45)
ALLOWED_GPU_REMOTE_ROOT = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "relative_path": resolved.relative_to(PROJECT_ROOT.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def portable_freshness_value(value: Any) -> Any:
    """Remove machine-local path projections from portable evidence records."""

    if isinstance(value, Mapping):
        has_relative_identity = isinstance(value.get("relative_path"), str)
        return {
            key: portable_freshness_value(item)
            for key, item in value.items()
            if not (has_relative_identity and key == "path")
        }
    if isinstance(value, list):
        return [portable_freshness_value(item) for item in value]
    return value


def build_plan() -> dict[str, Any]:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    live_readiness = readiness_verifier.verify()
    if (
        readiness.get("passed") is not True
        or readiness.get("ready_for_candidate_execution") is not True
        or readiness.get("candidate_metric_available") is not False
        or live_readiness.get("passed") is not True
        or live_readiness.get("ready_for_candidate_execution") is not True
        or live_readiness.get("candidate_metric_available") is not False
        or ((readiness.get("checks") or {}).get("verified_route_stat_sidecar") or {}).get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError("Taxi source readiness is not in the executable pre-candidate state")
    freshness_fields = ("competition_id", "source", "tests", "plan", "checks")
    stale_fields = [
        field
        for field in freshness_fields
        if portable_freshness_value(readiness.get(field))
        != portable_freshness_value(live_readiness.get(field))
    ]
    if stale_fields:
        raise RuntimeError(
            "Taxi persisted source readiness is stale: " + ", ".join(stale_fields)
        )
    bundle = readiness_verifier.verify_local_bundle(readiness_verifier.DEFAULT_BUNDLE)
    may_plan = json.loads(MAY_PLAN.read_text(encoding="utf-8"))
    if may_plan.get("status") != "approved_waiting_for_hpc_gpu_and_serial_queue":
        raise RuntimeError("May dependency plan is not in its approved waiting state")
    cache_collection = json.loads(TAXI_CACHE_COLLECTION.read_text(encoding="utf-8"))
    cache_manifest = cache_collection.get("cache_manifest") or {}
    if not (
        cache_collection.get("status") == "completed_and_verified"
        and cache_manifest.get("schema")
        == "evomind.mlebench.taxi_public_feature_cache.v1"
        and cache_manifest.get("status") == "completed"
        and cache_manifest.get("visibility_mode") == "PUBLIC_ONLY"
        and cache_manifest.get("seed") == 42
        and cache_manifest.get("folds") == 3
        and cache_manifest.get("max_rows") == 5_000_000
        and (cache_manifest.get("contracts") or {}).get("private_files_read") == []
        and (cache_manifest.get("contracts") or {}).get("private_labels_used") is False
    ):
        raise RuntimeError("Taxi verified PUBLIC_ONLY cache evidence is incomplete")
    remote_cache = str(Path(str(cache_manifest["artifacts"][0]["path"])).parent).replace("\\", "/")
    manifest_download = next(
        item for item in cache_collection["downloads"] if item["name"] == "cache_manifest.json"
    )
    sample_input = next(
        item
        for item in cache_manifest.get("inputs") or []
        if str(item.get("relative_path", "")).endswith("sample_submission.csv")
    )
    route_collection = json.loads(TAXI_ROUTE_STAT_COLLECTION.read_text(encoding="utf-8"))
    route_manifest_record = route_collection.get("manifest") or {}
    try:
        route_manifest_path = resolve_pinned_project_path(
            bind_legacy_record_to_project_anchor(
                route_manifest_record,
                "workspace/hpc/job89941_taxi_route_stats",
            ),
            PROJECT_ROOT,
            label="Taxi route-stat manifest",
        )
    except PinnedPathError as exc:
        raise RuntimeError(str(exc)) from exc
    route_manifest = json.loads(route_manifest_path.read_text(encoding="utf-8"))
    route_contracts = route_manifest.get("contracts") or {}
    if not (
        route_collection.get("status") == "completed_and_verified"
        and route_manifest_path.is_file()
        and route_manifest_record.get("bytes") == route_manifest_path.stat().st_size
        and route_manifest_record.get("sha256") == sha256_file(route_manifest_path)
        and route_manifest.get("schema") == "evomind.mlebench.taxi_route_stat_sidecar.v1"
        and route_manifest.get("status") == "completed"
        and route_manifest.get("cache_seed") == 42
        and route_manifest.get("folds") == 3
        and route_manifest.get("train_rows") == 5_000_000
        and (route_manifest.get("base_cache_manifest") or {}).get("sha256")
        == manifest_download["sha256"]
        and route_contracts.get("validation_targets_used") == 0
        and route_contracts.get("private_files_read") == []
        and route_contracts.get("private_labels_used") is False
        and route_contracts.get("gpu_used") is False
    ):
        raise RuntimeError("Taxi verified route-stat sidecar evidence is incomplete")
    remote_route_cache = str(route_collection.get("remote_cache") or "")
    if not remote_route_cache.startswith(ALLOWED_GPU_REMOTE_ROOT + "/"):
        raise RuntimeError("Taxi route-stat sidecar escaped the dedicated remote root")
    run_ids = [f"hpc88240_taxi_source_audited_route_s{seed}_20260728" for seed in SEEDS]
    return {
        "schema": "evomind.mlebench.taxi_source_audited_multiseed_plan.v1",
        "created_at": now_iso(),
        "status": "approved_waiting_for_may_terminal_and_gpu_idle",
        "competition_id": "new-york-city-taxi-fare-prediction",
        "objective": {
            "metric": "rmse",
            "direction": "minimize",
            "candidate_only": True,
            "expected_medal_gain": 1,
        },
        "seeds": list(SEEDS),
        "run_ids": run_ids,
        "source_readiness": record(READINESS),
        "runner_source": readiness["source"],
        "tests_source": readiness["tests"],
        "bundle": {
            "path": str(readiness_verifier.DEFAULT_BUNDLE.resolve()),
            "relative_path": readiness_verifier.DEFAULT_BUNDLE.resolve()
            .relative_to(PROJECT_ROOT.resolve())
            .as_posix(),
            "bytes": readiness_verifier.DEFAULT_BUNDLE.stat().st_size,
            "sha256": bundle["sha256"],
            "manifest_hash_count": bundle["manifest_hash_count"],
            "human_gate_preserved": bundle["human_gate_preserved"],
        },
        "public_precomputed_cache": {
            "required": True,
            "path": remote_cache,
            "manifest_path": str(cache_collection["remote_status"]["manifest"]["path"]),
            "manifest_bytes": int(manifest_download["bytes"]),
            "manifest_sha256": str(manifest_download["sha256"]),
            "schema": cache_manifest["schema"],
            "status": cache_manifest["status"],
            "visibility_mode": cache_manifest["visibility_mode"],
            "cache_seed": int(cache_manifest["seed"]),
            "train_rows": int(cache_manifest["train_rows"]),
            "feature_count": int(cache_manifest["feature_count"]),
            "build_seconds": float(cache_manifest["build_seconds"]),
            "collection_evidence": record(TAXI_CACHE_COLLECTION),
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        "public_sample_submission": {
            "remote_path": str(sample_input["path"]),
            "relative_path": str(sample_input["relative_path"]),
            "bytes": int(sample_input["bytes"]),
            "sha256": str(sample_input["sha256"]),
        },
        "verified_route_stat_sidecar": {
            "required": True,
            "path": remote_route_cache,
            "schema": route_manifest["schema"],
            "status": route_manifest["status"],
            "visibility_mode": route_manifest["visibility_mode"],
            "cache_seed": int(route_manifest["cache_seed"]),
            "folds": int(route_manifest["folds"]),
            "train_rows": int(route_manifest["train_rows"]),
            "test_rows": int(route_manifest["test_rows"]),
            "base_cache_manifest_sha256": str(
                route_manifest["base_cache_manifest"]["sha256"]
            ),
            "algorithm_source_sha256": str(route_manifest["algorithm_source_sha256"]),
            "manifest": record(route_manifest_path),
            "collection_evidence": record(TAXI_ROUTE_STAT_COLLECTION),
            "validation_targets_used": int(route_contracts["validation_targets_used"]),
            "private_labels_used": False,
            "gpu_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
        "serial_dependency": {
            "competition_id": "tabular-playground-series-may-2022",
            "plan": record(MAY_PLAN),
            "run_id": may_plan["run_id"],
            "run_ids": may_plan["run_ids"],
            "seeds": may_plan["seeds"],
            "all_seeds_terminal_before_taxi": True,
            "required_terminal_before_taxi": True,
            "natural_completion_only": True,
        },
        "compute_gate": {
            "target": "job88240",
            "gpu": "NVIDIA A40",
            "samples_required": 3,
            "max_memory_used_mib": 1024,
            "max_utilization_percent": 5,
            "compute_apps_required_empty": True,
            "launch_when_gate_false": False,
        },
        "training_contract": {
            "largest_predeclared_sample_rows": 5_000_000,
            "chunk_rows": 250_000,
            "folds": 3,
            "iterations": 1_400,
            "holdout_fraction": 0.05,
            "duplicate_safe_oof": True,
            "conflicting_targets_quarantined": True,
            "fold_local_route_statistics": True,
            "route_statistics_precomputed_on_cpu": True,
            "temporal_stress": True,
            "geographic_stress": True,
            "full_data_refit": True,
            "refit_iteration_source": "median_outer_fold_selected_iterations",
        },
        "runner_contract_args": [
            "--candidate-only",
            "--wave2-fast-kernels",
            "--taxi-max-train-rows",
            "5000000",
            "--taxi-chunk-rows",
            "250000",
            "--taxi-folds",
            "3",
            "--taxi-iterations",
            "1400",
            "--taxi-holdout-fraction",
            "0.05",
            "--taxi-cache-seed",
            "42",
            "--taxi-precomputed-cache-dir",
            remote_cache,
            "--taxi-require-precomputed-cache",
            "--taxi-route-stat-cache-dir",
            remote_route_cache,
            "--taxi-require-route-stat-cache",
        ],
        "promotion_contract": {
            "every_seed_random_oof_rmse_maximum": 2.85,
            "every_seed_temporal_stress_rmse_maximum": 3.10,
            "every_seed_geographic_stress_rmse_maximum": 3.35,
            "mean_random_oof_rmse_maximum": 2.82,
            "worst_seed_required": True,
            "all_three_external_seeds_required": True,
            "duplicate_isolation_required": True,
            "full_refit_required": True,
            "official_private_grader_approved": False,
            "kaggle_submission_approved": False,
        },
        "execution_contract": {
            "candidate_only": True,
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_allowed": False,
            "other_processes_may_be_modified": False,
            "local_gpu_allowed": False,
            "dedicated_remote_root_only": True,
            "verified_precomputed_cache_required": True,
            "gpu_feature_rebuild_forbidden": True,
            "verified_route_stat_cache_required": True,
            "gpu_route_stat_rebuild_forbidden": True,
        },
        "boundaries": {
            "private_labels_used": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
            "process_signals_sent": 0,
            "human_gate_preserved": True,
        },
        "claim_boundary": (
            "This plan authorizes bounded candidate computation only. Source readiness and future "
            "OOF metrics are not official scores, medals, or leaderboard proof."
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan()
    write_json_atomic(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
