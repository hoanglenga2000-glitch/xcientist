#!/usr/bin/env python3
"""Queue source-audited Taxi seeds after May and a fresh idle GPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import mlebench_remote_ops as ops  # noqa: E402

DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "taxi_source_audited_multiseed_plan_current.json"
)
DEFAULT_MAY_STATUS = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_may2022_nested_queue_v8_multiseed_cache_taxi_route"
    / "status_current.json"
)
DEFAULT_EVIDENCE_DIR = (
    PROJECT_ROOT
    / "workspace"
    / "hpc"
    / "job88240_taxi_source_audited_queue_v5_may_multiseed_route_cache"
)


class TaxiQueueError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxiQueueError(message)


def validate_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    resolved = Path(path).resolve()
    plan = json.loads(resolved.read_text(encoding="utf-8"))
    _require(
        plan.get("schema") == "evomind.mlebench.taxi_source_audited_multiseed_plan.v1",
        "Taxi queue plan schema changed",
    )
    _require(
        plan.get("status") == "approved_waiting_for_may_terminal_and_gpu_idle",
        "Taxi queue plan is not approved for candidate execution",
    )
    _require(plan.get("seeds") == [43, 44, 45], "Taxi queue seeds changed")
    _require(len(set(plan.get("run_ids") or [])) == 3, "Taxi queue run IDs are invalid")
    execution = plan.get("execution_contract") or {}
    _require(
        execution.get("candidate_only") is True
        and execution.get("official_grader_executed") is False
        and execution.get("kaggle_submission_executed") is False
        and execution.get("process_signals_allowed") is False
        and execution.get("other_processes_may_be_modified") is False
        and execution.get("local_gpu_allowed") is False,
        "Taxi queue execution boundary changed",
    )
    for name in ("source_readiness", "runner_source", "tests_source"):
        item = plan.get(name) or {}
        source = Path(str(item.get("path") or "")).resolve()
        _require(source.is_file(), f"Taxi queue {name} is missing")
        _require(
            source.stat().st_size == int(item.get("bytes", -1))
            and sha256_file(source) == item.get("sha256"),
            f"Taxi queue {name} hash drifted",
        )
    dependency = plan.get("serial_dependency") or {}
    dependency_plan = dependency.get("plan") or {}
    dependency_path = Path(str(dependency_plan.get("path") or "")).resolve()
    _require(
        dependency_path.is_file()
        and dependency_path.stat().st_size == int(dependency_plan.get("bytes", -1))
        and sha256_file(dependency_path) == dependency_plan.get("sha256"),
        "Taxi queue May dependency plan drifted",
    )
    bundle = plan.get("bundle") or {}
    bundle_path = Path(str(bundle.get("path") or "")).resolve()
    _require(bundle_path == ops.DEFAULT_BUNDLE.resolve(), "Taxi queue bundle path changed")
    verified = ops.verify_local_bundle(bundle_path)
    _require(verified["sha256"] == bundle.get("sha256"), "Taxi queue bundle hash drifted")
    cache = plan.get("public_precomputed_cache") or {}
    _require(
        cache.get("required") is True
        and cache.get("status") == "completed"
        and cache.get("visibility_mode") == "PUBLIC_ONLY"
        and cache.get("cache_seed") == 42
        and cache.get("private_labels_used") is False,
        "Taxi queue verified-cache contract changed",
    )
    cache_path = ops.ensure_remote_path(str(cache.get("path") or ""))
    _require(
        cache_path.startswith(ops.ALLOWED_GPU_REMOTE_ROOT + "/")
        and Path(cache_path).name == "cache",
        "Taxi queue cache path changed",
    )
    collection = cache.get("collection_evidence") or {}
    collection_path = Path(str(collection.get("path") or "")).resolve()
    _require(
        collection_path.is_file()
        and collection_path.stat().st_size == int(collection.get("bytes", -1))
        and sha256_file(collection_path) == collection.get("sha256"),
        "Taxi queue cache collection evidence drifted",
    )
    route = plan.get("verified_route_stat_sidecar") or {}
    _require(
        route.get("required") is True
        and route.get("schema") == "evomind.mlebench.taxi_route_stat_sidecar.v1"
        and route.get("status") == "completed"
        and route.get("visibility_mode") == "PUBLIC_ONLY"
        and route.get("cache_seed") == 42
        and route.get("folds") == 3
        and route.get("train_rows") == 5_000_000
        and route.get("base_cache_manifest_sha256") == cache.get("manifest_sha256")
        and route.get("validation_targets_used") == 0
        and route.get("private_labels_used") is False
        and route.get("gpu_used") is False,
        "Taxi queue verified route-stat sidecar contract changed",
    )
    route_path = ops.ensure_remote_path(str(route.get("path") or ""))
    expected_route_path = (
        ops.ALLOWED_GPU_REMOTE_ROOT
        + "/evomind_mle22/job89941_taxi_route_stats/"
        "taxi_route_stats_s42_v1_20260728/route_stats"
    )
    _require(route_path == expected_route_path, "Taxi queue route-stat path changed")
    for name in ("manifest", "collection_evidence"):
        item = route.get(name) or {}
        path = Path(str(item.get("path") or "")).resolve()
        _require(
            path.is_file()
            and path.stat().st_size == int(item.get("bytes", -1))
            and sha256_file(path) == item.get("sha256"),
            f"Taxi queue route-stat {name} evidence drifted",
        )
    execution = plan.get("execution_contract") or {}
    _require(
        execution.get("verified_route_stat_cache_required") is True
        and execution.get("gpu_route_stat_rebuild_forbidden") is True,
        "Taxi queue route-stat execution contract changed",
    )
    argv = list(plan.get("runner_contract_args") or [])
    _require(
        argv.count("--taxi-precomputed-cache-dir") == 1
        and argv.count("--taxi-require-precomputed-cache") == 1
        and argv.count("--taxi-cache-seed") == 1,
        "Taxi queue mandatory cache arguments changed",
    )
    _require(
        argv.count("--taxi-route-stat-cache-dir") == 1
        and argv.count("--taxi-require-route-stat-cache") == 1
        and argv[argv.index("--taxi-route-stat-cache-dir") + 1] == route_path,
        "Taxi queue mandatory route-stat arguments changed",
    )
    plan["_path"] = str(resolved)
    plan["_sha256"] = sha256_file(resolved)
    plan["_bundle_verification"] = verified
    return plan


def remote_run_terminal(status: Mapping[str, Any]) -> bool:
    process = status.get("process")
    checkpoint = status.get("checkpoint") if isinstance(status.get("checkpoint"), dict) else {}
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    completed = checkpoint.get("completed") if isinstance(checkpoint.get("completed"), dict) else {}
    return bool(
        process == "stopped"
        and (
            completed
            or summary.get("status") in {"candidate_complete", "partial_failure", "completed"}
            or summary.get("competition_count") == 1
        )
    )


def _base(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "evomind.hpc88240.taxi_source_audited_queue.v1",
        "created_at": now_iso(),
        "plan_path": plan["_path"],
        "plan_sha256": plan["_sha256"],
        "bundle_sha256": plan["_bundle_verification"]["sha256"],
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }


def run_once(
    plan: Mapping[str, Any],
    *,
    may_status_path: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    status_path = evidence_dir / "status_current.json"
    may_local = json.loads(Path(may_status_path).read_text(encoding="utf-8"))
    may_dependency = plan.get("serial_dependency") or {}
    may_plan_sha = str(
        (((plan.get("serial_dependency") or {}).get("plan") or {}).get("sha256") or "")
    )
    if may_local.get("plan_sha256") != may_plan_sha:
        result = {
            **_base(plan),
            "status": "waiting_for_matching_may_launch",
            "may_queue_status": may_local.get("status"),
            "expected_may_plan_sha256": may_plan_sha,
            "observed_may_plan_sha256": may_local.get("plan_sha256"),
        }
        write_json_atomic(status_path, result)
        return result
    if may_local.get("status") != "all_seeds_terminal":
        result = {
            **_base(plan),
            "status": "waiting_for_may_multiseed_terminal",
            "may_queue_status": may_local.get("status"),
        }
        write_json_atomic(status_path, result)
        return result
    expected_may_runs = list(may_dependency.get("run_ids") or [])
    completed_may_runs = [str(item.get("run_id")) for item in may_local.get("completed_seeds") or []]
    _require(
        may_dependency.get("all_seeds_terminal_before_taxi") is True
        and expected_may_runs
        and completed_may_runs == expected_may_runs,
        "Taxi queue May multiseed terminal evidence changed",
    )

    completed: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for seed, run_id in zip(plan["seeds"], plan["run_ids"], strict=True):
        try:
            remote = ops.read_remote_status(str(run_id))
        except (ops.RemoteOpsError, OSError, ValueError, json.JSONDecodeError):
            remote = None
        if remote is not None and remote_run_terminal(remote):
            completed.append({"seed": seed, "run_id": run_id, "status": remote})
            continue
        if remote is not None and remote.get("process") == "running":
            active = {"seed": seed, "run_id": run_id, "status": remote}
            break

        gate_path = evidence_dir / "gpu_gate_current.json"
        gate = ops.sample_gpu_idle_gate(interval_seconds=5)
        write_json_atomic(gate_path, gate)
        if gate.get("passed") is not True:
            result = {
                **_base(plan),
                "status": "waiting_for_gpu_idle",
                "completed_seeds": completed,
                "next_seed": seed,
                "gpu_gate": gate,
            }
            write_json_atomic(status_path, result)
            return result
        deployment = ops.deploy_bundle(ops.DEFAULT_BUNDLE, gate_path, max_gate_age=600)
        smoke = ops.cuda_smoke(ops.DEFAULT_BUNDLE, gate_path, max_gate_age=600)
        final_gate = ops.sample_gpu_idle_gate(interval_seconds=5)
        write_json_atomic(gate_path, final_gate)
        _require(final_gate.get("passed") is True, "Taxi queue final GPU gate failed")
        start = ops.start_run(
            ops.DEFAULT_BUNDLE,
            gate_path,
            run_id=str(run_id),
            waves=["Wave1"],
            competitions=["new-york-city-taxi-fare-prediction"],
            seed=int(seed),
            optimization_plan_name="medal_recovery_gpt56_current.json",
            resume=False,
            allow_concurrent_with_cpu_light=False,
            max_gate_age=600,
            runner_contract_args=list(plan["runner_contract_args"]),
        )
        active = {
            "seed": seed,
            "run_id": run_id,
            "deployment": deployment,
            "cuda_smoke": smoke,
            "gpu_gate": final_gate,
            "start": start,
        }
        break

    result = {
        **_base(plan),
        "status": "all_seeds_terminal" if len(completed) == len(plan["seeds"]) else "seed_active",
        "completed_seeds": completed,
        "active_seed": active,
    }
    write_json_atomic(status_path, result)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--may-status", type=Path, default=DEFAULT_MAY_STATUS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--deadline-hours", type=float, default=336.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 30 or args.deadline_hours <= 0:
        raise ValueError("Taxi queue timing contract is invalid")
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now().astimezone() + timedelta(hours=args.deadline_hours)
    while True:
        try:
            plan = validate_plan(args.plan)
            result = run_once(
                plan,
                may_status_path=Path(args.may_status).resolve(),
                evidence_dir=evidence_dir,
            )
        except Exception as exc:
            result = {
                "schema": "evomind.hpc88240.taxi_source_audited_queue_failure.v1",
                "created_at": now_iso(),
                "status": "transient_or_contract_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "process_signals_sent": 0,
                "other_processes_modified": False,
                "official_grader_executed": False,
                "kaggle_submission_executed": False,
            }
            write_json_atomic(evidence_dir / "failure_current.json", result)
            if args.once:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result["status"] == "all_seeds_terminal":
                return 0
            if args.once:
                return 0
        if datetime.now().astimezone() >= deadline:
            return 4
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
