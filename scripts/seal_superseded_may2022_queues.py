#!/usr/bin/env python3
"""Fail-close superseded local May-2022 queue watchers without stopping them."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HPC_ROOT = PROJECT_ROOT / "workspace" / "hpc"
DEFAULT_AUTHORITATIVE_DIR = (
    DEFAULT_HPC_ROOT / "job88240_may2022_nested_queue_v8b_multiseed_final"
)
DEFAULT_REPORT = DEFAULT_HPC_ROOT / "may2022_supersession_seal_current.json"
ACTIVE_OR_TERMINAL = {
    "training_launched",
    "seed_active",
    "all_seeds_terminal",
    "candidate_complete",
    "completed",
}


class SupersessionSealError(RuntimeError):
    """Raised when a queue cannot be truthfully sealed."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SupersessionSealError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def create_json_exclusive(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return True


def _require_unlaunched(status: Mapping[str, Any], status_path: Path) -> None:
    if status.get("status") in ACTIVE_OR_TERMINAL:
        raise SupersessionSealError(
            f"refusing to seal active or terminal queue: {status_path}"
        )
    required = {
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    for key, expected in required.items():
        if status.get(key) != expected:
            raise SupersessionSealError(
                f"queue safety field changed: {status_path}: {key}"
            )


def _first_seed(plan: Mapping[str, Any]) -> int:
    seeds = plan.get("seeds")
    if isinstance(seeds, list) and seeds:
        return int(seeds[0])
    if plan.get("seed") is not None:
        return int(plan["seed"])
    raise SupersessionSealError("May queue plan does not declare a seed")


def _claim_path(queue_dir: Path, status: Mapping[str, Any], plan: Mapping[str, Any]) -> Path:
    schema = str(status.get("schema") or "")
    if schema == "evomind.hpc88240.may2022_nested_queue.v1":
        return queue_dir / "launch_claim.json"
    if schema == "evomind.hpc88240.may2022_cached_queue.v1":
        return queue_dir / f"launch_claim_s{_first_seed(plan)}.json"
    raise SupersessionSealError(f"unsupported May queue schema: {schema}")


def seal_superseded_queues(
    *,
    hpc_root: Path = DEFAULT_HPC_ROOT,
    authoritative_dir: Path = DEFAULT_AUTHORITATIVE_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    hpc_root = Path(hpc_root).resolve()
    authoritative_dir = Path(authoritative_dir).resolve()
    authoritative_status_path = authoritative_dir / "status_current.json"
    authoritative_status = read_json(authoritative_status_path)
    _require_unlaunched(authoritative_status, authoritative_status_path)
    authoritative_plan_sha = str(authoritative_status.get("plan_sha256") or "")
    authoritative_bundle_sha = str(
        (authoritative_status.get("bundle") or {}).get("sha256") or ""
    )
    if len(authoritative_plan_sha) != 64 or len(authoritative_bundle_sha) != 64:
        raise SupersessionSealError("authoritative queue hashes are incomplete")

    records: list[dict[str, Any]] = []
    for queue_dir in sorted(hpc_root.glob("job88240_may2022_nested_queue*")):
        queue_dir = queue_dir.resolve()
        if queue_dir == authoritative_dir or not queue_dir.is_dir():
            continue
        status_path = queue_dir / "status_current.json"
        if not status_path.is_file():
            continue
        status = read_json(status_path)
        schema = str(status.get("schema") or "")
        if schema not in {
            "evomind.hpc88240.may2022_nested_queue.v1",
            "evomind.hpc88240.may2022_cached_queue.v1",
        }:
            continue
        _require_unlaunched(status, status_path)
        plan_path = Path(str(status.get("plan_path") or "")).resolve()
        if not plan_path.is_file():
            raise SupersessionSealError(f"missing superseded plan: {plan_path}")
        plan = read_json(plan_path)
        observed_plan_sha = sha256_file(plan_path)
        loaded_plan_sha = str(status.get("plan_sha256") or "")
        if len(loaded_plan_sha) != 64:
            raise SupersessionSealError(
                f"superseded queue has no frozen plan hash: {status_path}"
            )
        claim_path = _claim_path(queue_dir, status, plan)
        seal = {
            "schema": "evomind.local_queue_supersession_seal.v1",
            "created_at": now_iso(),
            "status": "superseded_without_launch",
            "queue_dir": str(queue_dir),
            "superseded_plan_path": str(plan_path),
            "superseded_loaded_plan_sha256": loaded_plan_sha,
            "current_plan_path_sha256": observed_plan_sha,
            "plan_path_hash_matches_loaded_status": observed_plan_sha == loaded_plan_sha,
            "superseded_by": str(authoritative_dir),
            "authoritative_plan_sha256": authoritative_plan_sha,
            "authoritative_bundle_sha256": authoritative_bundle_sha,
            "does_not_assert_remote_launch": True,
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        }
        created = create_json_exclusive(claim_path, seal)
        existing = read_json(claim_path)
        if existing.get("schema") != seal["schema"]:
            raise SupersessionSealError(
                f"pre-existing non-supersession claim requires review: {claim_path}"
            )
        records.append(
            {
                "queue_dir": str(queue_dir),
                "status_before_seal": status.get("status"),
                "loaded_plan_sha256": loaded_plan_sha,
                "current_plan_path_sha256": observed_plan_sha,
                "plan_path_hash_matches_loaded_status": observed_plan_sha
                == loaded_plan_sha,
                "claim_path": str(claim_path),
                "claim_sha256": sha256_file(claim_path),
                "created": created,
            }
        )

    report = {
        "schema": "evomind.local_queue_supersession_seal_report.v1",
        "created_at": now_iso(),
        "status": "sealed",
        "authoritative_queue_dir": str(authoritative_dir),
        "authoritative_plan_sha256": authoritative_plan_sha,
        "authoritative_bundle_sha256": authoritative_bundle_sha,
        "superseded_queue_count": len(records),
        "superseded_queues": records,
        "authoritative_queue_sealed": False,
        "watchers_stopped": 0,
        "process_signals_sent": 0,
        "other_processes_modified": False,
        "official_grader_executed": False,
        "kaggle_submission_executed": False,
    }
    write_json_atomic(report_path, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpc-root", type=Path, default=DEFAULT_HPC_ROOT)
    parser.add_argument(
        "--authoritative-dir", type=Path, default=DEFAULT_AUTHORITATIVE_DIR
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = seal_superseded_queues(
        hpc_root=args.hpc_root,
        authoritative_dir=args.authoritative_dir,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
