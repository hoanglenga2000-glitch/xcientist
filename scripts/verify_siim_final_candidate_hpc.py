#!/usr/bin/env python3
"""Validate an HPC SIIM plan and reuse the independent candidate verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import verify_siim_final_candidate as base  # noqa: E402

ALLOWED_REMOTE_ROOT = PurePosixPath(
    "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_remote_path(raw_path: str) -> None:
    try:
        PurePosixPath(raw_path).relative_to(ALLOWED_REMOTE_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"SIIM HPC path escaped the dedicated root: {raw_path}") from exc


def validate_hpc_plan(path: Path) -> dict[str, Any]:
    path = path.resolve()
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    require(
        plan.get("schema") == "evomind.siim.final_candidate_frozen_plan.v1",
        "SIIM HPC plan schema differs",
    )
    require(plan.get("status") == "frozen_waiting_prerequisites", "SIIM HPC plan is not frozen")
    target = plan.get("execution_target") or {}
    require(target.get("job_id") == 88240, "SIIM HPC plan targets a different job")
    require(target.get("gpu") == "NVIDIA A40", "SIIM HPC plan targets a different GPU")
    require(target.get("process_signals_allowed") is False, "SIIM HPC plan permits signals")
    training = plan.get("training") or {}
    require(training.get("candidate_only") is True, "SIIM HPC plan enables grading")
    require(training.get("private_labels_used") is False, "SIIM HPC plan uses private labels")
    require(training.get("official_grader_executed") is False, "SIIM HPC plan ran grader")
    require(training.get("kaggle_submission_executed") is False, "SIIM HPC plan submitted")
    require(training.get("outer_folds") == 5, "SIIM HPC outer folds changed")
    require(training.get("inner_folds") == 3, "SIIM HPC inner folds changed")
    require(training.get("backbone") == "convnext_small", "SIIM HPC primary backbone changed")
    require(
        training.get("secondary_backbone") == "efficientnet_v2_s",
        "SIIM HPC secondary backbone changed",
    )
    for name in (
        "data_root",
        "output_root",
        "allowed_root",
        "official_source_root",
        "torch_home",
    ):
        require_remote_path(str(training.get(name, "")))
    for name in ("full_runner", "wave0", "wave2", "adapter"):
        record = (plan.get("implementation") or {}).get(name) or {}
        source = Path(str(record.get("path", ""))).resolve()
        require(source.is_file(), f"SIIM HPC source is missing: {name}")
        require(sha256_file(source) == record.get("sha256"), f"SIIM HPC source drifted: {name}")
    require(
        (plan.get("launch_contract") or {}).get("process_signals_allowed") is False,
        "SIIM HPC launch contract permits signals",
    )
    plan["_plan_path"] = str(path)
    plan["_plan_sha256"] = sha256_file(path)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--queue-status", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = validate_hpc_plan(args.plan)
    run_dir = args.run_dir or (
        Path(plan["training"]["output_root"]) / plan["training"]["run_id"]
    )
    queue_status = args.queue_status or Path(plan["launch_contract"]["queue_status"])
    output = args.output or Path(run_dir) / "independent_verification.json"
    report = base.verify_run(plan, Path(run_dir), Path(queue_status))
    base.write_json_atomic(Path(output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
