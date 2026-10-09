#!/usr/bin/env python3
"""Refresh the May/Taxi frozen source manifest from canonical repository bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "workspace"
    / "mlebench_plans"
    / "may2022_nested_selection_execution_multiseed_hpc88240_v8_cache_taxi_route_20260728.json"
)


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def refresh(plan_path: Path) -> dict[str, object]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    records = plan.get("source_identity")
    if not isinstance(records, list) or len(records) < 7:
        raise RuntimeError("frozen source identity is incomplete")
    refreshed: list[dict[str, object]] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise RuntimeError("frozen source identity record is invalid")
        relative = str(raw.get("relative_path") or "")
        source = (PROJECT_ROOT / relative).resolve()
        source.relative_to(PROJECT_ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = canonical_bytes(source)
        refreshed.append(
            {
                **raw,
                "path": str(source),
                "relative_path": source.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    plan["source_identity"] = refreshed
    temporary = plan_path.with_suffix(plan_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, plan_path)
    return {"plan": str(plan_path), "refreshed_sources": len(refreshed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    print(json.dumps(refresh(args.plan), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
