from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.seal_superseded_may2022_queues import (
    SupersessionSealError,
    seal_superseded_queues,
)


PLAN_SHA = "1" * 64
BUNDLE_SHA = "2" * 64


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_queue(root: Path, name: str, *, cached: bool, authoritative: bool = False) -> Path:
    queue = root / name
    plan = root / f"{name}.json"
    write_json(plan, {"seeds": [42, 43, 44]} if cached else {"seed": 42})
    import hashlib

    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    write_json(
        queue / "status_current.json",
        {
            "schema": (
                "evomind.hpc88240.may2022_cached_queue.v1"
                if cached
                else "evomind.hpc88240.may2022_nested_queue.v1"
            ),
            "status": "waiting_for_cactus_terminal",
            "plan_path": str(plan),
            "plan_sha256": PLAN_SHA if authoritative else plan_sha,
            "bundle": {"sha256": BUNDLE_SHA},
            "process_signals_sent": 0,
            "other_processes_modified": False,
            "official_grader_executed": False,
            "kaggle_submission_executed": False,
        },
    )
    return queue


def test_seals_legacy_queues_and_excludes_authoritative(tmp_path: Path) -> None:
    authoritative = make_queue(
        tmp_path,
        "job88240_may2022_nested_queue_v8b_multiseed_final",
        cached=True,
        authoritative=True,
    )
    legacy_nested = make_queue(
        tmp_path, "job88240_may2022_nested_queue", cached=False
    )
    legacy_cached = make_queue(
        tmp_path, "job88240_may2022_nested_queue_v7_cache", cached=True
    )
    report_path = tmp_path / "report.json"

    report = seal_superseded_queues(
        hpc_root=tmp_path,
        authoritative_dir=authoritative,
        report_path=report_path,
    )

    assert report["superseded_queue_count"] == 2
    assert report["authoritative_queue_sealed"] is False
    assert not (authoritative / "launch_claim_s42.json").exists()
    assert (legacy_nested / "launch_claim.json").is_file()
    assert (legacy_cached / "launch_claim_s42.json").is_file()
    assert json.loads(
        (legacy_cached / "launch_claim_s42.json").read_text(encoding="utf-8")
    )["status"] == (
        "superseded_without_launch"
    )
    assert report_path.is_file()


def test_refuses_to_seal_active_queue(tmp_path: Path) -> None:
    authoritative = make_queue(
        tmp_path,
        "job88240_may2022_nested_queue_v8b_multiseed_final",
        cached=True,
        authoritative=True,
    )
    legacy = make_queue(tmp_path, "job88240_may2022_nested_queue_v7", cached=True)
    status_path = legacy / "status_current.json"
    status = json.loads(status_path.read_text())
    status["status"] = "seed_active"
    write_json(status_path, status)

    with pytest.raises(SupersessionSealError, match="active or terminal"):
        seal_superseded_queues(
            hpc_root=tmp_path,
            authoritative_dir=authoritative,
            report_path=tmp_path / "report.json",
        )
