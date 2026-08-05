from __future__ import annotations

import json
from pathlib import Path

from scripts import watch_taxi_multiseed_completion as watch


def test_waiting_taxi_queue_never_collects_or_stages(tmp_path: Path, monkeypatch):
    plan = watch.queue.validate_plan()
    queue_status = tmp_path / "taxi_status.json"
    queue_status.write_text(
        json.dumps(
            {"status": "waiting_for_may_launch", "plan_sha256": plan["_sha256"]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        watch.ops,
        "collect_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("collect called")),
    )
    monkeypatch.setattr(
        watch.stage,
        "stage_candidate_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stage called")),
    )

    result = watch.run_once(
        plan,
        queue_status_path=queue_status,
        evidence_dir=tmp_path / "evidence",
        package_dir=tmp_path / "package",
        sample_path=tmp_path / "sample.csv",
        staged_run_id="hg-taxi-test",
    )

    assert result["status"] == "waiting_for_taxi_seeds_terminal"
    assert result["process_signals_sent"] == 0
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_mismatched_plan_hash_fails_closed_before_collection(tmp_path: Path, monkeypatch):
    plan = watch.queue.validate_plan()
    queue_status = tmp_path / "taxi_status.json"
    queue_status.write_text(
        json.dumps({"status": "all_seeds_terminal", "plan_sha256": "0" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        watch.ops,
        "collect_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("collect called")),
    )

    result = watch.run_once(
        plan,
        queue_status_path=queue_status,
        evidence_dir=tmp_path / "evidence",
        package_dir=tmp_path / "package",
        sample_path=tmp_path / "sample.csv",
        staged_run_id="hg-taxi-test",
    )

    assert result["status"] == "waiting_for_matching_taxi_plan"
