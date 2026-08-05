from __future__ import annotations

import json
from pathlib import Path

from scripts import watch_dog_breed_multiseed_completion as watch


def _plan(tmp_path: Path) -> dict:
    return {
        "_path": str((tmp_path / "plan.json").resolve()),
        "_sha256": "a" * 64,
        "confirmation": {
            "run_ids": ["dog-full-s46", "dog-full-s47"],
            "seeds": [46, 47],
        },
    }


def _write_status(path: Path, plan: dict, status: str) -> None:
    path.write_text(
        json.dumps(
            {
                "plan_sha256": plan["_sha256"],
                "status": status,
                "completed_seeds": [
                    {"run_id": run_id}
                    for run_id in plan["confirmation"]["run_ids"]
                ],
            }
        ),
        encoding="utf-8",
    )


def test_waiting_queue_never_collects_or_aggregates(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    queue_status = tmp_path / "queue.json"
    _write_status(queue_status, plan, "full_seed_active")
    monkeypatch.setattr(
        watch.ops,
        "collect_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("collect called")),
    )
    monkeypatch.setattr(
        watch.aggregate,
        "aggregate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("aggregate called")),
    )

    result = watch.run_once(
        plan,
        queue_status_path=queue_status,
        evidence_dir=tmp_path,
        package_dir=tmp_path / "package",
        sample_path=tmp_path / "sample.csv",
        staged_run_id="dog-stage",
    )

    assert result["status"] == "waiting_for_dog_seeds_terminal"
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False


def test_upstream_gate_failure_is_terminal_without_collection(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path)
    queue_status = tmp_path / "queue.json"
    _write_status(queue_status, plan, "diagnostic_gate_failed")
    monkeypatch.setattr(
        watch.ops,
        "collect_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("collect called")),
    )

    result = watch.run_once(
        plan,
        queue_status_path=queue_status,
        evidence_dir=tmp_path,
        package_dir=tmp_path / "package",
        sample_path=tmp_path / "sample.csv",
        staged_run_id="dog-stage",
    )

    assert result["status"] == "upstream_gate_failed"
    assert result["candidate_ready_for_human_gate"] is False


def test_terminal_multiseed_collects_exact_runs_and_stages_human_gate(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _plan(tmp_path)
    queue_status = tmp_path / "queue.json"
    _write_status(queue_status, plan, "all_full_seeds_terminal")
    calls = {"collected": []}

    def collect(run_id: str):
        calls["collected"].append(run_id)
        return {"run_id": run_id, "passed": True}

    monkeypatch.setattr(watch.ops, "collect_run", collect)
    monkeypatch.setattr(
        watch,
        "stage_public_sample",
        lambda *_args, **_kwargs: {"sha256": "b" * 64, "downloaded": False},
    )
    monkeypatch.setattr(
        watch.aggregate,
        "aggregate",
        lambda **_kwargs: {
            "status": "ready_for_human_review_not_submitted",
            "candidate_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(watch.stage, "verify_human_gate_package", lambda _path: object())
    monkeypatch.setattr(
        watch.stage,
        "stage_candidate_run",
        lambda _path, *, run_id: {"status": "verified", "run_id": run_id},
    )

    result = watch.run_once(
        plan,
        queue_status_path=queue_status,
        evidence_dir=tmp_path,
        package_dir=tmp_path / "package",
        sample_path=tmp_path / "sample.csv",
        staged_run_id="dog-stage",
    )

    assert calls["collected"] == ["dog-full-s46", "dog-full-s47"]
    assert result["status"] == "verified_human_approval_pending"
    assert result["approved"] is False
    assert result["automatic_approval"] is False
    assert result["official_grader_executed"] is False
    assert result["kaggle_submission_executed"] is False
