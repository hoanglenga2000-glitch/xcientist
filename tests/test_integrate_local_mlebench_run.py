from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import integrate_local_mlebench_run as local_integration


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_completed_run(root: Path, run_id: str = "local_run") -> Path:
    run_dir = root / run_id
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "status": "passed",
            "training_started": True,
            "human_gate_preserved": True,
            "kaggle_submission_enabled": False,
        },
    )
    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "status": "passed",
            "competition_count": 1,
            "passed": 1,
            "failed": 0,
            "results": [
                {
                    "competition_id": "dog-breed-identification",
                    "status": "passed",
                    "valid_submission": True,
                    "official_grader_executed": True,
                    "mle_private_grader_score": 0.031,
                    "private_grader": {"upstream_report": {"any_medal": True}},
                }
            ],
        },
    )
    return run_dir


def test_validate_completed_local_run_accepts_official_single_task(tmp_path: Path):
    run_dir = make_completed_run(tmp_path)

    validated = local_integration.validate_completed_local_run(
        run_dir,
        "dog-breed-identification",
        allowed_root=tmp_path,
    )

    assert validated["run_id"] == "local_run"
    assert validated["score"] == pytest.approx(0.031)
    assert validated["any_medal"] is True


@pytest.mark.parametrize(
    ("manifest_patch", "summary_patch"),
    [
        ({"human_gate_preserved": False}, {}),
        ({"kaggle_submission_enabled": True}, {}),
        ({"status": "running"}, {}),
        ({}, {"status": "partial_failure"}),
        ({}, {"failed": 1}),
    ],
)
def test_validate_completed_local_run_fails_closed(
    tmp_path: Path,
    manifest_patch: dict,
    summary_patch: dict,
):
    run_dir = make_completed_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest.update(manifest_patch)
    summary.update(summary_patch)
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)

    with pytest.raises(ValueError):
        local_integration.validate_completed_local_run(
            run_dir,
            "dog-breed-identification",
            allowed_root=tmp_path,
        )


def test_validate_completed_local_run_requires_upstream_medal_boolean(tmp_path: Path):
    run_dir = make_completed_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["results"][0]["private_grader"]["upstream_report"].pop("any_medal")
    write_json(run_dir / "summary.json", summary)

    with pytest.raises(ValueError, match="any_medal"):
        local_integration.validate_completed_local_run(
            run_dir,
            "dog-breed-identification",
            allowed_root=tmp_path,
        )


def test_campaign_lock_retries_live_owner_then_succeeds(monkeypatch):
    calls = {"count": 0}
    clock = {"now": 0.0}

    def acquire() -> None:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError(local_integration.CAMPAIGN_LOCK_BUSY_MESSAGE)

    monkeypatch.setattr(local_integration.campaign, "acquire_lock", acquire)

    evidence = local_integration.acquire_campaign_lock_with_retry(
        timeout_seconds=20,
        poll_seconds=5,
        monotonic=lambda: clock["now"],
        sleep=lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    assert evidence == {
        "attempts": 3,
        "waited_seconds": 10.0,
        "timeout_seconds": 20.0,
        "poll_seconds": 5.0,
    }


def test_campaign_lock_timeout_is_bounded(monkeypatch):
    clock = {"now": 0.0}
    sleeps: list[float] = []

    def acquire() -> None:
        raise RuntimeError(local_integration.CAMPAIGN_LOCK_BUSY_MESSAGE)

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(local_integration.campaign, "acquire_lock", acquire)

    with pytest.raises(TimeoutError, match="campaign lock"):
        local_integration.acquire_campaign_lock_with_retry(
            timeout_seconds=12,
            poll_seconds=5,
            monotonic=lambda: clock["now"],
            sleep=sleep,
        )

    assert sleeps == [5, 5, 2]


def test_campaign_lock_does_not_mask_unexpected_errors(monkeypatch):
    def acquire() -> None:
        raise RuntimeError("unexpected lock corruption")

    monkeypatch.setattr(local_integration.campaign, "acquire_lock", acquire)

    with pytest.raises(RuntimeError, match="unexpected lock corruption"):
        local_integration.acquire_campaign_lock_with_retry(
            timeout_seconds=20,
            poll_seconds=5,
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
        )
