from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspace.hpc import a800_recovery_queue_supervisor as queue


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_terminal_failure_is_detected_without_exposing_full_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "LANE_DIR", tmp_path)
    run_id = "siim_failed_s42"
    write_json(
        tmp_path / run_id / "status_current.json",
        {
            "process": "stopped",
            "summary": {
                "status": "partial_failure",
                "results": [
                    {
                        "competition_id": "siim-isic-melanoma-classification",
                        "status": "failed",
                        "traceback": "Traceback\nOSError: [Errno 24] Too many open files\n",
                    }
                ],
            },
        },
    )

    failure = queue.read_terminal_failure(
        run_id, "siim-isic-melanoma-classification"
    )

    assert failure is not None
    assert failure["summary_status"] == "partial_failure"
    assert failure["failure_tail"] == "OSError: [Errno 24] Too many open files"
    assert "Traceback" not in failure["failure_tail"]


def test_invalid_completion_waits_for_terminal_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "WATCH_DIR", tmp_path)
    write_json(
        tmp_path / "siim_failed_s42_completion.json",
        {
            "run_id": "siim_failed_s42",
            "process": "stopped",
            "summary_status": "partial_failure",
            "official_grade_count": 0,
            "official_results": [],
            "collection": {"passed": True},
        },
    )


def test_adaptive_queue_uses_only_validated_a800_actions(tmp_path: Path) -> None:
    report = tmp_path / "adaptive.json"
    write_json(
        report,
        {
            "schema": "evomind.mlebench.gpt56_adaptive_loop.v1",
            "status": "passed",
            "ok": True,
            "validation_errors": [],
            "decision": {
                "schema": "evomind.mlebench.adaptive_decision.v1",
                "active_run_policy": [
                    {
                        "lane": "a800",
                        "run_id": "active-siim",
                        "action": "continue_active_runs",
                    }
                ],
                "next_actions": [
                    {
                        "priority": 1,
                        "action": "collect_and_grade",
                        "competition_id": "siim-isic-melanoma-classification",
                        "lane": "cpu",
                        "requires_active_run_completion": True,
                    },
                    {
                        "priority": 2,
                        "action": "queue_experiment",
                        "competition_id": "aptos2019-blindness-detection",
                        "lane": "a800",
                        "requires_active_run_completion": True,
                    },
                    {
                        "priority": 3,
                        "action": "retry_failed_after_active",
                        "competition_id": "ranzcr-clip-catheter-line-classification",
                        "lane": "a800",
                        "requires_active_run_completion": True,
                    },
                ],
            },
        },
    )

    tasks = queue.load_adaptive_tasks(report, active_run_id="active-siim")

    assert [item.competition_id for item in tasks] == [
        "aptos2019-blindness-detection",
        "ranzcr-clip-catheter-line-classification",
    ]
    assert tasks[1].runner_performance_overrides == ("--wave2-workers", "32")

    excluded = queue.load_adaptive_tasks(
        report,
        active_run_id="active-siim",
        excluded_competitions=frozenset({"ranzcr-clip-catheter-line-classification"}),
    )
    assert [item.competition_id for item in excluded] == [
        "aptos2019-blindness-detection"
    ]

    assert (
        queue.read_valid_completion(
            "siim_failed_s42", "siim-isic-melanoma-classification"
        )
        is None
    )


def test_failed_run_is_relaunched_with_same_task_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "LANE_DIR", tmp_path / "lane")
    monkeypatch.setattr(queue, "MARKER_PATH", tmp_path / "queue.json")
    task = queue.QueueTask(
        "siim-isic-melanoma-classification",
        "Wave0",
        "siim",
        ("--siim-workers", "8"),
    )
    completion = {
        "run_id": "siim_retry_s42",
        "official_results": [
            {
                "competition_id": task.competition_id,
                "mle_private_grader_score": 0.94,
            }
        ],
    }
    waits = iter(
        [
            queue.TerminalTaskFailure(
                {
                    "schema": "evomind.mlebench.a800_terminal_failure.v1",
                    "run_id": "siim_failed_s42",
                    "competition_id": task.competition_id,
                    "summary_status": "partial_failure",
                }
            ),
            completion,
        ]
    )
    launched: list[queue.QueueTask] = []

    def fake_wait(*args, **kwargs):
        outcome = next(waits)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(queue, "wait_for_completion", fake_wait)

    def fake_launch(candidate: queue.QueueTask, **kwargs):
        launched.append(candidate)
        return "siim_retry_s42", 1234

    monkeypatch.setattr(queue, "launch_task", fake_launch)

    result, final_run_id, final_bundle_sha256 = queue.wait_for_task_with_recovery(
        "siim_failed_s42",
        task,
        profile_dir=tmp_path / "profile",
        bundle=tmp_path / "bundle.tar.gz",
        poll_seconds=1,
        launch_attempts=2,
        run_attempts=2,
        completed=[],
        active_bundle_sha256=queue.LEGACY_ACTIVE_BUNDLE_SHA256,
    )

    assert result == completion
    assert final_run_id == "siim_retry_s42"
    assert final_bundle_sha256 == queue.LAUNCH_BUNDLE_SHA256
    assert launched == [task]
    assert launched[0].runner_performance_overrides == ("--siim-workers", "8")
    recovery = json.loads(
        (tmp_path / "lane" / "siim_failed_s42" / "terminal_failure_recovery.json").read_text()
    )
    assert recovery["run_attempt"] == 1


def test_resume_queue_preserves_current_run_and_skips_valid_parallel_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "LANE_DIR", tmp_path / "lane")
    selected = (
        queue.ADAPTIVE_TASK_SPECS["jigsaw-toxic-comment-classification-challenge"],
        queue.ADAPTIVE_TASK_SPECS["aptos2019-blindness-detection"],
        queue.ADAPTIVE_TASK_SPECS["aerial-cactus-identification"],
    )
    current_run_id = "a800_aptos_active_s42"
    write_json(
        tmp_path / "lane" / current_run_id / "launch_summary.json",
        {
            "schema": "evomind.mlebench.a800_lane_launch.v1",
            "competition_id": "aptos2019-blindness-detection",
            "run_id": current_run_id,
            "bundle_sha256": queue.LEGACY_ACTIVE_BUNDLE_SHA256,
            "start_passed": True,
            "cpu_light_concurrent": False,
        },
    )
    run_id = "a800_jigsaw_parallel_s42"
    write_json(
        tmp_path / "lane" / run_id / "launch_summary.json",
        {
            "schema": "evomind.mlebench.a800_lane_launch.v1",
            "competition_id": "jigsaw-toxic-comment-classification-challenge",
            "run_id": run_id,
            "bundle_sha256": queue.LEGACY_ACTIVE_BUNDLE_SHA256,
            "start_passed": True,
            "cpu_light_concurrent": True,
        },
    )
    marker = tmp_path / "queue.json"
    write_json(
        marker,
        {
            "schema": "evomind.mlebench.a800_recovery_queue.v1",
            "status": "waiting_for_official_completion",
            "current_run_id": current_run_id,
            "current_competition_id": "aptos2019-blindness-detection",
            "remaining": [
                "aptos2019-blindness-detection",
                "aerial-cactus-identification",
                "aptos2019-blindness-detection",
            ],
            "completed": [],
            "bundle_sha256": queue.LEGACY_ACTIVE_BUNDLE_SHA256,
            "active_bundle_sha256": queue.LEGACY_ACTIVE_BUNDLE_SHA256,
        },
    )

    current, remaining, completed, active_bundle_sha256 = queue.resume_queue_state(
        marker,
        current_run_id=current_run_id,
        current_competition_id="aptos2019-blindness-detection",
        selected_tasks=selected,
        parallel_runs={"jigsaw-toxic-comment-classification-challenge": run_id},
    )

    assert current.competition_id == "aptos2019-blindness-detection"
    assert [task.competition_id for task in remaining] == [
        "aerial-cactus-identification"
    ]
    assert completed == []
    assert active_bundle_sha256 == queue.LEGACY_ACTIVE_BUNDLE_SHA256
