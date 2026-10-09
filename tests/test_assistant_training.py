from __future__ import annotations

from xsci.assistant_training import _remote_candidate_complete


def _candidate_status() -> dict[str, object]:
    task = "tabular-playground-series-dec-2021"
    return {
        "process": "stopped",
        "manifest": {"status": "candidate_complete"},
        "summary": {"status": "candidate_complete"},
        "checkpoint": {
            "requested": [task],
            "completed": {task: {"status": "candidate_complete"}},
            "remaining": [],
        },
    }


def test_remote_candidate_complete_accepts_committed_stopped_worker() -> None:
    assert _remote_candidate_complete(_candidate_status()) is True


def test_remote_candidate_complete_rejects_incomplete_stopped_worker() -> None:
    status = _candidate_status()
    checkpoint = status["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["remaining"] = ["tabular-playground-series-dec-2021"]

    assert _remote_candidate_complete(status) is False


def test_remote_candidate_complete_rejects_failed_manifest() -> None:
    status = _candidate_status()
    manifest = status["manifest"]
    assert isinstance(manifest, dict)
    manifest["status"] = "failed"

    assert _remote_candidate_complete(status) is False
