import pytest

# workspace/ is gitignored operator state; the A800 supervisors live there on
# operator machines only, so a clean checkout skips (same convention as the
# video-production tests) instead of failing collection.
supervisor = pytest.importorskip(
    "workspace.hpc.a800_cpu_light_resume_supervisor",
    reason="workspace/hpc A800 supervisor is not present in this checkout (workspace/ is gitignored)",
)


def test_queue_allows_only_registered_parallel_run_after_completion():
    marker = {
        "schema": "evomind.mlebench.a800_recovery_queue.v1",
        "status": "completed",
        "parallel_runs": {"jigsaw": "run-jigsaw"},
    }

    assert supervisor.queue_allows_resume(marker, "run-jigsaw") is True
    assert supervisor.queue_allows_resume(marker, "another-run") is False


def test_queue_does_not_resume_while_gpu_queue_is_active():
    marker = {
        "schema": "evomind.mlebench.a800_recovery_queue.v1",
        "status": "started_and_watched",
        "parallel_runs": {"jigsaw": "run-jigsaw"},
    }

    assert supervisor.queue_allows_resume(marker, "run-jigsaw") is False
