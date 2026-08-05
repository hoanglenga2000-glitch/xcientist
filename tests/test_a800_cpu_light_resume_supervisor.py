from workspace.hpc import a800_cpu_light_resume_supervisor as supervisor


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
