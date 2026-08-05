from scripts.watch_job89941_taxi_cpu_candidate import classify_status


def test_classify_status_requires_matching_live_process_or_terminal_result() -> None:
    assert classify_status({"process": {"running": True, "cmdline_matches": True}}) == "running"
    assert classify_status({"process": {"running": True, "cmdline_matches": False}}) == (
        "process_identity_mismatch"
    )
    assert classify_status({"process": {"running": False}}) == "stopped_without_result"
    assert classify_status(
        {
            "process": {"running": False},
            "result": {"status": "candidate_complete", "passed": True},
        }
    ) == "terminal_result"
    assert classify_status({"result": {"status": "unknown"}}) == "invalid_result"
