from scripts.probe_job89941_taxi_cpu_progress import compute_delta


def test_compute_delta_detects_cpu_or_io_progress() -> None:
    previous = {
        "utime_ticks": 100,
        "stime_ticks": 20,
        "io": {"rchar": 1000, "wchar": 10},
        "log_bytes": 0,
    }
    current = {
        "utime_ticks": 150,
        "stime_ticks": 22,
        "io": {"rchar": 1000, "wchar": 10},
        "log_bytes": 0,
        "result_exists": False,
    }
    delta = compute_delta(current, previous)
    assert delta["utime_ticks"] == 50
    assert delta["making_progress"] is True


def test_compute_delta_marks_no_change_and_handles_first_sample() -> None:
    current = {
        "utime_ticks": 10,
        "stime_ticks": 2,
        "io": {"rchar": 100, "wchar": 5},
        "log_bytes": 0,
        "result_exists": False,
    }
    assert compute_delta(current, None) == {"available": False, "making_progress": None}
    delta = compute_delta(current, dict(current))
    assert delta["making_progress"] is False
