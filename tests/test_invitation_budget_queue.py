import concurrent.futures
import time
import threading

import pytest

from evomind_runtime.execution_progress import queued_resource
from evomind_runtime.research_budget import ResearchBudget


def test_five_callers_share_one_compute_slot(tmp_path):
    active = 0
    maximum = 0
    guard = threading.Lock()
    def work(index):
        nonlocal active, maximum
        with queued_resource(tmp_path, "gpu", f"run-{index}", wait_seconds=10):
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with guard:
                active -= 1
        return index
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        assert sorted(pool.map(work, range(5))) == list(range(5))
    assert maximum == 1


def test_trial_reservations_are_idempotent_and_failed_trials_count(tmp_path):
    budget = ResearchBudget(str(tmp_path / "study.sqlite3"), "study")
    assert budget.reserve("trial-1", "direct_tool_loop", "a"*64, independent_data=True)
    assert not budget.reserve("trial-1", "direct_tool_loop", "a"*64, independent_data=True)
    budget.settle("trial-1", 20, False)
    assert budget.summary()["failed"] == 1
    with pytest.raises(ValueError):
        budget.reserve("seen-holdout", "direct_tool_loop", "a"*64, independent_data=False)
    with pytest.raises(ValueError):
        budget.reserve("long-run", "direct_tool_loop", "a"*64, seconds=3600, independent_data=True)
