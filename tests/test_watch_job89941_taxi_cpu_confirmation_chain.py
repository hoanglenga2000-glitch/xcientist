from scripts.watch_job89941_taxi_cpu_confirmation_chain import (
    classify_collection,
    classify_parent_gate,
)


def test_classify_collection_requires_verified_candidate_ready() -> None:
    assert classify_collection({"status": "waiting_for_terminal_result"}) == "running"
    assert classify_collection(
        {
            "status": "completed_and_verified",
            "candidate_ready_for_multiseed_confirmation": True,
        }
    ) == "passed"
    assert classify_collection(
        {
            "status": "completed_and_verified",
            "candidate_ready_for_multiseed_confirmation": False,
        }
    ) == "gate_failed"
    assert classify_collection({"status": "stopped_without_result"}) == "failed"


def test_classify_parent_gate_stops_on_verified_parent_failure() -> None:
    assert classify_parent_gate({"ready": True}) == "ready"
    assert classify_parent_gate({"ready": False, "status": "waiting_for_result"}) == "waiting"
    assert (
        classify_parent_gate(
            {
                "ready": False,
                "status": "completed_and_verified",
                "parent_passed": False,
            }
        )
        == "gate_failed"
    )
