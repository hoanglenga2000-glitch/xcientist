from __future__ import annotations

import pytest

from research_agent_workstation.server.core.task_state_machine import TaskState, TaskStateMachine


def test_task_state_is_string_compatible_across_supported_python_versions():
    assert isinstance(TaskState.CREATED, str)
    assert str(TaskState.CREATED) == "CREATED"
    assert TaskState.CREATED.value == "CREATED"


def test_task_state_machine_enforces_transition_contract():
    machine = TaskStateMachine("task_01")

    transition = machine.transition(TaskState.IMPORTED, "fixture import")

    assert transition.from_state == "CREATED"
    assert transition.to_state == "IMPORTED"
    assert machine.snapshot()["state"] == "IMPORTED"
    with pytest.raises(ValueError, match="Illegal transition"):
        machine.transition(TaskState.COMPLETED, "skip gates")


def test_hpc_job_stays_manifest_prepared_until_dispatch_receipt_arrives():
    machine = TaskStateMachine("task_hpc")
    for state in (
        TaskState.IMPORTED,
        TaskState.UNDERSTANDING,
        TaskState.UNDERSTOOD,
        TaskState.PLANNING,
        TaskState.PLAN_WAITING_APPROVAL,
        TaskState.MANIFEST_PREPARED,
    ):
        machine.transition(state, "fixture")

    assert machine.state == TaskState.MANIFEST_PREPARED
    with pytest.raises(ValueError, match="remote_job_id"):
        machine.transition(TaskState.TRAINING_QUEUED, "no dispatch evidence")
    machine.transition(
        TaskState.TRAINING_QUEUED,
        "gateway accepted job",
        {
            "remote_job_id": "remote-1",
            "dispatch_receipt": {"job_id": "remote-1", "status": "accepted"},
        },
    )
    assert machine.state == TaskState.TRAINING_QUEUED
    assert not machine.can_transition(TaskState.TRAINING_DONE)
