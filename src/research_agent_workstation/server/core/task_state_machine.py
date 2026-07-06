from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TaskState(StrEnum):
    CREATED = "CREATED"
    IMPORTED = "IMPORTED"
    UNDERSTANDING = "UNDERSTANDING"
    UNDERSTOOD = "UNDERSTOOD"
    EDA_RUNNING = "EDA_RUNNING"
    EDA_DONE = "EDA_DONE"
    PLANNING = "PLANNING"
    PLAN_WAITING_APPROVAL = "PLAN_WAITING_APPROVAL"
    PLAN_APPROVED = "PLAN_APPROVED"
    CODE_GENERATING = "CODE_GENERATING"
    CODE_READY = "CODE_READY"
    TRAINING_RUNNING = "TRAINING_RUNNING"
    TRAINING_DONE = "TRAINING_DONE"
    REVIEWING = "REVIEWING"
    REVIEW_DONE = "REVIEW_DONE"
    SUBMISSION_WAITING_APPROVAL = "SUBMISSION_WAITING_APPROVAL"
    REPORT_GENERATING = "REPORT_GENERATING"
    REPORT_DONE = "REPORT_DONE"
    FINAL_WAITING_APPROVAL = "FINAL_WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WAITING_FIX = "WAITING_FIX"


ALLOWED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.CREATED: {TaskState.IMPORTED, TaskState.FAILED},
    TaskState.IMPORTED: {TaskState.UNDERSTANDING, TaskState.FAILED},
    TaskState.UNDERSTANDING: {TaskState.UNDERSTOOD, TaskState.FAILED},
    TaskState.UNDERSTOOD: {TaskState.EDA_RUNNING, TaskState.PLANNING, TaskState.FAILED},
    TaskState.EDA_RUNNING: {TaskState.EDA_DONE, TaskState.FAILED},
    TaskState.EDA_DONE: {TaskState.PLANNING, TaskState.FAILED},
    TaskState.PLANNING: {TaskState.PLAN_WAITING_APPROVAL, TaskState.FAILED},
    TaskState.PLAN_WAITING_APPROVAL: {TaskState.PLAN_APPROVED, TaskState.FAILED},
    TaskState.PLAN_APPROVED: {TaskState.CODE_GENERATING, TaskState.FAILED},
    TaskState.CODE_GENERATING: {TaskState.CODE_READY, TaskState.FAILED},
    TaskState.CODE_READY: {TaskState.TRAINING_RUNNING, TaskState.FAILED},
    TaskState.TRAINING_RUNNING: {TaskState.TRAINING_DONE, TaskState.FAILED, TaskState.WAITING_FIX},
    TaskState.WAITING_FIX: {TaskState.CODE_GENERATING, TaskState.FAILED},
    TaskState.TRAINING_DONE: {TaskState.REVIEWING, TaskState.FAILED},
    TaskState.REVIEWING: {TaskState.REVIEW_DONE, TaskState.FAILED},
    TaskState.REVIEW_DONE: {TaskState.SUBMISSION_WAITING_APPROVAL, TaskState.REPORT_GENERATING, TaskState.FAILED},
    TaskState.SUBMISSION_WAITING_APPROVAL: {TaskState.REPORT_GENERATING, TaskState.FAILED},
    TaskState.REPORT_GENERATING: {TaskState.REPORT_DONE, TaskState.FAILED},
    TaskState.REPORT_DONE: {TaskState.FINAL_WAITING_APPROVAL, TaskState.COMPLETED, TaskState.FAILED},
    TaskState.FINAL_WAITING_APPROVAL: {TaskState.COMPLETED, TaskState.FAILED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: {TaskState.WAITING_FIX},
}


@dataclass(slots=True)
class StateTransition:
    from_state: str
    to_state: str
    reason: str
    at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskStateMachine:
    task_id: str
    state: TaskState = TaskState.CREATED
    history: list[StateTransition] = field(default_factory=list)

    def can_transition(self, to_state: TaskState) -> bool:
        return to_state in ALLOWED_TRANSITIONS[self.state]

    def transition(self, to_state: TaskState, reason: str, metadata: dict[str, Any] | None = None) -> StateTransition:
        if not self.can_transition(to_state):
            raise ValueError(f"Illegal transition for {self.task_id}: {self.state} -> {to_state}")
        record = StateTransition(
            from_state=self.state.value,
            to_state=to_state.value,
            reason=reason,
            at=datetime.now().isoformat(timespec="seconds"),
            metadata=metadata or {},
        )
        self.history.append(record)
        self.state = to_state
        return record

    def require(self, expected: TaskState, action: str) -> None:
        if self.state != expected:
            raise RuntimeError(f"{action} requires state {expected.value}; current state is {self.state.value}.")

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "history": [asdict(transition) for transition in self.history],
        }
