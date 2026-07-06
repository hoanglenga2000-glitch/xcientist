from __future__ import annotations

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class TaskReaderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("TaskReaderAgent", "task_understanding")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        target = agent_input.task_profile.get("target")
        metric = agent_input.task_profile.get("metric")
        task_type = agent_input.task_profile.get("task_type")
        return AgentOutput(
            status="success",
            summary=f"Imported {task_type} task with target={target}, metric={metric}.",
            decisions=[f"Use {metric} as primary metric.", "Keep task as tabular MVP but schema remains extensible."],
            next_actions=["Generate task scaffold", "Inspect train/test files"],
        )
