from __future__ import annotations

from pathlib import Path

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class DataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("DataAgent", "eda")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        train_path = Path(str(agent_input.task_profile.get("train_path", "")))
        risks = []
        if not train_path.exists():
            risks.append("train_path_missing")
        return AgentOutput(
            status="success" if not risks else "needs_human",
            summary=f"Data contract checked for {agent_input.task_id}; train file exists={train_path.exists()}.",
            generated_artifacts=[],
            risk_flags=risks,
            next_actions=["Generate data_quality artifact", "Plan baseline preprocessing"],
        )
