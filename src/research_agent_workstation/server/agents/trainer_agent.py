from __future__ import annotations

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class TrainerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("TrainerAgent", "training")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(
            status="success",
            summary="Local Python runner completed baseline training and produced metric artifacts.",
            decisions=["Use local baseline result as experiment graph node."],
            generated_artifacts=[artifact.get("path", "") for artifact in agent_input.current_artifacts if artifact.get("path")],
            next_actions=["Review metrics", "Run submission check"],
        )
