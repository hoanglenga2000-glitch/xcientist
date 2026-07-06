from __future__ import annotations

from dataclasses import dataclass

from ..schemas.agent import AgentInput, AgentOutput


@dataclass(slots=True)
class BaseAgent:
    name: str
    stage: str

    def run(self, agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(status="skipped", summary=f"{self.name} has no implementation for {agent_input.stage}.")
