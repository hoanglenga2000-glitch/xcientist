from __future__ import annotations

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class WriterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("WriterAgent", "report_generation")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        evidence_count = len(agent_input.current_artifacts)
        return AgentOutput(
            status="success" if evidence_count else "needs_human",
            summary=f"Report draft generated from {evidence_count} registered artifacts.",
            decisions=["Claims without evidence must remain marked Needs Evidence."],
            next_actions=["Run FINAL_CLAIM_APPROVAL gate", "Export report after approval"],
        )
