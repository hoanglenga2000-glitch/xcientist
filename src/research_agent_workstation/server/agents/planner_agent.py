from __future__ import annotations

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("PlannerAgent", "experiment_planning")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        memories = agent_input.memory_context
        memory_hint = f"{len(memories)} related memory records available." if memories else "No related memory yet."
        return AgentOutput(
            status="waiting_gate",
            summary=f"Scaffold and first baseline plan created. {memory_hint}",
            decisions=["Start with local baseline before external Code Agent optimization.", "Require PLAN_APPROVAL before training."],
            next_actions=["Open PLAN_APPROVAL gate", "Export context for Codex/Claude Code if baseline is insufficient"],
            suggested_gate={"gate_type": "PLAN_APPROVAL", "risk_level": "medium"},
        )
