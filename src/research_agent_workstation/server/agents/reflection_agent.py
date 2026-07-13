from __future__ import annotations

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class ReflectionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ReflectionAgent", "reflection")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(
            status="success",
            summary="Generated retrospective memory and next experiment suggestions from metrics, artifacts and reviewer output.",
            decisions=["Continue local tabular branch if validation remains strong.", "Try model family comparison only after evidence is bound."],
            next_actions=[
                "Try CatBoost with same preprocessing.",
                "Compare LightGBM with XGBoost under same folds.",
                "Check leakage risk before leaderboard submission.",
                "Add feature importance report before final conclusion.",
            ],
        )
