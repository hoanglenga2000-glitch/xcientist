from __future__ import annotations

from ..schemas.agent import AgentInput, AgentOutput
from .base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ReviewerAgent", "validation_review")

    def run(self, agent_input: AgentInput) -> AgentOutput:
        metrics_artifact = [item for item in agent_input.current_artifacts if "metric" in item.get("name", "").lower() or "model_results" in item.get("name", "")]
        return AgentOutput(
            status="success" if metrics_artifact else "needs_human",
            summary=f"Reviewed {len(agent_input.current_artifacts)} artifacts; metric artifact found={bool(metrics_artifact)}.",
            decisions=["Submission requires valid submission_check and human approval."],
            evidence_refs=[item.get("artifact_id", "") for item in metrics_artifact],
            next_actions=["Open SUBMISSION_APPROVAL gate", "Bind report claims to evidence"],
        )
