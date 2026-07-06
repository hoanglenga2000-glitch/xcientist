from __future__ import annotations

from abc import abstractmethod

from .base import Adapter


class LLMAdapter(Adapter):
    provider = "llm"

    @abstractmethod
    def summarize_task(self, overview: str, data_profile: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_research_plan(self, task_profile: dict, eda_summary: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def explain_results(self, metrics: dict, evidence: list[dict]) -> str:
        raise NotImplementedError

    @abstractmethod
    def draft_report(self, report_context: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def review_claim(self, claim: str, evidence: list[dict]) -> dict:
        raise NotImplementedError


class RuleBasedLLMAdapter(LLMAdapter):
    provider = "rule_based"

    def summarize_task(self, overview: str, data_profile: dict) -> str:
        return f"Task uses {data_profile.get('train_rows', 'unknown')} training rows and target {data_profile.get('target', 'unknown')}."

    def generate_research_plan(self, task_profile: dict, eda_summary: dict) -> str:
        return "\n".join([
            "Use the configured tabular baseline workflow.",
            "Bind every metric and submission check to evidence artifacts.",
            "Require Human Gate before external submission.",
        ])

    def explain_results(self, metrics: dict, evidence: list[dict]) -> str:
        return f"Best model is {metrics.get('best_model', 'unknown')}; explanation is evidence-bound to {len(evidence)} artifacts."

    def draft_report(self, report_context: dict) -> str:
        return "Rule-based draft report generated from task profile, metrics and evidence manifest."

    def review_claim(self, claim: str, evidence: list[dict]) -> dict:
        return {"claim": claim, "status": "passed" if evidence else "needs_evidence", "evidence_count": len(evidence)}

