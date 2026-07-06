from __future__ import annotations

from pathlib import Path

from ..adapters.llm_adapter import LLMAdapter


class ReportService:
    def __init__(self, llm: LLMAdapter) -> None:
        self.llm = llm

    def generate_summary_report(self, output_dir: Path, context: dict) -> Path:
        report = output_dir / "workstation_report.md"
        lines = [
            "# Research Agent Workstation Run Summary",
            "",
            self.llm.draft_report(context),
            "",
            "## Evidence Boundary",
            "",
            "- All conclusions are tied to local artifacts.",
            "- Kaggle and GPU actions remain gated connectors.",
        ]
        report.write_text("\n".join(lines), encoding="utf-8")
        return report

