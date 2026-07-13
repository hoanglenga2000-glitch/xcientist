from __future__ import annotations

from pathlib import Path

from .code_agent_adapter import LocalTemplateCodeAgentAdapter
from ..schemas.agent import PatchResult


class CodexAdapter(LocalTemplateCodeAgentAdapter):
    provider = "codex"

    def fix_error(self, code_path: Path, error_log: str, run_context: dict) -> PatchResult:
        result = super().fix_error(code_path, error_log, run_context)
        result.source_agent = self.provider
        result.metadata["external_status"] = "reserved; export context and import patch are supported"
        return result

