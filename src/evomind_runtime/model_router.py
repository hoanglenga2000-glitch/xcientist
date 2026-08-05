from __future__ import annotations

import os

from .models import ModelDecision


class ModelRouter:
    """Explainable provider routing; transport failover remains in AgentMessageClient."""

    def route(self, purpose: str = "execution", policy: str = "balanced") -> ModelDecision:
        configured = []
        if os.getenv("ANTHROPIC_API_KEY"):
            configured.append(("anthropic", os.getenv("ANTHROPIC_MODEL", "claude")))
        if os.getenv("OPENAI_API_KEY"):
            configured.append(("openai", os.getenv("OPENAI_MODEL", "gpt")))
        if os.getenv("DEEPSEEK_API_KEY"):
            configured.append(("deepseek", os.getenv("DEEPSEEK_MODEL", "deepseek")))
        preferred = "anthropic" if purpose in {"planning", "review", "long_context"} else "openai"
        if policy == "economy":
            preferred = "deepseek"
        selected = next((item for item in configured if item[0] == preferred), configured[0] if configured else ("none", "none"))
        return ModelDecision(purpose, selected[0], selected[1], f"{policy} policy for {purpose}", [f"{p}:{m}" for p, m in configured if p != selected[0]], purpose in {"critical_review", "claim_audit"})
