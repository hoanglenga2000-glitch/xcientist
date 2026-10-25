from __future__ import annotations

from .models import ModelDecision


class ModelRouter:
    """Display metadata for the transport AgentMessageClient will actually call.

    Transport selection, STRICT, and failover stay in ``_resolve_transports``.
    This router only reports that same ordered list so a label cannot name a
    different provider or model than the call.
    """

    def route(self, purpose: str = "execution", policy: str = "balanced") -> ModelDecision:
        from research_os.agent.messaging import _resolve_transports

        transports = _resolve_transports()
        if not transports:
            selected = ("none", "none")
            chain: list[str] = []
        else:
            primary = transports[0]
            selected = (primary.name, primary.config.model)
            chain = [f"{item.name}:{item.config.model}" for item in transports[1:]]
        return ModelDecision(
            purpose,
            selected[0],
            selected[1],
            f"{policy} policy for {purpose}",
            chain,
            purpose in {"critical_review", "claim_audit"},
        )


def remember_served_model(store, session_id: str, provider: str, model: str) -> None:
    """Persist one model.response identity onto the assistant run, if this session is one.

    Non-assistant sessions have no run row. An empty observation must not wipe
    a model already recorded for an earlier turn.
    """
    provider_name = str(provider or "")
    model_name = str(model or "")
    if not provider_name and not model_name:
        return
    try:
        store.update_assistant_run(session_id, model_provider=provider_name, model=model_name)
    except KeyError:
        return
