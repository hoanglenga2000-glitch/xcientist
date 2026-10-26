from __future__ import annotations

import re
from dataclasses import replace

from .models import ModelDecision

_EFFORTS = frozenset({'low', 'medium', 'high'})
# User intensity is a chat-completions reasoning_effort. Only models that accept
# that field get it. gpt-4o / gpt-4.1 and other OpenAI-compatible ids do not.
_REASONING_MODEL = re.compile(
    r'^(?:o1|o3)(?:$|[-_.])|^o4-mini(?:$|[-_.])|^gpt-5(?:$|[-_.])|^gpt-6-astra(?:$|[-_.])',
    re.IGNORECASE,
)


def available_platform_models() -> list[dict[str, str]]:
    """Models the existing transport resolver can call on this node.

    A provider is listed only when its key is configured. ``route`` is not
    changed; the first entry is still the default ``platform_model``.
    """
    from research_os.agent.messaging import _resolve_transports

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for transport in _resolve_transports():
        provider = str(transport.name or '')
        model = str(getattr(transport.config, 'model', '') or '')
        if not provider or provider == 'none' or not model:
            continue
        item_id = f'{provider}:{model}'
        if item_id in seen:
            continue
        seen.add(item_id)
        rows.append({'id': item_id, 'provider': provider, 'model': model, 'label': model})
    return rows


def platform_transport_for(model_id: str):
    """Return the resolved transport for one catalog id, or None when it is not available."""
    if not isinstance(model_id, str) or not model_id:
        return None
    from research_os.agent.messaging import _resolve_transports

    for transport in _resolve_transports():
        if f'{transport.name}:{transport.config.model}' == model_id:
            return transport
    return None


def model_accepts_reasoning_effort(model: str) -> bool:
    return bool(_REASONING_MODEL.match(str(model or '').strip()))


def apply_reasoning_effort(transport, effort: str):
    """Stamp a user effort onto an OpenAI transport that can accept it.

    Non-reasoning models keep the config already built from the environment
    (including an operator ``OPENAI_REASONING_EFFORT``). Anthropic and DeepSeek
    wires are unchanged. An absent user effort never calls this.
    """
    if effort not in _EFFORTS or getattr(transport, 'name', '') != 'openai':
        return transport
    model = str(getattr(getattr(transport, 'config', None), 'model', '') or '')
    if not model_accepts_reasoning_effort(model):
        return transport
    transport.config = replace(transport.config, reasoning_effort=effort)
    return transport


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
