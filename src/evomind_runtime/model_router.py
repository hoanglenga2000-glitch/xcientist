from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from .models import ModelDecision

_EFFORTS = frozenset({'low', 'medium', 'high'})
# User intensity is a chat-completions reasoning_effort. Only models that accept
# that field get it. gpt-4o / gpt-4.1 and other OpenAI-compatible ids do not.
_REASONING_MODEL = re.compile(
    r'^(?:o1|o3)(?:$|[-_.])|^o4-mini(?:$|[-_.])|^gpt-5(?:$|[-_.])|^gpt-6(?:$|[-_.])',
    re.IGNORECASE,
)


# Platform models per provider (tri_vendor_v1, probed 2026-11-03 with one real
# call each). A provider is offered only when its key resolved a transport;
# each entry reuses that transport's endpoint and key with the model swapped.
#   openai   = api.zhjjq.tech Sub2API, GPT 6 and above
#   deepseek = Alibaba Bailian token-plan compatible-mode (国模)
# Anthropic extras come only from the probed lt4net catalog (see `_anthropic_catalog`).
PLATFORM_CATALOG: dict[str, tuple[str, ...]] = {
    'deepseek': ('deepseek-v4.1-flash', 'deepseek-v4-pro', 'qwen3.8-max', 'qwen3.8-flash', 'glm-5.3'),
    'openai': ('gpt-6-astra', 'gpt-6-luna', 'gpt-6-sol', 'gpt-6.1-sol'),
}

# The picker offers only the Claude 5.5 family from lt4net (the probed list keeps
# every id that answered so a run on an older id still resolves its name).
PICKER_CLAUDE_MODELS: tuple[str, ...] = ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5')

# t237 C0 / t219u: ids that fail reproducibly on live, kept out of the picker
# only. Hidden until root cause fixed; remove an id to re-list it.
PICKER_HIDDEN_MODEL_IDS: frozenset[str] = frozenset({
    'deepseek:deepseek-chat', 'deepseek:deepseek-reasoner', 'openai:gpt-4.1', 'openai:deepseek-flash',
    # 2026-11-03: lt4net answers claude-sonnet-5-5 with an empty turn (0/6 non-empty, 1 output token,
    # stop=end_turn; opus-5-5 and sonnet-5 6/6). Hidden until the gateway serves it again.
    'anthropic:claude-sonnet-5-5',
})

_LT4NET = json.loads((Path(__file__).with_name('lt4net_models.json')).read_text(encoding='utf-8'))
LT4NET_HOST: str = _LT4NET['gateway_host']
LT4NET_MODELS: tuple[str, ...] = tuple(_LT4NET['models'])
_LT4NET_ALIASES: dict[str, str] = dict(_LT4NET['aliases'])

# Product/UI default on the probed lt4net gateway. The node env model (e.g. the
# IOAI profile's claude-sonnet-5) stays the default everywhere else.
PRODUCT_DEFAULT_ANTHROPIC_MODEL = 'claude-opus-5-5'


def _anthropic_catalog(transport) -> tuple[str, ...]:
    """Probed lt4net Claude ids, only when the anthropic transport points at that gateway.

    Any other host (api.anthropic.com, a local proxy) gets no extras: nothing
    else was probed, so only the env default is offered there.
    """
    host = (urlparse(str(transport.config.base_url or '')).hostname or '').lower()
    if host != LT4NET_HOST:
        return ()
    return tuple(model for model in PICKER_CLAUDE_MODELS if model in LT4NET_MODELS)


def _catalog(transport) -> tuple[str, ...]:
    if transport.name == 'anthropic':
        return _anthropic_catalog(transport)
    return PLATFORM_CATALOG.get(str(transport.name or ''), ())


def _with_model(transport, model: str):
    """Same provider wire, endpoint, key and operator request profile; another model id."""
    return type(transport)(replace(transport.config, model=model))


def _product_default(transport):
    """The lt4net anthropic transport on the product default model; any other transport as is."""
    if transport.name != 'anthropic' or PRODUCT_DEFAULT_ANTHROPIC_MODEL not in _anthropic_catalog(transport):
        return transport
    if _LT4NET_ALIASES.get(transport.config.model, transport.config.model) == PRODUCT_DEFAULT_ANTHROPIC_MODEL:
        return transport
    return _with_model(transport, PRODUCT_DEFAULT_ANTHROPIC_MODEL)


def product_default_transports():
    """Resolved transports with the product default applied, or None when it changes nothing.

    Same providers, order and STRICT gate as ``_resolve_transports``; only the
    lt4net anthropic model is swapped. None keeps callers on the env path.
    """
    from research_os.agent.messaging import _resolve_transports

    resolved = _resolve_transports()
    swapped = [_product_default(item) for item in resolved]
    return swapped if any(a is not b for a, b in zip(swapped, resolved)) else None


def empty_failover_transport(provider: str, model: str):
    """t238 C7e R2: the one transport an implicit run may fall to after repeated empty answers.

    The default model of the first configured provider other than the one that answered empty
    (same ordered, STRICT-gated list as ``_resolve_transports``). Never a sibling model of the failed
    provider, never a picker-hidden id; None when nothing qualifies. ``model`` is kept for the record.
    """
    from research_os.agent.messaging import _resolve_transports

    del model
    failed = str(provider or '')
    for transport in map(_product_default, _resolve_transports()):
        name = str(transport.name or '')
        served = str(getattr(transport.config, 'model', '') or '')
        if not name or name == 'none' or not served or name == failed:
            continue
        if f'{name}:{served}' in PICKER_HIDDEN_MODEL_IDS:
            continue
        return transport
    return None


def _platform_transports() -> list[tuple[str, object]]:
    """(id, transport) for each callable platform model, default of each provider first.

    ``_resolve_transports`` still decides which providers exist (key gate,
    STRICT, primary order); this only widens the model list inside each one.
    The default is the env model, except on lt4net where the product default
    leads. An alias of the default (claude-sonnet-5.5 vs claude-sonnet-5-5) is
    listed once, under the default's spelling.
    """
    from research_os.agent.messaging import _resolve_transports

    rows: list[tuple[str, object]] = []
    seen: set[str] = set()
    for transport in map(_product_default, _resolve_transports()):
        provider = str(transport.name or '')
        default = str(getattr(transport.config, 'model', '') or '')
        if not provider or provider == 'none' or not default:
            continue
        for model in (default, *_catalog(transport)):
            key = f'{provider}:{_LT4NET_ALIASES.get(model, model)}'
            if key in seen:
                continue
            seen.add(key)
            rows.append((f'{provider}:{model}', transport if model == default else _with_model(transport, model)))
    return _drop_openai_duplicates_of_deepseek(rows)


def _chat_target(transport) -> tuple[str, str]:
    from research_os.llm_client import _openai_chat_url

    return _openai_chat_url(str(getattr(transport.config, 'base_url', '') or '')), str(transport.config.model)


def _drop_openai_duplicates_of_deepseek(rows: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """Skip an ``openai:`` row that calls the same chat URL and model as a ``deepseek:`` row.

    On live both wires point at Pezayo and ``OPENAI_MODEL=deepseek-flash``, so
    ``openai:deepseek-flash`` is the deepseek-flash line again, but on the governed
    path (t237 C4, MODEL_ROOT_CAUSE fix 2). Listing only; no transport changes.
    """
    deepseek = {_chat_target(t) for item_id, t in rows if item_id.startswith('deepseek:')}
    return [(item_id, t) for item_id, t in rows
            if not (item_id.startswith('openai:') and _chat_target(t) in deepseek)]


def available_platform_models() -> list[dict[str, str]]:
    """Models the existing transport resolver can call on this node.

    A provider is listed only when its key is configured, and then with every
    known model. ``route`` is not changed; the first entry is still the default
    ``platform_model``.
    """
    return [{'id': item_id, 'provider': transport.name, 'model': transport.config.model,
             'label': model_display_name(transport.config.model)}
            for item_id, transport in _platform_transports() if item_id not in PICKER_HIDDEN_MODEL_IDS]


def platform_transport_for(model_id: str):
    """Return a fresh transport for one listed id, or None when it is not available."""
    if not isinstance(model_id, str) or not model_id:
        return None
    for item_id, transport in _platform_transports():
        if item_id == model_id:
            return transport
    return None


# Served forms a gateway may report for the requested id (t237 C4.1). Pezayo
# answers deepseek-flash as deepseek/deepseek-v4.1-flash (MODEL_ROOT_CAUSE).
_SERVED_ALIASES: dict[str, frozenset[str]] = {'deepseek-flash': frozenset({'deepseek/deepseek-v4.1-flash'})}
_SNAPSHOT_SUFFIX = re.compile(r'-(?:\d{8}|\d{4}-\d{2}-\d{2})$')


def served_identity_matches(requested: str, served) -> bool:
    """True when the reported model is the requested one, not a substitute.

    Accepts the exact id, the gateway's ``vendor/<id>`` form, a dated provider
    snapshot of the id, an lt4net alias spelling, and the documented served
    aliases above. Anything else (e.g. deepseek-v4-pro answered as
    deepseek/deepseek-v4.1-flash) is another model.
    """
    requested = str(requested or '').strip()
    served = str(served or '').strip() if isinstance(served, str) else ''
    if not requested or not served:
        return False
    if served in _SERVED_ALIASES.get(requested, ()):
        return True
    bare = _SNAPSHOT_SUFFIX.sub('', served.rsplit('/', 1)[-1])
    canon = lambda name: _LT4NET_ALIASES.get(name, name)
    return canon(bare) == canon(_SNAPSHOT_SUFFIX.sub('', requested))


def require_served_identity(transport):
    """The same transport, but a reply from another model fails as model_identity_unconfirmed.

    Used only for an explicit picker selection; wire, endpoint, key and request
    are unchanged. Not retryable: repeating the request reaches the same channel.
    """
    base = type(transport)
    checked = _IDENTITY_CHECKED.get(base)
    if checked is None:
        def parse(self, body):
            turn = base.parse(self, body)
            if not served_identity_matches(self.config.model, turn.model):
                from .model_transport import ModelTransportError
                raise ModelTransportError('model_identity_unconfirmed', retryable=False)
            return turn
        checked = _IDENTITY_CHECKED[base] = type(f'IdentityChecked{base.__name__}', (base,), {'parse': parse})
    return checked(transport.config)


_IDENTITY_CHECKED: dict[type, type] = {}


_BRANDS = {'claude': 'Claude', 'gpt': 'GPT', 'deepseek': 'DeepSeek', 'grok': 'Grok', 'kimi': 'Kimi', 'mini': 'mini', 'glm': 'GLM'}
_NUMBER = re.compile(r'^\d+(?:\.\d+)*$')
_DEEPSEEK_VERSION = re.compile(r'^v(\d+(?:\.\d+)*)$')
# Gateway alias -> the model it serves. Pezayo answers deepseek-flash as
# deepseek/deepseek-v4.1-flash (model_observed, t237 C4 MODEL_ROOT_CAUSE).
_DISPLAY_ALIASES = {'deepseek-flash': 'deepseek-v4.1-flash'}


def model_display_name(model: str) -> str:
    """Product name with its exact version (claude-opus-4-8 -> Claude Opus 4.8).

    Brand-cased tokens; consecutive number tokens join with '.'; a date suffix
    is dropped. Mirror of ``productModelLabel`` in the web task-presentation
    module; both are checked against tests/fixtures/model_display_names.json.
    """
    name = re.sub(r'-\d{8}$', '', str(model or '').strip().lower())
    if not name or name == 'none':
        return ''
    if name == 'deepseek':
        return 'DeepSeek Chat'
    name = _DISPLAY_ALIASES.get(name, name)
    words: list[str] = []
    numeric = False
    for token in filter(None, re.split(r'[-_\s]+', name)):
        if words == ['DeepSeek'] and _DEEPSEEK_VERSION.match(token):
            token = token[1:]  # DeepSeek names its versions "DeepSeek 4 Pro", not "V4"
        if numeric and _NUMBER.match(token):
            words[-1] += '.' + token
            continue
        numeric = bool(_NUMBER.match(token))
        words.append(_BRANDS.get(token, token[:1].upper() + token[1:]))
    return re.sub(r'^GPT (?=\d)', 'GPT-', ' '.join(words))


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
    This router only reports that same ordered list (with the lt4net product
    default applied, as the runtime calls it) so a label cannot name a
    different provider or model than the call.
    """

    def route(self, purpose: str = "execution", policy: str = "balanced") -> ModelDecision:
        from research_os.agent.messaging import _resolve_transports

        transports = product_default_transports() or _resolve_transports()
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
