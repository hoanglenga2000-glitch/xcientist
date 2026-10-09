"""A single pinned user transport, with no process-global provider changes/fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from research_os.agent.messaging import AnthropicTransport, DeepSeekTransport, OpenAITransport
from research_os.llm_client import ProviderConfig
from .model_transport import _response_error
from .models import utc_now


def client_for_run(runtime, session, before_attempt=None):
    from .model_profiles import ModelProfiles
    from .tenant_access import AccessStore
    profile_store = ModelProfiles(runtime.runtime_root)
    owner = AccessStore(runtime.runtime_root).owner_for('session', session['id'])
    frozen = profile_store.run_binding(owner, session['id'])
    declared = (session.get('metadata') or {}).get('model_profile') or {}
    if any(frozen[key] != declared.get(key) for key in ('id', 'version')):
        raise ValueError('model_run_binding_mismatch')
    return PersonalModelClient(profile_store, owner, frozen, before_attempt=before_attempt, record_use=True)


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]')
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {redact(name, key): redact(item, key) for name, item in value.items()}
    return value


def post_public_json(url, headers, payload):
    """External-service boundary. Fixtures may replace this, never the task/store."""
    if os.environ.get('EVOMIND_TEST_FIXTURE_ROOT'):
        raise ValueError('model_isolated_network_disabled')
    request = json.dumps({'url': url, 'headers': headers, 'payload': payload}).encode()
    if len(request) > 3 * 1024 * 1024:
        raise ValueError('model_request_too_large')
    try:
        result = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('personal_model_http.py'))],
            input=request, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except subprocess.TimeoutExpired:
        raise ValueError('model_total_timeout') from None
    if result.returncode or len(result.stdout) > 2 * 1024 * 1024:
        raise ValueError('model_transport_failed')
    result = json.loads(result.stdout)
    if 'error' in result:
        raise ValueError(result['error'])
    return result['body']


class PersonalModelClient:
    def __init__(self, store, principal, binding, before_attempt=None, record_use=False):
        self.store, self.principal, self.binding = store, principal, binding
        self.before_attempt, self.record_use = before_attempt, record_use
        profile = store.get(principal, binding['id'], binding['version'])
        if not profile['enabled']:
            raise ValueError('model_profile_disabled')
        key = store.secrets.load(principal.tenant_id, principal.owner_id, binding['id'], binding['version'])
        config = ProviderConfig(profile['provider'], profile['base_url'], profile['model'], key)
        transport = {'anthropic': AnthropicTransport, 'deepseek': DeepSeekTransport, 'openai': OpenAITransport}[profile['provider']]
        self.transport = transport(config)
        self.timeout, self.max_retries = 30, 0

    @property
    def model(self):
        return self.transport.config.model

    def is_available(self):
        return True

    def _resolve_transports(self):
        return [self.transport]

    def send(self, messages, *, system, tools, max_tokens=4096, temperature=0.3):
        if self.before_attempt:
            self.before_attempt()
        # Disabling blocks the next request; edits keep this run's old revision.
        if not self.store.get(self.principal, self.binding['id'])['enabled']:
            raise ValueError('model_profile_disabled')
        url, headers, payload = self.transport.build(messages, system, tools, min(max_tokens, 4096), temperature)
        body = post_public_json(url, headers, payload)
        # A malicious provider must not reflect the credential into stored output.
        body = redact(body, self.transport.config.api_key)
        if self.transport.name != 'anthropic':
            _response_error(body, expected_model=self.model, tools=tools)
        elif (not isinstance(body, dict) or body.get('model') != self.model or 'error' in body
              or not isinstance(body.get('content'), list) or body.get('stop_reason') not in {'end_turn', 'tool_use'}):
            raise ValueError('model_response_invalid')
        turn = self.transport.parse(body)
        allowed = {tool.name for tool in tools}
        seen = set()
        for call in turn.tool_calls:
            if not call.id or call.id in seen or call.name not in allowed or not isinstance(call.input, dict):
                raise ValueError('model_tool_response_invalid')
            seen.add(call.id)
        if not turn.text and not turn.tool_calls:
            raise ValueError('model_response_empty')
        if self.record_use:
            with self.store.connect() as connection:
                connection.execute('UPDATE revisions SET used_at=? WHERE id=? AND version=?', (utc_now(), self.binding['id'], self.binding['version']))
        return turn
