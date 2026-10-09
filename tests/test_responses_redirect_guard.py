"""Exercise the actual urllib handler chain without opening any sockets."""
import io
import json
import socket
import urllib.error
import urllib.request
import urllib.response

import pytest

from evomind_runtime import model_transport, responses_transport as module
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig


SOURCE = 'https://fixture.invalid/v1/responses'
FIXTURE_CREDENTIAL = 'fixture-bearer-not-a-real-key'


def offline_response(monkeypatch, status, *, location=None, content=b''):
    """Replace only the network sink, retaining the production redirect chain."""
    observed = []

    def reject_socket(*args, **kwargs):
        raise AssertionError('network access is forbidden in redirect regression tests')

    def do_open(_handler, _http_class, request, **kwargs):
        observed.append({'url': request.full_url, 'method': request.get_method(),
                         'authorization_present': bool(request.get_header('Authorization'))})
        headers = {'Content-Type': 'text/event-stream', 'Retry-After': '17'}
        if location is not None:
            headers['Location'] = location
        response = urllib.response.addinfourl(io.BytesIO(content), headers, request.full_url, status)
        response.msg = 'offline fixture'
        return response

    monkeypatch.setattr(socket, 'create_connection', reject_socket)
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, 'do_open', do_open)
    return observed


@pytest.mark.parametrize('status', [301, 302, 303, 307, 308])
@pytest.mark.parametrize('location', [
    'https://other-fixture.invalid/v1/responses',
    'http://other-fixture.invalid/v1/responses',
    '/same-origin-but-unapproved-route',
])
def test_redirect_is_an_original_http_error_without_a_second_request(monkeypatch, status, location):
    observed = offline_response(monkeypatch, status, location=location)
    with pytest.raises(urllib.error.HTTPError) as caught:
        module.post_responses(SOURCE, {'Authorization': 'Bearer ' + FIXTURE_CREDENTIAL}, {}, 10)
    assert caught.value.code == status
    assert caught.value.url == SOURCE
    assert caught.value.headers['Location'] == location
    assert caught.value.headers['Retry-After'] == '17'
    assert observed == [{'url': SOURCE, 'method': 'POST', 'authorization_present': True}]
    caught.value.close()


@pytest.mark.parametrize('model', ['gpt-6-astra', 'gpt-5.6-sol'])
def test_governed_redirect_fails_once_and_keeps_credentials_out_of_errors(monkeypatch, model):
    monkeypatch.setenv('EVOMIND_MODEL_WIRE_PROTOCOL', 'responses')
    observed = offline_response(monkeypatch, 302, location='https://other-fixture.invalid/redirect')
    events = []
    raw = messaging.AgentMessageClient(max_retries=2, timeout=10, transports=[
        messaging.OpenAITransport(ProviderConfig('openai', SOURCE, model, FIXTURE_CREDENTIAL))])
    client = model_transport.governed_client(raw, events.append)
    with pytest.raises(model_transport.ModelTransportError) as caught:
        client.send([{'role': 'user', 'content': 'offline fixture'}], system='fixture', tools=[])
    assert caught.value.code == 'http_302'
    assert caught.value.retryable is False
    assert client.last_attempt_count == 1 and len(observed) == 1
    assert FIXTURE_CREDENTIAL not in str(caught.value) + json.dumps(events)
    assert events[0]['http_status'] == 302


def test_successful_sse_still_uses_the_frozen_endpoint(monkeypatch):
    response = {'id': 'response_fixture', 'model': 'gpt-5.6-sol', 'status': 'completed', 'output': [
        {'type': 'message', 'role': 'assistant', 'status': 'completed',
         'content': [{'type': 'output_text', 'text': 'fixture ready'}]}]}
    encoded = ('data: ' + json.dumps({'type': 'response.completed', 'response': response}) + '\n\n').encode()
    observed = offline_response(monkeypatch, 200, content=encoded)
    assert module.post_responses(SOURCE, {'Authorization': 'Bearer ' + FIXTURE_CREDENTIAL}, {}, 10) == response
    assert observed == [{'url': SOURCE, 'method': 'POST', 'authorization_present': True}]


def test_private_opener_replaces_the_default_redirect_handler():
    handlers = module._responses_opener.handlers
    assert any(isinstance(handler, module._NoRedirectHandler) for handler in handlers)
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers)
