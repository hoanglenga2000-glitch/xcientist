"""Strict response validation around the configured model client's transport.

Provider, model, endpoint and credentials remain owned by the original client.
Only model requests are retried here; no tool is executed by this module.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import queue
import subprocess
import sys
import time
import threading
import urllib.error
from urllib.parse import urlsplit
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research_os.agent import messaging
from research_os.llm_client import LLMError


RETRY_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRY_ENVELOPES = frozenset({"upstream_error", "server_error", "rate_limit_error", "timeout_error", "overloaded_error"})
GOVERNED_MODELS = frozenset({"gpt-6-astra", "gpt-5.6-sol", "gpt-5.5", "deepseek-v4-pro", "deepseek-flash"})
ACCEPTANCE_EXTRA_MODELS = frozenset({'gpt-5.6-terra', 'gpt-5.6-luna'})


def governed_model_names():
    """Extra candidates are isolated-process opt-in, never a default switch."""
    return GOVERNED_MODELS | ACCEPTANCE_EXTRA_MODELS if os.environ.get('EVOMIND_MODEL_ACCEPTANCE_EXTRAS') == '1' else GOVERNED_MODELS


def bind_run_client(store, session_id, client):
    """A pinned Run cannot bypass validation by resolving an ungoverned client."""
    from .siim_request_limits import configure_client
    configure_client(store, session_id, client)
    contract = getattr(client, 'contract', None)
    if contract:
        store.bind_model_contract(session_id, contract)
    else:
        try:
            store.admit_uncontracted_model_client(session_id)
        except ValueError as error:
            if str(error) == 'model_contract_missing_on_resume':
                raise ModelTransportError('model_contract_missing_on_resume', retryable=False) from None
            raise


class ModelTransportError(LLMError):
    def __init__(self, code: str, *, retryable: bool, status: int | None = None,
                 retry_after_seconds: float | None = None) -> None:
        self.code, self.retryable, self.http_status = code, retryable, status
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"LLMError/{code}")


def post_json_with_deadline(url, headers, payload, timeout):
    """Enforce wall time even if a gateway trickles bytes forever.

    Credentials travel only over private process stdin. Timeout terminates and
    reaps this exact HTTP worker, never the runtime or a training process.
    """
    deadline = time.monotonic() + timeout
    request = json.dumps({'url': url, 'headers': headers, 'payload': payload, 'timeout': timeout}).encode('utf-8')
    if len(request) > 16 * 1024 * 1024:
        raise ModelTransportError('model_request_too_large', retryable=False)
    process = subprocess.Popen([sys.executable, '-I', '-u', str(Path(__file__).with_name('model_deadline_worker.py'))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        out, _err = process.communicate(request, timeout=max(0.001, deadline-time.monotonic()))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        raise ModelTransportError('model_total_deadline_exceeded', retryable=False) from None
    if time.monotonic() > deadline:
        raise ModelTransportError('model_total_deadline_exceeded', retryable=False)
    if process.returncode or len(out) > 32 * 1024 * 1024:
        raise ModelTransportError('model_http_worker_failed', retryable=False)
    result = json.loads(out)
    if result.get('kind') == 'response':
        return result['body']
    if result.get('kind') == 'http_error':
        status = result['status']
        raise ModelTransportError(f'http_{status}', retryable=status in RETRY_HTTP, status=status,
            retry_after_seconds=retry_after_seconds({'Retry-After': result.get('retry_after')}))
    raise ModelTransportError('model_network_error' if result.get('kind')=='network_error' else 'model_response_protocol_error', retryable=True)


def retry_after_seconds(headers):
    """Respect a provider cooldown without logging its headers or error body."""
    value = headers.get('Retry-After') if headers else None
    if value is None:
        return None
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return delay if 0 <= delay < float('inf') else None


def post_stream_with_deadline(url, headers, payload, timeout, on_text, check_cancel=None):
    """Read real deltas over private worker pipes under the same total deadline."""
    deadline = time.monotonic() + timeout
    request = json.dumps({'url': url, 'headers': headers, 'payload': payload, 'timeout': timeout, 'stream': True}).encode()
    if len(request) > 16 * 1024 * 1024:
        raise ModelTransportError('model_request_too_large', retryable=False)
    process = subprocess.Popen([sys.executable, '-I', '-u', str(Path(__file__).with_name('model_deadline_worker.py'))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    events = queue.Queue()
    def read_worker():
        try:
            total = 0
            for _ in range(32768):
                line = process.stdout.readline(1_048_577)
                if not line:
                    break
                total += len(line)
                if len(line) > 1_048_576 or total > 32 * 1024 * 1024:
                    break
                events.put(json.loads(line))
        except (ValueError, OSError):
            pass
        finally:
            events.put({'kind': 'eof'})
    reader = threading.Thread(target=read_worker, daemon=True)
    reader.start()
    try:
        process.stdin.write(request)
        process.stdin.close()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelTransportError('model_total_deadline_exceeded', retryable=False)
            try:
                event = events.get(timeout=min(0.25, remaining))
            except queue.Empty:
                if check_cancel:
                    check_cancel()
                continue
            if not isinstance(event, dict):
                raise ModelTransportError('model_response_protocol_error', retryable=True)
            if event.get('kind') == 'text':
                if not isinstance(event.get('text'), str):
                    raise ModelTransportError('model_response_protocol_error', retryable=True)
                on_text(event['text'])
            elif event.get('kind') == 'response':
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    raise ModelTransportError('model_total_deadline_exceeded', retryable=False) from None
                if time.monotonic() > deadline:
                    raise ModelTransportError('model_total_deadline_exceeded', retryable=False)
                if process.returncode:
                    raise ModelTransportError('model_http_worker_failed', retryable=False)
                return event['body']
            elif event.get('kind') == 'http_error':
                status = int(event['status'])
                raise ModelTransportError(f'http_{status}', retryable=status in RETRY_HTTP, status=status,
                    retry_after_seconds=retry_after_seconds({'Retry-After': event.get('retry_after')}))
            elif event.get('code') == 'model_identity_unconfirmed':
                raise ModelTransportError('model_identity_unconfirmed', retryable=False)
            else:
                raise ModelTransportError('model_network_error' if event.get('kind') == 'network_error' else 'model_response_protocol_error', retryable=True)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        reader.join(timeout=1)
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()


def _response_error(body: Any, *, expected_model: str, tools: list[Any]) -> None:
    if not isinstance(body, dict):
        raise ModelTransportError("response_not_object", retryable=True)
    if "error" in body:
        error = body["error"]
        kind = str(error.get("type") or "") if isinstance(error, dict) else ""
        code = kind if kind in RETRY_ENVELOPES else "provider_error"
        raise ModelTransportError(code, retryable=kind in RETRY_ENVELOPES)
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ModelTransportError("response_choices_invalid", retryable=True)
    if body.get("model") != expected_model:
        raise ModelTransportError("model_identity_unconfirmed", retryable=False)
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelTransportError("response_message_missing", retryable=True)
    if message.get("refusal") or choice.get("finish_reason") in {"content_filter", "refusal"}:
        raise ModelTransportError("model_refusal", retryable=False)
    # A length-limited reasoning-only reply is truncated, not a transient empty
    # reply. Repeating the identical token-limited request cannot repair it.
    if choice.get("finish_reason") == "length":
        raise ModelTransportError("model_output_truncated", retryable=False)
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ModelTransportError("tool_calls_invalid", retryable=True)
    allowed = {item.name for item in tools}
    seen = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type", "function") != "function":
            raise ModelTransportError("tool_call_invalid", retryable=True)
        identifier, function = call.get("id"), call.get("function")
        if not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(function, dict):
            raise ModelTransportError("tool_call_identity_invalid", retryable=True)
        seen.add(identifier)
        if function.get("name") not in allowed:
            raise ModelTransportError("unoffered_tool_call", retryable=False)
        try:
            value = function.get("arguments")
            value = json.loads(value) if isinstance(value, str) else value
            if not isinstance(value, dict):
                raise ValueError("arguments_not_object")
        except (TypeError, ValueError):
            raise ModelTransportError("tool_arguments_invalid", retryable=True) from None
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ModelTransportError("response_content_invalid", retryable=True)
    if not calls and not str(content or "").strip():
        raise ModelTransportError("empty_model_response", retryable=True)


def response_metadata(body: Any, *, expected_model: str) -> dict:
    """Only types, counts, a fixed finish enum and reported numeric usage.

    Never persist response text, reasoning content, tool arguments, IDs,
    headers or arbitrary provider error strings in diagnostic events.
    """
    if not isinstance(body, dict):
        return {"response_object": False, "reported_usage_valid": False}
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices)==1 and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    finish = choice.get("finish_reason")
    allowed_finish = {"stop", "length", "tool_calls", "function_call", "content_filter", "refusal"}
    content, calls = message.get("content"), message.get("tool_calls")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    def count(value):
        return value if type(value) is int and 0 <= value <= 1_000_000_000 else None
    prompt, completion = count(usage.get("prompt_tokens")), count(usage.get("completion_tokens"))
    return {"response_object": True, "choice_count": len(choices) if isinstance(choices, list) else None,
            "model_identity_matches": body.get("model")==expected_model,
            "finish_reason": finish if isinstance(finish, str) and finish in allowed_finish else "other_or_missing",
            "content_characters": len(content) if isinstance(content, str) else None,
            "tool_call_count": len(calls) if isinstance(calls, list) else 0 if calls is None else None,
            "reasoning_field_present": "reasoning_content" in message or "reasoning" in message,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "reasoning_tokens": count(details.get("reasoning_tokens")),
            "reported_usage_valid": prompt is not None and completion is not None}


class _GovernedClient:
    manages_request_retries = True

    def __init__(self, client: Any, observer: Callable[[dict], None] | None,
                 before_attempt: Callable[[], None] | None = None,
                 text_observer: Callable[[dict], None] | None = None,
                 during_request: Callable[[], None] | None = None) -> None:
        self.client, self.observer = client, observer
        self.before_attempt = before_attempt
        self.text_observer = text_observer
        self.during_request = during_request
        # Freeze the selected transport. A retry may not select another model,
        # provider or endpoint, including after an environment/config change.
        self.transport = copy.deepcopy(client._resolve_transports()[0])
        protocol = os.environ.get('EVOMIND_MODEL_WIRE_PROTOCOL', 'chat_completions')
        if protocol not in {'chat_completions', 'responses'}:
            raise ModelTransportError('model_wire_protocol_invalid', retryable=False)
        if protocol == 'responses':
            from .responses_transport import OpenAIResponsesTransport
            self.transport = OpenAIResponsesTransport(self.transport.config)
        self.wire_protocol = getattr(self.transport, 'wire_protocol', 'chat_completions_v1')
        timeout_setting = os.environ.get('EVOMIND_MODEL_TIMEOUT_SECONDS')
        if timeout_setting is not None:
            try:
                selected_timeout = int(timeout_setting)
            except (TypeError, ValueError):
                raise ModelTransportError('model_timeout_configuration_invalid', retryable=False) from None
            if not 1 <= selected_timeout <= 3600:
                raise ModelTransportError('model_timeout_configuration_invalid', retryable=False)
            self.timeout = selected_timeout
        else:
            self.timeout = client.timeout
        self.max_retries = max(0, min(2, client.max_retries))
        address = urlsplit(self.transport.config.base_url)
        # This is EvoMind's verified 15-second, single-credential gateway.
        # Retrying at 2/4 seconds only hits the same cooldown, not the model.
        self.retry_floor_seconds = 16 if (self.wire_protocol == 'responses_stream_v1'
            and address.hostname in {'127.0.0.1', 'localhost', '::1'} and address.port == 65068) else 0
        config = self.transport.config
        route_path = os.environ.get('EVOMIND_MODEL_ROUTE_CONFIG_PATH')
        try:
            self.route_config_path = Path(route_path).resolve() if route_path else None
            route_sha = hashlib.sha256(self.route_config_path.read_bytes()).hexdigest() if self.route_config_path else None
        except (OSError, ValueError):
            raise ModelTransportError('model_route_configuration_unavailable', retryable=False) from None
        self.contract = {
            "schema": "evomind.model_execution_contract.v4",
            "provider": self.transport.name, "model": config.model,
            "endpoint_sha256": hashlib.sha256(config.base_url.encode()).hexdigest(),
            "reasoning_effort": config.reasoning_effort,
            "service_tier": config.service_tier,
            'credential_binding_sha256': hmac.new(config.api_key.encode(), b'evomind:model-credential-binding:v1', hashlib.sha256).hexdigest(),
            'route_config_sha256': route_sha,
            'wire_protocol': self.wire_protocol, 'timeout_seconds': self.timeout,
            'max_request_retries': self.max_retries, 'retry_floor_seconds': self.retry_floor_seconds,
        }
        self.last_attempt_count = 0

    def _check_route_config(self):
        if self.route_config_path is None:
            return
        try:
            actual = hashlib.sha256(self.route_config_path.read_bytes()).hexdigest()
        except OSError:
            raise ModelTransportError('model_route_configuration_unavailable', retryable=False) from None
        if actual != self.contract['route_config_sha256']:
            raise ModelTransportError('model_route_configuration_drift', retryable=False)

    @property
    def model(self) -> str:
        return self.transport.config.model

    def is_available(self) -> bool:
        return self.client.is_available()

    def send(self, messages, *, system, tools, max_tokens=4096, temperature=0.3):
        confirmed = getattr(self, 'confirmed_request_limits', None)
        deadline = time.monotonic() + confirmed['total_request_seconds'] if confirmed else None
        if confirmed:
            max_tokens = confirmed['max_output_tokens']
        self.last_attempt_count = 0
        last = ModelTransportError("no_model_transport", retryable=False)
        for transport in (self.transport,):
            url, headers, payload = transport.build(messages, system, tools, max_tokens, temperature)
            if self.wire_protocol == 'chat_completions_v1' and transport.name == "openai" and transport.config.model in governed_model_names():
                if tools:
                    for spec in payload["tools"]:
                        # Preserve optional tool parameters across compatible gateways.
                        # The runtime still validates all arguments and approvals.
                        spec["function"]["strict"] = False
                else:
                    payload.pop("tools", None)
            for attempt in range(self.max_retries + 1):
                self._check_route_config()
                if self.before_attempt:
                    self.before_attempt()
                if deadline is not None and getattr(self, 'confirmed_time_budget', None):
                    deadline = min(deadline, time.monotonic() + self.confirmed_time_budget())
                self._check_route_config()
                self.last_attempt_count += 1
                started = time.monotonic()
                checked_body = None
                try:
                    if self.wire_protocol == 'responses_stream_v1':
                        from .responses_transport import post_responses, as_chat_envelope
                        body = post_responses(url, headers, payload, self.timeout)
                        checked_body = as_chat_envelope(body)
                    elif self.text_observer is not None and self.wire_protocol == 'chat_completions_v1':
                        remaining = min(self.timeout, deadline - time.monotonic()) if deadline is not None else self.timeout
                        if remaining <= 0:
                            raise ModelTransportError('model_total_deadline_exceeded', retryable=False)
                        self.text_observer({'phase': 'start'})
                        body = post_stream_with_deadline(url, headers, payload, remaining,
                            lambda text: self.text_observer({'phase': 'delta', 'text': text}), self.during_request)
                        checked_body = body
                    else:
                        if deadline is not None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise ModelTransportError('model_total_deadline_exceeded', retryable=False)
                            body = post_json_with_deadline(url, headers, payload, min(self.timeout, remaining))
                        else:
                            body = messaging._post_json(url, headers, payload, self.timeout)
                        checked_body = body
                    _response_error(checked_body, expected_model=transport.config.model, tools=tools)
                    self._check_route_config()
                    turn = transport.parse(body)
                    if self.client._is_empty(turn):
                        raise ModelTransportError("empty_model_response", retryable=True)
                except ModelTransportError as error:
                    last = error
                except urllib.error.HTTPError as error:
                    status = int(error.code)
                    last = ModelTransportError(f"http_{status}", retryable=status in RETRY_HTTP, status=status,
                                               retry_after_seconds=retry_after_seconds(error.headers))
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    last = ModelTransportError("model_network_error", retryable=True)
                except (ValueError, TypeError):
                    last = ModelTransportError("model_response_protocol_error", retryable=True)
                else:
                    if self.text_observer is not None:
                        self.text_observer({'phase': 'complete'})
                    if self.observer:
                        self.observer({"status": "completed", "provider": turn.provider, "model": turn.model,
                                       "attempt": self.last_attempt_count, "elapsed_seconds": time.monotonic() - started,
                                       "input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens,
                                       "request_max_output_tokens": max_tokens,
                                       **({'request_limits': confirmed} if confirmed else {})})
                    return turn
                if self.text_observer is not None:
                    self.text_observer({'phase': 'reset'})
                if self.observer:
                    metadata = response_metadata(checked_body, expected_model=transport.config.model)
                    self.observer({"status": "failed", "provider": transport.name, "model": transport.config.model,
                                   "attempt": self.last_attempt_count, "elapsed_seconds": time.monotonic() - started,
                                   "error_code": last.code, "http_status": last.http_status,
                                   "retryable": last.retryable, "token_usage_known": metadata["reported_usage_valid"],
                                   "input_tokens": metadata.get("prompt_tokens"), "output_tokens": metadata.get("completion_tokens"),
                                   "request_max_output_tokens": max_tokens,
                                   "response_metadata": metadata, **({'request_limits': confirmed} if confirmed else {})})
                if not last.retryable:
                    raise last from None
                if attempt < self.max_retries:
                    delay = max(min(8, 2 ** (attempt + 1)), self.retry_floor_seconds, last.retry_after_seconds or 0)
                    if delay > 30:
                        # Do not ignore a longer provider hint or exceed our
                        # automatic retry budget. Leave recovery to a later Run action.
                        raise last from None
                    if deadline is not None and time.monotonic() + delay >= deadline:
                        raise ModelTransportError('model_total_deadline_exceeded', retryable=False) from None
                    time.sleep(delay)
        raise last from None


def governed_client(client: Any, observer: Callable[[dict], None] | None = None,
                    *, before_attempt: Callable[[], None] | None = None,
                    text_observer: Callable[[dict], None] | None = None,
                    during_request: Callable[[], None] | None = None):
    """Govern the selected Astra/Sol transport; preserve unrelated integrations."""
    if not hasattr(client, "_resolve_transports"):
        return client
    try:
        transports = client._resolve_transports()
    except LLMError:
        return client
    if not transports or transports[0].name != "openai" or transports[0].config.model not in governed_model_names():
        return client
    return _GovernedClient(client, observer, before_attempt, text_observer, during_request)
