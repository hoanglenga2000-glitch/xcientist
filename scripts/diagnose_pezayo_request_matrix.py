"""Bounded, no-tool-execution channel ablations with credential-free receipts.

Run on the EvoMind server. Never follows redirects or retries. An optional key
arrives over SSH stdin, is compared in memory, and never modifies configuration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

import requests
import yaml


NAMES = ['file_read', 'file_write', 'directory_hash', 'artifact_publish',
         'artifact_list', 'report_generate', 'report_status']
CANDIDATE = 'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v2'


def functions(case):
    basic = {'type': 'function', 'name': 'contract_probe',
             'description': 'Return the provided integer in a synthetic transport fixture.',
             'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}},
                            'required': ['value'], 'additionalProperties': False}, 'strict': False}
    if case.startswith('simple'):
        result = [copy.deepcopy(basic) for _ in range(7 if case == 'simple7' else 1)]
        for i, item in enumerate(result):
            if i:
                item['name'] += '_' + str(i)
        return result, 'Return one contract_probe call with value 37. No tool will run.'
    sys.path.insert(0, CANDIDATE)
    from evomind_runtime.tools import build_default_registry
    registry = build_default_registry()
    names = NAMES[:1] if case == 'actual1' else NAMES
    result = [{'type': 'function', 'name': name, 'description': registry.get(name).description,
               'parameters': registry.get(name).input_schema, 'strict': False} for name in names]
    return result, 'Return one file_read call with path inputs/data.json. No tool will run.'


def build_body(model, case, endpoint, stream, tier):
    specs, prompt = functions(case)
    body = {'model': model, 'stream': stream}
    if tier != 'omit':
        body['service_tier'] = tier
    if endpoint == 'responses':
        body.update(input=[{'role': 'user', 'content': prompt}], tools=specs,
                    reasoning={'effort': 'low'}, max_output_tokens=1024, store=False)
    else:
        body.update(messages=[{'role': 'user', 'content': prompt}],
                    tools=[{'type': 'function', 'function': {k: v for k, v in s.items() if k != 'type'}} for s in specs],
                    reasoning_effort='low', max_tokens=1024)
    return body


def identifier(value):
    """Only permit non-secret protocol identifiers, not provider prose."""
    value = str(value or '')
    if re.fullmatch(r'[a-zA-Z0-9_.:-]{1,100}', value) and not value.lower().startswith(('sk-', 'bearer')):
        return value
    return None


def summarize(body, endpoint, model, case):
    if not isinstance(body, dict):
        return {'valid': False, 'error_class': 'body_not_object'}
    if body.get('error'):
        err = body['error']
        message = str(err.get('message', '')) if isinstance(err, dict) else ''
        categories = [label for label, pattern in [
            ('capacity', r'unavailable|overload|capacity|no available|temporar'),
            ('quota', r'quota|balance|credit|余额|额度'),
            ('schema', r'schema|parameter|argument|参数'),
            ('unsupported', r'unsupported|not supported|不支持'),
            ('auth', r'auth|invalid.*key|permission|权限'),
            ('timeout', r'timeout|timed out|超时')]
            if re.search(pattern, message, re.I)]
        return {'valid': False, 'error_class': 'error_envelope', 'error_categories': categories,
                'error_type': identifier(err.get('type')) if isinstance(err, dict) else None,
                'error_code': identifier(err.get('code')) if isinstance(err, dict) else None}
    calls, final = [], None
    if endpoint == 'responses':
        final = body.get('status')
        calls = [x for x in body.get('output', []) if x.get('type') == 'function_call']
    else:
        choices = body.get('choices') or []
        if len(choices) == 1:
            final = choices[0].get('finish_reason')
            calls = [dict(x.get('function') or {}, call_id=x.get('id'))
                     for x in (choices[0].get('message') or {}).get('tool_calls', [])]
    expected = 'contract_probe' if case.startswith('simple') else 'file_read'
    expected_args = {'value': 37} if case.startswith('simple') else {'path': 'inputs/data.json'}
    args_ok = False
    if len(calls) == 1:
        try:
            args_ok = json.loads(calls[0].get('arguments', '')) == expected_args
        except (TypeError, ValueError):
            pass
    usage = body.get('usage') or {}
    return {'valid': body.get('model') == model and final in ('completed', 'tool_calls', 'stop')
                    and len(calls) == 1 and calls[0].get('name') == expected
                    and bool(calls[0].get('call_id')) and args_ok,
            'response_model': identifier(body.get('model')), 'terminal_status': identifier(final),
            'tool_call_count': len(calls), 'arguments_match': args_ok,
            'input_tokens': usage.get('input_tokens', usage.get('prompt_tokens')),
            'output_tokens': usage.get('output_tokens', usage.get('completion_tokens'))}


def read_stream(response, endpoint, started):
    first, events, terminal = None, [], None
    chat = {'choices': [{'message': {'tool_calls': []}, 'finish_reason': None}]}
    calls = {}
    total = 0
    non_sse = []
    for raw in response.iter_lines():
        if time.monotonic() - started > 100:
            raise TimeoutError('overall_response_deadline')
        total += len(raw)
        if total > 2_000_000:
            raise ValueError('response_size_limit')
        if not raw.startswith(b'data:'):
            if raw and not raw.startswith((b':', b'event:', b'id:')):
                non_sse.append(raw)
            continue
        data = raw[5:].strip()
        if data == b'[DONE]':
            break
        if not data:
            continue
        if first is None:
            first = time.monotonic() - started
        event = json.loads(data)
        if event.get('error'):
            terminal = event
            break
        if endpoint == 'responses':
            kind = identifier(event.get('type'))
            if kind and kind not in events:
                events.append(kind)
            if kind in ('response.completed', 'response.failed', 'response.incomplete'):
                terminal = event.get('response') or event
                break
        else:
            chat['model'] = event.get('model', chat.get('model'))
            if event.get('usage'):
                chat['usage'] = event['usage']
            for choice in event.get('choices') or []:
                if choice.get('finish_reason'):
                    chat['choices'][0]['finish_reason'] = choice['finish_reason']
                for delta in (choice.get('delta') or {}).get('tool_calls') or []:
                    call = calls.setdefault(delta['index'], {'id': None, 'function': {'name': '', 'arguments': ''}})
                    call['id'] = delta.get('id', call['id'])
                    for key in ('name', 'arguments'):
                        call['function'][key] += (delta.get('function') or {}).get(key) or ''
    if first is None and non_sse:
        try:
            terminal = json.loads(b'\n'.join(non_sse))
        except ValueError:
            pass
    if endpoint == 'chat' and terminal is None:
        chat['choices'][0]['message']['tool_calls'] = list(calls.values())
        terminal = chat
    return terminal, {'first_data_seconds': first, 'event_types': events, 'stream_bytes': total}


def read_json(response, started):
    # Compatibility servers may emit whitespace heartbeats indefinitely. An
    # idle timeout alone does not bound a non-streaming response's lifetime.
    chunks = bytearray()
    for chunk in response.iter_content(chunk_size=1):
        if time.monotonic() - started > 100:
            raise TimeoutError('overall_response_deadline')
        chunks.extend(chunk)
        if len(chunks) > 2_000_000:
            raise ValueError('response_size_limit')
    return json.loads(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gpt-6-astra', 'gpt-5.6-sol'])
    parser.add_argument('--case', choices=['inventory', 'simple1', 'simple7', 'actual1', 'actual7'], required=True)
    parser.add_argument('--endpoint', choices=['responses', 'chat'], default='responses')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--route', choices=['upstream', 'gateway'], default='upstream')
    parser.add_argument('--tier', choices=['omit', 'priority', 'default'], default='omit')
    parser.add_argument('--credential-stdin', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    # Never overwrite a previous diagnostic receipt, even after an SSH disconnect.
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as receipt:
        cfg = yaml.safe_load(Path('C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml').read_text(encoding='utf-8-sig'))
        providers = [x for x in cfg.get('openai-compatibility', []) if x.get('name') == 'pezayo']
        if len(providers) != 1 or len(providers[0].get('api-key-entries') or []) != 1:
            raise ValueError('ambiguous_protected_provider')
        provider = providers[0]
        origin = provider['base-url'].rstrip('/')
        if urlsplit(origin).scheme != 'https' or urlsplit(origin).hostname != 'api.pezayo.com':
            raise ValueError('unapproved_origin')
        entry = provider['api-key-entries'][0]
        key = entry['api-key']
        matches = None
        if args.credential_stdin:
            supplied = json.loads(sys.stdin.readline()).get('key')
            if not isinstance(supplied, str) or not re.fullmatch(r'sk-[A-Za-z0-9_-]{20,200}', supplied):
                raise ValueError('invalid_supplied_credential')
            matches = hmac.compare_digest(supplied, key)
            key = supplied
        if args.route == 'gateway':
            if matches is False:
                raise ValueError('gateway_credential_does_not_match_supplied_key')
            origin, key = 'http://127.0.0.1:65068/v1', cfg['api-keys'][0]
        base = origin if origin.endswith('/v1') else origin + '/v1'
        session = requests.Session()
        session.trust_env = False
        proxy = entry.get('proxy-url') or ''
        if args.route == 'upstream' and proxy.lower() not in ('', 'direct', 'direct://'):
            session.proxies = {'http': proxy, 'https': proxy}
        session.headers['Authorization'] = 'Bearer ' + key
        models = [None] if args.case == 'inventory' else args.models
        for index, model in enumerate(models):
            if index:
                time.sleep(16)  # isolate the gateway's 15-second cooldown
            started = time.monotonic()
            row = {'schema': 'evomind.channel_ablation.v1', 'case': args.case, 'route': args.route,
                   'endpoint': args.endpoint, 'stream': args.stream, 'tier': args.tier,
                   'requested_model': model, 'tools_executed': 0, 'retries': 0,
                   'supplied_credential_matches_config': matches, 'started_epoch': time.time()}
            try:
                if args.case == 'inventory':
                    response = session.get(base + '/models', timeout=(10, 45), allow_redirects=False)
                    row['http_status'] = response.status_code
                    value = response.json()
                    row['gpt_models'] = sorted({x['id'] for x in value.get('data', [])
                                                if isinstance(x, dict) and re.fullmatch(r'gpt-[A-Za-z0-9_.-]+', str(x.get('id', '')))})
                    row['inventory_only'] = True
                else:
                    body = build_body(model, args.case, args.endpoint, args.stream, args.tier)
                    row['payload_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                    row['tool_count'] = len(body['tools'])
                    route = '/responses' if args.endpoint == 'responses' else '/chat/completions'
                    with session.post(base + route, json=body, timeout=(10, 75), stream=True,
                                      allow_redirects=False) as response:
                        row['http_status'] = response.status_code
                        row['content_type'] = response.headers.get('Content-Type', '').split(';')[0]
                        if 'text/event-stream' in row['content_type']:
                            value, timing = read_stream(response, args.endpoint, started)
                            row.update(timing)
                        else:
                            value = read_json(response, started)
                        row.update(summarize(value, args.endpoint, model, args.case))
                        row['valid'] = bool(row['valid'] and 200 <= response.status_code < 300)
            except Exception as error:
                row.update(valid=False, error_class=type(error).__name__)
            row['elapsed_seconds'] = time.monotonic() - started
            encoded = json.dumps(row, ensure_ascii=True)
            receipt.write(encoded + '\n')
            receipt.flush()
            print(encoded, flush=True)


if __name__ == '__main__':
    main()
