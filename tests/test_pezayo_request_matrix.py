import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('channel_matrix', Path(__file__).parents[1] / 'scripts/diagnose_pezayo_request_matrix.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_simple_tool_count_ablation_preserves_prompt_and_first_schema():
    one = module.build_body('gpt-5.6-sol', 'simple1', 'responses', True, 'omit')
    seven = module.build_body('gpt-5.6-sol', 'simple7', 'responses', True, 'omit')
    assert one['input'] == seven['input']
    assert one['tools'][0] == seven['tools'][0]
    assert len(seven['tools']) == 7
    assert len({t['name'] for t in seven['tools']}) == 7


def test_error_receipts_only_contain_allowlisted_categories():
    result = module.summarize({'error': {'type': 'api_error', 'message': 'DO_NOT_LOG upstream unavailable',
        'code': 'sk-DO_NOT_LOG'}}, 'responses', 'gpt-5.6-sol', 'simple1')
    assert result['error_categories'] == ['capacity']
    assert result['error_code'] is None
    assert 'DO_NOT_LOG' not in json.dumps(result)


def test_json_error_mislabeled_as_sse():
    class Response:
        def iter_lines(self):
            yield b'{"error":{"type":"api_error","message":"Service temporarily unavailable"}}'
    value, timing = module.read_stream(Response(), 'responses', module.time.monotonic())
    assert value['error']['type'] == 'api_error'
    assert timing['first_data_seconds'] is None


def test_nonstream_heartbeats_have_a_total_deadline(monkeypatch):
    class Response:
        def iter_content(self, **_):
            yield b' '
            yield b' '
    clock = iter([1, 101])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    with pytest.raises(TimeoutError, match='deadline'):
        module.read_json(Response(), 0)


def test_inventory_never_substitutes_for_tool_validation():
    result = module.summarize({'data': [{'id': 'gpt-5.6-sol'}]}, 'responses', 'gpt-5.6-sol', 'simple1')
    assert result['valid'] is False
