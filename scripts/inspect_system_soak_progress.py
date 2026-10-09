"""Credential-free, read-only progress for an explicitly named isolated soak."""
import argparse
import json
from pathlib import Path
import re
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'C:/EMQA/sys0907-[a-z0-9-]+', args.root):
        raise ValueError('isolated_root_required')
    root = Path(args.root)
    value = json.loads((root / 'endurance.json').read_text(encoding='utf-8'))
    result = {key: value.get(key) for key in ['model', 'status', 'started_at', 'completed_at',
        'elapsed_seconds', 'qualified', 'service_soak_passed', 'runtime_closed_cleanly',
        'native_tools_executed', 'real_transport_failures']}
    database = root / 'isolated-workspace/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=3) as connection:
        connection.execute('PRAGMA query_only=ON')
        attempts = connection.execute("SELECT payload_json FROM events WHERE event_type='model.transport_attempt' ORDER BY rowid DESC LIMIT 8").fetchall()
        safe = ['status', 'provider', 'model', 'attempt', 'elapsed_seconds', 'input_tokens',
                'output_tokens', 'error_code', 'http_status', 'retryable']
        result['recent_attempts'] = [{key: payload.get(key) for key in safe if key in payload}
            for (encoded,) in attempts for payload in [json.loads(encoded)]]
        result['tool_states'] = connection.execute('SELECT status,COUNT(*) FROM tool_calls GROUP BY status').fetchall()
        result['successful_case_tools'] = []
        for case in value.get('cases', []):
            rows = connection.execute('SELECT tool_name,status,result_json FROM tool_calls WHERE session_id=?', (case['run_id'],)).fetchall()
            completed = [(name, json.loads(encoded)) for name, status, encoded in rows if status == 'completed']
            result['successful_case_tools'].append({'run_id': case['run_id'],
                'successful_tools': sorted({name for name, payload in completed if payload.get('ok')}),
                'hash_receipt_contains_expected_sha256': any(case.get('output_sha256') in json.dumps(payload.get('content', {}))
                    for name, payload in completed if name == 'directory_hash' and payload.get('ok') and case.get('output_sha256'))})
        result['recent_tools'] = []
        for name, status, encoded in connection.execute('SELECT tool_name,status,result_json FROM tool_calls ORDER BY rowid DESC LIMIT 8'):
            payload = json.loads(encoded)
            error = payload.get('error') or {}
            code = error.get('code') if isinstance(error, dict) else error
            error_text = str(error)
            result['recent_tools'].append({'name': name, 'status': status,
                'error_code': code if isinstance(code, str) and re.fullmatch(r'[a-z_]{1,80}', code) else None,
                'error_categories': [label for label, needle in [('directory_binding', 'directory_id'),
                    ('missing_argument', 'required'), ('path_not_found', 'not found'),
                    ('path_not_found', 'FileNotFoundError'), ('permission', 'PermissionError'),
                    ('unknown_directory', 'unknown directory')] if needle in error_text]})
        result['hash_directory_binding'] = []
        for encoded, metadata in connection.execute("SELECT t.arguments_json,s.metadata_json FROM tool_calls t JOIN sessions s ON s.id=t.session_id WHERE t.tool_name='directory_hash'"):
            arguments, session = json.loads(encoded), json.loads(metadata)
            expected = session.get('super_agent_directory_ids') or []
            result['hash_directory_binding'].append({
                'matches_session_directory': arguments.get('directory_id') in expected,
                'expected_directory_count': len(expected),
                'target_is_summary_file': arguments.get('relative_path') == 'outputs/summary.json'})
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
