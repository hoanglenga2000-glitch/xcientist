"""Authenticated local-candidate contracts. Only the named disposable preview."""
from __future__ import annotations
import hashlib
from http.cookiejar import CookieJar
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/product-launch-20261008'
ORIGIN = 'http://127.0.0.1:8102'


class User:
    def __init__(self, name):
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        self.csrf = ''
        self.username = name
        self.call('POST', '/api/auth/login', {'username': name, 'password': name + '-product-fixture-only'})
        self.csrf = self.call('GET', '/api/session/status')['csrf_token']

    def call(self, method, path, body=None, *, expected=200, binary=False, headers=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        request_headers = {'Origin': ORIGIN, 'Content-Type': 'application/octet-stream' if isinstance(body, bytes) else 'application/json', 'x-evomind-csrf': self.csrf, **(headers or {})}
        if isinstance(body, bytes): request_headers['X-Chunk-SHA256'] = hashlib.sha256(body).hexdigest()
        try:
            response = self.opener.open(urllib.request.Request(ORIGIN + path, raw, request_headers, method=method), timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status, content = response.status, response.read()
            assert status == expected, (method, path, status, content[:200])
            return content if binary else json.loads(content)


def main():
    a, b = User('alice'), User('bob')
    records = {}; checks = []; batch = uuid4().hex[:8]
    for user, peer in ((a, b), (b, a)):
        tasks = [user.call('POST', '/api/assistant/tasks', {'title': user.username + f' API 隔离任务 {index}',
                 'draft': '只属于当前账户的验收草稿', 'idempotency_key': f'product-contract-{batch}-{user.username}-{index}'}, expected=201)['task'] for index in (1, 2)]
        for task in tasks:
            peer.call('GET', '/api/assistant/tasks/' + task['id'], expected=404)
            peer.call('POST', '/api/assistant/tasks/' + task['id'] + '/draft', {'title': 'cross', 'draft': 'cross', 'version': task['version']}, expected=404)
            spoof = peer.call('GET', '/api/assistant/tasks', headers={'x-evomind-principal-id': user.username, 'x-evomind-tenant-id': 'tenant_' + 'a' * 24})
            assert task['id'] not in {row['id'] for row in spoof['tasks']}
        data = f'private fixture for {user.username}\n'.encode()
        upload = user.call('POST', '/api/assistant/uploads', {'name': user.username + '-private.txt', 'total_bytes': len(data)}, expected=201)
        peer.call('PUT', f"/api/assistant/uploads/{upload['id']}/chunks/0", data, expected=404)
        user.call('PUT', f"/api/assistant/uploads/{upload['id']}/chunks/0", data)
        file = user.call('POST', f"/api/assistant/uploads/{upload['id']}/complete", {})['attachment']
        user.call('POST', '/api/assistant/tasks/' + tasks[0]['id'] + '/files', {'attachment_ids': [file['id']], 'version': tasks[0]['version']})
        peer.call('GET', '/api/assistant/files/' + file['id'], expected=404)
        assert user.call('GET', '/api/assistant/files/' + file['id'], binary=True) == data
        assert user.call('GET', '/api/assistant/tasks/' + tasks[1]['id'])['files'] == []
        profiles = user.call('GET', '/api/assistant/model-profiles')['profiles']
        profile = next((row for row in profiles if row['enabled'] and row['model'] == 'isolated-ui-fixture'), None)
        if not profile:
            profile = user.call('POST', '/api/assistant/model-profiles', {'name': user.username + ' 隔离脚本', 'provider': 'openai',
                'base_url': 'https://api.example.com/v1', 'model': 'isolated-ui-fixture', 'api_key': 'fixture-only-public-test-key'}, expected=201)['profile']
        peer.call('GET', '/api/assistant/model-profiles/' + profile['id'], expected=404)
        run_body = {'prompt': 'Generate the isolated acceptance file.', 'user_task_id': tasks[0]['id'], 'attachment_ids': [file['id']],
                    'model_profile_id': profile['id'], 'model_profile_version': profile['version'], 'idempotency_key': user.username + '-product-contract-run-' + batch}
        run = user.call('POST', '/api/assistant/runs', run_body, expected=201)
        assert user.call('POST', '/api/assistant/runs', run_body)['id'] == run['id']
        for path in [f"/api/assistant/runs/{run['id']}", f"/api/assistant/runs/{run['id']}/events"]:
            peer.call('GET', path, expected=404)
        peer.call('POST', f"/api/assistant/runs/{run['id']}/actions", {'action': 'pause'}, expected=404)
        for _ in range(60):
            snapshot = user.call('GET', '/api/assistant/runs/' + run['id'])
            if snapshot['status'] in {'completed', 'failed', 'blocked'}: break
            time.sleep(0.2)
        assert snapshot['status'] == 'completed', snapshot['status']
        artifact = snapshot['artifacts'][0]
        peer.call('GET', '/api/assistant/artifacts/' + artifact['id'], expected=404)
        downloaded = user.call('GET', '/api/assistant/artifacts/' + artifact['id'] + '?download=1', binary=True)
        assert hashlib.sha256(downloaded).hexdigest() == artifact['sha256']
        assert b'GPU' in downloaded and artifact['run_id'] == run['id']
        user.call('GET', '/api/tasks', expected=403)
        records[user.username] = {'tasks': [task['id'] for task in tasks], 'run': run['id'], 'artifact': artifact['id'], 'sha256': artifact['sha256'], 'file': file['id'], 'model': profile['id']}
        checks.append(user.username + ': two owned tasks, spoof rejected, cross CRUD/file/upload/model/run/SSE/action/download rejected, duplicate Run reconciled, actual file hash passed')
    (OUT / 'authenticated-contract.json').write_text(json.dumps({'candidate': json.loads((OUT / 'preview.json').read_text())['candidate'],
        'records': records, 'checks': checks, 'provider': 'local script; not a real model', 'production_deployed': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': len(checks), 'evidence': str(OUT / 'authenticated-contract.json')}))


if __name__ == '__main__': main()
