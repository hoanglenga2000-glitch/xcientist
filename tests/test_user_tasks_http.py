"""User task contracts through the authenticated runtime HTTP surface.

Only temporary data is used. The first slice must not create any Run.
"""
from __future__ import annotations

from contextlib import contextmanager
import http.client
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
import threading
import pytest

from evomind_runtime.http_server import make_handler
from evomind_runtime.runtime import AgentRuntime


@pytest.fixture(autouse=True)
def no_external_model_credentials(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(('OPENAI_', 'ANTHROPIC_', 'DEEPSEEK_', 'CLAUDE_', 'EVOLUTION_')):
            monkeypatch.delenv(name)


@contextmanager
def task_api(root):
    runtime = AgentRuntime(root)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(runtime, 'isolated-task-http-test'))
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()

    def request(method, path, body=None, owner='alice', tenant='a', token='isolated-task-http-test', binary=False):
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json',
                   'X-EvoMind-Access-Scope': 'user.v1', 'X-EvoMind-Tenant-Id': 'tenant_' + tenant * 24,
                   'X-EvoMind-Principal-Id': owner}
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        try:
            if isinstance(body, bytes):
                headers['Content-Type'] = 'application/octet-stream'
                headers['X-Chunk-SHA256'] = hashlib.sha256(body).hexdigest()
            conn.request(method, path, body=body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None, headers=headers)
            response = conn.getresponse()
            content = response.read()
            return response.status, content if binary else json.loads(content)
        finally:
            conn.close()

    try:
        yield request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def test_uploaded_data_is_private_and_selected_for_one_task_with_durable_version(tmp_path):
    data = b'private input for Alice only\n'
    with task_api(tmp_path) as request:
        task = request('POST', '/v1/user-tasks', {'title': '资料任务', 'idempotency_key': 'files-task-one'})[1]['task']
        other = request('POST', '/v1/user-tasks', {'title': '独立任务', 'idempotency_key': 'files-task-two'})[1]['task']
        upload = request('POST', '/v1/uploads', {'name': 'private.txt', 'total_bytes': len(data)})[1]
        assert request('PUT', f"/v1/uploads/{upload['id']}/chunks/0", data)[0] == 200
        file = request('POST', f"/v1/uploads/{upload['id']}/complete", {})[1]['attachment']
        status, catalog = request('GET', '/v1/user-files')
        assert status == 200, catalog
        assert [item['id'] for item in catalog['files']] == [file['id']]
        assert 'path' not in catalog['files'][0]
        assert request('GET', '/v1/user-files', owner='bob')[1]['files'] == []
        assert request('GET', '/v1/user-files/' + file['id'], owner='bob')[0] == 404
        assert request('GET', '/v1/user-files/' + file['id'], binary=True) == (200, data)
        endpoint = '/v1/user-tasks/' + task['id'] + '/files'
        body = {'attachment_ids': [file['id']], 'version': 1}
        assert request('POST', endpoint, body, owner='bob')[0] == 404
        attached = request('POST', endpoint, body)
        assert attached[0] == 200, attached
        assert attached[1]['task']['version'] == 2
        assert request('POST', endpoint, {**body, 'attachment_ids': []})[0] == 409
        assert request('GET', '/v1/user-tasks/' + other['id'])[1]['files'] == []
        assert request('POST', '/v1/runs', {'prompt': 'Read my selected file', 'user_task_id': task['id'],
            'attachment_ids': [], 'idempotency_key': 'must-not-omit-files'})[0] == 409
        assert request('GET', '/v1/runs')[1]['runs'] == []
    with task_api(tmp_path) as request:
        persisted = request('GET', '/v1/user-tasks/' + task['id'])[1]
        assert persisted['files'][0]['sha256'] == hashlib.sha256(data).hexdigest()


def test_save_draft_is_owned_durable_idempotent_and_does_not_start_a_run(tmp_path):
    body = {'title': '阅读工业缺陷检测论文', 'draft': '请整理方法与待核实的结论。', 'idempotency_key': 'create-task-one'}
    with task_api(tmp_path) as request:
        status, created = request('POST', '/v1/user-tasks', body)
        assert status == 201, created
        task = created['task']
        assert task['title'] == body['title'] and task['draft'] == body['draft']
        assert task['status'] == 'draft' and task['version'] == 1
        assert request('GET', '/v1/runs')[1]['runs'] == []
        assert request('GET', '/v1/user-tasks')[1]['tasks'] == [task]
        assert request('POST', '/v1/user-tasks', body)[1]['task']['id'] == task['id']
        assert request('POST', '/v1/user-tasks', {**body, 'title': 'changed'})[0] == 409
        assert request('GET', '/v1/user-tasks/' + task['id'], owner='bob')[0] == 404
        assert request('GET', '/v1/user-tasks', owner='bob')[1]['tasks'] == []
        assert request('GET', '/v1/user-tasks', tenant='b')[1]['tasks'] == []
        assert request('GET', '/v1/user-tasks', token='wrong')[0] == 401
        assert request('POST', '/v1/user-tasks', {**body, 'owner_id': 'bob'})[0] == 400
    with task_api(tmp_path) as request:
        detail = request('GET', '/v1/user-tasks/' + task['id'])[1]
        assert detail['task'] == task
        assert detail['runs'] == []


def test_edit_draft_keeps_task_scope_and_rejects_stale_browser_versions(tmp_path):
    with task_api(tmp_path) as request:
        task = request('POST', '/v1/user-tasks', {'title': '任务 A', 'idempotency_key': 'draft-task-a'})[1]['task']
        other = request('POST', '/v1/user-tasks', {'title': '任务 B', 'idempotency_key': 'draft-task-b'})[1]['task']
        path = '/v1/user-tasks/' + task['id'] + '/draft'
        change = {'title': '任务 A', 'draft': '只保存在 A', 'version': 1}
        assert request('POST', path, change, owner='bob')[0] == 404
        status, edited = request('POST', path, change)
        assert status == 200, edited
        assert edited['task']['version'] == 2 and edited['task']['draft'] == '只保存在 A'
        assert request('POST', path, {**change, 'draft': '过期浏览器内容'})[0] == 409
        assert request('GET', '/v1/user-tasks/' + task['id'])[1]['task']['draft'] == '只保存在 A'
        assert request('GET', '/v1/user-tasks/' + other['id'])[1]['task']['draft'] == ''
        assert request('GET', '/v1/runs')[1]['runs'] == []


def test_run_admission_and_explicit_history_link_do_not_cross_task_or_owner(tmp_path):
    with task_api(tmp_path) as request:
        task = request('POST', '/v1/user-tasks', {'title': '任务 A', 'idempotency_key': 'link-task-a'})[1]['task']
        other = request('POST', '/v1/user-tasks', {'title': '任务 B', 'idempotency_key': 'link-task-b'})[1]['task']
        run_body = {'prompt': 'Say hello.', 'user_task_id': task['id'], 'idempotency_key': 'owned-run-admit'}
        assert request('POST', '/v1/runs', {**run_body, 'prompt': '[Selected task: unrelated_legacy_task] Say hello.'})[0] == 409
        assert request('POST', '/v1/runs', run_body, owner='bob')[0] == 404
        status, run = request('POST', '/v1/runs', run_body)
        assert status == 201, run
        assert run['user_task_id'] == task['id']
        assert run['conversation_id'] == task['conversation_id']
        assert request('GET', '/v1/user-tasks/' + task['id'])[1]['runs'][0]['id'] == run['id']
        assert request('GET', '/v1/user-tasks/' + other['id'])[1]['runs'] == []
        assert request('POST', '/v1/runs', run_body)[1]['id'] == run['id']
        assert request('POST', '/v1/runs', {**run_body, 'user_task_id': other['id']})[0] == 409
        legacy = request('POST', '/v1/runs', {'prompt': 'Say hello.', 'idempotency_key': 'old-owned-run'})[1]
        link_path = '/v1/user-tasks/' + task['id'] + '/links'
        assert request('POST', link_path, {'run_id': legacy['id']}, owner='bob')[0] == 404
        assert request('POST', link_path, {'run_id': legacy['id']})[0] == 200
        assert request('POST', link_path, {'run_id': legacy['id']})[0] == 200
        assert request('POST', '/v1/user-tasks/' + other['id'] + '/links', {'run_id': legacy['id']})[0] == 409
        linked = request('GET', '/v1/user-tasks/' + task['id'])[1]['runs']
        assert {item['id'] for item in linked} == {run['id'], legacy['id']}
        # Linking must not rewrite the old conversation or replay execution.
        assert request('GET', '/v1/runs/' + legacy['id'])[1]['conversation_id'] == legacy['conversation_id']
