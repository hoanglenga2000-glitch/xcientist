"""Real loopback owner-scoped export API; fixtures never call a model or GPU."""
import copy
import threading
import time
from pathlib import Path

import pytest

from test_invitation_tenant_access import api, ALICE, BOB, COLLEAGUE, new_run


def endpoint(run):
    return f"/v1/runs/{run['id']}/reports"


@pytest.mark.parametrize('lifecycle', ['completed', 'failed', 'cancelled', 'blocked', 'paused', 'running'])
def test_export_admission_preserves_execution_lifecycle_and_uses_no_model(api, monkeypatch, lifecycle):
    runtime, request = api
    run = new_run(request)
    runtime.store.update_assistant_run(run['id'], status=lifecycle, completed_at='frozen-time')
    runtime.store.update_session(run['id'], status=lifecycle)
    runtime.reports.stopping = True
    original = copy.deepcopy(runtime.store.get_assistant_run(run['id']))
    session = copy.deepcopy(runtime.store.get_session(run['id']))
    monkeypatch.setattr(runtime, 'invoke_tool', lambda *a, **kw: pytest.fail('must not re-enter direct tool execution'))
    status, response = request('POST', endpoint(run), body={'formats': ['markdown'], 'idempotency_key': 'fixture'})
    assert status == 202 and response['status'] == 'accepted', response
    job = response['result']['content']['report_job']
    assert job['run_id'] == run['id'] and job['status'] == 'queued'
    assert runtime.store.get_assistant_run(run['id']) == original
    assert runtime.store.get_session(run['id']) == session
    assert runtime.store.list_tool_calls(run['id']) == []
    again = request('POST', endpoint(run), body={'formats': ['markdown'], 'idempotency_key': 'new-key'})[1]
    assert again['result']['content']['report_job']['id'] == job['id']


@pytest.mark.parametrize('other', [BOB, COLLEAGUE])
def test_foreign_user_cannot_list_generate_read_or_resume_reports(api, other):
    runtime, request = api
    run = new_run(request)
    runtime.reports.stopping = True
    job = request('POST', endpoint(run), body={'formats': ['markdown']})[1]['result']['content']['report_job']
    for method, url, body in [('GET', endpoint(run), None), ('GET', endpoint(run)+'/'+job['id'], None),
                             ('POST', endpoint(run), {'formats': ['markdown']}),
                             ('POST', endpoint(run), {'report_id': job['id']})]:
        assert request(method, url, other, body) == (404, {'error': 'not_found'})
    assert len(runtime.reports.list(run['id'])) == 1


def test_unauthenticated_requests_rejected_before_report_dispatch(api):
    runtime, request = api
    run = new_run(request)
    for method in ['GET', 'POST']:
        status, response = request(method, endpoint(run), body={} if method == 'POST' else None,
                                   extra={'Authorization': ''})
        assert status == 401 and response == {'error': 'unauthorized'}
    assert runtime.reports.list(run['id']) == []


@pytest.mark.parametrize('body', [
    {'path': 'C:/outside/report.pdf'}, {'output_dir': '../outside'}, {'name': 'outside.pdf'},
    {'formats': []}, {'formats': 'pdf'}, {'formats': ['exe']}, {'artifact_ids': [12]},
    {'title': {}}, {'language': 'bad'}, {'report_id': '../report'}, {'idempotency_key': 42},
    {'report_id': 'report_'+'0'*32, 'formats': ['markdown']}, {'tool_name': 'shell_exec'},
])
def test_export_request_rejects_arbitrary_paths_and_invalid_fields(api, body):
    runtime, request = api
    run = new_run(request)
    assert request('POST', endpoint(run), body=body)[0] == 400
    assert runtime.reports.list(run['id']) == []


def test_observe_only_session_cannot_create_export(api):
    runtime, request = api
    run = new_run(request)
    runtime.store.update_session(run['id'], permission_level='observe')
    assert request('POST', endpoint(run), body={'formats': ['markdown']}) == (
        403, {'error': 'report_generation_not_permitted'})


def test_export_endpoint_does_not_open_generic_tools(api):
    runtime, request = api
    run = new_run(request)
    for name in ['report_generate', 'file_write', 'shell_exec', 'managed_tensor_train']:
        assert request('POST', f"/v1/sessions/{run['id']}/tools", body={'tool_name': name, 'arguments': {}})[0] == 403
    assert runtime.store.list_tool_calls(run['id']) == []


def test_export_ack_precedes_render_and_owner_only_artifact_download(api, monkeypatch):
    import evomind_runtime.report_jobs as jobs
    runtime, request = api
    entered, release = threading.Event(), threading.Event()
    def render(document, output):
        entered.set()
        assert release.wait(5)
        output.mkdir(parents=True)
        (output / 'report.md').write_text('fixture-only-report', encoding='utf-8')
        return {'schema': 'evomind.report_package.v2', 'document_sha256': 'a'*64, 'report_status': 'ready', 'files': []}
    monkeypatch.setattr(jobs, 'render_document', render)
    run = new_run(request)
    try:
        started = time.monotonic()
        status, response = request('POST', endpoint(run), body={'formats': ['markdown']})
        assert status == 202 and time.monotonic() - started < 1 and entered.wait(1)
        job = response['result']['content']['report_job']
        assert request('GET', endpoint(run))[1]['reports'][0]['id'] == job['id']
        assert not release.is_set()
        release.set()
        deadline = time.monotonic() + 5
        while runtime.reports.get(run['id'], job['id'])['status'] == 'running' and time.monotonic() < deadline:
            time.sleep(.01)
        final = runtime.reports.get(run['id'], job['id'])
        assert final['status'] == 'ready', final
        artifact = final['artifacts'][0]
        url = '/v1/artifacts/' + artifact['id']
        assert request('GET', url+'?download=1') == (200, b'fixture-only-report')
        assert request('GET', url+'/preview')[0] == 200
        for other in [BOB, COLLEAGUE]:
            assert request('GET', url+'?download=1', other) == (404, {'error': 'not_found'})
            assert request('GET', url+'/preview', other) == (404, {'error': 'not_found'})
        assert request('GET', url+'?download=1', extra={'Authorization': ''})[0] == 401
    finally:
        release.set()
