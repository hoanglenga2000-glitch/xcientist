"""Packaged Web -> runtime -> report/download acceptance in an isolated fixture.

No production activation, real model, HPC, Chrome or account provisioning.
The owner uses the real login endpoint. Additional principals are explicitly
synthetic signed fixture identities, not independent production login proof.
"""
import argparse
import hashlib
import http.client
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
import zipfile


def request(port, method, path, *, body=None, headers=None):
    headers = dict(headers or {})
    if body is not None:
        body = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
    try:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError('report_response_size_limit')
        received = dict((key.lower(), value) for key, value in response.getheaders())
        if 'application/json' in received.get('content-type', ''):
            data = json.loads(data)
        return response.status, received, data
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--web-sha256', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--build-id', required=True)
    parser.add_argument('--installed-manifest', type=Path, required=True)
    parser.add_argument('--installed-manifest-sha256', required=True)
    args = parser.parse_args()
    stage = args.stage_root.resolve(strict=True)
    stage.relative_to(Path('C:/ProgramData/EvoMind/staging').resolve(strict=True))
    spec = importlib.util.spec_from_file_location('report_export_services', stage/'verify_invitation_server_candidate.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    receipt_path = stage/'report-export-acceptance.json'
    if receipt_path.exists():
        raise ValueError('report_acceptance_already_exists')
    site = Path('C:/EMQA/sys0907-report-delivered-v2/site-packages')
    if helper.sha(args.installed_manifest) != args.installed_manifest_sha256:
        raise ValueError('installed_manifest_mismatch')
    installed = json.loads(args.installed_manifest.read_text())

    def verify_site():
        expected = {row['path']: row for row in installed['files']}
        actual = {path.relative_to(site).as_posix(): path for path in site.rglob('*') if path.is_file()}
        if len(expected) != 2187 or set(actual) != set(expected):
            raise ValueError('report_site_file_set_mismatch')
        for name, path in actual.items():
            if (path.is_symlink() or path.stat().st_size != expected[name]['bytes']
                    or helper.sha(path) != expected[name]['sha256']):
                raise ValueError('report_site_file_mismatch')

    verify_site()
    flow = stage/'report-export-flow'
    flow.mkdir(exist_ok=False)
    helper.extract(stage/'web.zip', args.web_sha256, flow/'web', 'operational-overlay-manifest.json')
    helper.extract(stage/'runtime.zip', args.runtime_sha256, flow/'runtime', 'runtime-hotfix-manifest.json')
    (flow/'workspace').mkdir()
    shutil.copyfile(stage/'fixture-schema.sqlite', flow/'workspace/workstation.sqlite')
    secret, password = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    identity = json.loads((flow/'web/runtime-build-manifest.json').read_text())['source_tree_sha256']
    environment = helper.fixture_environment(flow, secret, password, args.build_id, identity)
    environment['PYTHONPATH'] = os.pathsep.join((str(flow/'runtime'), str(site), str(helper.RELEASE/'src')))
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['MPLCONFIGDIR'] = str(flow/'matplotlib-cache')
    services = helper.FixtureServices(flow, environment, 'C:/Program Files/nodejs/node.exe')
    before, checks = helper.production_snapshot(), []
    result = {'schema': 'evomind.report_export_http_acceptance.v1', 'status': 'running',
              'build_id': args.build_id, 'web_sha256': args.web_sha256, 'runtime_sha256': args.runtime_sha256,
              'fixture_only': True, 'real_model_requests': 0, 'hpc_accessed': False, 'production_deployed': False,
              'browser_e2e': 'not_performed', 'additional_principals': 'synthetic_signed_fixture_sessions'}

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})
        if not passed:
            raise ValueError(name)

    try:
        services.start()
        port = services.web_port
        origin = 'http://127.0.0.1:' + str(port)
        status, headers, _ = request(port, 'POST', '/api/auth/login',
            body={'username': 'invitation_acceptance', 'password': password}, headers={'Origin': origin})
        check('report_real_owner_login', status == 200 and 'set-cookie' in headers)
        owner = {'Origin': origin, 'Cookie': headers['set-cookie'].split(';', 1)[0]}
        _, _, session = request(port, 'GET', '/api/session/status', headers=owner)
        owner[helper.CSRF] = session['csrf_token']
        others = []
        for username, tenant in [('colleague_fixture', session['tenant_id']), ('foreign_fixture', 'tenant_'+'b'*24)]:
            principal = {'Origin': origin, 'Cookie': helper.COOKIE+'='+helper.signed_principal(secret, username, tenant)}
            _, _, state = request(port, 'GET', '/api/session/status', headers=principal)
            principal[helper.CSRF] = state['csrf_token']
            others.append(principal)
        status, _, run = request(port, 'POST', '/api/assistant/runs', headers=owner,
            body={'prompt': 'ISOLATED_ACCEPTANCE_ECHO report transport fixture; not actual training', 'idempotency_key': 'report-fixture'})
        check('report_fixture_run_created', status == 201 and bool(run.get('id')))
        run_url = '/api/assistant/runs/' + run['id']
        deadline = time.monotonic()+20
        while time.monotonic()<deadline:
            _, _, current = request(port, 'GET', run_url, headers=owner)
            if current.get('terminal'):
                break
            time.sleep(.1)
        check('report_fixture_run_completed', current.get('status') == 'completed')
        url = run_url+'/reports'
        for method in ['GET', 'POST']:
            check('report_unauthenticated_'+method.lower(), request(port, method, url,
                body={} if method=='POST' else None, headers={'Origin': origin})[0] == 401)
        check('report_csrf_required', request(port, 'POST', url, body={},
            headers={key: value for key, value in owner.items() if key != helper.CSRF})[0] == 403)
        jobs = []
        for kind in ['diagnostic', 'training', 'inference']:
            status, _, answer = request(port, 'POST', url, headers=owner, body={
                'title': 'Synthetic '+kind+' transport fixture', 'report_kind': kind,
                'formats': ['markdown', 'html', 'docx', 'pdf'], 'language': 'zh-CN'})
            check(kind+'_report_accepted', status == 202 and answer.get('status') == 'accepted')
            job = answer['result']['content']['report_job']
            deadline = time.monotonic()+90
            while time.monotonic()<deadline:
                status, _, payload = request(port, 'GET', url+'?report_id='+job['id'], headers=owner)
                job = payload['report']
                if job['status'] in {'ready', 'partial', 'failed'}:
                    break
                time.sleep(.1)
            check(kind+'_report_terminal', status==200 and job['status'] in {'ready', 'partial'} and job['run_id']==run['id'])
            check(kind+'_execution_status_preserved', job['execution_status']=='completed'
                  and request(port, 'GET', run_url, headers=owner)[2]['status']=='completed')
            check(kind+'_not_scientific_validation', job['evidence_status'] != 'independently_verified')
            required = {'report.html', 'report.pdf', 'report.docx', 'report-bundle.zip'}
            artifacts = [row for row in job['artifacts'] if row['name'] in required]
            check(kind+'_required_formats_published', {row['name'] for row in artifacts} == required)
            for artifact in artifacts:
                path = '/api/assistant/artifacts/' + artifact['id']
                status, response_headers, data = request(port, 'GET', path+'?download=1', headers=owner)
                check(kind+'_'+artifact['name']+'_download', status==200 and isinstance(data, bytes)
                      and len(data)==artifact['bytes'] and hashlib.sha256(data).hexdigest()==artifact['sha256'])
                if artifact['name']=='report.docx':
                    with zipfile.ZipFile(io.BytesIO(data)) as archive:
                        check(kind+'_editable_word', 'word/document.xml' in archive.namelist())
                if artifact['name']=='report.pdf':
                    import fitz
                    with fitz.open(stream=data, filetype='pdf') as pdf:
                        text = ''.join(page.get_text(flags=fitz.TEXTFLAGS_TEXT & ~fitz.TEXT_PRESERVE_LIGATURES) for page in pdf)
                        check(kind+'_readable_pdf', pdf.page_count>0 and run['id'] in text)
                for index, other in enumerate(others):
                    check(kind+'_'+artifact['name']+'_other_'+str(index), request(port,'GET',path+'?download=1',headers=other)[0]==404)
                    check(kind+'_'+artifact['name']+'_preview_other_'+str(index), request(port,'GET',path+'?preview=1',headers=other)[0]==404)
            for index, other in enumerate(others):
                check(kind+'_list_other_'+str(index), request(port,'GET',url,headers=other)[0]==404)
                check(kind+'_read_other_'+str(index), request(port,'GET',url+'?report_id='+job['id'],headers=other)[0]==404)
                check(kind+'_generate_other_'+str(index), request(port,'POST',url,body={'formats':['markdown']},headers=other)[0]==404)
                check(kind+'_resume_other_'+str(index), request(port,'POST',url,body={'report_id':job['id']},headers=other)[0]==404)
            jobs.append(job['id'])
        latency=[]
        for _ in range(20):
            started=time.monotonic()
            status, _, answer=request(port,'POST',url,body={'report_id':jobs[0]},headers=owner)
            latency.append((time.monotonic()-started)*1000)
            check('report_idempotent_resume_'+str(len(latency)), status==202 and answer['result']['content']['report_job']['id']==jobs[0])
        p95=sorted(latency)[math.ceil(.95*len(latency))-1]
        check('report_ack_p95',p95<=1000,p95_ms=round(p95,3),samples=len(latency))
        check('three_frozen_report_jobs_only',len(request(port,'GET',url,headers=owner)[2]['reports'])==3)
        result.update(run_id=run['id'], report_ids=jobs)
    except Exception as error:
        result['error_class']=type(error).__name__
        result['error_code']=str(error) if isinstance(error,ValueError) and str(error).replace('_','').isalnum() else 'report_acceptance_failed'
    finally:
        try:
            services.stop()
            result['cleanup']='owned_fixture_processes_stopped'
        except Exception:
            result['cleanup']='unconfirmed'
        result['launch_records']=services.launch_records
        result['production_unchanged']=helper.production_snapshot()==before
        verify_site()
        result['report_site_unchanged']=True
        result['checks']=checks
        result['status']='passed' if checks and all(row['passed'] for row in checks) and 'error_code' not in result and result['production_unchanged'] and result['cleanup']=='owned_fixture_processes_stopped' else 'failed'
        helper.write_json(receipt_path,result)
        print(json.dumps({'status':result['status'],'checks':len(checks),'error_code':result.get('error_code'), 'production_unchanged':result['production_unchanged']}))
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
