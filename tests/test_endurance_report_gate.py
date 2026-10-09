"""Acceptance reports must have a real, source-bound asynchronous export job."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('report_soak_harness', REPOSITORY / 'scripts/run_model_endurance_acceptance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# The selected document interpreter needs no pytest installation. It executes
# the production report tools and worker, not a test renderer or model request.
REAL_REPORT_FIXTURE = r'''
import hashlib, json, sys, time
from pathlib import Path
from evomind_runtime.runtime import AgentRuntime
runtime = AgentRuntime(sys.argv[1])
try:
    run = runtime.assistant.create_run(prompt='Synthetic system report acceptance, not research results', start=False)
    session = runtime.get_session(run['id'])
    root = Path(session['workspace_root'])
    # This exact payload's SHA contains "ff" and exercises MuPDF's default
    # U+FB00 ligature extraction rather than relying on a random Run digest.
    content = json.dumps({'case_id':2,'count':4,'sum':13,'min':-5,'max':7})
    assert 'ff' in hashlib.sha256(content.encode()).hexdigest()
    for name, args in [
        ('file_write', {'path':'outputs/summary.json','content':content}),
        ('file_read', {'path':'outputs/summary.json'}),
        ('directory_hash', {'directory_id':session['metadata']['super_agent_directory_ids'][0], 'relative_path':'outputs/summary.json'}),
        ('artifact_publish', {'path':'outputs/summary.json'}),
    ]:
        result = runtime.invoke_tool(run['id'], name, args)
        assert result['status']=='completed' and result['result']['ok'] is True
    result = runtime.invoke_tool(run['id'], 'report_generate', {
        'title':'Synthetic system report acceptance', 'report_kind':'analysis', 'language':'zh-CN',
        'formats':['markdown','html','docx','pdf']})
    assert result['status']=='completed' and result['result']['ok'] is True
    identifier = result['result']['content']['report_job']['id']
    deadline = time.monotonic()+90
    while True:
        job = runtime.reports.get(run['id'], identifier)
        if job['status'] in {'ready','partial','failed'}:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('fixture_report_deadline')
        time.sleep(0.02)
    assert job['status'] in {'ready','partial'}, job.get('error_code')
    status = runtime.invoke_tool(run['id'], 'report_status', {'report_id':identifier})
    assert status['result']['ok'] is True
    print(json.dumps({'run_id':run['id'],'report_id':identifier,
        'summary_sha256':hashlib.sha256((root/'outputs/summary.json').read_bytes()).hexdigest()}), flush=True)
finally:
    assert runtime.close(timeout=10)
'''


@pytest.fixture(scope='module')
def real_report(tmp_path_factory):
    if not all(importlib.util.find_spec(name) for name in ('docx', 'fitz')):
        pytest.skip('pytest runtime lacks DOCX/PDF validators; real report gate is unverified')
    interpreter = os.environ.get('EVOMIND_REPORT_TEST_PYTHON') or sys.executable
    probe = subprocess.run([interpreter, '-c', 'import docx, fitz, matplotlib, psutil'],
                           capture_output=True, timeout=20)
    if probe.returncode:
        pytest.skip('selected document runtime lacks DOCX/PDF/figure dependencies; real report gate is unverified')
    workspace = tmp_path_factory.mktemp('real-report-gate')
    environment = {**os.environ, 'PYTHONPATH': str(REPOSITORY / 'src'), 'PYTHONUTF8': '1'}
    result = subprocess.run([interpreter, '-c', REAL_REPORT_FIXTURE, str(workspace)],
                            env=environment, capture_output=True, text=True, encoding='utf-8', timeout=110)
    assert result.returncode == 0, result.stderr[-3000:]
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    from evomind_runtime.runtime import AgentRuntime
    runtime = AgentRuntime(workspace)
    case = SimpleNamespace(runtime=runtime, **receipt)
    case.calls = runtime.store.list_tool_calls(case.run_id)
    case.job = runtime.reports.get(case.run_id, case.report_id)
    try:
        yield case
    finally:
        runtime.close(timeout=10)


def verify(case, calls=None, digest=None):
    return module.verify_report_job_evidence(case.runtime, case.run_id,
        case.calls if calls is None else calls, case.summary_sha256 if digest is None else digest)


def test_manual_same_named_text_files_do_not_substitute_for_report_job():
    calls = [{'session_id':'run_fixture', 'tool_name':name, 'status':'completed',
              'arguments':{'path':'outputs/report.pdf'}, 'result':{'ok':True}}
             for name in ['file_write', 'artifact_publish']]
    result = module.verify_report_job_evidence(SimpleNamespace(), 'run_fixture', calls, 'a'*64)
    assert result == {'passed':False, 'error_code':'report_generation_receipt_missing'}


def test_real_report_job_readable_formats_and_manifest_pass(real_report):
    result = verify(real_report)
    assert result['passed'], result
    assert result['report_id'] == real_report.report_id
    assert result['pdf_pages'] >= 1 and result['docx_tables'] >= 1
    assert result['checked_payload_files'] >= 5


def test_pdf_hash_with_typographic_ligature_roundtrips_exactly(real_report):
    import fitz
    import re
    artifact = next(item for item in real_report.job['artifacts'] if item['name'] == 'report.pdf')
    path = real_report.runtime.store.get_deliverable(artifact['id'])['path']
    with fitz.open(path) as pdf:
        text = ''.join(page.get_text(flags=fitz.TEXTFLAGS_TEXT & ~fitz.TEXT_PRESERVE_LIGATURES)
                       for page in pdf)
    assert 'ff' in real_report.summary_sha256
    assert real_report.summary_sha256 in re.sub(r'\s+', '', text)
    assert verify(real_report)['passed']


def test_resuming_an_old_job_without_a_creation_receipt_cannot_pass(real_report):
    calls = copy.deepcopy(real_report.calls)
    for call in calls:
        if call['tool_name'] == 'report_generate':
            call['arguments'] = {'report_id':real_report.report_id}
    assert verify(real_report, calls)['error_code'] == 'report_generation_receipt_missing'


def test_unrelated_old_job_id_is_not_replaced_with_available_files(real_report):
    calls = copy.deepcopy(real_report.calls)
    for call in calls:
        if call['tool_name'] == 'report_generate':
            call['result']['content']['report_job']['id'] = 'report_' + '0'*32
    assert not verify(real_report, calls)['passed']


def test_foreign_run_receipt_cannot_claim_current_job(real_report):
    calls = copy.deepcopy(real_report.calls)
    for call in calls:
        if call['tool_name'] == 'report_generate':
            call['result']['content']['report_job']['run_id'] = 'run_foreign'
    result = verify(real_report, calls)
    assert not result['passed'] and result['error_code'] == 'report_request_identity_invalid'


def test_job_must_include_the_current_summary_not_a_previous_version(real_report):
    result = verify(real_report, digest='0'*64)
    assert not result['passed'] and result['error_code'] == 'report_current_summary_source_missing'


def test_terminal_report_status_must_be_observed_by_a_successful_tool(real_report):
    calls = [call for call in real_report.calls if call['tool_name'] != 'report_status']
    result = verify(real_report, calls)
    assert not result['passed'] and result['error_code'] == 'report_terminal_receipt_missing'


@pytest.mark.parametrize('status', ['queued','running','failed'])
def test_nonterminal_or_failed_durable_job_cannot_pass(real_report, monkeypatch, status):
    job = copy.deepcopy(real_report.job)
    job['status'] = status
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    result = verify(real_report)
    assert not result['passed'] and result['error_code'] == 'report_job_not_terminal'


def test_durable_job_from_another_run_is_rejected(real_report, monkeypatch):
    job = copy.deepcopy(real_report.job)
    job['run_id'] = 'run_foreign'
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    assert verify(real_report)['error_code'] == 'report_job_wrong_run'


def test_manifest_cannot_claim_a_different_frozen_source(real_report, monkeypatch):
    job = copy.deepcopy(real_report.job)
    job['manifest']['document_sha256'] = '0'*64
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    assert verify(real_report)['error_code'] == 'report_manifest_source_mismatch'


def test_job_cannot_relabel_partial_report_as_ready(real_report, monkeypatch):
    job = copy.deepcopy(real_report.job)
    job['report_status'] = 'partial' if job['status'] == 'ready' else 'ready'
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    assert verify(real_report)['error_code'] == 'report_manifest_status_mismatch'


def test_manifest_payload_requires_a_matching_published_snapshot(real_report, monkeypatch):
    job = copy.deepcopy(real_report.job)
    job['artifacts'] = [artifact for artifact in job['artifacts'] if artifact['name'] != 'report.pdf']
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    assert verify(real_report)['error_code'] == 'report_payload_not_published'


def test_published_snapshot_owned_by_another_run_is_rejected(real_report, monkeypatch):
    original = real_report.runtime.store.get_deliverable
    target = next(artifact['id'] for artifact in real_report.job['artifacts'] if artifact['name'] == 'report.pdf')
    def get(identifier):
        row = original(identifier)
        return {**row, 'run_id':'run_foreign'} if identifier == target else row
    monkeypatch.setattr(real_report.runtime.store, 'get_deliverable', get)
    assert verify(real_report)['error_code'] == 'report_artifact_wrong_run'


def test_changed_manifest_bytes_are_not_accepted_as_registered_evidence(real_report):
    artifact = next(item for item in real_report.job['artifacts'] if item['name'] == 'report-manifest.json')
    path = Path(real_report.runtime.store.get_deliverable(artifact['id'])['path'])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b'\n')
        assert verify(real_report)['error_code'] == 'report_artifact_hash_mismatch'
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('name,content', [
    ('report.docx', b'plain text is not an editable Word document'),
    ('report.pdf', b'%PDF-1.7\nnot a parseable PDF document'),
])
def test_parsers_reject_false_formats_even_with_consistent_hash_metadata(real_report, monkeypatch, tmp_path, name, content):
    from evomind_runtime.report_document import canonical
    job = copy.deepcopy(real_report.job)
    original_get = real_report.runtime.store.get_deliverable
    root = Path(real_report.runtime.get_session(real_report.run_id)['workspace_root'])
    counterfeit = root / 'outputs' / ('gate-negative-' + tmp_path.name)
    counterfeit.mkdir()
    replacement_rows = {}

    def replace_artifact(filename, data):
        item = next(artifact for artifact in job['artifacts'] if artifact['name'] == filename)
        row = original_get(item['id'])
        path = counterfeit / filename
        path.write_bytes(data)
        changed = {'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}
        item.update(changed)
        replacement_rows[item['id']] = {**row, **changed, 'path':str(path)}
        return changed

    changed = replace_artifact(name, content)
    next(entry for entry in job['manifest']['files'] if entry['path'] == name).update(changed)
    replace_artifact('report-manifest.json', canonical(job['manifest']).encode())
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    monkeypatch.setattr(real_report.runtime.store, 'get_deliverable',
                        lambda identifier: replacement_rows.get(identifier) or original_get(identifier))
    result = verify(real_report)
    assert not result['passed'], result
    assert result['error_code'] == 'report_evidence_invalid'


@pytest.mark.parametrize('variant', ['wrong_hash', 'wrong_run', 'fullwidth_hash'])
def test_valid_pdf_with_wrong_ascii_identity_is_rejected(real_report, monkeypatch, tmp_path, variant):
    import fitz
    from evomind_runtime.report_document import canonical
    run_id = 'run_' + '0' * 32 if variant == 'wrong_run' else real_report.run_id
    sha = '0' * 64 if variant == 'wrong_hash' else real_report.summary_sha256
    if variant == 'fullwidth_hash':
        sha = ''.join(chr(ord(char) + 0xFEE0) for char in sha)
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 30), run_id, fontsize=8)
        page.insert_text((30, 55), sha, fontsize=8, fontname='china-s' if variant == 'fullwidth_hash' else 'helv')
        payload = pdf.tobytes()
    job = copy.deepcopy(real_report.job)
    original_get = real_report.runtime.store.get_deliverable
    root = Path(real_report.runtime.get_session(real_report.run_id)['workspace_root'])
    output = root / 'outputs' / ('gate-pdf-identity-' + tmp_path.name)
    output.mkdir()
    replacements = {}
    for filename, data in [('report.pdf', payload)]:
        item = next(item for item in job['artifacts'] if item['name'] == filename)
        row = original_get(item['id'])
        path = output / filename
        path.write_bytes(data)
        changed = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        item.update(changed)
        replacements[item['id']] = {**row, **changed, 'path': str(path)}
        next(entry for entry in job['manifest']['files'] if entry['path'] == filename).update(changed)
    data = canonical(job['manifest']).encode()
    item = next(item for item in job['artifacts'] if item['name'] == 'report-manifest.json')
    row = original_get(item['id'])
    path = output / 'report-manifest.json'
    path.write_bytes(data)
    changed = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    item.update(changed)
    replacements[item['id']] = {**row, **changed, 'path': str(path)}
    monkeypatch.setattr(real_report.runtime.reports, 'get', lambda *args, **kwargs: job)
    monkeypatch.setattr(real_report.runtime.store, 'get_deliverable',
                        lambda identifier: replacements.get(identifier) or original_get(identifier))
    assert verify(real_report)['error_code'] == 'report_pdf_content_identity_mismatch'
