"""Published source readiness is separate from successful tool execution."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from evomind_runtime.report_document import build_document, canonical
from evomind_runtime.runtime import AgentRuntime


@pytest.fixture
def runtime(tmp_path):
    value = AgentRuntime(tmp_path)
    # Unit controls inspect frozen jobs without requiring rendering dependencies.
    value.reports.stopping = True
    try:
        yield value
    finally:
        value.close(timeout=5)


def unpublished_summary(runtime, prompt='Analyze this synthetic fixture'):
    run = runtime.assistant.create_run(prompt=prompt, start=False)
    for name, arguments in [
        ('file_write', {'path':'outputs/summary.json', 'content':'{"count":4,"sum":13}'}),
        ('file_read', {'path':'outputs/summary.json'}),
    ]:
        outcome = runtime.invoke_tool(run['id'], name, arguments)
        assert outcome['status'] == 'completed' and outcome['result']['ok'] is True
    session = runtime.get_session(run['id'])
    outcome = runtime.invoke_tool(run['id'], 'directory_hash', {
        'directory_id':session['metadata']['super_agent_directory_ids'][0],
        'relative_path':'outputs/summary.json'})
    assert outcome['status'] == 'completed' and outcome['result']['ok'] is True
    return run


def generate(runtime, run_id, **arguments):
    result = runtime.invoke_tool(run_id, 'report_generate', {'formats':['markdown'], **arguments})
    assert result['status'] == 'completed' and result['result']['ok'] is True
    return result['result']['content']['report_job']


def publish(runtime, run_id):
    result = runtime.invoke_tool(run_id, 'artifact_publish', {'path':'outputs/summary.json'})
    assert result['status'] == 'completed' and result['result']['ok'] is True
    return result['result']['content']['artifact']


@pytest.mark.parametrize('explicit', [False, True])
def test_analysis_checks_without_published_sources_are_partial(runtime, explicit):
    run = unpublished_summary(runtime)
    arguments = {'report_kind':'analysis'} if explicit else {}
    document = build_document(runtime, run['id'], arguments)
    assert document['kind'] == 'analysis' and len(document['checks']) == 3
    assert document['sources'] == [] and document['report_status'] == 'partial'
    assert document['evidence_status'] == 'unverified'
    missing = '\n'.join(document['missing'])
    assert all(word in missing for word in ('artifact_publish', 'artifact_ids', 'new report', 'task record'))
    assert runtime.store.list_deliverables(run['id']) == []


def test_diagnostic_without_artifacts_remains_an_honest_task_record(runtime):
    run = runtime.assistant.create_run(prompt='Connection diagnostic task record', start=False)
    result = runtime.invoke_tool(run['id'], 'runtime_health', {})
    assert result['result']['ok'] is True
    document = build_document(runtime, run['id'], {'report_kind':'diagnostic'})
    assert document['sources'] == [] and document['checks']
    assert document['report_status'] == 'ready' and document['evidence_status'] == 'unverified'
    assert not any('No published analysis sources' in item for item in document['missing'])


def test_diagnostic_with_no_execution_evidence_is_still_partial(runtime):
    run = runtime.assistant.create_run(prompt='Connection diagnostic task record', start=False)
    document = build_document(runtime, run['id'], {'report_kind':'diagnostic'})
    assert document['report_status'] == 'partial'
    assert any('No execution or artifact evidence' in item for item in document['missing'])


def test_compact_job_and_tool_status_expose_missing_source_without_publishing(runtime):
    run = unpublished_summary(runtime)
    summary = generate(runtime, run['id'], report_kind='analysis')
    assert summary['source_count'] == 0 and summary['source_artifact_ids'] == []
    assert summary['report_status'] == 'partial' and summary['missing']
    public = runtime.reports.get(run['id'], summary['id'], include_document=False)
    assert 'document' not in public and public['source_count'] == 0
    result = runtime.invoke_tool(run['id'], 'report_status', {'report_id':summary['id']})
    current = result['result']['content']['report_job']
    assert current['missing'] == summary['missing'] and current['source_count'] == 0
    assert 'path' not in current and runtime.store.list_deliverables(run['id']) == []


def test_publish_then_explicit_source_binding_is_ready_and_preserves_old_snapshot(runtime):
    run = unpublished_summary(runtime)
    old = generate(runtime, run['id'], report_kind='analysis')
    old_source = runtime.reports._row(old['id'])['source_json']
    artifact = publish(runtime, run['id'])
    new = generate(runtime, run['id'], report_kind='analysis', artifact_ids=[artifact['id']])
    assert new['id'] != old['id'] and new['document_sha256'] != old['document_sha256']
    assert new['source_count'] == 1 and new['source_artifact_ids'] == [artifact['id']]
    assert new['report_status'] == 'ready' and new['missing'] == []
    assert new['evidence_status'] == 'hash_verified'
    assert runtime.reports._row(old['id'])['source_json'] == old_source
    result = runtime.invoke_tool(run['id'], 'report_generate', {'report_id':old['id']})
    resumed = result['result']['content']['report_job']
    assert resumed['source_count'] == 0 and resumed['report_status'] == 'partial'
    assert resumed['document_sha256'] == old['document_sha256']


def test_unknown_and_foreign_source_ids_remain_rejected(runtime):
    run = unpublished_summary(runtime)
    other = unpublished_summary(runtime)
    foreign = publish(runtime, other['id'])
    for identifier in ('artifact_unknown', foreign['id']):
        with pytest.raises(ValueError, match='not_owned'):
            build_document(runtime, run['id'], {'report_kind':'analysis', 'artifact_ids':[identifier]})


def test_public_missing_details_are_redacted_without_rewriting_frozen_source(runtime, monkeypatch):
    run = unpublished_summary(runtime)
    summary = generate(runtime, run['id'], report_kind='analysis')
    row = runtime.reports._row(summary['id'])
    document = json.loads(row['source_json'])
    document['missing'] = ['safe missing source', 'password=fixture-must-not-be-exposed']
    supplied = {**row, 'source_json':canonical(document)}
    monkeypatch.setattr(runtime.reports, '_row', lambda identifier: dict(supplied))
    result = runtime.reports.get(run['id'], summary['id'], include_document=False)
    assert 'fixture-must-not-be-exposed' not in json.dumps(result['missing'])
    assert result['missing'][0] == 'safe missing source'
    assert supplied['source_json'] == canonical(document)


def test_tool_description_explains_snapshot_order_and_new_report_recovery(runtime):
    description = runtime.registry.get('report_generate').description
    assert all(word in description for word in ('artifact_publish', 'artifact_ids', 'source_count/missing',
                                                'already published', 'new report', 'report_status'))


REAL_SOURCE_FLOW = r'''
import hashlib, json, sys, time
from pathlib import Path
from evomind_runtime.runtime import AgentRuntime
runtime = AgentRuntime(sys.argv[1])
def complete(run_id, result):
    identifier = result['result']['content']['report_job']['id']
    deadline = time.monotonic()+90
    while True:
        job = runtime.reports.get(run_id,identifier)
        if job['status'] in {'ready','partial','failed'}: break
        assert time.monotonic()<deadline, 'report_fixture_timeout'
        time.sleep(0.02)
    assert job['status']!='failed', job.get('error_code')
    response = runtime.invoke_tool(run_id,'report_status',{'report_id':identifier})
    return job,response['result']['content']['report_job']
try:
    run = runtime.assistant.create_run(prompt='Synthetic analysis source fixture',start=False)
    runtime.invoke_tool(run['id'],'file_write',{'path':'outputs/summary.json','content':'{"count":4,"sum":13}'})
    runtime.invoke_tool(run['id'],'file_read',{'path':'outputs/summary.json'})
    args={'report_kind':'analysis','language':'zh-CN','formats':['markdown','html','docx','pdf']}
    old,old_summary=complete(run['id'],runtime.invoke_tool(run['id'],'report_generate',args))
    old_source=runtime.reports._row(old['id'])['source_json']
    old_hash=hashlib.sha256(old_source.encode()).hexdigest()
    result=runtime.invoke_tool(run['id'],'artifact_publish',{'path':'outputs/summary.json'})
    source_id=result['result']['content']['artifact']['id']
    new,new_summary=complete(run['id'],runtime.invoke_tool(run['id'],'report_generate',{**args,'artifact_ids':[source_id]}))
    assert runtime.reports._row(old['id'])['source_json']==old_source
    from docx import Document
    word_artifact=next(item for item in old['artifacts'] if item['name']=='report.docx')
    word=Document(runtime.store.get_deliverable(word_artifact['id'])['path'])
    word_text='\n'.join(p.text for p in word.paragraphs)
    assert 'artifact_publish' in word_text and 'artifact_ids' in word_text
    print(json.dumps({'old_status':old['status'],'old_source_count':old_summary['source_count'],
        'old_missing_visible':bool(old_summary['missing']),'new_status':new['status'],
        'new_source_ids':new_summary['source_artifact_ids'],'expected_source_id':source_id,
        'snapshot_unchanged':hashlib.sha256(runtime.reports._row(old['id'])['source_json'].encode()).hexdigest()==old_hash,
        'word_recovery_hint':True}),flush=True)
finally:
    assert runtime.close(timeout=10)
'''


def test_real_renderer_preserves_partial_report_and_generates_new_bound_report(tmp_path):
    interpreter = os.environ.get('EVOMIND_REPORT_TEST_PYTHON') or sys.executable
    probe = subprocess.run([interpreter, '-c', 'import docx,fitz,matplotlib,psutil'], capture_output=True, timeout=20)
    if probe.returncode:
        pytest.skip('selected document runtime lacks rendering dependencies; real source flow unverified')
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'PYTHONPATH':str(root/'src'), 'PYTHONUTF8':'1'}
    result = subprocess.run([interpreter, '-c', REAL_SOURCE_FLOW, str(tmp_path/'source-flow')],
                            env=environment, capture_output=True, text=True, encoding='utf-8', timeout=150)
    assert result.returncode == 0, result.stderr[-3000:]
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence['old_status'] == 'partial' and evidence['old_source_count'] == 0
    assert evidence['old_missing_visible'] and evidence['word_recovery_hint'] and evidence['snapshot_unchanged']
    assert evidence['new_status'] == 'ready' and evidence['new_source_ids'] == [evidence['expected_source_id']]
