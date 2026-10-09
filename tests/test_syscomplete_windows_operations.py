"""All Windows operational commands are replaced by recording fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import syscomplete_windows_operations as module


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'node'
    (root/'config').mkdir(parents=True)
    (root/'config/node-config.json').write_text(json.dumps({'dedicated_user':'FixtureOwner',
        'network':{'runtime_port':58765,'web_port':58088,'llm_port':55068}}))
    monkeypatch.setattr(module.tx,'PRODUCTION_ROOT',root)
    calls=[]
    def run(arguments,**kwargs):
        calls.append((arguments,kwargs))
        if '-Action' in arguments:
            action=arguments[arguments.index('-Action')+1]
            return SimpleNamespace(returncode=0,stdout=json.dumps({'status':'completed','action':action}).encode(),stderr=b'')
        return SimpleNamespace(returncode=0,stdout=b'private-fixture-output',stderr=b'')
    spec={'root':str(root),'startup_side_effects_acknowledged':{
        'managed_hpc_bridge':True,'configured_hpc_reverification':True,'competition_pause':True}}
    helper=root/'stage/service-helper.ps1'
    helper.parent.mkdir()
    helper.write_bytes(b'Never executed fixture')
    monkeypatch.setattr(module,'SERVICE_HELPER',helper)
    spec['service_action_helper']={'path':str(helper),'sha256':module.tx.sha(helper)}
    return spec,calls,run


def test_constructing_adapter_is_read_only(fixture):
    spec,calls,run=fixture
    value=module.WindowsOperations(spec,run=run)
    assert calls==[] and value.ports==[58765,58088,55068]


def test_side_effect_acknowledgement_is_required_before_commands(fixture):
    spec,calls,run=fixture
    spec.pop('startup_side_effects_acknowledged')
    with pytest.raises(module.tx.Hold,match='not_acknowledged'):
        module.WindowsOperations(spec,run=run)
    assert calls==[]


def test_backup_permission_command_is_narrow_and_hidden(fixture):
    spec,calls,run=fixture
    value=module.WindowsOperations(spec,run=run)
    backup=Path(spec['root'])/'backups/fixture'
    backup.mkdir(parents=True)
    value.protect_backup(backup)
    args,options=calls[0]
    assert args[:2]==['icacls.exe',str(backup)]
    assert all('FixtureOwner' not in item for item in args)
    assert options['capture_output'] is True
    assert options['creationflags'] != 0


def test_acl_rejects_paths_outside_the_node(fixture,tmp_path):
    spec,calls,run=fixture
    value=module.WindowsOperations(spec,run=run)
    with pytest.raises(module.tx.Hold): value._acl(tmp_path/'outside')
    assert calls==[]


def test_only_hash_of_command_output_is_persisted(fixture):
    spec,calls,run=fixture
    value=module.WindowsOperations(spec,run=run)
    backup=Path(spec['root'])/'backups/fixture'
    backup.mkdir(parents=True)
    value.plan={'backup':str(backup)}
    value._command(['fixture-command'])
    records=list(backup.glob('command-*.json'))
    assert len(records)==1
    text=records[0].read_text()
    assert 'private-fixture-output' not in text and 'output_sha256' in text


@pytest.mark.parametrize('action',['Start','Stop'])
def test_service_uses_only_the_existing_hash_bound_helper(fixture,action):
    spec,calls,run=fixture
    value=module.WindowsOperations(spec,run=run)
    value._service(action)
    args=calls[0][0]
    assert str(module.SERVICE_HELPER) in args
    assert args[-4:]==['-Action',action,'-TimeoutMinutes','12']
    assert 'RotateSessionRestart' not in args


def test_rotation_or_helper_drift_never_dispatches(fixture):
    spec,calls,run=fixture
    value=module.WindowsOperations(spec,run=run)
    with pytest.raises(module.tx.Hold): value._service('RotateSessionRestart')
    Path(spec['service_action_helper']['path']).write_bytes(b'changed')
    with pytest.raises(module.tx.Hold): value._service('Start')
    assert calls==[]


def test_failed_command_does_not_print_its_raw_diagnostics(fixture,capsys):
    spec,calls,_run=fixture
    value=module.WindowsOperations(spec,run=lambda *a,**kw:SimpleNamespace(returncode=1,stdout=b'private-fixture-output',stderr=b'private-fixture-error'))
    with pytest.raises(module.tx.Hold,match='managed_command_failed'): value._command(['fixture'])
    assert capsys.readouterr().out==''
