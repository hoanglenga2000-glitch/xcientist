"""Offline policy/accounting tests; no model fitting, remote calls or GPU use."""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from evomind_runtime import ev_calibration_control as control


def fixture_context(tmp_path,monkeypatch):
    root=tmp_path/'case';(root/'inputs').mkdir(parents=True)
    (root/'inputs/task.json').write_text('{}');(root/'inputs/data-profile.json').write_text('{}')
    hashes={f'inputs/{n}':control.digest(root/'inputs'/n) for n in ['task.json','data-profile.json']}
    binding={'job_id':93207,'credential_profile':'fixture','allocation_generation':27,'profile_instance_id':'fixture','allocation_binding_id':'fixture'}
    p={'schema':'evomind.ev_calibration.runtime_policy.v1','campaign_id':control.CAMPAIGN,'enabled':True,
       'tenant_id':'tenant-fixture','owner_principal_id':'owner','managed_hpc_identity':binding,
       'cases':{'run-fixture':{'arm':'evomind','seed':17,'workspace_root':str(root),'input_hashes':hashes}}}
    path=tmp_path/'policy.json';path.write_text(json.dumps(p));monkeypatch.setattr(control,'POLICY_PATH',path)
    meta={'official_calibration':{'policy_sha256':control.digest(path)},'tenant_id':'tenant-fixture','owner_principal_id':'owner',
          'managed_hpc_identity':binding,'model_execution_contract':{'model':'deepseek-v4-pro','wire_protocol':'chat_completions_v1','endpoint_sha256':hashlib.sha256(b'https://api.pezayo.com/v1').hexdigest()},'run_allowed_tool_names':list(control.TOOLS)}
    return SimpleNamespace(session_id='run-fixture',workspace_root=root,metadata=meta)


def test_registered_context_and_summary_inputs(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch)
    control.guard_tool('file_read',{'path':'inputs/task.json'},context)
    control.guard_tool('file_write',{'path':'work/solutions/candidate.py'},context)


@pytest.mark.parametrize('name,args',[
    ('shell_exec',{'argv':['python','x.py']}),('kaggle_submit',{}),('memory_search',{}),
    ('file_read',{'path':'outputs/hpc/a/oof.npz'}),('file_read',{'path':'../other/training.csv'}),
    ('file_write',{'path':'inputs/task.json'}),('artifact_publish',{'path':'outputs/hpc/a/predictions.npz'})])
def test_forbidden_channels_fail(tmp_path,monkeypatch,name,args):
    context=fixture_context(tmp_path,monkeypatch)
    with pytest.raises(ValueError):control.guard_tool(name,args,context)


def test_tampered_input_or_binding_fails(tmp_path,monkeypatch):
    context=fixture_context(tmp_path,monkeypatch)
    (context.workspace_root/'inputs/task.json').write_text('{"changed":1}')
    with pytest.raises(ValueError):control.guard_tool('file_read',{'path':'inputs/task.json'},context)
    context.metadata['managed_hpc_identity']['job_id']=1
    with pytest.raises(ValueError):control.policy_for(context)


@pytest.mark.parametrize('text',[
    'import os\ndef fit_model(a,b,c,d,e): return os.listdir("/")',
    'import pandas as pd\ndef fit_model(a,b,c,d,e): return pd.read_csv("data.csv")',
    'def fit_model(a,b,c,d,e): return eval("1")',
    'from catboost import CatBoostClassifier\ndef fit_model(a,b,c,d,e): return CatBoostClassifier(thread_count=-1)',
    'import numpy as np\ndef fit_model(a,b,c,d,e): return np.load("x.npy")'])
def test_candidate_unsafe_source_is_rejected(text):
    with pytest.raises(ValueError):control.validate_source(text)


def test_fixed_baseline_fits_source_contract():
    path=Path(__file__).resolve().parents[1]/'scripts/ev_fixed_catboost_baseline.py'
    control.validate_source(path.read_text())


def test_budget_reserves_and_blocks_concurrent_work(tmp_path):
    ledger=control.CaseBudget(tmp_path/'budget.sqlite3');ledger.start_case('r')
    attempt,seconds=ledger.reserve('r','a'*64,10800)
    assert 10790<=seconds<=10800
    with pytest.raises(ValueError):ledger.reserve('r','b'*64,120)
    ledger.settle(attempt,30,True)
    assert ledger.summary()['charged_slot_seconds']==30


def test_uncertain_settlement_blocks_new_attempts(tmp_path):
    ledger=control.CaseBudget(tmp_path/'budget.sqlite3');ledger.start_case('r')
    attempt,seconds=ledger.reserve('r','a'*64,120)
    ledger.settle(attempt,5,False,uncertain=True)
    assert ledger.summary()['charged_slot_seconds']==120
    with pytest.raises(ValueError):ledger.reserve('r','a'*64,120)


def test_start_cannot_reset_case_clock(tmp_path):
    ledger=control.CaseBudget(tmp_path/'budget.sqlite3');ledger.start_case('r')
    with pytest.raises(ValueError):ledger.start_case('r')


def test_model_requests_stop_before_case_deadline(tmp_path,monkeypatch):
    ledger=control.CaseBudget(tmp_path/'budget.sqlite3');ledger.start_case('r')
    ledger.require_model_window('r')
    now=time.time();monkeypatch.setattr(control.time,'time',lambda:now+10700)
    with pytest.raises(RuntimeError):ledger.require_model_window('r')
