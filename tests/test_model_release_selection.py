import copy
import pytest
from scripts.model_release_selection import qualifies,select,PRIMARY,RESERVE

CANDIDATE='a'*64

def report(model):
    return {'schema':'evomind.model_service_endurance.v1','model':model,'status':'completed',
        'service_soak_passed':True,'runtime_closed_cleanly':True,'preflight_only':False,
        'elapsed_seconds':5400,'required_seconds':5400,'injected_faults':0,'gpu_actions':0,
        'candidate_manifest_sha256':CANDIDATE,'route':'configured_upstream','wire_protocol':'responses',
        'reasoning_effort':'low','service_tier':'omit','native_tools_executed':60,
        'cases':[{'run_id':f'run_{i}','index':i,'passed':True,'status':'completed','case_timeout':False,
            'tool_evidence':{'successful_tool_count':3,'required_tool_steps_passed':True,
                'hash_check_passed':True,'readback_passed':True}} for i in range(20)]}

def test_prefers_sol_only_after_qualification():
    assert select(report(PRIMARY),report(RESERVE),CANDIDATE)['selected_candidate_model']==PRIMARY
    pending=report(PRIMARY);pending['status']='running'
    assert select(pending,report(RESERVE),CANDIDATE)['selected_candidate_model']==RESERVE

def test_no_qualified_model_is_hold():
    assert select(None,None,CANDIDATE)['status']=='HOLD_no_qualified_model'

@pytest.mark.parametrize('field,value',[('elapsed_seconds',5399),('elapsed_seconds',True),('elapsed_seconds',float('nan')),
    ('candidate_manifest_sha256','b'*64),('model',RESERVE),('injected_faults',1),('preflight_only',True),
    ('native_tools_executed',59),('runtime_closed_cleanly',False),('service_soak_passed','true')])
def test_invalid_primary_cannot_displace_reserve(field,value):
    primary=report(PRIMARY);primary[field]=value
    assert not qualifies(primary,PRIMARY,CANDIDATE)
    assert select(primary,report(RESERVE),CANDIDATE)['selected_candidate_model']==RESERVE

def test_failed_case_and_duplicate_identity_are_not_passes():
    primary=report(PRIMARY)
    primary['cases'][0]['passed']=False
    assert not qualifies(primary,PRIMARY,CANDIDATE)
    primary=report(PRIMARY);primary['cases'][1]=copy.deepcopy(primary['cases'][0])
    assert not qualifies(primary,PRIMARY,CANDIDATE)

def test_selection_never_claims_live_worker_failover_or_deployment():
    result=select(report(PRIMARY),report(RESERVE),CANDIDATE)
    assert result['production_changed'] is False
    assert result['release_verdict'].startswith('HOLD')
    for key in ['silent_in_run_switch','cancel_existing_worker','resubmit_training','reset_budget',
                'authentication_or_permission_bypass','automatic_live_failover_verified']:
        assert result['failover_contract'][key] is False
