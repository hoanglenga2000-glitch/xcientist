import hashlib,importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('siim_private',Path(__file__).resolve().parents[1]/'scripts/siim_private_score_frozen.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture():
    return {'schema':'evomind.siim_private_score_freeze.v1','competition':'siim-isic-melanoma-classification',
        'all_candidates_frozen_before_test':True,'independent_verification_passed':True,'modeling_tasks_terminal':True,
        'private_feedback_used':False,'selected':{arm:{} for arm in ['fixed_baseline','evomind','aide']}}

def test_no_score_without_complete_pre_score_contract():
    p=fixture();raw=json.dumps(p).encode()
    assert set(module.validate_freeze(p,hashlib.sha256(raw).hexdigest(),raw))==set(p['selected'])

@pytest.mark.parametrize('field,value',[('all_candidates_frozen_before_test',False),('independent_verification_passed',False),('modeling_tasks_terminal',False),('private_feedback_used',True),('selected',{'evomind':{}})])
def test_bad_freeze_rejected_before_answer_access(field,value):
    p=fixture();p[field]=value;raw=json.dumps(p).encode()
    with pytest.raises(ValueError):module.validate_freeze(p,hashlib.sha256(raw).hexdigest(),raw)

def test_changed_freeze_rejected():
    p=fixture();raw=json.dumps(p).encode()
    with pytest.raises(ValueError):module.validate_freeze(p,'0'*64,raw)
