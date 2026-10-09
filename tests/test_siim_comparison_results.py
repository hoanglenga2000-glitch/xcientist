import pytest
from research_os.siim_comparison_results import ARMS,SEEDS,validate_completed_cases,select_by_oof,compare_frozen_scores

def fixture():
    protocol={'protocol_sha256':'a'*64,'data_manifest_sha256':'b'*64,'folds_sha256':'c'*64,'seconds_per_case':7200}
    cases=[dict(protocol,arm=a,seed=s,status='completed',oof_complete=True,official_score_seen_before_freeze=False,
                reload_processes=6,independent_saved_oof_recomputed=True,filesystem_isolation='landlock_abi5',private_answer_access_denied=True,
                human_modeling_interventions=0,oof_roc_auc=.7+s*.0001,runtime_seconds=600,submission_sha256=a+str(s),candidate_sha256=a+'source'+str(s)) for a in ARMS for s in SEEDS]
    return protocol,cases

def test_valid_complete_results_and_selection():
    p,cases=fixture();assert validate_completed_cases(cases,p)
    assert all(c['seed']==43 for c in select_by_oof(cases).values())

@pytest.mark.parametrize('field,value',[('status','running'),('oof_complete',False),('reload_processes',0),('independent_saved_oof_recomputed',False),('private_answer_access_denied',False),('runtime_seconds',7201),('protocol_sha256','d'*64),('human_modeling_interventions',1)])
def test_incomplete_or_mismatched_cases_rejected(field,value):
    p,cases=fixture();cases[3][field]=value
    with pytest.raises(ValueError):validate_completed_cases(cases,p)

def test_partial_cases_not_padded_with_zero():
    p,cases=fixture()
    with pytest.raises(ValueError):validate_completed_cases(cases[:-1],p)

def score_fixture():
    _,cases=fixture();selection=select_by_oof(cases)
    receipt={'status':'scored','competition':'siim-isic-melanoma-classification','metric':'roc_auc','evaluation_variant':'fixture',
        'scores':[dict(selection[a],score=s,test_answers_sha256='e'*64,test_rows=4142,test_labels_used_for_training=False) for a,s in zip(ARMS,[.8,.82,.81])]}
    return selection,receipt

def test_actual_score_differences_without_virtual_rank():
    selection,receipt=score_fixture();result=compare_frozen_scores(selection,receipt)
    assert result['evomind_minus_baseline']==pytest.approx(.02)
    assert result['evomind_minus_aide']==pytest.approx(.01)
    assert result['official_kaggle_rank'] is None

@pytest.mark.parametrize('field,value',[('test_rows',4000),('test_answers_sha256','z'*64),('submission_sha256','changed'),('candidate_sha256','changed'),('test_labels_used_for_training',True),('score',float('nan'))])
def test_invalid_score_receipt_rejected(field,value):
    selection,receipt=score_fixture();receipt['scores'][0][field]=value
    with pytest.raises(ValueError):compare_frozen_scores(selection,receipt)
