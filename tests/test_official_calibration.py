"""Synthetic fixtures validate contracts; none of these values are measured results."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from research_os.official_calibration import (
    build_frozen_folds, compare_official_scores, freeze_protocol, remaining_gpu_hours,
    select_frozen_candidate, validate_data_frames,
)

ROOT=Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT/'configs/benchmarks/ev_public_calibration_20260908.json').read_text())


def frames():
    train=pd.DataFrame({'id':range(20),'x':range(20),'Will_Buy_EV':[0,1]*10})
    test=pd.DataFrame({'id':[21,22],'x':[1,2]})
    sample=pd.DataFrame({'id':[21,22],'Will_Buy_EV':[0,0]})
    return train,test,sample


def test_protocol_is_exclusive(tmp_path):
    assert len(freeze_protocol(protocol(),tmp_path/'protocol.json'))==64
    with pytest.raises(FileExistsError):freeze_protocol(protocol(),tmp_path/'protocol.json')


def test_data_boundary_and_reproducible_folds():
    train,test,sample=frames();p=protocol()
    assert validate_data_frames(train,test,sample,p)['test_has_no_labels']
    first=build_frozen_folds(train,p);second=build_frozen_folds(train,p)
    pd.testing.assert_frame_equal(first,second)
    assert first.id.is_unique and first.fold.value_counts().to_dict()=={0:4,1:4,2:4,3:4,4:4}
    for fold in range(5):
        assert set(first.loc[first.fold==fold,'id']).isdisjoint(set(first.loc[first.fold!=fold,'id']))


def test_yes_no_official_labels_have_explicit_positive_encoding():
    train,test,sample=frames()
    expected=build_frozen_folds(train,protocol())
    train['Will_Buy_EV']=train['Will_Buy_EV'].map({0:'No',1:'Yes'})
    assert validate_data_frames(train,test,sample,protocol())['target_encoding']=={'No':0,'Yes':1}
    pd.testing.assert_frame_equal(build_frozen_folds(train,protocol()),expected)
    assert set(train.Will_Buy_EV)=={'No','Yes'}


@pytest.mark.parametrize('bad',['overlap','order','test_labels','missing_target'])
def test_data_corruption_is_rejected(bad):
    train,test,sample=frames()
    if bad=='overlap':test.loc[0,'id']=0
    elif bad=='order':sample=sample.iloc[::-1]
    elif bad=='test_labels':test['Will_Buy_EV']=0
    else:train.loc[0,'Will_Buy_EV']=float('nan')
    with pytest.raises(ValueError):validate_data_frames(train,test,sample,protocol())


def runs():
    return [{'arm':'evomind','seed':s,'status':'completed','oof_complete':True,'oof_roc_auc':.8,
             'runtime_seconds':t,'official_score_seen_before_freeze':False,'human_modeling_interventions':0,
             'protocol_sha256':'a'*64,'data_manifest_sha256':'b'*64,'folds_sha256':'c'*64}
            for s,t in [(17,30),(29,20),(43,40)]]


def test_frozen_selection_uses_oof_then_runtime():
    assert select_frozen_candidate(runs(),'evomind')['seed']==29


@pytest.mark.parametrize('field,value',[('oof_complete',False),('human_modeling_interventions',1),
                                      ('official_score_seen_before_freeze',True),('runtime_seconds',10801)])
def test_selection_rejects_invalid_runs(field,value):
    rows=runs();rows[0][field]=value
    with pytest.raises(ValueError):select_frozen_candidate(rows,'evomind')


def test_selection_rejects_different_frozen_fold_identity():
    rows=runs();rows[0]['folds_sha256']='d'*64
    with pytest.raises(ValueError,match='identity'):
        select_frozen_candidate(rows,'evomind')


def test_budget_charges_unsettled_reservations_and_failures():
    entries=[{'attempt_id':'a','settled':True,'actual_gpu_hours':21},
             {'attempt_id':'b','settled':False,'reserved_gpu_hours':3}]
    assert remaining_gpu_hours(entries)==0
    with pytest.raises(ValueError):remaining_gpu_hours(entries,.01)
    with pytest.raises(ValueError):remaining_gpu_hours(entries+[entries[0]])


def official_fixture():
    snapshot={'competition':'playground-series-s6e9','phase':'public','metric':'roc_auc',
              'captured_at':'synthetic-fixture','source_url':'https://example.invalid/fixture',
              'complete':True,'team_count':4,
              'rows':[{'team_id':str(i),'team':'fixture-'+str(i),'rank':i,'score':s} for i,s in enumerate([.9,.8,.7,.6],1)]}
    receipt={k:snapshot[k] for k in ['competition','phase','metric']}
    receipt.update(status='scored',source_url='https://example.invalid/fixture',prediction_sha256='a'*64,
                   frozen_prediction_sha256='a'*64)
    return snapshot,{**receipt,'submission_ref':'base','score':.7},{**receipt,'submission_ref':'ev','score':.8,'team_id':'2'}


def test_official_comparison_does_not_invent_rank():
    s,b,c=official_fixture();r=compare_official_scores(s,b,c)
    assert r['delta_auc']==pytest.approx(.1) and r['top25_boundary_score']==.9
    assert r['median_team_score']==pytest.approx(.75) and r['official_candidate_rank'] is None
    c['team_best_submission_ref']='ev'
    assert compare_official_scores(s,b,c)['official_candidate_rank']==2


@pytest.mark.parametrize('bad',['private','missing_rows','duplicate_team','prediction_changed','same_receipt','wrong_order'])
def test_bad_comparisons_fail_closed(bad):
    s,b,c=official_fixture()
    if bad=='private':c['phase']='private'
    elif bad=='missing_rows':s['rows'].pop()
    elif bad=='duplicate_team':s['rows'][1]['team_id']='1'
    elif bad=='prediction_changed':c['prediction_sha256']='b'*64
    elif bad=='same_receipt':c['submission_ref']='base'
    else:s['rows'][0]['score']=.5
    with pytest.raises(ValueError):compare_official_scores(s,b,c)
