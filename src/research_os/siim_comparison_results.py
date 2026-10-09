"""Evidence-only SIIM result collection; no network, training or grader access."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path

ARMS=('fixed_baseline','evomind','aide')
SEEDS=(17,29,43)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def validate_completed_cases(cases,protocol,expected_cases=None):
    wanted=expected_cases or {(arm,seed) for arm in ARMS for seed in SEEDS}
    if len(cases)!=len(wanted) or {(c.get('arm'),c.get('seed')) for c in cases}!=wanted:
        raise ValueError('complete_distinct_planned_cases_required')
    required_hashes={'protocol_sha256':protocol['protocol_sha256'],'data_manifest_sha256':protocol['data_manifest_sha256'],'folds_sha256':protocol['folds_sha256']}
    for case in cases:
        if case.get('status')!='completed' or case.get('oof_complete') is not True:raise ValueError('incomplete_case')
        if any(case.get(k)!=v for k,v in required_hashes.items()):raise ValueError('case_protocol_identity_mismatch')
        if case.get('official_score_seen_before_freeze') is not False:raise ValueError('test_feedback_isolation_not_verified')
        if case.get('reload_processes')!=6 or case.get('independent_saved_oof_recomputed') is not True:raise ValueError('independent_verification_missing')
        if case.get('filesystem_isolation')!='landlock_abi5' or case.get('private_answer_access_denied') is not True:raise ValueError('private_answer_isolation_missing')
        if case['arm']!='fixed_baseline' and case.get('human_modeling_interventions')!=0:raise ValueError('manual_modeling_intervention')
        score=case.get('oof_roc_auc');elapsed=case.get('runtime_seconds')
        if not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('invalid_complete_oof_score')
        if not isinstance(elapsed,(float,int)) or not math.isfinite(elapsed) or not 0<=elapsed<=protocol['seconds_per_case']:raise ValueError('case_budget_exceeded')
    return True

def select_by_oof(cases):
    return {arm:min((c for c in cases if c['arm']==arm),key=lambda c:(-c['oof_roc_auc'],c['runtime_seconds'],c['seed'])) for arm in ARMS}

def compare_frozen_scores(selection,receipt):
    if receipt.get('status')!='scored' or receipt.get('competition')!='siim-isic-melanoma-classification' or receipt.get('metric')!='roc_auc':raise ValueError('wrong_scoring_receipt')
    scores=receipt.get('scores',[])
    if len(scores)!=3 or {s.get('arm') for s in scores}!=set(ARMS):raise ValueError('three_distinct_arm_scores_required')
    if len({s.get('test_answers_sha256') for s in scores})!=1 or not all(s.get('test_answers_sha256') for s in scores):raise ValueError('different_hidden_tests')
    by_arm={s['arm']:s for s in scores}
    for arm in ARMS:
        row=by_arm[arm];candidate=selection[arm]
        if row.get('submission_sha256')!=candidate.get('submission_sha256') or row.get('candidate_sha256')!=candidate.get('candidate_sha256'):raise ValueError('prediction_or_model_changed')
        if row.get('test_rows')!=4142 or row.get('test_labels_used_for_training') is not False:raise ValueError('hidden_test_protocol_mismatch')
        value=row.get('score')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('invalid_test_score')
    return {'metric':'roc_auc','test_rows':4142,'scores':{a:by_arm[a]['score'] for a in ARMS},
            'evomind_minus_baseline':by_arm['evomind']['score']-by_arm['fixed_baseline']['score'],
            'evomind_minus_aide':by_arm['evomind']['score']-by_arm['aide']['score'],
            'evaluation_variant':receipt['evaluation_variant'],'official_kaggle_rank':None}
