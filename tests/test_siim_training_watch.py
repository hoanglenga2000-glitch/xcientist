import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('siim_training_watch_under_test',ROOT/'scripts/check_siim_training_watch.py')
watch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def evidence(now=1000,steps=80,phase='fold_fit',fold=2,attempt='attempt1',observer='o1',reserved=6000):
    return {'application':{'parent':{'status':'running'},'children':[],
        'budget':{'pending_reserved_seconds':reserved,'remaining_seconds':60000,'settlement_attention':bool(reserved)}},
        'hpc_observation':{'at':datetime.fromtimestamp(now,timezone.utc).isoformat(),'observer_session':observer,
            'attempt_id':attempt,'read_only':True,'hpc_identity_samples':5,'case_id':'fixed_baseline-seed17',
            'observed':{'progress':{'phase':phase,'fold':fold},'optimizer_progress':{'optimizer_steps':steps},
                        'gpu_summary':'0, 8843, 81920','completed_fit_events':[],'checkpoints':[]}},
        'case_results':[],'selected_candidates':None}


def test_steps_increase_proves_progress_even_when_gpu_sample_is_zero():
    first=watch.analyze(evidence(),now=1000)
    current=watch.analyze(evidence(now=1300,steps=350,observer='o2'),first,now=1300)
    assert current['state']=='training_progress' and current['optimizer_delta']==270


def test_one_unchanged_sample_does_not_mean_stuck():
    first=watch.analyze(evidence(),now=1000)
    current=watch.analyze(evidence(now=1300,observer='o2'),first,now=1300)
    assert current['state']=='unchanged' and current['unchanged_samples']==1


def test_two_unchanged_distinct_samples_over_ten_minutes_trigger_investigation():
    first=watch.analyze(evidence(),now=1000)
    second=watch.analyze(evidence(now=1300,observer='o2'),first,now=1300)
    current=watch.analyze(evidence(now=1601,observer='o3'),second,now=1601)
    assert current['state']=='suspected_stall'
    assert not current['training_restarted']


@pytest.mark.parametrize('change',[{'phase':'resume_reload'},{'fold':3},{'attempt':'attempt2','steps':1}])
def test_stage_fold_or_attempt_transition_resets_stall_counter(change):
    previous=watch.analyze(evidence(),now=1000)
    previous['unchanged_samples']=4
    current=watch.analyze(evidence(now=1900,observer='o2',**change),previous,now=1900)
    assert current['state'] in {'training_progress','progress_observed'}
    assert current['unchanged_samples']==0


def test_stale_observation_is_not_live_training_or_proof_of_a_stall():
    current=watch.analyze(evidence(),now=2000)
    assert current['state']=='observation_unavailable'


def test_same_old_observer_cannot_count_as_two_independent_samples():
    previous=watch.analyze(evidence(),now=1000)
    current=watch.analyze(evidence(now=1300),previous,now=1300)
    assert current['unchanged_samples']==0


def test_no_fit_and_no_controller_progress_is_also_detected():
    value=evidence(reserved=0)
    value['hpc_observation']=None
    first=watch.analyze(value,now=1000)
    second=watch.analyze(value,first,now=1300)
    current=watch.analyze(value,second,now=1601)
    assert current['state']=='suspected_controller_stall'


def test_parent_complete_does_not_replace_training_evidence():
    value=evidence(reserved=0)
    value['application']['parent']['status']='completed'
    assert watch.analyze(value,now=1000)['state']=='parent_finished_but_training_not_verified'


def test_nine_valid_training_records_still_require_final_artifact_audit():
    value=evidence(reserved=0)
    value['case_results']=[{'case_id':arm+str(seed),'arm':arm,'seed':seed,'status':'completed',
                          'oof_complete':True,'independent_saved_oof_recomputed':True,'reload_processes':6}
                         for arm in ['fixed_baseline','evomind','aide'] for seed in [17,29,43]]
    value['selected_candidates']={'selected':True}
    assert watch.analyze(value,now=1000)['state']=='training_records_complete_pending_final_audit'
    value['case_results'][0]['reload_processes']=0
    assert watch.analyze(value,now=1000)['verified_complete_cases']==8


def test_previous_state_reads_chinese_paths_as_explicit_utf8(tmp_path, monkeypatch):
    target=tmp_path/'state.json'
    value={'evidence_file':'D:/桌面/codex/科研港科技/监督.json'}
    target.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    original=Path.read_text
    def locale_guard(path,*args,**kwargs):
        assert kwargs.get('encoding')=='utf-8'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',locale_guard)
    assert watch.previous_state(target)==value
    assert watch.previous_state(tmp_path/'not-created.json')=={}


def test_saved_final_model_is_verification_not_still_fitting():
    value=evidence(phase='final_fit')
    observed=value['hpc_observation']['observed']
    observed.update(final_metrics_present=True,checkpoints=[{'name':'final.joblib'}],
                    post_fit_reload_passed=1,reload_checks=[{'model':'fold-0','pid':100,'max_abs_diff':0}])
    first=watch.analyze(value,now=1000)
    assert first['phase']=='independent_reload' and first['state']=='verification_progress'
    value['hpc_observation'].update(at=datetime.fromtimestamp(1601,timezone.utc).isoformat(),observer_session='o2')
    observed['reload_checks'].append({'model':'fold-1','pid':101,'max_abs_diff':0})
    observed['post_fit_reload_passed']=2
    current=watch.analyze(value,first,now=1601)
    assert current['state']=='verification_progress' and current['optimizer_delta']==0
    assert current['post_fit_reload_passed']==2 and current['unchanged_samples']==0
    observed['post_fit_reload_passed']=6
    assert watch.analyze(value,current,now=1602)['phase']=='final_artifact_validation'


def test_known_dummy_probe_is_not_a_completed_training_case_but_low_auc_alone_is_not_rejected():
    value=evidence(reserved=0)
    row={'case_id':'evomind-seed17','arm':'evomind','seed':17,'status':'completed',
         'oof_roc_auc':0.5,'oof_complete':True,'independent_saved_oof_recomputed':True,
         'reload_processes':6,'candidate_sha256':'2b29e63a1a32083d684f819749780418d4ae18c7b70e458e7dadf49f9bf20b69'}
    value['case_results']=[row]
    current=watch.analyze(value,now=1000)
    assert current['recorded_completed_cases']==1 and current['verified_complete_cases']==0
    assert current['state']=='diagnostic_probe_misclassified_as_complete'
    row['candidate_sha256']='0'*64
    assert watch.analyze(value,now=1000)['verified_complete_cases']==1


def test_new_parent_resume_operation_is_not_a_stall_carried_from_pause():
    value=evidence(reserved=0)
    value['hpc_observation']=None
    value['application']['parent'].update(status='paused',execution={'calls':[{'id':'old','status':'failed'}]})
    old=watch.analyze(value,now=1000)
    old['unchanged_samples']=4
    value['application']['parent'].update(status='running',execution={'calls':[{'id':'new','status':'running'}]})
    current=watch.analyze(value,old,now=2000)
    assert current['state']=='progress_observed' and current['unchanged_samples']==0
