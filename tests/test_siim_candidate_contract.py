import base64
import hashlib
from pathlib import Path
import pytest

from evomind_runtime.siim_candidate_contract import (
    candidate_kind, immutable_candidate_kind, controller_completed, admitted_metrics,
    reconcile_summaries, execution_receipt, dispatch_paths,
)

PROBE='''import numpy as np
class Dummy:
 def predict_proba(self, frame):
  p=np.full(len(frame),0.5)
  return np.column_stack([1-p,p])
def fit_model(x,y,xv,yv,seed):
 return Dummy()
'''
TRAINED='''import numpy as np
def fit_model(x,y,xv,yv,seed):
 model=LogisticRegression()
 model.fit(x,y)
 return model
'''


def metrics_fixture(tmp_path,source=TRAINED):
    root=tmp_path/'work/.calibration-control';root.mkdir(parents=True,exist_ok=True)
    attempt='a'*32
    (root/('attempt-'+attempt+'.py')).write_text('CANDIDATE='+repr(base64.b64encode(source.encode()).decode()),encoding='utf-8')
    return {'hpc_artifact_root':'/registered/case/'+attempt,'candidate_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'status':'completed','oof_complete':True,'oof_roc_auc':0.5,'reload_processes':6,
            'independent_saved_oof_recomputed':True}


def test_constant_probe_shape_is_diagnostic():
    assert candidate_kind(PROBE)=='diagnostic_constant_probe'
    assert candidate_kind(TRAINED)=='model_candidate'
    learned_prior=PROBE.replace('return Dummy()','unused=float(np.mean(y))\n return Dummy()')
    assert candidate_kind(learned_prior)=='model_candidate'


def test_low_score_alone_never_disqualifies_trained_candidate(tmp_path):
    metrics=metrics_fixture(tmp_path)
    assert metrics['oof_roc_auc']==0.5 and admitted_metrics(metrics,tmp_path)
    metrics=metrics_fixture(tmp_path,PROBE)
    assert not admitted_metrics(metrics,tmp_path)


def test_wrapper_hash_not_editable_solution_name_is_authoritative(tmp_path):
    metrics=metrics_fixture(tmp_path)
    assert immutable_candidate_kind(tmp_path,metrics)=='model_candidate'
    metrics['candidate_sha256']='0'*64
    with pytest.raises(ValueError,match='hash_changed'):
        immutable_candidate_kind(tmp_path,metrics)


@pytest.mark.parametrize('arm,status,expected',[
 ('evomind','blocked',False),('evomind','failed',False),('evomind','completed',True),
 ('fixed_baseline','completed',True),('fixed_baseline','running',False),
 ('aide','time_budget_reached',True),('aide','steps_completed',True),('aide','blocked',False)])
def test_controller_outcome_cannot_be_replaced_by_prior_success(arm,status,expected):
    assert controller_completed(arm,{'status':status}) is expected


def test_reconciliation_preserves_original_probe_score(tmp_path):
    metrics=metrics_fixture(tmp_path,PROBE)
    row={**metrics,'run_id':'run1','case_id':'evomind-seed17','arm':'evomind','seed':17}
    cases={'run1':{'workspace_root':str(tmp_path),'case_id':row['case_id'],'arm':'evomind','seed':17}}
    good,issues=reconcile_summaries([row],cases)
    assert good==[] and issues[0]['recorded_oof_auc_preserved']==0.5
    assert row['status']=='completed' and row['oof_roc_auc']==0.5


def test_exit_receipt_contains_code_and_hashes_not_raw_text():
    record=execution_receipt({'exit_code':137,'failure_type':'evidence','stderr_tail':'sensitive raw row'},20)
    assert record['exit_code']==137 and record['failure_type']=='evidence'
    assert record['stderr_bytes']>0 and 'sensitive' not in repr(record)
    assert execution_receipt({},1)['exit_code'] is None


def test_new_approved_operation_has_distinct_receipt_and_idempotency_key(tmp_path):
    parent='run_'+'a'*32
    old,key1=dispatch_paths(tmp_path,parent,'')
    new,key2=dispatch_paths(tmp_path,parent,'repair1')
    assert old!=new and key1!=key2
    assert (new,key2)==dispatch_paths(tmp_path,parent,'repair1')
    with pytest.raises(ValueError):dispatch_paths(tmp_path,parent,'../escape')


def test_suite_keeps_legacy_index_and_requires_controller_outcome():
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/siim_calibration_suite.py').read_text(encoding='utf-8')
    # Admission rests on hash-bound verified tool records (admitted_metrics), while the
    # controller outcome is still recorded and drives the deferral reason
    # (scripts/siim_calibration_suite.py admission comment: "widens evidence, not claims").
    assert 'controller_ok=controller_completed(' in source
    assert "'controller_not_completed' if not controller_ok" in source
    assert "'controller_completed':controller_ok" in source
    assert 'admitted_metrics(metrics,workspace)' in source
    assert "'admission_basis':'verified_candidate_evidence'" in source
    assert "save('case-results.json'" not in source
    assert "save('case-results-admitted-v2.json'" in source
