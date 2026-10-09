import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from research_os.siim_checkpoint_contract import inventory_complete_folds,validate_frozen_checkpoint,reconstruct_partial_oof

def fixture(tmp_path):
    root=tmp_path/'attempt';root.mkdir();code=b'def fit_model(*args): pass\n'
    expected={'arm':'fixed_baseline','seed':17,'candidate_sha256':hashlib.sha256(code).hexdigest(),'protocol_sha256':'a'*64,'data_manifest_sha256':'b'*64}
    (root/'task.json').write_text(json.dumps({**expected,'artifact_root':str(root)}))
    (root/'candidate.py').write_bytes(code);(root/'trusted-runner.py').write_text('# synthetic runner\n')
    (root/'fold-0.joblib').write_bytes(b'not-an-executable-pickle')
    np.save(root/'fold-0-reference.npy',np.array([.2,.8]),allow_pickle=False)
    event={'phase':'fold_fit','fold':0,'seed':17,'train_rows':8,'valid_rows':2,'auc':1.0,'ended':10.0}
    (root/'fit-events.jsonl').write_text(json.dumps(event)+'\n')
    return root,expected,event

def test_completed_fold_is_hashed_but_never_unpickled(tmp_path):
    root,expected,_=fixture(tmp_path);m=inventory_complete_folds(root,expected,tmp_path)
    assert m['remaining_folds']==[1,2,3,4]
    assert m['models_unpickled'] is False and m['ready_for_training'] is False
    assert validate_frozen_checkpoint(m,expected,root)

def test_tampered_checkpoint_rejected(tmp_path):
    root,expected,_=fixture(tmp_path);m=inventory_complete_folds(root,expected,tmp_path)
    (root/'fold-0.joblib').write_bytes(b'different-model')
    with pytest.raises(ValueError,match='changed_after_freeze'):validate_frozen_checkpoint(m,expected,root)

@pytest.mark.parametrize('field,value',[('seed',29),('fold',2),('auc',float('nan')),('valid_rows',3),('ended',-1)])
def test_bad_fold_receipt_rejected(tmp_path,field,value):
    root,expected,event=fixture(tmp_path);event[field]=value
    (root/'fit-events.jsonl').write_text(json.dumps(event)+'\n')
    with pytest.raises(ValueError):inventory_complete_folds(root,expected,tmp_path)

def test_truncated_fit_event_not_accepted(tmp_path):
    root,expected,event=fixture(tmp_path);(root/'fit-events.jsonl').write_text(json.dumps(event))
    with pytest.raises(ValueError,match='event_log_incomplete'):inventory_complete_folds(root,expected,tmp_path)

def test_wrong_candidate_identity_rejected(tmp_path):
    root,expected,_=fixture(tmp_path);expected['seed']=43
    with pytest.raises(ValueError,match='case_identity_mismatch'):inventory_complete_folds(root,expected,tmp_path)

def test_model_file_without_completed_event_not_reused(tmp_path):
    root,expected,_=fixture(tmp_path);(root/'fit-events.jsonl').write_text('')
    m=inventory_complete_folds(root,expected,tmp_path)
    assert m['complete_folds']==[] and m['remaining_folds']==list(range(5))

def test_nan_reference_is_rejected(tmp_path):
    root,expected,_=fixture(tmp_path);np.save(root/'fold-0-reference.npy',np.array([.2,np.nan]),allow_pickle=False)
    with pytest.raises(ValueError,match='reference_invalid'):inventory_complete_folds(root,expected,tmp_path)


def test_partial_reconstruction_never_fabricates_remaining_folds(tmp_path):
    import pandas as pd
    root,expected,_=fixture(tmp_path);m=inventory_complete_folds(root,expected,tmp_path)
    train=pd.DataFrame({'image_name':['x'+str(i) for i in range(10)],'target':[0,1]*5})
    folds=pd.DataFrame({'image_name':train.image_name,'fold':np.repeat(np.arange(5),2)})
    receipt,probabilities=reconstruct_partial_oof(m,expected,root,train,folds)
    assert receipt['complete_oof'] is False and receipt['complete_oof_auc'] is None
    assert receipt['oof_rows_covered']==2 and receipt['completed_folds'][0]['auc']==1.0
    assert np.isnan(probabilities[2:]).all()


def test_checkpoint_auc_must_match_frozen_reference(tmp_path):
    import pandas as pd
    root,expected,event=fixture(tmp_path);event['auc']=.5
    (root/'fit-events.jsonl').write_text(json.dumps(event)+'\n')
    m=inventory_complete_folds(root,expected,tmp_path)
    train=pd.DataFrame({'image_name':['x'+str(i) for i in range(10)],'target':[0,1]*5})
    folds=pd.DataFrame({'image_name':train.image_name,'fold':np.repeat(np.arange(5),2)})
    with pytest.raises(ValueError,match='auc_differs'):reconstruct_partial_oof(m,expected,root,train,folds)
