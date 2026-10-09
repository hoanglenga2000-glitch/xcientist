import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

spec=importlib.util.spec_from_file_location('ev_saved_validation',Path(__file__).resolve().parents[1]/'scripts/verify_ev_hpc_saved_predictions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def data():
    ids=np.arange(10);y=np.tile([0,1],5);folds=np.repeat(np.arange(5),2)
    test=np.array([10,11]);oof={'id':ids,'target':y,'fold':folds,'probability':np.tile([.2,.8],5)}
    pred={'id':test,'probability':np.array([.2,.8])}
    return [ids,y,ids,folds,test,oof,pred,test,pred['probability'],np,roc_auc_score]

def test_saved_predictions_recompute():
    result=module.validate_arrays(*data())
    assert result['oof_roc_auc']==1 and result['submission_npz_max_abs_diff']==0

@pytest.mark.parametrize('problem',['id','label','fold','nan','range','order','probability'])
def test_saved_prediction_corruption_is_rejected(problem):
    args=data()
    if problem=='id':args[5]['id']=args[0][::-1]
    elif problem=='label':args[5]['target']=1-args[1]
    elif problem=='fold':args[5]['fold']=args[3][::-1]
    elif problem=='nan':args[5]['probability'][0]=np.nan
    elif problem=='range':args[5]['probability'][0]=1.1
    elif problem=='order':args[7]=args[7][::-1]
    else:args[8]=np.array([.3,.8])
    with pytest.raises(ValueError):module.validate_arrays(*args)
