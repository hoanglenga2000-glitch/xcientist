"""Independent read-only saved-prediction verifier, intended for the managed HPC.

This script never fits or reloads a model and has no submission/network code.
The operator must obtain the complete current identity gate before dispatch.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

DATA=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/ev_public_calibration_20260908')
RESULTS=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/ev_calibration_results/ev_public_calibration_20260908')
EXPECTED_MANIFEST='b70258d89f57033b050728f9f929366d76ad8e9eae220907197b478b16011366'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def validate_arrays(train_ids,y,fold_ids,fold_values,test_ids,oof,test_prob,submission_ids,submission_prob,np,auc):
    if not np.array_equal(oof['id'],train_ids) or not np.array_equal(oof['target'],y):raise ValueError('oof_target_or_id_mismatch')
    if not np.array_equal(fold_ids,train_ids) or not np.array_equal(oof['fold'],fold_values):raise ValueError('oof_fold_mismatch')
    if not np.array_equal(test_prob['id'],test_ids) or not np.array_equal(submission_ids,test_ids):raise ValueError('submission_id_order_mismatch')
    for values,expected_rows in [(oof['probability'],len(train_ids)),(test_prob['probability'],len(test_ids)),(submission_prob,len(test_ids))]:
        values=np.asarray(values)
        if values.shape!=(expected_rows,) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError('invalid_saved_probability')
    difference=float(np.max(np.abs(submission_prob-test_prob['probability'])))
    if difference>1e-6:raise ValueError('submission_probability_mismatch')
    if set(np.unique(fold_values))!=set(range(5)):raise ValueError('invalid_fold_set')
    return {'oof_roc_auc':float(auc(y,oof['probability'])),
            'fold_roc_auc':[float(auc(y[fold_values==fold],oof['probability'][fold_values==fold])) for fold in range(5)],
            'submission_npz_max_abs_diff':difference,'train_rows':len(train_ids),'test_rows':len(test_ids)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root).resolve();relative=root.relative_to(RESULTS.resolve())
    if len(relative.parts)!=2:raise ValueError('candidate_root_rejected')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='8'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    manifest=DATA/'.evomind/data-manifest.json'
    if sha(manifest)!=EXPECTED_MANIFEST:raise ValueError('data_manifest_changed')
    info=json.loads(manifest.read_text())
    for name,row in info['files'].items():
        if sha(DATA/name)!=row['sha256']:raise ValueError('data_changed')
    sys.path.insert(0,str(DATA/'.evomind/preinstalled-only'))
    import numpy as np,pandas as pd
    from sklearn.metrics import roc_auc_score
    metric=json.loads((root/'metrics.json').read_text())
    if metric['data_manifest_sha256']!=EXPECTED_MANIFEST or metric['candidate_sha256']!=sha(root/'candidate.py'):
        raise ValueError('candidate_identity_mismatch')
    train=pd.read_csv(DATA/'train.csv',usecols=['id','Will_Buy_EV']);test=pd.read_csv(DATA/'test.csv',usecols=['id'])
    folds=pd.read_csv(DATA/'frozen-folds.csv');submission=pd.read_csv(root/'submission.csv')
    if list(submission)!=['id','Will_Buy_EV']:raise ValueError('submission_columns_mismatch')
    y=train.Will_Buy_EV.map({'No':0,'Yes':1}).to_numpy()
    with np.load(root/'oof.npz',allow_pickle=False) as oof,np.load(root/'test-predictions.npz',allow_pickle=False) as test_prob:
        result=validate_arrays(train.id.to_numpy(),y,folds.id.to_numpy(),folds.fold.to_numpy(),test.id.to_numpy(),
            oof,test_prob,submission.id.to_numpy(),submission.Will_Buy_EV.to_numpy(),np,roc_auc_score)
    delta=abs(result['oof_roc_auc']-metric['oof_roc_auc'])
    if delta>1e-6 or max(abs(a-b) for a,b in zip(result['fold_roc_auc'],metric['fold_roc_auc']))>1e-6:
        raise ValueError('saved_auc_mismatch')
    if sha(root/'submission.csv')!=metric['submission_sha256']:raise ValueError('submission_hash_mismatch')
    reloads=[json.loads((root/(key+'-reload.json')).read_text()) for key in [*(f'fold-{i}' for i in range(5)),'final']]
    if any(r['status']!='passed' or r['max_abs_diff']>1e-6 or r.get('auc_difference',0)>1e-6 for r in reloads):
        raise ValueError('reload_check_failed')
    result.update(status='passed',arm=metric['arm'],seed=metric['seed'],oof_auc_abs_diff=delta,
                  submission_sha256=metric['submission_sha256'],candidate_sha256=metric['candidate_sha256'],
                  reload_processes=len(reloads),max_reload_probability_difference=max(r['max_abs_diff'] for r in reloads),
                  read_only=True,model_fitting=False,raw_predictions_returned=False)
    print(json.dumps(result))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'failed','error_type':type(e).__name__,'raw_text_withheld':True}));raise SystemExit(2)
