"""HPC-only, once-per-freeze MLE-bench scorer. Never called by a modeling agent.

The operator must first validate all case outcomes and obtain a fresh managed
identity receipt. This program scores only the exact three frozen predictions.
It sends no data to a model or external service.
"""
from __future__ import annotations
import argparse,hashlib,importlib,io,json,math,os,sys,time
from pathlib import Path

ROOT=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')
SOURCE=ROOT/'mle-bench'
DATA=ROOT/'mlebench_official_data/siim-isic-melanoma-classification'
ARTIFACTS=ROOT/'siim_calibration_results/siim_mlebench_calibration_20260908'
SOURCE_HASHES={
 'mlebench/grade.py':'cd4634bb08c78f13582405c30456cc69b9ac1a95c301e893ca37bc5c1fae988d',
 'mlebench/grade_helpers.py':'7d55512a893699b2e17041f3cd3bd0c2aba955c73f50872b3c69238546b87005',
 'mlebench/competitions/siim-isic-melanoma-classification/config.yaml':'532406c066bbd9e7d2bb71bd650eedf187865e70ac0fd70b71b0b4956ad65d64',
 'mlebench/competitions/siim-isic-melanoma-classification/prepare.py':'b3743d7f2c6a38f98588fa52edff977a80a7578b0c0f4deb99703c4ed01ce264',
 'mlebench/competitions/siim-isic-melanoma-classification/grade.py':'82d14d2462a89fd2704f81761466d246c19f79f364123946b1b348428ee3793d'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate_freeze(freeze,expected_hash,file_bytes):
    if hashlib.sha256(file_bytes).hexdigest()!=expected_hash:raise ValueError('freeze_hash_changed')
    if freeze.get('schema')!='evomind.siim_private_score_freeze.v1' or freeze.get('competition')!='siim-isic-melanoma-classification':raise ValueError('invalid_freeze')
    if freeze.get('all_candidates_frozen_before_test') is not True or freeze.get('independent_verification_passed') is not True:raise ValueError('pre_score_verification_required')
    if freeze.get('modeling_tasks_terminal') is not True or freeze.get('private_feedback_used') is not False:raise ValueError('modeling_not_closed')
    selected=freeze.get('selected')
    if not isinstance(selected,dict) or set(selected)!={'fixed_baseline','evomind','aide'}:raise ValueError('three_frozen_arms_required')
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);args=p.parse_args()
    freeze_path=Path(args.freeze).resolve();freeze_path.relative_to(ARTIFACTS)
    raw=freeze_path.read_bytes();freeze=json.loads(raw);selected=validate_freeze(freeze,args.freeze_sha256,raw)
    for name,h in SOURCE_HASHES.items():
        if sha(SOURCE/name)!=h:raise ValueError('upstream_grader_source_changed')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    import numpy as np,pandas as pd
    predictions={}
    test_ids=pd.read_csv(DATA/'prepared/public/test.csv')['image_name']
    for arm,candidate in selected.items():
        prediction=Path(candidate['submission_path']).resolve();prediction.relative_to(ARTIFACTS)
        if prediction.is_symlink() or sha(prediction)!=candidate['submission_sha256']:raise ValueError('frozen_prediction_changed')
        frame=pd.read_csv(prediction)
        if list(frame)!=['image_name','target'] or not frame.image_name.equals(test_ids) or len(frame)!=4142:raise ValueError('submission_alignment_failed')
        values=frame.target.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('submission_probability_invalid')
        predictions[arm]=frame
    output=freeze_path.parent/'private-score.json';claim=freeze_path.parent/'private-score-claim.json'
    if output.exists() or claim.exists():raise ValueError('private_score_already_claimed')
    # An interrupted claim is retained for reconciliation; never silently rescore.
    with claim.open('x') as f:json.dump({'freeze_sha256':args.freeze_sha256,'started_epoch':time.time(),'private_labels_will_be_read_once':True},f)
    answers_raw=(DATA/'prepared/private/test.csv').read_bytes()
    answers_hash=hashlib.sha256(answers_raw).hexdigest();answers=pd.read_csv(io.BytesIO(answers_raw))
    if len(answers)!=4142 or not answers.image_name.is_unique or set(answers.target)!={0,1}:raise ValueError('hidden_answer_contract_failed')
    sys.path.insert(0,str(SOURCE))
    upstream=importlib.import_module('mlebench.competitions.siim-isic-melanoma-classification.grade')
    scores=[]
    for arm,prediction in predictions.items():
        score=float(upstream.grade(prediction,answers))
        if not math.isfinite(score) or not 0<=score<=1:raise ValueError('official_grader_invalid_score')
        candidate=selected[arm]
        scores.append({'arm':arm,'score':score,'submission_sha256':candidate['submission_sha256'],
                       'candidate_sha256':candidate['candidate_sha256'],'test_answers_sha256':answers_hash,
                       'test_rows':len(answers),'test_labels_used_for_training':False})
    result={'schema':'evomind.siim_private_scoring_receipt.v1','status':'scored','competition':'siim-isic-melanoma-classification',
        'metric':'roc_auc','phase':'mlebench_hidden_test','evaluation_variant':freeze['evaluation_variant'],
        'scores':scores,'grader_source_hashes':SOURCE_HASHES,'freeze_sha256':args.freeze_sha256,
        'private_answer_file_reads':1,'official_kaggle_submission':False,'official_kaggle_rank':None,'completed_epoch':time.time()}
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
