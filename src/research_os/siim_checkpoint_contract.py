"""Verify reusable complete-fold checkpoints without changing runs or budgets.

This module never imports a candidate, unpickles a model, opens private test
labels or starts training. A validated manifest is not an execution approval;
model reload still needs a fresh isolated process and available approved time.
"""
from __future__ import annotations
import hashlib,json,math,re
from pathlib import Path
from typing import Any

HEX=re.compile(r'[a-f0-9]{64}')
IDENTITY_KEYS=('arm','seed','candidate_sha256','protocol_sha256','data_manifest_sha256')

def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def bound_file(root: Path,name: str) -> Path:
    if Path(name).name!=name or name in {'','.','..'}:raise ValueError('checkpoint_filename_invalid')
    path=root/name
    if path.is_symlink() or path.resolve().parent!=root or not path.is_file():raise ValueError('checkpoint_file_missing_or_aliased')
    return path

def inventory_complete_folds(root: Path,expected: dict[str,Any],approved_artifact_root: Path) -> dict[str,Any]:
    original=Path(root)
    root=original.resolve(strict=True)
    if original.is_symlink():raise ValueError('checkpoint_root_aliased')
    root.relative_to(approved_artifact_root.resolve(strict=True))
    task_path=bound_file(root,'task.json')
    task=json.loads(task_path.read_text(encoding='utf-8'))
    if any(task.get(k)!=expected.get(k) for k in IDENTITY_KEYS):raise ValueError('checkpoint_case_identity_mismatch')
    if Path(task.get('artifact_root','')).resolve()!=root:raise ValueError('checkpoint_task_path_mismatch')
    source=bound_file(root,'candidate.py')
    if hash_file(source)!=task['candidate_sha256']:raise ValueError('checkpoint_candidate_source_changed')
    if any(not HEX.fullmatch(str(task[k])) for k in ['candidate_sha256','protocol_sha256','data_manifest_sha256']):raise ValueError('checkpoint_hash_invalid')
    runner=bound_file(root,'trusted-runner.py');runner_hash=hash_file(runner)
    if expected.get('trusted_runner_sha256') and runner_hash!=expected['trusted_runner_sha256']:raise ValueError('checkpoint_runner_changed')
    events_path=root/'fit-events.jsonl'
    if not events_path.exists():events=[]
    else:
        events_path=bound_file(root,'fit-events.jsonl')
        if events_path.stat().st_size>65536:raise ValueError('checkpoint_event_log_oversized')
        raw=events_path.read_text(encoding='utf-8')
        if raw and not raw.endswith('\n'):raise ValueError('checkpoint_event_log_incomplete')
        events=[json.loads(line) for line in raw.splitlines() if line.strip()]
    records=[];seen=set()
    import numpy as np
    for event in events:
        if event.get('phase') not in {'fold_fit','fold_reused'}:continue
        fold=event.get('fold')
        if type(fold) is not int or fold not in range(5) or fold in seen:raise ValueError('checkpoint_fold_duplicate_or_invalid')
        seen.add(fold)
        if fold!=len(records):raise ValueError('checkpoint_fold_order_gap')
        if event.get('seed')!=task['seed']:raise ValueError('checkpoint_fold_seed_mismatch')
        score=event.get('auc');ended=event.get('ended')
        if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('checkpoint_fold_score_invalid')
        if not isinstance(ended,(int,float)) or not math.isfinite(ended) or ended<0:raise ValueError('checkpoint_fold_end_invalid')
        model=bound_file(root,f'fold-{fold}.joblib');reference=bound_file(root,f'fold-{fold}-reference.npy')
        if event.get('phase')=='fold_reused':
            reload=json.loads(bound_file(root,f'resume-fold-{fold}-reload.json').read_text(encoding='utf-8'))
            if event.get('new_fit') is not False or reload.get('status')!='passed' or reload.get('max_abs_diff',float('inf'))>1e-6:
                raise ValueError('reused_checkpoint_reload_not_verified')
        if model.stat().st_size==0:raise ValueError('checkpoint_model_empty')
        probabilities=np.load(reference,allow_pickle=False)
        if probabilities.shape!=(event.get('valid_rows'),) or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():raise ValueError('checkpoint_reference_invalid')
        records.append({'fold':fold,'seed':task['seed'],'train_rows':event.get('train_rows'),'valid_rows':len(probabilities),
            'recorded_auc':score,'model':{'name':model.name,'bytes':model.stat().st_size,'sha256':hash_file(model)},
            'reference':{'name':reference.name,'bytes':reference.stat().st_size,'sha256':hash_file(reference)}})
    return {'schema':'evomind.siim_complete_fold_checkpoint_manifest.v1',
        'source_artifact_root':str(root),'identity':{k:task[k] for k in IDENTITY_KEYS},
        'trusted_runner_sha256':runner_hash,'task_sha256':hash_file(task_path),'complete_folds':records,
        'remaining_folds':[f for f in range(5) if f not in seen],'models_unpickled':False,
        'private_answers_read':False,'budget_reset':False,'ready_for_training':False,
        'next_required_check':'fresh isolated model reload plus approved remaining case and global time'}

def validate_frozen_checkpoint(manifest,expected,root):
    if manifest.get('schema')!='evomind.siim_complete_fold_checkpoint_manifest.v1':raise ValueError('checkpoint_manifest_schema_invalid')
    if manifest.get('identity')!={k:expected[k] for k in IDENTITY_KEYS}:raise ValueError('checkpoint_resume_identity_mismatch')
    root=Path(root).resolve(strict=True)
    if str(root)!=manifest.get('source_artifact_root'):raise ValueError('checkpoint_resume_root_changed')
    for record in manifest['complete_folds']:
        for field in ['model','reference']:
            row=record[field];path=bound_file(root,row['name'])
            if path.stat().st_size!=row['bytes'] or hash_file(path)!=row['sha256']:raise ValueError('checkpoint_changed_after_freeze')
    return True


def reconstruct_partial_oof(manifest,expected,root,training,folds):
    """Recompute completed-fold AUC from frozen training labels, never private labels.

    Missing folds remain NaN and complete_oof_auc stays None until all five
    folds exist. This does not run or reload any model or authorize a resume.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    validate_frozen_checkpoint(manifest,expected,root)
    if not training.image_name.equals(folds.image_name):raise ValueError('checkpoint_fold_ids_differ')
    if not training.image_name.is_unique or set(training.target)!={0,1}:raise ValueError('checkpoint_training_domain_invalid')
    if set(folds.fold)!=set(range(5)):raise ValueError('checkpoint_five_folds_required')
    result=np.full(len(training),np.nan);verified=[]
    for record in manifest['complete_folds']:
        mask=folds.fold.to_numpy()==record['fold']
        probabilities=np.load(bound_file(Path(root).resolve(),record['reference']['name']),allow_pickle=False)
        if int(mask.sum())!=record['valid_rows'] or len(training)-int(mask.sum())!=record['train_rows']:raise ValueError('checkpoint_fold_row_counts_differ')
        value=float(roc_auc_score(training.target.to_numpy()[mask],probabilities))
        if abs(value-record['recorded_auc'])>1e-12:raise ValueError('checkpoint_recorded_auc_differs')
        result[mask]=probabilities
        verified.append({'fold':record['fold'],'auc':value,'absolute_difference':abs(value-record['recorded_auc']),'rows':int(mask.sum())})
    complete=bool(np.isfinite(result).all())
    return {'completed_folds':verified,'complete_oof':complete,
            'complete_oof_auc':float(roc_auc_score(training.target,result)) if complete else None,
            'oof_rows_covered':int(np.isfinite(result).sum()),'oof_rows_expected':len(training),
            'private_labels_read':False,'model_fits':0,'missing_predictions_fabricated':False},result
