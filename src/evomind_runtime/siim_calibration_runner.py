"""Trusted HPC scorer. Candidate code receives fold-local frames, never network access."""
from __future__ import annotations
import argparse,base64,ctypes,errno,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_case(task,artifact_root):
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['CUDA_CACHE_DISABLE']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    cache=artifact_root/'cache';cache.mkdir(exist_ok=True)
    for name in ['MPLCONFIGDIR','XDG_CACHE_HOME','TMPDIR','TEMP','TMP','HF_HOME','TORCH_HOME']:
        os.environ[name]=str(cache)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    data=Path(task['persistent_data_root'])
    if not str(data).startswith('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/'):
        raise ValueError('data_root_rejected')
    manifest=data/'manifest.json'
    if digest(manifest)!=task['data_manifest_sha256']:raise ValueError('data_manifest_changed')
    info=json.loads(manifest.read_text())
    for name,row in info['files'].items():
        if digest(data/name)!=row['sha256']:raise ValueError('data_file_changed')
    import numpy as np,pandas as pd,joblib,torch,torchvision
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise ValueError('cuda_unavailable')
    for row in task['pretrained_weights']:
        if digest(Path(row['path']))!=row['sha256']:raise ValueError('pretrained_weights_changed')
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');folds=pd.read_csv(data/'frozen-folds.csv')
    if not np.array_equal(train.image_name.to_numpy(),folds.image_name.to_numpy()):raise ValueError('fold_alignment_failed')
    y=train.target.to_numpy(dtype=int)
    x=train.drop(columns=['image_name','patient_id','target']);xt=test.drop(columns=['image_name','patient_id'])
    code=artifact_root/'candidate.py'
    if digest(code)!=task['candidate_sha256']:raise ValueError('candidate_source_changed')
    helper=artifact_root/'isolation.py'
    if digest(helper)!=task['isolation_sha256']:raise ValueError('isolation_source_changed')
    isolated=importlib.util.spec_from_file_location('siim_isolation',helper)
    isolation=importlib.util.module_from_spec(isolated);isolated.loader.exec_module(isolation)
    # Init the driver before narrowing filesystem rights; all following children inherit the guard.
    torch.cuda.init()
    read=[str(data),str(Path(info['source_public_root'])/'jpeg/train'),str(Path(info['source_public_root'])/'jpeg/test'),sys.prefix,sys.base_prefix]
    read += [p for p in ['/usr','/lib','/lib64','/sys','/etc/ld.so.cache','/etc/localtime','/etc/passwd','/etc/nsswitch.conf','/proc/meminfo','/proc/cpuinfo','/proc/stat',f'/proc/{os.getpid()}'] if Path(p).exists()]
    read += [row['path'] for row in task['pretrained_weights']]
    devices=[str(p) for p in Path('/dev').glob('nvidia*') if p.exists()]
    devices += [p for p in ['/dev/null','/dev/zero','/dev/urandom','/dev/random'] if Path(p).exists()]
    write=[str(artifact_root),str(Path(task['output_dir'])),'/dev/shm']
    isolation.block_internet()
    isolation.restrict_filesystem(read,write,devices)
    prohibited=Path(info['source_public_root']).parent/'private/test.csv'
    try:
        probe=prohibited.open('rb')
    except PermissionError:
        pass
    else:
        probe.close();raise ValueError('private_answer_isolation_failed')
    spec=importlib.util.spec_from_file_location('siim_candidate',code)
    module=importlib.util.module_from_spec(spec);sys.modules['siim_candidate']=module;spec.loader.exec_module(module)
    return np,pd,joblib,roc_auc_score,info,train,test,folds,x,y,xt,module


def probabilities(model,features,np):
    classes=np.asarray(model.classes_)
    if classes.shape!=(2,) or set(classes.tolist())!={0,1}:raise ValueError('invalid_model_classes')
    p=np.asarray(model.predict_proba(features),dtype=float)
    if p.shape!=(len(features),2) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('invalid_model_probabilities')
    if not np.allclose(p.sum(axis=1),1,atol=1e-6,rtol=0):raise ValueError('probability_rows_invalid')
    return p[:,int(np.flatnonzero(classes==1)[0])]


def verify(task,root,model_key):
    np,pd,joblib,auc,info,train,test,folds,x,y,xt,module=load_case(task,root)
    model=joblib.load(root/(model_key+'.joblib'))
    features=xt if model_key=='final' else x.iloc[np.flatnonzero(folds.fold.to_numpy()==int(model_key.split('-')[1]))]
    actual=probabilities(model,features,np)
    expected=np.load(root/(model_key+'-reference.npy'),allow_pickle=False)
    difference=float(np.max(np.abs(actual-expected)))
    if difference>1e-6:raise ValueError('reload_prediction_mismatch')
    result={'status':'passed','model':model_key,'max_abs_diff':difference,'rows':len(actual),'pid':os.getpid()}
    if model_key!='final':
        mask=folds.fold.to_numpy()==int(model_key.split('-')[1])
        score=float(auc(y[mask],actual));reference=float(auc(y[mask],expected))
        if abs(score-reference)>1e-6:raise ValueError('reload_auc_mismatch')
        result.update(auc=score,auc_difference=abs(score-reference))
    (root/(model_key+'-reload.json')).write_text(json.dumps(result))


def stage_verified_checkpoints(task,root):
    import shutil
    checkpoint=task.get('resume_checkpoint')
    if not checkpoint:return []
    source=Path(checkpoint['source_artifact_root'])
    if source.resolve()!=source or source==root:raise ValueError('resume_source_path_invalid')
    source.relative_to(root.parent)
    for key in ['arm','seed','candidate_sha256','protocol_sha256','data_manifest_sha256']:
        if checkpoint['identity'].get(key)!=task.get(key):raise ValueError('resume_identity_mismatch')
    for name,key in [('task.json','task_sha256'),('trusted-runner.py','trusted_runner_sha256')]:
        if (source/name).is_symlink() or digest(source/name)!=checkpoint[key]:raise ValueError('resume_evidence_changed')
    if digest(source/'candidate.py')!=task['candidate_sha256']:raise ValueError('resume_candidate_changed')
    folds=[]
    for row in checkpoint['complete_folds']:
        fold=row['fold']
        if type(fold) is not int or fold!=len(folds) or fold not in range(5):raise ValueError('resume_fold_sequence_invalid')
        for kind,expected_name in [('model',f'fold-{fold}.joblib'),('reference',f'fold-{fold}-reference.npy')]:
            record=row[kind];path=source/expected_name
            if record['name']!=expected_name or path.is_symlink() or path.resolve().parent!=source:raise ValueError('resume_file_path_invalid')
            if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:raise ValueError('resume_file_changed')
            with path.open('rb') as old,(root/expected_name).open('xb') as new:shutil.copyfileobj(old,new)
            if digest(root/expected_name)!=record['sha256']:raise ValueError('resume_copy_mismatch')
        folds.append(fold)
    (root/'resume-lineage.json').write_text(json.dumps({'checkpoint':checkpoint,'models_refit':False,'copied_within_hpc':True},indent=2))
    return folds


def initialize_attempt(task,code_bytes,out):
    root=Path(task['artifact_root'])
    if not str(root).startswith('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/siim_calibration_results/'):
        raise ValueError('artifact_root_rejected')
    root.mkdir(parents=True,exist_ok=False)
    task={**task,'output_dir':str(out.resolve())}
    (root/'isolation.py').write_bytes(base64.b64decode(ISOLATION))
    (root/'candidate.py').write_bytes(code_bytes)
    (root/'task.json').write_text(json.dumps(task,sort_keys=True))
    # Persist the trusted runner separately so reload starts in a new interpreter.
    import shutil
    shutil.copyfile(__file__,root/'trusted-runner.py')
    return task,stage_verified_checkpoints(task,root)


def train_case(task,code_bytes,out):
    begin=time.monotonic();root=Path(task['artifact_root'])
    task=json.loads((root/'task.json').read_text())
    reused={row['fold']:row for row in (task.get('resume_checkpoint') or {}).get('complete_folds',[])}
    events=[]
    try:
        np,pd,joblib,auc,info,train,test,folds,x,y,xt,module=load_case(task,root)
        oof=np.full(len(train),np.nan);scores=[]
        log=root/'candidate-output.log'
        import contextlib
        for fold in range(5):
            valid=np.flatnonzero(folds.fold.to_numpy()==fold);training=np.flatnonzero(folds.fold.to_numpy()!=fold)
            if fold in reused:
                check=json.loads((root/(f'resume-fold-{fold}-reload.json')).read_text())
                if check.get('status')!='passed' or check['max_abs_diff']>1e-6:raise ValueError('resume_reload_not_verified')
                p=np.load(root/(f'fold-{fold}-reference.npy'),allow_pickle=False)
                if p.shape!=(len(valid),):raise ValueError('resume_validation_size_changed')
                value=float(auc(y[valid],p))
                if abs(value-reused[fold]['recorded_auc'])>1e-12:raise ValueError('resume_auc_changed')
                oof[valid]=p;scores.append(value)
                with (root/'fit-events.jsonl').open('a') as stream:stream.write(json.dumps({'phase':'fold_reused','fold':fold,'seed':task['seed'],'train_rows':len(training),'valid_rows':len(valid),'auc':value,'ended':time.monotonic()-begin,'new_fit':False})+'\n')
                continue
            event={'phase':'fold_fit','fold':fold,'seed':task['seed'],'train_rows':len(training),'valid_rows':len(valid),'started':time.monotonic()-begin}
            (root/'fit-progress.json').write_text(json.dumps({**event,'at_epoch':time.time(),'status':'fitting'}))
            with log.open('a') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                model=module.fit_model(x.iloc[training].copy(),y[training].copy(),x.iloc[valid].copy(),y[valid].copy(),task['seed'])
                p=probabilities(model,x.iloc[valid],np)
            oof[valid]=p;scores.append(float(auc(y[valid],p)))
            key='fold-'+str(fold);joblib.dump(model,root/(key+'.joblib'));np.save(root/(key+'-reference.npy'),p,allow_pickle=False)
            event.update(ended=time.monotonic()-begin,auc=scores[-1]);events.append(event)
            with (root/'fit-events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            del model
        if not np.isfinite(oof).all():raise ValueError('oof_incomplete')
        (root/'fit-progress.json').write_text(json.dumps({'phase':'final_fit','at_epoch':time.time(),'status':'fitting'}))
        with log.open('a') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            final=module.fit_model(x.copy(),y.copy(),None,None,task['seed'])
            ptest=probabilities(final,xt,np)
        joblib.dump(final,root/'final.joblib');np.save(root/'final-reference.npy',ptest,allow_pickle=False)
        events.append({'phase':'final_fit','seed':task['seed'],'train_rows':len(x),'ended':time.monotonic()-begin})
        with (root/'fit-events.jsonl').open('a') as stream:stream.write(json.dumps(events[-1])+'\n')
        del final
        np.savez_compressed(root/'oof.npz',id=train.image_name.to_numpy(dtype=str),target=y,fold=folds.fold.to_numpy(),probability=oof)
        np.savez_compressed(root/'test-predictions.npz',id=test.image_name.to_numpy(dtype=str),probability=ptest)
        pd.DataFrame({'image_name':test.image_name,'target':ptest}).to_csv(root/'submission.csv',index=False)
        for name,row in info['files'].items():
            if digest(Path(info['root'])/name)!=row['sha256']:raise ValueError('input_mutation_detected')
        elapsed=time.monotonic()-begin
        score=float(auc(y,oof))
        metrics={'schema':'evomind.siim_candidate_metrics.v1','status':'fit_complete_pending_reload','arm':task['arm'],'seed':task['seed'],
                 'oof_roc_auc':score,'fold_roc_auc':scores,'oof_complete':True,'runtime_seconds':elapsed,
                 'candidate_sha256':task['candidate_sha256'],'data_manifest_sha256':task['data_manifest_sha256'],
                 'folds_sha256':info['files']['frozen-folds.csv']['sha256'],'protocol_sha256':task['protocol_sha256'],
                 'train_rows':len(train),'test_rows':len(test),'fit_events':6-len(reused),'reused_folds':sorted(reused),'reload_processes':0,
                 'resource_limits':task.get('resource_limits',{}),
                 'filesystem_isolation':'landlock_abi5','private_answer_access_denied':True,'internet_socket_creation_blocked':True,'max_compute_threads':8,'cuda_visible_devices':'0',
                 'human_modeling_interventions':0 if task['arm'] in {'evomind','aide'} else None,'official_score_seen_before_freeze':False,
                 'official_score':None,'submission_sha256':digest(root/'submission.csv'),'hpc_artifact_root':str(root)}
        (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
        files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()}
        manifest={'schema':'evomind.siim_candidate_artifacts.v1','root':str(root),'files':files,'candidate_sha256':task['candidate_sha256']}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'status':'completed','oof_roc_auc':score,'runtime_seconds':elapsed}),flush=True)
        return 0
    except Exception as e:
        import traceback
        (root/'operator-error.txt').write_text(traceback.format_exc())
        safe={'status':'failed','error_type':type(e).__name__,'completed_fit_events':len(events),'runtime_seconds':time.monotonic()-begin,
              'hpc_artifact_root':str(root),'raw_outputs_withheld':True}
        (out/'failure.json').write_text(json.dumps(safe));print(json.dumps(safe),flush=True)
        return 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--out-dir');p.add_argument('--verify-root');p.add_argument('--model-key');p.add_argument('--fit-worker',action='store_true')
    args=p.parse_args()
    task=json.loads((Path(args.verify_root)/'task.json').read_text()) if args.verify_root else TASK
    required=task['python_executable']
    if str(Path(sys.prefix))!=str(Path(required).parent.parent):
        os.execv(required,[required,__file__,*sys.argv[1:]])
    if args.verify_root:
        root=Path(args.verify_root);verify(json.loads((root/'task.json').read_text()),root,args.model_key);return 0
    if args.fit_worker:return train_case(TASK,base64.b64decode(CANDIDATE),Path(args.out_dir))
    # The supervisor never imports agent code. Each fit/reload process independently
    # applies its filesystem guard, without inheriting a stale /proc/<parent> rule.
    begin=time.monotonic();out=Path(args.out_dir);root=Path(TASK['artifact_root'])
    try:
        task,reused=initialize_attempt(TASK,base64.b64decode(CANDIDATE),out)
        for fold in reused:
            key=f'fold-{fold}'
            (root/'fit-progress.json').write_text(json.dumps({'phase':'resume_reload','fold':fold,'at_epoch':time.time()}))
            remaining=max(1,int(TASK['worker_limit_seconds']-(time.monotonic()-begin)))
            with (root/'resume-reload-output.log').open('ab') as stream:
                reload=subprocess.run([sys.executable,str(root/'trusted-runner.py'),'--verify-root',str(root),'--model-key',key],stdout=stream,stderr=stream,timeout=min(remaining,600),close_fds=True)
            if reload.returncode:raise ValueError('resume_independent_reload_failed')
            import shutil
            shutil.copyfile(root/(key+'-reload.json'),root/('resume-'+key+'-reload.json'))
        with (out/'worker-process.log').open('ab') as stream:
            fit=subprocess.run([sys.executable,__file__,*sys.argv[1:],'--fit-worker'],stdout=stream,stderr=stream,timeout=max(1,int(TASK['worker_limit_seconds']-(time.monotonic()-begin))),close_fds=True)
        if fit.returncode:return fit.returncode
        metrics=json.loads((root/'metrics.json').read_text())
        for key in [*(f'fold-{i}' for i in range(5)),'final']:
            remaining=max(1,int(TASK['worker_limit_seconds']-(time.monotonic()-begin)))
            with (root/'reload-output.log').open('ab') as stream:
                result=subprocess.run([sys.executable,str(root/'trusted-runner.py'),'--verify-root',str(root),'--model-key',key],stdout=stream,stderr=stream,timeout=min(remaining,600),close_fds=True)
            if result.returncode:raise ValueError('independent_reload_failed')
        import numpy as np,pandas as pd
        from sklearn.metrics import roc_auc_score
        with np.load(root/'oof.npz',allow_pickle=False) as saved:
            truth=pd.read_csv(Path(TASK['persistent_data_root'])/'train.csv')['target'].to_numpy()
            if not np.array_equal(truth,saved['target']):raise ValueError('saved_target_mismatch')
            if abs(float(roc_auc_score(truth,saved['probability']))-metrics['oof_roc_auc'])>1e-6:raise ValueError('independent_auc_mismatch')
        metrics.update(status='completed',reload_processes=6,pre_resume_reload_processes=len(reused),runtime_seconds=time.monotonic()-begin,independent_saved_oof_recomputed=True)
        (root/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()}
        manifest={'schema':'evomind.siim_candidate_artifacts.v1','root':str(root),'files':files,'candidate_sha256':TASK['candidate_sha256']}
        (out/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'status':'completed','oof_roc_auc':metrics['oof_roc_auc'],'runtime_seconds':metrics['runtime_seconds']}));return 0
    except Exception as e:
        import traceback
        if root.is_dir():(root/'operator-error.txt').write_text(traceback.format_exc())
        (out/'failure.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'raw_outputs_withheld':True}))
        return 2


if __name__=='__main__':raise SystemExit(main())
