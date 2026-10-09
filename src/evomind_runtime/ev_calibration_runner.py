"""Trusted HPC scorer. Candidate code receives fold-local frames, never network access."""
from __future__ import annotations
import argparse,base64,ctypes,errno,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def block_internet():
    """Deny creation of IPv4/IPv6 sockets, inherited by verification children."""
    class Comparison(ctypes.Structure):
        _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Comparison)]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    context=lib.seccomp_init(0x7fff0000)
    if not context:raise RuntimeError('network_filter_creation_failed')
    try:
        number=lib.seccomp_syscall_resolve_name(b'socket')
        for family in (2,10):
            comparison=Comparison(0,4,family,0)
            if lib.seccomp_rule_add_array(context,0x00050000|errno.EPERM,number,1,ctypes.byref(comparison))!=0:
                raise RuntimeError('network_filter_rule_failed')
        if lib.seccomp_load(context)!=0:raise RuntimeError('network_filter_activation_failed')
    finally:lib.seccomp_release(context)
    import socket
    for family in (socket.AF_INET,socket.AF_INET6):
        try:
            attempted=socket.socket(family,socket.SOCK_STREAM)
        except PermissionError:
            continue
        else:
            attempted.close();raise RuntimeError('network_filter_probe_failed')


def load_case(task,artifact_root):
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[name]='8'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    allowed=sorted(os.sched_getaffinity(0))[:8]
    os.sched_setaffinity(0,allowed)
    data=Path(task['persistent_data_root'])
    if not str(data).startswith('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/'):
        raise ValueError('data_root_rejected')
    manifest=data/'.evomind/data-manifest.json'
    if digest(manifest)!=task['data_manifest_sha256']:raise ValueError('data_manifest_changed')
    info=json.loads(manifest.read_text())
    for name,row in info['files'].items():
        if digest(data/name)!=row['sha256']:raise ValueError('data_file_changed')
    sys.path.insert(0,info['dependency_root'])
    import importlib.metadata
    if any(importlib.metadata.version(name)!=version for name,version in info['dependencies'].items()):
        raise ValueError('dependency_version_changed')
    import numpy as np,pandas as pd,joblib
    from sklearn.metrics import roc_auc_score
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');folds=pd.read_csv(data/'frozen-folds.csv')
    if not np.array_equal(train.id.to_numpy(),folds.id.to_numpy()):raise ValueError('fold_alignment_failed')
    y=train.Will_Buy_EV.map({'No':0,'Yes':1}).to_numpy()
    x=train.drop(columns=['id','Will_Buy_EV']);xt=test.drop(columns=['id'])
    code=artifact_root/'candidate.py'
    if digest(code)!=task['candidate_sha256']:raise ValueError('candidate_source_changed')
    spec=importlib.util.spec_from_file_location('ev_candidate',code)
    module=importlib.util.module_from_spec(spec);sys.modules['ev_candidate']=module
    block_internet()
    spec.loader.exec_module(module)
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


def train_case(task,code_bytes,out):
    begin=time.monotonic()
    root=Path(task['artifact_root'])
    if not str(root).startswith('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/ev_calibration_results/'):
        raise ValueError('artifact_root_rejected')
    root.mkdir(parents=True,exist_ok=False)
    (root/'candidate.py').write_bytes(code_bytes)
    (root/'task.json').write_text(json.dumps(task,sort_keys=True))
    # Persist the trusted runner separately so reload starts in a new interpreter.
    import shutil
    shutil.copyfile(__file__,root/'trusted-runner.py')
    events=[]
    try:
        np,pd,joblib,auc,info,train,test,folds,x,y,xt,module=load_case(task,root)
        oof=np.full(len(train),np.nan);scores=[]
        log=root/'candidate-output.log'
        import contextlib
        for fold in range(5):
            valid=np.flatnonzero(folds.fold.to_numpy()==fold);training=np.flatnonzero(folds.fold.to_numpy()!=fold)
            event={'phase':'fold_fit','fold':fold,'seed':task['seed'],'train_rows':len(training),'valid_rows':len(valid),'started':time.monotonic()-begin}
            with log.open('a') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                model=module.fit_model(x.iloc[training].copy(),y[training].copy(),x.iloc[valid].copy(),y[valid].copy(),task['seed'])
                p=probabilities(model,x.iloc[valid],np)
            oof[valid]=p;scores.append(float(auc(y[valid],p)))
            key='fold-'+str(fold);joblib.dump(model,root/(key+'.joblib'));np.save(root/(key+'-reference.npy'),p,allow_pickle=False)
            event.update(ended=time.monotonic()-begin,auc=scores[-1]);events.append(event)
            with (root/'fit-events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            del model
        if not np.isfinite(oof).all():raise ValueError('oof_incomplete')
        with log.open('a') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            final=module.fit_model(x.copy(),y.copy(),None,None,task['seed'])
            ptest=probabilities(final,xt,np)
        joblib.dump(final,root/'final.joblib');np.save(root/'final-reference.npy',ptest,allow_pickle=False)
        events.append({'phase':'final_fit','seed':task['seed'],'train_rows':len(x),'ended':time.monotonic()-begin})
        with (root/'fit-events.jsonl').open('a') as stream:stream.write(json.dumps(events[-1])+'\n')
        del final
        np.savez_compressed(root/'oof.npz',id=train.id.to_numpy(),target=y,fold=folds.fold.to_numpy(),probability=oof)
        np.savez_compressed(root/'test-predictions.npz',id=test.id.to_numpy(),probability=ptest)
        pd.DataFrame({'id':test.id,'Will_Buy_EV':ptest}).to_csv(root/'submission.csv',index=False)
        for key in [*(f'fold-{i}' for i in range(5)),'final']:
            remaining=max(1,int(task['worker_limit_seconds']-(time.monotonic()-begin)))
            with (root/'reload-output.log').open('ab') as stream:
                result=subprocess.run([sys.executable,str(root/'trusted-runner.py'),'--verify-root',str(root),'--model-key',key],stdout=stream,stderr=stream,timeout=min(remaining,600),close_fds=True)
            if result.returncode:raise ValueError('independent_reload_failed')
        for name,row in info['files'].items():
            if digest(Path(info['root'])/name)!=row['sha256']:raise ValueError('input_mutation_detected')
        elapsed=time.monotonic()-begin
        score=float(auc(y,oof))
        metrics={'schema':'evomind.ev_candidate_metrics.v1','status':'completed','arm':task['arm'],'seed':task['seed'],
                 'oof_roc_auc':score,'fold_roc_auc':scores,'oof_complete':True,'runtime_seconds':elapsed,
                 'candidate_sha256':task['candidate_sha256'],'data_manifest_sha256':task['data_manifest_sha256'],
                 'folds_sha256':info['files']['frozen-folds.csv']['sha256'],'protocol_sha256':task['protocol_sha256'],
                 'train_rows':len(train),'test_rows':len(test),'fit_events':6,'reload_processes':6,
                 'internet_socket_creation_blocked':True,'max_compute_threads':8,'cuda_visible_devices':'0',
                 'human_modeling_interventions':0 if task['arm']=='evomind' else None,'official_score_seen_before_freeze':False,
                 'official_score':None,'submission_sha256':digest(root/'submission.csv'),'hpc_artifact_root':str(root)}
        (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
        files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()}
        manifest={'schema':'evomind.ev_candidate_artifacts.v1','root':str(root),'files':files,'candidate_sha256':task['candidate_sha256']}
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
    p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--out-dir');p.add_argument('--verify-root');p.add_argument('--model-key')
    args=p.parse_args()
    if args.verify_root:
        root=Path(args.verify_root);verify(json.loads((root/'task.json').read_text()),root,args.model_key);return 0
    # TASK and CANDIDATE are injected by the server-owned controller, not by the LLM.
    return train_case(TASK,base64.b64decode(CANDIDATE),Path(args.out_dir))


if __name__=='__main__':raise SystemExit(main())
