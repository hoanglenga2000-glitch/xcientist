"""Trusted data-only worker. Official HTTPS download, exact hashes, no model fitting."""
from __future__ import annotations
import contextlib,hashlib,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path

EXPECTED={
 'train.csv':'eae9eaa4e6378df405e755f853771d7e26d212bd93258349fc797b771021946a',
 'test.csv':'539263f6caabc40afd5e2f0bc0ab16b10a2d1177c565fc71b866f0181d836b34',
 'sample_submission.csv':'a9747a8b947e4e35505e3da4535a5a494978b012a7adb50973f13e598849dda5',
 'frozen-folds.csv':'0333a344834f5900051dc8360e908ac7102f27fc5cf31ff4121c73fb67343a28'}
ROOT=Path(os.environ['EVOMIND_COMPETITION_DATA_ROOT'])
RECEIPT=Path(os.environ['EVOMIND_COMPETITION_RECEIPT'])


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    began=time.monotonic();stage='preflight'
    r={'schema':'evomind.ev_official_data_provision.v1','competition':'playground-series-s6e9',
       'training_started':False,'gpu_training_hours':0,'secret_values_logged':False}
    try:
        expected='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/ev_public_calibration_20260908'
        if str(ROOT.resolve())!=expected:raise ValueError('Data root mismatch')
        final=ROOT/'.evomind/data-manifest.json'
        if final.exists():raise ValueError('Existing frozen data requires reconciliation')
        os.environ['CUDA_VISIBLE_DEVICES']=''
        key=Path(os.environ['EVOMIND_SECRET_KAGGLE_API_FILE']).read_text().strip()
        os.environ['KAGGLE_API_TOKEN']=key;key=None
        stage='official_api_download'
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate()
            api.competition_download_files('playground-series-s6e9',path=str(ROOT),force=False,quiet=True)
        os.environ.pop('KAGGLE_API_TOKEN',None)
        archive=ROOT/'playground-series-s6e9.zip'
        stage='archive_and_source_identity'
        with zipfile.ZipFile(archive) as z:
            members=z.infolist()
            if len(members)!=3 or {m.filename for m in members}!=set(EXPECTED)-{'frozen-folds.csv'}:raise ValueError('Unexpected archive')
            if sum(m.file_size for m in members)>256*1024*1024:raise ValueError('Archive exceeds bound')
            for m in members:
                if (m.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink')
                target=ROOT/m.filename
                if target.exists():
                    if sha(target)!=EXPECTED[m.filename]:raise ValueError('Existing CSV differs')
                    continue
                import shutil
                with z.open(m) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest)
        for name in set(EXPECTED)-{'frozen-folds.csv'}:
            if sha(ROOT/name)!=EXPECTED[name]:raise ValueError('Official CSV hash differs')
        stage='frozen_folds'
        import numpy as np,pandas as pd
        from sklearn.model_selection import StratifiedKFold
        train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');sample=pd.read_csv(ROOT/'sample_submission.csv')
        if len(train)!=668665 or len(test)!=286571 or list(sample)!=['id','Will_Buy_EV']:raise ValueError('Shape differs')
        if not test.id.equals(sample.id) or set(train.id)&set(test.id):raise ValueError('ID alignment failed')
        if any(f.id.isna().any() or not f.id.is_unique for f in [train,test,sample]):raise ValueError('Invalid IDs')
        y=train.Will_Buy_EV.map({'No':0,'Yes':1})
        if y.isna().any():raise ValueError('Invalid target domain')
        folds=np.full(len(train),-1,dtype=int)
        for fold,(_,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20260908).split(np.zeros(len(train)),y)):
            if (folds[valid]!=-1).any():raise ValueError('Fold overlap')
            folds[valid]=fold
        if (folds<0).any():raise ValueError('Incomplete folds')
        # Reproduce the frozen Windows CSV bytes, not a new split.
        pd.DataFrame({'id':train.id,'row_index':np.arange(len(train)),'fold':folds}).to_csv(ROOT/'frozen-folds.csv',index=False,lineterminator='\r\n')
        if sha(ROOT/'frozen-folds.csv')!=EXPECTED['frozen-folds.csv']:raise ValueError('Frozen fold hash differs')
        stage='isolated_dependencies'
        deps=ROOT/'.evomind/deps';deps.mkdir(parents=True,exist_ok=True)
        install=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--no-deps',
                                '--target',str(deps),'--report',str(ROOT/'.evomind/dependency-install.json'),
                                'lightgbm==4.6.0','xgboost==2.1.3'],capture_output=True,timeout=600)
        if install.returncode:raise ValueError('Dependency preparation failed')
        sys.path.insert(0,str(deps))
        import importlib.metadata
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','lightgbm','xgboost','scipy']}
        profile=[]
        for name in train.columns:
            if name in {'id','Will_Buy_EV'}:continue
            col=train[name]
            row={'name':name,'dtype':str(col.dtype),'missing':int(col.isna().sum()),'unique_count':int(col.nunique())}
            if pd.api.types.is_numeric_dtype(col):row.update(minimum=float(col.min()),maximum=float(col.max()),mean=float(col.mean()),std=float(col.std()))
            profile.append(row)
        manifest={'schema':'evomind.ev_persistent_data.v1','competition':'playground-series-s6e9','status':'FULL_DATA_READY',
                  'source':'official_kaggle_api','root':str(ROOT),'files':{n:{'sha256':h,'bytes':(ROOT/n).stat().st_size} for n,h in EXPECTED.items()},
                  'train_rows':len(train),'test_rows':len(test),'target_encoding':{'No':0,'Yes':1},'fold_count':5,'split_seed':20260908,
                  'profile':profile,'target_counts':{str(k):int(v) for k,v in y.value_counts().items()},
                  'dependencies':versions,'dependency_root':str(deps),'training_started':False}
        with final.open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2)
        for name in EXPECTED:(ROOT/name).chmod(0o444)
        final.chmod(0o444)
        r.update(status='completed',manifest_sha256=sha(final),persistent_root=str(ROOT),files=manifest['files'],
                 train_rows=len(train),test_rows=len(test),dependencies=versions)
    except Exception as e:
        r.update(status='failed',failure_stage=stage,error_type=type(e).__name__)
    finally:
        os.environ.pop('KAGGLE_API_TOKEN',None)
        r['elapsed_seconds']=time.monotonic()-began
        RECEIPT.write_text(json.dumps(r,indent=2))
    return 0 if r['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
