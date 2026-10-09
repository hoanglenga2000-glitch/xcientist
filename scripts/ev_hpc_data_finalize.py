"""Finalize already verified official files using existing software only; no download/fit."""
import hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
ROOT=Path(os.environ['EVOMIND_COMPETITION_DATA_ROOT']);OUT=Path(os.environ['EVOMIND_COMPETITION_RECEIPT'])
EXPECTED={
 'train.csv':'eae9eaa4e6378df405e755f853771d7e26d212bd93258349fc797b771021946a',
 'test.csv':'539263f6caabc40afd5e2f0bc0ab16b10a2d1177c565fc71b866f0181d836b34',
 'sample_submission.csv':'a9747a8b947e4e35505e3da4535a5a494978b012a7adb50973f13e598849dda5',
 'frozen-folds.csv':'0333a344834f5900051dc8360e908ac7102f27fc5cf31ff4121c73fb67343a28'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 started=time.monotonic();r={'schema':'evomind.ev_official_data_provision.v1','competition':'playground-series-s6e9','training_started':False,'gpu_training_hours':0}
 try:
  if str(ROOT.resolve())!='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/competition_data/ev_public_calibration_20260908':raise ValueError('root_mismatch')
  target=ROOT/'.evomind/data-manifest.json'
  if target.exists():raise ValueError('already_frozen')
  for name,h in EXPECTED.items():
   if sha(ROOT/name)!=h:raise ValueError('source_hash_mismatch')
  import numpy,pandas as pd,sklearn,catboost,scipy,joblib
  versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','catboost','scipy','joblib']}
  train=pd.read_csv(ROOT/'train.csv');test=pd.read_csv(ROOT/'test.csv');folds=pd.read_csv(ROOT/'frozen-folds.csv')
  if len(train)!=668665 or len(test)!=286571 or not train.id.equals(folds.id):raise ValueError('shape_or_fold_mismatch')
  y=train.Will_Buy_EV.map({'No':0,'Yes':1})
  if y.isna().any():raise ValueError('label_encoding_invalid')
  profile=[]
  for name in train:
   if name in {'id','Will_Buy_EV'}:continue
   col=train[name];numeric=pd.api.types.is_numeric_dtype(col)
   row={'name':name,'kind':'numeric' if numeric else 'categorical','missing':int(col.isna().sum()),'unique_count':int(col.nunique())}
   if numeric:row.update(minimum=float(col.min()),maximum=float(col.max()),mean=float(col.mean()),std=float(col.std()))
   profile.append(row)
  deps=ROOT/'.evomind/preinstalled-only';deps.mkdir(parents=True,exist_ok=True)
  manifest={'schema':'evomind.ev_persistent_data.v1','competition':'playground-series-s6e9','status':'FULL_DATA_READY','source':'official_kaggle_api',
            'root':str(ROOT),'files':{n:{'sha256':h,'bytes':(ROOT/n).stat().st_size} for n,h in EXPECTED.items()},
            'train_rows':len(train),'test_rows':len(test),'target_encoding':{'No':0,'Yes':1},'fold_count':5,'split_seed':20260908,
            'profile':profile,'target_counts':{str(k):int(v) for k,v in y.value_counts().items()},'dependencies':versions,'dependency_root':str(deps),
            'optional_dependency_install_used':False,'training_started':False}
  with target.open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2)
  for name in EXPECTED:(ROOT/name).chmod(0o444)
  target.chmod(0o444)
  r.update(status='completed',manifest_sha256=sha(target),manifest=manifest)
 except Exception as e:r.update(status='failed',error_type=type(e).__name__)
 r['elapsed_seconds']=time.monotonic()-started;OUT.write_text(json.dumps(r,indent=2))
 return 0 if r['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
