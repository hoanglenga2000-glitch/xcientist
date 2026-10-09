"""Freeze JPEG-only SIIM benchmark inputs, common folds and provenance on HPC."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
ROOT=Path('/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra')
SOURCE=ROOT/'mlebench_official_data/siim-isic-melanoma-classification/prepared/public'
DEST=ROOT/'competition_data/siim_mlebench_calibration_20260908'
EXPECTED={
 'train.csv':'d1827411b90c36fc9d68994360b3aab4a5c123e348e872511f338bef4f3f18ac',
 'test.csv':'4050281af8c6ed04d64b8ac24d1b151b64dbb1d4a4160ee72b0900012e5282ea',
 'sample_submission.csv':'0045a0af17b1b68c2f600c3e18dba0b5f5410fd47f47476cee1b6784897f2da9'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def build_folds(train,hashes):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    parent=list(range(len(train)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def join(a,b):parent[find(a)]=find(b)
    patients={};content={}
    for i,row in enumerate(train.itertuples()):
        patient=str(row.patient_id) if pd.notna(row.patient_id) else 'missing-'+str(i)
        if patient in patients:join(i,patients[patient])
        else:patients[patient]=i
        digest=hashes[row.image_name]
        if digest in content:join(i,content[digest])
        else:content[digest]=i
    groups=np.asarray([find(i) for i in range(len(train))]);folds=np.full(len(train),-1)
    y=train.target.to_numpy()
    for fold,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=20260908).split(np.zeros(len(y)),y,groups)):
        if set(groups[tr]) & set(groups[va]):raise ValueError('internal_group_leakage')
        if set(y[va])!={0,1}:raise ValueError('single_class_validation_fold')
        folds[va]=fold
    if (folds<0).any():raise ValueError('folds_incomplete')
    return pd.DataFrame({'image_name':train.image_name,'row_index':np.arange(len(train)),'fold':folds,'group':groups})

def main():
    import fcntl
    started=time.monotonic()
    DEST.mkdir(parents=True,exist_ok=True)
    lock=(DEST/'.prepare.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def progress(phase,**values):
        data={'phase':phase,'pid':os.getpid(),'at_epoch':time.time(),'elapsed_seconds':time.monotonic()-started,'training_started':False,**values}
        temporary=DEST/'prepare-progress.json.new'
        temporary.write_text(json.dumps(data));temporary.replace(DEST/'prepare-progress.json')
    progress('source_validation')
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    if (DEST/'manifest.json').exists():raise ValueError('frozen_dataset_already_exists')
    for name,h in EXPECTED.items():
        if sha(SOURCE/name)!=h:raise ValueError('prepared_source_csv_changed')
    import pandas as pd
    from PIL import Image
    train=pd.read_csv(SOURCE/'train.csv');test=pd.read_csv(SOURCE/'test.csv');sample=pd.read_csv(SOURCE/'sample_submission.csv')
    if len(train)!=28984 or len(test)!=4142 or set(train.target)!={0,1}:raise ValueError('data_shape_or_target_changed')
    if not test.image_name.equals(sample.image_name):raise ValueError('submission_order_mismatch')
    if not train.image_name.is_unique or not test.image_name.is_unique or set(train.image_name)&set(test.image_name):raise ValueError('image_id_leakage')
    rows=[];hashes={};splits={};image_shape_counts={};split_collision_hashes=set()
    cached={};journal=DEST/'image-audit-progress.jsonl'
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                item=json.loads(line);cached[(item['split'],item['image_name'])]=item
            except (ValueError,KeyError):continue
    journal_stream=journal.open('a')
    for split,frame in [('train',train),('test',test)]:
        directory=SOURCE/'jpeg'/split
        if {p.stem for p in directory.glob('*.jpg')}!=set(frame.image_name):raise ValueError('image_set_mismatch')
        for image_id in frame.image_name:
            p=directory/(image_id+'.jpg')
            if p.is_symlink() or p.resolve().parent!=directory.resolve():raise ValueError('image_path_escape')
            info=p.stat();old=cached.get((split,image_id))
            if old and old.get('bytes')==info.st_size and old.get('mtime_ns')==info.st_mtime_ns:
                digest=old['sha256'];shape=old['shape']
            else:
                digest=sha(p)
                with Image.open(p) as image:shape=str(image.size);image.verify()
                journal_stream.write(json.dumps({'split':split,'image_name':image_id,'bytes':info.st_size,'mtime_ns':info.st_mtime_ns,'sha256':digest,'shape':shape})+'\n')
            if digest in splits and splits[digest]!=split:split_collision_hashes.add(digest)
            splits[digest]=split;hashes[image_id]=digest
            image_shape_counts[shape]=image_shape_counts.get(shape,0)+1
            rows.append({'image_name':image_id,'split':split,'bytes':p.stat().st_size,'sha256':digest})
            if len(rows)%250==0:
                journal_stream.flush();progress('image_validation',completed_images=len(rows),total_images=33126)
    journal_stream.flush();journal_stream.close()
    # Only remove labeled training duplicates; the benchmark test membership is immutable.
    original_train_count=len(train)
    excluded=[name for name in train.image_name if hashes[name] in split_collision_hashes]
    train=train.loc[~train.image_name.isin(excluded)].reset_index(drop=True)
    if set(hashes[name] for name in train.image_name)&set(hashes[name] for name in test.image_name):raise ValueError('deduplication_failed')
    progress('freeze_group_folds',completed_images=len(rows),excluded_training_images=len(excluded))
    folds=build_folds(train,hashes)
    training_patients=set(train.patient_id.dropna());testing_patients=set(test.patient_id.dropna())
    features=['image_name','patient_id','sex','age_approx','anatom_site_general_challenge']
    public_train=train[features+['target']].copy();public_test=test[features].copy()
    public_train['image_path']=[str(SOURCE/'jpeg/train'/(i+'.jpg')) for i in train.image_name]
    public_test['image_path']=[str(SOURCE/'jpeg/test'/(i+'.jpg')) for i in test.image_name]
    DEST.mkdir(parents=True,exist_ok=True)
    for name,frame in [('train.csv',public_train),('test.csv',public_test),('sample_submission.csv',sample),('frozen-folds.csv',folds)]:
        with (DEST/name).open('x',newline='') as f:frame.to_csv(f,index=False,lineterminator='\n')
        (DEST/name).chmod(0o444)
    images=DEST/'image-manifest.json'
    with images.open('x') as f:json.dump(rows,f,sort_keys=True)
    exclusions=DEST/'excluded-training-images.json'
    with exclusions.open('x') as f:json.dump({'reason':'exact_image_duplicate_of_benchmark_test','image_names':excluded},f)
    manifest={'schema':'evomind.siim_frozen_data.v1','competition':'siim-isic-melanoma-classification',
        'mlebench_commit':'507f92e1138bb6e40dac5c6ee7a6758e6424bf97','root':str(DEST),'source_public_root':str(SOURCE),
        'source_csv_hashes':EXPECTED,'files':{name:{'sha256':sha(DEST/name),'bytes':(DEST/name).stat().st_size} for name in ['train.csv','test.csv','sample_submission.csv','frozen-folds.csv','image-manifest.json','excluded-training-images.json']},
        'train_rows':len(train),'test_rows':len(test),'target_counts':{str(k):int(v) for k,v in train.target.value_counts().items()},
        'fold_sizes':{str(k):int(v) for k,v in folds.fold.value_counts().items()},'internal_patient_and_content_groups_disjoint':True,
        'benchmark_train_test_patient_overlap':len(training_patients&testing_patients),'benchmark_test_membership_unchanged':True,
        'benchmark_training_membership_unchanged':not bool(excluded),'original_train_rows':original_train_count,
        'excluded_training_rows':len(excluded),'train_test_exact_duplicate_hashes_before':len(split_collision_hashes),
        'train_test_exact_duplicate_hashes_after':0,'evaluation_variant':'mlebench_siim_test_preserved_train_deduplicated_v1',
        'image_hash_count':len(rows),'image_dimensions':image_shape_counts,
        'excluded_columns':['diagnosis','benign_malignant'],'identifier_columns':['image_name','patient_id'],
        'test_tfrecords_exposed_to_training':False,'raw_training_csv_exposed_to_training':False,
        'private_answers_read':False,'network_calls':0,'training_started':False,'elapsed_seconds':time.monotonic()-started}
    with (DEST/'manifest.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2)
    (DEST/'manifest.json').chmod(0o444);images.chmod(0o444)
    progress('complete',completed_images=len(rows))
    print(json.dumps({'status':'frozen_data_ready','manifest_sha256':sha(DEST/'manifest.json'),
        'manifest':{k:v for k,v in manifest.items() if k!='image_dimensions'},'large_artifacts_retained_on_hpc':True}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        safe={'status':'failed','error_type':type(e).__name__,'detail':str(e) if isinstance(e,ValueError) else 'prepare_failed'}
        if DEST.is_dir():(DEST/('prepare-failure-'+str(time.time_ns())+'.json')).write_text(json.dumps(safe))
        print(json.dumps(safe));raise SystemExit(2)
