from __future__ import annotations
import json, subprocess, zipfile
from datetime import datetime
from pathlib import Path
ROOT=Path.cwd()
TASKS={
 'spaceship_titanic':'spaceship-titanic',
 'bike_sharing_demand':'bike-sharing-demand',
 'porto_seguro_safe_driver_prediction':'porto-seguro-safe-driver-prediction',
 'santander_customer_transaction_prediction':'santander-customer-transaction-prediction',
 'store_sales_time_series_forecasting':'store-sales-time-series-forecasting',
 'tabular_playground_series_aug_2022':'tabular-playground-series-aug-2022',
}
records=[]
for task, comp in TASKS.items():
    data_dir=ROOT/'tasks'/task/'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd=['python','-m','kaggle','competitions','download','-c',comp,'-p',str(data_dir),'--force']
    started=datetime.now().isoformat(timespec='seconds')
    rec={'task_id':task,'competition':comp,'data_dir':str(data_dir.relative_to(ROOT)).replace('\\','/'),'started_at':started,'command':'python -m kaggle competitions download -c <competition> -p <task_data_dir> --force','status':'pending'}
    try:
        proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=300)
        rec['returncode']=proc.returncode
        rec['stdout_tail']=proc.stdout[-1000:]
        rec['stderr_tail']=proc.stderr[-1000:]
        if proc.returncode!=0:
            rec['status']='failed_download'
        else:
            for z in data_dir.glob('*.zip'):
                try:
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(data_dir)
                except zipfile.BadZipFile:
                    pass
            # unzip nested csv.zip if any
            for z in data_dir.glob('*.zip'):
                try:
                    with zipfile.ZipFile(z) as zf:
                        names=zf.namelist()
                        if any(n.endswith('.csv') for n in names):
                            zf.extractall(data_dir)
                except Exception:
                    pass
            files={p.name:p.stat().st_size for p in data_dir.iterdir() if p.is_file()}
            rec['files']=files
            rec['has_train']=any(p.name.lower()=='train.csv' for p in data_dir.iterdir() if p.is_file())
            rec['has_test']=any(p.name.lower()=='test.csv' for p in data_dir.iterdir() if p.is_file())
            rec['has_sample_submission']=any(p.name.lower()=='sample_submission.csv' for p in data_dir.iterdir() if p.is_file())
            rec['status']='downloaded' if rec['has_train'] and rec['has_test'] and rec['has_sample_submission'] else 'downloaded_incomplete_schema'
    except Exception as e:
        rec['status']='exception'
        rec['error']=str(e)
    records.append(rec)
summary={'schema':'academic_research_os.kaggle_data_onboarding.v1','created_at':datetime.now().isoformat(timespec='seconds'),'records':records,'secret_policy':'No token value was read, printed, or written; Kaggle CLI used existing credential provider only.'}
out=ROOT/'workspace/kaggle_6_data_onboarding_20260624.json'
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'out':out.relative_to(ROOT).as_posix(),'statuses':{r['task_id']:r['status'] for r in records}},ensure_ascii=False,indent=2))
