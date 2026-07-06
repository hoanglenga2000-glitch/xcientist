from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
ROOT=Path.cwd()
TASKS=['digit_recognizer','titanic','playground_series_s6e6','house_prices_advanced_regression_techniques','spaceship_titanic','bike_sharing_demand','porto_seguro_safe_driver_prediction','santander_customer_transaction_prediction','store_sales_time_series_forecasting','tabular_playground_series_aug_2022']
ALIASES={'house_prices_advanced_regression_techniques':'house_prices'}
records=[]
for task in TASKS:
    local=ALIASES.get(task,task)
    task_dir=ROOT/'tasks'/local
    data_dir=task_dir/'data'
    config_candidates=[ROOT/'configs'/f'{local}.yaml',ROOT/'configs/generated'/f'{local}.yaml',ROOT/'configs'/f'{task}.yaml',ROOT/'configs/generated'/f'{task}.yaml']
    config=next((p for p in config_candidates if p.exists()),None)
    files={name:(data_dir/name).exists() for name in ['train.csv','test.csv','sample_submission.csv']}
    exp_dir=ROOT/'experiments'/local
    latest=None
    if exp_dir.exists():
        dirs=sorted([p for p in exp_dir.iterdir() if p.is_dir()])
        latest=dirs[-1].relative_to(ROOT).as_posix() if dirs else None
    runnable=bool(config and all(files.values()))
    if runnable:
        status='runnable_workstation_dataset_ready'
        next_action='run via workstation API/AgentOrchestrator with score_promotion_gate; no official submit'
    else:
        status='blocked_data_or_config_missing'
        missing=[]
        if not config: missing.append('config yaml')
        missing += [k for k,v in files.items() if not v]
        next_action='onboard Kaggle dataset into tasks/<task>/data and generate config before training: '+', '.join(missing)
    records.append({'task_id':task,'local_task_id':local,'config_path':config.relative_to(ROOT).as_posix() if config else None,'data_dir':data_dir.relative_to(ROOT).as_posix(),'data_files':files,'runnable':runnable,'latest_experiment_dir':latest,'status':status,'next_action':next_action})
summary={'schema':'academic_research_os.kaggle_10_readiness.v1','created_at':datetime.now().isoformat(timespec='seconds'),'runnable_count':sum(r['runnable'] for r in records),'task_count':len(records),'records':records,'claim_boundary':'Readiness only; not a score or leaderboard claim.'}
out=ROOT/'workspace/kaggle_10_task_readiness_20260623.json'
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
md=ROOT/'reports/KAGGLE_10_TASK_READINESS_20260623.md'
lines=['# Kaggle 10 Task Readiness','',f"- Created at: `{summary['created_at']}`",f"- Runnable datasets/configs: `{summary['runnable_count']}/{summary['task_count']}`",'','| task | runnable | config | data files | next action |','|---|---:|---|---|---|']
for r in records:
    files=', '.join([f'{k}:{"Y" if v else "N"}' for k,v in r['data_files'].items()])
    lines.append(f"| `{r['task_id']}` | `{r['runnable']}` | `{r['config_path']}` | {files} | {r['next_action']} |")
lines += ['','## Execution Rule','','Only `runnable=true` tasks may enter workstation training. Missing-data tasks must go through dataset onboarding and gates first.']
md.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'json':out.relative_to(ROOT).as_posix(),'md':md.relative_to(ROOT).as_posix(),'runnable_count':summary['runnable_count']},ensure_ascii=False,indent=2))
