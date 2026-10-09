"""Build the user-authorized, source-backed time-limit amendment without resetting history."""
import hashlib,json,shutil,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/siim-mlebench-calibration-20260908/official-limits-amendment'
PIN='507f92e1138bb6e40dac5c6ee7a6758e6424bf97'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,sort_keys=True)

def main():
    payload=OUT/'payload';payload.mkdir(exist_ok=False)
    old_path=OUT/'base/policy.json';old=json.loads(old_path.read_text(encoding='utf-8'))
    sources=[]
    paths=['README.md','agents/aide/config.yaml','mlebench/competitions/siim-isic-melanoma-classification/config.yaml','mlebench/competitions/siim-isic-melanoma-classification/description.md']
    for index,path in enumerate(paths):
        url='https://raw.githubusercontent.com/openai/mle-bench/'+PIN+'/'+path
        with urllib.request.urlopen(url,timeout=30) as r:raw=r.read(1024*1024)
        text=raw.decode()
        if index==0 and 'Runtime: 24 hours' not in text:raise ValueError('benchmark_time_source_changed')
        if index==1 and ('exec.timeout: 32400' not in text or 'time_limit: &time_limit 86400' not in text or 'step_count: &step_count 500' not in text):raise ValueError('aide_limit_source_changed')
        destination=payload/'sources'/('source-'+str(index)+Path(path).suffix);destination.parent.mkdir(exist_ok=True)
        with destination.open('xb') as f:f.write(raw)
        sources.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'path':destination.relative_to(payload).as_posix()})
    previous=json.loads((OUT.parent/'two-fold-independent-auc.json').read_text(encoding='utf-8'))
    resume_run=previous['run_id'];checkpoint=previous['checkpoint_manifest']
    if checkpoint['identity']['protocol_sha256']!=old['protocol_sha256'] or checkpoint['identity']['candidate_sha256']!=old['baseline_source_sha256']:
        raise ValueError('checkpoint_not_from_this_study')
    if [r['fold'] for r in checkpoint['complete_folds']]!=[0,1]:raise ValueError('checkpoint_coverage_changed')
    units={run:6 for run in old['cases']};units[resume_run]=4
    limits={'schema':'evomind.siim_source_based_limits.v1','case_ceiling_seconds':86400,'step_ceiling_seconds':32400,
        'max_agent_steps':500,'global_limit_seconds':86400,'remaining_work_units':units,'headroom_factor':1.15,
        'allocation_rule':'remaining_work_share_with_15_percent_headroom','local_allocation_is_not_an_official_competition_rule':True,
        'sources':sources,'benchmark_defaults_are_recommendations':True,'kaggle_original_training_time_limit_verified':False,
        'user_authorization':'统一按照比赛限制来自动调节','approved_at':datetime.now(timezone.utc).isoformat()}
    policy={**old,'supersedes_policy_sha256':sha(old_path),'resource_limits':limits,'resume_checkpoints':{resume_run:checkpoint}}
    modules=['siim_calibration_budget.py','siim_calibration_control.py','siim_calibration_runner.py']
    for name in modules:shutil.copyfile(ROOT/'src/evomind_runtime'/name,payload/name)
    for name in ['siim_calibration_suite.py','siim_aide_controller.py']:
        shutil.copyfile(ROOT/'scripts'/name,payload/name)
    policy['runner_sha256']=sha(payload/'siim_calibration_runner.py')
    for run,row in policy['cases'].items():
        prior=ROOT/'artifacts/siim-mlebench-calibration-20260908/runtime-payload-v2/cases'/row['case_id']
        for name in ['task.json','data-profile.json']:
            if sha(prior/name)!=row['input_hashes']['inputs/'+name]:raise ValueError('original_case_input_changed')
        task=json.loads((prior/'task.json').read_text(encoding='utf-8'))
        task['seconds_per_case']=86400
        task['resource_limits']={'case_ceiling_seconds':86400,'step_ceiling_seconds':32400,'max_agent_steps':500,
            'local_budget':'The server clips each attempt to the remaining global budget and the current allocation for unfinished experiments. Read the returned budget. Never assume 24 hours of exclusive GPU time.',
            'source_scope':'MLE-bench recommendation and official AIDE reference configuration, not an asserted SIIM external-training rule.'}
        task['execution']+=' timeout_seconds may request up to 32400; the server applies the stricter current budget.'
        target=payload/'cases'/row['case_id'];write(target/'task.json',task);shutil.copyfile(prior/'data-profile.json',target/'data-profile.json')
        row['input_hashes']={'inputs/'+name:sha(target/name) for name in ['task.json','data-profile.json']}
    write(payload/'policy.json',policy)
    amendment={'schema':'evomind.siim_resource_amendment.v1','parent_policy_sha256':sha(old_path),'new_policy_sha256':sha(payload/'policy.json'),
        'scientific_protocol_sha256_unchanged':old['protocol_sha256'],'data_and_folds_unchanged':True,'baseline_model_source_unchanged':True,
        'old_case_starts_preserved':True,'all_charged_attempts_preserved':True,'total_gpu_budget_expanded':False,
        'model':'deepseek-v4-pro','resource_limits':limits,'resume_run_id':resume_run,'verified_folds_to_reuse':[0,1]}
    write(payload/'resource-amendment.json',amendment)
    seal=json.loads((OUT/'base/seal.json').read_text(encoding='utf-8-sig'));baseline={r['path'].split('/')[-1]:r['sha256'] for r in seal['files'] if r['path'] in {'runtime/evomind_runtime/'+n for n in modules}}
    files={p.relative_to(payload).as_posix():sha(p) for p in payload.rglob('*') if p.is_file()}
    write(payload/'manifest.json',{'files':files,'baseline_module_hashes':baseline,'baseline_policy_sha256':sha(old_path),
        'baseline_seal_sha256':sha(OUT/'base/seal.json'),'new_policy_sha256':sha(payload/'policy.json')})
    print(json.dumps({'status':'source_limits_amendment_built','manifest_sha256':sha(payload/'manifest.json'),
        'policy_sha256':sha(payload/'policy.json'),'case_limit_hours':24,'step_limit_hours':9,'global_limit_hours':24,'reused_folds':2}))

if __name__=='__main__':main()
