from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
ROOT=Path.cwd()
readiness=json.loads((ROOT/'workspace/kaggle_10_task_readiness_20260623.json').read_text(encoding='utf-8-sig'))
progress=json.loads((ROOT/'workspace/kaggle_10_self_evolution_progress_20260623.json').read_text(encoding='utf-8-sig'))
orders=[]
for r in readiness['records']:
    if r['runnable']:
        if r['task_id']=='house_prices_advanced_regression_techniques':
            branch='regression_specialized_lightgbm_catboost_or_feature_engineering'
            objective='beat best-so-far RMSLE 0.122627; if not, hold candidate and write negative memory'
        elif r['task_id']=='titanic':
            branch='small_tabular_classification_feature_ablation_and_calibration'
            objective='beat best-so-far accuracy 0.838384; if not, hold candidate'
        elif r['task_id']=='playground_series_s6e6':
            branch='large_tabular_classification_hpc_when_available_or_fast_local_probe'
            objective='produce workstation score-gated candidate with balanced_accuracy evidence; do not use official submit'
        else:
            branch='baseline_then_ensemble'
            objective='establish baseline then candidate under promotion gate'
        status='ready_for_search_controller'
    else:
        branch='dataset_onboarding'
        objective='download/onboard data and generate config; no training until data gate passes'
        status='blocked_until_onboarded'
    orders.append({'task_id':r['task_id'],'status':status,'assigned_agent':'SearchControllerAgent' if r['runnable'] else 'DataOnboardingAgent','branch':branch,'objective':objective,'required_gates':['PLAN_APPROVAL','CODE_QUALITY','SCORE_PROMOTION_GATE','SUBMISSION_APPROVAL_HUMAN_ONLY'],'no_codex_training':True,'next_action':r['next_action']})
out={'schema':'academic_research_os.kaggle_10_search_controller_orders.v1','created_at':datetime.now().isoformat(timespec='seconds'),'policy':'robust baseline first; promotion only through score_promotion_gate; failed or non-improving candidates become retrospective memory','orders':orders}
path=ROOT/'workspace/kaggle_10_next_search_controller_orders_20260623.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
md=ROOT/'reports/KAGGLE_10_NEXT_SEARCH_CONTROLLER_ORDERS_20260623.md'
lines=['# Kaggle 10 Next Search Controller Orders','',f"- Created at: `{out['created_at']}`",f"- Policy: {out['policy']}",'','| task | agent | status | branch | objective |','|---|---|---|---|---|']
for o in orders:
    lines.append(f"| `{o['task_id']}` | {o['assigned_agent']} | {o['status']} | {o['branch']} | {o['objective']} |")
md.write_text('\n'.join(lines),encoding='utf-8')
# update progress with readiness/order evidence
progress['created_at']=datetime.now().isoformat(timespec='seconds')
progress['task_readiness_path']='workspace/kaggle_10_task_readiness_20260623.json'
progress['next_search_controller_orders_path']='workspace/kaggle_10_next_search_controller_orders_20260623.json'
progress['runnable_task_count']=readiness['runnable_count']
progress['workstation_promotion_gate_integrated']=True
for p,t in [(ROOT/'workspace/kaggle_10_task_readiness_20260623.json','readiness'),(ROOT/'reports/KAGGLE_10_TASK_READINESS_20260623.md','readiness_report'),(path,'search_orders'),(md,'search_orders_report'),(ROOT/'experiments/titanic/wr_2026-06-23T23-33-34.809327_1de68f06/score_promotion_gate.json','integrated_promotion_gate'),(ROOT/'experiments/titanic/wr_2026-06-23T23-33-34.809327_1de68f06/research_os_search_graph.json','integrated_search_graph')]:
    if p.exists() and not any(a.get('path')==p.relative_to(ROOT).as_posix() for a in progress.get('artifact_manifest',[])):
        import hashlib
        h=hashlib.sha256(); h.update(p.read_bytes())
        progress.setdefault('artifact_manifest',[]).append({'path':p.relative_to(ROOT).as_posix(),'sha256':h.hexdigest(),'type':t})
(ROOT/'workspace/kaggle_10_self_evolution_progress_20260623.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'orders':path.relative_to(ROOT).as_posix(),'report':md.relative_to(ROOT).as_posix(),'orders_count':len(orders)},ensure_ascii=False,indent=2))
