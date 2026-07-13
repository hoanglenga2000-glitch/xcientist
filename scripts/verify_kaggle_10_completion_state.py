from __future__ import annotations
import hashlib,json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path.cwd()
PROGRESS=ROOT/'workspace/kaggle_10_self_evolution_progress_20260623.json'
READINESS=ROOT/'workspace/kaggle_10_task_readiness_20260623.json'
ORDERS=ROOT/'workspace/kaggle_10_next_search_controller_orders_20260623.json'
CACHE=ROOT/'workspace/deepseek_cache_hit_rate_target_verification_20260623.json'
OUT=ROOT/'workspace/kaggle_10_completion_audit_20260624.json'
MD=ROOT/'reports/KAGGLE_10_COMPLETION_AUDIT_20260624.md'

def load(path): return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None
progress=load(PROGRESS) or {}
readiness=load(READINESS) or {}
orders=load(ORDERS) or {}
cache=load(CACHE) or {}
checks=[]
def check(name, passed, detail, evidence=None):
    checks.append({'name':name,'status':'passed' if passed else 'failed','detail':detail,'evidence':evidence})
# core checks
check('ten_tasks_registered', progress.get('task_count')==10 and len(orders.get('orders',[]))==10, f"task_count={progress.get('task_count')} orders={len(orders.get('orders',[]))}")
check('workstation_executor_only', progress.get('training_executor')=='workstation_api_or_workstation_agent_artifacts' and progress.get('codex_role')=='supervisor_only_no_direct_training', 'training executor and Codex role recorded')
check('no_official_submission', progress.get('official_submission_made') is False and all(not r.get('official_submission_made') for r in progress.get('results',[])), 'no official Kaggle submit recorded')
check('deepseek_cache_ge_80', cache.get('measured_80_percent_met') is True and (cache.get('manifest_stats') or {}).get('observed_hit_ratio',0)>=0.8, f"hit_ratio={(cache.get('manifest_stats') or {}).get('observed_hit_ratio')}")
check('promotion_gate_integrated', progress.get('workstation_promotion_gate_integrated') is True and len(progress.get('score_guard',{}).get('promotion_gate_evidence',[]))>=2, 'score promotion gate evidence exists', progress.get('score_guard',{}).get('promotion_gate_evidence'))
check('runnable_task_count_ge_5', readiness.get('runnable_count',0)>=5, f"runnable_count={readiness.get('runnable_count')}")
check('actual_score_evidence_ge_5', progress.get('tasks_with_actual_workstation_score_evidence',0)>=5, f"actual_score_tasks={progress.get('tasks_with_actual_workstation_score_evidence')}")
check('confirmed_improvements_ge_3', progress.get('tasks_with_confirmed_local_improvement',0)>=3, f"confirmed={progress.get('tasks_with_confirmed_local_improvement')}")
# evidence paths exist
missing=[]
for item in progress.get('artifact_manifest',[]):
    rel=item.get('path')
    if rel and not (ROOT/rel).exists():
        missing.append(rel)
check('artifact_manifest_paths_exist', not missing, f"missing={len(missing)}", missing[:20])
# blocked tasks are explicit
blocked=[r for r in readiness.get('records',[]) if not r.get('runnable')]
expected_blocked = max(0, 10 - int(readiness.get('runnable_count', 0)))
check('blocked_tasks_explicitly_marked', len(blocked)==expected_blocked and all('blocked' in r.get('status','') for r in blocked), f"blocked={len(blocked)} expected={expected_blocked}", [r['task_id'] for r in blocked])
# completion status
passed=sum(1 for c in checks if c['status']=='passed')
all_complete = readiness.get('runnable_count')==10 and progress.get('tasks_with_actual_workstation_score_evidence')==10
status='not_complete_but_progressing' if not all_complete else 'complete'
report={'schema':'academic_research_os.kaggle_10_completion_audit.v1','generated_at':datetime.now(timezone.utc).isoformat(),'status':status,'checks_passed':passed,'checks_total':len(checks),'checks':checks,'current_truth':{'task_count':progress.get('task_count'),'runnable_task_count':readiness.get('runnable_count'),'actual_score_evidence_tasks':progress.get('tasks_with_actual_workstation_score_evidence'),'confirmed_local_improvement_tasks':progress.get('tasks_with_confirmed_local_improvement'),'official_submission_made':progress.get('official_submission_made'),'deepseek_cache_hit_ratio':(cache.get('manifest_stats') or {}).get('observed_hit_ratio')},'remaining_work':['Onboard missing datasets/configs for 6 Kaggle tasks.','Run each newly onboarded task through workstation agents only.','Require score_promotion_gate for every candidate.','Do not claim leaderboard rank/medal until human-approved official submissions exist.']}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Kaggle 10 Completion Audit','',f"- Status: `{status}`",f"- Checks: `{passed}/{len(checks)}`",'', '## Current Truth','']
for k,v in report['current_truth'].items(): lines.append(f'- {k}: `{v}`')
lines += ['','## Checks','']
for c in checks: lines.append(f"- `{c['status']}` {c['name']}: {c['detail']}")
lines += ['','## Remaining Work','']+[f'- {x}' for x in report['remaining_work']]
MD.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'status':status,'checks':f'{passed}/{len(checks)}','json':OUT.relative_to(ROOT).as_posix(),'md':MD.relative_to(ROOT).as_posix()},ensure_ascii=False,indent=2))
