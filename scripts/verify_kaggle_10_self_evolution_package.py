from __future__ import annotations
import json, re, sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
TASKS=ROOT/'benchmark/kaggle_10_self_evolution/tasks_20260623.json'
SUP=ROOT/'workspace/kaggle_10_self_evolution_workstation_supervision_20260623.json'
PROBE=ROOT/'workspace/kaggle_10_access_probe_20260623.json'
OUT_JSON=ROOT/'workspace/kaggle_10_package_verification_20260623.json'
OUT_MD=ROOT/'reports/KAGGLE_10_PACKAGE_VERIFICATION_20260623.md'

VALID_MODALITY={'tabular','image','text','multimodal','time_series','other'}
VALID_MODE={'official','proxy','offline','dry_run'}
VALID_STATUS={'not_started','ready','running','blocked','completed','failed'}

def load(p:Path)->Any: return json.loads(p.read_text(encoding='utf-8-sig'))
def rel(p:Path)->str: return str(p.relative_to(ROOT)).replace('\\','/')

def normalized_task(task:dict[str,Any])->dict[str,Any]:
    task=dict(task)
    modality_map={
        'image_flat_pixels':'image',
        'tabular_mixed':'tabular',
        'tabular_time_features':'tabular',
        'tabular_anonymized':'tabular',
        'tabular_time_series':'time_series',
    }
    mode_map={
        'local_cv_proxy_no_submit':'proxy',
        'historical_evidence_plus_future_gate':'proxy',
        'blocked_until_data_access':'dry_run',
        'planned_workstation_round_no_submit':'dry_run',
    }
    status_map={
        'round2_completed_local_cv':'completed',
        'data_zip_available_pending_workstation_round':'ready',
        'historical_artifacts_present_pending_new_10_task_round':'ready',
        'pending_data_or_access_check':'not_started',
        'blocked_403_recorded':'blocked',
        'planned':'not_started',
    }
    task['raw_modality']=task.get('modality')
    task['raw_evaluation_mode']=task.get('evaluation_mode')
    task['raw_status']=task.get('status')
    task['modality']=modality_map.get(task.get('modality'),task.get('modality'))
    task['evaluation_mode']=mode_map.get(task.get('evaluation_mode'),task.get('evaluation_mode'))
    task['status']=status_map.get(task.get('status'),task.get('status'))
    task['notes']=task.get('notes','') + ' Normalized for benchmark_task.schema compatibility; raw fields preserved as raw_*.'
    return task

def main()->int:
    registry=load(TASKS)
    tasks=registry['tasks']
    normalized=[normalized_task(t) for t in tasks]
    registry['tasks']=normalized
    registry['normalization_note']='modality/evaluation_mode/status normalized to configs/schemas/benchmark_task.schema.json enums; raw values preserved.'
    TASKS.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')

    sup=load(SUP); probe=load(PROBE)
    checks=[]
    task_ids=[t['task_id'] for t in normalized]
    checks.append({'name':'task_count_10','passed':len(normalized)==10,'evidence':len(normalized)})
    checks.append({'name':'unique_task_ids','passed':len(set(task_ids))==10,'evidence':task_ids})
    checks.append({'name':'schema_enum_compatible','passed':all(t['modality'] in VALID_MODALITY and t['evaluation_mode'] in VALID_MODE and t['status'] in VALID_STATUS for t in normalized),'evidence':[{'task_id':t['task_id'],'modality':t['modality'],'evaluation_mode':t['evaluation_mode'],'status':t['status']} for t in normalized]})
    checks.append({'name':'kaggle_probe_10_results','passed':probe.get('status')=='passed' and len(probe.get('results',[]))==10,'evidence':{'status':probe.get('status'),'results':len(probe.get('results',[]))}})
    checks.append({'name':'workstation_runs_10_created','passed':sup.get('workstation_runs_created')==10 and sup.get('task_count')==10,'evidence':{'runs':sup.get('workstation_runs_created'),'tasks':sup.get('task_count')}})
    checks.append({'name':'codex_no_direct_training','passed':sup.get('training_started_by_codex') is False,'evidence':sup.get('training_started_by_codex')})
    checks.append({'name':'no_official_submission','passed':sup.get('official_submission_started') is False,'evidence':sup.get('official_submission_started')})
    contract_missing=[]
    for row in sup['supervision']:
        p=ROOT/'workspace'/'workstation_runs'/row['task_id']/row['workstation_run_id']/'kaggle_10_supervision_contract.json'
        if not p.exists(): contract_missing.append(rel(p))
    checks.append({'name':'per_task_supervision_contracts_exist','passed':not contract_missing,'evidence':contract_missing})
    secret_files=[TASKS,SUP,PROBE,ROOT/'workspace/kaggle_10_workstation_action_create_runs_20260623.json',ROOT/'workspace/kaggle_10_agent_work_orders_20260623.json']
    patterns=[re.compile(r'sk-[A-Za-z0-9]{20,}'),re.compile(r'KGAT_[A-Za-z0-9]{16,}'),re.compile(r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{6,}["\']'),re.compile(r'(?i)(api[_-]?key|token|secret)\s*[:=]\s*["\'][A-Za-z0-9_\-.]{16,}["\']')]
    hits=[]
    for f in secret_files:
        text=f.read_text(encoding='utf-8-sig',errors='replace')
        for i,line in enumerate(text.splitlines(),1):
            if any(p.search(line) for p in patterns): hits.append({'file':rel(f),'line':i})
    checks.append({'name':'secret_scan_clean_for_10_task_package','passed':not hits,'evidence':hits})
    passed=sum(1 for c in checks if c['passed'])
    ok=passed==len(checks)
    report={'ok':ok,'schema':'academic_research_os.kaggle_10_package_verification.v1','checks_passed':passed,'checks_total':len(checks),'checks':checks,'claim_boundary':['This verifies 10-task registry/run contracts, not completion of ten trainings.','Only existing completed training evidence remains digit_recognizer local CV.']}
    OUT_JSON.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Kaggle 10 Package Verification','',f'- Status: `{ "passed" if ok else "failed" }`',f'- Checks: `{passed}/{len(checks)}`','', '## Checks','']
    for c in checks: lines.append(f"- `{ 'passed' if c['passed'] else 'failed' }` {c['name']}")
    OUT_MD.write_text('\n'.join(lines),encoding='utf-8-sig')
    print(json.dumps({'status':'passed' if ok else 'failed','checks':f'{passed}/{len(checks)}','json':rel(OUT_JSON),'md':rel(OUT_MD)},ensure_ascii=False,indent=2))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
