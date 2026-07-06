from __future__ import annotations
import json, hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
TODAY='20260623'
CREATE_RUNS=ROOT/'workspace/kaggle_10_workstation_action_create_runs_20260623.json'
WORK_ORDERS=ROOT/'workspace/kaggle_10_agent_work_orders_20260623.json'
TASKS=ROOT/'benchmark/kaggle_10_self_evolution/tasks_20260623.json'
OUT_JSON=ROOT/'workspace/kaggle_10_self_evolution_workstation_supervision_20260623.json'
OUT_MD=ROOT/'reports/KAGGLE_10_SELF_EVOLUTION_WORKSTATION_SUPERVISION_20260623.md'

STAGES=[
 ('plan_gate','pending_human_plan_approval'),
 ('code_quality_gate','pending_code_quality_review'),
 ('execution_gate','blocked_until_resource_and_gate_ready'),
 ('submission_gate','blocked_until_candidate_and_human_approval'),
 ('final_report_gate','pending_evidence_completion'),
]
AGENTS=[
 ('OrchestratorAgent','task_decomposition'),
 ('TaskParserAgent','task_spec'),
 ('DataAuditAgent','data_audit'),
 ('RetrospectiveMemoryAgent','memory_retrieval'),
 ('SearchControllerAgent','mlevolve_progressive_search_plan'),
 ('ValidationContractAgent','xcientist_validation_contract'),
 ('CodeImplementationAgent','draft_only_code_plan'),
 ('ExecutionAgent','gated_execution_manifest'),
 ('ValidationAnalysisAgent','metrics_and_score_gate'),
 ('ClaimAuditAgent','claim_drift_audit'),
 ('ReportAgent','reproducibility_report'),
]

def rel(p:Path)->str:
    return str(p.relative_to(ROOT)).replace('\\','/')

def sha256(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_json(p:Path)->Any:
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write_json(p:Path,payload:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

def branch_plan(order:dict[str,Any])->dict[str,Any]:
    ctrl=order['layer_2_mlevolve_search_controller']
    return {
        'search_stage':'exploration_to_exploitation_queue',
        'baseline_first':ctrl['round_0_strategy'],
        'exploration_branches':ctrl['exploration_branches'],
        'exploitation_branches':ctrl['exploitation_branches'],
        'code_generation_modes':ctrl['code_generation_modes'],
        'promotion_rule':ctrl['best_so_far_invariant'],
        'metric_direction':ctrl['metric_direction'],
        'rollback_condition':'candidate missing required artifacts or fails risk gate or does not improve parent best',
    }

def main()->int:
    now=datetime.now().isoformat(timespec='seconds')
    create_runs=read_json(CREATE_RUNS)
    work_orders=read_json(WORK_ORDERS)['work_orders']
    tasks=read_json(TASKS)['tasks']
    by_order={o['task_id']:o for o in work_orders}
    by_task={t['task_id']:t for t in tasks}
    supervision=[]
    artifacts=[]
    all_ok=True
    for item in create_runs:
        task_id=item['task_id']
        run_id=item['run_id']
        ok=bool(item['ok'] and run_id)
        all_ok=all_ok and ok
        order=by_order[task_id]
        task=by_task[task_id]
        run_root=ROOT/'workspace'/'workstation_runs'/task_id/run_id
        manifest_path=run_root/'workstation_run_manifest.json'
        artifact_manifest_path=run_root/'artifact_manifest.json'
        gates=[{'gate_id':f'{run_id}_{name}','gate_type':name,'status':status,'required_before':name.replace('_gate',''),'human_required': name in {'plan_gate','execution_gate','submission_gate','final_report_gate'}} for name,status in STAGES]
        agent_trace=[{'agent_id':agent,'stage':stage,'status':'assigned','executor':'workstation_agent','codex_participation':'supervision_only_no_training','required_artifact':f'{stage}.json'} for agent,stage in AGENTS]
        score_policy={
            'task_id':task_id,
            'metric':task['metric'],
            'metric_direction':order['layer_2_mlevolve_search_controller']['metric_direction'],
            'current_best_evidence': None,
            'baseline_score':task.get('baseline_score'),
            'target_score':task.get('target_score'),
            'best_so_far_invariant':'Never promote a worse branch; preserve parent best and write failure memory.',
            'steady_improvement_definition':'Best-so-far is monotonic over promoted branches; individual attempts may fail but cannot replace protected best.',
        }
        if task_id=='digit_recognizer':
            score_policy['current_best_evidence']='workspace/new_kaggle_cache_training_round_summary_20260623.json'
            score_policy['current_best_score']=0.9669761904761904
            score_policy['next_safe_improvement_rounds']=[
                'SVM/RBF or calibrated linear branch on PCA/reduced features',
                'ExtraTrees/RF seed ensemble and OOF mode/rank blend',
                'CNN branch when GPU/HPC returns; compare against protected 0.966976 local CV',
            ]
        else:
            score_policy['current_best_score']=None
            score_policy['next_safe_improvement_rounds']=[
                'Round0 robust baseline and data audit',
                'Round1 branch search over first four allowed models',
                'Round2 exploitation only after valid OOF and schema artifacts exist',
            ]
        validation_contract={
            'contract_id':f'{run_id}_validation_contract',
            'task_id':task_id,
            'hypothesis':'A robust baseline plus controlled branch search can improve best-so-far without sacrificing auditability.',
            'implementation_requirement':'All training must be started by workstation ExecutionAgent from approved manifests; Codex does not run model training.',
            'metric':task['metric'],
            'acceptance_criteria':order['layer_3_xcientist_research_harness']['acceptance_criteria'],
            'risk_checklist':order['layer_3_xcientist_research_harness']['risk_checks'],
            'conclusion_boundary':order['layer_3_xcientist_research_harness']['claim_boundary'],
            'required_artifacts':['task_spec.json','data_audit.json','search_controller_decision.json','validation_contract.json','metrics.json','oof_predictions.*','submission_audit.json','claim_audit.json','report.md'],
        }
        run_supervision={
            'task_id':task_id,
            'competition_name':task['competition_name'],
            'workstation_run_id':run_id,
            'api_create_run_ok':ok,
            'run_manifest_path':rel(manifest_path),
            'artifact_manifest_path':rel(artifact_manifest_path),
            'training_started':False,
            'official_submission_started':False,
            'codex_role':'supervisor_only_no_direct_training',
            'gates':gates,
            'agent_trace_assigned':agent_trace,
            'mlevolve_search_plan':branch_plan(order),
            'xcientist_validation_contract':validation_contract,
            'score_policy':score_policy,
            'next_action':order['next_action'],
            'data_status':order['data_status'],
        }
        out=run_root/'kaggle_10_supervision_contract.json'
        write_json(out,run_supervision)
        report=run_root/'kaggle_10_supervision_report.md'
        report.write_text('\n'.join([
            f'# {task_id} Kaggle 10 Supervision Contract',
            '',
            f'- workstation_run_id: `{run_id}`',
            f'- competition: `{task["competition_name"]}`',
            '- Codex role: supervisor only; no direct training.',
            '- Training started: `false`',
            '- Official submission started: `false`',
            '',
            '## Search Plan',
            f"- Baseline: {run_supervision['mlevolve_search_plan']['baseline_first']}",
            f"- Exploration: {', '.join(run_supervision['mlevolve_search_plan']['exploration_branches'])}",
            f"- Exploitation: {', '.join(run_supervision['mlevolve_search_plan']['exploitation_branches'])}",
            '',
            '## Gates',
            *[f"- {g['gate_type']}: {g['status']}" for g in gates],
            '',
            '## Claim Boundary',
            validation_contract['conclusion_boundary'],
        ]),encoding='utf-8-sig')
        for p in [out,report,manifest_path,artifact_manifest_path]:
            if p.exists(): artifacts.append({'task_id':task_id,'path':rel(p),'sha256':sha256(p),'artifact_type':p.name})
        supervision.append(run_supervision)
    completed=[s for s in supervision if s['task_id']=='digit_recognizer']
    payload={
        'schema':'academic_research_os.kaggle_10_self_evolution_supervision.v1',
        'created_at':now,
        'task_count':len(supervision),
        'workstation_runs_created':sum(1 for s in supervision if s['api_create_run_ok']),
        'training_started_by_codex':False,
        'official_submission_started':False,
        'completed_score_evidence_tasks':len(completed),
        'current_score_evidence':completed,
        'supervision':supervision,
        'artifact_manifest':artifacts,
        'global_score_improvement_guard':{
            'policy':'best_so_far_monotonic_by_promotion_only',
            'meaning':'The system can attempt risky branches, but only validated improvements become the protected best; failures are memory, not regressions.',
            'current_proven_improvement':'digit_recognizer round1 0.912881 -> round2 0.966976 local CV',
            'not_yet_proven':'The other nine tasks have planned run contracts but no completed training evidence yet.'
        },
        'claim_boundary':['Ten workstation runs/contracts are created and agent work is assigned.','Only digit_recognizer currently has completed local-CV improvement evidence.','No official Kaggle submit, rank, medal, or ten-task score improvement is claimed.']
    }
    write_json(OUT_JSON,payload)
    lines=['# Kaggle 10 自进化工作站监督台账','',f'- Created at: `{now}`',f'- Workstation runs created: `{payload["workstation_runs_created"]}/{payload["task_count"]}`','- Codex direct training: `false`','- Official Kaggle submit: `false`','', '## Run Matrix','','| task | run_id | data ready | next action | current best evidence |','|---|---|---:|---|---|']
    for s in supervision:
        ev=s['score_policy'].get('current_best_evidence') or 'pending'
        lines.append(f"| `{s['task_id']}` | `{s['workstation_run_id']}` | {s['data_status']['all_required_csv_present']} | {s['next_action']} | {ev} |")
    lines += ['', '## Steady Improvement Guard','', payload['global_score_improvement_guard']['meaning'], '', f"- Proven so far: {payload['global_score_improvement_guard']['current_proven_improvement']}", f"- Not yet proven: {payload['global_score_improvement_guard']['not_yet_proven']}", '', '## Artifacts','', f'- JSON: `{rel(OUT_JSON)}`']
    OUT_MD.parent.mkdir(parents=True,exist_ok=True)
    OUT_MD.write_text('\n'.join(lines),encoding='utf-8-sig')
    print(json.dumps({'status':'passed','runs':payload['workstation_runs_created'],'tasks':payload['task_count'],'json':rel(OUT_JSON),'md':rel(OUT_MD)},ensure_ascii=False,indent=2))
    return 0 if all_ok else 1
if __name__=='__main__': raise SystemExit(main())
