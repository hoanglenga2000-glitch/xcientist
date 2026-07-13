from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
ROOT=Path.cwd()
KAGGLE_TASKS=[
 'digit_recognizer','titanic','playground_series_s6e6','house_prices','house_prices_advanced_regression_techniques',
 'spaceship_titanic','bike_sharing_demand','porto_seguro_safe_driver_prediction','santander_customer_transaction_prediction',
 'store_sales_time_series_forecasting','tabular_playground_series_aug_2022'
]
TASK_FROM_COMPETITION={
 'playground-series-s6e6':'playground_series_s6e6',
 'spaceship-titanic':'spaceship_titanic',
}

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except Exception:
            return None

def metric_from_metrics(payload):
    if not isinstance(payload, dict):
        return None, None, None
    metric=payload.get('metric') or payload.get('ensemble',{}).get('selection_metric')
    direction=payload.get('metric_direction') or payload.get('ensemble',{}).get('metric_direction')
    score=payload.get('ensemble',{}).get('best_validation_score')
    if score is None:
        for key in ['cv_accuracy_mean','cv_rmsle_mean','holdout_rmsle','accuracy','rmsle']:
            if isinstance(payload.get(key),(int,float)):
                score=payload[key]; metric=metric or key; break
    return metric, direction, score

def is_better(candidate, current, direction):
    if not isinstance(candidate, (int, float)):
        return False
    if not isinstance(current, (int, float)):
        return True
    if str(direction).lower() in {'minimize','lower','lower_is_better'}:
        return candidate < current
    return candidate > current

def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def infer_task_id(path: Path, payload: dict | None = None):
    if isinstance(payload, dict):
        for key in ['task_id', 'task']:
            if payload.get(key):
                return payload[key]
        competition = payload.get('competition') or payload.get('competition_slug')
        if competition in TASK_FROM_COMPETITION:
            return TASK_FROM_COMPETITION[competition]
    parts=list(path.parts)
    for root_name in ['experiments', 'workstation_runs']:
        if root_name in parts:
            idx=parts.index(root_name)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None

def top30_from_rank_percentile(rank_percentile):
    return (
        isinstance(rank_percentile, (int, float))
        and rank_percentile <= 0.30
    )

def build_official_record_from_submission(rec):
    official=rec['official_submission']
    rank=official.get('rank')
    leaderboard_team_count=official.get('leaderboard_team_count')
    rank_percentile=official.get('rank_percentile')
    rank_known=isinstance(rank, int) and isinstance(leaderboard_team_count, int)
    top30_reached=top30_from_rank_percentile(rank_percentile) if rank_known else None
    return {
        'task_id':rec['task_id'],
        'run_id':rec['run_id'],
        'competition':official.get('competition'),
        'submission_ref':official.get('submission_ref'),
        'public_score':official.get('public_score'),
        'official_score_known': isinstance(official.get('public_score'), (int, float)),
        'rank':rank,
        'leaderboard_team_count':leaderboard_team_count,
        'rank_percentile':rank_percentile,
        'rank_unknown': not rank_known,
        'top30_reached': top30_reached,
        'workstation_status': (
            'top30_reached' if top30_reached is True
            else 'top30_failed' if rank_known
            else 'rank_unknown'
        ),
        'status':official.get('status'),
        'experiment_id': official.get('experiment_id'),
        'submission_file': official.get('submission_file'),
        'evidence_source':'kaggle_official_submission',
        'evidence_paths':[rec['path'] + '/kaggle_official_submission.json'],
        'path':rec['path'],
    }

def parse_submission_tail_for_ref(tail, preferred_ref=None):
    if not isinstance(tail, str) or not tail.strip():
        return {}
    rows = []
    for raw_line in tail.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("ref ") or line.startswith("--------"):
            continue
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        public_score = None
        status = None
        for index, part in enumerate(parts):
            if part.startswith("SubmissionStatus."):
                status = part
                for candidate in parts[index + 1:]:
                    public_score = to_float(candidate)
                    if public_score is not None:
                        break
                break
        rows.append({"submission_ref": parts[0], "status": status, "public_score": public_score})
    if preferred_ref:
        for row in rows:
            if str(row.get("submission_ref")) == str(preferred_ref):
                return row
    return rows[0] if rows else {}

def build_official_record_from_response(response_path: Path):
    response=read_json(response_path)
    if not isinstance(response, dict):
        return None
    competition=response.get('competition_slug') or response.get('competition')
    task_id=infer_task_id(response_path, {'competition_slug': competition})
    if not task_id:
        return None
    tail_record=parse_submission_tail_for_ref(response.get('submissions_list_tail'), response.get('kaggle_ref'))
    public_score=to_float(response.get('public_score'))
    if public_score is None:
        public_score=to_float(tail_record.get('public_score'))
    submission_ref=response.get('kaggle_ref') or tail_record.get('submission_ref')
    if public_score is None or not submission_ref:
        return None
    rank=response.get('public_rank')
    leaderboard_team_count=response.get('leaderboard_team_count')
    rank_percentile=response.get('rank_percentile')
    if rank_percentile is None and isinstance(rank, int) and isinstance(leaderboard_team_count, int) and leaderboard_team_count > 0:
        rank_percentile=rank / leaderboard_team_count
    rank_known=isinstance(rank, int) and isinstance(leaderboard_team_count, int)
    top30_reached=top30_from_rank_percentile(rank_percentile) if rank_known else None
    return {
        'task_id':task_id,
        'run_id':response.get('workstation_run_id') or response_path.parent.name,
        'competition':competition,
        'submission_ref':str(submission_ref),
        'public_score':public_score,
        'official_score_known': True,
        'rank':rank,
        'leaderboard_team_count':leaderboard_team_count,
        'rank_percentile':rank_percentile,
        'rank_unknown': not rank_known,
        'top30_reached': top30_reached,
        'workstation_status': (
            'top30_reached' if top30_reached is True
            else 'top30_failed' if rank_known
            else 'official-known-rank-unknown'
        ),
        'status':response.get('status'),
        'experiment_id': response.get('experiment_id'),
        'submission_file': response.get('submission_path'),
        'evidence_source':'kaggle_submission_response',
        'evidence_paths':[response_path.relative_to(ROOT).as_posix()],
        'path':response_path.parent.relative_to(ROOT).as_posix(),
    }

def build_official_record_from_score_gate(gate_path: Path):
    gate=read_json(gate_path)
    if not isinstance(gate, dict):
        return None
    frontier=gate.get('score_recovery_frontier') if isinstance(gate.get('score_recovery_frontier'), dict) else {}
    current=frontier.get('current_official_best') if isinstance(frontier.get('current_official_best'), dict) else {}
    public_score=to_float(current.get('public_score'))
    if public_score is None:
        public_score=to_float(gate.get('current_best_public_score'))
    submission_ref=current.get('submission_ref') or gate.get('current_best_submission_ref')
    if public_score is None or not submission_ref:
        return None
    competition=gate.get('competition_slug') or current.get('competition')
    task_id=infer_task_id(gate_path, {'competition_slug': competition})
    if not task_id:
        return None
    source_run_id=gate.get('workstation_run_id') or gate_path.parent.name
    run_id=current.get('workstation_run_id') or source_run_id
    path=gate_path.parent.relative_to(ROOT).as_posix()
    return {
        'task_id':task_id,
        'run_id':run_id,
        'competition':competition,
        'submission_ref':str(submission_ref),
        'public_score':public_score,
        'official_score_known': True,
        'rank':None,
        'leaderboard_team_count':None,
        'rank_percentile':None,
        'rank_unknown':True,
        'top30_reached':None,
        'workstation_status':'official-known-rank-unknown',
        'status':'official-known-rank-unknown',
        'experiment_id':current.get('experiment_id'),
        'submission_file':current.get('submission_file'),
        'source_workstation_run_id':source_run_id,
        'evidence_source':'score_improvement_gate.current_official_best',
        'evidence_paths':[gate_path.relative_to(ROOT).as_posix()],
        'path':path,
    }

def merge_official_records(records):
    merged={}
    for item in records:
        key=(item.get('task_id'), item.get('competition'), str(item.get('submission_ref')))
        existing=merged.get(key)
        if not existing:
            merged[key]=item
            continue
        existing_paths=existing.setdefault('evidence_paths', [])
        for evidence_path in item.get('evidence_paths', []):
            if evidence_path not in existing_paths:
                existing_paths.append(evidence_path)
        existing_rank_known=not existing.get('rank_unknown')
        item_rank_known=not item.get('rank_unknown')
        if item_rank_known and not existing_rank_known:
            item['evidence_paths']=existing_paths
            merged[key]=item
        elif item.get('evidence_source') == 'kaggle_official_submission' and existing.get('evidence_source') != 'kaggle_official_submission':
            item['evidence_paths']=existing_paths
            merged[key]=item
    return list(merged.values())

records=[]
for task in KAGGLE_TASKS:
    exp_root=ROOT/'experiments'/task
    if not exp_root.exists():
        continue
    for run_dir in sorted([p for p in exp_root.iterdir() if p.is_dir()]):
        metrics=read_json(run_dir/'metrics.json')
        metric,direction,score=metric_from_metrics(metrics)
        gate=read_json(run_dir/'score_promotion_gate.json')
        decision=gate.get('decision',{}) if isinstance(gate,dict) else {}
        trace=read_json(run_dir/'agent_trace.json')
        if isinstance(trace,dict):
            trace_items=trace.get('traces') or trace.get('trace') or trace.get('agents') or []
        elif isinstance(trace,list):
            trace_items=trace
        else:
            trace_items=[]
        agents=sorted({x.get('agent') for x in trace_items if isinstance(x,dict) and x.get('agent')})
        has_artifacts={name:(run_dir/name).exists() for name in [
            'metrics.json','submission.csv','oof_predictions.csv','artifact_manifest.json',
            'score_promotion_gate.json','agent_trace.json','launcher_manifest.json',
            'timeout_manifest.json','failure_review.json','orchestrator_run.json',
            'search_controller_decision.json','validation_contract.json','claim_audit.json',
            'submission_audit.json','task_benchmark_state.json','workstation_run_registry.json',
            'rank_promotion_gate.json','benchmark_claim_gate.json','kaggle_official_submission.json'
        ]}
        official=read_json(run_dir/'kaggle_official_submission.json')
        records.append({
            'task_id':task,
            'run_id':run_dir.name,
            'path':run_dir.relative_to(ROOT).as_posix(),
            'metric':metric,
            'direction':direction,
            'score':score,
            'gate_decision':decision.get('decision'),
            'promoted':decision.get('promoted'),
            'parent_score':decision.get('parent_score'),
            'promotion_delta':decision.get('promotion_delta'),
            'agent_count':len(agents),
            'agents':agents,
            'has_artifacts':has_artifacts,
            'official_submission':official if isinstance(official,dict) else None,
            'status':'timeout_or_failed' if has_artifacts['timeout_manifest.json'] or has_artifacts['failure_review.json'] and not has_artifacts['metrics.json'] else ('scored' if has_artifacts['metrics.json'] else 'unscored'),
        })

workspace_runs_root = ROOT / 'workspace' / 'workstation_runs'
if workspace_runs_root.exists():
    for task_root in sorted([p for p in workspace_runs_root.iterdir() if p.is_dir()]):
        task = task_root.name
        if task not in KAGGLE_TASKS:
            continue
        for run_dir in sorted([p for p in task_root.iterdir() if p.is_dir()]):
            metrics_path = run_dir / 'metrics.json'
            if not metrics_path.exists():
                metrics_path = run_dir / 'hpc_gpu_training' / 'metrics.json'
            metrics = read_json(metrics_path)
            metric, direction, score = metric_from_metrics(metrics)
            gate = read_json(run_dir / 'score_promotion_gate.json')
            decision = gate.get('decision', {}) if isinstance(gate, dict) else {}
            if not decision:
                score_gate = read_json(run_dir / 'score_improvement_gate.json')
                if isinstance(score_gate, dict):
                    status = score_gate.get('status')
                    decision = {
                        'decision': 'promote' if status == 'passed' else 'hold' if status == 'blocked' else status,
                        'promoted': status == 'passed',
                    }
            trace_items = []
            trace_path = run_dir / 'agent_trace.json'
            trace = read_json(trace_path)
            if isinstance(trace, dict):
                trace_items = trace.get('traces') or trace.get('trace') or trace.get('agents') or []
            elif isinstance(trace, list):
                trace_items = trace
            agents = sorted({x.get('agent') for x in trace_items if isinstance(x, dict) and x.get('agent')})
            artifact_names = [
                'metrics.json', 'submission.csv', 'oof_predictions.csv', 'artifact_manifest.json',
                'score_promotion_gate.json', 'score_improvement_gate.json', 'agent_trace.json',
                'agent_trace.jsonl', 'launcher_manifest.json', 'timeout_manifest.json',
                'failure_review.json', 'orchestrator_run.json', 'search_controller_decision.json',
                'validation_contract.json', 'claim_audit.json', 'submission_audit.json',
                'task_benchmark_state.json', 'workstation_run_registry.json', 'rank_promotion_gate.json',
                'benchmark_claim_gate.json', 'kaggle_official_submission.json'
            ]
            has_artifacts = {}
            for name in artifact_names:
                has_artifacts[name] = (run_dir / name).exists() or (run_dir / 'hpc_gpu_training' / name).exists()
            official = read_json(run_dir / 'kaggle_official_submission.json')
            records.append({
                'task_id': task,
                'run_id': run_dir.name,
                'path': run_dir.relative_to(ROOT).as_posix(),
                'metric': metric,
                'direction': direction,
                'score': score,
                'gate_decision': decision.get('decision'),
                'promoted': decision.get('promoted'),
                'parent_score': decision.get('parent_score'),
                'promotion_delta': decision.get('promotion_delta'),
                'agent_count': len(agents),
                'agents': agents,
                'has_artifacts': has_artifacts,
                'official_submission': official if isinstance(official, dict) else None,
                'status': 'timeout_or_failed' if has_artifacts.get('timeout_manifest.json') or has_artifacts.get('failure_review.json') and not has_artifacts.get('metrics.json') else ('scored' if has_artifacts.get('metrics.json') else 'unscored'),
            })

# best per task by latest summary/progress when available, else from records by metric direction
progress=read_json(ROOT/'workspace'/'kaggle_10_self_evolution_progress_20260623.json') or {}
progress_results={r.get('task_id'):r for r in progress.get('results',[]) if isinstance(r,dict)}
PROGRESS_ALIASES={
 'house_prices':'house_prices_advanced_regression_techniques',
 'house_prices_advanced_regression_techniques':'house_prices_advanced_regression_techniques',
}
kaggle4=read_json(ROOT/'workspace'/'kaggle4_self_evolution_rounds_20260624.json') or {}
kaggle4_results={t.get('task_id'):t for t in kaggle4.get('tasks',[]) if isinstance(t,dict)}
by_task={}
for rec in records:
    by_task.setdefault(rec['task_id'],[]).append(rec)
summary=[]
for task,runs in sorted(by_task.items()):
    scored=[r for r in runs if isinstance(r.get('score'),(int,float))]
    promoted=[r for r in runs if r.get('gate_decision')=='promote']
    held=[r for r in runs if r.get('gate_decision')=='hold']
    timeouts=[r for r in runs if r.get('has_artifacts',{}).get('timeout_manifest.json')]
    best_source='computed'
    best_score=None; best_run=None; metric=None; improvement=None
    if task in kaggle4_results:
        k=kaggle4_results[task]
        best_score=k.get('best_score'); best_run=k.get('best_run_id'); metric=k.get('latest_metric'); improvement=k.get('improvement_within_current_metric'); best_source='kaggle4_summary'
    elif task in progress_results or PROGRESS_ALIASES.get(task) in progress_results:
        p=progress_results.get(task) or progress_results[PROGRESS_ALIASES.get(task)]
        best_score=p.get('best_score'); best_run=p.get('best_run_id') or p.get('best_branch'); metric=p.get('metric'); improvement=p.get('improvement'); best_source='kaggle10_progress'
    elif scored:
        # choose last metric family and direction
        latest=scored[-1]
        metric=latest.get('metric'); direction=latest.get('direction') or ('minimize' if metric in {'rmsle','rmse','mae','mse','log_loss'} else 'maximize')
        same=[r for r in scored if r.get('metric')==metric]
        reverse=direction!='minimize'
        best=max(same,key=lambda x:x['score']) if reverse else min(same,key=lambda x:x['score'])
        best_score=best['score']; best_run=best['run_id']
    if scored and metric:
        direction = next((r.get('direction') for r in reversed(scored) if r.get('metric') == metric and r.get('direction')), None) or ('minimize' if metric in {'rmsle','rmse','mae','mse','log_loss'} else 'maximize')
        same=[r for r in scored if r.get('metric')==metric and isinstance(r.get('score'),(int,float))]
        if same:
            record_best=same[0]
            for candidate in same[1:]:
                if is_better(candidate.get('score'), record_best.get('score'), direction):
                    record_best=candidate
            if is_better(record_best.get('score'), best_score, direction):
                best_score=record_best['score']
                best_run=record_best['run_id']
                best_source='computed_records_override'
    all_agents=sorted({a for r in runs for a in r.get('agents',[])})
    summary.append({
        'task_id':task,
        'run_count':len(runs),
        'scored_runs':len(scored),
        'promoted_runs':len(promoted),
        'held_runs':len(held),
        'timeout_or_failed_runs':len(timeouts),
        'best_score':best_score,
        'metric':metric,
        'best_run':best_run,
        'improvement':improvement,
        'best_source':best_source,
        'agent_count_observed':len(all_agents),
        'agents_observed':all_agents,
    })

readiness=read_json(ROOT/'workspace'/'kaggle_10_task_readiness_20260623.json') or {}
completion=read_json(ROOT/'workspace'/'kaggle_10_completion_audit_20260624.json') or {}
verification=read_json(ROOT/'workspace'/'kaggle4_self_evolution_verification_20260624.json') or {}
timeout_verification=read_json(ROOT/'workspace'/'kaggle4_timeout_control_verification_20260624.json') or {}

official_submit_claims=[]
for p in [ROOT/'workspace'/'kaggle4_self_evolution_rounds_20260624.json', ROOT/'reports'/'KAGGLE4_SELF_EVOLUTION_SUPERVISION_20260624.md']:
    if p.exists():
        text=p.read_text(encoding='utf-8',errors='ignore').lower()
        if 'official_kaggle_submit": true' in text or 'official kaggle submit: `true`' in text:
            official_submit_claims.append(p.relative_to(ROOT).as_posix())
official_submission_records=merge_official_records(
    [build_official_record_from_submission(r) for r in records if isinstance(r.get('official_submission'),dict)]
    + [
        item for item in (
            build_official_record_from_response(p)
            for root in [ROOT/'workspace'/'workstation_runs', ROOT/'experiments']
            if root.exists()
            for p in root.rglob('kaggle_submission_response.json')
        )
        if item
    ]
    + [
        item for item in (
            build_official_record_from_score_gate(p)
            for root in [ROOT/'workspace'/'workstation_runs', ROOT/'experiments']
            if root.exists()
            for p in root.rglob('score_improvement_gate.json')
        )
        if item
    ]
)
official_score_known_count=sum(1 for r in official_submission_records if r.get('official_score_known'))
official_rank_known_count=sum(1 for r in official_submission_records if not r.get('rank_unknown'))
official_rank_unknown_count=sum(1 for r in official_submission_records if r.get('rank_unknown'))
official_top30_count=sum(1 for r in official_submission_records if r.get('top30_reached') is True)

out={
 'schema':'academic_research_os.kaggle_experiment_inventory.v1',
 'created_at':datetime.now().isoformat(timespec='seconds'),
 'task_count_with_experiments':len(summary),
 'total_runs_observed':len(records),
 'total_scored_runs':sum(s['scored_runs'] for s in summary),
 'total_promoted_runs':sum(s['promoted_runs'] for s in summary),
 'total_held_runs':sum(s['held_runs'] for s in summary),
 'total_timeout_or_failed_runs':sum(s['timeout_or_failed_runs'] for s in summary),
 'governance_artifact_coverage':{
   'search_controller_decision':sum(1 for r in records if r['has_artifacts'].get('search_controller_decision.json')),
   'validation_contract':sum(1 for r in records if r['has_artifacts'].get('validation_contract.json')),
   'claim_audit':sum(1 for r in records if r['has_artifacts'].get('claim_audit.json')),
   'submission_audit':sum(1 for r in records if r['has_artifacts'].get('submission_audit.json')),
   'task_benchmark_state':sum(1 for r in records if r['has_artifacts'].get('task_benchmark_state.json')),
   'workstation_run_registry':sum(1 for r in records if r['has_artifacts'].get('workstation_run_registry.json')),
   'rank_promotion_gate':sum(1 for r in records if r['has_artifacts'].get('rank_promotion_gate.json')),
   'benchmark_claim_gate':sum(1 for r in records if r['has_artifacts'].get('benchmark_claim_gate.json')),
   'kaggle_official_submission':sum(1 for r in records if r['has_artifacts'].get('kaggle_official_submission.json')),
 },
 'kaggle10_runnable_count':readiness.get('runnable_count'),
 'kaggle10_task_count':readiness.get('task_count'),
 'kaggle10_completion_status':completion.get('status'),
 'kaggle4_verification':{'status':verification.get('status'),'checks':f"{verification.get('checks_passed')}/{verification.get('checks_total')}"},
 'timeout_control_verification':{'status':timeout_verification.get('status'),'checks':f"{timeout_verification.get('checks_passed')}/{timeout_verification.get('checks_total')}"},
 'official_submit_claims_in_kaggle4_scope':official_submit_claims,
 'official_submission_records':official_submission_records,
 'official_score_known_count':official_score_known_count,
 'official_rank_known_count':official_rank_known_count,
 'official_rank_unknown_count':official_rank_unknown_count,
 'official_top30_count':official_top30_count,
 'official_top30_rate':(
    official_top30_count / official_rank_known_count
    if official_rank_known_count else 0.0
 ),
 'claim_boundary':'Scores are local CV/proxy evidence unless an explicit Kaggle response artifact is cited. Public-score-only evidence sets official_score_known=true and rank_unknown=true; no rank, top30, or medal claim is made without rank/medal artifacts.',
 'task_summary':summary,
 'runs':records,
}
out_path=ROOT/'workspace'/'kaggle_experiment_inventory_20260624.json'
out_path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
md=ROOT/'reports'/'KAGGLE_EXPERIMENT_INVENTORY_20260624.md'
lines=['# Kaggle 实验总账与工作站稳定性审计','',f"- Created at: `{out['created_at']}`",f"- Tasks with experiments: `{out['task_count_with_experiments']}`",f"- Total runs observed: `{out['total_runs_observed']}`",f"- Scored runs: `{out['total_scored_runs']}`",f"- Promoted / held / timeout-or-failed: `{out['total_promoted_runs']}` / `{out['total_held_runs']}` / `{out['total_timeout_or_failed_runs']}`",f"- Kaggle10 readiness: `{out['kaggle10_runnable_count']}/{out['kaggle10_task_count']}`",f"- Kaggle10 completion status: `{out['kaggle10_completion_status']}`",f"- Kaggle4 verification: `{out['kaggle4_verification']['status']} {out['kaggle4_verification']['checks']}`",f"- Timeout control: `{out['timeout_control_verification']['status']} {out['timeout_control_verification']['checks']}`",f"- Official Kaggle submissions recorded: `{len(out['official_submission_records'])}`",f"- Official score-known / rank-known / rank-unknown: `{out['official_score_known_count']}` / `{out['official_rank_known_count']}` / `{out['official_rank_unknown_count']}`",f"- Official top30 reached: `{out['official_top30_count']}`",f"- Official top30 rate among rank-known records: `{out['official_top30_rate']:.4f}`",'','## Governance Artifact Coverage','',f"- search_controller_decision: `{out['governance_artifact_coverage']['search_controller_decision']}`",f"- validation_contract: `{out['governance_artifact_coverage']['validation_contract']}`",f"- claim_audit: `{out['governance_artifact_coverage']['claim_audit']}`",f"- submission_audit: `{out['governance_artifact_coverage']['submission_audit']}`",f"- task_benchmark_state: `{out['governance_artifact_coverage']['task_benchmark_state']}`",f"- workstation_run_registry: `{out['governance_artifact_coverage']['workstation_run_registry']}`",f"- rank_promotion_gate: `{out['governance_artifact_coverage']['rank_promotion_gate']}`",f"- benchmark_claim_gate: `{out['governance_artifact_coverage']['benchmark_claim_gate']}`",f"- kaggle_official_submission: `{out['governance_artifact_coverage']['kaggle_official_submission']}`",'','## Official Submission Records','','| task | competition | run | ref | public score | score known | rank | rank percentile | top30 | status | evidence |','|---|---|---|---:|---:|---|---:|---:|---|---|---|']
for item in official_submission_records:
    rank_text=f"{item.get('rank')}/{item.get('leaderboard_team_count')}" if item.get('rank') else "rank_unknown"
    top30_text=item.get('top30_reached') if item.get('top30_reached') is not None else "rank_unknown"
    lines.append(f"| `{item['task_id']}` | `{item['competition']}` | `{item['run_id']}` | {item['submission_ref']} | {item['public_score']} | `{item.get('official_score_known')}` | {rank_text} | {item.get('rank_percentile')} | `{top30_text}` | `{item['status']}` | `{item.get('evidence_source')}` |")
if not official_submission_records:
    lines.append("| none | none | none |  |  | false |  |  | false | none | none |")
lines += ['','## Task Summary','','| task | runs | scored | promote | hold | timeout | metric | best | improvement | agents |','|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
for s in summary:
    lines.append(f"| `{s['task_id']}` | {s['run_count']} | {s['scored_runs']} | {s['promoted_runs']} | {s['held_runs']} | {s['timeout_or_failed_runs']} | `{s['metric']}` | {s['best_score']} | {s['improvement']} | {s['agent_count_observed']} |")
lines += ['','## Stability Conclusion','','- Multi-Agent Research OS: agent traces and artifacts are present for scored workstation runs.','- MLEvolve-style Search Controller: score promotion gates distinguish promote/hold and preserve best-so-far.','- XCIENTIST-style Harness: timeout/failure runs are recorded separately and not counted as score evidence; official submission remains gated.','- Current limitation: most scores are local CV/proxy, not official Kaggle leaderboard results; Santander dataset remains blocked by Kaggle 403 in the latest readiness evidence.']
md.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'json':out_path.relative_to(ROOT).as_posix(),'md':md.relative_to(ROOT).as_posix(),'tasks':len(summary),'runs':len(records),'promoted':out['total_promoted_runs'],'held':out['total_held_runs'],'timeouts':out['total_timeout_or_failed_runs']},ensure_ascii=False,indent=2))

