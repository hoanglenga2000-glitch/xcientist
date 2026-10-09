"""Service-owner coordinator: native EvoMind model/tool loops, nine isolated cases."""
from __future__ import annotations
import argparse,hashlib,http.client,json,sqlite3,sys,time
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/runtime-extension'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']
from evomind_runtime.siim_calibration_control import CaseBudget,TOOLS
from evomind_runtime.siim_candidate_contract import admitted_metrics,controller_completed,reconcile_summaries


def run_suite():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-key',default='initial');parser.add_argument('--parent-run',default='');options=parser.parse_args()
    if not options.resume_key.replace('-','').replace('_','').isalnum() or len(options.resume_key)>40:raise ValueError('Invalid resume key')
    policy_path=BASE/'config/official-calibration/siim-mlebench-20260908.json'
    raw=policy_path.read_bytes();policy=json.loads(raw);policy_sha=hashlib.sha256(raw).hexdigest()
    output=STAGE/'service-output';output.mkdir(exist_ok=True)
    control_path=output/'suite-started.json'
    if control_path.exists() and not options.resume:raise ValueError('Suite already dispatched; reconcile instead of duplicate')
    if options.resume:
        original=json.loads(control_path.read_text())
        lineage={policy_sha,policy.get('supersedes_policy_sha256')}|set(policy.get('supersedes_chain') or [])
        if original['policy_sha256'] not in lineage:raise ValueError('Resume policy differs')
    config=json.loads((BASE/'config/node-config.json').read_text(encoding='utf-8-sig'))
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    parent_store=None
    if options.parent_run:
        from evomind_runtime.store import RuntimeStore
        from evomind_runtime.siim_calibration_web import require_parent
        parent_store=RuntimeStore(BASE/'data/workspace/runtime/runtime.sqlite3')
        require_parent(parent_store,options.parent_run,policy)
    def request(method,path,body=None,timeout=120):
        if options.parent_run and method!='GET':require_parent(parent_store,options.parent_run,policy)
        c=http.client.HTTPConnection('127.0.0.1',int(config['network']['runtime_port']),timeout=timeout)
        try:
            c.request(method,path,body=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();data=r.read(4*1024*1024+1)
            if len(data)>4*1024*1024 or r.status not in {200,201}:raise ValueError('runtime_request_rejected')
            return json.loads(data)
        finally:c.close()
    def save(name,value):
        p=output/name;tmp=p.with_suffix(p.suffix+'.new');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
    def defer(case_id,run_id,row,entry):
        # A wedged case must not strand the remaining registered cases, and an arm that
        # raises (rather than returning a non-completed status) is still only one case.
        deferred.append({'run_id':run_id,'case_id':case_id,'arm':row['arm'],'seed':row['seed'],**entry})
        save('progress.json',{'status':'running','active_case':case_id,'run_id':run_id,'completed_cases':len(summaries),
             'deferred':deferred,'budget':ledger.summary(),'official_submissions':0})
    ledger=CaseBudget(policy['budget_path'],policy['ev_budget_path'],policy.get('resource_limits'))
    step_ceiling=(policy.get('resource_limits') or {}).get('step_ceiling_seconds',7200)
    max_steps=(policy.get('resource_limits') or {}).get('max_agent_steps',24)
    if not options.resume:
        save('suite-started.json',{'schema':'evomind.siim_suite_start.v1','at':datetime.now(timezone.utc).isoformat(),
                                  'policy_sha256':policy_sha,'cases':list(policy['cases']),'automatic_submission':False})
    else:
        if ledger.summary()['settlement_attention']:raise ValueError('Resume blocked by unsettled execution')
        save('resume-'+str(time.time_ns())+'.json',{'at':datetime.now(timezone.utc).isoformat(),'reason':options.resume_key,'counters_reset':False})
    admitted_path=output/'case-results-admitted-v2.json'
    original_path=admitted_path if admitted_path.exists() else output/'case-results.json'
    original_summaries=json.loads(original_path.read_text(encoding='utf-8')) if original_path.exists() else []
    summaries,corrections=reconcile_summaries(original_summaries,policy['cases'])
    if corrections:
        save('completion-admission-audit-'+str(time.time_ns())+'.json',{'source_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
             'findings':corrections,'original_records_preserved':True,'budget_and_clocks_reset':False})
    save('case-results-admitted-v2.json',summaries)
    deferred=[]
    # Priority is the arm that still has no admitted case: the registered comparison needs
    # three frozen arms, so an arm with zero admissions gates the whole deliverable, while an
    # arm that already has an admitted seed can afford to be retried last.
    for run_id,row in sorted(policy['cases'].items(),key=lambda item:({'aide':0,'fixed_baseline':1,'evomind':2}[item[1]['arm']],item[1]['seed'])):
        if any(r['run_id']==run_id for r in summaries):continue
        if row['arm']=='aide' and not (STAGE/'siim_aide_controller.py').is_file():
            save('progress.json',{'status':'awaiting_aide_adapter','completed_cases':len(summaries),'budget':ledger.summary(),'official_submissions':0});return 2
        started=time.monotonic();case_id=row['case_id']
        if hashlib.sha256(policy_path.read_bytes()).hexdigest()!=policy_sha:raise ValueError('policy_changed')
        with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
            exists=c.execute('SELECT 1 FROM sessions WHERE id=?',(run_id,)).fetchone() is not None
        if not exists:
            request('POST','/v1/sessions',{'session_id':run_id,'title':'SIIM '+case_id,'objective':'SIIM MLE-bench comparison '+case_id,
                 'permission_level':'workspace-write','workspace_root':row['workspace_root'],
                 'metadata':{'managed_hpc_identity':policy['managed_hpc_identity'],'tenant_id':policy['tenant_id'],
                             'owner_principal_id':policy['owner_principal_id'],'project_id':'siim_mlebench_calibration_20260908',
                             'siim_calibration':{'policy_sha256':policy_sha},'run_allowed_tool_names':sorted(TOOLS),
                             'independent_context':True,'no_official_score_feedback':True}})
        else:
            current=request('GET','/v1/sessions/'+run_id)
            if current['metadata'].get('siim_calibration')!={'policy_sha256':policy_sha}:raise ValueError('Existing case binding differs')
            if current['metadata'].get('user_pause_requested') or current.get('status') in {'paused', 'pausing', 'blocked', 'failed'}:
                if not options.resume:raise ValueError('Case is paused; explicit resume required')
                request('POST','/v1/sessions/'+run_id+'/resume',{})
        if options.parent_run:
            require_parent(parent_store,options.parent_run,policy)
            current=parent_store.get_session(run_id)
            metadata={**current['metadata'],'siim_web_parent_run':options.parent_run}
            parent_store.update_session(run_id,metadata_json=metadata)
            parent_store.append_event(run_id,'siim.web_parent_bound',{'parent_run_id':options.parent_run,
                'prior_history_preserved':True,'prior_history_was_web_started':False})
        workspace=Path(row['workspace_root']);(workspace/'inputs').mkdir(exist_ok=True)
        (workspace/'work/solutions').mkdir(parents=True,exist_ok=True);(workspace/'outputs').mkdir(exist_ok=True)
        for name in ['task.json','data-profile.json']:
            body=(STAGE/'cases'/case_id/name).read_bytes()
            if hashlib.sha256(body).hexdigest()!=row['input_hashes']['inputs/'+name]:raise ValueError('case_input_changed')
            destination=workspace/'inputs'/name
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=row['input_hashes']['inputs/'+name]:raise ValueError('Existing case input differs')
            else:
                with destination.open('xb') as f:f.write(body)
        save('progress.json',{'status':'running','active_case':case_id,'run_id':run_id,'completed_cases':len(summaries),'budget':ledger.summary()})
        with sqlite3.connect(policy['budget_path']) as c:clock=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()
        if clock is None:ledger.start_case(run_id)
        if row['arm']=='fixed_baseline':
            code=(STAGE/'baseline.py').read_bytes()
            if hashlib.sha256(code).hexdigest()!=policy['baseline_source_sha256']:raise ValueError('baseline_source_changed')
            destination=workspace/'work/solutions/baseline.py'
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=policy['baseline_source_sha256']:raise ValueError('Existing baseline changed')
            else:
                with destination.open('xb') as f:f.write(code)
            for name,args in [('training_route',{'path':'inputs/task.json','task_description':'Registered SIIM comparison on the managed HPC; no local fitting.'}),('hpc_verify',{})]:
                result=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':name,'arguments':args,'idempotency_key':case_id+'-'+name+'-schema-v2-'+options.resume_key},timeout=180)
                if not (result.get('result') or {}).get('ok'):raise ValueError('baseline_preflight_failed')
            try:
                result=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':'hpc_execute_solution',
                     'arguments':{'solution_id':case_id,'script_path':'work/solutions/baseline.py','data_dir':'inputs','timeout_seconds':step_ceiling},
                     'idempotency_key':case_id+'-fixed-fit-'+options.resume_key},timeout=step_ceiling+300)
            except Exception as error:
                defer(case_id,run_id,row,{'reason':'arm_execution_exception','error_class':type(error).__name__,'code':str(error)[:200]})
                continue
        elif row['arm']=='aide':
            from siim_aide_controller import execute_aide_case
            try:
                result=execute_aide_case(run_id,row,policy,request,resume_key=options.resume_key)
            except Exception as error:
                defer(case_id,run_id,row,{'reason':'arm_execution_exception','error_class':type(error).__name__,'code':str(error)[:200]})
                continue
        else:
            prompt=('You are running one independent SIIM image and metadata experiment. Read inputs/task.json and inputs/data-profile.json first. '
                    'Design the feature processing and classifier yourself, using only the available packages and those summaries. '
                    'The raw frames remain on HPC and the protected scorer supplies fold-local frames to your fit_model function. '
                    'Follow the exact function and tool interface in task.json. Before fitting call training_route, then hpc_verify. '
                    'Write each candidate to work/solutions/<name>.py and invoke hpc_execute_solution with data_dir="inputs", no competition argument. '
                    'You may improve candidates using complete OOF AUC within the source-based time ceilings and automatic remaining-work allocation returned by the tools. '
                    'Do not read other experiments, competition solutions, leaderboards, memory, raw predictions, or perform downloads/submissions. '
                    'Select your highest complete OOF candidate; record its source name, candidate hash and returned OOF metric in outputs/summary.md and publish it. '
                    'Use the frozen seed '+str(row['seed'])+'. No human model-design intervention is permitted.')
            if options.resume:
                prompt+=(' Infrastructure clarification: fit_model is called by the protected runner; no CLI entry point is needed. '
                         'Admission error line numbers refer to your own module. Constant interface probes are diagnostic only, '
                         'not trained candidates. Failed or blocked controller execution cannot complete a case. '
                         'Your existing candidates, case start, charges and remaining allocation are unchanged. '
                         'Do not read forbidden raw logs; use the sanitized execution receipt and diagnostics returned by the tool.')
            diagnostic_path=output/('infrastructure-diagnostics-'+run_id+'.json')
            if options.resume and diagnostic_path.is_file():
                diagnostic=json.loads(diagnostic_path.read_text())
                if diagnostic.get('run_id')!=run_id or diagnostic.get('case_id')!=case_id:raise ValueError('Diagnostic case mismatch')
                prompt+=(' Infrastructure notice: sanitized worker error metadata is now available. '
                         'Automatically recovered diagnostics from your own previous attempts follow; no modeling recommendation is included. '
                         'Keep using the original case clock, candidates and remaining budget. '+json.dumps(diagnostic))
                save('diagnostic-feedback-'+run_id+'-'+options.resume_key+'.json',{
                    'run_id':run_id,'diagnostic_sha256':hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
                    'modeling_recommendations_provided':False,'case_clock_reset':False})
            try:
                result=request('POST','/v1/sessions/'+run_id+'/messages',{'content':prompt,'max_steps':max_steps},timeout=86700)
            except Exception as error:
                defer(case_id,run_id,row,{'reason':'arm_execution_exception','error_class':type(error).__name__,'code':str(error)[:200]})
                continue
        with sqlite3.connect(policy['budget_path']) as c:case_started=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()[0]
        elapsed=time.time()-case_started
        controller_ok=controller_completed(row['arm'],result)
        # Admission rests on the immutable tool records, not on the controller's closing
        # prose: a run that ends non-completed can still carry candidates that were fitted,
        # independently recomputed and hash-bound. Constant probes are still rejected by
        # admitted_metrics, so this widens evidence, not claims.
        db=BASE/'data/workspace/runtime/runtime.sqlite3'
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
            c.execute('PRAGMA query_only=ON')
            rows=c.execute("SELECT result_json FROM tool_calls WHERE session_id=? AND tool_name='hpc_execute_solution' AND status='completed'",(run_id,)).fetchall()
            accepted=[]
            for (raw_result,) in rows:
                record=json.loads(raw_result);content=record.get('content') or {};metrics=content.get('metrics') or {}
                if record.get('ok') and admitted_metrics(metrics,workspace):accepted.append(metrics)
            model_events=[json.loads(r[0]) for r in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq",(run_id,))]
        if not accepted:
            deferred.append({'run_id':run_id,'case_id':case_id,'arm':row['arm'],'seed':row['seed'],
                 'reason':'controller_not_completed' if not controller_ok else 'no_verified_complete_candidate',
                 'controller_status':result.get('status')})
            save('progress.json',{'status':'running','active_case':case_id,'run_id':run_id,
                 'completed_cases':len(summaries),'deferred':deferred,'budget':ledger.summary(),'official_submissions':0})
            continue
        best=min(accepted,key=lambda x:(-x['oof_roc_auc'],x['runtime_seconds']))
        summary={**best,'run_id':run_id,'case_id':case_id,'candidate_fit_seconds':best['runtime_seconds'],
                 'runtime_seconds':elapsed,'successful_candidates':len(accepted),'model_events':model_events,
                 'status':'completed','official_score_seen_before_freeze':False,
                 'admission_basis':'verified_candidate_evidence','controller_completed':controller_ok,
                 'controller_status':result.get('status')}
        summaries.append(summary);save(case_id+'-admitted-'+options.resume_key+'.json',summary);save('case-results-admitted-v2.json',summaries)
        ledger.mark_complete(run_id)
        if ledger.summary()['settlement_attention']:raise ValueError('settlement_requires_reconciliation')
    if deferred:
        # The study is not complete; evidence stays auditable and the next dispatch retries these cases.
        save('progress.json',{'status':'blocked','active_case':deferred[0]['case_id'],'run_id':deferred[0]['run_id'],
             'reason':'cases_deferred','deferred':deferred,'completed_cases':len(summaries),
             'budget':ledger.summary(),'official_submissions':0})
        return 2
    selected={arm:min((x for x in summaries if x['arm']==arm),key=lambda x:(-x['oof_roc_auc'],x['runtime_seconds'])) for arm in ['fixed_baseline','evomind','aide']}
    save('selected-candidates.json',{'schema':'evomind.siim_selected_before_private_scoring.v1','at':datetime.now(timezone.utc).isoformat(),
                                    'selected':selected,'official_scores_seen':False,'official_submissions':0,'budget':ledger.summary()})
    save('progress.json',{'status':'candidates_frozen_pending_private_grader_review','completed_cases':9,'budget':ledger.summary(),'official_submissions':0})
    return 0


def main():
    from evomind_runtime.execution_progress import resource_lease
    with resource_lease(STAGE/'service-output/suite-leases','siim-calibration-suite'):
        return run_suite()


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        p=STAGE/'service-output/suite-error.json';p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({'status':'blocked','error_type':type(e).__name__,'code':str(e) if isinstance(e,ValueError) else 'suite_execution_interrupted','official_submissions':0}),encoding='utf-8')
        raise SystemExit(2)
