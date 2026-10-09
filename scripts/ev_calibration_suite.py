"""Service-owner coordinator: native EvoMind model/tool loops, six isolated cases."""
from __future__ import annotations
import argparse,hashlib,http.client,json,sqlite3,sys,time
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/ev-public-calibration-20260908/runtime-extension'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']
from evomind_runtime.ev_calibration_control import CaseBudget,TOOLS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--resume-key',default='initial');options=parser.parse_args()
    if not options.resume_key.replace('-','').replace('_','').isalnum() or len(options.resume_key)>40:raise ValueError('Invalid resume key')
    policy_path=BASE/'config/official-calibration/ev-public-20260908.json'
    raw=policy_path.read_bytes();policy=json.loads(raw);policy_sha=hashlib.sha256(raw).hexdigest()
    output=STAGE/'service-output';output.mkdir(exist_ok=True)
    control_path=output/'suite-started.json'
    if control_path.exists() and not options.resume:raise ValueError('Suite already dispatched; reconcile instead of duplicate')
    if options.resume:
        original=json.loads(control_path.read_text())
        if original['policy_sha256']!=policy_sha:raise ValueError('Resume policy differs')
    config=json.loads((BASE/'config/node-config.json').read_text(encoding='utf-8-sig'))
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def request(method,path,body=None,timeout=120):
        c=http.client.HTTPConnection('127.0.0.1',int(config['network']['runtime_port']),timeout=timeout)
        try:
            c.request(method,path,body=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();data=r.read(4*1024*1024+1)
            if len(data)>4*1024*1024 or r.status not in {200,201}:raise ValueError('runtime_request_rejected')
            return json.loads(data)
        finally:c.close()
    def save(name,value):
        p=output/name;tmp=p.with_suffix(p.suffix+'.new');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
    ledger=CaseBudget(policy['budget_path'])
    if not options.resume:
        save('suite-started.json',{'schema':'evomind.ev_suite_start.v1','at':datetime.now(timezone.utc).isoformat(),
                                  'policy_sha256':policy_sha,'cases':list(policy['cases']),'automatic_submission':False})
    else:
        if ledger.summary()['settlement_attention']:raise ValueError('Resume blocked by unsettled execution')
        save('resume-'+str(time.time_ns())+'.json',{'at':datetime.now(timezone.utc).isoformat(),'reason':options.resume_key,'counters_reset':False})
    summaries=json.loads((output/'case-results.json').read_text()) if (output/'case-results.json').exists() else []
    for run_id,row in policy['cases'].items():
        if any(r['run_id']==run_id for r in summaries):continue
        started=time.monotonic();case_id=row['case_id']
        if hashlib.sha256(policy_path.read_bytes()).hexdigest()!=policy_sha:raise ValueError('policy_changed')
        with sqlite3.connect((BASE/'data/workspace/runtime/runtime.sqlite3').as_uri()+'?mode=ro',uri=True) as c:
            exists=c.execute('SELECT 1 FROM sessions WHERE id=?',(run_id,)).fetchone() is not None
        if not exists:
            request('POST','/v1/sessions',{'session_id':run_id,'title':'EV '+case_id,'objective':'EV official calibration '+case_id,
                 'permission_level':'workspace-write','workspace_root':row['workspace_root'],
                 'metadata':{'managed_hpc_identity':policy['managed_hpc_identity'],'tenant_id':policy['tenant_id'],
                             'owner_principal_id':policy['owner_principal_id'],'project_id':'ev_public_calibration_20260908',
                             'official_calibration':{'policy_sha256':policy_sha},'run_allowed_tool_names':sorted(TOOLS),
                             'independent_context':True,'no_official_score_feedback':True}})
        else:
            current=request('GET','/v1/sessions/'+run_id)
            if current['metadata'].get('official_calibration')!={'policy_sha256':policy_sha}:raise ValueError('Existing case binding differs')
            if current['metadata'].get('user_pause_requested'):
                if not options.resume:raise ValueError('Case is paused; explicit resume required')
                request('POST','/v1/sessions/'+run_id+'/resume',{})
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
            for name,args in [('training_route',{'path':'inputs/task.json','task_description':'Registered EV calibration on the managed HPC; no local fitting.'}),('hpc_verify',{})]:
                result=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':name,'arguments':args,'idempotency_key':case_id+'-'+name+'-schema-v2'},timeout=180)
                if not (result.get('result') or {}).get('ok'):raise ValueError('baseline_preflight_failed')
            result=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':'hpc_execute_solution',
                 'arguments':{'solution_id':case_id,'script_path':'work/solutions/baseline.py','data_dir':'inputs','timeout_seconds':10800},
                 'idempotency_key':case_id+'-fixed-fit-'+options.resume_key},timeout=11200)
        else:
            prompt=('You are running one independent EV prediction experiment. Read inputs/task.json and inputs/data-profile.json first. '
                    'Design the feature processing and classifier yourself, using only the available packages and those summaries. '
                    'The raw frames remain on HPC and the protected scorer supplies fold-local frames to your fit_model function. '
                    'Follow the exact function and tool interface in task.json. Before fitting call training_route, then hpc_verify. '
                    'Write each candidate to work/solutions/<name>.py and invoke hpc_execute_solution with data_dir="inputs", no competition argument. '
                    'You may improve candidates using complete OOF AUC within the remaining three-hour case budget. '
                    'Do not read other experiments, competition solutions, leaderboards, memory, raw predictions, or perform downloads/submissions. '
                    'Select your highest complete OOF candidate; record its source name, candidate hash and returned OOF metric in outputs/summary.md and publish it. '
                    'Use the frozen seed '+str(row['seed'])+'. No human model-design intervention is permitted.')
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
            result=request('POST','/v1/sessions/'+run_id+'/messages',{'content':prompt,'max_steps':24},timeout=11200)
        with sqlite3.connect(policy['budget_path']) as c:case_started=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()[0]
        elapsed=time.time()-case_started
        # Read actual tool records; prose alone never supplies an accepted score.
        db=BASE/'data/workspace/runtime/runtime.sqlite3'
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
            c.execute('PRAGMA query_only=ON')
            rows=c.execute("SELECT result_json FROM tool_calls WHERE session_id=? AND tool_name='hpc_execute_solution' AND status='completed'",(run_id,)).fetchall()
            accepted=[]
            for (raw_result,) in rows:
                record=json.loads(raw_result);content=record.get('content') or {};metrics=content.get('metrics') or {}
                if record.get('ok') and metrics.get('oof_complete') is True:accepted.append(metrics)
            model_events=[json.loads(r[0]) for r in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq",(run_id,))]
        if not accepted:
            save('progress.json',{'status':'blocked','active_case':case_id,'run_id':run_id,'reason':'no_verified_complete_candidate','completed_cases':len(summaries),'budget':ledger.summary()})
            return 2
        best=min(accepted,key=lambda x:(-x['oof_roc_auc'],x['runtime_seconds']))
        summary={**best,'run_id':run_id,'case_id':case_id,'candidate_fit_seconds':best['runtime_seconds'],
                 'runtime_seconds':elapsed,'successful_candidates':len(accepted),'model_events':model_events,
                 'status':'completed','official_score_seen_before_freeze':False}
        summaries.append(summary);save(case_id+'.json',summary);save('case-results.json',summaries)
        if ledger.summary()['settlement_attention']:raise ValueError('settlement_requires_reconciliation')
    selected={arm:min((x for x in summaries if x['arm']==arm),key=lambda x:(-x['oof_roc_auc'],x['runtime_seconds'])) for arm in ['fixed_baseline','evomind']}
    save('selected-candidates.json',{'schema':'evomind.ev_selected_before_official_scoring.v1','at':datetime.now(timezone.utc).isoformat(),
                                    'selected':selected,'official_scores_seen':False,'official_submissions':0,'budget':ledger.summary()})
    save('progress.json',{'status':'awaiting_two_explicit_submission_confirmations','completed_cases':6,'budget':ledger.summary(),'official_submissions':0})
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        p=STAGE/'service-output/suite-error.json';p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({'status':'blocked','error_type':type(e).__name__,'code':str(e) if isinstance(e,ValueError) else 'suite_execution_interrupted','official_submissions':0}),encoding='utf-8')
        raise SystemExit(2)
