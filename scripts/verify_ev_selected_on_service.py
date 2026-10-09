"""Managed, read-only re-computation of the six completed EV results before submission."""
import hashlib,http.client,json,shlex,sqlite3,sys,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/ev-public-calibration-20260908/runtime-extension'
sys.path[:0]=[str(BASE/'bundle/runtime'),'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']

def main():
    policy=json.loads((BASE/'config/official-calibration/ev-public-20260908.json').read_text())
    selected_path=STAGE/'service-output/selected-candidates.json'
    if not selected_path.is_file():raise ValueError('six_cases_and_selection_required')
    selected=json.loads(selected_path.read_text());cases=json.loads((STAGE/'service-output/case-results.json').read_text())
    if len(cases)!=6 or set((c['arm'],c['seed']) for c in cases)!={(a,s) for a in ['fixed_baseline','evomind'] for s in [17,29,43]}:
        raise ValueError('six_distinct_cases_required')
    if selected.get('official_scores_seen') is not False or selected.get('official_submissions')!=0:raise ValueError('pre_score_freeze_required')
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA query_only=ON');budget=c.execute('SELECT charged,status FROM attempts').fetchall()
        if sum(r[0] for r in budget)>86400 or any(r[1] not in {'completed','failed'} for r in budget):raise ValueError('budget_or_settlement_failed')
    for case in cases:
        if case['status']!='completed' or not case['oof_complete'] or case['runtime_seconds']>10800:raise ValueError('case_contract_failed')
    token=(BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def api(method,path,body):
        c=http.client.HTTPConnection('127.0.0.1',8765,timeout=180)
        try:
            c.request(method,path,json.dumps(body).encode(),{'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();raw=r.read(2*1024*1024)
            if r.status not in {200,201}:raise ValueError('audit_api_failed')
            return json.loads(raw)
        finally:c.close()
    binding=policy['managed_hpc_identity'];session='session_ev_frozen_audit_'+str(time.time_ns())
    api('POST','/v1/sessions',{'session_id':session,'objective':'Independently recompute six frozen OOF and test-file checks; no fitting/submission.',
        'permission_level':'observe','workspace_root':str(BASE/'data/acceptance/ev-calibration-20260908/observer'),
        'metadata':{'managed_hpc_identity':binding,'tenant_id':policy['tenant_id'],'owner_principal_id':policy['owner_principal_id'],
                    'run_allowed_tool_names':['hpc_verify']}})
    verified=api('POST','/v1/sessions/'+session+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'fresh-final-validation-identity'})
    evidence=(verified.get('result') or {}).get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not hpc_identity_evidence_complete(evidence,expected_profile=binding['credential_profile'],expected_job_id=binding['job_id']):raise ValueError('audit_identity_failed')
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client=connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],session),timeout=30)
    raw=(STAGE/'verify_ev_hpc_saved_predictions.py').read_bytes();source_sha=hashlib.sha256(raw).hexdigest()
    target='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/ev_calibration_results/verify-saved-'+source_sha+'.py'
    results=[]
    try:
        sftp=client.open_sftp()
        try:
            try:
                with sftp.open(target,'rb') as stream:existing=stream.read(131073)
            except FileNotFoundError:
                with sftp.open(target,'wx') as stream:stream.write(raw)
                sftp.chmod(target,0o444)
                with sftp.open(target,'rb') as stream:existing=stream.read(131073)
            if hashlib.sha256(existing).hexdigest()!=source_sha:raise ValueError('verifier_upload_changed')
        finally:sftp.close()
        for case in cases:
            root=case['hpc_artifact_root']
            if not root.startswith(policy['artifact_root']+'/'+case['case_id']+'/'):raise ValueError('case_root_mismatch')
            _,stdout,stderr=client.exec_command('python3 '+shlex.quote(target)+' --root '+shlex.quote(root),timeout=120)
            raw=stdout.read(65537);status=stdout.channel.recv_exit_status()
            if status or len(raw)>65536:raise ValueError('independent_prediction_validation_failed')
            result=json.loads(raw)
            if result['status']!='passed' or result['submission_sha256']!=case['submission_sha256'] or abs(result['oof_roc_auc']-case['oof_roc_auc'])>1e-6:
                raise ValueError('independent_result_mismatch')
            results.append({**result,'run_id':case['run_id'],'case_id':case['case_id']})
    finally:client.close()
    for arm in ['fixed_baseline','evomind']:
        best=min((c for c in cases if c['arm']==arm),key=lambda c:(-c['oof_roc_auc'],c['runtime_seconds']))
        if best['candidate_sha256']!=selected['selected'][arm]['candidate_sha256']:raise ValueError('selection_rule_mismatch')
    receipt={'schema':'evomind.ev_pre_submission_verification.v1','status':'passed','at':datetime.now(timezone.utc).isoformat(),
        'results':results,'verifier_sha256':source_sha,'selected_manifest_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        'hpc_identity':evidence,'charged_slot_seconds':sum(r[0] for r in budget),'model_fitting':False,'official_submissions':0}
    output=STAGE/'service-output/pre-submission-verification.json'
    with output.open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps({'status':'passed','verified_cases':len(results),'receipt':str(output),'official_submissions':0}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'blocked','error_type':type(e).__name__,'detail':str(e) if isinstance(e,ValueError) else 'verification_failed'}));raise SystemExit(2)
