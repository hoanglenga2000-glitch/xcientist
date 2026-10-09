"""Read failed EV attempt diagnostics only, after a fresh managed identity gate.

No samples, exception messages, log lines or credentials leave the HPC.
"""
import hashlib, http.client, inspect, json, shlex, sqlite3, sys, time
from pathlib import Path
from datetime import datetime, timezone

BASE = Path('C:/ProgramData/EvoMind')
STAGE = BASE/'staging/ev-public-calibration-20260908/runtime-extension'
sys.path[:0] = [str(BASE/'bundle/runtime'), 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/src']


def main():
    policy = json.loads((BASE/'config/official-calibration/ev-public-20260908.json').read_text())
    with sqlite3.connect(Path(policy['budget_path']).as_uri()+'?mode=ro', uri=True) as c:
        c.execute('PRAGMA query_only=ON')
        attempts = c.execute("SELECT id,run_id FROM attempts WHERE status='failed'").fetchall()
    roots = []
    for attempt, run in attempts:
        case = policy['cases'][run]
        if case['arm'] == 'evomind':
            roots.append({'attempt_id': attempt, 'case_id': case['case_id'],
                          'root': policy['artifact_root']+'/'+case['case_id']+'/'+attempt})
    token = (BASE/'data/workspace/runtime/runtime.token').read_text().strip()
    def api(method, path, body):
        c = http.client.HTTPConnection('127.0.0.1',8765,timeout=180)
        try:
            c.request(method,path,json.dumps(body).encode(),{'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            r=c.getresponse();data=r.read(2*1024*1024)
            if r.status not in {200,201}: raise ValueError('audit_api_failed')
            return json.loads(data)
        finally: c.close()
    binding=policy['managed_hpc_identity'];session='session_ev_failure_audit_'+str(time.time_ns())
    api('POST','/v1/sessions',{'session_id':session,'objective':'Read-only redacted EV failure diagnostics.',
        'permission_level':'observe','workspace_root':str(BASE/'data/acceptance/ev-calibration-20260908/observer'),
        'metadata':{'managed_hpc_identity':binding,'tenant_id':policy['tenant_id'],
                    'owner_principal_id':policy['owner_principal_id'],'run_allowed_tool_names':['hpc_verify']}})
    result=api('POST','/v1/sessions/'+session+'/tools',{'tool_name':'hpc_verify','arguments':{},'idempotency_key':'failure-audit-fresh-identity'})
    evidence=(result.get('result') or {}).get('content') or {}
    from xsci.terminal_tools import hpc_identity_evidence_complete
    if not hpc_identity_evidence_complete(evidence,expected_profile=binding['credential_profile'],expected_job_id=binding['job_id']):
        raise ValueError('audit_identity_failed')
    from evomind_runtime.tools import _load_bound_hpc_config
    from research_agent_workstation.server.core.gpu_credentials import connect_ssh
    client=connect_ssh(_load_bound_hpc_config(binding['credential_profile'],binding['job_id'],session),timeout=30)
    from evomind_runtime.ev_calibration_diagnostics import summarize_error
    program = 'import json,re,hashlib\n'+inspect.getsource(summarize_error)+'\n'+'''import json,re,hashlib
from pathlib import Path
rows=ROOTS_VALUE
for row in rows:
 p=Path(row.pop('root'))/'operator-error.txt'
 if not p.is_file() or p.is_symlink() or p.stat().st_size>65536:
  row['status']='diagnostic_not_available';continue
 row.update(summarize_error(p.read_bytes()))
print(json.dumps(rows))
'''.replace('ROOTS_VALUE',repr(roots))
    try:
        _,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(program),timeout=30)
        data=stdout.read(131072);code=stdout.channel.recv_exit_status()
        if code: raise ValueError('audit_remote_read_failed')
        findings=json.loads(data)
    finally: client.close()
    receipt={'schema':'evomind.ev_failure_audit.v1','at':datetime.now(timezone.utc).isoformat(),
             'read_only':True,'fresh_identity_samples':5,'findings':findings}
    (STAGE/'service-output'/('failure-audit-'+str(time.time_ns())+'.json')).write_text(json.dumps(receipt,indent=2))
    for run,case in policy['cases'].items():
        own=[row for row in findings if row['case_id']==case['case_id']]
        if own:
            path=STAGE/'service-output'/('infrastructure-diagnostics-'+run+'.json')
            with path.open('x') as f:json.dump({'run_id':run,'case_id':case['case_id'],'findings':own},f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':
    try: main()
    except Exception as e:
        (STAGE/'service-output/failure-audit-error.json').write_text(json.dumps({'error_type':type(e).__name__,'at':datetime.now(timezone.utc).isoformat()}))
        raise SystemExit(2)
