"""Invoke only hpc_verify through the existing service owner; no model or training.

The service loads its own named DPAPI profile. This controller never decrypts
credentials, changes a binding, or connects directly to an HPC address.
"""
from datetime import datetime, timezone
import argparse
import http.client
import json
from pathlib import Path
import sqlite3

ROOT=Path('C:/ProgramData/EvoMind')
STAGE=ROOT/'staging/invitation-7925595228da-sys1'
OLD_RUN='run_0c57bbde44e94a988843c2f6981f3c65'
SESSION='session_syscomplete_gpt55_hpc_identity_v3'
PROFILE='tenant_ba0ef9d3767f2fb385b856e0_job93207_g27'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--continue-schema-rejected',action='store_true')
    args=parser.parse_args()
    checkpoint=STAGE/'hpc-readonly-dispatch.json'
    receipt=STAGE/'hpc-readonly-result.json'
    if args.continue_schema_rejected:
        previous=json.loads(receipt.read_text())
        if (previous.get('session_id')!=SESSION or previous.get('status')!='failed'
                or previous.get('result',{}).get('summary')!='invalid tool arguments'
                or previous.get('result',{}).get('content')!={}):
            raise ValueError('previous_action_not_a_schema_rejection')
        checkpoint=STAGE/'hpc-readonly-dispatch-corrected.json'
        receipt=STAGE/'hpc-readonly-result-corrected.json'
    if checkpoint.exists() or receipt.exists():
        raise ValueError('existing_preflight_requires_reconciliation')
    database=ROOT/'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        row=connection.execute('SELECT metadata_json FROM sessions WHERE id=?',(OLD_RUN,)).fetchone()
        identity=json.loads(row[0])['managed_hpc_identity']
        exists=connection.execute('SELECT 1 FROM sessions WHERE id=?',(SESSION,)).fetchone()
        if exists and not args.continue_schema_rejected:
            raise ValueError('preflight_session_already_exists')
        if args.continue_schema_rejected:
            rows=connection.execute('SELECT status,result_json FROM tool_calls WHERE session_id=?',(SESSION,)).fetchall()
            if len(rows)!=1 or rows[0][0]!='failed' or json.loads(rows[0][1]).get('summary')!='invalid tool arguments':
                raise ValueError('existing_tool_requires_reconciliation')
    if (identity.get('credential_profile')!=PROFILE or int(identity.get('job_id',0))!=93207
            or int(identity.get('allocation_generation',0))!=27
            or identity.get('profile_instance_id')!='274b6681-9a34-4c8e-990b-fec4e725c6ba'):
        raise ValueError('bound_allocation_changed')
    config=json.loads((ROOT/'config/node-config.json').read_text(encoding='utf-8-sig'))
    port=int(config['network']['runtime_port'])
    token=(ROOT/'data/workspace/runtime/runtime.token').read_text().strip()
    def request(method,path,body=None):
        data=json.dumps(body).encode() if body is not None else None
        connection=http.client.HTTPConnection('127.0.0.1',port,timeout=240)
        try:
            connection.request(method,path,body=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            response=connection.getresponse()
            raw=response.read(2*1024*1024+1)
            if len(raw)>2*1024*1024 or response.status not in {200,201}:
                raise ValueError('managed_preflight_http_failed')
            return json.loads(raw)
        finally:
            connection.close()
    plan={'schema':'evomind.hpc_readonly_dispatch.v1','session_id':SESSION,'profile':PROFILE,'job_id':93207,
          'started_at':datetime.now(timezone.utc).isoformat(),'tool':'hpc_verify',
          'model_requests':0,'training_started':False,'binding_changed':False}
    with checkpoint.open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,indent=2)
    body={'session_id':SESSION,'objective':'Read-only allocation identity preflight. No execution or downloads.',
        'permission_level':'observe','workspace_root':str(ROOT/'data/acceptance/gpt55-hpc-readonly-v3'),
        'metadata':{'managed_hpc_identity':identity,'tenant_id':identity['tenant_id'],
            'owner_principal_id':identity['owner_principal_id'],'run_allowed_tool_names':['hpc_verify'],
            'acceptance_scope':'readonly_identity_no_model_no_training'}}
    created=request('GET','/v1/sessions/'+SESSION) if args.continue_schema_rejected else request('POST','/v1/sessions',body)
    if created.get('id')!=SESSION:
        raise ValueError('preflight_session_identity_mismatch')
    if created.get('metadata',{}).get('managed_hpc_identity')!=identity or created.get('permission_level')!='observe':
        raise ValueError('preflight_identity_or_permission_changed')
    tools=request('GET','/v1/tools')['tools']
    schema=next(tool['input_schema'] for tool in tools if tool['name']=='hpc_verify')
    if schema.get('properties')!={}:
        raise ValueError('current_hpc_verify_schema_requires_review')
    result=request('POST','/v1/sessions/'+SESSION+'/tools',{'tool_name':'hpc_verify',
        'arguments':{},'idempotency_key':'syscomplete-v3-readonly-identity-schema-corrected' if args.continue_schema_rejected else 'syscomplete-v3-readonly-identity-once'})
    payload={'schema':'evomind.hpc_readonly_result.v1',**plan,'finished_at':datetime.now(timezone.utc).isoformat(),
             'status':result.get('status'),'result':result.get('result'),'replayed':result.get('replayed',False)}
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=True,indent=2)
    evidence=(result.get('result') or {}).get('content') or {}
    print(json.dumps({'status':result.get('status'),'ok':(result.get('result') or {}).get('ok'),
        'session_id':SESSION,'identity_status':evidence.get('status'),'samples_passed':evidence.get('samples_passed'),
        'read_only':evidence.get('read_only'),'training_started':False}))
    return 0 if (result.get('result') or {}).get('ok') is True else 2


if __name__=='__main__':
    raise SystemExit(main())
