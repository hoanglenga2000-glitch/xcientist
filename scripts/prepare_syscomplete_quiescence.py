"""Derive a bounded quiescence receipt from persistent synchronous exit evidence.

Reads node metadata and selected non-label receipt files only. No model, SSH,
GPU, process signal, competition execution, or historical-record mutation.
This is not a live inventory of arbitrary processes on the HPC machine.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

if __package__:
    from . import syscomplete_activation_transaction as tx
else:
    import syscomplete_activation_transaction as tx

SAFE_RECEIPTS={'download-progress.json','execution-result.json','capacity-probe.json','verification.json',
    'replay-manifest.json','independent-reload-receipt.json','model-manifest.json','environment.lock.json'}
LOCAL={'artifact_import','file_write','artifact_bundle','artifact_preview'}
EXTERNAL={'hpc_execute_solution','managed_model_prepare'}
IDENTITY_FIELDS={'designated_proxy_path_verified','pinned_gateway_host_key_verified','allocation_role_authenticated',
    'expected_host_uuid_match','expected_gpu_uuid_match','expected_gpu_model_and_memory_match','allowed_remote_root_match'}


def derive(root, output, hpc_reference):
    root=tx.clean_path(root)
    output=tx.clean_path(output,root/'staging')
    tx.require(not output.exists(),'quiescence_receipt_already_exists')
    proof,_=tx.checked_ref(hpc_reference)
    evidence=(proof.get('result') or {}).get('content') or {}
    tx.require((proof.get('result') or {}).get('ok') is True and proof.get('status')=='completed'
        and proof.get('training_started') is False and evidence.get('schema')=='evomind.hpc.identity_receipt.v2'
        and evidence.get('samples_passed')==5 and evidence.get('read_only') is True and evidence.get('signals_sent')==0
        and evidence.get('other_processes_modified') is False and all(evidence.get(key) is True for key in IDENTITY_FIELDS),
        'hpc_identity_receipt_incomplete')
    tx.require(len(evidence.get('samples',[]))==5 and all(sample.get('complete') is True
        and all(sample.get(key) is True for key in IDENTITY_FIELDS) for sample in evidence['samples']), 'hpc_identity_samples_incomplete')
    before=tx.execution_snapshot(root)
    with tx.read_only(root/'data/workspace/runtime/runtime.sqlite3') as connection:
        session=dict(connection.execute('SELECT * FROM sessions WHERE id=?',(before['session_id'],)).fetchone())
        calls=[dict(row) for row in connection.execute('SELECT * FROM tool_calls WHERE session_id=? ORDER BY id',(before['session_id'],))]
    identity=json.loads(session['metadata_json'])['managed_hpc_identity']
    tx.require(evidence['job_id']==int(identity['job_id']) and evidence['credential_profile']==identity['credential_profile'],
               'quiescence_allocation_binding_mismatch')
    workspace=tx.clean_path(session['workspace_root'],root/'data')
    sealed=json.loads((root/'state/bundle-integrity.json').read_text(encoding='utf-8-sig'))
    executor={}
    for name in ('hpc_runtime_overlay.py','tools.py'):
        relative='runtime/evomind_runtime/'+name
        entry=next(row for row in sealed['files'] if row['path']==relative)
        actual=root/'bundle'/relative
        tx.require(tx.sha(actual)==entry['sha256'],'producer_executor_source_changed')
        executor[name]=entry['sha256']
    records=[]
    pending=[]
    for call in calls:
        name=call['tool_name']
        tx.require(call['status'] in {'completed','failed'},'quiescence_unsettled_tool')
        if name in LOCAL:
            records.append({'call_id':call['id'],'state':'not_external'})
            continue
        tx.require(name in EXTERNAL,'quiescence_unknown_tool')
        result=json.loads(call['result_json'])
        content=result.get('content') or {}
        tx.require(type(content.get('exit_code')) is int and 0<=content['exit_code']<=255
            and 'remote_settlement_unconfirmed' not in str(content.get('error',''))
            and content.get('status') in {'completed','failed'} and content.get('run_id')==tx.PAUSE_RUN,
            'remote_execution_terminal_receipt_missing')
        solution=content.get('solution_id','')
        remote=content.get('remote_dir','')
        tx.require(re.fullmatch('[A-Za-z0-9_-]{1,120}',solution) is not None
            and remote.startswith(evidence['remote_root'].rstrip('/')+'/') and tx.PAUSE_RUN in remote
            and remote.endswith('/solutions/'+solution), 'remote_worker_identity_unconfirmed')
        attempts=set()
        receipts=[]
        for item in content.get('local_artifacts',[]):
            path=Path(item['path'])
            try:
                relative=path.relative_to(workspace).as_posix()
            except ValueError:
                raise tx.Hold('worker_artifact_outside_run') from None
            match=re.search('/attempts/([a-f0-9]{32})/output/',relative)
            tx.require(match is not None,'worker_attempt_identity_missing')
            attempts.add(match.group(1))
            if path.name not in SAFE_RECEIPTS:
                continue  # In particular, do not read label arrays or datasets.
            tx.clean_path(path,workspace)
            tx.require(0<path.stat().st_size<=2*1024*1024 and path.stat().st_size==item['bytes']
                and tx.sha(path)==item['sha256'],'worker_receipt_integrity_mismatch')
            receipts.append({'path':relative,'sha256':item['sha256'],'bytes':item['bytes']})
        tx.require(len(attempts)==1 and bool(receipts),'worker_attempt_or_receipt_unconfirmed')
        row={'schema':'evomind.synchronous_hpc_exit_receipt.v1','run_id':tx.PAUSE_RUN,'call_id':call['id'],
             'tool_name':name,'solution_id':solution,'attempt_id':next(iter(attempts)),
             'remote_solution':remote,'exit_code':content['exit_code'],'tool_status':call['status'],
             'completed_at':call['completed_at'],'executor_sha256':executor,'receipt_files':receipts,
             'hpc_identity_receipt_sha256':hpc_reference['sha256'],'identity_verified':True,
             'scope':'persistent_synchronous_managed_executor_exit_not_general_live_process_inventory'}
        relative='calls/'+call['id']+'.json'
        data=(tx.canonical(row)+'\n').encode()
        records.append({'call_id':call['id'],'state':'terminal_verified','identity_verified':True,
                        'evidence':{'path':str(output.parent/relative),'sha256':tx.digest_bytes(data)}})
        pending.append((output.parent/relative,data))
    tx.require(tx.execution_snapshot(root)==before,'execution_changed_during_quiescence_audit')
    receipt={'schema':'evomind.syscomplete_quiescence_inventory.v1','run_id':tx.PAUSE_RUN,
        'observed_at':datetime.now(timezone.utc).isoformat(),'tool_receipts_sha256':before['tool_receipts_sha256'],
        'budget_sha256':before['budget_sha256'],'all_started_actions_accounted':True,
        'unknown_workers':[],'active_external_workers':[],'calls':records,
        'scope':'managed_invocations_only_persistent_exit_status_and_hash_bound_attempt_receipts',
        'live_hpc_process_inventory_performed':False,'hpc_identity_receipt_sha256':hpc_reference['sha256'],
        'historical_data_modified':False,'private_labels_read':False,'training_started':False}
    for path,data in pending:
        tx.write_new(path,data)
    tx.write_new(output,(tx.canonical(receipt)+'\n').encode())
    return {'status':'prepared','calls':len(records),'external_exit_receipts':len(pending),'sha256':tx.sha(output),
            'scope':receipt['scope'],'live_hpc_process_inventory_performed':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hpc-proof',type=Path,required=True)
    parser.add_argument('--hpc-proof-sha256',required=True)
    args=parser.parse_args()
    print(tx.canonical(derive(args.root,args.output,{'path':str(args.hpc_proof),'sha256':args.hpc_proof_sha256})))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(tx.canonical({'status':'HOLD','error_code':str(error) if isinstance(error,tx.Hold) else type(error).__name__}))
        raise SystemExit(2)
