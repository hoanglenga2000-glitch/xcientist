"""Evidence-based Sol-primary / GPT-5.5-reserve selection, not live failover.

No service/config mutation. Existing pinned Runs keep their model; an already
submitted training worker is never cancelled or resubmitted by this policy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PRIMARY='gpt-5.6-sol'
RESERVE='gpt-5.5'


def qualifies(report,model,candidate):
    if not isinstance(report,dict): return False
    duration=report.get('elapsed_seconds')
    cases=report.get('cases',[])
    if not (report.get('model')==model and report.get('schema')=='evomind.model_service_endurance.v1'
        and report.get('status')=='completed' and report.get('service_soak_passed') is True
        and report.get('runtime_closed_cleanly') is True and report.get('preflight_only') is False
        and type(duration) in {int,float} and math.isfinite(duration) and duration>=5400
        and report.get('required_seconds')==5400 and report.get('injected_faults')==0
        and report.get('gpu_actions')==0 and report.get('candidate_manifest_sha256')==candidate
        and report.get('route')=='configured_upstream' and report.get('wire_protocol')=='responses'
        and report.get('reasoning_effort')=='low' and report.get('service_tier')=='omit'
        and isinstance(cases,list) and len(cases)>=20): return False
    if len({c.get('run_id') for c in cases})!=len(cases) or len({c.get('index') for c in cases})!=len(cases): return False
    total=0
    for case in cases:
        proof=case.get('tool_evidence',{})
        count=proof.get('successful_tool_count')
        if not (case.get('passed') is True and case.get('status')=='completed' and case.get('case_timeout') is False
                and proof.get('required_tool_steps_passed') is True and proof.get('hash_check_passed') is True
                and proof.get('readback_passed') is True and type(count) is int and count>0): return False
        if case.get('needs_report') and proof.get('report_job_evidence',{}).get('passed') is not True: return False
        total+=count
    return type(report.get('native_tools_executed')) is int and report['native_tools_executed']==total and total>=50


def select(primary,reserve,candidate):
    primary_ok=qualifies(primary,PRIMARY,candidate)
    reserve_ok=qualifies(reserve,RESERVE,candidate)
    chosen=PRIMARY if primary_ok else RESERVE if reserve_ok else None
    return {'schema':'evomind.model_release_selection.v1','preferred_model':PRIMARY,'reserve_model':RESERVE,
        'selected_candidate_model':chosen,'primary_endurance_passed':primary_ok,'reserve_endurance_passed':reserve_ok,
        'candidate_manifest_sha256':candidate,'selection_rule':'qualified_sol_first_otherwise_qualified_gpt55',
        'status':'candidate_selected' if chosen else 'HOLD_no_qualified_model',
        'production_changed':False,'release_verdict':'HOLD_until_system_and_chrome_gates',
        'failover_contract':{'silent_in_run_switch':False,'cancel_existing_worker':False,'resubmit_training':False,
            'reset_budget':False,'authentication_or_permission_bypass':False,
            'requires_recorded_safe_checkpoint':True,'automatic_live_failover_verified':False},
        'pending':['current_channel_preflight','execution_reconciliation','fault_and_permission_gates',
                   'managed_training_continuity','transactional_activation','real_chrome_e2e']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary-receipt',type=Path)
    parser.add_argument('--reserve-receipt',type=Path,required=True)
    parser.add_argument('--candidate-manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    primary=json.loads(args.primary_receipt.read_text()) if args.primary_receipt else None
    reserve=json.loads(args.reserve_receipt.read_text())
    result=select(primary,reserve,args.candidate_manifest_sha256)
    result['receipt_hashes']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
        [('primary',args.primary_receipt),('reserve',args.reserve_receipt)] if path}
    with args.output.open('x',encoding='utf-8') as output:
        json.dump(result,output,ensure_ascii=True,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
