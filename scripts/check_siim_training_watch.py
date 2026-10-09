"""One bounded supervision tick. Never trains, restarts, scores, or changes a budget.

Called by the Codex thread heartbeat. The application service's existing observer
performs fresh full HPC identity verification before inspecting the exact worker.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from prepare_siim_web_bridge import connect, ROOT

PARENT = 'run_3eb4898a120847c19784eb9cdc9dd4a2'
BASE = 'C:/ProgramData/EvoMind'
STAGE = BASE+'/staging/siim-mlebench-calibration-20260908/runtime-extension'
REMOTE_OUTPUT = STAGE+'/service-output'
OUT = ROOT/'artifacts/siim-mlebench-calibration-20260908/supervision'
PYTHON = 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe'
# Source-audited constant predictors. Keep their scores/history, but do not
# confuse a successful interface probe with a learned model. This is not an
# AUC threshold: a genuine trained model scoring 0.5 is still reported as such.
DIAGNOSTIC_ONLY_SOURCES = {
    '9a9299b26f0c8ae7e3aacb763825e263f0dd7b05315c0d94a25fe422dccf480d',
    '1761cc3d5fbd0e369b3b6b0bda305202139c4cd25a122bf8b1aed8732e6a9f32',
    '2b29e63a1a32083d684f819749780418d4ae18c7b70e458e7dadf49f9bf20b69',
}


def read_json(sftp, path, optional=False):
    try:
        with sftp.open(path, 'rb') as stream:
            raw = stream.read(2*1024*1024+1)
    except FileNotFoundError:
        if optional:
            return None
        raise
    if len(raw) > 2*1024*1024:
        raise ValueError('oversized_observation')
    return json.loads(raw)


def powershell(client, command):
    encoded = base64.b64encode(("$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';"+command).encode('utf-16le')).decode()
    _, stdout, _ = client.exec_command('powershell.exe -NoProfile -EncodedCommand '+encoded, timeout=45)
    raw = stdout.read(512*1024+1)
    if len(raw)>512*1024 or stdout.channel.recv_exit_status():
        raise ValueError('application_observation_failed')
    return json.loads(raw)


def verified_script(sftp, remote, local):
    with sftp.open(remote, 'rb') as stream:
        actual = hashlib.sha256(stream.read(256*1024)).hexdigest()
    if actual != hashlib.sha256(local.read_bytes()).hexdigest():
        raise ValueError('observer_script_changed')


def previous_state(path):
    # Receipts contain Chinese workspace paths and are always written as UTF-8.
    # Never let the Windows service/desktop locale select GBK on the next tick.
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def analyze(current, previous=None, now=None):
    now = time.time() if now is None else now
    previous = previous or {}
    app = current['application']
    parent = app['parent']
    budget = app['budget']
    cases = current.get('case_results') or []
    probe_cases = [row for row in cases if row.get('candidate_sha256') in DIAGNOSTIC_ONLY_SOURCES]
    accepted = [row for row in cases if row.get('candidate_sha256') not in DIAGNOSTIC_ONLY_SOURCES
                and row.get('status')=='completed' and row.get('oof_complete') is True
                and row.get('independent_saved_oof_recomputed') is True and row.get('reload_processes')==6]
    complete = (len(accepted)==9 and len({row.get('case_id') for row in accepted})==9
                and {(row.get('arm'),row.get('seed')) for row in accepted} ==
                {(arm,seed) for arm in ['fixed_baseline','evomind','aide'] for seed in [17,29,43]}
                and budget['pending_reserved_seconds']==0 and not budget['settlement_attention']
                and bool(current.get('selected_candidates')))
    observed = (current.get('hpc_observation') or {}).get('observed') or {}
    receipt = current.get('hpc_observation') or {}
    progress = observed.get('progress') or {}
    if observed.get('final_metrics_present') and any(row.get('name')=='final.joblib' for row in observed.get('checkpoints') or []):
        progress = {**progress,'phase':'independent_reload' if observed.get('post_fit_reload_passed',0)<6 else 'final_artifact_validation'}
    opt = observed.get('optimizer_progress') or {}
    timestamp = receipt.get('at')
    age = now-datetime.fromisoformat(timestamp).timestamp() if timestamp else None
    children = app.get('children') or []
    child_signature = [(row['run_id'],row['status'],row.get('model_transport_event_count'),
                        [(call['id'],call['status']) for call in row.get('calls') or []]) for row in children]
    parent_execution = parent.get('execution') or {}
    parent_signature = [parent['status'], parent_execution.get('model_transport_event_count'),
                        [(call['id'],call['status']) for call in parent_execution.get('calls') or []]]
    signature = [receipt.get('attempt_id'), progress.get('phase'),progress.get('fold'),opt.get('optimizer_steps'),
                 observed.get('completed_fit_events'),observed.get('checkpoints'),
                 observed.get('final_metrics_present'),observed.get('reload_checks'),child_signature,parent_signature,len(accepted)]
    signature_hash = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    changed = signature_hash != previous.get('signature_sha256')
    last_change = now if changed else previous.get('last_progress_at',now)
    fresh = age is not None and -60<=age<=150 and receipt.get('read_only') is True and receipt.get('hpc_identity_samples')==5
    old_observer = previous.get('observer_session')
    independent_sample = (fresh and receipt.get('observer_session') != old_observer) or budget['pending_reserved_seconds']==0
    unchanged = 0 if changed else previous.get('unchanged_samples',0)+int(independent_sample)
    stall = (fresh or budget['pending_reserved_seconds']==0) and not changed and unchanged>=2 and now-last_change>=600
    state = 'training_progress' if fresh and changed and progress.get('phase') in {'fold_fit','final_fit'} else 'progress_observed' if changed else 'unchanged'
    if fresh and changed and progress.get('phase') in {'independent_reload','final_artifact_validation'}:
        state = 'verification_progress'
    if budget['pending_reserved_seconds'] and not fresh:
        state = 'observation_unavailable'
    if stall:
        state = 'suspected_stall' if budget['pending_reserved_seconds'] else 'suspected_controller_stall'
    if parent['status'] in {'blocked','failed','cancelled','paused','pausing'}:
        state = 'requires_attention'
    if probe_cases:
        state = 'diagnostic_probe_misclassified_as_complete'
    if budget['remaining_seconds']<=0 and budget['pending_reserved_seconds']==0:
        state = 'budget_exhausted'
    if complete:
        state = 'training_records_complete_pending_final_audit'
    elif parent['status']=='completed':
        state = 'parent_finished_but_training_not_verified'
    old_opt = previous.get('optimizer_steps')
    same_attempt = previous.get('attempt_id') == receipt.get('attempt_id')
    delta = opt.get('optimizer_steps',0)-old_opt if same_attempt and isinstance(old_opt,int) and isinstance(opt.get('optimizer_steps'),int) else None
    return {'schema':'evomind.siim_supervision.v1','checked_at':datetime.fromtimestamp(now,timezone.utc).isoformat(),
            'state':state,'parent_run_id':PARENT,'parent_status':parent['status'],
            'case_id':receipt.get('case_id'),'attempt_id':receipt.get('attempt_id'),
            'phase':progress.get('phase'),'fold':progress.get('fold'),
            'optimizer_steps':opt.get('optimizer_steps'),'optimizer_delta':delta,
            'post_fit_reload_passed':observed.get('post_fit_reload_passed'),
            'gpu_summary':observed.get('gpu_summary'),'verified_complete_cases':len(accepted),'total_cases':9,
            'recorded_completed_cases':len(cases),
            'candidate_findings':[{'case_id':row['case_id'],'candidate_sha256':row['candidate_sha256'],
                                   'reason':'source_verified_constant_probe_not_learned_model'} for row in probe_cases],
            'observation_age_seconds':age,'observer_session':receipt.get('observer_session'),
            'signature_sha256':signature_hash,'last_progress_at':last_change,'unchanged_samples':unchanged,
            'budget':budget,'training_restarted':False,'official_score':None,
            'completion_requires_complete_oof_reload_and_predictions':True}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    previous=previous_state(OUT/'state.json')
    client=connect()
    try:
        sftp=client.open_sftp()
        inspector=BASE+'/staging/siim-mlebench-calibration-20260908/web-entry-repair/inspect_siim_web_run.py'
        verified_script(sftp,inspector,ROOT/'scripts/inspect_siim_web_run.py')
        command="& '"+PYTHON+"' -X utf8 '"+inspector+"' --run "+PARENT+'; exit $LASTEXITCODE'
        app=powershell(client,command)
        if app['parent']['id']!=PARENT:
            raise ValueError('parent_identity_mismatch')
        observation=None
        if app['budget']['pending_reserved_seconds']>0:
            launcher=STAGE+'/start_siim_calibration_observe.ps1'
            verified_script(sftp,launcher,ROOT/'scripts/start_siim_calibration_observe.ps1')
            verified_script(sftp,STAGE+'/siim_calibration_observe.py',ROOT/'scripts/siim_calibration_observe.py')
            start=time.time()
            dispatch=powershell(client,"$active=@(Get-ScheduledTask|Where-Object {$_.TaskName -like 'EvoMind-SIIM-Observe-*' -and $_.State -eq 'Running'});if($active.Count){[ordered]@{status='observer_already_running'}|ConvertTo-Json -Compress}else{& '"+launcher+"'}")
            deadline=time.monotonic()+40
            while time.monotonic()<deadline:
                observations=[entry for entry in sftp.listdir_attr(REMOTE_OUTPUT)
                              if entry.filename.startswith('observation-') and entry.filename.endswith('.json')
                              and entry.filename!='observation-error.json' and entry.st_mtime>=start-2]
                if observations:
                    candidate=read_json(sftp,REMOTE_OUTPUT+'/'+max(observations,key=lambda x:x.st_mtime).filename)
                    if datetime.fromisoformat(candidate['at']).timestamp()>=start-2:
                        observation=candidate
                        break
                if dispatch['status']=='observer_already_running':
                    break
                time.sleep(2)
            app=powershell(client,command)
        admitted=read_json(sftp,REMOTE_OUTPUT+'/case-results-admitted-v2.json',optional=True)
        legacy=read_json(sftp,REMOTE_OUTPUT+'/case-results.json',optional=True)
        current={'application':app,'hpc_observation':observation,'legacy_case_results':legacy,
                 'case_results':admitted if admitted is not None else legacy,
                 'selected_candidates':read_json(sftp,REMOTE_OUTPUT+'/selected-candidates.json',optional=True)}
    finally:
        client.close()
    status=analyze(current,previous)
    evidence=OUT/(stamp+'.json')
    with evidence.open('x',encoding='utf-8') as stream:
        json.dump(current,stream,ensure_ascii=False,indent=2)
    status['evidence_file']=str(evidence)
    status['evidence_sha256']=hashlib.sha256(evidence.read_bytes()).hexdigest()
    temp=OUT/'state.new.json'
    temp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(OUT/'state.json')
    print(json.dumps(status,ensure_ascii=True))


if __name__=='__main__':
    main()
