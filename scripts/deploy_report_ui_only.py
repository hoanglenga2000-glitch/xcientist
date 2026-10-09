"""Publish an exact web overlay while retaining the recovered GPT-5.5 backend.

No backend/launcher/credential/budget mutation; no historical recovery. Normal
mode stages and validates. --apply additionally swaps one config field and
uses the existing managed stop/start helper. Rollback never restores a DB.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time

ROOT=Path('C:/ProgramData/EvoMind')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def require(value,code):
    if not value:raise ValueError(code)
def protected():
    paths=[ROOT/'bundle/scripts/Start-Node.ps1',ROOT/'bundle/runtime/run_python_runtime.py',ROOT/'state/bundle-integrity.json']
    paths.extend((ROOT/'bundle/runtime/evomind_runtime').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def write_new(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=True,indent=2)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    require(sha(args.manifest)==args.manifest_sha256,'manifest_mismatch')
    manifest=json.loads(args.manifest.read_text());stage=args.manifest.resolve().parent
    stage.relative_to(ROOT/'staging')
    for name,digest in manifest['files'].items():
        require('/' not in name and '\\' not in name and sha(stage/name)==digest,'staged_file_changed')
    if manifest.get('preference_isolation_receipt'):
        receipt_name=manifest['preference_isolation_receipt']
        require(receipt_name in manifest['files'],'isolation_receipt_unbound')
        accepted=json.loads((stage/receipt_name).read_text())
        require(accepted.get('status')=='passed' and accepted.get('production_unchanged') is True
            and accepted.get('web_sha256')==manifest['files']['web.zip'] and accepted.get('build_id')==manifest['build_id']
            and accepted.get('model_requests')==0 and accepted.get('hpc_accessed') is False
            and accepted.get('checks') and all(c.get('passed') is True for c in accepted['checks']), 'settings_isolation_failed')
    build=manifest['build_id'];require(re.fullmatch('overlay-invitation-beta-[a-f0-9]{12}-sys1',build),'build_id_invalid')
    target=ROOT/'web-overlays'/build
    require(target.resolve()==target,'target_alias')
    guard=load(stage/'guard.py','report_release_guard')
    helper=load(stage/'invitation_release_transaction.py','report_release_preflight')
    verifier=load(stage/'verify_invitation_server_candidate.py','report_release_archive')
    require(sha(ROOT/'bundle/scripts/Start-Node.ps1')=='4af54177257c6cb2336246cfc41863e6413e10e3e1c00afb1a82e63b99c24c6d','gpt55_launcher_changed')
    require(sha(ROOT/'bundle/runtime/run_python_runtime.py')=='55ef6200195f104c989fcb054dfb68f82c1a9c888be04bd80839c7cd2bd62a67','manual_recovery_entry_changed')
    cfg_path=ROOT/'config/node-config.json';old_bytes=cfg_path.read_bytes();cfg=json.loads(old_bytes.decode('utf-8-sig'))
    require(cfg['hpc']['state']!='active','global_hpc_startup_requires_review')
    require(Path(cfg['web_runtime_root'])!=target,'already_activated_read_receipt')
    before=protected();work=guard.current_work()
    renderer_patch=manifest.get('report_renderer_patch')
    renderer=ROOT/'bundle/runtime/evomind_runtime/report_render.py'
    seal_path=ROOT/'state/bundle-integrity.json'
    if renderer_patch:
        require(renderer_patch['file']=='report_render.py' and sha(renderer)==renderer_patch['before_sha256']
            and sha(stage/'report_render.py')==renderer_patch['after_sha256'],'report_renderer_binding_failed')
    jobs=ROOT/'data/workspace/runtime/report-jobs.sqlite3'
    if jobs.exists():
        with sqlite3.connect(jobs.as_uri()+'?mode=ro',uri=True) as c:
            require(c.execute("SELECT count(*) FROM jobs WHERE status NOT IN ('ready','partial','failed')").fetchone()[0]==0,'active_report_job_prevents_restart')
    if not target.exists():
        verifier.extract(stage/'web.zip',manifest['files']['web.zip'],target,'operational-overlay-manifest.json')
    package=json.loads((target/'operational-overlay-manifest.json').read_text())
    require(package['overlay_id']==build,'overlay_identity_mismatch')
    for row in package['files']:
        require(sha(target/row['path'])==row['sha256'],'extracted_web_changed')
    # Explicit permissions on existing files first, then inheritance for future files.
    account=str(cfg['dedicated_user'])
    subprocess.run(['takeown.exe','/F',str(target),'/A','/R','/D','Y'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(['icacls.exe',str(target),'/inheritance:r','/grant:r',account+':RX','*S-1-5-18:F','*S-1-5-32-544:F','/T','/C','/Q'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run(['icacls.exe',str(target),'/grant:r',account+':(OI)(CI)RX','*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F','/Q'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    source=json.loads((target/'release-source-manifest.json').read_text())
    helper.production_preflight(ROOT,target,build,source['source_tree_sha256'],source['database_schema_sha256'],
        stage/('preflight-'+str(time.time_ns())+'.json'),schema_source_identity=source['production_schema_sha256'])
    if not args.apply:
        print(json.dumps({'status':'web_staged_preflight_passed','build_id':build,'production_changed':False}));return
    backup=ROOT/'backups'/('report-ui-20260908-'+build.split('-')[-2]);require(not backup.exists(),'prior_attempt_requires_reconciliation')
    with guard.mutex():
        require(protected()==before and cfg_path.read_bytes()==old_bytes and guard.current_work()==work,'state_changed_before_stop')
        backup.mkdir(parents=True)
        subprocess.run(['icacls.exe',str(backup),'/inheritance:r','/grant:r','*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        (backup/'node-config.json').write_bytes(old_bytes)
        write_new(backup/'before.json',{'protected_files':before,'work':work,'old_web':cfg['web_runtime_root']})
        new_cfg={**cfg,'web_runtime_root':str(target)}
        new_bytes=(json.dumps(new_cfg,ensure_ascii=True,indent=2)+'\n').encode()
        switched=False
        renderer_changed=False
        expected_protected=dict(before)
        try:
            print(json.dumps({'phase':'stopping_for_web_update'}),flush=True)
            guard.service('Stop',backup);guard.stopped()
            require(guard.current_work()==work and protected()==before and cfg_path.read_bytes()==old_bytes,'state_changed_during_stop')
            db=ROOT/'data/workspace/runtime/runtime.sqlite3'
            with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c,sqlite3.connect(backup/'runtime.sqlite3') as copy:c.backup(copy)
            if manifest.get('preference_isolation_receipt'):
                settings_db=Path(cfg['data_root'])/'prisma/workstation.db'
                require(settings_db.resolve()==(ROOT/'data/prisma/workstation.db').resolve(), 'settings_database_path_changed')
                with sqlite3.connect(settings_db.as_uri()+'?mode=ro',uri=True) as c,sqlite3.connect(backup/'workstation.db') as copy:c.backup(copy)
            if renderer_patch:
                (backup/'report_render.py').write_bytes(renderer.read_bytes())
                (backup/'bundle-integrity.json').write_bytes(seal_path.read_bytes())
                renderer.write_bytes((stage/'report_render.py').read_bytes());renderer_changed=True
                seal=json.loads(seal_path.read_text(encoding='utf-8-sig'))
                row=next(row for row in seal['files'] if row['path']=='runtime/evomind_runtime/report_render.py')
                row.update(sha256=sha(renderer),size=renderer.stat().st_size)
                seal['sealed_at_utc']=datetime.now(timezone.utc).isoformat()
                seal_path.write_text(json.dumps(seal,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
                expected_protected[str(renderer.relative_to(ROOT))]=sha(renderer)
                expected_protected[str(seal_path.relative_to(ROOT))]=sha(seal_path)
            temp=cfg_path.with_name('node-config.report-ui.tmp');require(not temp.exists(),'temporary_config_exists')
            temp.write_bytes(new_bytes);os.replace(temp,cfg_path);switched=True
            guard.service('Start',backup)
            health=helper.wait_health(build)
            require(protected()==expected_protected and cfg_path.read_bytes()==new_bytes,'protected_state_changed')
            require(guard.current_work()==work,'unexpected_execution_after_restart')
            result={'status':'web_updated','build_id':build,'model':'gpt-5.5','model_and_launcher_unchanged':True,
                'report_renderer_updated':bool(renderer_patch),'backend_other_files_unchanged':True,
                'automatic_recovery':'still_manual','history_replayed':False,'database_restored':False,'backup':str(backup),'health':health}
            write_new(backup/'result.json',result);print(json.dumps(result),flush=True)
        except Exception as error:
            guard.service('Stop',backup);guard.stopped()
            if renderer_changed:
                require(sha(renderer)==renderer_patch['after_sha256'],'report_renderer_external_change')
                renderer.write_bytes((backup/'report_render.py').read_bytes())
                seal_path.write_bytes((backup/'bundle-integrity.json').read_bytes())
            if switched:
                require(cfg_path.read_bytes()==new_bytes,'external_config_changed_rollback_held')
                temp=cfg_path.with_name('node-config.report-ui-rollback.tmp');temp.write_bytes(old_bytes);os.replace(temp,cfg_path)
            guard.service('Start',backup)
            helper.wait_health(Path(cfg['web_runtime_root']).name)
            write_new(backup/'failure.json',{'status':'rolled_back','error_class':type(error).__name__,'database_restored':False})
            raise

if __name__=='__main__':main()
