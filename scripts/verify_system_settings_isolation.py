"""Real packaged Next/SQLite preference isolation; no runtime/model/HPC needed."""
import argparse
import importlib.util
import json
import secrets
import shutil
import socket
import sqlite3
import subprocess
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--web-sha256',required=True)
    p.add_argument('--fixture-sha256',required=True)
    p.add_argument('--node',default='C:/Program Files/nodejs/node.exe')
    args=p.parse_args()
    stage=args.stage.resolve()
    stage.relative_to(Path('C:/ProgramData/EvoMind/staging').resolve())
    spec=importlib.util.spec_from_file_location('settings_harness',stage/'verify_invitation_server_candidate.py')
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    flow=stage/'settings-isolation';flow.mkdir(exist_ok=False)
    v.extract(stage/'web.zip',args.web_sha256,flow/'web','operational-overlay-manifest.json')
    fixture=stage/'fixture-schema.sqlite'
    assert v.sha(fixture)==args.fixture_sha256
    (flow/'workspace').mkdir()
    shutil.copyfile(fixture,flow/'workspace/workstation.sqlite')
    db=sqlite3.connect(flow/'workspace/workstation.sqlite')
    db.execute('INSERT INTO settings (key,value_json,updated_at) VALUES (?,?,?)',('legacy_test_secret',json.dumps({'token':'synthetic-must-not-leak'}),'2026-09-08T00:00:00Z'))
    db.commit();db.close()
    build=json.loads((flow/'web/runtime-build-manifest.json').read_text())
    secret=secrets.token_urlsafe(48)
    env=v.fixture_environment(flow,secret,secrets.token_urlsafe(32),build['build_id'],build['source_tree_sha256'])
    with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env['PORT']=str(port)
    origin='http://127.0.0.1:'+str(port)
    checks=[];process=None
    def check(name,passed):
        checks.append({'name':name,'passed':bool(passed)})
        if not passed: raise RuntimeError(name)
    def start():
        process=subprocess.Popen([args.node,'server.js'],cwd=flow/'web',env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(200):
            if process.poll() is not None: raise RuntimeError('fixture_web_exited')
            try:
                if v.request(port,'GET','/login',timeout=1)[0]==200:return process
            except OSError:pass
            time.sleep(.1)
        process.terminate();process.wait(10);raise RuntimeError('fixture_start_timeout')
    def auth(username,tenant):
        cookie=v.COOKIE+'='+v.signed_principal(secret,username,tenant)
        h={'Cookie':cookie,'Origin':origin}
        status,_,session=v.request(port,'GET','/api/session/status',headers=h)
        check('session_'+username,status==200)
        h[v.CSRF]=session['csrf_token'];return h
    before=v.production_snapshot()
    with (flow/'web.out.log').open('wb') as out,(flow/'web.err.log').open('wb') as err:
      try:
        process=start()
        a=auth('alice','tenant_'+'a'*24);b=auth('bob','tenant_'+'a'*24);c=auth('carol','tenant_'+'c'*24)
        check('unauthenticated_denied',v.request(port,'GET','/api/settings')[0]==401)
        check('csrf_denied',v.request(port,'PATCH','/api/settings',body={'settings':{'general':{'theme':'light'}}},headers={'Cookie':a['Cookie'],'Origin':origin})[0]==403)
        status,_,result=v.request(port,'GET','/api/settings',headers=a)
        check('global_settings_not_exposed',status==200 and set(result['settings'])=={'general','language'} and 'synthetic-must-not-leak' not in json.dumps(result))
        for payload in [{'compute':{'mode':'local'}},{'general':{'theme':'invalid'}},{'credentials':{'token':'test'}},[]]:
            check('invalid_payload_denied',v.request(port,'PATCH','/api/settings',body={'settings':payload},headers=a)[0]==400)
        status,_,saved=v.request(port,'PATCH','/api/settings',body={'settings':{'general':{'theme':'light'},'language':{'ui_language':'en-US'}}},headers=a)
        check('save_own_preferences',status==200 and saved['settings']['general']['theme']=='light')
        for label,h in [('same_tenant',b),('different_tenant',c)]:
            status,_,value=v.request(port,'GET','/api/settings',headers=h)
            check(label+'_isolated',status==200 and value['settings']['general']['theme']=='dark' and value['settings']['language']['ui_language']=='zh-CN')
        status,_,saved=v.request(port,'PATCH','/api/settings',body={'settings':{'language':{'report_language':'en-US'}}},headers=a)
        check('partial_merge',status==200 and saved['settings']['general']['theme']=='light' and saved['settings']['language']['ui_language']=='en-US')
        process.terminate();process.wait(10);process=start()
        status,_,saved=v.request(port,'GET','/api/settings',headers=a)
        check('restart_persistence',status==200 and saved['settings']['language']['report_language']=='en-US')
        db=sqlite3.connect(flow/'workspace/workstation.sqlite')
        check('legacy_rows_preserved',json.loads(db.execute("SELECT value_json FROM settings WHERE key='legacy_test_secret'").fetchone()[0])=={'token':'synthetic-must-not-leak'})
        db.close()
      except Exception as error:
        checks.append({'name':'unexpected_failure','passed':False,'error_class':type(error).__name__})
        raise
      finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(10)
        after=v.production_snapshot()
        checks.append({'name':'production_unchanged','passed':before==after})
        result={'status':'passed' if checks and all(c['passed'] for c in checks) else 'failed','checks':checks,'model_requests':0,'hpc_accessed':False,'production_unchanged':before==after,
                'web_sha256':args.web_sha256,'build_id':build['build_id']}
        v.write_json(flow/'result.json',result)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
