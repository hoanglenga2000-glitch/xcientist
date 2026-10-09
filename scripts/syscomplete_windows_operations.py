"""Windows service adapter for a separately verified cutover spec.

No actions occur on import. The CLI validates by default. --apply additionally
requires the exact plan hash and an explicit startup-side-effect acknowledgement.
Never use this module to rotate credentials, restore a DB, or re-seal arbitrary code.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

if __package__:
    from . import syscomplete_activation_transaction as tx
else:
    import syscomplete_activation_transaction as tx

SERVICE_HELPER=Path('C:/SecureInput/Invoke-ServiceAccountAction.ps1')


class WindowsOperations:
    def __init__(self, spec, *, run=subprocess.run):
        tx.require(os.name == 'nt' and Path(spec['root']) == tx.PRODUCTION_ROOT, 'production_windows_root_required')
        tx.require(spec.get('startup_side_effects_acknowledged') == {
            'managed_hpc_bridge': True, 'configured_hpc_reverification': True, 'competition_pause': True},
            'managed_startup_side_effects_not_acknowledged')
        self.spec, self.root, self.run = spec, Path(spec['root']), run
        self.plan = None
        self.stopping = []
        config = json.loads((self.root/tx.CONFIG_REL).read_text(encoding='utf-8-sig'))
        self.user = str(config['dedicated_user'])
        tx.require(re.fullmatch('[A-Za-z0-9_.-]{1,64}', self.user) is not None, 'service_owner_invalid')
        self.ports = [int(config['network'][key]) for key in ('runtime_port', 'web_port', 'llm_port')]
        tx.require(len(set(self.ports))==3 and all(1024<port<65536 for port in self.ports), 'service_ports_invalid')

    def mutex(self):
        return tx.deployment_mutex()

    def _command(self, arguments, *, timeout=240, environment=None):
        result = self.run(arguments, capture_output=True, timeout=timeout, env=environment,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        # Raw service output can contain protected values. Persist only its hash.
        if self.plan:
            path = Path(self.plan['backup']) / ('command-'+str(time.time_ns())+'.json')
            tx.write_new(path, (tx.canonical({'exit_code': result.returncode,
                'output_sha256': hashlib.sha256(result.stdout+result.stderr).hexdigest()})+'\n').encode())
        tx.require(result.returncode==0, 'managed_command_failed')
        return result.stdout

    def _acl(self, path, *, private=False):
        tx.clean_path(path,self.root)
        grants=['*S-1-5-18:(OI)(CI)F','*S-1-5-32-544:(OI)(CI)F']
        if not private:
            grants.append(self.user+':(OI)(CI)RX')
        self._command(['icacls.exe',str(path),'/inheritance:r','/grant:r',*grants,'/T','/Q'])

    def protect_backup(self, path):
        self._acl(path,private=True)

    def checkpoint(self, phase):
        print(tx.canonical({'phase':phase,'release_verdict':'HOLD'}),flush=True)

    def preflight(self, plan):
        self.plan=plan
        web=Path(plan['web_target'])
        runtime=Path(plan['backup'])/'candidate-runtime/evomind_runtime'
        dependency=Path(plan['gates']['dependencies']['target'])
        for folder in (web,runtime,dependency.parent):
            self._acl(folder)
        source=json.loads((web/'release-source-manifest.json').read_text(encoding='utf-8-sig'))
        reference=self.spec['operational_preflight_helper']
        helper_path=tx.clean_path(reference['path'])
        tx.require(helper_path.is_relative_to(Path(plan['stage'])) and helper_path.name=='invitation_release_transaction.py'
            and tx.sha(helper_path)==tx.hash_value(reference['sha256']), 'preflight_helper_changed')
        loader=importlib.util.spec_from_file_location('sealed_preflight_helper',helper_path)
        helper=importlib.util.module_from_spec(loader)
        loader.loader.exec_module(helper)
        helper.production_preflight(self.root,web,plan['build_id'],source['source_tree_sha256'],
            source['database_schema_sha256'],Path(plan['backup'])/'startup-preflight.json',
            schema_source_identity=source['production_schema_sha256'])
        manifest=next(row for row in plan['launcher_support_files'] if row['dependency_binding'])
        verifier=next(row for row in plan['launcher_support_files'] if not row['dependency_binding'])
        config=json.loads((self.root/tx.CONFIG_REL).read_text(encoding='utf-8-sig'))
        release=Path(config['release_root'])
        raw=self._command([str(release/'.venv/Scripts/python.exe'),'-I','-S','-B',verifier['source'],
            '--manifest',str(self.root/manifest['target']),'--manifest-sha256',manifest['after_sha256'],
            '--runtime-root',str(self.root/'bundle/runtime'),'--release-root',str(release)])
        proof=json.loads(raw)
        tx.require(proof.get('status')=='verified' and proof.get('binding_sha256')==manifest['after_sha256']
            and Path(proof['site_packages'])==dependency and proof.get('installed_file_count')==2187
            and proof.get('import_origin_count')==15, 'actual_dependency_launch_gate_failed')
        tx.write_new(Path(plan['backup'])/'actual-dependency-launch-proof.json',(tx.canonical(proof)+'\n').encode())

    def _service(self, action):
        reference=self.spec['service_action_helper']
        path=tx.clean_path(reference['path'])
        tx.require(action in {'Stop','Start'} and path==SERVICE_HELPER
            and tx.sha(path)==tx.hash_value(reference['sha256']), 'service_action_helper_changed')
        raw=self._command(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(path),
                           '-Action',action,'-TimeoutMinutes','12'],timeout=900)
        answer=json.loads(raw.decode('utf-8-sig'))
        tx.require(answer.get('status')=='completed' and answer.get('action')==action, 'managed_service_action_unconfirmed')

    def stop(self):
        import psutil
        state_path=self.root/'state/node-processes.json'
        if state_path.exists():
            state=json.loads(state_path.read_text(encoding='utf-8-sig'))
            for row in state.get('records',[]):
                try:
                    process=psutil.Process(int(row['pid']))
                    tx.require(process.username().split('\\')[-1].casefold()==self.user.casefold(), 'service_process_owner_mismatch')
                    for member in [process,*process.children(recursive=True)]:
                        self.stopping.append((member.pid,member.create_time()))
                except psutil.NoSuchProcess:
                    continue
        self._service('Stop')
        self.assert_stopped()
        return {'stopped':True,'identity_verified':True,'managed_processes_remaining':[]}

    def assert_stopped(self):
        import psutil
        for pid,created in self.stopping:
            try:
                tx.require(abs(psutil.Process(pid).create_time()-created)>0.001, 'managed_process_still_alive')
            except psutil.NoSuchProcess:
                pass
        listeners=[row for row in psutil.net_connections(kind='tcp') if row.status=='LISTEN'
                   and row.laddr and row.laddr.port in self.ports]
        tx.require(not listeners,'managed_listener_still_present')

    def verify_integrity(self, build):
        script=r"""
$ErrorActionPreference='Stop'
. 'C:/ProgramData/EvoMind/bundle/scripts/lib/Runtime.ps1'
$cfg=Read-NodeConfig
Assert-BundleIntegrity -Config $cfg | Out-Null
Assert-RuntimeArtifactIntegrity -Config $cfg | Out-Null
$web=Resolve-WebRuntimeIdentity -Config $cfg
if (-not $web.verified -or $web.overlay_id -cne $env:EVOMIND_EXPECTED_BUILD) { throw 'WEB_IDENTITY_MISMATCH' }
'INTEGRITY_PASS'
"""
        raw=self._command(['powershell.exe','-NoProfile','-Command',script],
                          environment={**os.environ,'EVOMIND_EXPECTED_BUILD':build})
        tx.require(raw.decode('utf-8-sig').strip()=='INTEGRITY_PASS','managed_integrity_unconfirmed')

    def start(self):
        self.assert_stopped()
        self._service('Start')

    def verify_ready(self, build):
        observations=[]
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            connection=http.client.HTTPConnection('127.0.0.1',self.ports[1],timeout=5)
            try:
                connection.request('GET','/api/healthz')
                response=connection.getresponse()
                value=json.loads(response.read(1024*1024))
                if response.status==200 and value.get('status')=='ready' and value.get('build_id')==build:
                    observations.append({'build_id':build,'status':'ready','observed_at':datetime.now(timezone.utc).isoformat()})
                    if len(observations)==3:
                        return {'status':'ready','build_id':build,'samples':observations}
                else:
                    observations=[]
            except (OSError,ValueError,http.client.HTTPException):
                observations=[]
            finally:
                connection.close()
            time.sleep(1)
        raise tx.Hold('candidate_health_not_confirmed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    spec,_=tx.checked_ref({'path':str(args.spec),'sha256':args.spec_sha256})
    saved,_=tx.checked_ref({'path':str(args.plan),'sha256':args.plan_sha256})
    plan=tx.validate(spec)
    tx.require(plan==saved,'activation_plan_or_live_state_changed')
    if not args.apply:
        print(tx.canonical({'status':'validated_not_activated','release_verdict':'HOLD'}))
        return 0
    result=tx.activate(spec,plan,WindowsOperations(spec))
    print(tx.canonical(result))
    return 0 if result['status']=='canary_validation_pending' else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(tx.canonical({'status':'HOLD','error_class':type(error).__name__,'raw_diagnostics_withheld':True}))
        raise SystemExit(2)
