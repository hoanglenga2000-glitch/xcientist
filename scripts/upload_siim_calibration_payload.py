"""Upload only manifest-listed SIIM files through the pinned application-host route."""
import hashlib,json,subprocess
from pathlib import Path,PurePosixPath
import paramiko
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/siim-mlebench-calibration-20260908/runtime-payload-v2'
DEST='C:/ProgramData/EvoMind/staging/siim-mlebench-calibration-20260908/runtime-extension'

def main():
    settings={}
    for line in subprocess.check_output(['ssh','-G','evomind-shanghai'],text=True,stderr=subprocess.DEVNULL).splitlines():
        key,_,value=line.partition(' ')
        if key in {'hostname','port','user','identityfile','userknownhostsfile','proxycommand','proxyjump'}:settings.setdefault(key,value)
    if settings['hostname']!='101.43.51.15' or settings['user']!='Administrator' or int(settings['port'])!=22:raise ValueError('application_host_binding_changed')
    if any(settings.get(k) not in {None,'none'} for k in ['proxycommand','proxyjump']):raise ValueError('application_route_changed')
    client=paramiko.SSHClient();client.load_host_keys(settings['userknownhostsfile']);client.set_missing_host_key_policy(paramiko.RejectPolicy())
    key=paramiko.Ed25519Key.from_private_key_file(settings['identityfile'])
    client.connect(settings['hostname'],port=22,username=settings['user'],pkey=key,look_for_keys=False,allow_agent=False,timeout=20)
    client.get_transport().set_keepalive(15)
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    entries={rel:SOURCE/rel for rel in manifest['files']};entries['manifest.json']=SOURCE/'manifest.json'
    for name in ['apply_siim_calibration_runtime.py','siim_calibration_suite.py','start_siim_calibration_suite.ps1']:
        entries[name]=ROOT/'scripts'/name
    try:
        sftp=client.open_sftp();created=set()
        def directory(path):
            if path in created:return
            try:sftp.stat(path)
            except FileNotFoundError:
                parent=path.rsplit('/',1)[0]
                if parent.startswith('C:/ProgramData/EvoMind/staging/siim-mlebench-calibration-20260908'):directory(parent)
                sftp.mkdir(path)
            created.add(path)
        for rel,source in entries.items():
            if PurePosixPath(rel).is_absolute() or '..' in PurePosixPath(rel).parts:raise ValueError('unsafe_payload_path')
            raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest()
            if rel in manifest['files'] and h!=manifest['files'][rel]:raise ValueError('local_payload_changed')
            target=DEST+'/'+rel;directory(target.rsplit('/',1)[0]);sftp.put(str(source),target)
            with sftp.open(target,'rb') as stream:
                if hashlib.sha256(stream.read()).hexdigest()!=h:raise ValueError('uploaded_payload_changed')
        sftp.close();print(json.dumps({'status':'uploaded_and_hash_verified','files':len(entries),'hpc_files_transferred':0}))
    finally:client.close()

if __name__=='__main__':main()
