"""One pinned SSH connection for upload, hash readback and an exactly scoped resume."""
import base64,hashlib,json,subprocess
from pathlib import Path
import paramiko

ROOT=Path(__file__).resolve().parents[1]
REMOTE='C:/ProgramData/EvoMind/staging/ev-public-calibration-20260908/runtime-extension'


def main():
    settings={}
    text=subprocess.check_output(['ssh','-G','evomind-shanghai'],text=True,stderr=subprocess.DEVNULL)
    for line in text.splitlines():
        key,_,value=line.partition(' ')
        if key in {'hostname','port','user','identityfile','userknownhostsfile','proxycommand','proxyjump'}:settings.setdefault(key,value)
    if settings['hostname']!='101.43.51.15' or settings['user']!='Administrator' or int(settings['port'])!=22:
        raise ValueError('Saved host binding differs')
    if any(settings.get(k) not in {None,'none'} for k in ['proxycommand','proxyjump']):raise ValueError('Unexpected SSH routing')
    key=paramiko.Ed25519Key.from_private_key_file(settings['identityfile'])
    client=paramiko.SSHClient();client.load_host_keys(settings['userknownhostsfile']);client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(settings['hostname'],port=22,username=settings['user'],pkey=key,look_for_keys=False,allow_agent=False,timeout=15,banner_timeout=20,auth_timeout=20)
        client.get_transport().set_keepalive(10)
        sftp=client.open_sftp();hashes={}
        try:
            for name in ['ev_calibration_suite.py','start_ev_calibration_suite.ps1']:
                raw=(ROOT/'scripts'/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
                target=REMOTE+'/'+name;sftp.put(str(ROOT/'scripts'/name),target)
                with sftp.open(target,'rb') as f:readback=f.read()
                if hashlib.sha256(readback).hexdigest()!=hashes[name]:raise ValueError('Uploaded source differs')
        finally:sftp.close()
        command="$ProgressPreference='SilentlyContinue'; & '"+REMOTE+"/start_ev_calibration_suite.ps1' -Resume -ResumeKey envfix-v3"
        encoded=base64.b64encode(command.encode('utf-16le')).decode()
        _,stdout,stderr=client.exec_command('powershell -NoProfile -EncodedCommand '+encoded,timeout=30)
        output=stdout.read(65536);code=stdout.channel.recv_exit_status()
        if code:raise ValueError('Verified launcher rejected resume')
        print(json.dumps({'status':'uploaded_verified_and_dispatched','source_hashes':hashes,'response':output.decode('utf-8-sig',errors='replace').strip()}))
    finally:client.close()


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'failed','error_type':type(e).__name__}));raise SystemExit(2)
