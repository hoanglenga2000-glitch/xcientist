"""Update only the read-only observer, preserving its pinned prior source."""
import hashlib
import json
from prepare_siim_web_bridge import connect,ROOT
from check_siim_training_watch import powershell,STAGE


def main():
    client=connect()
    target=STAGE+'/siim_calibration_observe.py'
    backup=STAGE+'/siim_calibration_observe.before-reload-v2.py'
    temporary=STAGE+'/siim_calibration_observe.reload-v2-new.py'
    before='28e40da9ee4074ef6094084a9f5ddfa2fd2d2bce79f9b01ea329c85571fd0e4c'
    source=ROOT/'scripts/siim_calibration_observe.py'
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        active=powershell(client,"[ordered]@{running=@(Get-ScheduledTask|Where-Object {$_.TaskName -like 'EvoMind-SIIM-Observe-*' -and $_.State -eq 'Running'}).Count}|ConvertTo-Json -Compress")
        if active['running']:
            raise ValueError('observer_is_running')
        sftp=client.open_sftp()
        with sftp.open(target,'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest()!=before:
                raise ValueError('observer_baseline_changed')
        for path in [backup,temporary]:
            try:
                sftp.stat(path)
            except FileNotFoundError:
                continue
            raise ValueError('previous_update_requires_reconciliation')
        sftp.put(str(source),temporary)
        with sftp.open(temporary,'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest()!=after:
                raise ValueError('observer_upload_changed')
        sftp.rename(target,backup)
        try:
            sftp.rename(temporary,target)
        except Exception:
            sftp.rename(backup,target)
            raise
        receipt={'status':'observer_projection_updated','before_sha256':before,'after_sha256':after,
                 'training_or_service_restarted':False,'hpc_mutations':0,'remote_backup':backup}
        (ROOT/'artifacts/siim-mlebench-calibration-20260908/supervision/observer-reload-v2/deployment.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(receipt))
    finally:
        client.close()


if __name__=='__main__':
    main()
