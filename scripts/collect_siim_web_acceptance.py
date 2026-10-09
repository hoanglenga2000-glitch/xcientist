"""Collect bounded web-dispatch evidence from the application server only."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from prepare_siim_web_bridge import connect, OUT

RUN = 'run_3eb4898a120847c19784eb9cdc9dd4a2'
BASE = 'C:/ProgramData/EvoMind'
STAGE = BASE+'/staging/siim-mlebench-calibration-20260908/runtime-extension/service-output'


def main():
    destination = OUT/('evidence-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    destination.mkdir()
    client = connect()
    try:
        command = "& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -X utf8 '"+BASE+"/staging/siim-mlebench-calibration-20260908/web-entry-repair/inspect_siim_web_run.py' --run "+RUN+'; exit $LASTEXITCODE'
        _, stdout, stderr = client.exec_command('powershell.exe -NoProfile -EncodedCommand '+base64.b64encode(command.encode('utf-16le')).decode(), timeout=30)
        raw = stdout.read(128*1024)
        if stdout.channel.recv_exit_status():
            raise ValueError('application_inspection_failed')
        receipt = json.loads(raw)
        if receipt.get('parent', {}).get('id') != RUN:
            raise ValueError('wrong_run_evidence')
        (destination/'application-run.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        sftp = client.open_sftp()
        sources = {'activation.json': BASE+'/backups/siim-web-entry-20260909/result.json',
                   'web-dispatch.json': STAGE+'/web-dispatches/'+RUN+'.json'}
        observations = [item for item in sftp.listdir_attr(STAGE) if item.filename.startswith('observation-') and item.filename != 'observation-error.json' and item.filename.endswith('.json')]
        if observations:
            sources['hpc-observation.json'] = STAGE+'/'+max(observations, key=lambda x:x.st_mtime).filename
        for name, source in sources.items():
            with sftp.open(source, 'rb') as stream:
                raw = stream.read(128*1024)
            value = json.loads(raw)
            (destination/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        client.close()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}
    (destination/'manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(json.dumps({'evidence_directory': str(destination), 'parent_status': receipt['parent']['status'], 'files': list(hashes)}))


if __name__ == '__main__':
    main()
