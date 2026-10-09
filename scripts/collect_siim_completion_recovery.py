"""Collect only the application-host completion correction and recovery receipts."""
from datetime import datetime,timezone
import hashlib,json
from prepare_siim_web_bridge import connect,ROOT
from check_siim_training_watch import read_json,BASE,REMOTE_OUTPUT

if __name__=='__main__':
    out=ROOT/'artifacts/siim-mlebench-calibration-20260908/completion-recovery'/('evidence-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    client=connect()
    try:
        sftp=client.open_sftp()
        sources={'activation.json':BASE+'/backups/siim-completion-recovery-20260909/result.json',
            'web-operation.json':REMOTE_OUTPUT+'/web-dispatches/run_3eb4898a120847c19784eb9cdc9dd4a2--completion-recovery-20260909a.json',
            'case-results-original.json':REMOTE_OUTPUT+'/case-results.json',
            'case-results-admitted-v2.json':REMOTE_OUTPUT+'/case-results-admitted-v2.json'}
        audits=[x for x in sftp.listdir_attr(REMOTE_OUTPUT) if x.filename.startswith('completion-admission-audit-') and x.filename.endswith('.json')]
        if audits:sources['admission-audit.json']=REMOTE_OUTPUT+'/'+max(audits,key=lambda x:x.st_mtime).filename
        for name,path in sources.items():
            value=read_json(sftp,path)
            (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:client.close()
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'evidence':str(out),'files':len(manifest)}))
