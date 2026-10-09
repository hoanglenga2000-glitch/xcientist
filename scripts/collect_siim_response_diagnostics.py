"""Preserve response-shape evidence without any response or reasoning text."""
from datetime import datetime,timezone
import hashlib,json
from prepare_siim_web_bridge import connect,ROOT
from check_siim_training_watch import powershell,PYTHON,STAGE,BASE,read_json,REMOTE_OUTPUT

if __name__=='__main__':
    out=ROOT/'artifacts/siim-mlebench-calibration-20260908/response-diagnostics'/('evidence-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    client=connect()
    try:
        result=powershell(client,"& '"+PYTHON+"' -X utf8 '"+STAGE+"/inspect_siim_case.py' --run run_cc7970d443b54b6189a9aaec6dea9fd9 --compact")
        evidence={'run_id':result['run_id'],'status':result['status'],'model_attempts':result['model_attempts'],'budget':result['budget']}
        (out/'model-response-metadata.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
        sftp=client.open_sftp()
        for name,path in {'activation.json':BASE+'/backups/siim-response-diagnostics-20260909/result.json',
            'operation.json':REMOTE_OUTPUT+'/web-dispatches/run_3eb4898a120847c19784eb9cdc9dd4a2--response-shape-20260909a.json'}.items():
            (out/name).write_text(json.dumps(read_json(sftp,path),ensure_ascii=False,indent=2),encoding='utf-8')
    finally:client.close()
    manifest={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in out.iterdir()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'evidence':str(out),'files':len(manifest)}))
