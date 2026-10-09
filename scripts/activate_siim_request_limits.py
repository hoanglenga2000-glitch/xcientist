"""Invoke the existing exact-hash deployment transaction on the app host."""
import sys
import json
from prepare_siim_web_bridge import connect, ROOT
from check_siim_training_watch import powershell
import hashlib

stage='pause-recovery' if '--pause-recovery' in sys.argv else 'web-request-limits'
installer='apply_siim_pause_recovery.py' if '--pause-recovery' in sys.argv else 'apply_siim_request_limits.py'
manifest=ROOT/'artifacts/siim-mlebench-calibration-20260908'/stage/'payload/manifest.json'
digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
client=connect()
try:
    command="& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -X utf8 'C:/ProgramData/EvoMind/staging/siim-mlebench-calibration-20260908/"+stage+'/'+installer+"' --manifest-sha256 "+digest
    if '--apply' in sys.argv:command+=' --apply'
    command+='; exit $LASTEXITCODE'
    print(json.dumps(powershell(client,command)))
finally:
    client.close()
