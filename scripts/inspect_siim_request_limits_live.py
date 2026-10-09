"""Read only, secret-free application request-limit and dispatch inspection."""
from prepare_siim_web_bridge import connect
from check_siim_training_watch import powershell
import base64

client = connect()
try:
    command = r"""
$base='C:/ProgramData/EvoMind'
$probe=@'
import sqlite3,json
c=sqlite3.connect('file:C:/ProgramData/EvoMind/data/workspace/runtime/runtime.sqlite3?mode=ro',uri=True)
r=c.execute('SELECT status,metadata_json FROM sessions WHERE id=?',('run_cc7970d443b54b6189a9aaec6dea9fd9',)).fetchone()
m=json.loads(r[1])
events=[json.loads(x[0]) for x in c.execute("SELECT payload_json FROM events WHERE session_id=? AND event_type='model.transport_attempt' ORDER BY seq DESC LIMIT 2",('run_cc7970d443b54b6189a9aaec6dea9fd9',))]
print(json.dumps({'status':r[0],'user_pause_requested':m.get('user_pause_requested'),'events':[{k:e.get(k) for k in ['status','error_code','request_limits','input_tokens','output_tokens']} for e in events]}))
'@
$child=& 'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv/Scripts/python.exe' -c $probe | ConvertFrom-Json
$backup=$base+'/backups/siim-web-request-limits-20260910'
$activation=if(Test-Path ($backup+'/result.json')){Get-Content -Raw ($backup+'/result.json') | ConvertFrom-Json}else{$null}
$failure=if(Test-Path ($backup+'/failure.json')){Get-Content -Raw ($backup+'/failure.json') | ConvertFrom-Json}else{$null}
$services=@(Get-Service '*EvoMind*' | Select-Object Name,Status)
$services+=@($child)
$services+=@(Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'apply_siim_request_limits|Start-Node.ps1'} | Select-Object ProcessId,Name,CreationDate)
$transport=Get-Content -LiteralPath ($base+'/bundle/runtime/evomind_runtime/model_transport.py')
$matches=@($transport | Select-String -Pattern 'confirmed|timeout_setting|max_tokens=|max_tokens =')
$dispatch=Get-ChildItem -LiteralPath ($base+'/staging/siim-mlebench-calibration-20260908/runtime-extension/service-output/web-dispatches') -Filter '*request-limits-recovery-20260910b.json'
$receipts=@($dispatch | ForEach-Object {Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json} | Select-Object parent_run_id,status,operation_id,approval_id,coordinator_exit_code,result)
[ordered]@{activation=$activation;failure=$failure;services=$services;request_limits_module_present=(Test-Path ($base+'/bundle/runtime/evomind_runtime/siim_request_limits.py'));recovery_module_present=(Test-Path ($base+'/bundle/runtime/evomind_runtime/siim_confirmed_recovery.py')); recovery_config_present=(Test-Path ($base+'/config/official-calibration/siim-confirmed-recovery-20260909.json')); transport_lines=@($matches | ForEach-Object {$_.Line}); dispatches=$receipts} | ConvertTo-Json -Depth 6 -Compress
"""
    start = command.index("$probe=@'\n")
    end = command.index("\n'@", start)
    encoded = base64.b64encode(command[start+len("$probe=@'\n"):end].encode()).decode()
    command = command[:start] + "$probe=\"exec(__import__('base64').b64decode('" + encoded + "'))\"" + command[end+3:]
    print(powershell(client, command))
finally:
    client.close()
