# Enroll job 98456 as an additional pool node profile for the tenant (does NOT switch the active binding; switch is a separate step).
# Runs as EvoMindSvc. HPC password comes from the LocalMachine DPAPI handoff blob and is deleted after use.
$ErrorActionPreference='Stop'
[Threading.Thread]::CurrentThread.CurrentUICulture = [Globalization.CultureInfo]::GetCultureInfo('en-US')
$out='C:\ProgramData\EvoMind\logs\job98456_pool_enroll_result.json'
$passFile='C:\ProgramData\EvoMind\state\.handoff-job98456.dpapi'
$profilesRoot='C:\Users\EvoMindSvc\AppData\Roaming\ResearchAgentWorkstation\profiles'
$tenantId='tenant_ba0ef9d3767f2fb385b856e0'
$srcProfile=Join-Path $profilesRoot "${tenantId}_job98270_g46"
$jobId=98456; $gen=47; $allocBinding='job-98456'; $roleUser='aimslab-cxl-gpu'
$profileInstanceId=[guid]::NewGuid().ToString('D')
$tenantProfile="${tenantId}_job${jobId}_g${gen}"
$tenantRoot="C:\ProgramData\EvoMind\byoa\tenants\$tenantId"
$result=[ordered]@{ schema='evomind.pool_node_enroll.v1'; at_utc=(Get-Date).ToUniversalTime().ToString('o'); job_id=$jobId; profile=$tenantProfile; steps=@() }
function Step($n,$ok,$d){ $script:result.steps += [ordered]@{name=$n; ok=[bool]$ok; detail=[string]$d}; Write-Output "STEP $n ok=$ok $d" }
$secureHpc=$null
try {
  if([Security.Principal.WindowsIdentity]::GetCurrent().Name -notmatch '\\EvoMindSvc$'){ throw 'EVOMINDSVC_REQUIRED' }
  if(Test-Path (Join-Path $profilesRoot $tenantProfile)){ throw "PROFILE_EXISTS $tenantProfile" }
  Add-Type -AssemblyName System.Security
  $blob=[IO.File]::ReadAllBytes($passFile); Remove-Item -LiteralPath $passFile -Force
  $plain=[Security.Cryptography.ProtectedData]::Unprotect($blob,[Text.Encoding]::UTF8.GetBytes('evomind-job98456-handoff'),'LocalMachine')
  $secureHpc=New-Object Security.SecureString
  foreach($ch in [Text.Encoding]::UTF8.GetChars($plain)){ $secureHpc.AppendChar($ch) }
  [Array]::Clear($plain,0,$plain.Length)
  Step 'load_hpc_pass' $true 'loaded'
  $socks=Import-Clixml -LiteralPath (Join-Path $srcProfile 'hpc_socks_credential.xml')
  if($socks -isnot [Management.Automation.PSCredential]){ throw 'SOCKS_CREDENTIAL_INVALID' }
  $env:EVOMIND_HPC_PROFILE_INSTANCE_ID=$profileInstanceId
  $inst = & 'C:\ProgramData\EvoMind\bundle\scripts\Install-TenantHpcProfile.ps1' `
    -TenantId $tenantId -OwnerPrincipalId 'admin' -JobId $jobId -AllocationGeneration $gen `
    -ProfileInstanceId $profileInstanceId -AllocationBindingId $allocBinding -HpcRoleAccount $roleUser `
    -ProxyUser ([string]$socks.UserName) -SecureProxyPassword $socks.Password -SecureHpcPassword $secureHpc `
    -TrustedKnownHostsPath 'C:\ProgramData\EvoMind\trust\aimslab-gateway-known_hosts'
  Step 'install_profile' $true (($inst | Out-String).Trim())
  $receipt=$null
  for($i=1; $i -le 3 -and -not $receipt; $i++){
    try {
      $node=Get-Content 'C:\ProgramData\EvoMind\config\node-config.json' -Raw|ConvertFrom-Json
      $python=Join-Path ([string]$node.release_root) '.venv\Scripts\python.exe'
      $env:EVOMIND_HPC_CREDENTIAL_PROFILE=$tenantProfile; $env:EVOMIND_SIIM_HPC_JOB_ID=[string]$jobId
      $env:PYTHONPATH="C:\ProgramData\EvoMind\bundle\runtime;$([string]$node.release_root)\src"
      $act = @(& $python 'C:\ProgramData\EvoMind\operations\activate_tenant_hpc_binding_inv.py' --tenant-id $tenantId --job-id $jobId --allocation-generation $gen --profile-instance-id $profileInstanceId --state-root 'C:\ProgramData\EvoMind\byoa') | Out-String
      Remove-Item Env:EVOMIND_HPC_CREDENTIAL_PROFILE,Env:EVOMIND_SIIM_HPC_JOB_ID,Env:PYTHONPATH -ErrorAction SilentlyContinue
      $receipt = $act.Trim() | ConvertFrom-Json
      if(-not $receipt.connection_ready){ throw ("ACTIVATION_NOT_READY " + $receipt.code) }
    } catch { Step "activate_try$i" $false $_.Exception.Message; Start-Sleep 10 }
  }
  if(-not $receipt -or -not $receipt.connection_ready){ throw 'ACTIVATION_FAILED' }
  Step 'activate' $true ("gpu=$($receipt.gpu_uuids -join ',') host=$($receipt.host_uuid) mem=$($receipt.gpu_memory_total_mib)")
  $pool=[ordered]@{
    schema='evomind.tenant_hpc_pool_node.v1'; tenant_id=$tenantId; job_id=$jobId; state='active'
    credential_profile=$tenantProfile; allocation_generation=$gen; profile_instance_id=$profileInstanceId
    allocation_binding_id=$allocBinding; expected_host_uuid=[string]$receipt.host_uuid
    expected_gpu_uuid=((@($receipt.gpu_uuids) | % { ([string]$_).ToLower() }) -join ',')
    gpu_name=[string]$receipt.gpu_name; gpu_memory_total_mib=[int]$receipt.gpu_memory_total_mib
    container_binding_sha256=[string]$receipt.container_binding_sha256; dedicated=$true
    remote_root='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'; updated_at_utc=(Get-Date).ToUniversalTime().ToString('o')
  }
  $poolDir=Join-Path $tenantRoot 'pool-nodes'; New-Item -ItemType Directory -Force $poolDir | Out-Null
  [IO.File]::WriteAllText((Join-Path $poolDir "job$jobId.json"), (($pool|ConvertTo-Json -Depth 6)+"`n"), [Text.UTF8Encoding]::new($false))
  Step 'pool_record' $true "pool-nodes\job$jobId.json"
  $result.ok=$true
} catch {
  $result.ok=$false; Step 'fatal' $false (($_ | Out-String).Trim())
} finally {
  if($secureHpc){ $secureHpc.Dispose() }
  Remove-Item Env:EVOMIND_HPC_PROFILE_INSTANCE_ID -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $passFile -Force -ErrorAction SilentlyContinue
  [IO.File]::WriteAllText($out, (($result|ConvertTo-Json -Depth 8)+"`n"), [Text.UTF8Encoding]::new($false))
}
