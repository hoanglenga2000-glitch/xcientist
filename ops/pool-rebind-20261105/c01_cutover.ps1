# Cut the training pool over from expired job 98270 (A800) to job 98456 (A40). Admin. Backup first; any failure restores all.
# Changes: gpu-pool.json, tenant hpc-binding.json, node-config.hpc.{job_id,credential_profile}, gpu-vram.json broker off
# (the vLLM/TTS services lived on 98270), python package (gpu_scheduler/tenant_training follow the pool file).
$ErrorActionPreference='Stop'
. 'C:\ProgramData\EvoMind\bundle\scripts\lib\Runtime.ps1'
Assert-Administrator
$stg='C:\ProgramData\EvoMind\staging\pool-98456-20261105'
$T='C:\ProgramData\EvoMind\byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0'
$pool='C:\ProgramData\EvoMind\state\gpu-pool.json'
$bind="$T\hpc-binding.json"
$vram='C:\ProgramData\EvoMind\data\workspace\runtime\config\gpu-vram.json'
$node=Get-Content "$T\pool-nodes\job98456.json" -Raw | ConvertFrom-Json
if($node.state -ne 'active' -or $node.job_id -ne 98456 -or $node.credential_profile -ne 'tenant_ba0ef9d3767f2fb385b856e0_job98456_g47'){ throw 'POOL_NODE_98456_INVALID' }
$meta=Get-Content 'C:\Users\EvoMindSvc\AppData\Roaming\ResearchAgentWorkstation\profiles\tenant_ba0ef9d3767f2fb385b856e0_job98456_g47\hpc_ssh_metadata.json' -Raw | ConvertFrom-Json
if($meta.profile_state -ne 'active' -or -not $meta.expected_gpu_inventory){ throw 'PROFILE_98456_NOT_ACTIVE_WITH_INVENTORY' }
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'); $b="C:\ProgramData\EvoMind\backups\pre-pool-98456-$stamp"
New-Item -ItemType Directory -Path $b | Out-Null
Copy-Item $pool "$b\gpu-pool.json"; Copy-Item $bind "$b\hpc-binding.json"; Copy-Item $vram "$b\gpu-vram.json"; Copy-Item $script:ConfigPath "$b\node-config.json"
Set-Content "$stg\backup-dir.txt" $b
"BACKUP $b"
$enc=New-Object Text.UTF8Encoding($false)
try {
  [IO.File]::WriteAllText("$pool.tmp", '{"gpu_pool":{"node_id":"job98456","hpc_job_id":98456,"service_tenant_id":"tenant_ba0ef9d3767f2fb385b856e0","tenant_training":true,"gpu_model":"A40","gpu_vram_mb":49140}}', $enc); Move-Item "$pool.tmp" $pool -Force
  $nb=[ordered]@{ schema='evomind.tenant_hpc_binding.v1'; tenant_id='tenant_ba0ef9d3767f2fb385b856e0'; username='admin'; state='active'; reason='current_five_of_five_identity_activation_verified'
    job_id=98456; credential_profile=[string]$node.credential_profile; allocation_binding_id=[string]$node.allocation_binding_id; allocation_generation=47
    profile_instance_id=[string]$node.profile_instance_id; expected_host_uuid=[string]$node.expected_host_uuid; expected_gpu_uuid=[string]$node.expected_gpu_uuid
    container_binding_sha256=[string]$node.container_binding_sha256; remote_root='/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra'; updated_at_utc=(Get-Date).ToUniversalTime().ToString('o') }
  [IO.File]::WriteAllText("$bind.tmp", (($nb | ConvertTo-Json -Depth 4)+"`n"), $enc); Move-Item "$bind.tmp" $bind -Force
  # The memory broker manages vLLM/TTS on the old card only: off for the A40 (nothing to free there).
  [IO.File]::WriteAllText("$vram.tmp", '{"gpu_broker_enabled": false}'+"`n", $enc); Move-Item "$vram.tmp" $vram -Force
  $cfg=Read-NodeConfig
  $cfg.hpc.job_id=98456; $cfg.hpc.credential_profile='tenant_ba0ef9d3767f2fb385b856e0_job98456_g47'
  Write-JsonAtomic -Path $script:ConfigPath -Value $cfg -Depth 12
  "CONFIG_OK pool=job98456 binding=g47 broker=off node-config.hpc.job_id=$((Read-NodeConfig).hpc.job_id)"
} catch {
  "CUTOVER_FAILED: $($_.Exception.Message)"
  Copy-Item "$b\gpu-pool.json" $pool -Force; Copy-Item "$b\hpc-binding.json" $bind -Force; Copy-Item "$b\gpu-vram.json" $vram -Force
  Copy-Item "$b\node-config.json" "$script:ConfigPath.restore.tmp" -Force; Move-Item "$script:ConfigPath.restore.tmp" $script:ConfigPath -Force
  "restored"; exit 1
}
