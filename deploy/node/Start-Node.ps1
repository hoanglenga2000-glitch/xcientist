param([switch]$ForceRestart)
# Hotfix D6: make the start path idempotent about role listeners.
# A leftover role listener (most often the cliproxy gateway on its port) makes the new
# instance exit immediately, the start aborts, and the cleanup tears down the SOCKS bridge
# started earlier in the same run.  Clearing a listener is allowed only when it is
# provably this node's own role binary owned by the dedicated service account.
function Clear-StaleRoleListener {
  param(
    [Parameter(Mandatory)][string]$Role,
    [Parameter(Mandatory)][int]$Port,
    [Parameter(Mandatory)][string[]]$ExpectedProcessNames,
    [Parameter(Mandatory)][string]$ExpectedOwner
  )
  $rows = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
  if ($rows.Count -eq 0) { return }
  foreach ($row in $rows) {
    $ownerId = [int]$row.OwningProcess
    $process = Get-Process -Id $ownerId -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $name = [string]$process.ProcessName
    if ($ExpectedProcessNames -notcontains $name) {
      throw "ROLE_PORT_FOREIGN_OWNER: $Role port=$Port pid=$ownerId process=$name"
    }
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId" -ErrorAction SilentlyContinue
    $owner = if ($cim) { Invoke-CimMethod -InputObject $cim -MethodName GetOwner -ErrorAction SilentlyContinue } else { $null }
    $ownerName = if ($owner) { "$($owner.Domain)\$($owner.User)" } else { '' }
    if ($ownerName -ine $ExpectedOwner) {
      throw "ROLE_PORT_OWNER_NOT_ALLOWED: $Role port=$Port pid=$ownerId owner=$ownerName"
    }
    Stop-Process -Id $ownerId -Force -ErrorAction Stop
    $process.WaitForExit(15000) | Out-Null
    $deadline = (Get-Date).AddSeconds(8)
    while ((Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
      Start-Sleep -Milliseconds 250
    }
    if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
      throw "ROLE_PORT_CLEAR_TIMEOUT: $Role port=$Port pid=$ownerId"
    }
  }
}


. (Join-Path $PSScriptRoot 'lib\Runtime.ps1')
$config = Read-NodeConfig
Assert-DedicatedUserContext -Config $config
Assert-NoLegacyGpuEnvironment
Assert-AllowedWindowsPrivateIp -Address ([string]$config.network.windows_private_ip)
Assert-ExactRelease | Out-Null
Assert-BundleIntegrity -Config $config | Out-Null
Assert-RuntimeArtifactIntegrity -Config $config | Out-Null

$paths = Get-RuntimePaths
if (Test-Path -LiteralPath $paths.state) {
  if (-not $ForceRestart) {
    $statusScript = Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'scripts\Get-NodeStatus.ps1'
    $statusText = $null
    $statusExit = 1
    $priorErrorActionPreference = $ErrorActionPreference
    try {
      # A stale process-state file is expected after an ungraceful stop or a
      # whole-machine reboot. Let the probe return non-zero so the cleanup and
      # restart path below can run instead of promoting native stderr to a
      # terminating parent-script error.
      $ErrorActionPreference = 'Continue'
      $statusText = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $statusScript 2>$null
      $statusExit = $LASTEXITCODE
    } finally {
      $ErrorActionPreference = $priorErrorActionPreference
    }
    if ($statusExit -eq 0) { $statusText; exit 0 }
  }
  & (Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'scripts\Stop-Node.ps1')
}

$secretPaths = Get-NodeSecretPaths
$upstreamCredential = Import-DpapiCredentialChecked -Path $secretPaths.upstream -ExpectedUserName '__PEZAYO_UPSTREAM__'
$clientCredential = Import-DpapiCredentialChecked -Path $secretPaths.client -ExpectedUserName '__CLIPROXY_CLIENT__'
$sessionCredential = Import-DpapiCredentialChecked -Path $secretPaths.session -ExpectedUserName '__WORKSTATION_SESSION__'
$upstreamKey = $upstreamCredential.GetNetworkCredential().Password
$clientKey = $clientCredential.GetNetworkCredential().Password
$sessionSecret = $sessionCredential.GetNetworkCredential().Password
if ($upstreamKey.Length -lt 16 -or $clientKey.Length -lt 32 -or $sessionSecret.Length -lt 32) { throw 'NODE_SECRET_LENGTH_GATE_FAILED' }
$adminUsername = [string]$config.authentication.username
$adminPasswordScrypt = [string]$config.authentication.admin_password_scrypt
if ($adminUsername -notmatch '^[A-Za-z0-9._-]{1,64}$' -or $adminPasswordScrypt -notmatch '^scrypt\$16384\$8\$1\$[a-f0-9]{32}\$[a-f0-9]{64}$') {
  throw 'PUBLIC_LOGIN_CONFIGURATION_INVALID'
}

$gatewayConfig = New-CliproxyRuntimeConfig -UpstreamKey $upstreamKey -ClientKey $clientKey
$release = Resolve-ConfigPath ([string]$config.release_root)
$data = Resolve-ConfigPath ([string]$config.data_root)
$python = Join-Path $release '.venv\Scripts\python.exe'
$pythonw = Join-Path $release '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw)) { $pythonw = $python }
$node = (Get-Command node.exe -ErrorAction Stop).Source
$gatewayBinary = 'C:\ProgramData\EvoMind\bin\cliproxy-7.2.128.exe'
$webIdentity = Resolve-WebRuntimeIdentity -Config $config
$webStandalone = [string]$webIdentity.root
$expectedWebBuild = [string](Get-Content -LiteralPath ([string]$webIdentity.runtime_manifest_path) -Raw | ConvertFrom-Json).build_id
if (-not $expectedWebBuild) { throw 'WEB_EXPECTED_BUILD_MISSING' }
$runtimeScript = Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'runtime\run_python_runtime.py'
$byoaPythonRoot = Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'runtime'
foreach ($required in @($gatewayBinary,$python,(Join-Path $webStandalone 'server.js'),$runtimeScript)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "NODE_RUNTIME_FILE_MISSING: $required" }
}
$gatewayInstallEvidence = Get-Content -LiteralPath 'C:\ProgramData\EvoMind\state\cliproxy-install.json' -Raw | ConvertFrom-Json
if ($gatewayInstallEvidence.schema -ne 'evomind.cliproxy_install.v1' -or $gatewayInstallEvidence.version -ne '7.2.128' -or $gatewayInstallEvidence.binary_sha256 -ne (Get-Sha256File $gatewayBinary)) {
  throw 'CLIPROXY_RUNTIME_PROVENANCE_GATE_FAILED'
}

$records = @()
$hpcBridgeStarted = $false
try {
  Start-SafeHpcBridge -ReleaseRoot $release -BundleRoot (Resolve-ConfigPath ([string]$config.bundle_root)) -Port 7890 | Out-Null
  $hpcBridgeStarted = $true
  $queueWatcher = Start-ManagedProcess -Role 'tenant-enrollment-queue' -FilePath (Get-Command pwsh.exe -ErrorAction Stop).Source -ArgumentList @('-NoLogo','-NoProfile','-NonInteractive','-File',(Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'scripts\Watch-TenantEnrollmentQueue.ps1')) `
    -WorkingDirectory $data -Environment @{WORKSTATION_BYOA_STATE_ROOT='C:\ProgramData\EvoMind\byoa';PYTHONPATH=($byoaPythonRoot+';'+(Join-Path $release 'src'))} -LogsRoot $paths.logs
  $records += Get-ProcessRecord -Process $queueWatcher -Role 'tenant_enrollment_queue'
  if ($config.hpc.state -eq 'active') {
    $env:EVOMIND_SIIM_HPC_JOB_ID = [string]$config.hpc.job_id
    $env:EVOMIND_HPC_CREDENTIAL_PROFILE = [string]$config.hpc.credential_profile
    try {
      Assert-ServiceAllocationBinding -Config $config | Out-Null
      $startupHpcEvidence = Join-Path (Resolve-ConfigPath ([string]$config.state_root)) 'hpc-startup-live-reverify.json'
      & $python (Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'runtime\verify_hpc_cloud_node.py') --job-id ([int]$config.hpc.job_id) --samples 5 --output $startupHpcEvidence
      if ($LASTEXITCODE -ne 0) { throw 'ACTIVE_HPC_STARTUP_FIVE_SAMPLE_REVERIFY_FAILED' }
      $hpcProof = Get-Content -LiteralPath $startupHpcEvidence -Raw | ConvertFrom-Json
      if (-not $hpcProof.connection_ready -or -not $hpcProof.five_of_five_identity_samples) { throw 'ACTIVE_HPC_STARTUP_EVIDENCE_GATE_FAILED' }
    } catch {
      $hpcFailure = $_.Exception.Message
      try { Stop-SafeHpcBridge -Port 7890 | Out-Null } catch {}
      $hpcBridgeStarted = $false
      try { & $python (Join-Path (Resolve-ConfigPath ([string]$config.bundle_root)) 'runtime\manage_hpc_profile_lifecycle.py') freeze --profile ([string]$config.hpc.credential_profile) --reason startup_identity_reverify_failed | Out-Null } catch {}
      $config.hpc.state = 'blocked'
      $config.hpc.reason = $hpcFailure
      Write-JsonAtomic -Path $script:ConfigPath -Value $config
    } finally {
      Remove-Item Env:EVOMIND_SIIM_HPC_JOB_ID -ErrorAction SilentlyContinue
      Remove-Item Env:EVOMIND_HPC_CREDENTIAL_PROFILE -ErrorAction SilentlyContinue
    }
  }
  $baseEnv = Get-BaseRuntimeEnvironment -Config $config -ClientKey $clientKey -SessionSecret $sessionSecret
  $baseEnv['PYTHONPATH'] = $byoaPythonRoot + ';' + (Join-Path $release 'src')
  # BEGIN EVOMIND_GPT55_PROFILE_V1
  # EVOMIND_MODEL_PROFILE_V2 (2026-10-10): operator-selectable child-process model profile.
  # Default = TEMPORARY Claude route (claude-sonnet-5 via api.lt4net.org, Anthropic /v1/messages)
  # because the Pezayo deepseek-flash balance is exhausted. The Pezayo profile is kept intact.
  # Rollback without resealing: set node-config.json "model_profile": "pezayo_deepseek_flash", restart.
  # Never changes any User/Machine environment; credentials stay CurrentUser DPAPI files.
  $modelProfile = 'claude_lt4net_temp'
  if ($config.PSObject.Properties.Name -contains 'model_profile' -and [string]$config.model_profile) { $modelProfile = [string]$config.model_profile }
  if ($modelProfile -notin @('claude_lt4net_temp','claude_opus55_temp','pezayo_deepseek_flash','gpt6_sub2api')) { throw "MODEL_PROFILE_INVALID: $modelProfile" }
  $deepseekCredential = Import-DpapiCredentialChecked -Path (Join-Path $secretPaths.root 'pezayo_ev_deepseek_v4_pro.xml') -ExpectedUserName '__PEZAYO_DEEPSEEK_V4_PRO__'
  $baseEnv['OPENAI_API_KEY'] = $deepseekCredential.GetNetworkCredential().Password
  $baseEnv['OPENAI_BASE_URL'] = 'https://api.pezayo.com/v1'
  $baseEnv['OPENAI_MODEL'] = 'deepseek-flash'
  # Hotfix D4: DeepSeek connector names: the web connector projection and the DeepSeek-backed
  # Code Agent fallback read DEEPSEEK_*; publish the same already-decrypted
  # gateway credential under those names so they report the working path.
  $baseEnv['DEEPSEEK_API_KEY'] = $deepseekCredential.GetNetworkCredential().Password
  $baseEnv['DEEPSEEK_BASE_URL'] = 'https://api.pezayo.com/v1'
  $baseEnv['DEEPSEEK_MODEL'] = 'deepseek-flash'
  $baseEnv['OPENAI_REASONING_EFFORT'] = ''
  $baseEnv['OPENAI_SERVICE_TIER'] = ''
  $baseEnv['EVOLUTION_PRIMARY_PROVIDER'] = 'openai'
  $baseEnv['EVOLUTION_PROVIDER_STRICT'] = 'true'
  $baseEnv['LLM_PROVIDER'] = 'openai'
  $baseEnv['EVOMIND_MODEL_WIRE_PROTOCOL'] = 'chat_completions'
  $baseEnv['EVOMIND_MODEL_TIMEOUT_SECONDS'] = '180'
  $baseEnv['EVOMIND_MODEL_ROUTE_CONFIG_PATH'] = 'C:/ProgramData/EvoMind/state/ev-deepseek-route-20260908.json'
  $modelBypassTargets = @('api.pezayo.com')
  # RUN291 (2026-10-22, owner-approved, temporary): claude_opus55_temp = same lt4net credential,
  # model claude-opus-5-5, strict route (no fallback). Rollback: node-config model_profile -> claude_lt4net_temp, restart.
  $claudeModel = 'claude-sonnet-5'; $claudeRoutePath = 'C:\ProgramData\EvoMind\state\ev-claude-lt4net-route-20261010.json'
  if ($modelProfile -eq 'claude_opus55_temp') { $claudeModel = 'claude-opus-5-5'; $claudeRoutePath = 'C:\ProgramData\EvoMind\state\ev-claude-opus55-lt4net-route-temp.json' }
  if ($modelProfile -in @('claude_lt4net_temp','claude_opus55_temp')) {
    $claudeCredential = Import-DpapiCredentialChecked -Path (Join-Path $secretPaths.root 'newapi_lt4net_claude_temp.xml') -ExpectedUserName '__NEWAPI_LT4NET_CLAUDE_TEMP__'
    $baseEnv['ANTHROPIC_API_KEY'] = $claudeCredential.GetNetworkCredential().Password
    # Anthropic transport appends /v1/messages to this root.
    $baseEnv['ANTHROPIC_BASE_URL'] = 'https://api.lt4net.org'
    $baseEnv['CLAUDE_CODE_MODEL'] = $claudeModel
    $baseEnv['ANTHROPIC_MODEL'] = $claudeModel
    $baseEnv['EVOLUTION_PRIMARY_PROVIDER'] = 'anthropic'
    $baseEnv['LLM_PROVIDER'] = 'anthropic'
    $baseEnv['EVOMIND_MODEL_ROUTE_CONFIG_PATH'] = $claudeRoutePath.Replace('\','/')
    # Resilient failover (2026-10-11): honor route.fallback_to_other_models so AgentMessageClient
    # Layer-2 can try Pezayo deepseek-flash (already injected above) if Claude/lt4net fails.
    # Keep Claude primary via EVOLUTION_PRIMARY_PROVIDER=anthropic; do not fail closed on single-provider outage.
    $claudeRouteFallback = $false
    try {
      $claudeRouteObj = Get-Content -LiteralPath $claudeRoutePath -Raw -Encoding UTF8 | ConvertFrom-Json
      if ($claudeRouteObj.PSObject.Properties.Name -contains 'fallback_to_other_models' -and [bool]$claudeRouteObj.fallback_to_other_models) { $claudeRouteFallback = $true }
    } catch {}
    if ($claudeRouteFallback) {
      $baseEnv['EVOLUTION_PROVIDER_STRICT'] = 'false'
    } else {
      $baseEnv['EVOLUTION_PROVIDER_STRICT'] = 'true'
    }
    $modelBypassTargets += 'api.lt4net.org'
  }
  # GPT6_SUB2API (2026-11-02, owner-approved t361u): default model gpt-6-astra via Sub2API group 18 (admin-direct).
  # Credential = CurrentUser DPAPI newapi_gpt6_admin_direct.xml. Rollback: node-config model_profile -> claude_lt4net_temp, restart.
  if ($modelProfile -eq 'gpt6_sub2api') {
    $gpt6Credential = Import-DpapiCredentialChecked -Path (Join-Path $secretPaths.root 'newapi_gpt6_admin_direct.xml') -ExpectedUserName '__NEWAPI_GPT6_ADMIN_DIRECT__'
    $baseEnv['OPENAI_API_KEY'] = $gpt6Credential.GetNetworkCredential().Password
    $baseEnv['OPENAI_BASE_URL'] = 'https://api.zhjjq.tech/v1'
    $baseEnv['OPENAI_MODEL'] = 'gpt-6-astra'
    $baseEnv['EVOLUTION_PRIMARY_PROVIDER'] = 'openai'
    $baseEnv['LLM_PROVIDER'] = 'openai'
    $baseEnv['EVOMIND_MODEL_ROUTE_CONFIG_PATH'] = ''
    $modelBypassTargets += 'api.zhjjq.tech'
  }
  $baseEnv['EVOMIND_MODEL_PROFILE'] = $modelProfile
  foreach ($modelField in @('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL',
      'OPENAI_REASONING_EFFORT','OPENAI_SERVICE_TIER','EVOLUTION_PRIMARY_PROVIDER',
      'EVOLUTION_PROVIDER_STRICT','LLM_PROVIDER','ANTHROPIC_API_KEY','ANTHROPIC_BASE_URL','CLAUDE_CODE_MODEL')) {
    $baseEnv[$modelField + '_FILE'] = ''
  }
  # urllib honors NO_PROXY for the model origin. Keep all other proxy settings
  # and existing bypass entries intact, including the designated HPC route.
  foreach ($modelBypassTarget in $modelBypassTargets) {
    $modelNoProxy = if ($baseEnv.ContainsKey('NO_PROXY')) {
      [string]$baseEnv['NO_PROXY']
    } else {
      [Environment]::GetEnvironmentVariable('NO_PROXY', 'Process')
    }
    $modelBypassHosts = @($modelNoProxy -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($modelBypassHosts -notcontains '*' -and $modelBypassHosts -notcontains $modelBypassTarget) {
      $baseEnv['NO_PROXY'] = if ([string]::IsNullOrWhiteSpace($modelNoProxy)) {
        $modelBypassTarget
      } else {
        $modelNoProxy + ',' + $modelBypassTarget
      }
    }
  }
  # END EVOMIND_GPT55_PROFILE_V1
  # Hotfix D6: make the start path idempotent about role listeners
  $roleOwner = "$env:COMPUTERNAME\$($config.dedicated_user)"
  Clear-StaleRoleListener -Role 'llm-gateway' -Port 65068 -ExpectedProcessNames @('cliproxy-7.2.128','cliproxy') -ExpectedOwner $roleOwner
  Clear-StaleRoleListener -Role 'python-runtime' -Port ([int]$config.network.runtime_port) -ExpectedProcessNames @('pythonw','python') -ExpectedOwner $roleOwner
  Clear-StaleRoleListener -Role 'web' -Port ([int]$config.network.web_port) -ExpectedProcessNames @('node') -ExpectedOwner $roleOwner

  $gateway = Start-ManagedProcess -Role 'llm-gateway' -FilePath $gatewayBinary -ArgumentList @('-config', $gatewayConfig) `
    -WorkingDirectory (Split-Path -Parent $gatewayBinary) -Environment @{} -LogsRoot $paths.logs
  $records += Get-ProcessRecord -Process $gateway -Role 'llm_gateway'
  if (-not (Wait-TcpPort -Address '127.0.0.1' -Port ([int]$config.network.llm_port) -TimeoutSeconds 30)) { throw 'LLM_GATEWAY_LISTENER_TIMEOUT' }
  $headers = @{ Authorization = "Bearer $clientKey" }
  $models = Invoke-RestMethod -Uri 'http://127.0.0.1:65068/v1/models' -Headers $headers -TimeoutSec 20
  if (@($models.data | Where-Object id -eq 'gpt-5.6-sol').Count -ne 1) { throw 'LLM_MODEL_ROUTE_MISSING' }

  $runtime = Start-ManagedProcess -Role 'python-runtime' -FilePath $pythonw -ArgumentList @('-X','utf8',$runtimeScript,'--workspace',$data,'--port',[string]$config.network.runtime_port) `
    -WorkingDirectory $data -Environment $baseEnv -LogsRoot $paths.logs
  $records += Get-ProcessRecord -Process $runtime -Role 'python_runtime'
  if (-not (Wait-TcpPort -Address '127.0.0.1' -Port ([int]$config.network.runtime_port) -TimeoutSeconds 45)) { throw 'PYTHON_RUNTIME_LISTENER_TIMEOUT' }
  $runtimeTokenPath = Join-Path $data 'workspace\runtime\runtime.token'
  if (-not (Test-Path -LiteralPath $runtimeTokenPath -PathType Leaf)) { throw 'PYTHON_RUNTIME_TOKEN_MISSING' }
  $runtimeToken = (Get-Content -LiteralPath $runtimeTokenPath -Raw).Trim()
  $runtimeHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$($config.network.runtime_port)/v1/health" -Headers @{Authorization="Bearer $runtimeToken"} -TimeoutSec 10
  if ($runtimeHealth.status -ne 'ready') { throw 'PYTHON_RUNTIME_HEALTH_FAILED' }

  $webEnv = @{} + $baseEnv
  $webEnv['EVOMIND_PYTHON'] = $python
  $webEnv['HOSTNAME'] = '127.0.0.1'
  $webEnv['PORT'] = [string]$config.network.web_port
  $webEnv['npm_package_version'] = '0.3.0'
  $webEnv['WORKSTATION_ADMIN_USERNAME'] = $adminUsername
  $webEnv['WORKSTATION_ADMIN_PASSWORD_SCRYPT'] = $adminPasswordScrypt
  $webEnv['WORKSTATION_ACCOUNTS_DB'] = 'C:\ProgramData\EvoMind\accounts\accounts.db'
  $webEnv['WORKSTATION_BYOA_STATE_ROOT'] = 'C:\ProgramData\EvoMind\byoa'
  $clockAnchorPath = Join-Path (Resolve-ConfigPath ([string]$config.state_root)) 'service-clock-anchor.json'
  $serviceInstanceId = [guid]::NewGuid().ToString('D')
  $bootTime = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
  $bootBytes = [Text.Encoding]::UTF8.GetBytes($bootTime)
  $bootHasher = [Security.Cryptography.SHA256]::Create()
  try { $bootId = ([BitConverter]::ToString($bootHasher.ComputeHash($bootBytes))).Replace('-','').ToLowerInvariant() } finally { $bootHasher.Dispose() }
  $clockEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  $clockAnchor = [ordered]@{schema='evomind.service_clock_anchor.v1';service_instance_id=$serviceInstanceId;boot_id=$bootId;windows_utc_epoch_ms=$clockEpoch;qpc_timestamp=[Diagnostics.Stopwatch]::GetTimestamp();qpc_frequency=[Diagnostics.Stopwatch]::Frequency;created_at_utc=[DateTimeOffset]::FromUnixTimeMilliseconds($clockEpoch).ToString('o');web_monotonic_origin='system_qpc'}
  Write-JsonAtomic -Path $clockAnchorPath -Value $clockAnchor
  Set-OwnerOnlyAcl -Path $clockAnchorPath
  $webEnv['EVOMIND_SERVICE_CLOCK_ANCHOR_PATH'] = $clockAnchorPath
  $webEnv['EVOMIND_SERVICE_INSTANCE_ID'] = $serviceInstanceId
  $webEnv['EVOMIND_BOOT_ID'] = $bootId
  $webEnv['EVOMIND_CLOCK_QPC_TIMESTAMP'] = [string]$clockAnchor.qpc_timestamp
  $webEnv['EVOMIND_CLOCK_QPC_FREQUENCY'] = [string]$clockAnchor.qpc_frequency
  $webEnv['EVOMIND_CLOCK_EPOCH_MS'] = [string]$clockAnchor.windows_utc_epoch_ms
  $clockProbeCode = "const a=BigInt(process.env.EVOMIND_CLOCK_QPC_TIMESTAMP),f=BigInt(process.env.EVOMIND_CLOCK_QPC_FREQUENCY),e=Number(process.env.EVOMIND_CLOCK_EPOCH_MS),n=process.hrtime.bigint(),an=a*1000000000n/f;process.stdout.write(String(e+Number((n-an)/1000000n)))"
  $savedClockEnv = @{};foreach($key in @('EVOMIND_CLOCK_QPC_TIMESTAMP','EVOMIND_CLOCK_QPC_FREQUENCY','EVOMIND_CLOCK_EPOCH_MS')){$savedClockEnv[$key]=[Environment]::GetEnvironmentVariable($key,'Process');[Environment]::SetEnvironmentVariable($key,[string]$webEnv[$key],'Process')}
  try{$derivedClock=[long](& $node -e $clockProbeCode);if($LASTEXITCODE-ne0-or[Math]::Abs($derivedClock-[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())-ge1000){throw 'SERVICE_CLOCK_NODE_QPC_SELF_CHECK_FAILED'}}finally{foreach($key in $savedClockEnv.Keys){[Environment]::SetEnvironmentVariable($key,$savedClockEnv[$key],'Process')}}
  $web = Start-ManagedProcess -Role 'web' -FilePath $node -ArgumentList @('--title=evomind-windows-cloud','server.js') `
    -WorkingDirectory $webStandalone -Environment $webEnv -LogsRoot $paths.logs
  $records += Get-ProcessRecord -Process $web -Role 'web'
  if (-not (Wait-TcpPort -Address '127.0.0.1' -Port ([int]$config.network.web_port) -TimeoutSeconds 90)) { throw 'WEB_LISTENER_TIMEOUT' }
  $health = $null
  $healthDeadline = [DateTime]::UtcNow.AddSeconds(90)
  do {
    try {
      $candidateHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$($config.network.web_port)/api/healthz" -TimeoutSec 15 -ErrorAction Stop
      if (
        [bool]$candidateHealth.ok -and
        [string]$candidateHealth.status -eq 'ready' -and
        [string]$candidateHealth.build_id -eq $expectedWebBuild -and
        @($candidateHealth.failures).Count -eq 0
      ) {
        $health = $candidateHealth
        break
      }
    } catch {
      $candidateHealth = $null
    }
    if ([DateTime]::UtcNow -lt $healthDeadline) { Start-Sleep -Milliseconds 500 }
  } while ([DateTime]::UtcNow -lt $healthDeadline)
  if (-not $health) { throw 'WEB_HEALTH_FAILED' }

  $authPath = Write-UbuntuAuthMaterial -SessionSecret $sessionSecret
  $state = [ordered]@{
    schema='evomind.windows_node_process_state.v1'
    started_at_utc=[DateTime]::UtcNow.ToString('o')
    owner=[Security.Principal.WindowsIdentity]::GetCurrent().Name
    release_commit=$script:ReleaseCommit
    records=$records
    public_edge_auth_path=$authPath
    gpu_state=[string]$config.hpc.state
    service_clock_anchor=[ordered]@{path=$clockAnchorPath;sha256=(Get-Sha256File $clockAnchorPath);service_instance_id=$serviceInstanceId;boot_id=$bootId;node_qpc_self_check=$true}
  }
  Write-JsonAtomic -Path $paths.state -Value $state
  Set-OwnerOnlyAcl -Path $paths.state
  [ordered]@{ status='started'; release_commit=$script:ReleaseCommit; listeners=@('127.0.0.1:65068','127.0.0.1:8765','127.0.0.1:8088','127.0.0.1:7890'); public_hostname=[string]$config.network.public_hostname; gpu_state=$config.hpc.state } | ConvertTo-Json -Depth 5
} catch {
  foreach ($record in @($records | Select-Object -Last 99)) {
    try { Stop-VerifiedProcessRecord -Record $record | Out-Null } catch {}
  }
  if ($hpcBridgeStarted) { try { Stop-SafeHpcBridge -Port 7890 | Out-Null } catch {} }
  throw
} finally {
  $upstreamKey = $null; $clientKey = $null; $sessionSecret = $null; $adminPasswordScrypt = $null
  Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:WORKSTATION_SESSION_SECRET -ErrorAction SilentlyContinue
}
