param(
  [ValidateSet("restart", "start", "status", "smoke")]
  [string]$Command = "restart",
  [int]$Port = 8088,
  [string]$HostName = "127.0.0.1",
  [string]$OpenAIModel = "gpt-5.6-sol",
  [string]$OpenAIBaseUrl = "http://127.0.0.1:65068/v1",
  [string]$DeepSeekModel = "deepseek-v4-pro",
  [string]$ClaudeModel = "claude-opus-4-8",
  [switch]$AllowRealExternal,
  [switch]$AllowResourceBlockers,
  [switch]$SkipFullAcceptance,
  [switch]$Build
)

$ErrorActionPreference = "Stop"
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
  # Best effort for legacy Windows PowerShell.
}
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$ManagedSecretsDir = if ($env:EVOMIND_SECRETS_DIR) { [IO.Path]::GetFullPath($env:EVOMIND_SECRETS_DIR) } else { Join-Path $env:APPDATA "EvoMind\secrets" }

function Resolve-DpapiStateFile([string]$Name) {
  $managed = Join-Path $ManagedSecretsDir $Name
  $legacy = Join-Path $StateDir $Name
  $candidates = @($managed, $legacy) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
  if ($candidates.Count -eq 0) { return $managed }
  # The migration preserves the legacy source. Selecting the newest copy keeps
  # post-migration rotations made by an older credential manager visible while
  # central storage remains the preferred destination on equal timestamps.
  return ($candidates | Sort-Object @{ Expression = { (Get-Item -LiteralPath $_).LastWriteTimeUtc }; Descending = $true }, @{ Expression = { if ($_ -eq $managed) { 0 } else { 1 } }; Descending = $false } | Select-Object -First 1)
}

$OpenAICredentialPath = Resolve-DpapiStateFile "openai_api_key.xml"
$OpenAIMetadataPath = Resolve-DpapiStateFile "openai_gateway_metadata.json"
$DeepSeekCredentialPath = Resolve-DpapiStateFile "deepseek_api_key.xml"
$ClaudeCredentialPath = Resolve-DpapiStateFile "anthropic_api_key.xml"
$KaggleCredentialPath = Resolve-DpapiStateFile "kaggle_api_token.xml"
$HpcSshCredentialPath = Resolve-DpapiStateFile "hpc_ssh_credential.xml"
$HpcSshMetadataPath = Resolve-DpapiStateFile "hpc_ssh_metadata.json"
$KaggleAccessTokenUserName = "__KAGGLE_API_TOKEN__"
$AuditJsonPath = Join-Path $Root "docs\verified_workstation_launch_audit.json"
$AuditMarkdownPath = Join-Path $Root "docs\verified_workstation_launch_audit.md"

function Get-PythonExe {
  $candidates = @()
  if ($env:WORKSTATION_PYTHON) { $candidates += $env:WORKSTATION_PYTHON }
  $candidates += @(
    (Join-Path $Root "runtime\python\python.exe"),
    (Join-Path $Root "runtime\python.exe"),
    (Join-Path $Root ".venv\Scripts\python.exe"),
    "python.exe",
    "python"
  )
  foreach ($candidate in $candidates) {
    try {
      $resolved = Get-Command $candidate -ErrorAction Stop
      if ($resolved.Source) { return $resolved.Source }
    } catch {}
    if (Test-Path $candidate) { return $candidate }
  }
  throw "Python executable was not found."
}

function Enable-InstalledDpapiSecrets {
  $loaded = [ordered]@{
    openai = $false
    deepseek = $false
    claude = $false
    kaggle = $false
    hpc_ssh = $false
  }

  if (Test-Path $OpenAICredentialPath) {
    if (-not (Test-Path $OpenAIMetadataPath)) {
      throw "OpenAI gateway metadata is missing."
    }
    $credential = Import-Clixml -LiteralPath $OpenAICredentialPath
    $metadata = Get-Content -LiteralPath $OpenAIMetadataPath -Raw | ConvertFrom-Json
    $configuredBaseUrl = ([string]$metadata.base_url).TrimEnd('/')
    $configuredModel = [string]$metadata.model
    $expectedBaseUrl = $OpenAIBaseUrl.TrimEnd('/')
    if ($configuredBaseUrl -ne $expectedBaseUrl) {
      throw "OpenAI gateway metadata does not match the approved loopback endpoint."
    }
    if ([string]::IsNullOrWhiteSpace($configuredModel)) {
      $configuredModel = $OpenAIModel
    }
    $env:OPENAI_API_KEY = $credential.GetNetworkCredential().Password
    $env:OPENAI_BASE_URL = $configuredBaseUrl
    $env:OPENAI_MODEL = $configuredModel
    $env:EVOLUTION_PRIMARY_PROVIDER = "openai"
    $env:EVOLUTION_PROVIDER_STRICT = "1"
    $env:LLM_PROVIDER = "openai"
    $loaded.openai = $true
  }

  if (Test-Path $DeepSeekCredentialPath) {
    $credential = Import-Clixml -Path $DeepSeekCredentialPath
    $env:DEEPSEEK_API_KEY = $credential.GetNetworkCredential().Password
    $env:DEEPSEEK_MODEL = $DeepSeekModel
    $env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    $loaded.deepseek = $true
  }

  if (Test-Path $ClaudeCredentialPath) {
    $credential = Import-Clixml -Path $ClaudeCredentialPath
    $env:ANTHROPIC_API_KEY = $credential.GetNetworkCredential().Password
    $env:CLAUDE_CODE_MODEL = $ClaudeModel
    $loaded.claude = $true
  }

  if (Test-Path $KaggleCredentialPath) {
    $credential = Import-Clixml -Path $KaggleCredentialPath
    $secret = $credential.GetNetworkCredential().Password
    if ($credential.UserName -eq $KaggleAccessTokenUserName -or $secret -match "^KGAT_[A-Za-z0-9_-]{16,}$") {
      Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
      Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue
      $env:KAGGLE_API_TOKEN = $secret
    } else {
      Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
      $env:KAGGLE_USERNAME = $credential.UserName
      $env:KAGGLE_KEY = $secret
    }
    $loaded.kaggle = $true
  }

  if (Test-Path $HpcSshCredentialPath) {
    $credential = Import-Clixml -Path $HpcSshCredentialPath
    $metadata = $null
    if (Test-Path $HpcSshMetadataPath) {
      $metadata = Get-Content -LiteralPath $HpcSshMetadataPath -Raw | ConvertFrom-Json
    }
    $env:GPU_SSH_HOST = if ($metadata -and $metadata.host) { [string]$metadata.host } else { "100.85.169.63" }
    $env:GPU_SSH_PORT = if ($metadata -and $metadata.port) { [string]$metadata.port } else { "1235" }
    $env:GPU_SSH_USER = $credential.UserName
    $env:GPU_SSH_PASSWORD = $credential.GetNetworkCredential().Password
    if ($metadata -and ($null -ne $metadata.socks_host) -and ([string]$metadata.socks_host).Length -gt 0) {
      $env:GPU_SSH_SOCKS_HOST = [string]$metadata.socks_host
    } else {
      Remove-Item Env:GPU_SSH_SOCKS_HOST -ErrorAction SilentlyContinue
    }
    if ($metadata -and ($null -ne $metadata.socks_port) -and ([string]$metadata.socks_port).Length -gt 0) {
      $env:GPU_SSH_SOCKS_PORT = [string]$metadata.socks_port
    } else {
      Remove-Item Env:GPU_SSH_SOCKS_PORT -ErrorAction SilentlyContinue
    }
    $allowedRemoteWorkspace = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
    if (-not $metadata -or ([string]$metadata.remote_workspace).TrimEnd('/') -ne $allowedRemoteWorkspace) {
      throw "HPC metadata is missing or outside the dedicated EvoMind remote root."
    }
    $env:GPU_REMOTE_WORKSPACE = $allowedRemoteWorkspace
    $loaded.hpc_ssh = $true
  }

  $loaded
}

function Invoke-JsonCommand {
  param(
    [string]$Label,
    [string]$Executable,
    [string[]]$Arguments,
    [switch]$AllowFailure
  )
  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $output = & $Executable @Arguments 2>&1
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  [ordered]@{
    label = $Label
    command = "$Executable $($Arguments -join ' ')"
    exit_code = $exitCode
    output = ($output -join "`n")
    ok = ($exitCode -eq 0)
    allow_failure = [bool]$AllowFailure
  }
  if ($exitCode -ne 0 -and -not $AllowFailure) {
    throw "Verified workstation step failed: $Label"
  }
}

function Invoke-SmokeSuite {
  param(
    [hashtable]$Loaded,
    [string]$Python
  )
  $results = @()
  $baseUrl = "http://127.0.0.1:$Port"

  $results += Invoke-JsonCommand -Label "backend_resource_status" -Executable $Python -Arguments @(
    (Join-Path $Root "scripts\verify_backend_resource_status.py"),
    "--url",
    $baseUrl
  )

  if ($Loaded.openai) {
    $results += Invoke-JsonCommand -Label "openai_gateway_smoke" -Executable $Python -Arguments @(
      (Join-Path $Root "scripts\verify_openai_gateway.py"),
      "--output",
      (Join-Path $Root "workspace\llm\openai_gateway_smoke_current.json")
    )
  } elseif ($Loaded.deepseek) {
    $results += Invoke-JsonCommand -Label "deepseek_smoke" -Executable $Python -Arguments @(
      (Join-Path $Root "scripts\verify_deepseek_provider.py"),
      "--url",
      $baseUrl,
      "--require-configured"
    )
  }

  $gatewayArgs = @(
    (Join-Path $Root "scripts\verify_external_resource_gateways.py"),
    "--url",
    $baseUrl
  )
  if ($AllowRealExternal) {
    $gatewayArgs += "--allow-real-external"
  }
  $results += Invoke-JsonCommand -Label "external_gateway_smoke" -Executable $Python -Arguments $gatewayArgs -AllowFailure:$AllowResourceBlockers

  $kaggleArgs = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $Root "scripts\manage_kaggle_secret.ps1"),
    "smoke"
  )
  if ($AllowRealExternal) {
    $kaggleArgs += "-AllowRealExternal"
  }
  $results += Invoke-JsonCommand -Label "kaggle_secret_smoke" -Executable "powershell" -Arguments $kaggleArgs

  if (-not $SkipFullAcceptance) {
    $results += Invoke-JsonCommand -Label "full_acceptance" -Executable $Python -Arguments @(
      (Join-Path $Root "scripts\run_full_acceptance.py"),
      "--dashboard-url",
      $baseUrl
    )
  }

  $results += Invoke-JsonCommand -Label "plaintext_secret_scan" -Executable $Python -Arguments @(
    (Join-Path $Root "scripts\verify_no_plaintext_secrets.py")
  )

  $results
}

function Get-StringSha256 {
  param([string]$Text)
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
  } finally {
    $sha.Dispose()
  }
}

function Convert-ResultSummary {
  param([object[]]$Results)
  $Results | ForEach-Object {
    $output = [string]$_.output
    [ordered]@{
      label = $_.label
      command = $_.command
      exit_code = $_.exit_code
      ok = $_.ok
      allow_failure = $_.allow_failure
      output_sha256 = Get-StringSha256 $output
      output_excerpt = if ($output.Length -gt 900) { $output.Substring(0, 900) } else { $output }
    }
  }
}

function Write-Utf8FileAtomic {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Value
  )
  $directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Path $directory -Force | Out-Null
  $temporary = Join-Path $directory ("." + [System.IO.Path]::GetFileName($Path) + "." + $PID + "." + [guid]::NewGuid().ToString("N") + ".tmp")
  [System.IO.File]::WriteAllText($temporary, $Value, [System.Text.UTF8Encoding]::new($false))
  try {
    for ($attempt = 1; $attempt -le 8; $attempt++) {
      try {
        Move-Item -LiteralPath $temporary -Destination $Path -Force -ErrorAction Stop
        return
      } catch {
        if ($attempt -eq 8) { throw }
        Start-Sleep -Milliseconds (75 * $attempt)
      }
    }
  } finally {
    if (Test-Path -LiteralPath $temporary) {
      Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
  }
}

function Write-VerifiedAuditReport {
  param(
    [string]$LaunchCommand,
    [hashtable]$Loaded,
    [object[]]$Results
  )
  $resultSummaries = @(Convert-ResultSummary -Results $Results)
  $overallPassed = -not ($resultSummaries | Where-Object { -not $_.ok -and -not $_.allow_failure })
  $statusText = if ($overallPassed) { "passed" } else { "failed" }
  $remainingRequirements = @()
  if (-not $Loaded.openai) {
    $remainingRequirements += "Optional local gateway credential for streaming/tool calls; deterministic fallback is active"
  }
  if (-not $Loaded.kaggle) {
    $remainingRequirements += "KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY"
  }
  if (-not $Loaded.hpc_ssh) {
    $remainingRequirements += "GPU SSH environment credential"
  }
  $report = [ordered]@{
    status = $statusText
    generated_at = (Get-Date).ToString("s")
    command = $LaunchCommand
    dashboard_url = "http://${HostName}:$Port"
    dpapi_loaded = $Loaded
    active_llm = [ordered]@{
      provider = $(if ($Loaded.openai -and $gatewayReady) { "openai" } else { "local_fallback" })
      model = $(if ($Loaded.openai -and $gatewayReady) { $env:OPENAI_MODEL } else { "deterministic" })
      base_url = $(if ($Loaded.openai) { $env:OPENAI_BASE_URL } else { $OpenAIBaseUrl })
      streaming = [bool]($Loaded.openai -and $gatewayReady)
      tool_calling = [bool]($Loaded.openai -and $gatewayReady)
    }
    allow_real_external = [bool]$AllowRealExternal
    allow_resource_blockers = [bool]$AllowResourceBlockers
    skipped_full_acceptance = [bool]$SkipFullAcceptance
    secret_policy = "No secret values are written to this audit report; only DPAPI presence booleans, command labels, exit codes, hashes and short verifier excerpts are recorded."
    result_summaries = $resultSummaries
    remaining_external_requirements = $remainingRequirements
  }
  $json = $report | ConvertTo-Json -Depth 8
  Write-Utf8FileAtomic -Path $AuditJsonPath -Value ($json + "`n")

  $lines = @()
  $lines += "# Verified Workstation Launch Audit"
  $lines += ""
  $lines += "- Generated at: $($report.generated_at)"
  $lines += "- Status: $($report.status)"
  $lines += "- Dashboard: $($report.dashboard_url)"
  $lines += "- Active LLM: $($report.active_llm.provider) / $($report.active_llm.model)"
  $lines += "- OpenAI base URL: $($report.active_llm.base_url)"
  $lines += "- OpenAI streaming enabled: $($report.active_llm.streaming)"
  $lines += "- OpenAI tool calling enabled: $($report.active_llm.tool_calling)"
  $lines += "- OpenAI DPAPI: $($Loaded.openai)"
  $lines += "- DeepSeek DPAPI: $($Loaded.deepseek)"
  $lines += "- Claude DPAPI: $($Loaded.claude)"
  $lines += "- Kaggle DPAPI: $($Loaded.kaggle)"
  $lines += "- HPC SSH DPAPI: $($Loaded.hpc_ssh)"
  $lines += "- Real external calls: $([bool]$AllowRealExternal)"
  $lines += "- Resource blockers allowed: $([bool]$AllowResourceBlockers)"
  $lines += ""
  $lines += "## Check Results"
  $lines += ""
  foreach ($item in $resultSummaries) {
    $lines += "- " + $item.label + ": exit " + $item.exit_code + ", ok " + $item.ok + ", sha256 " + $item.output_sha256
  }
  $lines += ""
  $lines += "## Remaining External Conditions"
  $lines += ""
  foreach ($item in $report.remaining_external_requirements) {
    $lines += "- $item"
  }
  $lines += ""
  $lines += "## Security Note"
  $lines += ""
  $lines += $report.secret_policy
  Write-Utf8FileAtomic -Path $AuditMarkdownPath -Value (($lines -join "`n") + "`n")

  [ordered]@{
    json = $AuditJsonPath
    markdown = $AuditMarkdownPath
  }
}

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
$python = Get-PythonExe
$loaded = Enable-InstalledDpapiSecrets
$env:WORKSTATION_HOST = $HostName
$env:WORKSTATION_PORT = [string]$Port
$env:WORKSTATION_PYTHON = $python
$env:OPENAI_BASE_URL = $(if ($loaded.openai) { $env:OPENAI_BASE_URL } else { $OpenAIBaseUrl })
if (-not $loaded.openai) {
  $env:WORKSTATION_LOCAL_FALLBACK = "1"
  $env:EVOLUTION_PROVIDER_STRICT = "0"
  $env:LLM_PROVIDER = "local_fallback"
}
$gatewayArgs = @((Join-Path $Root "scripts\manage_local_gateway.py"), $(if ($Command -eq "status") { "status" } else { "start" }), "--base-url", $env:OPENAI_BASE_URL)
$gatewayOutput = & $python @gatewayArgs 2>&1
$gatewayExitCode = $LASTEXITCODE
if ($gatewayExitCode -ne 0) {
  throw "Local gateway lifecycle probe failed unexpectedly."
}
$gatewayStatus = $null
try { $gatewayStatus = ($gatewayOutput -join "`n") | ConvertFrom-Json } catch {}
$gatewayReady = [bool]($gatewayStatus -and $gatewayStatus.ready)
if (-not $gatewayReady) {
  $env:WORKSTATION_LOCAL_FALLBACK = "1"
  $env:EVOLUTION_PROVIDER_STRICT = "0"
  $env:LLM_PROVIDER = "local_fallback"
}

if ($Command -eq "status") {
  $managerOutput = & $python (Join-Path $Root "scripts\manage_workstation_dashboard.py") status --host $HostName --port $Port 2>&1
  $managerExitCode = $LASTEXITCODE
  $managerStatus = $null
  try { $managerStatus = ($managerOutput -join "`n") | ConvertFrom-Json } catch {}
  Write-Output ([ordered]@{
    status = "ok"
    dpapi_loaded = $loaded
    active_llm = [ordered]@{
      provider = $(if ($loaded.openai -and $gatewayReady) { "openai" } else { "local_fallback" })
      model = $(if ($loaded.openai -and $gatewayReady) { $env:OPENAI_MODEL } else { "deterministic" })
      base_url = $(if ($loaded.openai) { $env:OPENAI_BASE_URL } else { $OpenAIBaseUrl })
      streaming = [bool]($loaded.openai -and $gatewayReady)
      tool_calling = [bool]($loaded.openai -and $gatewayReady)
    }
    gateway = $gatewayStatus
    dashboard = $managerStatus
    dashboard_status_exit_code = $managerExitCode
    credential_paths = @{
      openai = $OpenAICredentialPath
      openai_metadata = $OpenAIMetadataPath
      deepseek = $DeepSeekCredentialPath
      claude = $ClaudeCredentialPath
      kaggle = $KaggleCredentialPath
      hpc_ssh = $HpcSshCredentialPath
      hpc_ssh_metadata = $HpcSshMetadataPath
    }
    dashboard_url = "http://${HostName}:$Port"
  } | ConvertTo-Json -Depth 5)
  exit 0
}

if ($Command -eq "start" -or $Command -eq "restart") {
  $managerCommand = if ($Command -eq "start") { "start" } else { "restart" }
  $managerArgs = @((Join-Path $Root "scripts\manage_workstation_dashboard.py"), $managerCommand, "--host", $HostName, "--port", [string]$Port, "--timeout", "90")
  if ($Build) { $managerArgs += "--build" }
  $managerOutput = & $python @managerArgs 2>&1
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  $managerStatus = $null
  try { $managerStatus = ($managerOutput -join "`n") | ConvertFrom-Json } catch {}
  Write-Output ([ordered]@{
    status = "passed"
    command = $Command
    dashboard_url = "http://${HostName}:$Port"
    dashboard = $managerStatus
    gateway = $gatewayStatus
    provider = $(if ($loaded.openai -and $gatewayReady) { "openai" } else { "local_fallback" })
    model = $(if ($loaded.openai -and $gatewayReady) { $env:OPENAI_MODEL } else { "deterministic" })
  } | ConvertTo-Json -Depth 8)
  exit 0
}

$smokeResults = Invoke-SmokeSuite -Loaded $loaded -Python $python
$auditPaths = Write-VerifiedAuditReport -LaunchCommand $Command -Loaded $loaded -Results $smokeResults
Write-Output ([ordered]@{
  status = "passed"
  command = $Command
  dashboard_url = "http://${HostName}:$Port"
  dpapi_loaded = $loaded
  gateway = $gatewayStatus
  allow_real_external = [bool]$AllowRealExternal
  allow_resource_blockers = [bool]$AllowResourceBlockers
  skipped_full_acceptance = [bool]$SkipFullAcceptance
  audit_paths = $auditPaths
  results = $smokeResults
} | ConvertTo-Json -Depth 8)
