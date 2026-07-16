param(
  [ValidateSet("restart", "start", "status", "smoke")]
  [string]$Command = "restart",
  [int]$Port = 8088,
  [string]$DeepSeekModel = "deepseek-v4-pro",
  [string]$ClaudeModel = "claude-opus-4-8",
  [switch]$AllowRealExternal,
  [switch]$AllowResourceBlockers,
  [switch]$SkipFullAcceptance,
  [switch]$RunFullAcceptance
)

$ErrorActionPreference = "Stop"
if ($SkipFullAcceptance -and $RunFullAcceptance) {
  throw "SkipFullAcceptance and RunFullAcceptance cannot be combined."
}
$ShouldRunFullAcceptance = [bool]$RunFullAcceptance
try {
  [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
  $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
  # Best effort for legacy Windows PowerShell.
}
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
. (Join-Path $Root "scripts\dpapi_credential_store.ps1")
$StateDir = Get-EvoMindCredentialStateDirectory
$DeepSeekCredentialPath = Join-Path $StateDir "deepseek_api_key.xml"
$ClaudeCredentialPath = Join-Path $StateDir "anthropic_api_key.xml"
$KaggleCredentialPath = Join-Path $StateDir "kaggle_api_token.xml"
$HpcStorePaths = Get-EvoMindHpcCredentialStorePaths $StateDir
$KaggleAccessTokenUserName = "__KAGGLE_API_TOKEN__"
$AuditJsonPath = Join-Path $Root "docs\verified_workstation_launch_audit.json"
$AuditMarkdownPath = Join-Path $Root "docs\verified_workstation_launch_audit.md"

function Get-PythonExe {
  $candidates = @(
    "C:\codex-python\python.exe",
    "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
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

function Import-VerifiedDpapiCredential {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string]$ExpectedUser = ""
  )
  Assert-EvoMindCredentialDestination $Path
  Protect-EvoMindCredentialPath -Path $Path
  $credential = Import-Clixml -LiteralPath $Path
  if ($credential -isnot [System.Management.Automation.PSCredential] -or $credential.Password.Length -eq 0) {
    throw "DPAPI credential structure is invalid."
  }
  if ($ExpectedUser -and $credential.UserName -ne $ExpectedUser) {
    throw "DPAPI credential identity is invalid."
  }
  return $credential
}

function Enable-InstalledDpapiSecrets {
  $managedEnvironment = @(
    "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "CLAUDE_API_KEY", "CLAUDE_API_KEY_FILE", "CLAUDE_CODE_MODEL",
    "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY",
    "EVOMIND_HPC_HOST", "EVOMIND_HPC_PORT", "EVOMIND_HPC_USER", "EVOMIND_HPC_PASSWORD",
    "EVOMIND_HPC_REMOTE_WORKSPACE", "EVOMIND_HPC_SOCKS_HOST", "EVOMIND_HPC_SOCKS_PORT",
    "GPU_SSH_HOST", "GPU_SSH_PORT", "GPU_SSH_USER", "GPU_SSH_PASSWORD", "GPU_SSH_KEY_PATH",
    "GPU_REMOTE_WORKSPACE", "GPU_SSH_SOCKS_HOST", "GPU_SSH_SOCKS_PORT"
  )
  foreach ($name in $managedEnvironment) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
  }

  $loaded = [ordered]@{
    deepseek = $false
    claude = $false
    kaggle = $false
    hpc_ssh = $false
    credential_errors = [ordered]@{}
  }

  $storeMutex = Enter-EvoMindCredentialStoreLock
  try {
    $verifiedStateDir = Initialize-EvoMindCredentialStateDirectory
    if ([System.IO.Path]::GetFullPath($verifiedStateDir) -ne [System.IO.Path]::GetFullPath($StateDir)) {
      throw "DPAPI state directory verification failed."
    }

    if (Test-Path $DeepSeekCredentialPath) {
      try {
        $credential = Import-VerifiedDpapiCredential $DeepSeekCredentialPath "__DEEPSEEK_API_KEY__"
        $env:DEEPSEEK_API_KEY = $credential.GetNetworkCredential().Password
        $env:DEEPSEEK_MODEL = $DeepSeekModel
        $env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
        $loaded.deepseek = $true
      } catch {
        Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
        $loaded.credential_errors.deepseek = "invalid_or_unreadable"
      }
    }

    if (Test-Path $ClaudeCredentialPath) {
      try {
        $credential = Import-VerifiedDpapiCredential $ClaudeCredentialPath
        $env:ANTHROPIC_API_KEY = $credential.GetNetworkCredential().Password
        $env:CLAUDE_CODE_MODEL = $ClaudeModel
        $loaded.claude = $true
      } catch {
        Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
        $loaded.credential_errors.claude = "invalid_or_unreadable"
      }
    }

    if (Test-Path $KaggleCredentialPath) {
      $credential = Import-VerifiedDpapiCredential $KaggleCredentialPath
      $secretValue = $credential.GetNetworkCredential().Password
      if ($credential.UserName -eq $KaggleAccessTokenUserName -or $secretValue -match "^KGAT_[A-Za-z0-9_-]{16,}$") {
        Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
        Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue
        $env:KAGGLE_API_TOKEN = $secretValue
      } else {
        if ($credential.UserName -notmatch "^[A-Za-z0-9_.-]{1,128}$") { throw "Kaggle credential identity is invalid." }
        Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
        $env:KAGGLE_USERNAME = $credential.UserName
        $env:KAGGLE_KEY = $secretValue
      }
      $secretValue = $null
      $loaded.kaggle = $true
    }

    $hpcGeneration = Resolve-EvoMindHpcCredentialGeneration -StateDirectory $StateDir -MigrateLegacy
    if ($null -ne $hpcGeneration) {
      $credential = $hpcGeneration.Credential
      $metadata = $hpcGeneration.Metadata
      $sshHost = [string]$metadata.host
      $sshPort = [string]$metadata.port
      $remoteWorkspace = [string]$metadata.remote_workspace
      $secretValue = $credential.GetNetworkCredential().Password

      $env:EVOMIND_HPC_HOST = $sshHost
      $env:EVOMIND_HPC_PORT = $sshPort
      $env:EVOMIND_HPC_USER = $credential.UserName
      $env:EVOMIND_HPC_PASSWORD = $secretValue
      $env:EVOMIND_HPC_REMOTE_WORKSPACE = $remoteWorkspace
      $env:GPU_SSH_HOST = $sshHost
      $env:GPU_SSH_PORT = $sshPort
      $env:GPU_SSH_USER = $credential.UserName
      $env:GPU_SSH_PASSWORD = $secretValue
      $env:GPU_REMOTE_WORKSPACE = $remoteWorkspace
      if (([string]$metadata.socks_host).Length -gt 0) {
        $env:EVOMIND_HPC_SOCKS_HOST = [string]$metadata.socks_host
        $env:EVOMIND_HPC_SOCKS_PORT = [string]$metadata.socks_port
        $env:GPU_SSH_SOCKS_HOST = [string]$metadata.socks_host
        $env:GPU_SSH_SOCKS_PORT = [string]$metadata.socks_port
      } else {
        Remove-Item Env:EVOMIND_HPC_SOCKS_HOST -ErrorAction SilentlyContinue
        Remove-Item Env:EVOMIND_HPC_SOCKS_PORT -ErrorAction SilentlyContinue
        Remove-Item Env:GPU_SSH_SOCKS_HOST -ErrorAction SilentlyContinue
        Remove-Item Env:GPU_SSH_SOCKS_PORT -ErrorAction SilentlyContinue
      }
      $secretValue = $null
      $loaded.hpc_ssh = $true
    }
  } finally {
    Exit-EvoMindCredentialStoreLock $storeMutex
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
  $output = @()
  $exitCode = -1
  try {
    $resolvedCommand = Get-Command $Executable -ErrorAction Stop
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    $output = & $Executable @Arguments 2>&1
    $exitCode = $LASTEXITCODE
  } catch {
    $output = @([string]$_.Exception.Message)
    $exitCode = -1
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

  $kaggleReadinessArgs = @(
    (Join-Path $Root "scripts\verify_kaggle_dpapi_readiness.py"),
    "--write-report"
  )
  if ($AllowRealExternal) {
    $kaggleReadinessArgs += "--allow-real-external"
  }
  $results += Invoke-JsonCommand -Label "kaggle_dpapi_readiness" -Executable $Python -Arguments $kaggleReadinessArgs

  $results += Invoke-JsonCommand -Label "backend_resource_status" -Executable $Python -Arguments @(
    (Join-Path $Root "scripts\verify_backend_resource_status.py"),
    "--url",
    $baseUrl
  )

  if ($Loaded.deepseek) {
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
  $results += Invoke-JsonCommand -Label "kaggle_secret_smoke" -Executable "powershell" -Arguments $kaggleArgs

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

function Get-AllowlistedResultSignals {
  param(
    [string]$Label,
    [string]$Output
  )
  $signals = [ordered]@{
    code_agent_configured_not_invoked = $false
    code_agent_smoke_tested = $false
    gpu_configured_not_invoked = $false
    gpu_smoke_tested = $false
    gpu_resource_blocked = $false
    kaggle_configured_not_invoked = $false
    kaggle_authenticated_real_api = $false
    human_gate_required_for_submission = $false
  }
  try {
    $payload = $Output | ConvertFrom-Json -ErrorAction Stop
  } catch {
    return $signals
  }
  if ($Label -eq "external_gateway_smoke") {
    $codeAgent = if ($payload.code_agent) { $payload.code_agent } else { $payload.claude }
    $signals.code_agent_configured_not_invoked = ([string]$codeAgent.status -eq "configured_not_invoked")
    $signals.code_agent_smoke_tested = ([string]$codeAgent.status -eq "configured_smoke_tested")
    $signals.gpu_configured_not_invoked = ([string]$payload.gpu.status -eq "configured_not_invoked")
    $signals.gpu_smoke_tested = ([string]$payload.gpu.status -eq "configured_smoke_tested")
    $signals.gpu_resource_blocked = ([string]$payload.gpu.status -eq "configured_resource_blocked")
  } elseif ($Label -eq "kaggle_secret_smoke") {
    $signals.kaggle_configured_not_invoked = ([string]$payload.verification_state -eq "configured_not_invoked")
    $signals.kaggle_authenticated_real_api = (
      [string]$payload.verification_state -eq "authenticated_real_api" -and
      [bool]$payload.real_external_called
    )
    $signals.human_gate_required_for_submission = [bool]$payload.human_gate_required_for_submission
  } elseif ($Label -eq "kaggle_dpapi_readiness") {
    $signals.kaggle_authenticated_real_api = (
      [string]$payload.credential_status -eq "authenticated_real_api" -and
      [bool]$payload.authenticated
    )
    $signals.human_gate_required_for_submission = [bool]$payload.human_gate_required_for_submission
  }
  return $signals
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
      signals = Get-AllowlistedResultSignals -Label ([string]$_.label) -Output $output
    }
  }
}

function Write-PendingAuditReport {
  param([string]$RunId)
  $pending = [ordered]@{
    status = "running"
    run_id = $RunId
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    command = $Command
    dashboard_url = "http://127.0.0.1:$Port"
    claim_boundary = "This run has not completed verification. It must not be used as readiness evidence."
  }
  Set-Content -LiteralPath $AuditJsonPath -Value ($pending | ConvertTo-Json -Depth 4) -Encoding UTF8
  Set-Content -LiteralPath $AuditMarkdownPath -Value @(
    "# Verified Workstation Launch Audit",
    "",
    "- Status: running",
    "- Run ID: $RunId",
    "",
    $pending.claim_boundary
  ) -Encoding UTF8
}

function Write-VerifiedAuditReport {
  param(
    [string]$LaunchCommand,
    [string]$RunId,
    [hashtable]$Loaded,
    [object[]]$Results,
    [object]$DashboardRuntime
  )
  $resultSummaries = @(Convert-ResultSummary -Results $Results)
  $overallPassed = -not ($resultSummaries | Where-Object { -not $_.ok -and -not $_.allow_failure })
  $deepSeekRuntimeVerified = [bool]($resultSummaries | Where-Object { $_.label -eq "deepseek_smoke" -and $_.ok })
  $gatewayRuntimeVerified = [bool]($resultSummaries | Where-Object {
    $_.label -eq "external_gateway_smoke" -and $_.ok -and $_.signals.code_agent_smoke_tested
  })
  $providerRuntimeVerified = $deepSeekRuntimeVerified -or $gatewayRuntimeVerified
  $statusText = if (-not $overallPassed) {
    "failed"
  } elseif ($providerRuntimeVerified) {
    "passed"
  } else {
    "local_ready_external_unverified"
  }
  $remainingRequirements = @()
  if (-not $Loaded.claude -and -not $Loaded.deepseek) {
    $remainingRequirements += "Protected Anthropic or DeepSeek credential"
  } elseif (-not $providerRuntimeVerified) {
    $remainingRequirements += "Current live LLM provider smoke"
  }
  if (-not $Loaded.kaggle) {
    $remainingRequirements += "KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY"
  }
  if (-not $Loaded.hpc_ssh) {
    $remainingRequirements += "GPU SSH environment credential"
  }
  $report = [ordered]@{
    status = $statusText
    run_id = $RunId
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    command = $LaunchCommand
    dashboard_url = "http://127.0.0.1:$Port"
    dashboard_runtime = $DashboardRuntime
    dpapi_loaded = $Loaded
    external_provider_runtime_verified = [bool]$providerRuntimeVerified
    allow_real_external = [bool]$AllowRealExternal
    allow_resource_blockers = [bool]$AllowResourceBlockers
    skipped_full_acceptance = (-not $ShouldRunFullAcceptance)
    secret_policy = "No secret values or raw command output are written to this audit report; only DPAPI presence booleans, runtime identity, command labels, exit codes, hashes and allowlisted status signals are recorded."
    claim_boundary = if ($providerRuntimeVerified) {
      "The local production gateway and at least one external LLM provider passed a current runtime smoke. GPU, Kaggle submission, benchmark parity and model quality require separate evidence."
    } else {
      "The local production gateway passed, but no external LLM provider was invoked successfully in this run. Provider availability, GPU, Kaggle submission, benchmark parity and model quality are not verified."
    }
    result_summaries = $resultSummaries
    remaining_external_requirements = $remainingRequirements
  }
  $json = $report | ConvertTo-Json -Depth 8
  Set-Content -LiteralPath $AuditJsonPath -Value $json -Encoding UTF8

  $lines = @()
  $lines += "# Verified Workstation Launch Audit"
  $lines += ""
  $lines += "- Generated at: $($report.generated_at)"
  $lines += "- Status: $($report.status)"
  $lines += "- Dashboard: $($report.dashboard_url)"
  $lines += "- Run ID: $RunId"
  $lines += "- Dashboard mode: $($DashboardRuntime.mode)"
  $lines += "- Build ID: $($DashboardRuntime.build_id)"
  $lines += "- Source digest: $($DashboardRuntime.source_digest)"
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
  $lines += ""
  $lines += "## Claim Boundary"
  $lines += ""
  $lines += $report.claim_boundary
  Set-Content -LiteralPath $AuditMarkdownPath -Value ($lines -join "`n") -Encoding UTF8

  [ordered]@{
    json = $AuditJsonPath
    markdown = $AuditMarkdownPath
    status = $statusText
    external_provider_runtime_verified = [bool]$providerRuntimeVerified
  }
}

$python = Get-PythonExe
$env:WORKSTATION_PYTHON = $python
$runId = [Guid]::NewGuid().ToString("N")
if ($Command -ne "status") {
  Write-PendingAuditReport -RunId $runId
}
$loaded = Enable-InstalledDpapiSecrets

if ($Command -eq "status") {
  Write-Output ([ordered]@{
    status = "ok"
    dpapi_loaded = $loaded
    credential_paths = @{
      deepseek = $DeepSeekCredentialPath
      claude = $ClaudeCredentialPath
      kaggle = $KaggleCredentialPath
      hpc_ssh_current = $HpcStorePaths.PointerPath
      hpc_ssh_generations = $HpcStorePaths.GenerationsDirectory
    }
    workstation_python = $env:WORKSTATION_PYTHON
    dashboard_url = "http://127.0.0.1:$Port"
  } | ConvertTo-Json -Depth 5)
  exit 0
}

function Invoke-DashboardManagerJson {
  param([string[]]$Arguments)
  $previousErrorActionPreference = $ErrorActionPreference
  $raw = @()
  $exitCode = -1
  try {
    # Windows PowerShell 5.1 can promote native stderr records to terminating
    # errors under Stop even when the process returns success and valid JSON.
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    $raw = @(& $python (Join-Path $Root "scripts\manage_workstation_dashboard.py") @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  if ($exitCode -ne 0) {
    $diagnostic = @($raw | Select-Object -Last 20 | ForEach-Object { [string]$_ }) -join [Environment]::NewLine
    throw "Dashboard manager failed with exit code $exitCode.$([Environment]::NewLine)$diagnostic"
  }
  try {
    ($raw -join "`n") | ConvertFrom-Json
  } catch {
    throw "Dashboard manager did not return valid JSON."
  }
}

$dashboardPayload = if ($Command -eq "start" -or $Command -eq "restart") {
  $managerCommand = if ($Command -eq "start") { "start" } else { "restart" }
  Invoke-DashboardManagerJson -Arguments @($managerCommand, "--port", [string]$Port, "--force", "--timeout", "90", "--build")
} else {
  Invoke-DashboardManagerJson -Arguments @("status", "--port", [string]$Port)
}
$dashboardRuntime = if ($dashboardPayload.runtime_state) { $dashboardPayload.runtime_state } else { $dashboardPayload }
if (
  $dashboardRuntime.mode -ne "start" -or
  [string]::IsNullOrWhiteSpace([string]$dashboardRuntime.build_id) -or
  [string]::IsNullOrWhiteSpace([string]$dashboardRuntime.source_digest) -or
  -not [bool]$dashboardRuntime.build_requested
) {
  throw "Verified workstation requires a fresh production build identity."
}

$smokeResults = Invoke-SmokeSuite -Loaded $loaded -Python $python
if ($ShouldRunFullAcceptance) {
  $smokeResults += Invoke-JsonCommand -Label "full_acceptance" -Executable $python -Arguments @(
    (Join-Path $Root "scripts\run_full_acceptance.py"),
    "--dashboard-url",
    "http://127.0.0.1:$Port",
    "--skip-verified-launch-audit"
  )
}
$auditPaths = Write-VerifiedAuditReport `
  -LaunchCommand $Command `
  -RunId $runId `
  -Loaded $loaded `
  -Results $smokeResults `
  -DashboardRuntime $dashboardRuntime
$auditVerification = Invoke-JsonCommand -Label "verified_launch_audit" -Executable $python -Arguments @(
  (Join-Path $Root "scripts\verify_verified_workstation_launch_audit.py")
) -AllowFailure
if (-not $auditVerification.ok) {
  $failedReport = Get-Content -LiteralPath $AuditJsonPath -Raw | ConvertFrom-Json
  $failedReport.status = "failed_verification"
  $failedReport | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $AuditJsonPath -Encoding UTF8
  throw "Verified workstation audit self-check failed."
}
Write-Output ([ordered]@{
  status = $auditPaths.status
  command = $Command
  dashboard_url = "http://127.0.0.1:$Port"
  dpapi_loaded = $loaded
  allow_real_external = [bool]$AllowRealExternal
  allow_resource_blockers = [bool]$AllowResourceBlockers
  skipped_full_acceptance = (-not $ShouldRunFullAcceptance)
  external_provider_runtime_verified = $auditPaths.external_provider_runtime_verified
  audit_paths = $auditPaths
  audit_verification_sha256 = Get-StringSha256 ([string]$auditVerification.output)
  results = $smokeResults
} | ConvertTo-Json -Depth 8)
