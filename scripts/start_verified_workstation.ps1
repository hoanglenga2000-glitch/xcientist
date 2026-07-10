param(
  [ValidateSet("restart", "start", "status", "smoke")]
  [string]$Command = "restart",
  [int]$Port = 8088,
  [string]$DeepSeekModel = "deepseek-v4-pro",
  [string]$ClaudeModel = "claude-opus-4-8",
  [switch]$AllowRealExternal,
  [switch]$AllowResourceBlockers,
  [switch]$SkipFullAcceptance
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
$DeepSeekCredentialPath = Join-Path $StateDir "deepseek_api_key.xml"
$ClaudeCredentialPath = Join-Path $StateDir "anthropic_api_key.xml"
$KaggleCredentialPath = Join-Path $StateDir "kaggle_api_token.xml"
$HpcSshCredentialPath = Join-Path $StateDir "hpc_ssh_credential.xml"
$HpcSshMetadataPath = Join-Path $StateDir "hpc_ssh_metadata.json"
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

function Enable-InstalledDpapiSecrets {
  $loaded = [ordered]@{
    deepseek = $false
    claude = $false
    kaggle = $false
    hpc_ssh = $false
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
    $env:GPU_REMOTE_WORKSPACE = if ($metadata -and $metadata.remote_workspace) { [string]$metadata.remote_workspace } else { "/hpc2hdd/home/aimslab/research_agent_workstation" }
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
  if (-not $Loaded.claude) {
    $remainingRequirements += "ANTHROPIC_API_KEY"
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
    dashboard_url = "http://127.0.0.1:$Port"
    dpapi_loaded = $Loaded
    allow_real_external = [bool]$AllowRealExternal
    allow_resource_blockers = [bool]$AllowResourceBlockers
    skipped_full_acceptance = [bool]$SkipFullAcceptance
    secret_policy = "No secret values are written to this audit report; only DPAPI presence booleans, command labels, exit codes, hashes and short verifier excerpts are recorded."
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
  Set-Content -LiteralPath $AuditMarkdownPath -Value ($lines -join "`n") -Encoding UTF8

  [ordered]@{
    json = $AuditJsonPath
    markdown = $AuditMarkdownPath
  }
}

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
$python = Get-PythonExe
$loaded = Enable-InstalledDpapiSecrets

if ($Command -eq "status") {
  Write-Output ([ordered]@{
    status = "ok"
    dpapi_loaded = $loaded
    credential_paths = @{
      deepseek = $DeepSeekCredentialPath
      claude = $ClaudeCredentialPath
      kaggle = $KaggleCredentialPath
      hpc_ssh = $HpcSshCredentialPath
      hpc_ssh_metadata = $HpcSshMetadataPath
    }
    dashboard_url = "http://127.0.0.1:$Port"
  } | ConvertTo-Json -Depth 5)
  exit 0
}

if ($Command -eq "start" -or $Command -eq "restart") {
  $managerCommand = if ($Command -eq "start") { "start" } else { "restart" }
  & $python (Join-Path $Root "scripts\manage_workstation_dashboard.py") $managerCommand --port $Port --force --timeout 90
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$smokeResults = Invoke-SmokeSuite -Loaded $loaded -Python $python
$auditPaths = Write-VerifiedAuditReport -LaunchCommand $Command -Loaded $loaded -Results $smokeResults
Write-Output ([ordered]@{
  status = "passed"
  command = $Command
  dashboard_url = "http://127.0.0.1:$Port"
  dpapi_loaded = $loaded
  allow_real_external = [bool]$AllowRealExternal
  allow_resource_blockers = [bool]$AllowResourceBlockers
  skipped_full_acceptance = [bool]$SkipFullAcceptance
  audit_paths = $auditPaths
  results = $smokeResults
} | ConvertTo-Json -Depth 8)
