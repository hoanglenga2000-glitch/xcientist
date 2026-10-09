param(
  [ValidateSet("install-key", "status", "restart-dashboard", "smoke")]
  [string]$Command = "status",
  [string]$ApiKey = "",
  [string]$Model = "claude-opus-4-8",
  [int]$Port = 8088,
  [switch]$AllowRealExternal
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "anthropic_api_key.xml"

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

function Import-ClaudeCredential {
  if (-not (Test-Path $CredentialPath)) {
    throw "Claude API key is not installed. Run: powershell -ExecutionPolicy Bypass -File scripts\manage_claude_secret.ps1 install-key -ApiKey <key>"
  }
  Import-Clixml -Path $CredentialPath
}

function Set-ClaudeEnvironment {
  $credential = Import-ClaudeCredential
  $env:ANTHROPIC_API_KEY = $credential.GetNetworkCredential().Password
  $env:CLAUDE_CODE_MODEL = $Model
}

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

if ($Command -eq "install-key") {
  if (-not $ApiKey) {
    throw "ApiKey is required for install-key."
  }
  if ($ApiKey -notmatch "^sk-ant-[A-Za-z0-9_-]{20,}$") {
    throw "ApiKey format is not an Anthropic sk-ant-* token."
  }
  $secure = ConvertTo-SecureString $ApiKey -AsPlainText -Force
  $credential = [pscredential]::new("anthropic", $secure)
  $credential | Export-Clixml -Path $CredentialPath
  Write-Output (@{
    status = "installed"
    credential_installed = $true
    credential_path = $CredentialPath
    model = $Model
    allow_real_external_required_for_sdk_smoke = $true
  } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "status") {
  Write-Output (@{
    status = $(if (Test-Path $CredentialPath) { "configured" } else { "not_configured" })
    credential_installed = (Test-Path $CredentialPath)
    credential_path = $CredentialPath
    model = $Model
    smoke_command = "powershell -ExecutionPolicy Bypass -File scripts\manage_claude_secret.ps1 smoke -AllowRealExternal"
  } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "restart-dashboard") {
  Set-ClaudeEnvironment
  $python = Get-PythonExe
  & $python (Join-Path $Root "scripts\manage_workstation_dashboard.py") restart --port $Port --force --timeout 90
  exit $LASTEXITCODE
}

if ($Command -eq "smoke") {
  $python = Get-PythonExe
  $verifyArgs = @(
    (Join-Path $Root "scripts\verify_external_resource_gateways.py"),
    "--url",
    "http://127.0.0.1:$Port"
  )
  if (Test-Path $CredentialPath) {
    Set-ClaudeEnvironment
    if ($AllowRealExternal) {
      $verifyArgs += "--allow-real-external"
    }
  }
  & $python @verifyArgs
  exit $LASTEXITCODE
}
