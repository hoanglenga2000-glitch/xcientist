[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet("install-token", "status", "smoke", "remove")]
  [string]$Command = "status",
  [System.Security.SecureString]$SecureApiToken,
  [string]$Username = "",
  [System.Security.SecureString]$SecureKey,
  [switch]$SecretFromStdin,
  [switch]$AllowRealExternal
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}

$CredentialPath = $null
$CredentialUserMarker = "__KAGGLE_API_TOKEN__"
$StoreMutex = $null

function Write-Json([System.Collections.IDictionary]$Payload) {
  Write-Output ($Payload | ConvertTo-Json -Depth 8 -Compress)
}

function Read-Credential {
  if (-not (Test-Path -LiteralPath $CredentialPath)) {
    return $null
  }
  Assert-EvoMindCredentialDestination $CredentialPath
  Protect-EvoMindCredentialPath -Path $CredentialPath
  $credential = Import-Clixml -LiteralPath $CredentialPath
  if ($credential -isnot [System.Management.Automation.PSCredential]) {
    throw [System.IO.InvalidDataException]::new("Credential format is invalid.")
  }
  if ([string]::IsNullOrWhiteSpace($credential.UserName) -or $credential.Password.Length -eq 0) {
    throw [System.IO.InvalidDataException]::new("Credential structure is invalid.")
  }
  return $credential
}

function Get-ToolStatus {
  $python = Get-Command python -ErrorAction SilentlyContinue
  $version = ""
  if ($python) {
    $versionOutput = & $python.Source -c "import importlib.metadata; print(importlib.metadata.version('kaggle'))" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $version = ([string]$versionOutput).Trim()
    }
  }
  $cli = Get-Command kaggle -ErrorAction SilentlyContinue
  return [ordered]@{
    python_package_installed = (-not [string]::IsNullOrWhiteSpace($version))
    python_package_version = $version
    cli_path = $(if ($cli) { $cli.Source } else { "" })
  }
}

function Get-StatusPayload {
  $credential = Read-Credential
  $installed = $null -ne $credential
  $tokenType = if (-not $installed) {
    "none"
  } elseif ($credential.UserName -eq $CredentialUserMarker) {
    "access_token"
  } else {
    "legacy_username_key"
  }
  return [ordered]@{
    status = $(if ($installed) { "configured" } else { "not_configured" })
    configured = $installed
    credential_installed = $installed
    credential_status = $(if ($installed) { "configured_unverified" } else { "not_configured" })
    credential_path = $CredentialPath
    token_type = $tokenType
    storage = "windows_dpapi_current_user"
    human_gate_required_for_submission = $true
    tool_status = Get-ToolStatus
  }
}

function Invoke-RealSmoke([System.Management.Automation.PSCredential]$Credential) {
  $python = Get-Command python -ErrorAction Stop
  $oldToken = $env:KAGGLE_API_TOKEN
  $oldUsername = $env:KAGGLE_USERNAME
  $oldKey = $env:KAGGLE_KEY
  try {
    $secretValue = $Credential.GetNetworkCredential().Password
    if ($Credential.UserName -eq $CredentialUserMarker) {
      $env:KAGGLE_API_TOKEN = $secretValue
      Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue
      Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue
    } else {
      Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
      $env:KAGGLE_USERNAME = $Credential.UserName
      $env:KAGGLE_KEY = $secretValue
    }
    $secretValue = $null
    # Windows PowerShell promotes native stderr to ErrorRecord. Kaggle may emit a
    # harmless client-version warning on stderr even when the API call succeeds.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
      $ErrorActionPreference = "Continue"
      $result = & $python.Source -c "from kaggle.api.kaggle_api_extended import KaggleApi; api=KaggleApi(); api.authenticate(); response=api.competitions_list(); rows=getattr(response, 'competitions', response) or []; print(len(rows))" 2>$null
      $pythonExitCode = $LASTEXITCODE
    } finally {
      $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($pythonExitCode -ne 0) {
      throw [System.InvalidOperationException]::new("Kaggle API smoke failed.")
    }
    $count = 0
    if (-not [int]::TryParse(([string]$result).Trim(), [ref]$count)) {
      throw [System.IO.InvalidDataException]::new("Kaggle API response was invalid.")
    }
    return $count
  } finally {
    if ($null -eq $oldToken) { Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue } else { $env:KAGGLE_API_TOKEN = $oldToken }
    if ($null -eq $oldUsername) { Remove-Item Env:KAGGLE_USERNAME -ErrorAction SilentlyContinue } else { $env:KAGGLE_USERNAME = $oldUsername }
    if ($null -eq $oldKey) { Remove-Item Env:KAGGLE_KEY -ErrorAction SilentlyContinue } else { $env:KAGGLE_KEY = $oldKey }
  }
}

try {
  . (Join-Path $PSScriptRoot "dpapi_credential_store.ps1")
  $StoreMutex = Enter-EvoMindCredentialStoreLock
  try {
    $StateDir = Initialize-EvoMindCredentialStateDirectory
    $CredentialPath = Join-Path $StateDir "kaggle_api_token.xml"

    switch ($Command) {
    "install-token" {
      if (-not [string]::IsNullOrWhiteSpace($Username)) {
        if ($Username -notmatch "^[A-Za-z0-9_.-]{1,128}$") {
          throw [System.ArgumentException]::new("Kaggle username is invalid.")
        }
        if ($null -ne $SecureApiToken) {
          throw [System.ArgumentException]::new("Access-token and username/key modes cannot be combined.")
        }
        $credentialUser = $Username
        $secureValue = Read-EvoMindSecureInput `
          -Provided $SecureKey `
          -Prompt "Kaggle legacy API key" `
          -FromStdin:$SecretFromStdin
      } else {
        if ($null -ne $SecureKey) {
          throw [System.ArgumentException]::new("A legacy key requires a username.")
        }
        $credentialUser = $CredentialUserMarker
        $secureValue = Read-EvoMindSecureInput `
          -Provided $SecureApiToken `
          -Prompt "Kaggle API token" `
          -FromStdin:$SecretFromStdin
      }
      if ($secureValue.Length -eq 0) {
        throw [System.ArgumentException]::new("Credential input was empty.")
      }
      try {
        $credential = [System.Management.Automation.PSCredential]::new($credentialUser, $secureValue)
        $temporaryPath = Write-EvoMindCredentialTemp `
          -Credential $credential `
          -StateDirectory $StateDir `
          -Prefix "kaggle-credential"
        Commit-EvoMindCredentialFiles @(
          [ordered]@{ TemporaryPath = $temporaryPath; DestinationPath = $CredentialPath }
        )
      } finally {
        $secureValue.Dispose()
      }
      Write-Json (Get-StatusPayload)
      break
    }
    "status" {
      Write-Json (Get-StatusPayload)
      break
    }
    "smoke" {
      $payload = Get-StatusPayload
      $toolsReady = $payload.tool_status.python_package_installed -and -not [string]::IsNullOrWhiteSpace($payload.tool_status.cli_path)
      if (-not $toolsReady) {
        throw [System.InvalidOperationException]::new("Kaggle toolchain is unavailable.")
      }
      if (-not $payload.credential_installed) {
        if ($AllowRealExternal) {
          throw [System.InvalidOperationException]::new("Kaggle credential is unavailable.")
        }
        $payload.status = "not_configured"
        $payload.verification_state = "not_configured"
        $payload.local_smoke = "passed"
        $payload.real_external_called = $false
        Write-Json $payload
        break
      }
      $payload.local_smoke = "passed"
      $payload.real_external_called = [bool]$AllowRealExternal
      if ($AllowRealExternal) {
        $payload.competition_count = Invoke-RealSmoke (Read-Credential)
        $payload.status = "passed"
        $payload.credential_status = "authenticated_real_api"
        $payload.verification_state = "authenticated_real_api"
      } else {
        $payload.status = "configured_unverified"
        $payload.credential_status = "configured_unverified"
        $payload.verification_state = "configured_not_invoked"
      }
      Write-Json $payload
      break
    }
    "remove" {
      Remove-EvoMindCredentialFile $CredentialPath
      if (Test-Path -LiteralPath $CredentialPath) {
        throw [System.IO.IOException]::new("Credential removal failed.")
      }
      Write-Json (Get-StatusPayload)
      break
    }
    }
  } finally {
    Exit-EvoMindCredentialStoreLock $StoreMutex
    $StoreMutex = $null
  }
} catch {
  $installed = $false
  if (-not [string]::IsNullOrWhiteSpace([string]$CredentialPath)) {
    $installed = Test-Path -LiteralPath $CredentialPath -PathType Leaf
  }
  Write-Json ([ordered]@{
    status = "failed"
    configured = $false
    credential_installed = $installed
    credential_path = $CredentialPath
    error_code = "credential_operation_failed"
    error_type = $_.Exception.GetType().Name
  })
  exit 1
}
exit 0
