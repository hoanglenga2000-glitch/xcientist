[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet("install", "install-key", "status", "remove")]
  [string]$Command = "status",
  [System.Security.SecureString]$SecureApiKey,
  [switch]$SecretFromStdin
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}

$CredentialPath = $null
$CredentialUser = "__DEEPSEEK_API_KEY__"
$StoreMutex = $null

function Write-Json([System.Collections.IDictionary]$Payload) {
  Write-Output ($Payload | ConvertTo-Json -Depth 6 -Compress)
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
  if ($credential.UserName -ne $CredentialUser -or $credential.Password.Length -eq 0) {
    throw [System.IO.InvalidDataException]::new("Credential structure is invalid.")
  }
  return $credential
}

try {
  . (Join-Path $PSScriptRoot "dpapi_credential_store.ps1")
  $StoreMutex = Enter-EvoMindCredentialStoreLock
  try {
    $StateDir = Initialize-EvoMindCredentialStateDirectory
    $CredentialPath = Join-Path $StateDir "deepseek_api_key.xml"

    switch ($Command) {
    { $_ -in @("install", "install-key") } {
      $secureValue = Read-EvoMindSecureInput `
        -Provided $SecureApiKey `
        -Prompt "DeepSeek API key" `
        -FromStdin:$SecretFromStdin
      if ($secureValue.Length -eq 0) {
        throw [System.ArgumentException]::new("Credential input was empty.")
      }
      try {
        $credential = [System.Management.Automation.PSCredential]::new($CredentialUser, $secureValue)
        $temporaryPath = Write-EvoMindCredentialTemp `
          -Credential $credential `
          -StateDirectory $StateDir `
          -Prefix "deepseek-api-key"
        Commit-EvoMindCredentialFiles @(
          [ordered]@{ TemporaryPath = $temporaryPath; DestinationPath = $CredentialPath }
        )
      } finally {
        $secureValue.Dispose()
      }
      if ($null -eq (Read-Credential)) {
        throw [System.IO.InvalidDataException]::new("Credential verification failed.")
      }
      Write-Json ([ordered]@{
        status = "configured"
        credential_installed = $true
        credential_path = $CredentialPath
        provider = "deepseek"
        storage = "windows_dpapi_current_user"
      })
      break
    }
    "status" {
      $credential = Read-Credential
      Write-Json ([ordered]@{
        status = $(if ($null -ne $credential) { "configured" } else { "not_configured" })
        credential_installed = ($null -ne $credential)
        credential_path = $CredentialPath
        provider = "deepseek"
        storage = "windows_dpapi_current_user"
      })
      break
    }
    "remove" {
      Remove-EvoMindCredentialFile $CredentialPath
      if (Test-Path -LiteralPath $CredentialPath) {
        throw [System.IO.IOException]::new("Credential removal failed.")
      }
      Write-Json ([ordered]@{
        status = "not_configured"
        credential_installed = $false
        credential_path = $CredentialPath
        provider = "deepseek"
        storage = "windows_dpapi_current_user"
      })
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
    credential_installed = $installed
    credential_path = $CredentialPath
    error_code = "credential_operation_failed"
    error_type = $_.Exception.GetType().Name
  })
  exit 1
}
exit 0
