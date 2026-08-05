[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet("install-credential", "install-password", "set-metadata", "status", "remove")]
  [string]$Command = "status",
  [string]$User = "",
  [System.Security.SecureString]$SecurePassword,
  [switch]$SecretFromStdin,
  [Alias("Host")]
  [string]$HostName = "",
  [int]$Port = 0,
  [string]$RemoteWorkspace = "",
  [string]$SocksHost = "",
  [int]$SocksPort = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}

$StateDir = $null
$StorePaths = $null
$StoreMutex = $null
$PreparedMetadata = $null
$SecureValue = $null

function Write-Json([System.Collections.IDictionary]$Payload) {
  Write-Output ($Payload | ConvertTo-Json -Depth 8 -Compress)
}

function Assert-SafeHost([string]$Value, [switch]$AllowEmpty) {
  Assert-EvoMindNetworkHost $Value -AllowEmpty:$AllowEmpty
}

function Assert-SafeRemoteWorkspace([string]$Value) {
  Assert-EvoMindRemoteWorkspace $Value
}

function New-MetadataPayload {
  Assert-SafeHost $HostName
  Assert-SafeHost $SocksHost -AllowEmpty
  Assert-SafeRemoteWorkspace $RemoteWorkspace
  if ($Port -lt 1 -or $Port -gt 65535) {
    throw [System.ArgumentOutOfRangeException]::new("SSH port is invalid.")
  }
  $hasSocksHost = -not [string]::IsNullOrWhiteSpace($SocksHost)
  $hasSocksPort = $SocksPort -gt 0
  if ($hasSocksHost -ne $hasSocksPort) {
    throw [System.ArgumentException]::new("SOCKS host and port must be configured together.")
  }
  if ($hasSocksPort -and $SocksPort -gt 65535) {
    throw [System.ArgumentOutOfRangeException]::new("SOCKS port is invalid.")
  }
  return [ordered]@{
    host = $HostName.Trim()
    port = $Port
    remote_workspace = $RemoteWorkspace.Trim()
    socks_host = $SocksHost.Trim()
    socks_port = $(if ($hasSocksPort) { $SocksPort } else { 0 })
  }
}

function Get-StatusPayload {
  $current = Resolve-EvoMindHpcCredentialGeneration -StateDirectory $StateDir -MigrateLegacy
  $configured = $null -ne $current
  $credential = if ($configured) { $current.Credential } else { $null }
  $metadata = if ($configured) { $current.Metadata } else { $null }
  return [ordered]@{
    status = $(if ($configured) { "configured" } else { "not_configured" })
    credential_installed = $configured
    current_pointer_path = $StorePaths.PointerPath
    generations_path = $StorePaths.GenerationsDirectory
    generation_id = $(if ($configured) { $current.GenerationId } else { $null })
    credential_path = $(if ($configured) { $current.CredentialPath } else { $null })
    metadata_path = $(if ($configured) { $current.MetadataPath } else { $null })
    user = $(if ($configured) { $credential.UserName } else { $null })
    host = $(if ($configured) { [string]$metadata.host } else { $null })
    port = $(if ($configured) { [int]$metadata.port } else { $null })
    remote_workspace = $(if ($configured) { [string]$metadata.remote_workspace } else { $null })
    socks_host = $(if ($configured) { [string]$metadata.socks_host } else { $null })
    socks_port = $(if ($configured) { [int]$metadata.socks_port } else { $null })
    storage = "windows_dpapi_current_user_generation_v1"
  }
}

try {
  . (Join-Path $PSScriptRoot "dpapi_credential_store.ps1")
  if ($Command -in @("install-credential", "install-password")) {
    if ($User -notmatch "^[A-Za-z0-9._-]{1,128}$") {
      throw [System.ArgumentException]::new("SSH user is invalid.")
    }
    $PreparedMetadata = New-MetadataPayload
    $SecureValue = Read-EvoMindSecureInput `
      -Provided $SecurePassword `
      -Prompt "HPC SSH password" `
      -FromStdin:$SecretFromStdin
    if ($SecureValue.Length -eq 0) {
      throw [System.ArgumentException]::new("Credential input was empty.")
    }
  } elseif ($Command -eq "set-metadata") {
    $PreparedMetadata = New-MetadataPayload
  }

  $StoreMutex = Enter-EvoMindCredentialStoreLock
  try {
    $StateDir = Initialize-EvoMindCredentialStateDirectory
    $StorePaths = Get-EvoMindHpcCredentialStorePaths $StateDir

    switch ($Command) {
    { $_ -in @("install-credential", "install-password") } {
      $credential = [System.Management.Automation.PSCredential]::new($User, $SecureValue)
      [void](New-EvoMindHpcCredentialGeneration `
        -StateDirectory $StateDir `
        -Credential $credential `
        -Metadata $PreparedMetadata)
      Write-Json (Get-StatusPayload)
      break
    }
    "set-metadata" {
      $current = Resolve-EvoMindHpcCredentialGeneration -StateDirectory $StateDir -MigrateLegacy
      if ($null -eq $current) {
        throw [System.InvalidOperationException]::new("HPC credential is not configured.")
      }
      [void](New-EvoMindHpcCredentialGeneration `
        -StateDirectory $StateDir `
        -Credential $current.Credential `
        -Metadata $PreparedMetadata)
      Write-Json (Get-StatusPayload)
      break
    }
    "status" {
      Write-Json (Get-StatusPayload)
      break
    }
    "remove" {
      Remove-EvoMindHpcCredentialStore $StateDir
      Write-Json (Get-StatusPayload)
      break
    }
    }
  } finally {
    Exit-EvoMindCredentialStoreLock $StoreMutex
    $StoreMutex = $null
  }
} catch {
  Write-Json ([ordered]@{
    status = "failed"
    credential_installed = $false
    current_pointer_path = $(if ($null -ne $StorePaths) { $StorePaths.PointerPath } else { $null })
    generations_path = $(if ($null -ne $StorePaths) { $StorePaths.GenerationsDirectory } else { $null })
    error_code = "credential_operation_failed"
    error_type = $_.Exception.GetType().Name
  })
  exit 1
} finally {
  if ($null -ne $SecureValue) {
    $SecureValue.Dispose()
  }
}
exit 0
