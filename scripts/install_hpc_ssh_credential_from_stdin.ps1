param(
  [Parameter(Mandatory = $true)]
  [string]$User,
  [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')]
  [string]$Profile = "default",
  [string]$HostName = "",
  [int]$Port = 0,
  [string]$RemoteWorkspace = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra",
  [string]$SocksHost = "127.0.0.1",
  [int]$SocksPort = 7890,
  [string]$JumpHost = "",
  [int]$JumpPort = 22,
  [string]$JumpUser = "",
  [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$')]
  [string]$AllocationBindingId = "",
  [int]$AllocationGeneration = 0,
  [switch]$FromStdin
)

$ErrorActionPreference = "Stop"
if (-not $HostName -or $Port -le 0) {
  throw "HostName and Port are required; HPC allocation metadata has no unsafe default."
}
if ($Profile -in @('.', '..')) {
  throw "Profile must be one safe path segment."
}
$JobId = 0
$IsJobProfile = $false
if ($Profile -match '^job([0-9]+)$') {
  $JobId = [int]$Matches[1]
  $IsJobProfile = $true
}
if ($Profile -ne "default" -and -not $IsJobProfile) {
  throw "Named HPC profiles must use job<job_id>."
}
if ($IsJobProfile) {
  if ([string]::IsNullOrWhiteSpace($AllocationBindingId)) {
    throw "AllocationBindingId is required for a named job profile."
  }
  if ($AllocationGeneration -le 0) {
    throw "AllocationGeneration must be a positive integer for a named job profile."
  }
}
if ($RemoteWorkspace.TrimEnd('/') -ne "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra") {
  throw "RemoteWorkspace must be the dedicated EvoMind HPC root."
}
$secureInput = $null
$plainInput = $null
if ($FromStdin) {
  $plainInput = [Console]::In.ReadLine()
  if ([string]::IsNullOrWhiteSpace($plainInput)) {
    throw "Password was not provided on stdin."
  }
  $secureInput = ConvertTo-SecureString $plainInput -AsPlainText -Force
  $plainInput = $null
} else {
  $secureInput = Read-Host "HPC SSH password" -AsSecureString
}
if ($null -eq $secureInput -or $secureInput.Length -le 0) {
  throw "Password was not provided."
}

try {
  if ([string]::IsNullOrWhiteSpace($env:APPDATA) -or -not [System.IO.Path]::IsPathRooted($env:APPDATA)) {
    throw "APPDATA must be an absolute path."
  }
  $stateRoot = Join-Path $env:APPDATA "ResearchAgentWorkstation"
  $stateDir = if ($Profile -eq "default") {
    $stateRoot
  } else {
    Join-Path (Join-Path $stateRoot "profiles") $Profile
  }
  $credentialPath = Join-Path $stateDir "hpc_ssh_credential.xml"
  $metadataPath = Join-Path $stateDir "hpc_ssh_metadata.json"
  if ($IsJobProfile -and (Test-Path -LiteralPath $stateDir)) {
    throw "Named job profile already exists; freeze/retire it and securely enroll a new allocation generation instead of overwriting it."
  }
  $profileInstanceId = [guid]::NewGuid().ToString("D")
  $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  $metadata = [ordered]@{
    schema = $(if ($IsJobProfile) { "evomind.hpc.dpapi_profile.v2" } else { "evomind.hpc.dpapi_profile.legacy.v1" })
    credential_profile = $Profile
    job_id = $JobId
    host = $HostName
    port = $Port
    remote_workspace = "/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra"
    socks_host = $SocksHost
    socks_port = $SocksPort
    jump_host = $JumpHost
    jump_port = $JumpPort
    jump_user = $(if ($JumpHost -and $JumpUser) { $JumpUser } elseif ($JumpHost) { $User } else { "" })
    known_hosts_path = "known_hosts"
    updated_at = $timestamp
  }
  if ($IsJobProfile) {
    $metadata["profile_state"] = "provisioning"
    $metadata["profile_state_reason"] = "awaiting_identity_bootstrap"
    $metadata["allocation_binding_id"] = $AllocationBindingId
    $metadata["allocation_generation"] = $AllocationGeneration
    $metadata["profile_instance_id"] = $profileInstanceId
    $metadata["lifecycle_revision"] = 1
    $metadata["created_at"] = $timestamp
    $metadata["state_changed_at"] = $timestamp
  }
  $metadataJson = $metadata | ConvertTo-Json -Depth 6
  if ($IsJobProfile) {
    $profilesRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $stateDir))
    New-Item -ItemType Directory -Path $profilesRoot -Force | Out-Null
    $stagingDir = [System.IO.Path]::GetFullPath(
      (Join-Path $profilesRoot ("." + $Profile + "." + [guid]::NewGuid().ToString("N") + ".tmp"))
    )
    $profilesPrefix = $profilesRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $stagingDir.StartsWith($profilesPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
      throw "Profile staging directory escaped the managed profiles root."
    }
    try {
      New-Item -ItemType Directory -Path $stagingDir | Out-Null
      $stagedCredentialPath = Join-Path $stagingDir "hpc_ssh_credential.xml"
      $stagedMetadataPath = Join-Path $stagingDir "hpc_ssh_metadata.json"
      [pscredential]::new($User, $secureInput) | Export-Clixml -LiteralPath $stagedCredentialPath
      Set-Content -LiteralPath $stagedMetadataPath -Value $metadataJson -Encoding UTF8
      Move-Item -LiteralPath $stagingDir -Destination $stateDir
    } finally {
      if (
        $null -ne $stagingDir -and
        $stagingDir.StartsWith($profilesPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $stagingDir)
      ) {
        Remove-Item -LiteralPath $stagingDir -Recurse -Force
      }
    }
  } else {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    [pscredential]::new($User, $secureInput) | Export-Clixml -LiteralPath $credentialPath
    Set-Content -LiteralPath $metadataPath -Value $metadataJson -Encoding UTF8
  }
  Write-Output (@{
    status = "installed"
    credential_profile = $Profile
    profile_state = $(if ($IsJobProfile) { "provisioning" } else { "legacy" })
    allocation_generation = $(if ($IsJobProfile) { $AllocationGeneration } else { 0 })
    profile_instance_id = $(if ($IsJobProfile) { $profileInstanceId } else { "" })
    credential_installed = (Test-Path -LiteralPath $credentialPath)
    metadata_installed = (Test-Path -LiteralPath $metadataPath)
    credential_path = $credentialPath
    metadata_path = $metadataPath
  } | ConvertTo-Json -Depth 4)
} finally {
  if ($null -ne $secureInput) {
    $secureInput.Dispose()
  }
  $secureInput = $null
  $plainInput = $null
}
