param(
  [Parameter(Mandatory = $true)]
  [string]$User,
  [Parameter(Mandatory = $true)]
  [string]$HostName,
  [Parameter(Mandatory = $true)]
  [ValidateRange(1, 65535)]
  [int]$Port,
  [Parameter(Mandatory = $true)]
  [string]$RemoteWorkspace,
  [string]$SocksHost = "",
  [ValidateRange(0, 65535)]
  [int]$SocksPort = 0
)

$ErrorActionPreference = "Stop"
if ([Environment]::UserInteractive) {
  $secureInput = Read-Host "HPC SSH password" -AsSecureString
  if ($secureInput.Length -eq 0) {
    throw "Password was not provided."
  }
  try {
    & (Join-Path $PSScriptRoot "manage_hpc_ssh_secret.ps1") install-credential `
      -User $User `
      -SecurePassword $secureInput `
      -HostName $HostName `
      -Port $Port `
      -RemoteWorkspace $RemoteWorkspace `
      -SocksHost $SocksHost `
      -SocksPort $SocksPort
  } finally {
    $secureInput.Dispose()
    $secureInput = $null
  }
} else {
  & (Join-Path $PSScriptRoot "manage_hpc_ssh_secret.ps1") install-credential `
    -User $User `
    -SecretFromStdin `
    -HostName $HostName `
    -Port $Port `
    -RemoteWorkspace $RemoteWorkspace `
    -SocksHost $SocksHost `
    -SocksPort $SocksPort
}
