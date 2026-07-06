param(
  [Parameter(Mandatory = $true)]
  [string]$User,
  [string]$HostName = "100.85.169.63",
  [int]$Port = 1235,
  [string]$RemoteWorkspace = "/hpc2hdd/home/aimslab/research_agent_workstation",
  [string]$SocksHost = "127.0.0.1",
  [int]$SocksPort = 7890
)

$ErrorActionPreference = "Stop"
$password = ""
if ([Environment]::UserInteractive) {
  $secureInput = Read-Host "HPC SSH password" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureInput)
  try {
    $password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
  } finally {
    if ($bstr -ne [IntPtr]::Zero) {
      [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    $secureInput = $null
  }
} else {
  $password = [Console]::In.ReadLine()
}
if ([string]::IsNullOrWhiteSpace($password)) {
  throw "Password was not provided on stdin."
}

try {
  & (Join-Path $PSScriptRoot "manage_hpc_ssh_secret.ps1") install-credential `
    -User $User `
    -Password $password `
    -HostName $HostName `
    -Port $Port `
    -RemoteWorkspace $RemoteWorkspace `
    -SocksHost $SocksHost `
    -SocksPort $SocksPort
} finally {
  $password = $null
}
