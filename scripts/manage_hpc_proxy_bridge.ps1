param(
  [ValidateSet("install-credential", "start", "stop", "status", "test")]
  [string]$Command = "status",
  [string]$ProxyUser = "",
  [string]$ProxyPassword = "",
  [int]$ListenPort = 7890
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "hpc_socks_credential.xml"
$PidPath = Join-Path $StateDir "hpc_socks_bridge.pid"
$OutLog = Join-Path $StateDir "hpc_socks_bridge.out.log"
$ErrLog = Join-Path $StateDir "hpc_socks_bridge.err.log"
$BridgeScript = Join-Path $Root "scripts\hpc_socks_bridge.py"

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

function Import-HpcCredential {
  if (-not (Test-Path $CredentialPath)) {
    throw "HPC SOCKS credential is not installed. Run: powershell -ExecutionPolicy Bypass -File scripts\manage_hpc_proxy_bridge.ps1 install-credential -ProxyUser <user> -ProxyPassword <password>"
  }
  Import-Clixml -Path $CredentialPath
}

function Test-PortListening([int]$Port) {
  $conn = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn) {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    return [pscustomobject]@{ listening = $true; pid = $conn.OwningProcess; process = $proc.ProcessName }
  }
  [pscustomobject]@{ listening = $false; pid = $null; process = $null }
}

function Stop-Bridge {
  $status = Test-PortListening $ListenPort
  if ($status.listening -and $status.process -match "python") {
    Stop-Process -Id $status.pid -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path $PidPath) {
    $pidValue = Get-Content $PidPath -ErrorAction SilentlyContinue
    if ($pidValue) {
      Stop-Process -Id ([int]$pidValue) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidPath -Force -ErrorAction SilentlyContinue
  }
}

New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

if ($Command -eq "install-credential") {
  if (-not $ProxyUser -or -not $ProxyPassword) {
    throw "ProxyUser and ProxyPassword are required for install-credential."
  }
  $secure = ConvertTo-SecureString $ProxyPassword -AsPlainText -Force
  $credential = [pscredential]::new($ProxyUser, $secure)
  $credential | Export-Clixml -Path $CredentialPath
  Write-Output (@{ status = "installed"; credential_path = $CredentialPath } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "stop") {
  Stop-Bridge
  Write-Output (@{ status = "stopped"; listen_port = $ListenPort } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "start") {
  $current = Test-PortListening $ListenPort
  if ($current.listening) {
    Write-Output (@{ status = "already_running"; listen_port = $ListenPort; pid = $current.pid; process = $current.process } | ConvertTo-Json -Depth 4)
    exit 0
  }
  $credential = Import-HpcCredential
  $envBlock = [System.Environment]::GetEnvironmentVariables()
  $envBlock["HPC_SOCKS_USER"] = $credential.UserName
  $envBlock["HPC_SOCKS_PASSWORD"] = $credential.GetNetworkCredential().Password
  $python = Get-PythonExe
  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = $python
  $psi.Arguments = "`"$BridgeScript`" --listen-port $ListenPort"
  $psi.WorkingDirectory = $Root
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  foreach ($key in $envBlock.Keys) {
    $psi.Environment[$key] = [string]$envBlock[$key]
  }
  $process = [System.Diagnostics.Process]::Start($psi)
  Set-Content -Path $PidPath -Value $process.Id -Encoding ASCII
  Start-Sleep -Milliseconds 900
  $out = $process.StandardOutput.ReadLine()
  if ($out) { Add-Content -Path $OutLog -Value $out -Encoding UTF8 }
  $status = Test-PortListening $ListenPort
  Write-Output (@{ status = $(if ($status.listening) { "started" } else { "failed" }); listen_port = $ListenPort; pid = $process.Id; process = $status.process; log = $OutLog } | ConvertTo-Json -Depth 4)
  exit $(if ($status.listening) { 0 } else { 1 })
}

if ($Command -eq "test") {
  $status = Test-PortListening $ListenPort
  if (-not $status.listening) {
    throw "HPC SOCKS bridge is not listening on 127.0.0.1:$ListenPort."
  }
  $python = Get-PythonExe
  $test = & $python (Join-Path $Root "scripts\verify_hpc_socks_gateway.py") --proxy-host 127.0.0.1 --proxy-port $ListenPort --dest-host 100.85.169.63 --dest-port 1235
  Write-Output $test
  exit 0
}

$status = Test-PortListening $ListenPort
Write-Output (@{ status = $(if ($status.listening) { "running" } else { "not_running" }); listen_port = $ListenPort; pid = $status.pid; process = $status.process; credential_installed = (Test-Path $CredentialPath) } | ConvertTo-Json -Depth 4)
