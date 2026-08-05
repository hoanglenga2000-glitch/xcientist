param(
  [ValidateSet("install-credential", "start", "stop", "status", "test")]
  [string]$Command = "status",
  [string]$ProxyUser = "",
  [string]$ProxyPassword = "",
  [int]$ListenPort = 7890,
  [ValidateSet("upstream", "direct")]
  [string]$RouteMode = "upstream"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "hpc_socks_credential.xml"
$MetadataPath = Join-Path $StateDir "hpc_ssh_metadata.json"
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

function Get-CurrentHpcTarget {
  if (-not (Test-Path -LiteralPath $MetadataPath -PathType Leaf)) {
    throw "HPC SSH metadata is not installed."
  }
  $metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
  $hostName = [string]$metadata.host
  $targetPort = [int]$metadata.port
  if ([string]::IsNullOrWhiteSpace($hostName) -or $targetPort -lt 1 -or $targetPort -gt 65535) {
    throw "HPC SSH metadata does not contain a valid current target."
  }
  [pscustomobject]@{ host = $hostName; port = $targetPort }
}

function Test-PortListening([int]$Port) {
  $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
  $listener = $listeners | Where-Object {
    $_.Port -eq $Port -and [Net.IPAddress]::IsLoopback($_.Address)
  } | Select-Object -First 1
  if ($listener) {
    $ownerPid = $null
    if (Test-Path $PidPath) {
      $rawPid = (Get-Content -LiteralPath $PidPath -Raw -ErrorAction SilentlyContinue).Trim()
      $parsedPid = 0
      if ([int]::TryParse($rawPid, [ref]$parsedPid)) {
        $ownerPid = $parsedPid
      }
    }
    $proc = if ($ownerPid) { Get-Process -Id $ownerPid -ErrorAction SilentlyContinue } else { $null }
    return [pscustomobject]@{ listening = $true; pid = $ownerPid; process = $proc.ProcessName }
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
  $credential = if ($RouteMode -eq "upstream") { Import-HpcCredential } else { $null }
  $target = if ($RouteMode -eq "direct") { Get-CurrentHpcTarget } else { $null }
  $python = Get-PythonExe
  $oldProxyUser = [Environment]::GetEnvironmentVariable("HPC_SOCKS_USER", "Process")
  $oldProxyPassword = [Environment]::GetEnvironmentVariable("HPC_SOCKS_PASSWORD", "Process")
  try {
    if ($RouteMode -eq "upstream") {
      $env:HPC_SOCKS_USER = $credential.UserName
      $env:HPC_SOCKS_PASSWORD = $credential.GetNetworkCredential().Password
    } else {
      Remove-Item Env:\HPC_SOCKS_USER -ErrorAction SilentlyContinue
      Remove-Item Env:\HPC_SOCKS_PASSWORD -ErrorAction SilentlyContinue
    }
    Set-Content -LiteralPath $OutLog -Value "" -Encoding UTF8
    Set-Content -LiteralPath $ErrLog -Value "" -Encoding UTF8
    $bridgeArguments = @(
      "`"$BridgeScript`"",
      "--listen-port", [string]$ListenPort
    )
    if ($RouteMode -eq "direct") {
      $bridgeArguments += @("--disable-upstream", "--direct-destination", "$($target.host):$($target.port)")
    }
    $process = Start-Process `
      -FilePath $python `
      -ArgumentList $bridgeArguments `
      -WorkingDirectory $Root `
      -WindowStyle Hidden `
      -RedirectStandardOutput $OutLog `
      -RedirectStandardError $ErrLog `
      -PassThru
  } finally {
    if ($null -eq $oldProxyUser) { Remove-Item Env:\HPC_SOCKS_USER -ErrorAction SilentlyContinue } else { $env:HPC_SOCKS_USER = $oldProxyUser }
    if ($null -eq $oldProxyPassword) { Remove-Item Env:\HPC_SOCKS_PASSWORD -ErrorAction SilentlyContinue } else { $env:HPC_SOCKS_PASSWORD = $oldProxyPassword }
  }
  Set-Content -Path $PidPath -Value $process.Id -Encoding ASCII
  Start-Sleep -Milliseconds 900
  $status = Test-PortListening $ListenPort
  Write-Output (@{ status = $(if ($status.listening) { "started" } else { "failed" }); listen_port = $ListenPort; route_mode = $RouteMode; pid = $process.Id; process = $status.process; log = $OutLog } | ConvertTo-Json -Depth 4)
  exit $(if ($status.listening) { 0 } else { 1 })
}

if ($Command -eq "test") {
  $status = Test-PortListening $ListenPort
  if (-not $status.listening) {
    throw "HPC SOCKS bridge is not listening on 127.0.0.1:$ListenPort."
  }
  $target = Get-CurrentHpcTarget
  $python = Get-PythonExe
  $test = & $python (Join-Path $Root "scripts\verify_hpc_socks_gateway.py") --proxy-host 127.0.0.1 --proxy-port $ListenPort --dest-host $target.host --dest-port $target.port
  $testExitCode = $LASTEXITCODE
  Write-Output $test
  exit $testExitCode
}

$status = Test-PortListening $ListenPort
Write-Output (@{ status = $(if ($status.listening) { "running" } else { "not_running" }); listen_port = $ListenPort; pid = $status.pid; process = $status.process; credential_installed = (Test-Path $CredentialPath) } | ConvertTo-Json -Depth 4)
