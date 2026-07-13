param(
  [Parameter(Mandatory = $true)]
  [string]$UpstreamHost,
  [Parameter(Mandatory = $true)]
  [ValidateRange(1, 65535)]
  [int]$UpstreamPort,
  [Parameter(Mandatory = $true)]
  [string]$PortalUrl
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Manager = Join-Path $Root "scripts\manage_hpc_proxy_bridge.ps1"
$parsedPortal = $null
if (-not [System.Uri]::TryCreate($PortalUrl, [System.UriKind]::Absolute, [ref]$parsedPortal) -or
    $parsedPortal.Scheme -notin @("http", "https") -or
    -not [string]::IsNullOrEmpty($parsedPortal.UserInfo)) {
  throw "PortalUrl must be an absolute HTTP(S) URL without embedded credentials."
}

powershell -ExecutionPolicy Bypass -File $Manager start `
  -UpstreamHost $UpstreamHost `
  -UpstreamPort $UpstreamPort | Out-Null

$chromeCandidates = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
  "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
  "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
  "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
)
$browser = $chromeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browser) {
  throw "Chrome/Edge was not found."
}

$profile = Join-Path $env:USERPROFILE ".evomind\hpc-browser-profile"
Start-Process -FilePath $browser -ArgumentList @(
  "--proxy-server=socks5://127.0.0.1:7890",
  "--user-data-dir=`"$profile`"",
  "--no-first-run",
  "--new-window",
  $parsedPortal.AbsoluteUri
)
