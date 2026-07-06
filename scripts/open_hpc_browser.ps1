$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Manager = Join-Path $Root "scripts\manage_hpc_proxy_bridge.ps1"

powershell -ExecutionPolicy Bypass -File $Manager start | Out-Null

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

$profile = Join-Path $env:USERPROFILE "hkust-hpc-chrome-data"
$url = "http://100.85.169.63:1234/appform/login"
Start-Process -FilePath $browser -ArgumentList @(
  "--proxy-server=socks5://127.0.0.1:7890",
  "--user-data-dir=`"$profile`"",
  "--no-first-run",
  "--new-window",
  $url
)
