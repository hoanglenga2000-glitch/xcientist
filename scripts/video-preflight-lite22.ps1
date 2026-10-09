[CmdletBinding()]
param(
  [string]$ReportPath = "video-production/evomind-lite22-training-20260808/qa/preflight.json"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$workspace = (Resolve-Path -LiteralPath ".").Path
$target = [System.IO.Path]::GetFullPath((Join-Path $workspace $ReportPath))
if (-not $target.StartsWith($workspace + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "ReportPath must remain inside the workspace."
}
$targetDirectory = Split-Path -Parent $target
New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null

$obs = "C:\Program Files\obs-studio\bin\64bit\obs64.exe"
$obsWebSocket = "C:\Program Files\obs-studio\obs-plugins\64bit\obs-websocket.dll"
$jianying = Get-ChildItem (Join-Path $env:LOCALAPPDATA "JianyingPro\Apps") -Filter JianyingPro.exe -Recurse -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$draftRoot = Join-Path $env:LOCALAPPDATA "JianyingPro\User Data\Projects\com.lveditor.draft"
$videoPython = "E:\EvoMind-release-validation\video-tooling\.venv\Scripts\python.exe"

Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class EvoMindDpiNative {
  [DllImport("user32.dll")]
  public static extern bool SetProcessDPIAware();
}
'@
[EvoMindDpiNative]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName System.Windows.Forms
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
$pymediainfo = $null
if (Test-Path -LiteralPath $videoPython -PathType Leaf) {
  $pymediainfo = (& $videoPython -c "import pymediainfo; print(pymediainfo.__version__)" 2>$null | Select-Object -First 1)
}

$health = $null
try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:8088/api/healthz" -TimeoutSec 10
} catch {
  $health = $null
}

$checks = [ordered]@{
  obs_present = Test-Path -LiteralPath $obs -PathType Leaf
  obs_signature_valid = (Test-Path -LiteralPath $obs -PathType Leaf) -and ((Get-AuthenticodeSignature $obs).Status -eq "Valid")
  obs_websocket_present = Test-Path -LiteralPath $obsWebSocket -PathType Leaf
  jianying_present = $null -ne $jianying
  jianying_signature_valid = ($null -ne $jianying) -and ((Get-AuthenticodeSignature $jianying.FullName).Status -eq "Valid")
  jianying_draft_root_present = Test-Path -LiteralPath $draftRoot -PathType Container
  ffmpeg_present = $null -ne $ffmpeg
  ffprobe_present = $null -ne $ffprobe
  video_python_present = Test-Path -LiteralPath $videoPython -PathType Leaf
  pymediainfo_present = -not [string]::IsNullOrWhiteSpace([string]$pymediainfo)
  screen_at_least_1080p = $bounds.Width -ge 1920 -and $bounds.Height -ge 1080
  dashboard_healthy = ($null -ne $health) -and $health.status -eq "ready"
}

$failed = @($checks.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object Key)
$report = [ordered]@{
  schema = "evomind.lite22_video.preflight.v1"
  generated_at = (Get-Date).ToString("o")
  status = if ($failed.Count -eq 0) { "passed" } else { "failed" }
  failed_checks = $failed
  checks = $checks
  environment = [ordered]@{
    obs = $obs
    obs_websocket = $obsWebSocket
    jianying = if ($jianying) { $jianying.FullName } else { $null }
    draft_root = $draftRoot
    video_python = $videoPython
    pymediainfo = $pymediainfo
    screen = "{0}x{1}" -f $bounds.Width, $bounds.Height
    dashboard_status = if ($health) { $health.status } else { $null }
    dashboard_version = if ($health) { $health.version } else { $null }
  }
  recording_contract = [ordered]@{
    resolution = "1920x1080"
    fps = 30
    real_product_footage_minimum = 0.70
    secrets_visible = $false
  }
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $target -Encoding UTF8
$report | ConvertTo-Json -Depth 8
if ($failed.Count -ne 0) { exit 1 }
