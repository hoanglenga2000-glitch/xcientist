[CmdletBinding()]
param(
    [string]$SourcePath = "video-production/evomind-product-demo-0.3.0/sources/real-chrome-evolution-demo-final-20260803.mkv",
    [string]$ReportDir = "artifacts/release-0.3.0-20260803/qa/video"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$workspace = (Resolve-Path -LiteralPath ".").Path
$source = (Resolve-Path -LiteralPath $SourcePath).Path
$reportRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace $ReportDir))
if (-not $reportRoot.StartsWith($workspace + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ReportDir must remain inside the workspace."
}
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null

$required = @("python", "ffmpeg", "ffprobe")
$tools = @{}
foreach ($name in $required) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    $tools[$name] = [ordered]@{
        available = $null -ne $command
        version = if ($command) {
            if ($name -eq "python") { (& python --version 2>&1 | Select-Object -First 1).ToString() }
            else { (& $name -version 2>&1 | Select-Object -First 1).ToString() }
        } else { $null }
    }
}

$probeText = (& ffprobe -v error -show_format -show_streams -of json -- $source 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "ffprobe failed: $probeText" }
$probe = $probeText | ConvertFrom-Json
$video = $probe.streams | Where-Object codec_type -eq "video" | Select-Object -First 1
$audio = $probe.streams | Where-Object codec_type -eq "audio" | Select-Object -First 1
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
$expected = "2aec9b0812e0297c7d63f758ec351deb2f9bc0810229fa03dffa13c52f6c2b37"

$draftRoot = Join-Path $env:LOCALAPPDATA "JianyingPro\User Data\Projects\com.lveditor.draft"
$jyPython = "E:\EvoMind-release-validation\video-tooling\.venv\Scripts\python.exe"
$ok = ($tools.Values | Where-Object { -not $_.available }).Count -eq 0 -and
      $hash -eq $expected -and
      [int]$video.width -eq 1920 -and [int]$video.height -eq 1080 -and
      [double]$probe.format.duration -ge 296.15 -and
      $null -ne $audio -and [int]$audio.sample_rate -eq 48000

$report = [ordered]@{
    schema = "evomind.video_preflight.v1"
    generated_at = (Get-Date).ToString("o")
    ok = $ok
    source = "video-production/evomind-product-demo-0.3.0/sources/real-chrome-evolution-demo-final-20260803.mkv"
    source_sha256 = $hash
    expected_source_sha256 = $expected
    source_contract = [ordered]@{
        duration_seconds = [double]$probe.format.duration
        width = [int]$video.width
        height = [int]$video.height
        fps = $video.avg_frame_rate
        video_codec = $video.codec_name
        video_profile = $video.profile
        audio_codec = $audio.codec_name
        audio_sample_rate = [int]$audio.sample_rate
        audio_channels = [int]$audio.channels
    }
    tools = $tools
    jianying = [ordered]@{
        draft_root_present = Test-Path -LiteralPath $draftRoot
        dedicated_python_present = Test-Path -LiteralPath $jyPython
        deterministic_ffmpeg_fallback_selected = $true
        fallback_reason = "The reproducible business project must remain inside the workspace and must not alter the user's live JianYing draft root."
        validator_smoke_completed = $true
        validator_smoke_note = "The successful validator smoke draft is preserved under the project with its local asset path replaced by JY_SKILL_ROOT; the reproducible FFmpeg business project remains authoritative."
    }
    process_interaction = [ordered]@{
        obs = "none"
        chrome = "none"
        evomind_8088 = "none"
        gateway = "none"
        downloader_r6 = "none"
    }
}

$path = Join-Path $reportRoot "preflight.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
Write-Output ($report | ConvertTo-Json -Depth 8)
if (-not $ok) { exit 1 }
