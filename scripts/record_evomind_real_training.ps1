[CmdletBinding()]
param(
    [ValidateSet("start", "stop", "status", "controller")]
    [string]$Mode = "status",
    [string]$OutputPath = "video-production/evomind-real-gpu-training-20260808/sources/evomind-real-training-continuous.mkv",
    [string]$StopSignal = "video-production/evomind-real-gpu-training-20260808/runtime/stop-recording.signal",
    [string]$StatePath = "video-production/evomind-real-gpu-training-20260808/runtime/recording-state.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-ProjectPath([string]$Value) {
    $candidate = if ([IO.Path]::IsPathRooted($Value)) { [IO.Path]::GetFullPath($Value) } else { [IO.Path]::GetFullPath((Join-Path $projectRoot $Value)) }
    if (-not $candidate.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Recording path escapes the project root: $candidate"
    }
    return $candidate
}

$output = Resolve-ProjectPath $OutputPath
$stop = Resolve-ProjectPath $StopSignal
$state = Resolve-ProjectPath $StatePath
$runtimeDir = Split-Path -Parent $state
$sourceDir = Split-Path -Parent $output
New-Item -ItemType Directory -Force -Path $runtimeDir, $sourceDir | Out-Null

if ($Mode -eq "start") {
    if (Test-Path -LiteralPath $state) {
        $existing = Get-Content -LiteralPath $state -Raw | ConvertFrom-Json
        if ($existing.controller_pid -and (Get-Process -Id $existing.controller_pid -ErrorAction SilentlyContinue)) {
            throw "A recording controller is already running: PID $($existing.controller_pid)"
        }
    }
    Remove-Item -LiteralPath $stop -Force -ErrorAction SilentlyContinue
    $pwsh = (Get-Command pwsh.exe -ErrorAction Stop).Source
    $arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
        "-Mode", "controller", "-OutputPath", $output,
        "-StopSignal", $stop, "-StatePath", $state
    )
    $controller = Start-Process -FilePath $pwsh -ArgumentList $arguments -WindowStyle Hidden -PassThru
    [ordered]@{
        schema = "evomind.real_screen_recording.state.v1"
        status = "starting"
        controller_pid = $controller.Id
        output_mkv = $output
        output_mp4 = [IO.Path]::ChangeExtension($output, ".mp4")
        stop_signal = $stop
        started_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $state -Encoding utf8
    Get-Content -LiteralPath $state -Raw
    exit 0
}

if ($Mode -eq "stop") {
    if (-not (Test-Path -LiteralPath $state)) { throw "Recording state not found" }
    New-Item -ItemType File -Force -Path $stop | Out-Null
    $deadline = (Get-Date).AddMinutes(4)
    do {
        Start-Sleep -Milliseconds 500
        $current = Get-Content -LiteralPath $state -Raw | ConvertFrom-Json
        if ($current.status -in @("completed", "failed")) { break }
    } while ((Get-Date) -lt $deadline)
    Get-Content -LiteralPath $state -Raw
    if ($current.status -ne "completed") { exit 1 }
    exit 0
}

if ($Mode -eq "status") {
    if (Test-Path -LiteralPath $state) { Get-Content -LiteralPath $state -Raw } else { '{"status":"not_started"}' }
    exit 0
}

$ffmpeg = (Get-Command ffmpeg.exe -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe.exe -ErrorAction Stop).Source
$mp4 = [IO.Path]::ChangeExtension($output, ".mp4")
$stderrLog = Join-Path $runtimeDir "ffmpeg-recording.stderr.log"
$startedAt = Get-Date

try {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $ffmpeg
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in @(
        "-hide_banner", "-y", "-f", "gdigrab", "-framerate", "30", "-draw_mouse", "1", "-i", "desktop",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", $output
    )) { [void]$info.ArgumentList.Add($argument) }
    $recorder = [Diagnostics.Process]::new()
    $recorder.StartInfo = $info
    if (-not $recorder.Start()) { throw "FFmpeg did not start" }
    $stdoutTask = $recorder.StandardOutput.ReadToEndAsync()
    $stderrTask = $recorder.StandardError.ReadToEndAsync()
    [ordered]@{
        schema = "evomind.real_screen_recording.state.v1"
        status = "recording"
        controller_pid = $PID
        ffmpeg_pid = $recorder.Id
        output_mkv = $output
        output_mp4 = $mp4
        stop_signal = $stop
        started_at = $startedAt.ToUniversalTime().ToString("o")
        capture = "ffmpeg-gdigrab-continuous-desktop"
        fps = 30
        draw_mouse = $true
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $state -Encoding utf8

    while (-not (Test-Path -LiteralPath $stop)) {
        if ($recorder.HasExited) { throw "FFmpeg exited before the stop signal" }
        Start-Sleep -Milliseconds 250
    }
    $recorder.StandardInput.WriteLine("q")
    $recorder.StandardInput.Flush()
    if (-not $recorder.WaitForExit(30000)) { throw "FFmpeg did not stop cleanly after q" }
    [IO.File]::WriteAllText($stderrLog, $stderrTask.Result, [Text.UTF8Encoding]::new($false))
    if ($recorder.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $output)) { throw "FFmpeg capture failed with exit code $($recorder.ExitCode)" }

    & $ffmpeg -hide_banner -y -loglevel error -i $output -map 0:v:0 -c copy -movflags +faststart $mp4
    if ($LASTEXITCODE -ne 0) { throw "MKV to MP4 remux failed" }
    & $ffmpeg -hide_banner -v error -i $mp4 -f null NUL
    $decodeExit = $LASTEXITCODE
    if ($decodeExit -ne 0) { throw "Full MP4 decode failed" }
    $probe = & $ffprobe -v error -show_entries "format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt" -of json $mp4 | ConvertFrom-Json
    $hash = (Get-FileHash -LiteralPath $mp4 -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{
        schema = "evomind.real_screen_recording.state.v1"
        status = "completed"
        controller_pid = $PID
        ffmpeg_pid = $recorder.Id
        output_mkv = $output
        output_mp4 = $mp4
        capture = "ffmpeg-gdigrab-continuous-desktop"
        started_at = $startedAt.ToUniversalTime().ToString("o")
        completed_at = (Get-Date).ToUniversalTime().ToString("o")
        wall_clock_seconds = [Math]::Round(((Get-Date) - $startedAt).TotalSeconds, 3)
        decode_exit_code = $decodeExit
        sha256 = $hash
        ffprobe = $probe
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $state -Encoding utf8
}
catch {
    if ($null -ne $stderrTask -and $stderrTask.IsCompleted) {
        [IO.File]::WriteAllText($stderrLog, $stderrTask.Result, [Text.UTF8Encoding]::new($false))
    }
    [ordered]@{
        schema = "evomind.real_screen_recording.state.v1"
        status = "failed"
        controller_pid = $PID
        output_mkv = $output
        output_mp4 = $mp4
        error = $_.Exception.Message
        failed_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $state -Encoding utf8
    throw
}
