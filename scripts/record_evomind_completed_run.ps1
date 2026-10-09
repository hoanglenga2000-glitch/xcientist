[CmdletBinding()]
param(
    [string]$RunId = "evomind_dec2021_demo_20260808_121338_f00001",
    [string]$OutputPath = "video-production/evomind-real-gpu-training-20260808/sources/evomind-job91051-dec2021-v7-completed-clean.mkv",
    [int]$DebugPort = 9338,
    [switch]$ResetProfile
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$output = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputPath))
$mp4 = [IO.Path]::ChangeExtension($output, ".mp4")
$runtimeDir = Join-Path $projectRoot "video-production/evomind-real-gpu-training-20260808/runtime"
$profileDir = Join-Path $runtimeDir "chrome-clean-recording-profile"
$statePath = Join-Path $runtimeDir "job91051-v7-completed-recording-state.json"
$stderrPath = Join-Path $runtimeDir "job91051-v7-completed-ffmpeg.stderr.log"
$chromeLog = Join-Path $runtimeDir "job91051-v7-completed-chrome.log"

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class EvoMindRecordingWindow {
    [DllImport("user32.dll")]
    public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int X, int Y);
    [DllImport("user32.dll")]
    public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extraInfo);
    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int index);
}
'@

New-Item -ItemType Directory -Force -Path (Split-Path $output -Parent), $runtimeDir | Out-Null
$existingProfileProcesses = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$profileDir*" }
foreach ($item in $existingProfileProcesses) {
    Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($existingProfileProcesses) { Start-Sleep -Seconds 2 }
if ($ResetProfile -and (Test-Path -LiteralPath $profileDir)) {
    $resolvedProfile = (Resolve-Path -LiteralPath $profileDir).Path
    $resolvedRuntime = (Resolve-Path -LiteralPath $runtimeDir).Path
    if (-not $resolvedProfile.StartsWith($resolvedRuntime + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clear Chrome profile outside the video runtime directory"
    }
    Remove-Item -LiteralPath $resolvedProfile -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$preferences = Join-Path $profileDir "Default/Preferences"
New-Item -ItemType Directory -Force -Path (Split-Path $preferences -Parent) | Out-Null
$python = Join-Path $projectRoot ".venv/Scripts/python.exe"
$preferenceScript = @'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
d.setdefault("translate", {})["enabled"] = False
d.setdefault("profile", {})["exit_type"] = "Normal"
d["exited_cleanly"] = True
p.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
'@
$preferenceScript | & $python - $preferences
if ($LASTEXITCODE -ne 0) { throw "Failed to normalize the clean Chrome recording profile" }

$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$ffmpeg = (Get-Command ffmpeg.exe -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe.exe -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $chrome)) { throw "Chrome executable not found" }

function Wait-DevTools {
    param([int]$Port)
    $deadline = (Get-Date).AddSeconds(25)
    do {
        try {
            return Invoke-RestMethod "http://127.0.0.1:$Port/json/version" -TimeoutSec 2
        } catch {
            Start-Sleep -Milliseconds 500
        }
    } while ((Get-Date) -lt $deadline)
    throw "Chrome DevTools endpoint did not become ready"
}

function Get-PageTarget {
    param([int]$Port)
    $deadline = (Get-Date).AddSeconds(20)
    do {
        $response = Invoke-RestMethod "http://127.0.0.1:$Port/json" -TimeoutSec 3
        $targets = @($response | ForEach-Object { $_ })
        $target = $targets | Where-Object { $_.type -eq "page" -and $_.url -like "http://127.0.0.1:8088/*" } | Select-Object -First 1
        if ($target) { return $target }
        Start-Sleep -Milliseconds 400
    } while ((Get-Date) -lt $deadline)
    throw "EvoMind Chrome page target was not found"
}

function Send-Cdp {
    param(
        [Net.WebSockets.ClientWebSocket]$Socket,
        [int]$Id,
        [string]$Method,
        [hashtable]$Params = @{}
    )
    $payload = @{ id = $Id; method = $Method; params = $Params } | ConvertTo-Json -Depth 20 -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
    $segment = [ArraySegment[byte]]::new($bytes)
    $Socket.SendAsync($segment, [Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    $buffer = New-Object byte[] 65536
    do {
        $stream = [IO.MemoryStream]::new()
        do {
            $chunk = [ArraySegment[byte]]::new($buffer)
            $result = $Socket.ReceiveAsync($chunk, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
            if ($result.Count -gt 0) { $stream.Write($buffer, 0, $result.Count) }
        } while (-not $result.EndOfMessage)
        $text = [Text.Encoding]::UTF8.GetString($stream.ToArray())
        $message = $text | ConvertFrom-Json
    } while ($message.id -ne $Id)
    if ($message.error) { throw "CDP $Method failed: $($message.error.message)" }
    return $message.result
}

function Evaluate-Cdp {
    param(
        [Net.WebSockets.ClientWebSocket]$Socket,
        [int]$Id,
        [string]$Expression,
        [switch]$AwaitPromise
    )
    $params = @{ expression = $Expression; returnByValue = $true }
    if ($AwaitPromise) { $params.awaitPromise = $true }
    $response = Send-Cdp -Socket $Socket -Id $Id -Method "Runtime.evaluate" -Params $params
    if ($response.exceptionDetails) {
        $description = [string]$response.exceptionDetails.exception.description
        if (-not $description) { $description = [string]$response.exceptionDetails.text }
        throw "Browser evaluation failed at CDP id $Id`: $description"
    }
    return $response.result.value
}

function Write-State([string]$Status, [hashtable]$Extra = @{}) {
    $payload = [ordered]@{
        schema = "evomind.real_screen_recording.state.v1"
        status = $Status
        run_id = $RunId
        output_mkv = $output
        output_mp4 = $mp4
        capture = "ffmpeg-gdigrab-continuous-desktop"
        chrome_profile = $profileDir
        created_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    foreach ($entry in $Extra.GetEnumerator()) { $payload[$entry.Key] = $entry.Value }
    $payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $statePath -Encoding utf8
}

$chromeProcess = $null
$recorder = $null
$socket = $null
$startedAt = Get-Date
try {
    Write-State "starting"
    $assistantUrl = "http://127.0.0.1:8088/?page=assistant&task=tabular-playground-series-dec-2021"
    $bootstrapFile = Join-Path $projectRoot "web/research-agent-workstation/.runtime-logs/dashboard.bootstrap.once"
    $launchUrl = $assistantUrl
    if (Test-Path -LiteralPath $bootstrapFile) {
        $candidate = (Get-Content -LiteralPath $bootstrapFile -Raw).Trim()
        $parsed = [Uri]$candidate
        if ($parsed.Scheme -ne "http" -or $parsed.Host -ne "127.0.0.1" -or $parsed.Port -ne 8088 -or $parsed.Fragment -notlike "#bootstrap=*") {
            throw "Dashboard bootstrap URL failed the loopback-only recording contract"
        }
        $launchUrl = $candidate
    }
    $chromeArgs = @(
        "--user-data-dir=$profileDir",
        "--remote-debugging-port=$DebugPort",
        "--remote-allow-origins=*",
        "--disable-infobars",
        "--hide-crash-restore-bubble",
        "--disable-extensions",
        "--disable-translate",
        "--disable-features=Translate,TranslateUI",
        "--disable-session-crashed-bubble",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
        $launchUrl
    )
    $chromeProcess = Start-Process -FilePath $chrome -ArgumentList $chromeArgs -PassThru
    [void](Wait-DevTools -Port $DebugPort)
    $target = @(Get-PageTarget -Port $DebugPort)[0]
    $socket = [Net.WebSockets.ClientWebSocket]::new()
    $webSocketUrl = [string]$target.webSocketDebuggerUrl
    $socket.ConnectAsync([Uri]$webSocketUrl, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    [void](Send-Cdp -Socket $socket -Id 1 -Method "Runtime.enable")
    [void](Send-Cdp -Socket $socket -Id 2 -Method "Page.enable")
    Start-Sleep -Seconds 3
    $sessionReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $sessionStatus = Evaluate-Cdp -Socket $socket -Id (300 + $attempt) -Expression "fetch('/api/session/status',{cache:'no-store'}).then(r=>r.status)" -AwaitPromise
            if ([int]$sessionStatus -eq 200) { $sessionReady = $true; break }
        } catch {
            # The bootstrap component may still be replacing the page context.
        }
        Start-Sleep -Seconds 1
    }
    if (-not $sessionReady) {
        $sessionDebug = $null
        try {
            $sessionDebug = Evaluate-Cdp -Socket $socket -Id 399 -Expression "({href:location.origin+location.pathname+location.search,has_bootstrap:location.hash.startsWith('#bootstrap='),title:document.title,body:(document.body?.innerText||'').slice(0,120)})"
        } catch {}
        throw "The clean Chrome profile did not establish the loopback session: $($sessionDebug | ConvertTo-Json -Compress)"
    }
    [void](Send-Cdp -Socket $socket -Id 3 -Method "Page.navigate" -Params @{ url = $assistantUrl })
    Start-Sleep -Seconds 5
    $expression = "localStorage.setItem('evomind_assistant_training_run_v1', '$RunId'); true"
    [void](Evaluate-Cdp -Socket $socket -Id 4 -Expression $expression)
    [void](Send-Cdp -Socket $socket -Id 41 -Method "Page.reload" -Params @{ ignoreCache = $true })
    Start-Sleep -Seconds 8
    $visibleChrome = $null
    $windowDeadline = (Get-Date).AddSeconds(8)
    do {
        $profileProcess = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*$profileDir*" } |
            Select-Object -First 1
        if ($profileProcess) {
            $candidateProcess = Get-Process -Id $profileProcess.ProcessId -ErrorAction SilentlyContinue
            if ($candidateProcess -and $candidateProcess.MainWindowHandle -ne 0) { $visibleChrome = $candidateProcess }
        }
        if (-not $visibleChrome) { Start-Sleep -Milliseconds 300 }
    } while (-not $visibleChrome -and (Get-Date) -lt $windowDeadline)
    if (-not $visibleChrome) { throw "The clean Chrome recording window handle was not found" }
    [void][EvoMindRecordingWindow]::ShowWindowAsync($visibleChrome.MainWindowHandle, 3)
    [void][EvoMindRecordingWindow]::SetForegroundWindow($visibleChrome.MainWindowHandle)
    Start-Sleep -Seconds 1
    $preflightExpression = "(async()=>{const r=await fetch('/api/assistant/runs/$RunId',{cache:'no-store'}); const j=await r.json().catch(()=>({})); return {href:location.href,title:document.title,http:r.status,run:j?.run?.run_id||'',status:j?.run?.status||'',card:document.querySelector('[data-ui-training-run-card]')?.innerText||''};})()"
    $preflight = Evaluate-Cdp -Socket $socket -Id 5 -Expression $preflightExpression -AwaitPromise
    if ($preflight.http -ne 200 -or $preflight.run -ne $RunId -or $preflight.status -ne "completed" -or $preflight.card -notlike "*$RunId*") {
        $safePreflight = @{ href = $preflight.href; title = $preflight.title; http = $preflight.http; run = $preflight.run; status = $preflight.status; card_present = [bool]$preflight.card }
        throw "Completed Run did not pass the authenticated Chrome preflight: $($safePreflight | ConvertTo-Json -Compress)"
    }

    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $ffmpeg
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in @(
        "-hide_banner", "-y", "-f", "gdigrab", "-framerate", "30", "-draw_mouse", "1", "-i", "desktop",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", $output
    )) { [void]$info.ArgumentList.Add($argument) }
    $recorder = [Diagnostics.Process]::new()
    $recorder.StartInfo = $info
    if (-not $recorder.Start()) { throw "FFmpeg did not start" }
    $stdoutTask = $recorder.StandardOutput.ReadToEndAsync()
    $stderrTask = $recorder.StandardError.ReadToEndAsync()
    Write-State "recording" @{ ffmpeg_pid = $recorder.Id; chrome_pid = $chromeProcess.Id; started_at = $startedAt.ToUniversalTime().ToString("o") }

    # Completed run card: top, card, logs and artifact links.
    Start-Sleep -Seconds 8
    [void](Evaluate-Cdp -Socket $socket -Id 6 -Expression "window.scrollTo({top: document.body.scrollHeight * 0.45, behavior: 'smooth'}); true")
    Start-Sleep -Seconds 8
    [void](Evaluate-Cdp -Socket $socket -Id 7 -Expression "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'}); true")
    Start-Sleep -Seconds 8

    # Open the same Run's generated report through EvoMind's authenticated artifact route.
    $reportUrl = "http://127.0.0.1:8088/api/artifacts?path=workspace/assistant_runs/$RunId/report.html"
    [void](Send-Cdp -Socket $socket -Id 8 -Method "Page.navigate" -Params @{ url = $reportUrl })
    Start-Sleep -Seconds 9
    $reportPreflight = Evaluate-Cdp -Socket $socket -Id 9 -Expression "({href:location.href,title:document.title,heading:document.querySelector('h1')?.innerText||''})"
    if ($reportPreflight.heading -notlike "*真实 GPU 训练报告*") { throw "Generated Run report did not become visible in Chrome" }
    [void](Evaluate-Cdp -Socket $socket -Id 10 -Expression "window.scrollTo({top: document.body.scrollHeight * 0.42, behavior: 'smooth'}); true")
    Start-Sleep -Seconds 9
    [void](Evaluate-Cdp -Socket $socket -Id 11 -Expression "window.scrollTo({top: document.body.scrollHeight * 0.78, behavior: 'smooth'}); true")
    Start-Sleep -Seconds 9
    [void](Evaluate-Cdp -Socket $socket -Id 12 -Expression "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'}); true")
    Start-Sleep -Seconds 10
    [void](Evaluate-Cdp -Socket $socket -Id 13 -Expression "window.scrollTo({top: 0, behavior: 'smooth'}); true")
    Start-Sleep -Seconds 6

    $recorder.StandardInput.WriteLine("q")
    $recorder.StandardInput.Flush()
    if (-not $recorder.WaitForExit(30000)) { throw "FFmpeg did not stop cleanly after q" }
    [IO.File]::WriteAllText($stderrPath, $stderrTask.Result, [Text.UTF8Encoding]::new($false))
    if ($recorder.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $output)) { throw "FFmpeg capture failed" }
    & $ffmpeg -hide_banner -y -loglevel error -i $output -map 0:v:0 -c copy -movflags +faststart $mp4
    if ($LASTEXITCODE -ne 0) { throw "MKV to MP4 remux failed" }
    & $ffmpeg -hide_banner -v error -i $mp4 -f null NUL
    if ($LASTEXITCODE -ne 0) { throw "Full MP4 decode failed" }
    $probe = & $ffprobe -v error -show_entries "format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt" -of json $mp4 | ConvertFrom-Json
    $hash = (Get-FileHash -LiteralPath $mp4 -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-State "completed" @{
        started_at = $startedAt.ToUniversalTime().ToString("o")
        completed_at = (Get-Date).ToUniversalTime().ToString("o")
        wall_clock_seconds = [Math]::Round(((Get-Date) - $startedAt).TotalSeconds, 3)
        decode_exit_code = 0
        sha256 = $hash
        ffprobe = $probe
    }
}
catch {
    Write-State "failed" @{ error = $_.Exception.Message }
    throw
}
finally {
    if ($recorder -and -not $recorder.HasExited) {
        try { $recorder.StandardInput.WriteLine("q"); $recorder.StandardInput.Flush(); [void]$recorder.WaitForExit(10000) } catch {}
    }
    if ($socket) {
        try { [void](Send-Cdp -Socket $socket -Id 900 -Method "Browser.close") } catch {}
        try { $socket.CloseAsync([Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "done", [Threading.CancellationToken]::None).GetAwaiter().GetResult() } catch {}
        $socket.Dispose()
    }
    if ($chromeProcess -and -not $chromeProcess.HasExited) {
        try { Stop-Process -Id $chromeProcess.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
}

Get-Content -LiteralPath $statePath -Raw
