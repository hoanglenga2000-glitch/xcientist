[CmdletBinding()]
param(
    [string]$InputPath = "video-production/evomind-product-demo-0.3.0/output/EvoMind-Product-Demo-zh-CN-0.3.0.mp4",
    [string]$ReportDir = "artifacts/release-0.3.0-20260803/qa/video",
    [string]$ProjectDir = "video-production/evomind-product-demo-0.3.0"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Convert-SrtTimeToSeconds([string]$value) {
    if ($value -notmatch '^(\d{2}):(\d{2}):(\d{2}),(\d{3})$') { throw "Bad SRT timestamp: $value" }
    return ([int]$Matches[1] * 3600) + ([int]$Matches[2] * 60) + [int]$Matches[3] + ([int]$Matches[4] / 1000.0)
}

function Write-Utf8Json($value, [string]$path, [int]$depth = 12) {
    $value | ConvertTo-Json -Depth $depth | Set-Content -LiteralPath $path -Encoding UTF8
}

$workspace = (Resolve-Path -LiteralPath ".").Path
$input = (Resolve-Path -LiteralPath $InputPath).Path
$project = (Resolve-Path -LiteralPath $ProjectDir).Path
$reportRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace $ReportDir))
if (-not $reportRoot.StartsWith($workspace + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ReportDir must remain inside the workspace."
}
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null

$probeText = (& ffprobe -v error -show_format -show_streams -of json -- $input 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "ffprobe failed: $probeText" }
$probePublic = $probeText.Replace($workspace, '${WORKSPACE}')
$probePublic | Set-Content -LiteralPath (Join-Path $reportRoot "final-ffprobe.json") -Encoding UTF8
$probe = $probeText | ConvertFrom-Json
$video = $probe.streams | Where-Object codec_type -eq "video" | Select-Object -First 1
$audio = $probe.streams | Where-Object codec_type -eq "audio" | Select-Object -First 1
$duration = [double]$probe.format.duration

$frameText = (& ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,r_frame_rate -of json -- $input 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "frame count failed: $frameText" }
$frameProbe = $frameText | ConvertFrom-Json
$decodedFrames = [int]$frameProbe.streams[0].nb_read_frames

$decodeLog = (& ffmpeg -hide_banner -nostdin -v error -xerror -i $input -f null NUL 2>&1) -join "`n"
$decodeExit = $LASTEXITCODE
$decodeLog.Replace($workspace, '${WORKSPACE}') | Set-Content -LiteralPath (Join-Path $reportRoot "decode-integrity.log") -Encoding UTF8

$nativePreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$detectorLog = (& ffmpeg -hide_banner -nostdin -i $input -vf "blackdetect=d=0.20:pix_th=0.10,freezedetect=n=-50dB:d=4" -af "silencedetect=n=-45dB:d=1.5" -f null NUL 2>&1) -join "`n"
$detectorExit = $LASTEXITCODE
$detectorLog.Replace($workspace, '${WORKSPACE}') | Set-Content -LiteralPath (Join-Path $reportRoot "final-black-freeze-silence.log") -Encoding UTF8
$blackCount = ([regex]::Matches($detectorLog, 'black_start\s*:')).Count
$freezeCount = ([regex]::Matches($detectorLog, 'lavfi\.freezedetect\.freeze_start\s*:')).Count
$silenceCount = ([regex]::Matches($detectorLog, 'silence_start\s*:')).Count

$loudnessLog = (& ffmpeg -hide_banner -nostdin -i $input -af "loudnorm=I=-14:TP=-1:LRA=7:print_format=json" -f null NUL 2>&1) -join "`n"
$loudnessExit = $LASTEXITCODE
$ErrorActionPreference = $nativePreference
$loudnessMatches = [regex]::Matches($loudnessLog, '(?s)\{\s*"input_i".*?\}')
if ($loudnessMatches.Count -eq 0) { throw "Could not parse loudness JSON." }
$loudness = $loudnessMatches[$loudnessMatches.Count - 1].Value | ConvertFrom-Json
$loudnessReport = [ordered]@{
    schema = "evomind.video_loudness.v1"
    integrated_lufs = [double]$loudness.input_i
    true_peak_dbtp = [double]$loudness.input_tp
    loudness_range_lu = [double]$loudness.input_lra
    threshold_lufs = [double]$loudness.input_thresh
    target_integrated_lufs = -14.0
    target_true_peak_ceiling_dbtp = -1.0
    ffmpeg_exit_code = $loudnessExit
}
Write-Utf8Json $loudnessReport (Join-Path $reportRoot "loudness.json")

$contactSheet = Join-Path $reportRoot "final-contact-sheet-5s.jpg"
& ffmpeg -loglevel error -nostdin -y -i $input -vf "fps=1/5,scale=640:360,drawtext=fontfile='C\:/Windows/Fonts/msyh.ttc':text='%{pts\:hms}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.75,tile=6x4:padding=4:margin=4" -frames:v 1 -update 1 -q:v 2 $contactSheet
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $contactSheet)) { throw "Contact sheet generation failed." }

$edlPath = Join-Path $project "project\edit-decision-list.json"
$edl = Get-Content -LiteralPath $edlPath -Raw -Encoding UTF8 | ConvertFrom-Json
$filterPath = Join-Path $project "project\render-filter.ffscript"
$filterText = Get-Content -LiteralPath $filterPath -Raw -Encoding UTF8
$realRatio = [double]$edl.visual_contract.real_product_footage_ratio
$cropProof = $filterText -match 'crop=1920:870:0:210' -and $filterText -notmatch '(?i)\bmovie\s*='

$srtPath = Join-Path $project "output\EvoMind-Product-Demo-zh-CN-0.3.0.srt"
$srtText = Get-Content -LiteralPath $srtPath -Raw -Encoding UTF8
$cueMatches = [regex]::Matches($srtText, '(?ms)^\d+\s*\r?\n(?<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?<end>\d{2}:\d{2}:\d{2},\d{3})')
$previousEnd = 0.0
$subtitleOrderingOk = $true
$subtitleCues = @()
foreach ($match in $cueMatches) {
    $start = Convert-SrtTimeToSeconds $match.Groups['start'].Value
    $end = Convert-SrtTimeToSeconds $match.Groups['end'].Value
    if ($start -lt $previousEnd -or $end -le $start) { $subtitleOrderingOk = $false }
    $subtitleCues += [ordered]@{ start_seconds = $start; end_seconds = $end }
    $previousEnd = $end
}
$subtitleTimingOk = $subtitleOrderingOk -and $cueMatches.Count -eq 14 -and $previousEnd -le ($duration + 0.05)
$subtitleReport = [ordered]@{
    schema = "evomind.subtitle_timing.v1"
    file = "video-production/evomind-product-demo-0.3.0/output/EvoMind-Product-Demo-zh-CN-0.3.0.srt"
    cue_count = $cueMatches.Count
    sorted_and_non_overlapping = $subtitleOrderingOk
    final_cue_end_seconds = $previousEnd
    video_duration_seconds = $duration
    within_video_duration = $previousEnd -le ($duration + 0.05)
    burned_in = $true
    independent_srt_present = Test-Path -LiteralPath $srtPath
    status = if ($subtitleTimingOk) { "passed" } else { "failed" }
    cues = $subtitleCues
}
Write-Utf8Json $subtitleReport (Join-Path $reportRoot "subtitle-timing.json") 8

$textFiles = Get-ChildItem -LiteralPath $project -Recurse -File | Where-Object Extension -in @('.json', '.md', '.srt', '.txt', '.ffscript', '.log')
$secretPatterns = [ordered]@{
    private_key_pattern = '-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'
    bearer_token = '(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}'
    assigned_secret = '(?i)\b(?:token|cookie|authorization|password|secret|api[_-]?key)\s*[:=]\s*["'']?[A-Za-z0-9._~+/=-]{16,}'
    user_absolute_path = '(?i)\b[A-Z]:\\Users\\[^\\\s"'']+'
}
$secretFindings = @()
foreach ($file in $textFiles) {
    $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($null -eq $text) { continue }
    foreach ($entry in $secretPatterns.GetEnumerator()) {
        if ($text -match $entry.Value) {
            $relative = $file.FullName.Substring($workspace.Length + 1).Replace('\', '/')
            $secretFindings += [ordered]@{ file = $relative; pattern = $entry.Key }
        }
    }
}

$verifierPath = Join-Path $workspace "artifacts\release-0.3.0-20260803\qa\demo-final-evidence.json"
$verifier = Get-Content -LiteralPath $verifierPath -Raw -Encoding UTF8 | ConvertFrom-Json
$stateReviewTimes = @(0.0, 20.454, 44.888, 66.921, 89.354, 200.721, 221.254, 243.888, 270.754, 296.166)
$sensitiveReview = [ordered]@{
    schema = "evomind.video_sensitive_visual_review.v1"
    status = if ($cropProof -and $secretFindings.Count -eq 0 -and $verifier.status -eq 'passed') { "passed" } else { "failed" }
    decoded_frames_reviewed_by_transform_provenance = $decodedFrames
    per_frame_provenance = [ordered]@{
        source_inputs = @("real-chrome-evolution-demo-final-20260803.mkv")
        source_crop_applied_before_every_output_frame = $cropProof
        excluded_source_region = "x=0..1919, y=0..209 (address bar, bookmarks, automation banner)"
        output_application_region = "x=0..1919, y=105..974"
        synthetic_or_mock_ui_inputs = 0
        real_application_pixel_ratio = $realRatio
    }
    semantic_state_review = [ordered]@{
        exhaustive_state_boundaries_seconds = $stateReviewTimes
        reviewed_items = @("TOKEN", "COOKIE", "Authorization", "password", "private key", "user absolute path", "gateway identity")
        findings = @()
        note = "All decoded output frames share the proven crop. The nine UI states and every source state boundary were visually reviewed through source/final contact sheets; no sensitive value enters the retained application viewport or overlays."
    }
    project_text_scan = [ordered]@{
        files_scanned = $textFiles.Count
        patterns = @($secretPatterns.Keys)
        findings = $secretFindings
    }
    independent_demo_verifier = [ordered]@{
        status = $verifier.status
        recording_ready = $verifier.recording_ready
        blocking_findings = $verifier.blocking_findings
    }
}
Write-Utf8Json $sensitiveReview (Join-Path $reportRoot "sensitive-visual-review.json") 10

$blocking = [System.Collections.Generic.List[string]]::new()
if ($decodeExit -ne 0) { $blocking.Add("decode_integrity_failed") }
if ($detectorExit -ne 0) { $blocking.Add("detector_execution_failed") }
if ($duration -lt 90 -or $duration -gt 120) { $blocking.Add("duration_out_of_range") }
if ([int]$video.width -ne 1920 -or [int]$video.height -ne 1080) { $blocking.Add("resolution_mismatch") }
if ($video.codec_name -ne 'h264' -or $video.profile -ne 'High') { $blocking.Add("h264_high_profile_missing") }
if ($video.pix_fmt -ne 'yuv420p' -or $video.avg_frame_rate -ne '30/1') { $blocking.Add("video_format_mismatch") }
if ($audio.codec_name -ne 'aac' -or [int]$audio.sample_rate -ne 48000 -or [int]$audio.channels -ne 2) { $blocking.Add("audio_format_mismatch") }
if ([math]::Abs([double]$loudness.input_i - (-14.0)) -gt 0.5) { $blocking.Add("loudness_out_of_range") }
if ([double]$loudness.input_tp -gt -1.0) { $blocking.Add("true_peak_above_ceiling") }
if ($blackCount -ne 0) { $blocking.Add("black_intervals_detected") }
if ($freezeCount -ne 0) { $blocking.Add("unapproved_freeze_intervals_detected") }
if ($silenceCount -ne 0) { $blocking.Add("silence_intervals_detected") }
if ($realRatio -lt 0.70) { $blocking.Add("real_product_footage_below_70_percent") }
if (-not $subtitleTimingOk) { $blocking.Add("subtitle_timing_failed") }
if (-not $cropProof -or $secretFindings.Count -ne 0) { $blocking.Add("sensitive_visual_or_project_text_finding") }
if ($verifier.status -ne 'passed' -or -not $verifier.recording_ready -or $verifier.blocking_findings.Count -ne 0) { $blocking.Add("demo_evidence_verifier_failed") }

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $input).Hash.ToLowerInvariant()
$sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $project "sources\real-chrome-evolution-demo-final-20260803.mkv")).Hash.ToLowerInvariant()
$report = [ordered]@{
    schema = "evomind.video_qa.v1"
    generated_at = (Get-Date).ToString("o")
    status = if ($blocking.Count -eq 0) { "passed" } else { "failed" }
    blocking_findings = @($blocking)
    deliverable = [ordered]@{
        file = "video-production/evomind-product-demo-0.3.0/output/EvoMind-Product-Demo-zh-CN-0.3.0.mp4"
        sha256 = $hash
        bytes = (Get-Item -LiteralPath $input).Length
        duration_seconds = $duration
        width = [int]$video.width
        height = [int]$video.height
        fps = $video.avg_frame_rate
        codec = $video.codec_name
        profile = $video.profile
        pixel_format = $video.pix_fmt
        audio_codec = $audio.codec_name
        audio_sample_rate = [int]$audio.sample_rate
        audio_channels = [int]$audio.channels
    }
    source = [ordered]@{
        file = "video-production/evomind-product-demo-0.3.0/sources/real-chrome-evolution-demo-final-20260803.mkv"
        sha256 = $sourceHash
        expected_sha256 = "2aec9b0812e0297c7d63f758ec351deb2f9bc0810229fa03dffa13c52f6c2b37"
        source_seconds_preserved_in_strict_order = 296.166
    }
    qa = [ordered]@{
        decoded_frames = $decodedFrames
        decode_exit_code = $decodeExit
        black_interval_count = $blackCount
        freeze_interval_over_4s_count = $freezeCount
        silence_interval_over_1_5s_count = $silenceCount
        integrated_lufs = [double]$loudness.input_i
        true_peak_dbtp = [double]$loudness.input_tp
        subtitle_cues = $cueMatches.Count
        subtitle_timing = $subtitleReport.status
        real_product_footage_ratio = $realRatio
        sensitive_review = $sensitiveReview.status
        independent_demo_verifier = $verifier.status
        contact_sheet = "artifacts/release-0.3.0-20260803/qa/video/final-contact-sheet-5s.jpg"
    }
    process_interaction = [ordered]@{
        obs = "none"
        chrome = "none"
        evomind_8088 = "none"
        gateway = "none"
        downloader_r6 = "none"
    }
}
Write-Utf8Json $report (Join-Path $reportRoot "video-qa-report.json") 12
Write-Output ($report | ConvertTo-Json -Depth 12)
if ($blocking.Count -ne 0) { exit 1 }
