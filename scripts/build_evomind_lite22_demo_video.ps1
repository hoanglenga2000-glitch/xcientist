param(
    [Parameter(Mandatory = $true)]
    [string]$SourceVideo,

    [string]$OutputRoot = "video-production/evomind-lite22-training-20260808/final"
)

$ErrorActionPreference = "Stop"

$projectRoot = if ($PSScriptRoot) {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    (Get-Location).Path
}
$sourcePath = (Resolve-Path $SourceVideo).Path
$outputDir = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputRoot))
$audioDir = Join-Path $outputDir "narration-clips"
$edgeTts = Join-Path $projectRoot ".venv\Scripts\edge-tts.exe"
if (-not (Test-Path -LiteralPath $edgeTts)) { throw "edge-tts is missing from the project virtual environment" }
New-Item -ItemType Directory -Force -Path $outputDir, $audioDir | Out-Null

$segments = @(
    [pscustomobject]@{ Start = 0.0;  End = 8.0;  Text = "这是 EvoMind 的小白训练演示。整个操作只从一句自然语言开始。" },
    [pscustomobject]@{ Start = 8.0;  End = 17.0; Text = "系统已经绑定 Dec 二零二一研究任务，左侧显示远程 GPU 和数据资源就绪。" },
    [pscustomobject]@{ Start = 17.0; End = 27.0; Text = "用户只要说：检查数据和 GPU，自动选择模型，完成训练验证，并展示模型、分数和报告，不提交 Kaggle。" },
    [pscustomobject]@{ Start = 27.0; End = 37.0; Text = "助手先完成数据、模型和 GPU 门禁，再进入受控训练流程。官方提交始终由人工门禁控制。" },
    [pscustomobject]@{ Start = 37.0; End = 50.0; Text = "随后，GPT 五点六调用实验结果工具，读取三次真实运行、九次迭代和六次晋升。最佳本地准确率是零点九五九六七三。" },
    [pscustomobject]@{ Start = 50.0; End = 61.0; Text = "最佳证据来自七月四日的运行，最佳实验是 EXP 零零二。界面同时列出三份可审计 summary 文件。" },
    [pscustomobject]@{ Start = 61.0; End = 73.0; Text = "最后，助手直接读取 Lite 二十二报告。二十二项都有训练和有效结果文件，二十一项完成私有评分，十三项达到离线门槛，其中六项为 Gold 等效。" },
    [pscustomobject]@{ Start = 73.0; End = 81.5; Text = "Dec 二零二一私有评分是零点九五八三六，高于零点九五六六门槛。唯一未评分的 Dog Breed 被质量门禁拦下。全程没有 Kaggle 官方提交或线上排名。" }
)

function Format-SrtTimestamp([double]$seconds) {
    $span = [TimeSpan]::FromSeconds($seconds)
    return "{0:00}:{1:00}:{2:00},{3:000}" -f [math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds, $span.Milliseconds
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$srtLines = New-Object System.Collections.Generic.List[string]
$audioInputs = New-Object System.Collections.Generic.List[string]
$audioFilters = New-Object System.Collections.Generic.List[string]
$mixLabels = New-Object System.Collections.Generic.List[string]

for ($i = 0; $i -lt $segments.Count; $i++) {
    $segment = $segments[$i]
    $clipPath = Join-Path $audioDir ("narration-{0:00}.mp3" -f ($i + 1))
    if (Test-Path -LiteralPath $clipPath) { Remove-Item -LiteralPath $clipPath -Force }
    & $edgeTts --voice "zh-CN-XiaoxiaoNeural" --rate="+5%" --text $segment.Text --write-media $clipPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $clipPath)) {
        throw "edge-tts failed for narration segment $($i + 1)"
    }

    $srtLines.Add([string]($i + 1))
    $srtLines.Add("$(Format-SrtTimestamp $segment.Start) --> $(Format-SrtTimestamp $segment.End)")
    $srtLines.Add($segment.Text)
    $srtLines.Add("")

    $slot = [double]($segment.End - $segment.Start)
    $durationRaw = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $clipPath
    $clipDuration = [double]::Parse(($durationRaw | Select-Object -First 1), [Globalization.CultureInfo]::InvariantCulture)
    $tempo = if ($clipDuration -gt $slot) { $clipDuration / $slot } else { 1.0 }
    $slotText = $slot.ToString("0.######", [Globalization.CultureInfo]::InvariantCulture)
    $tempoText = $tempo.ToString("0.######", [Globalization.CultureInfo]::InvariantCulture)
    $delayMs = [int][math]::Round($segment.Start * 1000)
    $inputIndex = $i + 1
    $label = "n$($i + 1)"
    $audioInputs.Add("-i")
    $audioInputs.Add($clipPath)
    $audioFilters.Add("[$inputIndex`:a]aresample=48000,atempo=$tempoText,atrim=0:$slotText,apad=pad_dur=$slotText,atrim=0:$slotText,adelay=$delayMs|$delayMs[$label]")
    $mixLabels.Add("[$label]")
}

$srtPath = Join-Path $outputDir "evomind-lite22-demo.zh-CN.srt"
[System.IO.File]::WriteAllLines($srtPath, $srtLines, $utf8NoBom)

$finalPath = Join-Path $outputDir "EvoMind-Lite22-小白训练演示-1080p.mp4"
$filter = ($audioFilters -join ";") + ";anoisesrc=color=pink:amplitude=0.012:sample_rate=48000:duration=81.5[bed];" +
    ($mixLabels -join "") + "[bed]amix=inputs=$($segments.Count + 1):normalize=0,loudnorm=I=-14:TP=-1:LRA=7[mix];" +
    "[0:v]subtitles='evomind-lite22-demo.zh-CN.srt':force_style='FontName=Microsoft YaHei,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101828,BorderStyle=1,Outline=2,Shadow=0,MarginV=46'[base];" +
    "color=c=0x14B8A6@0.92:s=320x8:r=30:d=81.5[runner];[base][runner]overlay=x='mod(t*240,1600)':y=1072:eval=frame:shortest=1," +
    "eq=brightness='0.006*sin(2*PI*3*t)':eval=frame[v]"

Push-Location $outputDir
try {
    $args = New-Object System.Collections.Generic.List[string]
    foreach ($value in @("-y", "-hide_banner", "-loglevel", "warning", "-i", $sourcePath)) { $args.Add($value) }
    foreach ($value in $audioInputs) { $args.Add($value) }
    foreach ($value in @(
        "-filter_complex", $filter,
        "-map", "[v]", "-map", "[mix]",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.1",
        "-pix_fmt", "yuv420p", "-r", "30", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-shortest", $finalPath
    )) { $args.Add($value) }
    & ffmpeg @args
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg export failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

$probe = & ffprobe -v error -show_entries "format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels" -of json $finalPath
$result = [ordered]@{
    schema = "evomind.video_export.v1"
    source = $sourcePath
    output = $finalPath
    subtitles = $srtPath
    narration_clips = $segments.Count
    ffprobe = ($probe | ConvertFrom-Json)
}
$resultPath = Join-Path $outputDir "export-result.json"
[System.IO.File]::WriteAllText($resultPath, ($result | ConvertTo-Json -Depth 10), $utf8NoBom)
$result | ConvertTo-Json -Depth 10
