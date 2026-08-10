[CmdletBinding()]
param([switch]$SkipMaster)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$videoRoot = "E:\EvoMind-release-validation\evomind-commercial-video-v2-20260722-165206"
$delivery = "E:\EvoMind-release-validation\evomind-v5-professional-video-20260723"
$python = "E:\EvoMind-release-validation\video-tooling\.venv\Scripts\python.exe"
$videoSkillScripts = Join-Path $env:USERPROFILE ".codex\skills\video-production-pro\scripts"
$preflightScript = Join-Path $videoSkillScripts "video-preflight.ps1"
$qaScript = Join-Path $videoSkillScripts "video-qa.ps1"
$capabilityGate = Join-Path $PSScriptRoot "verify_v5_capability_contract.py"
$evidenceGate = Join-Path $PSScriptRoot "verify_v5_evidence.py"
$publicTextGate = Join-Path $PSScriptRoot "verify_v5_public_text.py"
$capabilityOut = Join-Path $delivery "qa\v5-capability-contract.json"
$evidenceOut = Join-Path $delivery "qa\v5-evidence-gate.json"
$publicTextOut = Join-Path $delivery "qa\v5-public-text-gate.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Video tooling Python is missing." }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $evidenceOut) | Out-Null
& powershell -NoProfile -ExecutionPolicy Bypass -File $preflightScript -OutputJson "$delivery\qa\v5-preflight.json"
if ($LASTEXITCODE -ne 0) { throw "V5 video preflight failed." }
& $python $capabilityGate --workspace-root $repo --json-out $capabilityOut
if ($LASTEXITCODE -ne 0) { throw "V5 product capability contract failed." }
& $python $publicTextGate --video-root $videoRoot --json-out $publicTextOut
if ($LASTEXITCODE -ne 0) { throw "V5 public wording or reviewed visual baseline gate failed." }
& $python $evidenceGate --workspace-root $repo --video-root $videoRoot --ffprobe "D:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe" --require-captures --json-out $evidenceOut
if ($LASTEXITCODE -ne 0) { throw "V5 evidence gate blocked audio generation and video build." }
& $python "$PSScriptRoot\generate_audio_v5.py"
if ($LASTEXITCODE -ne 0) { throw "V5 narration/audio generation failed." }
& $python $publicTextGate --video-root $videoRoot --require-generated --json-out $publicTextOut
if ($LASTEXITCODE -ne 0) { throw "Generated subtitles failed the V5 public wording gate." }

$audioDelivery = Join-Path $delivery "audio"
$captionDelivery = Join-Path $delivery "captions"
$recordingDelivery = Join-Path $delivery "source-recordings"
$evidenceDelivery = Join-Path $delivery "evidence"
foreach ($directory in @($audioDelivery, $captionDelivery, $recordingDelivery, $evidenceDelivery)) {
  New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
Copy-Item -LiteralPath (Join-Path $videoRoot "audio\final-mix-v5.wav") -Destination $audioDelivery -Force
Copy-Item -LiteralPath (Join-Path $videoRoot "audio\narration-manifest-v5.json") -Destination $audioDelivery -Force
Copy-Item -LiteralPath (Join-Path $videoRoot "narration-v5.json") -Destination $audioDelivery -Force
Copy-Item -LiteralPath (Join-Path $videoRoot "subtitles-v5.srt") -Destination $captionDelivery -Force
Copy-Item -LiteralPath (Join-Path $videoRoot "subtitles-v5.ass") -Destination $captionDelivery -Force
Copy-Item -LiteralPath $evidenceOut -Destination (Join-Path $evidenceDelivery "v5-evidence-map.json") -Force
Get-ChildItem -LiteralPath (Join-Path $videoRoot "captures") -Filter "v5-*.mp4" -File |
  Copy-Item -Destination $recordingDelivery -Force

$arguments = @("$PSScriptRoot\build_video_v5.py")
if ($SkipMaster) { $arguments += "--skip-master" }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "V5 evidence gate or video build failed." }
powershell -ExecutionPolicy Bypass -File $qaScript -InputPath "$delivery\EvoMind-Enterprise-AI-Workflow-V5.mp4" -ReportDir "$delivery\qa\v5-final"
if ($LASTEXITCODE -ne 0) { throw "V5 video QA failed." }
