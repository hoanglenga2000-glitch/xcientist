[CmdletBinding()]
param(
    [string]$ProjectDir = "video-production/evomind-product-demo-0.3.0",
    [string]$QaDir = "artifacts/release-0.3.0-20260803/qa/video",
    [string]$Version = "0.3.0",
    [switch]$ReplaceExisting
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$workspace = (Resolve-Path -LiteralPath ".").Path
$project = (Resolve-Path -LiteralPath $ProjectDir).Path
$qa = (Resolve-Path -LiteralPath $QaDir).Path
$qaReportPath = Join-Path $qa "video-qa-report.json"
$qaReport = Get-Content -LiteralPath $qaReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($qaReport.status -ne "passed" -or $qaReport.blocking_findings.Count -ne 0) {
    throw "Video QA must be passed with zero blockers before packaging."
}

$expectedSourceSha256 = "2aec9b0812e0297c7d63f758ec351deb2f9bc0810229fa03dffa13c52f6c2b37"
$expectedFinalSha256 = "de1c34ccb66a61bd747c8c1a15e6cebbbb4f8875fbb5fd2241f9931d1f11f797"
$sourceMediaPath = Join-Path $project "sources\real-chrome-evolution-demo-final-20260803.mkv"
$finalMediaPath = Join-Path $project "output\EvoMind-Product-Demo-zh-CN-$Version.mp4"
$actualSourceSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceMediaPath).Hash.ToLowerInvariant()
$actualFinalSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $finalMediaPath).Hash.ToLowerInvariant()
if ($actualSourceSha256 -ne $expectedSourceSha256) {
    throw "Frozen Chrome master hash drifted: $actualSourceSha256"
}
if ($actualFinalSha256 -ne $expectedFinalSha256 -or $qaReport.deliverable.sha256 -ne $expectedFinalSha256) {
    throw "Frozen final MP4 hash drifted or no longer matches QA: $actualFinalSha256"
}

$edlValidationPath = Join-Path $qa "edl-validation.json"
$edlValidation = Get-Content -LiteralPath $edlValidationPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($edlValidation.status -ne "passed" -or $edlValidation.blocking_findings.Count -ne 0) {
    throw "EDL validation must be passed with zero blockers before packaging."
}

$stage = Join-Path $qa "evidence-pack-$Version"
$zip = Join-Path $qa "EvoMind-video-evidence-$Version.zip"
if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)) {
    if (-not $ReplaceExisting) { throw "Evidence package already exists; pass -ReplaceExisting to preserve and replace it." }
    $supersededRoot = Join-Path $qa ("superseded\video-evidence-{0}-{1}" -f $Version, (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ'))
    New-Item -ItemType Directory -Path $supersededRoot -Force | Out-Null
    if (Test-Path -LiteralPath $stage -PathType Container) {
        [IO.Directory]::Move($stage, (Join-Path $supersededRoot (Split-Path -Leaf $stage)))
    }
    if (Test-Path -LiteralPath $zip -PathType Leaf) {
        [IO.File]::Move($zip, (Join-Path $supersededRoot (Split-Path -Leaf $zip)))
    }
}
New-Item -ItemType Directory -Path $stage | Out-Null

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$workspaceJsonEscaped = $workspace.Replace('\', '\\')
$workspaceForward = $workspace.Replace('\', '/')
$toolchainRoot = 'E:\EvoMind-release-validation'
$jianyingProjectManifestPath = Join-Path $project "project\jianying-editable-0.3.0\evomind-jianying-project-manifest.json"
$jianyingProjectManifest = Get-Content -LiteralPath $jianyingProjectManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $jianyingProjectManifest.passed) { throw "Formal JianYing project manifest is not passed." }
$jianyingDraftRoot = Split-Path -Parent ([string]$jianyingProjectManifest.live_draft_path)
$publicTextReplacements = @(
    [pscustomobject]@{ From = $jianyingDraftRoot.Replace('\', '\\'); To = '${JY_DRAFT_ROOT}' },
    [pscustomobject]@{ From = $jianyingDraftRoot.Replace('\', '/'); To = '${JY_DRAFT_ROOT}' },
    [pscustomobject]@{ From = $jianyingDraftRoot; To = '${JY_DRAFT_ROOT}' },
    [pscustomobject]@{ From = $workspaceJsonEscaped; To = '${WORKSPACE}' },
    [pscustomobject]@{ From = $workspaceForward; To = '${WORKSPACE}' },
    [pscustomobject]@{ From = $workspace; To = '${WORKSPACE}' },
    [pscustomobject]@{ From = $toolchainRoot.Replace('\', '\\'); To = '${VIDEO_TOOLCHAIN}' },
    [pscustomobject]@{ From = $toolchainRoot.Replace('\', '/'); To = '${VIDEO_TOOLCHAIN}' },
    [pscustomobject]@{ From = $toolchainRoot; To = '${VIDEO_TOOLCHAIN}' },
    [pscustomobject]@{ From = 'C\:/Windows'; To = '${WINDOWS}' },
    [pscustomobject]@{ From = 'C:\\Windows'; To = '${WINDOWS}' },
    [pscustomobject]@{ From = 'C:/Windows'; To = '${WINDOWS}' },
    [pscustomobject]@{ From = 'C:\Windows'; To = '${WINDOWS}' }
)
$publicTextExtensions = @('.json', '.md', '.txt', '.log', '.srt', '.ps1', '.py', '.ffscript')

function Convert-ToPublicEvidenceText([string]$path) {
    $item = Get-Item -LiteralPath $path
    if ($item.Extension -notin $publicTextExtensions -and $item.Name -ne 'draft_settings') { return }
    $text = [System.IO.File]::ReadAllText($item.FullName, $utf8NoBom)
    foreach ($replacement in $publicTextReplacements) {
        $text = $text.Replace($replacement.From, $replacement.To)
    }
    [System.IO.File]::WriteAllText($item.FullName, $text, $utf8NoBom)
    if ($item.Extension -eq '.json') {
        try { $null = $text | ConvertFrom-Json } catch { throw "Sanitized JSON is invalid: $path`n$_" }
    }
}

function Copy-EvidenceFile([string]$source, [string]$relativeDestination) {
    $resolved = (Resolve-Path -LiteralPath $source).Path
    $destination = Join-Path $stage $relativeDestination
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $resolved -Destination $destination
    Convert-ToPublicEvidenceText $destination
}

# Frozen source, delivery, deterministic project and exact audio used by render.
Copy-EvidenceFile $sourceMediaPath "media/source/real-chrome-evolution-demo-final-20260803.mkv"
Copy-EvidenceFile $finalMediaPath "media/final/EvoMind-Product-Demo-zh-CN-$Version.mp4"
Copy-EvidenceFile (Join-Path $project "output\EvoMind-Product-Demo-zh-CN-$Version.srt") "media/final/EvoMind-Product-Demo-zh-CN-$Version.srt"
foreach ($name in @("final-mix.wav", "narration-timeline.wav", "evomind-ambient-bed.wav")) {
    Copy-EvidenceFile (Join-Path $project "assets\$name") "project/assets/$name"
}
Get-ChildItem -LiteralPath (Join-Path $project "assets\narration-segments") -File | Where-Object Extension -in @('.mp3', '.txt') | ForEach-Object {
    Copy-EvidenceFile $_.FullName ("project/assets/narration-segments/" + $_.Name)
}
Get-ChildItem -LiteralPath (Join-Path $project "assets\headers") -File | ForEach-Object {
    Copy-EvidenceFile $_.FullName ("project/assets/headers/" + $_.Name)
}
Get-ChildItem -LiteralPath (Join-Path $project "project") -Recurse -File | Where-Object {
    $_.FullName -notlike ((Join-Path $project "project\jianying-validator-smoke") + "*")
} | ForEach-Object {
    $relative = $_.FullName.Substring((Join-Path $project "project").Length + 1).Replace('\', '/')
    Copy-EvidenceFile $_.FullName ("project/edit/" + $relative)
}
Get-ChildItem -LiteralPath (Join-Path $project "analysis") -File | ForEach-Object {
    Copy-EvidenceFile $_.FullName ("qa/source-analysis/" + $_.Name)
}

# Only immutable media-QA inputs belong in the archive. The ZIP verification and
# release README are generated after packaging and must never recurse into it.
$qaFileNames = @(
    'decode-integrity.log',
    'edl-validation.json',
    'final-black-freeze-silence.log',
    'final-contact-sheet-10s.jpg',
    'final-contact-sheet-5s.jpg',
    'final-ffprobe.json',
    'final-frame-110s.png',
    'final-frame-115s.png',
    'final-frame-117s.png',
    'final-frame-119s.png',
    'loudness.json',
    'preflight.json',
    'sensitive-visual-review.json',
    'subtitle-timing.json',
    'video-qa-report.json'
)
foreach ($name in $qaFileNames) {
    Copy-EvidenceFile (Join-Path $qa $name) ("qa/final/" + $name)
}
Copy-EvidenceFile (Join-Path $workspace "artifacts\release-0.3.0-20260803\qa\demo-final-evidence.json") "qa/independent-verifier/demo-final-evidence.json"
foreach ($scriptName in @("build_evomind_product_demo.py", "build_evomind_jianying_project.py", "video-preflight.ps1", "video-qa.ps1", "video-package-evidence.ps1")) {
    Copy-EvidenceFile (Join-Path $workspace "scripts\$scriptName") ("project/scripts/" + $scriptName)
}

$readme = @"
# EvoMind Video Evidence $Version

This archive contains the frozen real Chrome master, the 119.444-second H.264/AAC
Chinese product film, independent SRT, deterministic EDL/filter/build scripts,
the exact narration/music mix inputs, source analysis, final media QA and the
independent demo verifier report.

The archive also includes `qa/final/edl-validation.json`, which proves strict
source ordering, complete source-time coverage, and the measured real-product
footage ratios used by the public claim map.

Evidence boundary: all scores are local public-validation or local sealed-holdout
facts. The film does not claim an official Kaggle rank, medal, SIIM private score,
or an official MLE-Bench result.

Rebuild from the repository root with:

    python scripts/build_evomind_product_demo.py
    powershell -ExecutionPolicy Bypass -File scripts/video-qa.ps1

The manifest uses relative archive paths and SHA-256 for every file.
All ZIP entry names use POSIX `/` separators. Local paths in staged text are
portable placeholders: `${WORKSPACE}, `${VIDEO_TOOLCHAIN}, and `${WINDOWS}.
"@
[System.IO.File]::WriteAllText((Join-Path $stage "README.md"), $readme, $utf8NoBom)

$items = @()
Get-ChildItem -LiteralPath $stage -Recurse -File | Where-Object Name -ne "MANIFEST-SHA256.json" | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
    $items += [ordered]@{
        path = $relative
        bytes = $_.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    schema = "evomind.video_evidence_manifest.v1"
    version = $Version
    generated_at = (Get-Date).ToString("o")
    source_sha256 = $expectedSourceSha256
    final_sha256 = $qaReport.deliverable.sha256
    qa_status = $qaReport.status
    blocking_findings = @($qaReport.blocking_findings)
    file_count = $items.Count
    files = $items
}
$manifestPath = Join-Path $stage "MANIFEST-SHA256.json"
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), $utf8NoBom)

$operatorName = -join ([char[]]@(0x666F, 0x6D69, 0x4F1F))
$operatorAliasPattern = '\b' + ('jing', 'hw' -join '') + '\b'
$gatewayPattern = '(?i)(?:47\.97\.124\.121|43\.130\.57\.88|119\.3\.164\.120|10\.120\.\d{1,3}\.\d{1,3}|ai\.ltcraft\.cn|' + ('job', '90353' -join '') + ')'
$sensitivePatterns = [ordered]@{
    local_drive_path = '(?i)(?<![A-Za-z0-9])(?:[A-Z]:(?:[\\/]|\\\\)|[A-Z]\\:[\\/])[^\x00-\x1F\\/:*?"<>|]{2,}(?:(?:[\\/]|\\\\)[^\x00-\x1F\\/:*?"<>|]{1,})+'
    absolute_workspace_path = '(?i)' + [regex]::Escape($workspace)
    user_identity = '(?i)(?:' + [regex]::Escape($operatorName) + '|Jing\s+Haowei|' + $operatorAliasPattern + ')'
    gateway_or_infrastructure_identity = $gatewayPattern
    private_key_pattern = '-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'
    bearer_token = '(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}'
    assigned_secret = '(?i)\b(?:token|cookie|authorization|password|secret|api[_-]?key)\s*[:=]\s*["'']?[A-Za-z0-9._~+/=-]{16,}'
    bootstrap_or_query_token = '(?i)(?:[#?&](?:bootstrap|token|auth)=)[A-Za-z0-9._~+/=-]{12,}'
    jwt = '\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\b'
}

function Add-SensitiveFindings([byte[]]$bytes, [string]$logicalPath, $findings) {
    $extension = [System.IO.Path]::GetExtension($logicalPath)
    $name = [System.IO.Path]::GetFileName($logicalPath)
    $utf8Text = [System.Text.Encoding]::UTF8.GetString($bytes)
    $unicodeText = [System.Text.Encoding]::Unicode.GetString($bytes)
    $isText = $extension -in $publicTextExtensions -or $name -eq 'draft_settings'
    if ($isText) {
        $decoded = @($utf8Text, $unicodeText)
    } else {
        # Every binary byte is mapped through Latin-1, then only durable printable
        # strings are searched. This detects embedded metadata/secrets without
        # treating random two- or three-byte codec payloads as a drive path.
        $latinText = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
        $asciiRuns = @([regex]::Matches($latinText, '[ -~]{10,}') | ForEach-Object Value)
        $unicodeRuns = @([regex]::Matches($unicodeText, '[ -~]{10,}') | ForEach-Object Value)
        $decoded = @(($asciiRuns + $unicodeRuns) -join "`n")
    }
    $foundPatterns = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    if ($utf8Text.Contains($workspace) -or $unicodeText.Contains($workspace)) {
        $null = $foundPatterns.Add('absolute_workspace_path')
        $findings.Add([ordered]@{ path = $logicalPath; pattern = 'absolute_workspace_path' })
    }
    if ($utf8Text.Contains($operatorName) -or $unicodeText.Contains($operatorName)) {
        $null = $foundPatterns.Add('user_identity')
        $findings.Add([ordered]@{ path = $logicalPath; pattern = 'user_identity' })
    }
    foreach ($pattern in $sensitivePatterns.GetEnumerator()) {
        if ($foundPatterns.Contains($pattern.Key)) { continue }
        foreach ($text in $decoded) {
            if ([regex]::IsMatch($text, $pattern.Value)) {
                $null = $foundPatterns.Add($pattern.Key)
                $findings.Add([ordered]@{ path = $logicalPath; pattern = $pattern.Key })
                break
            }
        }
    }
}

$stageFiles = @(Get-ChildItem -LiteralPath $stage -Recurse -File | Sort-Object FullName)
$stageSensitiveFindings = [System.Collections.Generic.List[object]]::new()
$stageBytesScanned = [long]0
foreach ($file in $stageFiles) {
    $relative = $file.FullName.Substring($stage.Length + 1).Replace('\', '/')
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $stageBytesScanned += $bytes.LongLength
    Add-SensitiveFindings $bytes $relative $stageSensitiveFindings
}
if ($stageSensitiveFindings.Count -ne 0) {
    throw "Sensitive data remained in evidence stage: $($stageSensitiveFindings | ConvertTo-Json -Compress)"
}

# ZipArchive.CreateEntry stores the supplied name verbatim. Supplying normalized
# relative paths here avoids Compress-Archive's Windows backslash entry names.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipStream = [System.IO.File]::Open($zip, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$createArchive = [System.IO.Compression.ZipArchive]::new($zipStream, [System.IO.Compression.ZipArchiveMode]::Create, $false, $utf8NoBom)
try {
    foreach ($file in $stageFiles) {
        $relative = $file.FullName.Substring($stage.Length + 1).Replace('\', '/')
        $entry = $createArchive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
        $entryStream = $entry.Open()
        $sourceStream = [System.IO.File]::OpenRead($file.FullName)
        try { $sourceStream.CopyTo($entryStream) } finally {
            $sourceStream.Dispose()
            $entryStream.Dispose()
        }
    }
} finally {
    $createArchive.Dispose()
    $zipStream.Dispose()
}
if (-not (Test-Path -LiteralPath $zip)) { throw "ZIP creation failed." }

$expectedEntryNames = @($items.path) + @('MANIFEST-SHA256.json')
$manifestMissing = [System.Collections.Generic.List[string]]::new()
$manifestExtra = [System.Collections.Generic.List[string]]::new()
$manifestHashMismatches = [System.Collections.Generic.List[string]]::new()
$manifestSizeMismatches = [System.Collections.Generic.List[string]]::new()
$invalidEntryNames = [System.Collections.Generic.List[string]]::new()
$duplicateEntryNames = [System.Collections.Generic.List[string]]::new()
$zipSensitiveFindings = [System.Collections.Generic.List[object]]::new()
$zipBytesScanned = [long]0
$archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
try {
    $entryCount = $archive.Entries.Count
    $actualEntryNames = @($archive.Entries | ForEach-Object FullName)
    $seenNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($name in $actualEntryNames) {
        $parts = @($name.Split('/'))
        if ($name.Contains('\') -or $name.StartsWith('/') -or $name -match '^[A-Za-z]:' -or $parts -contains '' -or $parts -contains '.' -or $parts -contains '..') {
            $invalidEntryNames.Add($name)
        }
        if (-not $seenNames.Add($name)) { $duplicateEntryNames.Add($name) }
    }
    foreach ($name in $expectedEntryNames) {
        if ($name -notin $actualEntryNames) { $manifestMissing.Add($name) }
    }
    foreach ($name in $actualEntryNames) {
        if ($name -notin $expectedEntryNames) { $manifestExtra.Add($name) }
    }

    $manifestByPath = @{}
    foreach ($item in $items) { $manifestByPath[$item.path] = $item }
    foreach ($entry in $archive.Entries) {
        $stream = $entry.Open()
        $memory = [System.IO.MemoryStream]::new()
        try {
            $stream.CopyTo($memory)
            $bytes = $memory.ToArray()
        } finally {
            $memory.Dispose()
            $stream.Dispose()
        }
        $zipBytesScanned += $bytes.LongLength
        Add-SensitiveFindings $bytes $entry.FullName $zipSensitiveFindings
        if ($entry.FullName -eq 'MANIFEST-SHA256.json') {
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try {
                $entryManifestHash = [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
            } finally {
                $sha.Dispose()
            }
            $stageManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant()
            if ($entryManifestHash -ne $stageManifestHash) {
                $manifestHashMismatches.Add('MANIFEST-SHA256.json')
            }
            continue
        }
        $item = $manifestByPath[$entry.FullName]
        if ($null -eq $item) { continue }
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $actualHash = [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
        if ($actualHash -ne $item.sha256) { $manifestHashMismatches.Add($entry.FullName) }
        if ($bytes.LongLength -ne [long]$item.bytes -or $entry.Length -ne [long]$item.bytes) { $manifestSizeMismatches.Add($entry.FullName) }
    }
} finally {
    $archive.Dispose()
}

$packageBlockers = [System.Collections.Generic.List[string]]::new()
if ($invalidEntryNames.Count -ne 0 -or $duplicateEntryNames.Count -ne 0) { $packageBlockers.Add('zip_entry_name_audit_failed') }
if ($manifestMissing.Count -ne 0 -or $manifestExtra.Count -ne 0 -or $manifestHashMismatches.Count -ne 0 -or $manifestSizeMismatches.Count -ne 0) { $packageBlockers.Add('manifest_exactness_failed') }
if ($stageSensitiveFindings.Count -ne 0 -or $zipSensitiveFindings.Count -ne 0) { $packageBlockers.Add('sensitive_data_scan_failed') }
if ('qa/final/edl-validation.json' -notin $actualEntryNames) { $packageBlockers.Add('edl_validation_missing') }
$allEntriesExact = $manifestMissing.Count -eq 0 -and $manifestExtra.Count -eq 0 -and $manifestHashMismatches.Count -eq 0 -and $manifestSizeMismatches.Count -eq 0
$entryNamesArePosix = $invalidEntryNames.Count -eq 0 -and $duplicateEntryNames.Count -eq 0
$verification = [ordered]@{
    schema = "evomind.video_evidence_zip_verification.v1"
    generated_at = (Get-Date).ToString('o')
    status = if ($packageBlockers.Count -eq 0) { "passed" } else { "failed" }
    zip = "artifacts/release-0.3.0-20260803/qa/video/EvoMind-video-evidence-$Version.zip"
    sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
    bytes = (Get-Item -LiteralPath $zip).Length
    entry_count = $entryCount
    manifest_file_count = $items.Count
    expected_entry_count = $items.Count + 1
    posix_entry_names = $entryNamesArePosix
    invalid_entry_names = @($invalidEntryNames)
    duplicate_entry_names = @($duplicateEntryNames)
    manifest_missing_entries = @($manifestMissing)
    manifest_extra_entries = @($manifestExtra)
    manifest_hash_mismatches = @($manifestHashMismatches)
    manifest_size_mismatches = @($manifestSizeMismatches)
    all_entries_read_without_error = $true
    all_manifest_entries_exact = $allEntriesExact
    edl_validation = [ordered]@{
        included = 'qa/final/edl-validation.json' -in $actualEntryNames
        status = $edlValidation.status
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $edlValidationPath).Hash.ToLowerInvariant()
    }
    sensitive_data_scan = [ordered]@{
        patterns = @($sensitivePatterns.Keys)
        stage = [ordered]@{
            files_scanned = $stageFiles.Count
            bytes_scanned = $stageBytesScanned
            findings = @($stageSensitiveFindings)
        }
        zip = [ordered]@{
            entries_scanned = $entryCount
            bytes_scanned = $zipBytesScanned
            findings = @($zipSensitiveFindings)
        }
    }
    media_integrity = [ordered]@{
        source_sha256 = $actualSourceSha256
        final_sha256 = $actualFinalSha256
        source_unchanged = $actualSourceSha256 -eq $expectedSourceSha256
        final_unchanged = $actualFinalSha256 -eq $expectedFinalSha256
    }
    video_qa_status = $qaReport.status
    blocking_findings = @($packageBlockers)
}
[System.IO.File]::WriteAllText((Join-Path $qa "evidence-zip-verification.json"), ($verification | ConvertTo-Json -Depth 10), $utf8NoBom)
Write-Output ($verification | ConvertTo-Json -Depth 6)
if ($packageBlockers.Count -ne 0) { exit 1 }
