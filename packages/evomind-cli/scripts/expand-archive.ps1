param(
  [Parameter(Mandatory = $true)][string]$Archive,
  [Parameter(Mandatory = $true)][string]$Destination
)
$ErrorActionPreference = "Stop"
$archivePath = [IO.Path]::GetFullPath($Archive)
$destinationPath = [IO.Path]::GetFullPath($Destination)
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) { throw "Archive does not exist." }
if ([IO.Path]::GetExtension($archivePath) -ne ".zip") { throw "Only ZIP archives are accepted." }
if (-not (Test-Path -LiteralPath $destinationPath -PathType Container)) { throw "Destination does not exist." }
if (@(Get-ChildItem -LiteralPath $destinationPath -Force).Count -ne 0) { throw "Destination must be empty." }
$destinationPrefix = $destinationPath.TrimEnd('\') + '\'
$reserved = '^(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)'
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$kinds = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase)
$maxEntries = 100000
$maxUncompressed = [int64]8GB
$maxMember = [int64]2GB
$maxRatio = 1000.0
$totalUncompressed = [int64]0
$totalCompressed = [int64]0
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
  if ($zip.Entries.Count -eq 0 -or $zip.Entries.Count -gt $maxEntries) {
    throw "ZIP entry-count limit exceeded: $($zip.Entries.Count)"
  }
  foreach ($entry in $zip.Entries) {
    if ([string]::IsNullOrWhiteSpace([string]$entry.FullName) -or ([string]$entry.FullName).Length -gt 1024) {
      throw "Unsafe ZIP path length: $($entry.FullName)"
    }
    $relative = ([string]$entry.FullName).Replace('/', '\')
    $isDirectory = $relative.EndsWith('\')
    $normalizedRelative = $relative.TrimEnd('\')
    $parts = @($normalizedRelative.Split('\', [StringSplitOptions]::None))
    $unsafePart = @($parts | Where-Object {
      [string]::IsNullOrEmpty($_) -or $_ -eq '..' -or $_ -eq '.' -or $_.Length -gt 255 -or
      $_ -match '[\x00-\x1f<>:"|?*]' -or $_ -match '[. ]$' -or $_ -match $reserved
    })
    if ([string]::IsNullOrWhiteSpace($normalizedRelative) -or [IO.Path]::IsPathRooted($normalizedRelative) -or $normalizedRelative -match '^[A-Za-z]:' -or $unsafePart.Count -gt 0) {
      throw "Unsafe ZIP entry: $($entry.FullName)"
    }
    if (-not $seen.Add($normalizedRelative)) { throw "Case-fold ZIP path collision: $($entry.FullName)" }
    for ($index = 1; $index -lt $parts.Count; $index++) {
      $parentKey = [string]::Join('\', $parts[0..($index - 1)])
      if ($kinds.ContainsKey($parentKey) -and $kinds[$parentKey] -eq 'file') {
        throw "ZIP file/directory prefix collision: $($entry.FullName)"
      }
    }
    if (-not $isDirectory) {
      $prefix = $normalizedRelative + '\'
      foreach ($existing in $kinds.Keys) {
        if ($existing.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
          throw "ZIP file/directory prefix collision: $($entry.FullName)"
        }
      }
    }
    $kinds[$normalizedRelative] = $(if ($isDirectory) { 'directory' } else { 'file' })
    if ($entry.Length -lt 0 -or $entry.CompressedLength -lt 0 -or $entry.Length -gt $maxMember) {
      throw "ZIP member-size limit exceeded: $($entry.FullName)"
    }
    if ($isDirectory -and $entry.Length -ne 0) { throw "ZIP directory contains a payload: $($entry.FullName)" }
    $totalUncompressed += [int64]$entry.Length
    $totalCompressed += [int64]$entry.CompressedLength
    if ($totalUncompressed -gt $maxUncompressed) { throw "ZIP uncompressed-size limit exceeded." }
    if ($entry.Length -gt 0 -and ([double]$entry.Length / [Math]::Max([double]$entry.CompressedLength, 1.0)) -gt $maxRatio) {
      throw "ZIP member compression-ratio limit exceeded: $($entry.FullName)"
    }
    $target = [IO.Path]::GetFullPath((Join-Path $destinationPath $normalizedRelative))
    if (-not $target.StartsWith($destinationPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "ZIP entry escapes destination: $($entry.FullName)" }
    $unixType = (($entry.ExternalAttributes -shr 16) -band 0xF000)
    if ($unixType -eq 0xA000) { throw "ZIP symbolic links are forbidden: $($entry.FullName)" }
  }
  if ($totalUncompressed -gt 0 -and ([double]$totalUncompressed / [Math]::Max([double]$totalCompressed, 1.0)) -gt $maxRatio) {
    throw "ZIP aggregate compression-ratio limit exceeded."
  }
} finally {
  $zip.Dispose()
}
Expand-Archive -LiteralPath $archivePath -DestinationPath $destinationPath -Force
$reparse = Get-ChildItem -LiteralPath $destinationPath -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1
if ($reparse) { throw "Expanded archive contains a reparse point: $($reparse.FullName)" }
