[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Destination
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
[System.IO.Directory]::CreateDirectory($destinationPath) | Out-Null

$targets = @(
  "src/research_os/agent/llm_refinement_workflow.py",
  "src/research_os/agent/multi_agent.py",
  "web/research-agent-workstation/src/app/api/tasks/[taskId]/refinement/route.ts",
  "web/research-agent-workstation/src/components/workstation/screens/ReportStudioScreen.tsx",
  "scripts/start_verified_workstation.ps1",
  "tests/test_llm_refinement_workflow.py",
  "workspace/current_run.json",
  "workspace/evomind_runs/qwen7b_refine_20260722175524_79b092/run.json",
  "workspace/evomind_runs/qwen7b_refine_20260722175524_79b092/task_graph.json",
  "workspace/evomind_runs/qwen7b_refine_20260722175524_79b092/events.jsonl",
  "workspace/tasks/evomind-qwen7b-finetune/refinements/refine_20260722164606_4da2b0.json"
)

$hashes = foreach ($relative in $targets) {
  $source = Join-Path $root $relative
  if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    continue
  }
  $target = Join-Path $destinationPath $relative
  [System.IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
  Copy-Item -LiteralPath $source -Destination $target -Force
  $hash = Get-FileHash -LiteralPath $source -Algorithm SHA256
  [pscustomobject]@{
    Path = $relative.Replace("\", "/")
    Length = (Get-Item -LiteralPath $source).Length
    SHA256 = $hash.Hash
  }
}

$hashes | Sort-Object Path | Export-Csv -LiteralPath (Join-Path $destinationPath "sha256-before.csv") -NoTypeInformation -Encoding UTF8
git -C $root status --short --branch | Out-File -LiteralPath (Join-Path $destinationPath "git-status-before.txt") -Encoding UTF8

$videoRoot = "E:\EvoMind-release-validation\evomind-commercial-video-v2-20260722-165206"
if (Test-Path -LiteralPath $videoRoot -PathType Container) {
  Get-ChildItem -LiteralPath $videoRoot -Recurse -File |
    Select-Object FullName, Length, LastWriteTimeUtc |
    Export-Csv -LiteralPath (Join-Path $destinationPath "video-inventory-before.csv") -NoTypeInformation -Encoding UTF8
}

[pscustomobject]@{
  status = "created"
  destination = $destinationPath
  files_backed_up = @($hashes).Count
} | ConvertTo-Json -Depth 3
