param(
  [string]$BaseUrl = "http://127.0.0.1:8088",
  [switch]$SkipClickSmoke
)

$ErrorActionPreference = "Stop"

$pages = @(
  "overview",
  "control",
  "data",
  "report",
  "code",
  "gpu",
  "evidence",
  "literature",
  "runtime",
  "settings"
)

$root = Split-Path -Parent $PSScriptRoot
$appShell = Join-Path $root "web/research-agent-workstation/src/components/workstation/AppShell.tsx"
$sidebar = Join-Path $root "web/research-agent-workstation/src/components/workstation/Sidebar.tsx"
$screens = Join-Path $root "web/research-agent-workstation/src/components/workstation/Screens.tsx"
$button = Join-Path $root "web/research-agent-workstation/src/components/ui/button.tsx"

function Read-Utf8([string]$Path) {
  return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

$sourceChecks = @()
foreach ($file in @($appShell, $sidebar, $screens, $button)) {
  $text = Read-Utf8 $file
  $sourceChecks += [pscustomobject]@{
    File = (Split-Path $file -Leaf)
    DataUiAction = ([regex]::Matches($text, "data-ui-action")).Count
    Buttons = ([regex]::Matches($text, "<Button|<button")).Count
    ImgTags = ([regex]::Matches($text, "<img\b")).Count
    HasOverlayMarker = [bool]($text -match "DesignFidelity|hasDesignFidelityAsset|design fidelity reference|image overlay")
  }
}

$routeChecks = @()
foreach ($page in $pages) {
  $response = Invoke-WebRequest -UseBasicParsing "$BaseUrl/?page=$page" -TimeoutSec 30
  $routeChecks += [pscustomobject]@{
    Page = $page
    Status = $response.StatusCode
    HasCss = [bool]($response.Content -like "*_next/static/css*")
    HasOldOverlay = [bool](
      $response.Content -like "*DesignFidelityPage*" -or
      $response.Content -like "*design fidelity reference*" -or
      $response.Content -like "*hasDesignFidelityAsset*"
    )
  }
}

$clickChecks = @()
if (-not $SkipClickSmoke) {
  $items = @(
    @{
      action = "ui_component_click"
      task_id = "playground_series_s6e6"
      metadata = @{
        page = "settings"
        component_type = "button"
        action_id = "save_settings_changes"
        label = "保存更改"
        disabled = $false
      }
    },
    @{
      action = "ui_component_click"
      task_id = "playground_series_s6e6"
      metadata = @{
        page = "literature"
        component_type = "button"
        action_id = "rag_build_agent_context"
        label = "构建 Agent Context"
        disabled = $false
      }
    },
    @{
      action = "ui_component_click"
      task_id = "playground_series_s6e6"
      metadata = @{
        page = "code"
        component_type = "button"
        action_id = "blocked_send_to_hpc"
        label = "Send to HPC"
        disabled = $true
      }
    }
  )

  foreach ($item in $items) {
    $json = $item | ConvertTo-Json -Depth 8
    $tmp = Join-Path $env:TEMP ("ui-click-" + $item.metadata.action_id + ".json")
    [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
    $raw = curl.exe -s -X POST "$BaseUrl/api/workstation-actions" -H "Content-Type: application/json" --data-binary "@$tmp"
    $parsed = $raw | ConvertFrom-Json
    $clickChecks += [pscustomobject]@{
      Action = $item.metadata.action_id
      Ok = $parsed.ok
      Status = $parsed.status
      Artifact = $parsed.artifact
    }
  }
}

$failures = @()
$failures += $sourceChecks | Where-Object { $_.HasOverlayMarker -or ($_.File -ne "button.tsx" -and $_.Buttons -gt 0 -and $_.DataUiAction -eq 0) }
$failures += $routeChecks | Where-Object { $_.Status -ne 200 -or -not $_.HasCss -or $_.HasOldOverlay }
if (-not $SkipClickSmoke) {
  $failures += $clickChecks | Where-Object { -not $_.Ok }
}

[pscustomobject]@{
  BaseUrl = $BaseUrl
  CheckedAt = (Get-Date).ToString("o")
  Pages = $pages.Count
  SourceChecks = $sourceChecks
  RouteChecks = $routeChecks
  ClickChecks = $clickChecks
  Passed = ($failures.Count -eq 0)
  FailureCount = $failures.Count
} | ConvertTo-Json -Depth 8

if ($failures.Count -gt 0) {
  exit 1
}
