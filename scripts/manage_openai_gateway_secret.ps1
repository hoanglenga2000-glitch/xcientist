param(
  [ValidateSet("install-from-cockpit", "install-from-stdin", "optimize-profile", "status", "smoke", "benchmark-profiles", "evomind-tool-smoke", "mlebench-wave-plan", "mlebench-wave2-plan", "mlebench-medal-recovery-plan", "mlebench-recovery-deep-review", "mlebench-unscored-deep-review", "mlebench-jigsaw-audit", "mlebench-dog-audit", "mlebench-adaptive-loop")]
  [string]$Command = "status",
  [string]$CockpitKeyId = "",
  [string]$BaseUrl = "http://127.0.0.1:65068/v1",
  [string]$Model = "gpt-5.6-sol",
  [ValidateSet("low", "medium", "high", "xhigh", "max", "ultra")]
  [string]$InteractiveReasoningEffort = "low",
  [ValidateSet("low", "medium", "high", "xhigh", "max", "ultra")]
  [string]$ResearchReasoningEffort = "high",
  [ValidateSet("auto", "default", "flex", "priority")]
  [string]$ServiceTier = "priority"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "openai_api_key.xml"
$MetadataPath = Join-Path $StateDir "openai_gateway_metadata.json"
$CockpitManifest = Join-Path $env:USERPROFILE ".antigravity_cockpit\codex_local_access_sidecar\manifest.json"

function Assert-LocalGatewayUrl([string]$Value) {
  $uri = [Uri]$Value
  if ($uri.Scheme -ne "http" -or $uri.Host -notin @("127.0.0.1", "localhost") -or $uri.Port -ne 65068) {
    throw "The configured OpenAI gateway must be the Cockpit loopback endpoint on port 65068."
  }
  if ($uri.AbsolutePath.TrimEnd('/') -ne "/v1") {
    throw "The configured OpenAI gateway path must end at /v1."
  }
}

function Save-GatewayCredential([Security.SecureString]$SecureValue, [string]$Source) {
  if ($null -eq $SecureValue -or $SecureValue.Length -lt 20) {
    throw "The gateway API key is missing or too short."
  }
  New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
  [pscredential]::new("__OPENAI_API_KEY__", $SecureValue) |
    Export-Clixml -LiteralPath $CredentialPath
  $metadata = [ordered]@{
    provider = "openai"
    base_url = $BaseUrl.TrimEnd('/')
    model = $Model
    interactive_reasoning_effort = $InteractiveReasoningEffort
    research_reasoning_effort = $ResearchReasoningEffort
    service_tier = $ServiceTier
    source = $Source
    updated_at = (Get-Date).ToString("s")
  }
  Set-Content -LiteralPath $MetadataPath -Value ($metadata | ConvertTo-Json -Depth 4) -Encoding UTF8
  Write-Output ([ordered]@{
    status = "installed"
    credential_installed = (Test-Path -LiteralPath $CredentialPath)
    metadata_installed = (Test-Path -LiteralPath $MetadataPath)
    provider = "openai"
    base_url = $metadata.base_url
    model = $metadata.model
    interactive_reasoning_effort = $metadata.interactive_reasoning_effort
    research_reasoning_effort = $metadata.research_reasoning_effort
    service_tier = $metadata.service_tier
    source = $Source
  } | ConvertTo-Json -Depth 4)
}

function Enable-GatewayEnvironment {
  if (-not (Test-Path -LiteralPath $CredentialPath) -or -not (Test-Path -LiteralPath $MetadataPath)) {
    throw "The OpenAI-compatible gateway credential is not installed."
  }
  $credential = Import-Clixml -LiteralPath $CredentialPath
  $metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
  Assert-LocalGatewayUrl ([string]$metadata.base_url)
  $env:OPENAI_API_KEY = $credential.GetNetworkCredential().Password
  $env:OPENAI_BASE_URL = [string]$metadata.base_url
  $env:OPENAI_MODEL = [string]$metadata.model
  $env:OPENAI_REASONING_EFFORT = $(
    if ($metadata.interactive_reasoning_effort) {
      [string]$metadata.interactive_reasoning_effort
    } else {
      $InteractiveReasoningEffort
    }
  )
  $env:OPENAI_SERVICE_TIER = $(
    if ($metadata.service_tier) { [string]$metadata.service_tier } else { $ServiceTier }
  )
  $env:EVOLUTION_PRIMARY_PROVIDER = "openai"
  $env:EVOLUTION_PROVIDER_STRICT = "1"
  return $metadata
}

Assert-LocalGatewayUrl $BaseUrl
New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

if ($Command -eq "install-from-cockpit") {
  if (-not $CockpitKeyId) {
    throw "CockpitKeyId is required; key selection is fail-closed."
  }
  if (-not (Test-Path -LiteralPath $CockpitManifest)) {
    throw "Cockpit local-access manifest was not found."
  }
  $manifest = Get-Content -LiteralPath $CockpitManifest -Raw | ConvertFrom-Json
  $entry = @($manifest.apiKeys | Where-Object { $_.id -eq $CockpitKeyId -and $_.enabled }) | Select-Object -First 1
  if (-not $entry -or [string]::IsNullOrWhiteSpace([string]$entry.key)) {
    throw "The selected enabled Cockpit API key was not found."
  }
  $secure = ConvertTo-SecureString ([string]$entry.key) -AsPlainText -Force
  try {
    Save-GatewayCredential $secure "cockpit_manifest:$CockpitKeyId"
  } finally {
    $secure.Dispose()
    $entry = $null
    $manifest = $null
  }
  exit 0
}

if ($Command -eq "install-from-stdin") {
  $secure = Read-Host "OpenAI-compatible gateway API key" -AsSecureString
  try {
    Save-GatewayCredential $secure "interactive_stdin"
  } finally {
    if ($null -ne $secure) { $secure.Dispose() }
  }
  exit 0
}

if ($Command -eq "optimize-profile") {
  if (-not (Test-Path -LiteralPath $CredentialPath) -or -not (Test-Path -LiteralPath $MetadataPath)) {
    throw "The OpenAI-compatible gateway credential is not installed."
  }
  $metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
  Assert-LocalGatewayUrl ([string]$metadata.base_url)
  $metadata | Add-Member -NotePropertyName interactive_reasoning_effort -NotePropertyValue $InteractiveReasoningEffort -Force
  $metadata | Add-Member -NotePropertyName research_reasoning_effort -NotePropertyValue $ResearchReasoningEffort -Force
  $metadata | Add-Member -NotePropertyName service_tier -NotePropertyValue $ServiceTier -Force
  $metadata | Add-Member -NotePropertyName optimized_at -NotePropertyValue (Get-Date).ToString("s") -Force
  Set-Content -LiteralPath $MetadataPath -Value ($metadata | ConvertTo-Json -Depth 4) -Encoding UTF8
  Write-Output ([ordered]@{
    status = "optimized"
    provider = $metadata.provider
    base_url = $metadata.base_url
    model = $metadata.model
    interactive_reasoning_effort = $metadata.interactive_reasoning_effort
    research_reasoning_effort = $metadata.research_reasoning_effort
    service_tier = $metadata.service_tier
    credential_modified = $false
  } | ConvertTo-Json -Depth 4)
  exit 0
}

if ($Command -eq "status") {
  $metadata = $null
  if (Test-Path -LiteralPath $MetadataPath) {
    $metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
  }
  Write-Output ([ordered]@{
    status = $(if ((Test-Path -LiteralPath $CredentialPath) -and $metadata) { "configured" } else { "not_configured" })
    credential_installed = (Test-Path -LiteralPath $CredentialPath)
    metadata_installed = (Test-Path -LiteralPath $MetadataPath)
    provider = $(if ($metadata) { $metadata.provider } else { "openai" })
    base_url = $(if ($metadata) { $metadata.base_url } else { $BaseUrl })
    model = $(if ($metadata) { $metadata.model } else { $Model })
    interactive_reasoning_effort = $(if ($metadata -and $metadata.interactive_reasoning_effort) { $metadata.interactive_reasoning_effort } else { $InteractiveReasoningEffort })
    research_reasoning_effort = $(if ($metadata -and $metadata.research_reasoning_effort) { $metadata.research_reasoning_effort } else { $ResearchReasoningEffort })
    service_tier = $(if ($metadata -and $metadata.service_tier) { $metadata.service_tier } else { $ServiceTier })
  } | ConvertTo-Json -Depth 4)
  exit 0
}

$activeMetadata = Enable-GatewayEnvironment
if ($Command -like "mlebench-*") {
  $env:OPENAI_REASONING_EFFORT = $(
    if ($activeMetadata.research_reasoning_effort) {
      [string]$activeMetadata.research_reasoning_effort
    } else {
      $ResearchReasoningEffort
    }
  )
}
$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  $python = "python"
}
if ($Command -eq "evomind-tool-smoke") {
  $output = Join-Path $Root "workspace\llm\evomind_native_tool_loop_current.json"
  & $python (Join-Path $Root "scripts\verify_evomind_native_tool_loop.py") --workspace $Root --output $output
} elseif ($Command -eq "benchmark-profiles") {
  $output = Join-Path $Root "workspace\llm\gpt56_performance_profiles_current.json"
  & $python (Join-Path $Root "scripts\benchmark_openai_gateway_profiles.py") --output $output
} elseif ($Command -eq "mlebench-wave-plan") {
  $output = Join-Path $Root "workspace\mlebench_plans\wave1_gpt56_current.json"
  & $python (Join-Path $Root "scripts\generate_mlebench_wave_plan.py") --workspace $Root --output $output
} elseif ($Command -eq "mlebench-wave2-plan") {
  $output = Join-Path $Root "workspace\mlebench_plans\wave2_gpt56_current.json"
  $progress = Join-Path $Root "workspace\mlebench_progress\lite11_current.json"
  & $python (Join-Path $Root "scripts\generate_mlebench_wave_plan.py") `
    --workspace $Root --wave Wave2 --progress-report $progress --output $output
} elseif ($Command -eq "mlebench-medal-recovery-plan") {
  $output = Join-Path $Root "workspace\mlebench_plans\medal_recovery_gpt56_current.json"
  $progress = Join-Path $Root "workspace\mlebench_progress\lite11_current.json"
  $thresholds = Join-Path $Root "workspace\mlebench_progress\wave2_thresholds.json"
  & $python (Join-Path $Root "scripts\generate_mlebench_medal_recovery_plan.py") `
    --progress $progress --wave2-thresholds $thresholds --output $output
} elseif ($Command -eq "mlebench-recovery-deep-review") {
  $output = Join-Path $Root "workspace\mlebench_plans\medal_recovery_deep_gpt56_review_current.json"
  & $python (Join-Path $Root "scripts\generate_mlebench_recovery_deep_review.py") --output $output
} elseif ($Command -eq "mlebench-unscored-deep-review") {
  $output = Join-Path $Root "workspace\mlebench_plans\unscored8_deep_gpt56_review_current.json"
  & $python (Join-Path $Root "scripts\generate_mlebench_unscored_deep_review.py") --output $output
} elseif ($Command -eq "mlebench-jigsaw-audit") {
  $output = Join-Path $Root "workspace\mlebench_plans\jigsaw_gpt56_audit_current.json"
  & $python (Join-Path $Root "scripts\generate_jigsaw_gpt56_audit.py") --output $output
} elseif ($Command -eq "mlebench-dog-audit") {
  $output = Join-Path $Root "workspace\mlebench_plans\dog_breed_gpt56_audit_current.json"
  & $python (Join-Path $Root "scripts\generate_dog_gpt56_audit.py") --output $output
} elseif ($Command -eq "mlebench-adaptive-loop") {
  $output = Join-Path $Root "workspace\llm\mlebench_gpt56_adaptive_loop_current.json"
  & $python (Join-Path $Root "scripts\run_mlebench_gpt56_adaptive_loop.py") `
    --workspace $Root --output $output
} else {
  $output = Join-Path $Root "workspace\llm\openai_gateway_smoke_current.json"
  & $python (Join-Path $Root "scripts\verify_openai_gateway.py") --output $output
}
exit $LASTEXITCODE
