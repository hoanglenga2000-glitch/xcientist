param(
  [int]$TimeoutSeconds = 180,
  [string[]]$Case = @(),
  [string]$Suite = "configs\evaluation\assistant_novice_v1.json",
  [string]$Output = "workspace\evaluation\assistant_novice_quality_gpt56_current.json",
  [string]$MarkdownOutput = "workspace\evaluation\assistant_novice_quality_gpt56_current.md"
)

# Runs the novice quality gate against the model route pinned by the suite.
# The suite's required_route must match the route the production web node
# uses (bundle\scripts\Start-Node.ps1), so the gate measures what users get.
# Credentials are read from the current user's DPAPI store and only exported
# to this process; nothing is printed or written to disk.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$managedSecrets = Join-Path $env:APPDATA "EvoMind\secrets"
$legacySecrets = Join-Path $env:APPDATA "ResearchAgentWorkstation"

function Resolve-NewestFile([string]$Name) {
  $candidates = @(
    (Join-Path $managedSecrets $Name),
    (Join-Path $legacySecrets $Name)
  ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
  if (@($candidates).Count -eq 0) { return $null }
  return $candidates |
    Sort-Object @{ Expression = { (Get-Item -LiteralPath $_).LastWriteTimeUtc }; Descending = $true } |
    Select-Object -First 1
}

function Import-RouteCredential([object[]]$Candidates) {
  foreach ($candidate in $Candidates) {
    $path = Resolve-NewestFile ([string]$candidate.file)
    if (-not $path) { continue }
    $credential = Import-Clixml -LiteralPath $path
    if ($credential -isnot [System.Management.Automation.PSCredential] -or $credential.UserName -ne [string]$candidate.user) {
      throw "Model route credential has an unexpected structure: $([string]$candidate.file)"
    }
    return $credential
  }
  throw "No DPAPI credential is available for the suite's model route."
}

$suitePath = if ([IO.Path]::IsPathRooted($Suite)) { $Suite } else { Join-Path $Root $Suite }
$suiteJson = Get-Content -LiteralPath $suitePath -Raw -Encoding UTF8 | ConvertFrom-Json
$provider = [string]$suiteJson.required_provider
$model = [string]$suiteJson.required_model
$route = $suiteJson.required_route
if ($provider -notin @("openai", "anthropic") -or -not $model) {
  throw "Quality suite must pin openai or anthropic provider and a required model."
}

$restore = @{}
function Set-ScopedEnv([string]$Name, [string]$Value) {
  if (-not $restore.ContainsKey($Name)) { $restore[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process") }
  [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

if ($null -ne $route -and [string]$route.base_url) {
  $baseUrl = ([string]$route.base_url).TrimEnd("/")
  $uri = [Uri]$baseUrl
  if ($uri.Scheme -ne "https") {
    throw "Quality suite route must be an https endpoint."
  }
  $credential = Import-RouteCredential @($route.credential_candidates)
  $secret = $credential.GetNetworkCredential().Password
  if ($provider -eq "anthropic") {
    # Anthropic transport appends /v1/messages to ANTHROPIC_BASE_URL.
    $anthropicRoot = $baseUrl
    if ($uri.AbsolutePath.TrimEnd("/") -eq "/v1") {
      $anthropicRoot = ($uri.GetLeftPart([UriPartial]::Authority)).TrimEnd("/")
    } elseif ($uri.AbsolutePath.TrimEnd("/") -notin @("", "/")) {
      throw "Anthropic quality suite route must be https://host or https://host/v1."
    }
    Set-ScopedEnv "ANTHROPIC_API_KEY" $secret
    Set-ScopedEnv "ANTHROPIC_BASE_URL" $anthropicRoot
    Set-ScopedEnv "CLAUDE_CODE_MODEL" $model
    Set-ScopedEnv "LLM_PROVIDER" "anthropic"
    Set-ScopedEnv "EVOMIND_MODEL_WIRE_PROTOCOL" $(if ([string]$route.wire_protocol) { [string]$route.wire_protocol } else { "anthropic_messages" })
  } else {
    if ($uri.AbsolutePath.TrimEnd("/") -ne "/v1") {
      throw "OpenAI-compatible quality suite route must be an https /v1 endpoint."
    }
    Set-ScopedEnv "OPENAI_API_KEY" $secret
    Set-ScopedEnv "DEEPSEEK_API_KEY" $secret
    Set-ScopedEnv "OPENAI_BASE_URL" $baseUrl
    Set-ScopedEnv "DEEPSEEK_BASE_URL" $baseUrl
    Set-ScopedEnv "DEEPSEEK_MODEL" $model
    Set-ScopedEnv "OPENAI_REASONING_EFFORT" ""
    Set-ScopedEnv "OPENAI_SERVICE_TIER" ""
    Set-ScopedEnv "LLM_PROVIDER" "openai"
    Set-ScopedEnv "EVOMIND_MODEL_WIRE_PROTOCOL" ([string]$route.wire_protocol)
  }
  Set-ScopedEnv "EVOMIND_MODEL_TIMEOUT_SECONDS" ([string]$route.timeout_seconds)
  $noProxy = [Environment]::GetEnvironmentVariable("NO_PROXY", "Process")
  $hosts = @($noProxy -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
  if ($hosts -notcontains "*" -and $hosts -notcontains $uri.Host) {
    Set-ScopedEnv "NO_PROXY" ((@($hosts) + $uri.Host) -join ",")
  }
} elseif ($provider -eq "anthropic") {
  throw "Anthropic quality suite requires required_route.base_url and credential_candidates."
} else {
  $credentialPath = Resolve-NewestFile "openai_api_key.xml"
  $metadataPath = Resolve-NewestFile "openai_gateway_metadata.json"
  if (-not $credentialPath -or -not $metadataPath) {
    throw "OpenAI loopback gateway DPAPI profile is incomplete."
  }
  $credential = Import-Clixml -LiteralPath $credentialPath
  $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
  $baseUrl = ([string]$metadata.base_url).TrimEnd("/")
  $uri = [Uri]$baseUrl
  if ($uri.Scheme -ne "http" -or $uri.Host -notin @("127.0.0.1", "localhost", "::1") -or $uri.Port -ne 65068 -or $uri.AbsolutePath.TrimEnd("/") -ne "/v1") {
    throw "OpenAI gateway metadata is not bound to the approved loopback endpoint."
  }
  Set-ScopedEnv "OPENAI_API_KEY" $credential.GetNetworkCredential().Password
  Set-ScopedEnv "OPENAI_BASE_URL" $baseUrl
}
if ($provider -eq "openai") {
  Set-ScopedEnv "OPENAI_MODEL" $model
}
Set-ScopedEnv "EVOLUTION_PRIMARY_PROVIDER" $provider
Set-ScopedEnv "EVOLUTION_PROVIDER_STRICT" "1"

$arguments = @(
  "run", "python", "scripts/evaluate_evomind_novice_agent.py",
  "--live", "--strict", "--timeout", [string]$TimeoutSeconds,
  "--suite", $suitePath,
  "--output", $Output,
  "--markdown-output", $MarkdownOutput
)
foreach ($caseId in $Case) {
  $arguments += @("--case", $caseId)
}

Push-Location $Root
try {
  & uv @arguments
  exit $LASTEXITCODE
} finally {
  Pop-Location
  foreach ($name in $restore.Keys) {
    [Environment]::SetEnvironmentVariable($name, $restore[$name], "Process")
  }
}
