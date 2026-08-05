param(
  [int]$TimeoutSeconds = 180,
  [string[]]$Case = @(),
  [string]$Output = "workspace\evaluation\assistant_novice_quality_gpt56_current.json",
  [string]$MarkdownOutput = "workspace\evaluation\assistant_novice_quality_gpt56_current.md"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$managedSecrets = Join-Path $env:APPDATA "EvoMind\secrets"
$legacySecrets = Join-Path $env:APPDATA "ResearchAgentWorkstation"

function Resolve-NewestFile([string]$Name) {
  $candidates = @(
    (Join-Path $managedSecrets $Name),
    (Join-Path $legacySecrets $Name)
  ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
  if ($candidates.Count -eq 0) { return $null }
  return $candidates |
    Sort-Object @{ Expression = { (Get-Item -LiteralPath $_).LastWriteTimeUtc }; Descending = $true } |
    Select-Object -First 1
}

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

$env:OPENAI_API_KEY = $credential.GetNetworkCredential().Password
$env:OPENAI_BASE_URL = $baseUrl
$env:OPENAI_MODEL = "gpt-5.6-sol"
$env:EVOLUTION_PRIMARY_PROVIDER = "openai"
$env:EVOLUTION_PROVIDER_STRICT = "1"

$arguments = @(
  "run", "python", "scripts/evaluate_evomind_novice_agent.py",
  "--live", "--strict", "--timeout", [string]$TimeoutSeconds,
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
  Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
}
