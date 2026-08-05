param(
  [Parameter(Mandatory = $true)]
  [string]$PromptOutput,
  [Parameter(Mandatory = $true)]
  [string]$ReviewOutput,
  [int]$TimeoutSeconds = 900,
  [int]$MaxTokens = 30000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "openai_api_key.xml"
$MetadataPath = Join-Path $StateDir "openai_gateway_metadata.json"
$ExpectedBaseUrl = "http://127.0.0.1:65068/v1"
$ExpectedModel = "gpt-5.6-sol"
$Python = Join-Path $Root "workspace\release-venv\Scripts\python.exe"
$Generator = Join-Path $Root "scripts\generate_mlebench_recovery_deep_review.py"

foreach ($path in @($CredentialPath, $MetadataPath, $Python, $Generator)) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required secure review dependency is missing: $path"
  }
}

$metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
$baseUrl = ([string]$metadata.base_url).TrimEnd('/')
$model = [string]$metadata.model
if ($baseUrl -ne $ExpectedBaseUrl) {
  throw "Secure gateway metadata does not match the required loopback endpoint."
}
if ($model -ne $ExpectedModel) {
  throw "Secure gateway metadata does not match the required GPT-5.6 model."
}

$credential = Import-Clixml -LiteralPath $CredentialPath
$secret = $credential.GetNetworkCredential().Password
if ([string]::IsNullOrWhiteSpace($secret)) {
  throw "The DPAPI-protected gateway credential is empty."
}

$previousKey = $env:OPENAI_API_KEY
$previousBaseUrl = $env:OPENAI_BASE_URL
$previousModel = $env:OPENAI_MODEL
try {
  $env:OPENAI_API_KEY = $secret
  $env:OPENAI_BASE_URL = $baseUrl
  $env:OPENAI_MODEL = $model
  & $Python $Generator `
    --prompt-output $PromptOutput `
    --output $ReviewOutput `
    --base-url $baseUrl `
    --model $model `
    --timeout $TimeoutSeconds `
    --max-tokens $MaxTokens
  if ($LASTEXITCODE -ne 0) {
    throw "GPT-5.6 recovery review exited with code $LASTEXITCODE."
  }
} finally {
  $secret = $null
  if ($null -eq $previousKey) {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
  } else {
    $env:OPENAI_API_KEY = $previousKey
  }
  if ($null -eq $previousBaseUrl) {
    Remove-Item Env:OPENAI_BASE_URL -ErrorAction SilentlyContinue
  } else {
    $env:OPENAI_BASE_URL = $previousBaseUrl
  }
  if ($null -eq $previousModel) {
    Remove-Item Env:OPENAI_MODEL -ErrorAction SilentlyContinue
  } else {
    $env:OPENAI_MODEL = $previousModel
  }
}
