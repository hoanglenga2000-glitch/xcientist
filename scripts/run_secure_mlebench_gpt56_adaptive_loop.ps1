param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ControllerArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$StateDir = Join-Path $env:APPDATA "ResearchAgentWorkstation"
$CredentialPath = Join-Path $StateDir "openai_api_key.xml"
$MetadataPath = Join-Path $StateDir "openai_gateway_metadata.json"
$ExpectedBaseUrl = "http://127.0.0.1:65068/v1"
$ExpectedModel = "gpt-5.6-sol"
$Python = Join-Path $Root "workspace\release-venv\Scripts\python.exe"
$Controller = Join-Path $Root "scripts\run_mlebench_gpt56_adaptive_loop.py"

foreach ($path in @($CredentialPath, $MetadataPath, $Python, $Controller)) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Required secure adaptive-loop dependency is missing: $path"
  }
}

$metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
$baseUrl = ([string]$metadata.base_url).TrimEnd('/')
$model = [string]$metadata.model
if ($baseUrl -ne $ExpectedBaseUrl) {
  throw "Secure gateway metadata does not match the loopback endpoint."
}
if ($model -ne $ExpectedModel) {
  throw "Secure gateway metadata does not match GPT-5.6."
}

$credential = Import-Clixml -LiteralPath $CredentialPath
$secret = $credential.GetNetworkCredential().Password
if ([string]::IsNullOrWhiteSpace($secret)) {
  throw "The DPAPI-protected gateway credential is empty."
}

$previousKey = $env:OPENAI_API_KEY
$previousBaseUrl = $env:OPENAI_BASE_URL
$previousModel = $env:OPENAI_MODEL
$previousProvider = $env:EVOLUTION_PRIMARY_PROVIDER
$previousStrict = $env:EVOLUTION_PROVIDER_STRICT
try {
  $env:OPENAI_API_KEY = $secret
  $env:OPENAI_BASE_URL = $baseUrl
  $env:OPENAI_MODEL = $model
  $env:EVOLUTION_PRIMARY_PROVIDER = "openai"
  $env:EVOLUTION_PROVIDER_STRICT = "1"
  & $Python $Controller @ControllerArgs
  if ($LASTEXITCODE -ne 0) {
    throw "GPT-5.6 adaptive controller exited with code $LASTEXITCODE."
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
  if ($null -eq $previousProvider) {
    Remove-Item Env:EVOLUTION_PRIMARY_PROVIDER -ErrorAction SilentlyContinue
  } else {
    $env:EVOLUTION_PRIMARY_PROVIDER = $previousProvider
  }
  if ($null -eq $previousStrict) {
    Remove-Item Env:EVOLUTION_PROVIDER_STRICT -ErrorAction SilentlyContinue
  } else {
    $env:EVOLUTION_PROVIDER_STRICT = $previousStrict
  }
}
