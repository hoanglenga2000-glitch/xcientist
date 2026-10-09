$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$stage='C:\ProgramData\EvoMind\staging\ev-deepseek-v4-pro-20260908'
$result=Join-Path $stage 'service-output\key-install.json'
$bytes=$null
$secret=$null
try {
  . 'C:\ProgramData\EvoMind\bundle\scripts\lib\Common.ps1'
  Assert-DedicatedUserContext
  Add-Type -AssemblyName System.Security
  $paths=Get-NodeSecretPaths
  $destination=Join-Path $paths.root 'pezayo_ev_deepseek_v4_pro.xml'
  if(Test-Path -LiteralPath $destination){throw 'New model credential already exists'}
  $envelope=Get-Content -Raw -LiteralPath (Join-Path $stage 'model-key.machine-dpapi.json')|ConvertFrom-Json
  if($envelope.schema -ne 'evomind.model_key.machine_dpapi_transit.v1'){throw 'Wrong envelope'}
  $bytes=[Security.Cryptography.ProtectedData]::Unprotect([Convert]::FromBase64String($envelope.ciphertext),$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)
  $plain=[Text.Encoding]::UTF8.GetString($bytes)
  $secret=ConvertTo-SecureString $plain -AsPlainText -Force
  $plain=$null
  New-DpapiCredentialFile -Path $destination -UserName '__PEZAYO_DEEPSEEK_V4_PRO__' -Secret $secret
  Import-DpapiCredentialChecked -Path $destination -ExpectedUserName '__PEZAYO_DEEPSEEK_V4_PRO__'|Out-Null
  $receipt=@{status='installed';dpapi_scope='CurrentUser';service_user=(Get-CurrentShortUserName);old_credentials_unchanged=$true;secret_logged=$false}
  [IO.File]::WriteAllText($result,($receipt|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
}catch{[IO.File]::WriteAllText($result,(@{status='failed';error_type=$_.Exception.GetType().Name}|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false));exit 1}
finally{if($null -ne $bytes){[Array]::Clear($bytes,0,$bytes.Length)};if($null -ne $secret){$secret.Dispose()};$plain=$null}
