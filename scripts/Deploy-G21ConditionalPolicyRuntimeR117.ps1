param(
  [Parameter(Mandatory)][string]$ApprovalPath,
  [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$approvalSchema = 'evomind.g21_conditional_policy_runtime_deployment_approval.r117.v1'
$releaseSchema = 'evomind.g21_conditional_policy_runtime.r117.v1'
$receiptSchema = 'evomind.g21_conditional_policy_runtime_test_receipt.r117.v1'
$fixedRun = 'run_7b1efb878afb40f396db431e91f093a5'
$fixedAllocation = 'G21'
$fixedWebBuild = 'overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'
$baselineZipSha = '2aa31006f33bb53de56afe6888c9eb03bc8597fa5334e3831e6fd788fda877c2'
$baselineTreeSha = '7e7b0babec6146640dc56cb42c05bbe37765bca2aca6348f42c5bf569ab69b38'
$oldSpecRawSha = '380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f'
$oldSpecCanonicalSha = 'a569bae731723fc7046817df737876d6dc7674eadd9f7b812fe7b4a87e521655'
$newSpecRawSha = 'da8c3fbe8b14107dfadf96ad906b4cf982bc714ebf6420225a1e3722cd41a752'
$newSpecCanonicalSha = 'fbb9939dff28c355c2798bf32fce104845337064f42f9c280f2b6906cccf4036'
$evidenceRawSha = '47df860192502afcaacd7b1b6bc7fc4d86110c2e665278c8b7b3ea421893b7a7'
$evidenceCanonicalSha = 'f81701f13aeebef146659e091cacfc4e39ae18df345ee6b58be2ad5ef40829f0'
$migrationId = 'g21-conditional-strong-baseline-v2'
$hostAlias = 'evomind-shanghai'
$runnerName = 'Deploy-G21ConditionalPolicyRuntimeR117Remote.ps1'
$runner = Join-Path $PSScriptRoot $runnerName

function Sha([string]$Path){
  $stream=[IO.File]::OpenRead($Path);$hash=[Security.Cryptography.SHA256]::Create()
  try{([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}finally{$stream.Dispose();$hash.Dispose()}
}
function Read-Object([string]$Path,[string]$Code){
  if(-not(Test-Path -LiteralPath $Path -PathType Leaf)){throw "$Code`:MISSING"}
  try{$value=Get-Content -LiteralPath $Path -Raw -Encoding UTF8|ConvertFrom-Json}catch{throw "$Code`:INVALID_JSON"}
  if($null-eq$value-or$value-is[array]){throw "$Code`:OBJECT_REQUIRED"};$value
}
function Assert-Sha([object]$Value,[string]$Code){if([string]$Value-cnotmatch'^[a-f0-9]{64}$'){throw "$Code`:SHA_REJECTED"}}
function Resolve-BoundInput([string]$Directory,[object]$Entry,[string]$Code){
  if($null-eq$Entry){throw "$Code`:ENTRY_MISSING"};$fileName=[string]$Entry.filename
  if($fileName-cnotmatch'^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$'-or[IO.Path]::GetFileName($fileName)-cne$fileName){throw "$Code`:FILENAME_REJECTED"}
  Assert-Sha $Entry.sha256 $Code;$path=[IO.Path]::GetFullPath((Join-Path $Directory $fileName))
  if([IO.Path]::GetDirectoryName($path).TrimEnd('\')-cne[IO.Path]::GetFullPath($Directory).TrimEnd('\')){throw "$Code`:PATH_ESCAPE"}
  if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "$Code`:MISSING"};if((Sha $path)-cne[string]$Entry.sha256){throw "$Code`:HASH_REJECTED"}
  [ordered]@{path=$path;filename=$fileName;sha256=[string]$Entry.sha256}
}
function Read-ZipManifest([string]$Path){
  Add-Type -AssemblyName System.IO.Compression.FileSystem;$archive=[IO.Compression.ZipFile]::OpenRead($Path)
  try{$seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase);$runtime=@();$manifest=@()
    foreach($entry in $archive.Entries){$name=([string]$entry.FullName).Replace('\','/')
      if(-not$name-or$name.StartsWith('/')-or$name-match'(^|/)\.\.(/|$)|^[A-Za-z]:'-or-not$seen.Add($name)){throw "R117_ZIP_UNSAFE:$name"}
      if($name-ceq'goal-runtime-overlay-manifest.json'-or$name-ceq'runtime-hotfix-manifest.json'){throw'R117_STALE_MANIFEST_REJECTED'}
      if($name-ceq'conditional-policy-runtime-manifest.json'){$manifest+=$entry}elseif($name-cmatch'^evomind_runtime/[A-Za-z0-9_]+\.py$'){$runtime+=$entry}elseif(-not$name.EndsWith('/')){throw "R117_ZIP_EXTRA_REJECTED:$name"}
    }
    if($runtime.Count-ne32-or$manifest.Count-ne1){throw'R117_ZIP_FILESET_REJECTED'}
    $reader=[IO.StreamReader]::new($manifest[0].Open(),[Text.Encoding]::UTF8,$true);try{$raw=$reader.ReadToEnd()}finally{$reader.Dispose()}
    try{$value=$raw|ConvertFrom-Json}catch{throw'R117_EMBEDDED_MANIFEST_JSON_REJECTED'}
    [ordered]@{value=$value;raw_sha256=([BitConverter]::ToString(([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($raw)))).Replace('-','').ToLowerInvariant()}
  }finally{$archive.Dispose()}
}
function Read-Approval([string]$Path){
  $full=[IO.Path]::GetFullPath($Path);$approval=Read-Object $full 'R117_APPROVAL';$directory=[IO.Path]::GetDirectoryName($full)
  if([string]$approval.schema-cne$approvalSchema-or[string]$approval.status-cne'approved_for_single_runtime_transaction'-or$approval.one_time-isnot[bool]-or-not[bool]$approval.one_time-or
    [string]$approval.deployment_id-cnotmatch'^[a-f0-9]{32}$'-or[string]$approval.run_id-cne$fixedRun-or[string]$approval.allocation-cne$fixedAllocation-or[string]$approval.web_build_id-cne$fixedWebBuild-or
    -not[bool]$approval.preflight.run_terminal_required-or[int]$approval.preflight.effective_active_required-ne0-or[int]$approval.preflight.pending_approvals_required-ne0-or[int]$approval.preflight.current_runtime_file_count-ne32-or
    [string]$approval.preflight.current_runtime_tree_sha256-cne$baselineTreeSha-or-not[bool]$approval.preflight.no_concurrent_transaction-or
    [string]$approval.baseline.zip_sha256-cne$baselineZipSha-or[string]$approval.baseline.runtime_tree_sha256-cne$baselineTreeSha-or[int]$approval.baseline.runtime_file_count-ne32-or
    [int]$approval.candidate.runtime_file_count-ne32-or[int]$approval.candidate.changed_file_count-ne5-or[int]$approval.candidate.unchanged_file_count-ne27-or
    [string]$approval.policy_migration.migration_id-cne$migrationId-or[string]$approval.policy_migration.old_spec_file_sha256-cne$oldSpecRawSha-or[string]$approval.policy_migration.old_spec_canonical_sha256-cne$oldSpecCanonicalSha-or
    [string]$approval.policy_migration.new_spec_file_sha256-cne$newSpecRawSha-or[string]$approval.policy_migration.new_spec_canonical_sha256-cne$newSpecCanonicalSha-or
    [string]$approval.policy_migration.policy_evidence_file_sha256-cne$evidenceRawSha-or[string]$approval.policy_migration.policy_evidence_canonical_sha256-cne$evidenceCanonicalSha-or
    -not[bool]$approval.policy_migration.managed_put_only-or-not[bool]$approval.policy_migration.isolated_body_required-or-not[bool]$approval.policy_migration.idempotent_replay_required-or
    [string]$approval.policy_migration.event_type-cne'goal.policy_migrated'-or[int]$approval.policy_migration.exact_event_count-ne1-or
    [string]$approval.deployer.service_action_path-cne'C:\SecureInput\Invoke-ServiceAccountAction.ps1'-or-not[bool]$approval.deployer.direct_scheduled_task_start_forbidden-or-not[bool]$approval.deployer.direct_process_kill_forbidden-or-not[bool]$approval.deployer.direct_sqlite_forbidden-or
    -not[bool]$approval.boundaries.runtime_only-or-not[bool]$approval.boundaries.web_unchanged-or-not[bool]$approval.boundaries.hpc_access_forbidden-or-not[bool]$approval.boundaries.gpu_touch_forbidden-or-not[bool]$approval.boundaries.weather_forbidden-or-not[bool]$approval.boundaries.training_forbidden_during_deployment){throw'R117_APPROVAL_CONTRACT_REJECTED'}
  try{$created=[DateTimeOffset]::Parse([string]$approval.created_at);$expires=[DateTimeOffset]::Parse([string]$approval.expires_at)}catch{throw'R117_APPROVAL_TIME_REJECTED'}
  $now=[DateTimeOffset]::UtcNow;if($expires-le$created-or$now-lt$created.AddMinutes(-5)-or$now-ge$expires-or($expires-$created).TotalMinutes-gt60){throw'R117_APPROVAL_EXPIRED_OR_TIME_REJECTED'}
  $names=@('candidate','source_manifest','build_result','test_receipt','old_spec','new_spec','policy_evidence','verifier','wrapper','remote_runner');$inputs=[ordered]@{}
  foreach($name in $names){$inputs[$name]=Resolve-BoundInput $directory $approval.bound_inputs.$name "R117_$($name.ToUpperInvariant())"}
  if($inputs.wrapper.filename-cne[IO.Path]::GetFileName($PSCommandPath)-or(Sha $PSCommandPath)-cne$inputs.wrapper.sha256){throw'R117_WRAPPER_HASH_REJECTED'}
  if($inputs.remote_runner.filename-cne$runnerName-or(Sha $runner)-cne$inputs.remote_runner.sha256){throw'R117_REMOTE_RUNNER_HASH_REJECTED'}
  if($inputs.candidate.sha256-cne[string]$approval.candidate.sha256-or$inputs.source_manifest.sha256-cne[string]$approval.candidate.source_manifest_sha256-or$inputs.build_result.sha256-cne[string]$approval.candidate.build_result_sha256-or
    $inputs.test_receipt.sha256-cne[string]$approval.verification.test_receipt_sha256-or$inputs.verifier.sha256-cne[string]$approval.verification.candidate_verifier_sha256){throw'R117_BOUND_INPUT_LINK_REJECTED'}
  $envelope=Read-ZipManifest $inputs.candidate.path;$manifest=Read-Object $inputs.source_manifest.path 'R117_SOURCE_MANIFEST';$build=Read-Object $inputs.build_result.path 'R117_BUILD_RESULT';$receipt=Read-Object $inputs.test_receipt.path 'R117_TEST_RECEIPT'
  if((Sha $inputs.source_manifest.path)-cne$envelope.raw_sha256){throw'R117_EMBEDDED_MANIFEST_BINDING_REJECTED'}
  if([string]$manifest.schema-cne$releaseSchema-or[string]$manifest.run_id-cne$fixedRun-or[string]$manifest.allocation-cne$fixedAllocation-or[string]$manifest.baseline_zip_sha256-cne$baselineZipSha-or[string]$manifest.baseline_runtime_tree_sha256-cne$baselineTreeSha-or
    [int]$manifest.candidate_runtime_file_count-ne32-or[int]$manifest.changed_file_count-ne5-or[int]$manifest.unchanged_file_count-ne27-or[string]$manifest.candidate_runtime_tree_sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256-or
    [string]$manifest.old_goal_spec_file_sha256-cne$oldSpecRawSha-or[string]$manifest.old_goal_spec_canonical_sha256-cne$oldSpecCanonicalSha-or[string]$manifest.new_goal_spec_file_sha256-cne$newSpecRawSha-or[string]$manifest.new_goal_spec_canonical_sha256-cne$newSpecCanonicalSha-or
    [string]$manifest.policy_evidence_file_sha256-cne$evidenceRawSha-or[string]$manifest.policy_evidence_canonical_sha256-cne$evidenceCanonicalSha-or[bool]$manifest.production_deployable-or[bool]$manifest.production_deployed-or
    [string]$build.schema-cne$releaseSchema-or[string]$build.status-cne'built_and_verified'-or[string]$build.zip_sha256-cne$inputs.candidate.sha256-or[string]$build.source_manifest_sha256-cne$inputs.source_manifest.sha256-or[string]$build.candidate_runtime_tree_sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256-or
    [bool]$build.production_deployed-or[bool]$build.hpc_accessed-or[bool]$build.gpu_touched-or[int]$build.remote_writes-ne0-or
    [string]$receipt.schema-cne$receiptSchema-or[string]$receipt.status-cne'verified'-or[string]$receipt.candidate_zip_sha256-cne$inputs.candidate.sha256-or[string]$receipt.source_manifest_sha256-cne$inputs.source_manifest.sha256-or[string]$receipt.build_result_sha256-cne$inputs.build_result.sha256-or[string]$receipt.candidate_runtime_tree_sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256-or
    @($receipt.verification_rounds).Count-ne2-or[bool]$receipt.production_deployed-or[bool]$receipt.hpc_accessed-or[bool]$receipt.gpu_touched-or[int]$receipt.remote_writes-ne0){throw'R117_RELEASE_BINDING_REJECTED'}
  [ordered]@{path=$full;directory=$directory;approval=$approval;inputs=$inputs}
}

$bound=Read-Approval $ApprovalPath;$approval=$bound.approval;$approvalSha=Sha $bound.path;$transactionId=[string]$approval.deployment_id
if($ValidateOnly){[ordered]@{schema=$approvalSchema;status='validated_not_deployed';deployment_id=$transactionId;run_id=$fixedRun;allocation=$fixedAllocation;approval_sha256=$approvalSha;candidate_sha256=$bound.inputs.candidate.sha256;candidate_tree_sha256=[string]$approval.candidate.installed_runtime_tree_sha256;hpc_accessed=$false;gpu_touched=$false;remote_writes=0;production_action_performed=$false}|ConvertTo-Json -Compress;exit 0}
$output=Join-Path $bound.directory "r117-conditional-policy-production-$transactionId.json";$errorOutput=Join-Path $bound.directory "r117-conditional-policy-production-$transactionId.error.json"
if((Test-Path -LiteralPath $output)-or(Test-Path -LiteralPath $errorOutput)){throw'R117_DEPLOY_OUTPUT_EXISTS'}
$remote="C:\SecureInput\r117-conditional-policy-$transactionId";$create="if(Test-Path -LiteralPath '$remote'){throw 'R117_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -;if($LASTEXITCODE-ne0){throw'R117_REMOTE_CREATE_FAILED'}
$uploads=@(
  [pscustomobject]@{path=$bound.path;name='approval.json';sha=$approvalSha},
  [pscustomobject]@{path=$bound.inputs.candidate.path;name='runtime.zip';sha=$bound.inputs.candidate.sha256},
  [pscustomobject]@{path=$bound.inputs.source_manifest.path;name='source-manifest.json';sha=$bound.inputs.source_manifest.sha256},
  [pscustomobject]@{path=$bound.inputs.build_result.path;name='build-result.json';sha=$bound.inputs.build_result.sha256},
  [pscustomobject]@{path=$bound.inputs.test_receipt.path;name='test-receipt.json';sha=$bound.inputs.test_receipt.sha256},
  [pscustomobject]@{path=$bound.inputs.old_spec.path;name='old-spec.json';sha=$bound.inputs.old_spec.sha256},
  [pscustomobject]@{path=$bound.inputs.new_spec.path;name='new-spec.json';sha=$bound.inputs.new_spec.sha256},
  [pscustomobject]@{path=$bound.inputs.policy_evidence.path;name='policy-evidence.json';sha=$bound.inputs.policy_evidence.sha256},
  [pscustomobject]@{path=$bound.inputs.remote_runner.path;name=$runnerName;sha=$bound.inputs.remote_runner.sha256}
)
foreach($item in $uploads){scp $item.path "${hostAlias}:$remote\$($item.name)";if($LASTEXITCODE-ne0){throw "R117_UPLOAD_FAILED:$($item.name)"}}
$remoteChecks=@($uploads|ForEach-Object{"[pscustomobject]@{name='$($_.name)';sha='$($_.sha)'}"})-join',';$verify="function sh([string]`$p){`$s=[IO.File]::OpenRead(`$p);`$h=[Security.Cryptography.SHA256]::Create();try{([BitConverter]::ToString(`$h.ComputeHash(`$s))).Replace('-','').ToLowerInvariant()}finally{`$s.Dispose();`$h.Dispose()}};`$files=@($remoteChecks);foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((sh `$p)-cne`$i.sha){throw 'R117_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' '$runnerName'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'R117_REMOTE_AST_REJECTED'}"
$verify|ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -;if($LASTEXITCODE-ne0){throw'R117_REMOTE_VERIFY_FAILED'}
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\$runnerName -TransactionId $transactionId -RemoteRoot $remote -ApprovalSha256 $approvalSha"
if($LASTEXITCODE-ne0){scp "${hostAlias}:$remote\transaction-$transactionId-error.json" $errorOutput 2>$null;throw'R117_TRANSACTION_FAILED'}
scp "${hostAlias}:$remote\transaction-$transactionId-result.json" $output;if($LASTEXITCODE-ne0-or-not(Test-Path -LiteralPath $output -PathType Leaf)){throw'R117_RESULT_DOWNLOAD_FAILED'}
$result=Read-Object $output 'R117_RESULT'
if([string]$result.schema-cne'evomind.g21_conditional_policy_runtime_deployment_result.r117.v1'-or[string]$result.status-cne'installed_and_verified'-or[string]$result.transaction_id-cne$transactionId-or[string]$result.approval_sha256-cne$approvalSha-or
  [string]$result.run_before.run_id-cne$fixedRun-or-not[bool]$result.run_before.terminal-or[int]$result.run_before.effective_active_count-ne0-or[int]$result.run_before.pending_approval_count-ne0-or
  [string]$result.run_after.run_id-cne$fixedRun-or-not[bool]$result.run_after.terminal-or[int]$result.run_after.effective_active_count-ne0-or[int]$result.run_after.pending_approval_count-ne0-or
  [string]$result.installed_runtime_tree.sha256-cne[string]$approval.candidate.installed_runtime_tree_sha256-or[int]$result.installed_runtime_tree.file_count-ne32-or
  [string]$result.goal.spec_sha256-cne$newSpecCanonicalSha-or[string]$result.goal_previous_spec_sha256-cne$oldSpecCanonicalSha-or[int]$result.goal_policy_migrated_event_count-ne1-or-not[bool]$result.policy_replay_idempotent-or-not[bool]$result.policy_restart_reuse-or
  @($result.loopback.samples).Count-ne3-or@($result.loopback.samples|Where-Object{-not[bool]$_.exact}).Count-ne0-or@($result.public.samples).Count-ne3-or@($result.public.samples|Where-Object{-not[bool]$_.exact}).Count-ne0-or
  -not[bool]$result.web_tree_unchanged-or-not[bool]$result.hpc_binding_unchanged-or-not[bool]$result.profile_unchanged-or-not[bool]$result.byoa_unchanged-or[bool]$result.hpc_accessed-or[bool]$result.gpu_touched-or[int]$result.hpc_remote_writes-ne0-or[int]$result.direct_signals_sent-ne0-or[bool]$result.other_processes_modified-or
  [string]$result.web_build_id-cne$fixedWebBuild-or-not[bool]$result.fresh_role_pids-or-not[bool]$result.fresh_listener_pids){throw'R117_RESULT_REJECTED'}
[ordered]@{status='installed_and_verified';transaction_id=$transactionId;result_path=$output;result_sha256=Sha $output;run_id=$fixedRun;hpc_accessed=$false;gpu_touched=$false}|ConvertTo-Json -Compress
