param(
  [Parameter(Mandatory)][string]$ApprovalPath,
  [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$approvalSchema = 'evomind.g21_goal_runtime_deployment_approval.v1'
$manifestSchema = 'evomind.g21_goal_runtime_overlay.clean.v1'
$receiptSchema = 'evomind.g21_goal_runtime_test_receipt.v1'
$fixedRun = 'run_7b1efb878afb40f396db431e91f093a5'
$fixedAllocation = 'G21'
$fixedWebBuild = 'overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc'
$r115ZipSha = 'fc65d7134f9f53fa6eda21ee2fcd804a330d9184a83d5206a6b7469a217d89ba'
$r115TreeSha = 'b92cc96dae5ac1c1ed3641999eec56eda782b4959ac8709a0c0584ac1438955a'
$goalSpecFileSha = '380a3b2c037c8067a4618717cb7aaf8847d323d4be401a88d64009468132848f'
$humanFileSha = '5b6ed3d912ad1cd4655c96f832629cf9603a4b0ad58ab6e27c01861fadd69c5f'
$rejectedCandidateSha = '37929a1c9baf682e4f946450f834f10fe3b7a78323084ae96948bd0e92f8de8d'
$rejectionEvidenceSha = '8900c32d41a47561715dc215ec2e560e338c2d164d6fa410554b7a2b83b5fb83'
$hostAlias = 'evomind-shanghai'
$runner = Join-Path $PSScriptRoot 'Deploy-G21GoalRuntimeR116Remote.ps1'
$candidateVerifier = Join-Path $PSScriptRoot 'verify_g21_goal_runtime_r116_candidate.py'

function Sha([string]$Path) {
  (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-Object([string]$Path, [string]$Code) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Code`:MISSING" }
  try { $value = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json }
  catch { throw "$Code`:INVALID_JSON" }
  if ($null -eq $value -or $value -is [array]) { throw "$Code`:OBJECT_REQUIRED" }
  $value
}

function Assert-Sha([object]$Value, [string]$Code) {
  if ([string]$Value -cnotmatch '^[a-f0-9]{64}$') { throw "$Code`:SHA_REJECTED" }
}

function Resolve-BoundInput([string]$Directory, [object]$Name, [string]$Code) {
  $fileName = [string]$Name
  if ($fileName -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$' -or [IO.Path]::GetFileName($fileName) -cne $fileName) {
    throw "$Code`:FILENAME_REJECTED"
  }
  $path = [IO.Path]::GetFullPath((Join-Path $Directory $fileName))
  if ([IO.Path]::GetDirectoryName($path).TrimEnd('\') -cne [IO.Path]::GetFullPath($Directory).TrimEnd('\')) {
    throw "$Code`:PATH_ESCAPE"
  }
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "$Code`:MISSING" }
  $path
}

function Assert-BoundFile([string]$Directory, [object]$Entry, [string]$NameField, [string]$ShaField, [string]$Code) {
  if ($null -eq $Entry) { throw "$Code`:ENTRY_MISSING" }
  $path = Resolve-BoundInput $Directory $Entry.$NameField $Code
  Assert-Sha $Entry.$ShaField $Code
  if ((Sha $path) -cne [string]$Entry.$ShaField) { throw "$Code`:HASH_REJECTED" }
  $path
}

function Read-ZipManifest([string]$ArchivePath) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
  try {
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $manifestEntries = @()
    foreach ($entry in $archive.Entries) {
      $name = ([string]$entry.FullName).Replace('\','/')
      if (-not $name -or $name.StartsWith('/') -or $name -match '(^|/)\.\.(/|$)|^[A-Za-z]:' -or -not $seen.Add($name)) {
        throw "R116_CANDIDATE_ZIP_UNSAFE:$name"
      }
      if ($name -ceq 'runtime-hotfix-manifest.json') { throw 'R116_STALE_RUNTIME_MANIFEST_REJECTED' }
      if ($name -ceq 'goal-runtime-overlay-manifest.json') { $manifestEntries += $entry }
    }
    if ($manifestEntries.Count -ne 1) { throw 'R116_GOAL_MANIFEST_COUNT_REJECTED' }
    $reader = [IO.StreamReader]::new($manifestEntries[0].Open(), [Text.Encoding]::UTF8, $true)
    try { $raw = $reader.ReadToEnd() } finally { $reader.Dispose() }
    try { $manifest = $raw | ConvertFrom-Json } catch { throw 'R116_EMBEDDED_MANIFEST_JSON_REJECTED' }
    [ordered]@{ manifest = $manifest; raw_sha256 = ([BitConverter]::ToString(([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($raw)))).Replace('-','').ToLowerInvariant() }
  } finally { $archive.Dispose() }
}

function Assert-TestReceipt([string]$Path, [object]$Entry, [object]$Approval) {
  $receipt = Read-Object $Path 'R116_TEST_RECEIPT'
  $checks = $receipt.checks
  $targeted = $receipt.test_artifacts.targeted_goal_builder_junit
  $regression = $receipt.test_artifacts.runtime_regression_junit
  if (
    [string]$Entry.schema -cne $receiptSchema -or
    [string]$receipt.schema -cne $receiptSchema -or
    [string]$receipt.status -cne 'passed' -or
    [string]$receipt.run_id -cne $fixedRun -or
    [string]$receipt.allocation -cne $fixedAllocation -or
    [string]$receipt.candidate_zip_sha256 -cne [string]$Approval.candidate.zip_sha256 -or
    [string]$receipt.source_manifest_sha256 -cne [string]$Approval.candidate.source_manifest_sha256 -or
    [int]$checks.dual_extract_rounds -ne 2 -or
    [int]$checks.python_files_compiled -ne 32 -or
    [int]$checks.goal_http_smoke_rounds -ne 2 -or
    [int]$checks.baseline_unchanged_files -ne 25 -or
    [int]$checks.changed_files -ne 7 -or
    [int]$checks.unsafe_entries -ne 0 -or
    [int]$checks.duplicate_entries -ne 0 -or
    [int]$checks.symlinks -ne 0 -or
    [string]$checks.secret_scan -cne 'passed' -or
    [string]$checks.git_diff_check -cne 'passed' -or
    [bool]$checks.hpc_accessed -or [bool]$checks.gpu_touched -or [int]$checks.remote_writes -ne 0 -or
    [int]$targeted.tests -lt 55 -or [int]$targeted.failures -ne 0 -or [int]$targeted.errors -ne 0 -or [string]$targeted.sha256 -cnotmatch '^[a-f0-9]{64}$' -or
    [int]$regression.tests -lt 112 -or [int]$regression.failures -ne 0 -or [int]$regression.errors -ne 0 -or [string]$regression.sha256 -cnotmatch '^[a-f0-9]{64}$' -or
    [string]$receipt.bootstrap_artifact_sha256 -cne [string]$Approval.bootstrap.artifact_sha256 -or
    [string]$receipt.verifier.filename -cne [IO.Path]::GetFileName($candidateVerifier) -or
    [string]$receipt.verifier.sha256 -cne [string]$Approval.verification.candidate_verifier_sha256
  ) { throw 'R116_TEST_RECEIPT_CONTRACT_REJECTED' }
}

function Read-Approval([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path)
  $approval = Read-Object $full 'R116_APPROVAL'
  $directory = [IO.Path]::GetDirectoryName($full)
  if (
    [string]$approval.schema -cne $approvalSchema -or
    [string]$approval.status -cne 'approved' -or
    $approval.single_use -isnot [bool] -or -not [bool]$approval.single_use -or
    [string]$approval.deployment_id -cnotmatch '^[a-f0-9]{32}$' -or
    [string]$approval.scope.mode -cne 'runtime_only' -or
    [string]$approval.scope.run_id -cne $fixedRun -or
    [string]$approval.scope.allocation -cne $fixedAllocation -or
    [string]$approval.scope.expected_web_build_id -cne $fixedWebBuild -or
    [string]$approval.scope.expected_current_runtime_tree_sha256 -cne $r115TreeSha -or
    [bool]$approval.scope.hpc_accessed -or [bool]$approval.scope.gpu_touched -or
    [string]$approval.baseline.release -cne 'r115-progress-parser-r114-base' -or
    [string]$approval.baseline.zip_sha256 -cne $r115ZipSha -or
    [string]$approval.baseline.runtime_tree_sha256 -cne $r115TreeSha -or
    [int]$approval.baseline.runtime_file_count -ne 30 -or
    [string]$approval.candidate.release -cne 'r116-g21-goal-r115-base' -or
    [string]$approval.candidate.zip_sha256 -ceq $rejectedCandidateSha -or
    [int]$approval.candidate.runtime_file_count -ne 32 -or
    [int]$approval.candidate.changed_file_count -ne 7 -or
    [string]$approval.bootstrap.goal_spec_file_sha256 -cne $goalSpecFileSha -or
    [string]$approval.bootstrap.human_baseline_file_sha256 -cne $humanFileSha -or
    [string]$approval.bootstrap.goal_record_status -cne 'blocked' -or
    [string]$approval.deployer.service_action_path -cne 'C:\SecureInput\Invoke-ServiceAccountAction.ps1' -or
    -not [bool]$approval.prohibited.web_replacement -or
    -not [bool]$approval.prohibited.direct_scheduled_task_start -or
    -not [bool]$approval.prohibited.process_kill -or
    -not [bool]$approval.prohibited.direct_sqlite_write -or
    -not [bool]$approval.prohibited.hpc_access -or
    -not [bool]$approval.prohibited.chrome -or
    -not [bool]$approval.prohibited.training -or
    [string]$approval.rejected_candidate.zip_sha256 -cne $rejectedCandidateSha -or
    [string]$approval.rejected_candidate.evidence_sha256 -cne $rejectionEvidenceSha -or
    [string]$approval.rejected_candidate.status -cne 'NOT_DEPLOYABLE'
  ) { throw 'R116_APPROVAL_CONTRACT_REJECTED' }
  try {
    $issued = [DateTimeOffset]::Parse([string]$approval.issued_at_utc)
    $expires = [DateTimeOffset]::Parse([string]$approval.expires_at_utc)
  } catch { throw 'R116_APPROVAL_TIME_REJECTED' }
  $now = [DateTimeOffset]::UtcNow
  if ($expires -le $issued -or $now -lt $issued.AddMinutes(-5) -or $now -ge $expires -or ($expires - $issued).TotalHours -gt 24) {
    throw 'R116_APPROVAL_EXPIRED_OR_TIME_REJECTED'
  }
  foreach ($value in @(
    $approval.candidate.zip_sha256, $approval.candidate.source_manifest_sha256,
    $approval.candidate.build_result_sha256, $approval.candidate.embedded_manifest_sha256,
    $approval.candidate.installed_runtime_tree_sha256, $approval.bootstrap.goal_spec_sha256,
    $approval.bootstrap.goal_board_file_sha256, $approval.bootstrap.goal_board_sha256,
    $approval.bootstrap.artifact_sha256,
    $approval.deployer.wrapper_sha256, $approval.deployer.remote_runner_sha256,
    $approval.verification.candidate_verifier_sha256
  )) { Assert-Sha $value 'R116_APPROVAL' }
  if ([string]$approval.deployer.wrapper_filename -cne [IO.Path]::GetFileName($PSCommandPath) -or (Sha $PSCommandPath) -cne [string]$approval.deployer.wrapper_sha256) {
    throw 'R116_WRAPPER_HASH_REJECTED'
  }
  if ([string]$approval.deployer.remote_runner_filename -cne [IO.Path]::GetFileName($runner) -or -not (Test-Path -LiteralPath $runner -PathType Leaf) -or (Sha $runner) -cne [string]$approval.deployer.remote_runner_sha256) {
    throw 'R116_REMOTE_RUNNER_HASH_REJECTED'
  }
  if ([string]$approval.verification.candidate_verifier_filename -cne [IO.Path]::GetFileName($candidateVerifier) -or -not (Test-Path -LiteralPath $candidateVerifier -PathType Leaf) -or (Sha $candidateVerifier) -cne [string]$approval.verification.candidate_verifier_sha256) {
    throw 'R116_CANDIDATE_VERIFIER_HASH_REJECTED'
  }
  $runtime = Assert-BoundFile $directory $approval.candidate 'zip_filename' 'zip_sha256' 'R116_RUNTIME'
  $sourceManifestPath = Assert-BoundFile $directory $approval.candidate 'source_manifest_filename' 'source_manifest_sha256' 'R116_SOURCE_MANIFEST'
  $buildResultPath = Assert-BoundFile $directory $approval.candidate 'build_result_filename' 'build_result_sha256' 'R116_BUILD_RESULT'
  $goalSpecPath = Assert-BoundFile $directory $approval.bootstrap 'goal_spec_filename' 'goal_spec_file_sha256' 'R116_GOAL_SPEC'
  $humanPath = Assert-BoundFile $directory $approval.bootstrap 'human_baseline_filename' 'human_baseline_file_sha256' 'R116_HUMAN_BASELINE'
  $boardPath = Assert-BoundFile $directory $approval.bootstrap 'goal_board_filename' 'goal_board_file_sha256' 'R116_GOAL_BOARD'
  $bootstrapArtifactPath = Assert-BoundFile $directory $approval.bootstrap 'artifact_filename' 'artifact_sha256' 'R116_BOOTSTRAP_ARTIFACT'
  $rejectionEvidencePath = Assert-BoundFile $directory $approval.rejected_candidate 'evidence_filename' 'evidence_sha256' 'R116_REJECTION_EVIDENCE'
  $sourceManifest = Read-Object $sourceManifestPath 'R116_SOURCE_MANIFEST'
  $buildResult = Read-Object $buildResultPath 'R116_BUILD_RESULT'
  $zipManifest = Read-ZipManifest $runtime
  if (
    [string]$sourceManifest.schema -cne $manifestSchema -or
    [string]$sourceManifest.run_id -cne $fixedRun -or
    [string]$sourceManifest.allocation -cne $fixedAllocation -or
    [string]$sourceManifest.baseline_zip_sha256 -cne $r115ZipSha -or
    [int]$sourceManifest.candidate_runtime_file_count -ne 32 -or
    [int]$sourceManifest.file_count -ne 32 -or
    [int]$sourceManifest.changed_file_count -ne 7 -or
    [string]$sourceManifest.candidate_runtime_tree_sha256 -cne [string]$approval.candidate.installed_runtime_tree_sha256 -or
    [string]$sourceManifest.bootstrap_artifact_sha256 -cne [string]$approval.bootstrap.artifact_sha256 -or
    [bool]$sourceManifest.production_deployable -or [bool]$sourceManifest.production_deployed -or
    [string]$zipManifest.raw_sha256 -cne [string]$approval.candidate.embedded_manifest_sha256 -or
    [string]$approval.candidate.source_manifest_sha256 -cne [string]$approval.candidate.embedded_manifest_sha256 -or
    [string]$buildResult.schema -cne $manifestSchema -or [string]$buildResult.status -cne 'built_and_verified' -or
    [string]$buildResult.zip_sha256 -cne [string]$approval.candidate.zip_sha256 -or
    [string]$buildResult.manifest_sha256 -cne [string]$approval.candidate.source_manifest_sha256 -or
    [string]$buildResult.candidate_runtime_tree_sha256 -cne [string]$approval.candidate.installed_runtime_tree_sha256 -or
    [string]$buildResult.bootstrap_artifact_sha256 -cne [string]$approval.bootstrap.artifact_sha256 -or
    [int]$buildResult.candidate_runtime_file_count -ne 32 -or
    @($buildResult.verification_rounds).Count -ne 2 -or
    @($buildResult.verification_rounds | Where-Object { [string]$_.status -cne 'passed' }).Count -ne 0 -or
    [bool]$buildResult.production_deployable -or [bool]$buildResult.production_deployed -or
    [bool]$buildResult.hpc_accessed -or [bool]$buildResult.gpu_touched -or [int]$buildResult.remote_writes -ne 0
  ) { throw 'R116_CANDIDATE_BINDING_REJECTED' }
  $spec = Read-Object $goalSpecPath 'R116_GOAL_SPEC'
  $human = Read-Object $humanPath 'R116_HUMAN_BASELINE'
  $board = Read-Object $boardPath 'R116_GOAL_BOARD'
  $bootstrapArtifact = Read-Object $bootstrapArtifactPath 'R116_BOOTSTRAP_ARTIFACT'
  $rejectionEvidence = Read-Object $rejectionEvidencePath 'R116_REJECTION_EVIDENCE'
  if (
    [string]$spec.schema -cne 'evomind.five_competition_goal.v1' -or [string]$spec.run_id -cne $fixedRun -or [string]$spec.allocation -cne $fixedAllocation -or
    [string]$spec.human_baseline_evidence_sha256 -cne $humanFileSha -or
    [string]$human.schema -cne 'evomind.five_competition_human_baseline_audit.v1' -or [string]$human.run_id -cne $fixedRun -or [string]$human.classification -cne 'HUMAN_BASELINE_UNDEFINED' -or
    [bool]$human.completion.training_authorized_by_this_artifact -or
    [string]$board.schema -cne 'evomind.goal-board.v2' -or [string]$board.run_id -cne $fixedRun -or [string]$board.allocation -cne $fixedAllocation -or
    [string]$board.goal_record_status -cne 'blocked' -or [int]$board.completion_count -ne 0 -or [int]$board.weather_actions -ne 0 -or @($board.competitions).Count -ne 5 -or
    [string]$bootstrapArtifact.schema -cne 'evomind.g21_goal_runtime_bootstrap.v1' -or [string]$bootstrapArtifact.run_id -cne $fixedRun -or [string]$bootstrapArtifact.goal_spec_file_sha256 -cne $goalSpecFileSha -or
    [string]$bootstrapArtifact.human_baseline_evidence_file_sha256 -cne $humanFileSha -or [string]$bootstrapArtifact.goal_board_file_sha256 -cne [string]$approval.bootstrap.goal_board_file_sha256 -or
    [string]$bootstrapArtifact.goal_board_sha256 -cne [string]$approval.bootstrap.goal_board_sha256 -or [bool]$bootstrapArtifact.production_deployable -or
    [string]$rejectionEvidence.schema -cne 'evomind.g21_goal_runtime_candidate_rejection.v1' -or
    [string]$rejectionEvidence.status -cne 'NOT_DEPLOYABLE' -or
    -not @($rejectionEvidence.candidate_files | Where-Object { [string]$_.sha256 -ceq $rejectedCandidateSha }).Count
  ) { throw 'R116_BOOTSTRAP_CONTRACT_REJECTED' }
  $receiptEntries = @($approval.test_receipts)
  if ($receiptEntries.Count -lt 1) { throw 'R116_INDEPENDENT_TEST_RECEIPT_REQUIRED' }
  $receiptPaths = @()
  foreach ($entry in $receiptEntries) {
    $receiptPath = Assert-BoundFile $directory $entry 'filename' 'sha256' 'R116_TEST_RECEIPT'
    Assert-TestReceipt $receiptPath $entry $approval
    $receiptPaths += $receiptPath
  }
  [ordered]@{
    path = $full; directory = $directory; value = $approval; runtime = $runtime;
    source_manifest = $sourceManifestPath; build_result = $buildResultPath;
    goal_spec = $goalSpecPath; human_baseline = $humanPath; goal_board = $boardPath;
    bootstrap_artifact = $bootstrapArtifactPath;
    rejection_evidence = $rejectionEvidencePath;
    test_receipts = $receiptPaths
  }
}

$bound = Read-Approval $ApprovalPath
$approval = $bound.value
$approvalSha = Sha $bound.path
$transactionId = [string]$approval.deployment_id
if ($ValidateOnly) {
  [ordered]@{
    schema = $approvalSchema; status = 'validated_not_deployed'; deployment_id = $transactionId;
    run_id = $fixedRun; allocation = $fixedAllocation; approval_sha256 = $approvalSha;
    candidate_sha256 = [string]$approval.candidate.zip_sha256; test_receipt_count = @($bound.test_receipts).Count;
    hpc_accessed = $false; gpu_touched = $false; remote_writes = 0
  } | ConvertTo-Json -Compress
  exit 0
}

$output = Join-Path $bound.directory "r116-goal-runtime-production-$transactionId.json"
$errorOutput = Join-Path $bound.directory "r116-goal-runtime-production-$transactionId.error.json"
foreach ($path in @($output, $errorOutput)) { if (Test-Path -LiteralPath $path) { throw "R116_DEPLOY_OUTPUT_EXISTS:$path" } }
$remote = "C:\SecureInput\r116-goal-runtime-$transactionId"
$create = "if(Test-Path -LiteralPath '$remote'){throw 'R116_REMOTE_EXISTS'};New-Item -ItemType Directory -Path '$remote'|Out-Null"
$create | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'R116_REMOTE_CREATE_FAILED' }
$uploads = @(
  [pscustomobject]@{ path=$bound.path; name='approval.json'; sha=$approvalSha },
  [pscustomobject]@{ path=$runner; name=[IO.Path]::GetFileName($runner); sha=Sha $runner },
  [pscustomobject]@{ path=$bound.runtime; name='runtime.zip'; sha=Sha $bound.runtime },
  [pscustomobject]@{ path=$bound.source_manifest; name='source-manifest.json'; sha=Sha $bound.source_manifest },
  [pscustomobject]@{ path=$bound.build_result; name='build-result.json'; sha=Sha $bound.build_result },
  [pscustomobject]@{ path=$bound.goal_spec; name='goal-spec.json'; sha=Sha $bound.goal_spec },
  [pscustomobject]@{ path=$bound.human_baseline; name='human-baseline.json'; sha=Sha $bound.human_baseline },
  [pscustomobject]@{ path=$bound.goal_board; name='goal-board.json'; sha=Sha $bound.goal_board },
  [pscustomobject]@{ path=$bound.bootstrap_artifact; name='bootstrap.json'; sha=Sha $bound.bootstrap_artifact },
  [pscustomobject]@{ path=$bound.rejection_evidence; name='rejected-candidate.json'; sha=Sha $bound.rejection_evidence }
)
for ($index=0; $index -lt @($bound.test_receipts).Count; $index++) {
  $uploads += [pscustomobject]@{ path=$bound.test_receipts[$index]; name="test-receipt-$($index+1).json"; sha=Sha $bound.test_receipts[$index] }
}
foreach ($item in $uploads) {
  scp $item.path "${hostAlias}:$remote\$($item.name)"
  if ($LASTEXITCODE -ne 0) { throw "R116_UPLOAD_FAILED:$($item.name)" }
}
$remoteChecks = @($uploads | ForEach-Object { "[pscustomobject]@{name='$($_.name)';sha='$($_.sha)'}" }) -join ','
$verify = "`$files=@($remoteChecks);foreach(`$i in `$files){`$p=Join-Path '$remote' `$i.name;if((Get-FileHash -LiteralPath `$p -Algorithm SHA256).Hash.ToLowerInvariant()-cne`$i.sha){throw 'R116_REMOTE_HASH_REJECTED'}};`$tokens=`$null;`$errors=`$null;[Management.Automation.Language.Parser]::ParseFile((Join-Path '$remote' '$([IO.Path]::GetFileName($runner))'),[ref]`$tokens,[ref]`$errors)|Out-Null;if(@(`$errors).Count){throw 'R116_REMOTE_AST_REJECTED'}"
$verify | ssh $hostAlias powershell.exe -NoProfile -NonInteractive -Command -
if ($LASTEXITCODE -ne 0) { throw 'R116_REMOTE_VERIFY_FAILED' }
ssh $hostAlias "powershell.exe -NoProfile -NonInteractive -File $remote\$([IO.Path]::GetFileName($runner)) -TransactionId $transactionId -RemoteRoot $remote -ApprovalSha256 $approvalSha"
if ($LASTEXITCODE -ne 0) {
  scp "${hostAlias}:$remote\transaction-$transactionId-error.json" $errorOutput 2>$null
  throw 'R116_TRANSACTION_FAILED'
}
scp "${hostAlias}:$remote\transaction-$transactionId-result.json" $output
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) { throw 'R116_RESULT_DOWNLOAD_FAILED' }
$result = Read-Object $output 'R116_RESULT'
if (
  [string]$result.schema -cne 'evomind.g21_goal_runtime_deployment_result.v1' -or [string]$result.status -cne 'installed_and_verified' -or
  [string]$result.transaction_id -cne $transactionId -or [string]$result.approval_sha256 -cne $approvalSha -or
  [string]$result.run_before.run_id -cne $fixedRun -or -not [bool]$result.run_before.terminal -or [int]$result.run_before.effective_active_count -ne 0 -or [int]$result.run_before.pending_approval_count -ne 0 -or
  [string]$result.run_after.run_id -cne $fixedRun -or -not [bool]$result.run_after.terminal -or [int]$result.run_after.effective_active_count -ne 0 -or [int]$result.run_after.pending_approval_count -ne 0 -or
  [string]$result.installed_runtime_tree.sha256 -cne [string]$approval.candidate.installed_runtime_tree_sha256 -or [int]$result.installed_runtime_tree.file_count -ne 32 -or
  @($result.loopback.samples).Count -ne 3 -or @($result.loopback.samples | Where-Object { -not [bool]$_.exact }).Count -ne 0 -or
  @($result.public.samples).Count -ne 3 -or @($result.public.samples | Where-Object { -not [bool]$_.exact }).Count -ne 0 -or
  [string]$result.goal.id -cne 'goal_g21_five_competition' -or [string]$result.goal.run_id -cne $fixedRun -or [string]$result.goal.status -cne 'blocked' -or
  [string]$result.goal.spec_sha256 -cne [string]$approval.bootstrap.goal_spec_sha256 -or [string]$result.goal.board_sha256 -cne [string]$approval.bootstrap.goal_board_sha256 -or
  [string]$result.goal.human_baseline_sha256 -cne $humanFileSha -or [int]$result.goal_created_event_count -ne 1 -or [int]$result.goal_record_count -ne 1 -or
  [string]$result.session_goal_metadata.goal_id -cne [string]$result.goal.id -or [string]$result.session_goal_metadata.goal_key -cne [string]$result.goal.goal_key -or
  [string]$result.session_goal_metadata.goal_spec_sha256 -cne [string]$result.goal.spec_sha256 -or [string]$result.session_goal_metadata.goal_board_sha256 -cne [string]$result.goal.board_sha256 -or
  [string]$result.session_goal_metadata.goal_human_baseline_sha256 -cne $humanFileSha -or
  [int64]$result.goal_event_window.after_seq -ne [int64]$result.run_before.last_event_seq -or [int64]$result.goal_event_window.goal_created_seq -le [int64]$result.goal_event_window.after_seq -or
  [int64]$result.goal_event_window.goal_created_seq -gt [int64]$result.goal_event_window.upper_seq -or [int]$result.goal_event_window.page_count -lt 1 -or [int]$result.goal_event_window.event_count -lt 1 -or
  -not [bool]$result.hpc_binding_unchanged -or -not [bool]$result.profile_unchanged -or -not [bool]$result.byoa_unchanged -or
  [bool]$result.hpc_accessed -or [bool]$result.gpu_touched -or [int]$result.hpc_remote_writes -ne 0 -or [int]$result.direct_signals_sent -ne 0 -or [bool]$result.other_processes_modified -or
  -not [bool]$result.production_writes_performed -or [int]$result.production_write_category_count -ne 5 -or @($result.production_write_categories).Count -ne 5 -or @($result.managed_service_actions).Count -ne 4 -or
  [string]$result.web_build_id -cne $fixedWebBuild -or
  -not [bool]$result.web_tree_unchanged -or
  [string]$result.web_tree_before.sha256 -cne [string]$result.web_tree_after.sha256 -or
  -not [bool]$result.fresh_role_pids -or -not [bool]$result.fresh_listener_pids
) { throw 'R116_RESULT_REJECTED' }
[ordered]@{ status='installed_and_verified'; transaction_id=$transactionId; result_path=$output; result_sha256=Sha $output; run_id=$fixedRun; goal_id='goal_g21_five_competition'; hpc_accessed=$false; gpu_touched=$false } | ConvertTo-Json -Compress
