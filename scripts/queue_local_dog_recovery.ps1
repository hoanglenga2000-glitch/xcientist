[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ParentRunId,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localRoot = Join-Path $root "workspace\local_gpu"
$runsRoot = Join-Path $localRoot "mlebench_lite_runs"
$parentRunDir = Join-Path $runsRoot $ParentRunId
$parentSummary = Join-Path $parentRunDir "summary.json"
$parentManifest = Join-Path $parentRunDir "manifest.json"
$statusPath = Join-Path $localRoot "local_dog_recovery_queue_current.json"
$python = (Resolve-Path (Join-Path $root "workspace\local-gpu-venv\Scripts\python.exe")).Path
$completionWatcher = Join-Path $localRoot "watch_local_mlebench_completion.ps1"
$competition = "dog-breed-identification"
$recoveryPrefix = "local4060_dog_breed_imagenethead_s42_"

function Write-AtomicState([System.Collections.IDictionary]$Payload) {
    $temporary = "$statusPath.tmp"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Get-RunProcesses([string]$RunId) {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(.exe)?$' -and
        $_.CommandLine -match [regex]::Escape($RunId)
    } | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine)
}

function Get-OtherOwnedTrainingProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(.exe)?$' -and
        $_.CommandLine -match 'run_mlebench_lite_full.py' -and
        $_.CommandLine -notmatch [regex]::Escape($ParentRunId)
    } | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine)
}

function Find-ExistingRecovery {
    @(Get-ChildItem -LiteralPath $runsRoot -Directory -Filter "$recoveryPrefix*" `
        -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
}

while ($true) {
    $parentProcesses = Get-RunProcesses $ParentRunId
    if (Test-Path -LiteralPath $parentSummary) {
        $summary = Get-Content -LiteralPath $parentSummary -Raw | ConvertFrom-Json
        $result = @($summary.results | Where-Object { $_.competition_id -eq $competition })
        if ($result.Count -ne 1) {
            throw "Parent Dog Breed summary does not contain exactly one competition result"
        }
        if ($parentProcesses.Count -gt 0) {
            Write-AtomicState ([ordered]@{
                schema = "evomind.local_dog_recovery_queue.v1"
                created_at = (Get-Date).ToString("o")
                status = "waiting_for_parent_exit"
                parent_run_id = $ParentRunId
                parent_process_ids = @($parentProcesses | ForEach-Object { $_.ProcessId })
                no_process_signals_sent = $true
                human_gate_preserved = $true
                kaggle_submission_enabled = $false
            })
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $upstream = $result[0].private_grader.upstream_report
        $anyMedal = $null -ne $upstream -and $upstream.any_medal -is [bool] `
            -and $upstream.any_medal
        if ($anyMedal) {
            Write-AtomicState ([ordered]@{
                schema = "evomind.local_dog_recovery_queue.v1"
                created_at = (Get-Date).ToString("o")
                status = "recovery_not_required_parent_medal"
                parent_run_id = $ParentRunId
                official_score = $result[0].mle_private_grader_score
                any_medal = $true
                no_process_signals_sent = $true
                human_gate_preserved = $true
                kaggle_submission_enabled = $false
            })
            exit 0
        }

        $existing = Find-ExistingRecovery
        if ($existing.Count -gt 0) {
            Write-AtomicState ([ordered]@{
                schema = "evomind.local_dog_recovery_queue.v1"
                created_at = (Get-Date).ToString("o")
                status = "existing_recovery_preserved"
                parent_run_id = $ParentRunId
                recovery_run_id = $existing[0].Name
                recovery_run_dir = $existing[0].FullName
                no_process_signals_sent = $true
                human_gate_preserved = $true
                kaggle_submission_enabled = $false
            })
            exit 0
        }

        $otherOwned = Get-OtherOwnedTrainingProcesses
        if ($otherOwned.Count -gt 0) {
            Write-AtomicState ([ordered]@{
                schema = "evomind.local_dog_recovery_queue.v1"
                created_at = (Get-Date).ToString("o")
                status = "waiting_for_local_training_lane"
                parent_run_id = $ParentRunId
                active_owned_process_ids = @($otherOwned | ForEach-Object { $_.ProcessId })
                no_process_signals_sent = $true
                human_gate_preserved = $true
                kaggle_submission_enabled = $false
            })
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $recoveryRunId = "$recoveryPrefix$stamp"
        $stdoutPath = Join-Path $localRoot "$recoveryRunId.launch.stdout.log"
        $stderrPath = Join-Path $localRoot "$recoveryRunId.launch.stderr.log"
        $arguments = @(
            "scripts\run_mlebench_lite_full.py",
            "--data-root", "workspace\local_gpu\mlebench_official_data",
            "--output-root", "workspace\local_gpu\mlebench_lite_runs",
            "--allowed-root", ".",
            "--official-source-root", "external-projects\mle-bench",
            "--waves", "Wave2",
            "--competitions", $competition,
            "--run-id", $recoveryRunId,
            "--seed", "42",
            "--phase-a-scope", "requested",
            "--wave2-dog-breed-batch-size", "16",
            "--wave2-workers", "4",
            "--wave2-vision-folds", "5",
            "--wave2-dog-breed-epochs", "10",
            "--wave2-fast-kernels"
        )
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

        Start-Sleep -Seconds 10
        $manifestPath = Join-Path $runsRoot "$recoveryRunId\manifest.json"
        if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue) -and
            -not (Test-Path -LiteralPath $manifestPath)) {
            throw "Dog Breed recovery process exited before writing its manifest"
        }

        $shell = (Get-Process -Id $PID).Path
        $watcherArguments = @(
            "-NoProfile",
            "-File", $completionWatcher,
            "-RunId", $recoveryRunId,
            "-Competition", $competition,
            "-PollSeconds", "$PollSeconds"
        )
        $watcher = Start-Process -FilePath $shell -ArgumentList $watcherArguments `
            -WorkingDirectory $root -WindowStyle Hidden -PassThru

        Write-AtomicState ([ordered]@{
            schema = "evomind.local_dog_recovery_queue.v1"
            created_at = (Get-Date).ToString("o")
            status = "recovery_launched"
            parent_run_id = $ParentRunId
            parent_summary_status = $summary.status
            parent_gate_passed = $result[0].promotion_gate.passed
            parent_official_score = $result[0].mle_private_grader_score
            parent_any_medal = $false
            recovery_run_id = $recoveryRunId
            recovery_process_id = $process.Id
            completion_watcher_pid = $watcher.Id
            command = @($python) + $arguments
            code_sha256 = (Get-FileHash -LiteralPath (
                Join-Path $root "scripts\mlebench_wave2_adapters.py"
            ) -Algorithm SHA256).Hash.ToLowerInvariant()
            stdout_path = $stdoutPath
            stderr_path = $stderrPath
            no_process_signals_sent = $true
            human_gate_preserved = $true
            kaggle_submission_enabled = $false
        })
        exit 0
    }

    if ($parentProcesses.Count -eq 0 -and (Test-Path -LiteralPath $parentManifest)) {
        Write-AtomicState ([ordered]@{
            schema = "evomind.local_dog_recovery_queue.v1"
            created_at = (Get-Date).ToString("o")
            status = "parent_missing_without_terminal_summary"
            parent_run_id = $ParentRunId
            no_process_signals_sent = $true
            human_gate_preserved = $true
            kaggle_submission_enabled = $false
        })
        exit 4
    }

    Write-AtomicState ([ordered]@{
        schema = "evomind.local_dog_recovery_queue.v1"
        created_at = (Get-Date).ToString("o")
        status = "waiting_for_parent_terminal_summary"
        parent_run_id = $ParentRunId
        parent_process_ids = @($parentProcesses | ForEach-Object { $_.ProcessId })
        poll_seconds = $PollSeconds
        no_process_signals_sent = $true
        human_gate_preserved = $true
        kaggle_submission_enabled = $false
    })
    Start-Sleep -Seconds $PollSeconds
}
