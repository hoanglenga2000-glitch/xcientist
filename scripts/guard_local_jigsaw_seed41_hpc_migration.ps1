[CmdletBinding()]
param(
    [string]$QueueStatus = '',
    [string]$Seed40Verification = '',
    [string]$Seed41RunDirectory = '',
    [string]$Output = '',
    [string]$TargetTaskName = 'EvoMind-Jigsaw-Early-After-May',
    [int]$PollSeconds = 2,
    [double]$DeadlineHours = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $QueueStatus) {
    $QueueStatus = Join-Path $root 'workspace\local_gpu\jigsaw_confirmation_early_queue.json'
}
if (-not $Seed40Verification) {
    $Seed40Verification = Join-Path $root 'workspace\local_gpu\mlebench_lite_runs\local4060_jigsaw_xfmr1_s40_confirm_20260727\independent_verification.json'
}
if (-not $Seed41RunDirectory) {
    $Seed41RunDirectory = Join-Path $root 'workspace\local_gpu\mlebench_lite_runs\local4060_jigsaw_xfmr1_s41_confirm_20260727'
}
if (-not $Output) {
    $Output = Join-Path $root 'workspace\local_gpu\jigsaw_local_seed41_hpc_migration_guard.json'
}
if ($PollSeconds -lt 1 -or $DeadlineHours -le 0) {
    throw 'Migration guard timing contract is invalid.'
}

function Write-AtomicJson {
    param([System.Collections.IDictionary]$Payload)
    $directory = Split-Path -Parent $Output
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $temporary = "$Output.$PID.tmp"
    [System.IO.File]::WriteAllText(
        $temporary,
        ($Payload | ConvertTo-Json -Depth 20) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $Output -Force
}

function Get-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-OwnedQueueProcesses {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -match 'queue_jigsaw_confirmation_after_may\.py' -or
            $_.CommandLine -match 'persistent_manifests[\\/]jigsaw_early_after_may\.json'
        )
    } | Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

$task = Get-ScheduledTask -TaskName $TargetTaskName -ErrorAction Stop
$action = $task.Actions | Select-Object -First 1
if (
    -not $action.Arguments -or
    $action.Arguments -notmatch 'persistent_manifests[\\/]jigsaw_early_after_may\.json'
) {
    throw 'Target Scheduled Task is not the frozen EvoMind early Jigsaw queue.'
}

$deadline = (Get-Date).AddHours($DeadlineHours)
while ((Get-Date) -lt $deadline) {
    $queue = Get-JsonFile -Path $QueueStatus
    $status = if ($null -ne $queue) { [string]$queue.status } else { '' }
    if (
        $status -eq 'training_seed_41' -or
        (Test-Path -LiteralPath $Seed41RunDirectory)
    ) {
        Write-AtomicJson -Payload ([ordered]@{
            schema = 'evomind.local_jigsaw_hpc_migration_guard.v1'
            created_at = (Get-Date -Format o)
            status = 'missed_safe_boundary_no_action_taken'
            queue_status = $status
            seed41_run_directory_present = (Test-Path -LiteralPath $Seed41RunDirectory)
            target_task_name = $TargetTaskName
            target_task_signaled = $false
            training_processes_signaled = 0
            other_processes_signaled = 0
        })
        exit 3
    }

    $completed = @()
    if ($null -ne $queue -and $null -ne $queue.completed_seeds) {
        $completed = @($queue.completed_seeds | ForEach-Object { [int]$_ })
    }
    $activeSeed = if (
        $null -ne $queue -and
        $queue.PSObject.Properties.Name -contains 'active_seed' -and
        $null -ne $queue.active_seed
    ) { [int]$queue.active_seed } else { -1 }
    $trainingPid = if (
        $null -ne $queue -and
        $queue.PSObject.Properties.Name -contains 'training_pid'
    ) { $queue.training_pid } else { $null }
    $atBoundary = (
        $status -eq 'waiting_for_stable_gpu_idle_seed_41' -and
        $completed -contains 40 -and
        $activeSeed -eq 41 -and
        $null -eq $trainingPid
    )
    if (-not $atBoundary) {
        Write-AtomicJson -Payload ([ordered]@{
            schema = 'evomind.local_jigsaw_hpc_migration_guard.v1'
            created_at = (Get-Date -Format o)
            status = 'waiting_for_post_seed40_pre_seed41_boundary'
            queue_status = $status
            completed_seeds = $completed
            target_task_name = $TargetTaskName
            deadline = $deadline.ToString('o')
            target_task_signaled = $false
            training_processes_signaled = 0
            other_processes_signaled = 0
        })
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $verification = Get-JsonFile -Path $Seed40Verification
    $verified = (
        $null -ne $verification -and
        [string]$verification.status -eq 'promotion_gate_passed' -and
        $verification.full_contract_valid -eq $true -and
        $verification.private_labels_used -eq $false -and
        $verification.official_grader_executed -eq $false -and
        $verification.kaggle_submission_executed -eq $false -and
        [int]$verification.process_signals_sent -eq 0
    )
    if (-not $verified) {
        Write-AtomicJson -Payload ([ordered]@{
            schema = 'evomind.local_jigsaw_hpc_migration_guard.v1'
            created_at = (Get-Date -Format o)
            status = 'boundary_seen_but_seed40_verification_not_valid'
            target_task_name = $TargetTaskName
            target_task_signaled = $false
            training_processes_signaled = 0
            other_processes_signaled = 0
        })
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $ownedBefore = @(Get-OwnedQueueProcesses)
    if ($ownedBefore.Count -lt 1) {
        Write-AtomicJson -Payload ([ordered]@{
            schema = 'evomind.local_jigsaw_hpc_migration_guard.v1'
            created_at = (Get-Date -Format o)
            status = 'safe_boundary_queue_already_absent'
            target_task_name = $TargetTaskName
            target_task_signaled = $false
            training_processes_signaled = 0
            other_processes_signaled = 0
        })
        exit 0
    }

    Disable-ScheduledTask -TaskName $TargetTaskName | Out-Null
    Stop-ScheduledTask -TaskName $TargetTaskName
    Start-Sleep -Seconds 3
    $ownedAfter = @(Get-OwnedQueueProcesses)
    $seed41Present = Test-Path -LiteralPath $Seed41RunDirectory
    $finalStatus = if ($ownedAfter.Count -eq 0 -and -not $seed41Present) {
        'local_seed41_launch_prevented_hpc_migration_active'
    } else {
        'owned_queue_residual_requires_review'
    }
    Write-AtomicJson -Payload ([ordered]@{
        schema = 'evomind.local_jigsaw_hpc_migration_guard.v1'
        created_at = (Get-Date -Format o)
        status = $finalStatus
        queue_status_at_boundary = $status
        seed40_verification_path = $Seed40Verification
        seed41_run_directory_present = $seed41Present
        target_task_name = $TargetTaskName
        target_task_state = [string](Get-ScheduledTask -TaskName $TargetTaskName).State
        owned_queue_processes_before = @($ownedBefore)
        owned_queue_processes_after = @($ownedAfter)
        target_task_signaled = $true
        owned_queue_stop_requests = 1
        training_processes_signaled = 0
        other_processes_signaled = 0
        process_scope = 'Only the EvoMind early Jigsaw queue at the verified post-seed40 pre-seed41 boundary.'
        official_grader_executed = $false
        kaggle_submission_executed = $false
    })
    exit $(if ($finalStatus -eq 'local_seed41_launch_prevented_hpc_migration_active') { 0 } else { 4 })
}

Write-AtomicJson -Payload ([ordered]@{
    schema = 'evomind.local_jigsaw_hpc_migration_guard.v1'
    created_at = (Get-Date -Format o)
    status = 'timeout_waiting_for_safe_boundary'
    target_task_name = $TargetTaskName
    target_task_signaled = $false
    training_processes_signaled = 0
    other_processes_signaled = 0
})
exit 4
