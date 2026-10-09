from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "Build-AssistantInteractionWebReleaseR122.ps1"
WRAPPER = ROOT / "scripts" / "Deploy-AssistantInteractionWebR122.ps1"
REMOTE = ROOT / "scripts" / "Deploy-AssistantInteractionWebR122Remote.ps1"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builder_is_r112_frozen_and_has_exact_three_file_production_allowlist() -> None:
    source = text(BUILDER)
    assert "2284252cf7ae98a1d3db08e9117d863d839e55174d428df2e7e00dc1a7d8eaa2" in source
    assert "overlay-controlled-secret-binding-r112-standalone-f38c40f2dbdc" in source
    assert "782333e5552c26ad31be24503c05eed9844af79362df333e3d3641a612e48da1" in source
    block = source[source.index("$patches=@(") : source.index("$testInputs=@(")]
    paths = re.findall(r"relative='([^']+)'", block)
    assert paths == [
        "src/components/workstation/screens/AssistantScreen.tsx",
        "src/components/workstation/screens/assistant-activity.mjs",
        "src/components/workstation/screens/assistant-activity.d.mts",
    ]
    assert "production_allowlist_count=3" in source
    assert "$origin=Join-Path $baselineRoot $name" in source
    assert "Copy-DirectoryContents $origin (Join-Path $buildRoot $name)" in source
    assert "Copy-Item -LiteralPath $patch.source" in source
    assert "Copy-Item -LiteralPath (Join-Path $web 'src')" not in source


def test_builder_keeps_both_targeted_tests_out_of_the_product_patch() -> None:
    source = text(BUILDER)
    assert "assistant-activity.test.ts" in source
    assert "assistant-super-agent-contract.test.ts" in source
    assert source.count("included_in_production=$false") >= 3
    assert "$test_in_production=$false" not in source  # field is emitted as a literal property, not a mutable variable
    assert "test_in_production=$false" in source
    assert "R122_TEST_FILE_PACKAGED" not in source  # packaging rejection is enforced by the remote verifier
    assert "Assert-SourceScope $releaseRoot" in source


def test_builder_requires_targeted_tests_typecheck_build_and_two_independent_rounds() -> None:
    source = text(BUILDER)
    assert "--test --experimental-strip-types" in source
    assert "assistant_super_agent_contract='passed'" in source
    assert "--noEmit --incremental false" in source
    assert "build --webpack" in source
    assert "foreach($round in 1..2)" in source
    assert "Invoke-ReleaseSmoke" in source
    assert "Verify-RegularManifest" in source
    assert "Test-ZipSafety" in source
    assert "function New-SafeZip" in source
    assert "ZipArchiveMode]::Create" in source
    assert "normalized_paths=$true" in source
    assert "CreateFromDirectory" not in source
    assert "Assert-SourceScope $target $baselineRoot $patches $testRelative -Production" in source
    assert "function New-SmokeSessionSecret" in source
    assert "RandomNumberGenerator]::Create()" in source
    assert "RandomNumberGenerator]::GetBytes" not in source
    assert "function Invoke-HttpResponse" in source
    assert "Add-Type -AssemblyName System.Net.Http" in source
    assert "Invoke-WebRequest -Uri $Uri -Headers $Headers" in source
    assert "Wait-Web \"http://127.0.0.1:$runtimePort/v1/health\" 30 $headers" in source
    assert "SkipHttpErrorCheck" not in source


def test_remote_runner_fails_closed_on_run_g24_runtime_and_hpc_boundaries() -> None:
    source = text(REMOTE)
    assert "run_7b1efb878afb40f396db431e91f093a5" in source
    assert "$fixedJob=93015" in source
    assert "$fixedGeneration=24" in source
    assert "effective_active_count=0" in source
    assert "pending_approval_count=0" in source
    assert "not[bool]$s.terminal" in source
    assert "R122_RUN_GATE_REJECTED" in source
    assert "R122_RUNTIME_BASELINE_REJECTED" in source
    assert "Assert-BoundarySame" in source
    assert "R122_HPC_BYOA_BOUNDARY_CHANGED" in source
    assert "Invoke-Managed Stop" in source
    assert "Invoke-Managed Start" in source
    assert "Restore-Config" in source
    assert "restored_and_verified" in source
    assert "Start-ScheduledTask" not in source
    assert "Stop-Process" not in source
    assert "$rolePid=[int]$r[0].pid" in source
    assert "$pid=[int]$r[0].pid" not in source
    assert "$profileHome=if($profileInfo)" in source
    assert "$home=if($profileInfo)" not in source
    assert "$lines.Sort([StringComparer]::Ordinal)" in source
    assert "ordinal_sort=$true" in source


def test_remote_runner_revalidates_candidate_and_rejects_test_packaging() -> None:
    source = text(REMOTE)
    assert "R122_REMOTE_RUNNER_SELF_HASH_REJECTED" in source
    assert "production_allowlist_count-ne3" in source
    assert "R122_TEST_FILE_PACKAGED" in source
    assert "Verify-Manifest $extract $op" in source
    assert "changed_source_files" in source
    assert "Get-Health 'http://127.0.0.1:8088/api/healthz'" in source
    assert "Get-Health 'https://evomind.zhjjq.tech/api/healthz'" in source
    assert "hpc_remote_writes=0" in source
    assert "training_started=$false" in source


def test_local_wrapper_is_hash_bound_and_preflights_before_transaction() -> None:
    source = text(WRAPPER)
    for name in (
        "CandidateZipSha256",
        "SourceManifestSha256",
        "BuildResultSha256",
        "RemoteRunnerSha256",
    ):
        assert f"[string]${name}" in source
    preflight = source.index("-PreflightOnly")
    transaction = source.index('ssh $HostAlias "powershell.exe -NoProfile -NonInteractive -File', preflight + 1)
    assert preflight < transaction
    assert "R122_REMOTE_PREFLIGHT_REJECTED" in source
    assert "run_unchanged" in source
    assert "runtime_unchanged" in source
    assert "hpc_binding_unchanged" in source
    assert "byoa_unchanged" in source
    assert "profile_unchanged" in source
