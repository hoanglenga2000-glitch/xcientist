from __future__ import annotations

import json
from pathlib import Path
import pytest
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r119-g21-hpc-asset-probe-r118-base-final-20260831")
BASELINE = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r118-g21-open-polymer-legacy-adoption-r117-base-final-20260831\evomind-runtime-r118-g21-open-polymer-legacy-adoption-r117-base.zip")
PREFIX = "evomind-runtime-r119-g21-hpc-asset-probe-r118-base"
WRAPPER = ROOT / "scripts/Deploy-G21HpcAssetProbeRuntimeR119.ps1"
RUNNER = ROOT / "scripts/Deploy-G21HpcAssetProbeRuntimeR119Remote.ps1"
BOARD_SHA = "6b908173083889ac21f8afea52528b4b5e35d2ff862a880a6065d65b169f443c"


DEPLOY_ARTIFACTS_ROOT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts")


def _require_deploy_artifacts() -> None:
    # Frozen release artifacts live in the out-of-repo deploy archive. Skip only when the
    # whole archive is absent on this machine; a missing file inside an existing archive
    # still fails the release-binding assertions below.
    if not DEPLOY_ARTIFACTS_ROOT.is_dir():
        pytest.skip("EvoMind-Cloud-Deploy release artifacts archive is not present on this machine")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=180)


def test_candidate_is_exact_one_file_overlay_with_independent_receipt() -> None:
    _require_deploy_artifacts()
    build = json.loads((ARTIFACT / f"{PREFIX}-build-result.json").read_text(encoding="utf-8"))
    receipt = json.loads((ARTIFACT / f"{PREFIX}-independent-receipt.json").read_text(encoding="utf-8"))
    assert build["status"] == "built_and_verified"
    assert build["zip_sha256"] == "9368c7962632fb976e7a8ba8ede0d3108989e96c00c4dd75bca154008d438206"
    assert build["candidate_runtime_tree_sha256"] == "906868d22a841f591fb84c7a505b75acee3c8df58f179e83d3d4585601d03975"
    assert build["candidate_runtime_file_count"] == 32
    assert build["changed_file_count"] == 1
    assert build["unchanged_file_count"] == 31
    assert build["changed_files"] == ["evomind_runtime/tools.py"]
    assert all(row["py_compile"] == 32 for row in build["verification_rounds"])
    assert receipt["status"] == "verified"
    assert receipt["candidate_zip_sha256"] == build["zip_sha256"]
    assert receipt["junit_receipts"][0]["tests"] == 114
    assert receipt["junit_receipts"][0]["failures"] == 0
    assert receipt["production_deployed"] is False
    assert receipt["hpc_accessed"] is False


def test_builder_dry_run_reconstructs_from_r118_without_dirty_tools_copy(tmp_path: Path) -> None:
    _require_deploy_artifacts()
    result = _run(
        sys.executable,
        "scripts/build_g21_hpc_asset_probe_runtime_r119.py",
        "--baseline-zip",
        str(BASELINE),
        "--output-dir",
        str(tmp_path / "unused"),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["changed_files"] == ["evomind_runtime/tools.py"]
    assert payload["changed_file_count"] == 1
    assert payload["unchanged_file_count"] == 31
    source = (ROOT / "scripts/build_g21_hpc_asset_probe_runtime_r119.py").read_text(encoding="utf-8")
    assert "_extract(" in source and "_ast_scope(" in source
    assert 'added != {"_hpc_asset_probe_remote_source", "_parse_hpc_asset_probe_payload", "_hpc_asset_probe"}' in source


def test_remote_deployer_is_runtime_only_and_preserves_goal_database_and_hpc() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert BOARD_SHA in wrapper and BOARD_SHA in runner
    assert "R119_CURRENT_RUNTIME_NOT_R118" in runner
    assert "Read-GoalGate" in runner
    assert "Read-ToolSchema" in runner
    assert "hpc_asset_probe" in runner
    assert "database_mutation_attempted=$false" in runner
    assert "database_raw_hash_compared_after_restart=$false" in runner
    assert "goal_unchanged=$true" in runner
    assert "goal_put_calls=0" in runner
    assert "hpc_tool_calls=0" in runner
    assert "single_scp_envelope" in wrapper
    assert "ZipFile]::CreateFromDirectory" in wrapper
    assert wrapper.count("scp ") == 3
    assert "single_ssh_envelope" not in wrapper
    assert "foreach($item in $uploads){scp" not in wrapper
    assert "upload-envelope.zip" in wrapper and "Expand-Archive" in wrapper
    assert "Invoke-ManagedAction Stop" in runner and "Invoke-ManagedAction Start" in runner
    assert "Backup-Database" in runner and "Restore-Database" in runner
    for forbidden in (
        "Invoke-LegacyAdoption",
        "Read-LegacyAdoptionState",
        "goal.policy_migrated",
        "'goal.updated'",
        "-Method Put",
        "hpc_verify",
        "hpc_execute_solution",
        "Start-ScheduledTask",
        "Stop-Process",
        "Get-FileHash",
    ):
        assert forbidden not in runner
    assert "validation_only_non_authorizing" in (ROOT / "scripts/new_g21_hpc_asset_probe_runtime_r119_deployment_approval.py").read_text(encoding="utf-8")


def test_ps7_and_ps5_parse_wrapper_and_remote() -> None:
    for executable in (shutil.which("pwsh"), shutil.which("powershell.exe")):
        assert executable
        for script in (WRAPPER, RUNNER):
            command = (
                "$t=$null;$e=$null;"
                f"[Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$t,[ref]$e)|Out-Null;"
                "if(@($e).Count){$e|ForEach-Object{$_.Message};exit 1}else{'PARSE_OK'}"
            )
            result = _run(str(executable), "-NoProfile", "-NonInteractive", "-Command", command)
            assert result.returncode == 0, result.stderr + result.stdout
            assert "PARSE_OK" in result.stdout


def test_fresh_validation_only_approval_passes_ps7_and_ps5_without_production(tmp_path: Path) -> None:
    _require_deploy_artifacts()
    output = tmp_path / "approval"
    result = _run(
        sys.executable,
        "scripts/new_g21_hpc_asset_probe_runtime_r119_deployment_approval.py",
        "--candidate-zip", str(ARTIFACT / f"{PREFIX}.zip"),
        "--source-manifest", str(ARTIFACT / f"{PREFIX}-source-manifest.json"),
        "--build-result", str(ARTIFACT / f"{PREFIX}-build-result.json"),
        "--test-receipt", str(ARTIFACT / f"{PREFIX}-independent-receipt.json"),
        "--old-spec", "configs/g21_five_competition_goal_v1_frozen.json",
        "--new-spec", "configs/g21_five_competition_goal.json",
        "--policy-evidence", "configs/g21_conditional_strong_baseline_policy_evidence.json",
        "--verifier", "scripts/verify_g21_hpc_asset_probe_runtime_r119_candidate.py",
        "--wrapper", str(WRAPPER),
        "--remote-runner", str(RUNNER),
        "--output-dir", str(output),
        "--validate-only",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    approval = output / "evomind-runtime-r119-g21-hpc-asset-probe-deployment-approval.json"
    document = json.loads(approval.read_text(encoding="utf-8"))
    assert document["status"] == "validation_only_non_authorizing"
    assert document["production_deployable"] is False
    assert document["production_invariants"]["goal_board_sha256"] == BOARD_SHA
    for executable in (shutil.which("pwsh"), shutil.which("powershell.exe")):
        validated = _run(str(executable), "-NoProfile", "-NonInteractive", "-File", str(WRAPPER), "-ApprovalPath", str(approval), "-ValidateOnly")
        assert validated.returncode == 0, validated.stderr + validated.stdout
        payload = json.loads(validated.stdout.strip().splitlines()[-1])
        assert payload["status"] == "validated_not_deployed"
        assert payload["production_action_performed"] is False
        assert payload["hpc_accessed"] is False
        assert payload["gpu_touched"] is False
