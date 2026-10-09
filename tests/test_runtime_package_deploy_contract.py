from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_package_deployer_accepts_only_terminal_quiescent_run_states() -> None:
    remote = (ROOT / "scripts" / "Deploy-RuntimePackageR1Remote.ps1").read_text(encoding="utf-8-sig")
    wrapper = (ROOT / "scripts" / "Deploy-RuntimePackageR30.ps1").read_text(encoding="utf-8-sig")
    terminal_states = "@('blocked','failed','cancelled')"

    assert terminal_states in remote
    assert terminal_states in wrapper
    assert "@($snapshot.active_tool_calls).Count-ne0" in remote
    assert "$pendingApprovalCount-ne0" in remote
    assert "pending_approval_count=0" in remote
    assert "[int]$result.target_run_preflight.active_tool_count-eq0" in wrapper
    assert "[int]$result.target_run_preflight.pending_approval_count-eq0" in wrapper
    assert "RUNTIME_PACKAGE_TARGET_RUN_NOT_QUIESCENT" in remote
    for unsafe in ("'running'", "'recovering'", "'waiting_approval'", "'completed'"):
        assert unsafe not in terminal_states


def test_cancelled_run_remains_bound_to_the_exact_expected_run() -> None:
    remote = (ROOT / "scripts" / "Deploy-RuntimePackageR1Remote.ps1").read_text(encoding="utf-8-sig")
    assert "[string]$snapshot.id-ne$ExpectedRunId" in remote
    assert "if(-not$ExpectedRunId){return $null}" in remote
    assert "target_run_preflight=$targetRunPreflight" in remote


def test_r113_deployer_updates_and_restores_runtime_build_manifest_atomically() -> None:
    remote = (ROOT / "scripts" / "Deploy-RuntimePackageR113Remote.ps1").read_text(encoding="utf-8-sig")
    wrapper = (ROOT / "scripts" / "Deploy-RuntimePackageR113.ps1").read_text(encoding="utf-8-sig")

    assert "ExpectedCurrentBuildId" in remote
    assert "-ExpectedCurrentBuildId $ExpectedCurrentBuildId" in wrapper
    assert "Deploy-RuntimePackageR113Remote.ps1" in wrapper
    assert "b43185064b18791518662e3a526d68dda59e4a8c18bac4a2d3fa8bc0d9fd7d8c" in wrapper
    assert "Start-ManagedNodeVerified" in remote
    assert "long_start_verified" in remote
    assert "AddMinutes(25)" in remote
    assert "$runtimeManifest.build_id=$ExpectedBuildId" not in remote
    assert "$overlayManifest.overlay_id=$ExpectedBuildId" not in remote
