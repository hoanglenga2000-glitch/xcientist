from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Install-SealReleaseWebOverlayScopeR1Remote.ps1"


def test_hotfix_removes_recursive_web_overlay_parent_seal() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    replacement = source.split("$newBlock = @'", 1)[1].split("'@", 1)[0]

    assert "'C:\\ProgramData\\EvoMind\\web-overlays') | Where-Object" not in replacement
    assert "$webOverlayRoot = 'C:\\ProgramData\\EvoMind\\web-overlays'" in replacement
    assert "recursive_parent_scan=$false" in replacement
    assert "operational-overlay-manifest.json" in replacement
    assert "runtime-build-manifest.json" in replacement
    assert "WEB_OVERLAY_ACTIVE_IDENTITY_REJECTED" in replacement
    assert "Resolve-WebRuntimeIdentity" not in replacement
    assert "Set-SealedTreeAcl -Path $webOverlayRoot" not in replacement
    assert "'/T'" not in replacement


def test_hotfix_is_transactional_and_preserves_external_boundaries() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "SEAL_SCOPE_SOURCE_SHA_REJECTED" in source
    assert "Assert-PowerShellAst $candidate" in source
    assert "[switch]$PreflightOnly" in source
    assert "status='preflight_verified'" in source
    assert "Global\\EvoMind-Byoa-V12-Deployment" in source
    assert "SEAL_SCOPE_HPC_BINDING_CHANGED" in source
    assert "service_restarted = $false" in source
    assert "hpc_accessed = $false" in source
    assert "gpu_touched = $false" in source
