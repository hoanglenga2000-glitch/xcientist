from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r119-g21-hpc-asset-probe-r118-base-final-20260831\evomind-runtime-r119-g21-hpc-asset-probe-r118-base.zip")
ARTIFACT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\r120-g21-terminal-preserving-probe-r119-base-v3-20260901")
ZIP = ARTIFACT / "evomind-runtime-r120-g21-terminal-preserving-probe-r119-base.zip"
MANIFEST = ARTIFACT / "evomind-runtime-r120-g21-terminal-preserving-probe-r119-base-source-manifest.json"
BUILD = ARTIFACT / "evomind-runtime-r120-g21-terminal-preserving-probe-r119-base-build-result.json"
BASELINE_SHA = "9368c7962632fb976e7a8ba8ede0d3108989e96c00c4dd75bca154008d438206"
BASELINE_TREE = "906868d22a841f591fb84c7a505b75acee3c8df58f179e83d3d4585601d03975"
CANDIDATE_TREE = "5998ea21823c121c23a421bb145d8e2266bb79ab59eccc91a16122c3626b61ce"
SOURCE_SHA = "519820314f63d2e5ae6ba354e044ea7df55c544a6c0ab85de43102279af0af4d"
CANDIDATE_HTTP_SHA = "6d2d17f1257736683c4bc590f346c183c9feb25e06cce7a77d63d35843e23a9d"


DEPLOY_ARTIFACTS_ROOT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts")


def _require_deploy_artifacts() -> None:
    # Frozen release artifacts live in the out-of-repo deploy archive. Skip only when the
    # whole archive is absent on this machine; a missing file inside an existing archive
    # still fails the release-binding assertions below.
    if not DEPLOY_ARTIFACTS_ROOT.is_dir():
        pytest.skip("EvoMind-Cloud-Deploy release artifacts archive is not present on this machine")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r120_candidate_is_exact_http_server_overlay_on_r119() -> None:
    _require_deploy_artifacts()
    assert sha(BASELINE) == BASELINE_SHA
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    assert manifest["schema"] == "evomind.g21_terminal_preserving_probe_runtime.r120.v1"
    assert manifest["baseline_runtime_tree_sha256"] == BASELINE_TREE
    assert manifest["candidate_runtime_tree_sha256"] == CANDIDATE_TREE
    assert manifest["source_http_server_raw_sha256"] == SOURCE_SHA
    assert manifest["source_http_server_packaged_sha256"] == CANDIDATE_HTTP_SHA
    assert manifest["changed_files"] == ["evomind_runtime/http_server.py"]
    assert manifest["changed_file_count"] == 1
    assert manifest["unchanged_file_count"] == 31
    assert manifest["cancel_workaround_forbidden"] is True
    assert build["status"] == "built_and_verified"
    assert len(build["verification_rounds"]) == 2
    assert all(row["py_compile"] == 32 for row in build["verification_rounds"])
    assert all(row["terminal_preserving_smoke"]["success_and_failure_preserved"] for row in build["verification_rounds"])
    with zipfile.ZipFile(BASELINE) as old_zip, zipfile.ZipFile(ZIP) as new_zip:
        old = {name: old_zip.read(name) for name in old_zip.namelist() if name.startswith("evomind_runtime/")}
        new = {name: new_zip.read(name) for name in new_zip.namelist() if name.startswith("evomind_runtime/")}
    assert set(old) == set(new)
    changed = [name for name in sorted(old) if old[name] != new[name]]
    assert changed == ["evomind_runtime/http_server.py"]
    assert hashlib.sha256(new[changed[0]]).hexdigest() == CANDIDATE_HTTP_SHA
    declared = {item["path"]: item for item in manifest["files"]}
    assert declared[changed[0]]["sha256"] == CANDIDATE_HTTP_SHA


def test_builder_dry_run_is_bound_to_r119_and_exact_ast_scope(tmp_path: Path) -> None:
    _require_deploy_artifacts()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_g21_terminal_preserving_probe_runtime_r120.py"),
            "--baseline-zip", str(BASELINE),
            "--output-dir", str(tmp_path / "must-not-exist"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run_verified"
    assert payload["candidate_runtime_tree_sha256"] == CANDIDATE_TREE
    assert payload["changed_files"] == ["evomind_runtime/http_server.py"]
    assert not (tmp_path / "must-not-exist").exists()
    source = (ROOT / "scripts/build_g21_terminal_preserving_probe_runtime_r120.py").read_text(encoding="utf-8")
    assert 'added_defs != {"_terminal_direct_tool_state", "_invoke_direct_tool"}' in source
    assert 'changed_defs != {"make_handler"}' in source
    assert SOURCE_SHA in source and CANDIDATE_HTTP_SHA in source


def test_deployer_is_runtime_only_and_never_calls_hpc_or_probe() -> None:
    wrapper = (ROOT / "scripts/Deploy-G21TerminalPreservingProbeRuntimeR120.ps1").read_text(encoding="utf-8")
    remote = (ROOT / "scripts/Deploy-G21TerminalPreservingProbeRuntimeR120Remote.ps1").read_text(encoding="utf-8")
    assert "single_scp_envelope" in wrapper
    assert "Invoke-ServiceAccountAction.ps1" in remote
    assert "Backup-Database" in remote and "Restore-Database" in remote
    assert "Global\\EvoMind-Byoa-V12-Deployment" in remote
    assert "evomind_runtime/http_server.py" in wrapper and "evomind_runtime/http_server.py" in remote
    assert CANDIDATE_TREE in wrapper and CANDIDATE_TREE in remote
    assert "/v1/sessions/$fixedRun/tools" not in remote
    assert "Post-Api" not in remote
    assert "Start-ScheduledTask" not in remote
    assert "Stop-Process" not in remote
    assert "10.120." not in wrapper + remote
    assert "return[" not in remote


def test_release_scripts_compile() -> None:
    scripts = [
        ROOT / "scripts/build_g21_terminal_preserving_probe_runtime_r120.py",
        ROOT / "scripts/verify_g21_terminal_preserving_probe_runtime_r120_candidate.py",
        ROOT / "scripts/new_g21_terminal_preserving_probe_runtime_r120_deployment_approval.py",
    ]
    for path in scripts:
        completed = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, f"{path}: {completed.stderr}"
