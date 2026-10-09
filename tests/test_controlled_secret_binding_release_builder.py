from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Build-ControlledSecretBindingReleaseR112.ps1"
OLD_ZIP = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts\overlay-controlled-secret-binding-r112.zip")
OLD_ZIP_SHA256 = "10e59a178ac4594d5157914f379af08883eae8c347fc2df605c9d8de110c2ff8"


DEPLOY_ARTIFACTS_ROOT = Path(r"D:\AI-Outputs\EvoMind-Cloud-Deploy\artifacts")


def _require_deploy_artifacts() -> None:
    # Frozen release artifacts live in the out-of-repo deploy archive. Skip only when the
    # whole archive is absent on this machine; a missing file inside an existing archive
    # still fails the release-binding assertions below.
    if not DEPLOY_ARTIFACTS_ROOT.is_dir():
        pytest.skip("EvoMind-Cloud-Deploy release artifacts archive is not present on this machine")


def test_old_unverified_zip_is_preserved_and_never_reused() -> None:
    _require_deploy_artifacts()
    assert OLD_ZIP.is_file()
    assert hashlib.sha256(OLD_ZIP.read_bytes()).hexdigest() == OLD_ZIP_SHA256
    source = SCRIPT.read_text(encoding="utf-8")
    assert OLD_ZIP_SHA256 in source
    assert "overlay-controlled-secret-binding-r112-standalone-v5.zip" in source
    assert "Remove-Item -LiteralPath $oldEvidenceZip" not in source


def test_builder_promotes_standalone_root_and_excludes_build_residue() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = [
        "$standaloneRoot = Join-Path $buildRoot '.next\\standalone'",
        "Copy-DirectoryContents $standaloneRoot $releaseRoot",
        "Copy-DirectoryContents (Join-Path $buildRoot '.next\\static')",
        "Copy-DirectoryContents (Join-Path $buildRoot 'public')",
        "Copy-DirectoryContents (Join-Path $buildRoot 'support')",
        "root_server",
        "root_next",
        "nested_standalone",
        "cache_files",
        "Remove-BuildCacheDirectories",
        "file_extra",
        "unsafe_entries",
        "duplicate_entries",
        "symlink_entries",
        "Start-ReleaseSmoke",
    ]
    for marker in required:
        assert marker in source
    assert "Compress-Archive -LiteralPath (Join-Path $stage '*')" not in source
    assert "CreateFromDirectory($releaseRoot" in source
    assert re.search(r"Get-ChildItem[^\n]+-Recurse -File", source)
    assert "R112_RELEASE_ROOT_CLOSURE_REJECTED:root_server=" in source
    assert "cache_samples=" in source
    assert "stderr_sha256=" in source
    assert "C:\\codex-python\\python.exe" in source
    assert source.index("Write-Json $sourceManifestOutput") > source.rindex("Start-ReleaseSmoke")


def test_cache_classifier_rejects_build_residue_but_allows_next_runtime_webpack() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    function = re.search(
        r"function Test-BuildCacheRelative\(\[string\]\$Relative\) \{[\s\S]*?\n\}",
        source,
    )
    assert function is not None
    positive = [
        ".next/cache/webpack/client.pack",
        ".next/dev/server/route.js",
        "node_modules/.cache/compiler/item",
        ".webpack-cache/index.pack",
        "webpack-cache/server/index.pack",
        "build-data/workstation.db",
    ]
    negative = [
        "node_modules/next/dist/build/webpack/cache-invalidation.js",
        "node_modules/next/dist/build/webpack/plugins/flight-manifest-plugin.js",
        ".next/server/chunks/runtime.js",
        ".next/static/chunks/app.js",
        "support/python-runtime/evomind_runtime/tools.py",
    ]
    pwsh = shutil.which("pwsh") or shutil.which("pwsh.exe")
    assert pwsh
    command = (
        function.group(0)
        + "\n$positive=" + json.dumps(positive, ensure_ascii=True).replace('[', '@(').replace(']', ')')
        + "\n$negative=" + json.dumps(negative, ensure_ascii=True).replace('[', '@(').replace(']', ')')
        + "\n[ordered]@{positive=@($positive|ForEach-Object{Test-BuildCacheRelative $_});negative=@($negative|ForEach-Object{Test-BuildCacheRelative $_})}|ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["positive"] == [True] * len(positive)
    assert payload["negative"] == [False] * len(negative)


def test_builder_allowlist_is_exactly_five_functional_files() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"\bthrow[\"']", source)
    expected_hashes = {
        "496617aafd56cdf6eea89fe6fbb807320f8a1ea926737552a0e2cfdf5b8e36ec",
        "ac48f5dc170606b4ee0907a19e5c9757c246461ce9f926cdb8375a1be0eebac2",
        "9486e1941bb9e073018fababffb42795e5e6ab1c34fd61a7304906fc9402dca1",
        "bf52a8fa66a5b15bf06594cfc9714cd5597610048902b2facfc1180c9555a72b",
        "0f4062dffd0828b90be6405b8b68dbe89389165e7e925438badaf49acee84c81",
    }
    for digest in expected_hashes:
        assert source.count(digest) == 1
    assert source.count("relative=") == 5


def test_old_zip_directory_entries_are_not_regular_file_extras() -> None:
    _require_deploy_artifacts()
    with zipfile.ZipFile(OLD_ZIP) as archive:
        manifest = json.loads(archive.read("operational-overlay-manifest.json"))
        regular_files = {item.filename for item in archive.infolist() if not item.is_dir()}
        listed = {item["path"] for item in manifest["files"]}
        assert regular_files == listed | {"operational-overlay-manifest.json"}
        assert sum(item.is_dir() for item in archive.infolist()) == 213
