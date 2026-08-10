from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_release_bundle.ps1"


def _builder_text() -> str:
    return BUILDER.read_text(encoding="utf-8-sig")


def test_output_containment_is_segment_aware_and_reparse_safe() -> None:
    text = _builder_text()

    assert "[IO.Path]::GetRelativePath($parentFull, $candidateFull)" in text
    assert "$segments[0] -eq \"..\"" in text
    assert "[IO.FileAttributes]::ReparsePoint" in text
    assert ".StartsWith($ArtifactsRoot" not in text


def test_publish_uses_owned_random_staging_explicit_replacement_and_rollback() -> None:
    text = _builder_text()

    assert "[switch]$ReplaceExisting" in text
    assert '".evomind-release-staging-$TransactionId"' in text
    assert "$SentinelName = \".evomind-release-root.json\"" in text
    assert "Read-OwnedOutputSentinel" in text
    assert "[IO.Directory]::Move($OutputRoot, $RollbackRoot)" in text
    assert "[IO.Directory]::Move($StagingRoot, $OutputRoot)" in text
    assert "[IO.Directory]::Move($RollbackRoot, $OutputRoot)" in text
    assert "Remove-Item -LiteralPath $OutputRoot" not in text


def test_release_build_cannot_reuse_unbound_next_output() -> None:
    text = _builder_text()

    assert "if ($SkipBuild)" in text
    assert "SkipBuild is disabled for release publishing" in text
    assert "next_tree_sha256" in text
    assert "build_id = $BuildId" in text
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in text
    assert "source_dirty = $false" in text


def test_native_calls_are_wrapped_and_npm_ls_problems_are_rejected() -> None:
    text = _builder_text()

    assert "$exitCode = $LASTEXITCODE" in text
    assert text.count("$exitCode = $LASTEXITCODE") == 2
    assert "Invoke-NativeCommand -FilePath $Npm" in text
    assert "Invoke-NativeText -FilePath $Git" in text
    assert "Invoke-NativeCommand -FilePath $HostPython" in text
    assert '$DependencyTree.PSObject.Properties["problems"]' in text
    # Ignore hashtable/property assignments such as ``node = $NodeVersion``;
    # only flag a bare executable followed by command arguments.
    raw_native = re.compile(
        r"^\s*(?:git|npm|node|python|pip|uv)(?:\.exe)?\s+(?![=:])",
        re.MULTILINE | re.IGNORECASE,
    )
    assert raw_native.search(text) is None


def test_builder_uses_repo_contracts_and_hash_locked_exact_wheelhouse() -> None:
    text = _builder_text()

    assert '"configs\\release\\toolchain.json"' in text
    assert '"configs\\release\\release-contract.json"' in text
    assert "artifacts\\launch-fix" not in text
    assert '"export", "--frozen"' in text
    assert text.count('"--require-hashes"') >= 2
    assert '"--dry-run", "--ignore-installed"' in text
    assert "pip-wheelhouse-validation.json" in text


def test_toolchain_contract_pins_patch_versions() -> None:
    contract = json.loads((ROOT / "configs/release/toolchain.json").read_text(encoding="utf-8"))

    assert re.fullmatch(r"\d+\.\d+\.\d+", contract["node"]["version"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", contract["npm"]["version"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", contract["python"]["version"])
    assert contract["python"]["abi"] == "cp312"
    assert contract["python"]["platform"] == "win_amd64"


def test_release_contract_is_repository_owned_and_publication_time_is_separate() -> None:
    contract = json.loads((ROOT / "configs/release/release-contract.json").read_text(encoding="utf-8"))

    assert contract["schema"] == "evomind.release.contract.v2"
    assert contract["builder"]["atomic_directory_switch"] is True
    assert contract["builder"]["rollback_retained"] is True
    assert contract["reproducibility"]["published_time_outside_bundle"] is True
    assert contract["reproducibility"]["runtime_build_manifest_bound_to_source"] is True
    assert contract["supply_chain"]["wheelhouse_exact_set_required"] is True
    assert contract["web"]["runtime_build_manifest"] == "app/runtime-build-manifest.json"
