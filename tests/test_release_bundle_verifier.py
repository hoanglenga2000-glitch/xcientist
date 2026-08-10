from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_verifier():
    path = ROOT / "scripts" / "verify_release_bundle.py"
    spec = importlib.util.spec_from_file_location("release_bundle_verifier_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_payload(path: Path, value: str | bytes = "fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def finalize_inner_bundle(bundle: Path, verifier, version: str = "0.3.0") -> dict:
    for name in verifier.INTEGRITY_FILES:
        (bundle / name).unlink(missing_ok=True)
    files = []
    for path in sorted((item for item in bundle.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(bundle).as_posix()
        size = path.stat().st_size
        files.append({"path": relative, "size": size, "bytes": size, "sha256": verifier.sha256(path)})
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(f"{item['path']}\0{item['sha256']}\0{item['size']}\n".encode("utf-8"))
    manifest = {
        "schema": verifier.INNER_MANIFEST_SCHEMA,
        "schema_version": verifier.INNER_SCHEMA_VERSION,
        "product": "EvoMind",
        "version": version,
        "platform": "win-x64",
        "git_commit": "b" * 40,
        "source_digest": "a" * 64,
        "build_id": "fixture-build-id",
        "build_utc": "2026-08-10T00:00:00Z",
        "runtime_compatibility": {
            "node": {"bundled": "v22.0.0", "minimum": "20.9.0"},
            "python": {
                "bundled": "3.12.9/cp312/win_amd64/bundled",
                "minimum": "3.12",
                "maximum_exclusive": "3.13",
            },
        },
        "payload_bytes": sum(item["bytes"] for item in files),
        "file_count": len(files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }
    write_json(bundle / "release-manifest.json", manifest)
    manifest_digest = verifier.sha256(bundle / "release-manifest.json")
    (bundle / "release-manifest.sha256").write_text(
        f"{manifest_digest}  release-manifest.json\n", encoding="ascii"
    )
    sums = {item["path"]: item["sha256"] for item in files}
    sums["release-manifest.json"] = manifest_digest
    write_json(bundle / "SHA256SUMS.json", sums)
    return manifest


def make_inner_bundle(tmp_path: Path, verifier, version: str = "0.3.0") -> Path:
    bundle = tmp_path / "EvoMind"
    for directory in (
        "app/.next/static",
        "app/public",
        "app/prisma/migrations",
        "runtime/python/src/xsci",
    ):
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    payloads = {
        "app/server.js": "server",
        "app/.next/BUILD_ID": "fixture-build-id\n",
        "app/prisma/migrations/20260810000000_baseline/migration.sql": (
            'CREATE TABLE "tasks" ("id" TEXT PRIMARY KEY NOT NULL);\n'
        ),
        "runtime/node/node.exe": b"node",
        "runtime/python/python.exe": b"python",
        "runtime/python/src/xsci/__init__.py": "",
        "scripts/release_db_migrate.py": "pass\n",
        "scripts/manage_workstation_dashboard.py": "pass\n",
        "install.ps1": "exit 0\n",
        "start.ps1": "exit 0\n",
        "stop.ps1": "exit 0\n",
        "status.ps1": "exit 0\n",
        "runtime/wheels/xcientist-0.3.0-py3-none-any.whl": b"wheel",
        "runtime/python/pyproject.toml": (
            '[project]\nname = "xcientist"\nversion = "0.3.0"\nrequires-python = ">=3.10"\n'
        ),
        "runtime/python/uv.lock": (
            'version = 1\nrequires-python = ">=3.10"\n'
            '[[package]]\nname = "xcientist"\nversion = "0.3.0"\n'
        ),
        "metadata/python-locked-requirements.txt": "fixture==1.0.0 --hash=sha256:" + "0" * 64 + "\n",
    }
    for relative, value in payloads.items():
        write_payload(bundle / relative, value)

    wheel = bundle / "runtime/wheels/xcientist-0.3.0-py3-none-any.whl"
    write_json(bundle / "runtime/wheels/wheelhouse-manifest.json", {
        "schema": "evomind.release.wheelhouse-manifest.v2",
        "platform": "win_amd64",
        "python": "3.12",
        "abi": "cp312",
        "uv_lock_sha256": verifier.sha256(bundle / "runtime/python/uv.lock"),
        "locked_requirements_sha256": verifier.sha256(
            bundle / "metadata/python-locked-requirements.txt"
        ),
        "wheel_count": 1,
        "wheels": [{
            "name": wheel.name,
            "size": wheel.stat().st_size,
            "sha256": verifier.sha256(wheel),
            "origin": "first-party-source-build",
            "package": "xcientist",
            "version": version,
        }],
    })
    write_json(bundle / "metadata/package-lock.json", {
        "name": "research-agent-workstation",
        "version": version,
        "lockfileVersion": 3,
        "requires": True,
        "packages": {
            "": {"name": "research-agent-workstation", "version": version},
        },
    })
    write_json(bundle / "metadata/node-sbom.cdx.json", {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "research-agent-workstation",
                "version": version,
            },
        },
        "components": [{"type": "library", "name": "fixture", "version": "1.0.0"}],
    })
    write_json(
        bundle / "metadata/release-contract.json",
        json.loads((ROOT / "configs/release/release-contract.json").read_text(encoding="utf-8")),
    )
    scratch = tmp_path / "schema-scratch"
    migration = verifier.migrate(
        scratch / "workstation.db",
        bundle / "app/prisma/migrations",
    )
    assert migration["ok"] is True
    database_identity = verifier.runtime_schema_identity(
        scratch / "workstation.db",
        bundle / "app/prisma/migrations",
    )
    write_json(bundle / "app/runtime-build-manifest.json", {
        "schema": "evomind.runtime_build.v1",
        "commit_hash": "b" * 40,
        "source_dirty": False,
        "source_tree_sha256": "a" * 64,
        "build_id": "fixture-build-id",
        "build_time": "2026-08-10T00:00:00Z",
        "backend_version": version,
        "frontend_version": version,
        "database_schema_version": database_identity["version"],
        "database_schema_sha256": database_identity["sha256"],
    })
    shutil.rmtree(scratch)
    finalize_inner_bundle(bundle, verifier, version)
    return bundle


def outer_manifest(verifier, version: str = "0.3.0") -> dict:
    return {"schema": verifier.OUTER_MANIFEST_SCHEMA, "version": version}


def test_inner_manifest_exactly_covers_bundle_and_runtime_metadata(tmp_path: Path) -> None:
    verifier = load_verifier()
    bundle = make_inner_bundle(tmp_path, verifier)

    manifest = verifier.verify_inner_manifest(bundle, outer_manifest(verifier))

    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["payload_bytes"] == sum(item["bytes"] for item in manifest["files"])


def test_inner_manifest_rejects_extra_files_and_outer_version_drift(tmp_path: Path) -> None:
    verifier = load_verifier()
    bundle = make_inner_bundle(tmp_path, verifier)
    write_payload(bundle / "unexpected.txt", "not declared")

    with pytest.raises(RuntimeError, match="file-tree coverage mismatch"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier))

    (bundle / "unexpected.txt").unlink()
    with pytest.raises(RuntimeError, match="inner and outer release versions differ"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier, "0.3.1"))


def test_inner_manifest_rejects_control_hash_and_summary_drift(tmp_path: Path) -> None:
    verifier = load_verifier()
    bundle = make_inner_bundle(tmp_path, verifier)
    (bundle / "release-manifest.sha256").write_text("0" * 64 + "  release-manifest.json\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="release-manifest.sha256"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier))

    finalize_inner_bundle(bundle, verifier)
    sums = json.loads((bundle / "SHA256SUMS.json").read_text(encoding="utf-8"))
    sums["extra"] = "0" * 64
    write_json(bundle / "SHA256SUMS.json", sums)
    with pytest.raises(RuntimeError, match="SHA256SUMS"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier))


def test_inner_manifest_rejects_aggregate_and_semantic_metadata_drift(tmp_path: Path) -> None:
    verifier = load_verifier()
    bundle = make_inner_bundle(tmp_path, verifier)
    manifest_path = bundle / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregate_sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="aggregate_sha256"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier))

    bundle = make_inner_bundle(tmp_path / "semantic", verifier)
    sbom_path = bundle / "metadata/node-sbom.cdx.json"
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    sbom["bomFormat"] = "unknown"
    write_json(sbom_path, sbom)
    finalize_inner_bundle(bundle, verifier)
    with pytest.raises(RuntimeError, match="CycloneDX"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier))


def test_inner_manifest_rejects_stale_release_runtime_contract(tmp_path: Path) -> None:
    verifier = load_verifier()
    bundle = make_inner_bundle(tmp_path, verifier)
    contract_path = bundle / "metadata/release-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["format_version"] = 1
    write_json(contract_path, contract)
    finalize_inner_bundle(bundle, verifier)

    with pytest.raises(RuntimeError, match="release runtime contract is incomplete"):
        verifier.verify_inner_manifest(bundle, outer_manifest(verifier))


def test_cli_is_installed_from_real_npm_tgz_into_disposable_prefix(tmp_path: Path) -> None:
    verifier = load_verifier()
    npm_found = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm_found:
        pytest.skip("npm is not installed")
    # Resolve drive-root-relative PATH shims before switching cwd to tmp_path.
    # A Windows entry such as ``\下载\npm.cmd`` otherwise follows the temporary
    # directory's drive and points at a different, nonexistent executable.
    npm_value = str(Path(npm_found).resolve())
    source = tmp_path / "source"
    write_json(source / "package.json", {
        "name": "@evomind-ai/cli",
        "version": "0.3.0",
        "type": "module",
        "bin": {"evomind": "bin/evomind.mjs"},
        "files": ["bin", "keys"],
    })
    write_payload(source / "bin/evomind.mjs", "process.stdout.write('{}\\n');\n")
    write_payload(
        source / "keys/release-ed25519-public.pem",
        "-----BEGIN PUBLIC KEY-----\nfixture\n-----END PUBLIC KEY-----\n",
    )
    (tmp_path / "packed").mkdir()
    packed = subprocess.run(
        [npm_value, "pack", "--json", "--pack-destination", str(tmp_path / "packed")],
        cwd=source,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    metadata = json.loads(packed.stdout)
    tgz = tmp_path / "packed" / metadata[0]["filename"]

    cli, package = verifier.install_cli_from_tgz(tgz, tmp_path / "prefix", Path(npm_value), os.environ.copy())

    assert package["name"] == "@evomind-ai/cli"
    assert cli.read_text(encoding="utf-8") == (source / "bin/evomind.mjs").read_text(encoding="utf-8")
    assert (tmp_path / "prefix").resolve() in cli.resolve().parents
    assert source.resolve() not in cli.resolve().parents


def test_cli_installer_rejects_repository_source_entrypoint(tmp_path: Path) -> None:
    verifier = load_verifier()
    source_cli = tmp_path / "packages/evomind-cli/bin/evomind.mjs"
    write_payload(source_cli, "")
    npm = Path(shutil.which("npm.cmd") or shutil.which("npm") or source_cli)

    with pytest.raises(RuntimeError, match="--cli-tgz"):
        verifier.install_cli_from_tgz(source_cli, tmp_path / "prefix", npm, os.environ.copy())


def test_release_cleanup_removes_long_content_addressed_cache_paths(tmp_path: Path) -> None:
    verifier = load_verifier()
    cache = tmp_path / "npm-cache" / "_cacache" / "content-v2" / "sha512" / "f5" / "5f"
    payload = cache / ("a" * 128)
    write_payload(payload, b"fixture")

    verifier.remove_tree_with_retry(tmp_path)

    assert not tmp_path.exists()
