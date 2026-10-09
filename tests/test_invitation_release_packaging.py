from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_invitation_release.py"
    spec = importlib.util.spec_from_file_location("invitation_package_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/absolute", "web\\escape", "web//ambiguous"])
def test_archive_paths_are_not_normalized_into_an_allowed_path(name):
    with pytest.raises(ValueError, match="candidate_archive_path_rejected"):
        builder().safe_name(name)


def make_baseline(path, *, tamper=False, extra=False):
    payload = b"fixture = True\n"
    manifest = {"schema": "evomind.invitation_baseline.v1", "web_build_id": "fixture-build", "file_count": 1,
                "files": [{"path": "runtime/fixture.py", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("runtime/fixture.py", b"changed" if tamper else payload)
        bundle.writestr("baseline-manifest.json", json.dumps(manifest))
        if extra:
            bundle.writestr("runtime/unlisted.py", "unlisted")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_is_hash_bound_and_restores_only_declared_sources(tmp_path):
    archive = tmp_path / "baseline.zip"
    digest = make_baseline(archive)
    output = tmp_path / "staged"
    result = builder().unpack_baseline(archive, digest, output, "fixture-build")
    assert result["file_count"] == 1
    assert (output / "runtime/fixture.py").read_bytes() == b"fixture = True\n"


@pytest.mark.parametrize("fault", ["archive_hash", "file_hash", "extra_file", "build_identity"])
def test_baseline_mismatch_stops_before_becoming_a_candidate(tmp_path, fault):
    archive = tmp_path / "baseline.zip"
    digest = make_baseline(archive, tamper=fault == "file_hash", extra=fault == "extra_file")
    with pytest.raises(ValueError, match="baseline_"):
        builder().unpack_baseline(archive, "0" * 64 if fault == "archive_hash" else digest,
                                 tmp_path / "staged", "different-build" if fault == "build_identity" else "fixture-build")


def release_fixture(tmp_path):
    build = builder()
    release, runtime = tmp_path / "release", tmp_path / "runtime"
    for directory in (release / "node_modules/.prisma/client", release / "support/python-runtime/evomind_runtime", runtime / "evomind_runtime"):
        directory.mkdir(parents=True)
    (release / "server.js").write_text("fixture-server")
    schema = release / "node_modules/.prisma/client/schema.prisma"
    schema.write_text("fixture-schema")
    for directory in (release / "support/python-runtime/evomind_runtime", runtime / "evomind_runtime"):
        (directory / "fixture.py").write_text("fixture = True\n")
    rows = [{"path": "web/fixture", "bytes": 1, "sha256": "1" * 64}]
    source = {"schema": "evomind.invitation_source.v1", "build_id": "fixture-new", "source_tree_sha256": build.tree_sha(rows),
              "production_schema_sha256": build.sha(schema), "database_schema_sha256": "d" * 64,
              "files": rows, "changes": [], "work_root": str(tmp_path)}
    commit = "664a636ddd419a66f73cc10820c0a16429784866"
    identity = {"schema": "evomind.runtime_build.v1", "commit_hash": commit, "backend_version": "0.3.0",
                "frontend_version": "0.3.0", "database_schema_version": "fixture", "database_schema_sha256": "0" * 64}
    operational = {"schema": "evomind.web_operational_overlay.v1", "base_commit": commit, "entrypoint": "server.js",
                   "source_identity_sha256": "0" * 64}
    build.publish_release_manifests(release, runtime, source, identity, operational)
    return build, release, runtime, source


def test_release_replaces_stale_source_and_schema_identities(tmp_path):
    build, release, runtime, source = release_fixture(tmp_path)
    accepted = build.validate_release_identity(release, runtime, source)
    assert accepted["source_identity_sha256"] == source["source_tree_sha256"]
    assert accepted["database_schema_sha256"] == source["database_schema_sha256"]
    assert accepted["prisma_schema_source_sha256"] == source["production_schema_sha256"]
    assert accepted["database_schema_sha256"] != accepted["prisma_schema_source_sha256"]
    assert accepted["production_startup_preflight"] == "pending"


@pytest.mark.parametrize("fault", ["source_identity", "runtime_identity", "schema_identity", "schema_version", "build_id", "missing_file", "changed_file", "extra_file", "runtime_mirror", "source_ledger"])
def test_release_contract_rejects_inconsistent_or_modified_candidates(tmp_path, fault):
    build, release, runtime, source = release_fixture(tmp_path)
    if fault in {"source_identity", "runtime_identity", "schema_identity", "schema_version", "build_id"}:
        path = release / ("operational-overlay-manifest.json" if fault == "source_identity" else "runtime-build-manifest.json")
        value = json.loads(path.read_text())
        key = {"source_identity": "source_identity_sha256", "runtime_identity": "source_tree_sha256",
               "schema_identity": "database_schema_sha256", "schema_version": "database_schema_version", "build_id": "build_id"}[fault]
        value[key] = "0" * 64 if fault.endswith("identity") else ""
        build.json_file(path, value)
    elif fault == "missing_file":
        (release / "server.js").unlink()
    elif fault == "changed_file":
        (release / "server.js").write_text("tampered")
    elif fault == "extra_file":
        (release / "unlisted.js").write_text("extra")
    elif fault == "runtime_mirror":
        (runtime / "evomind_runtime/fixture.py").write_text("changed")
    else:
        source["files"][0]["bytes"] = 2
    with pytest.raises(ValueError, match="candidate_"):
        build.validate_release_identity(release, runtime, source)


def test_packaged_release_unpack_is_hash_and_file_set_bound(tmp_path):
    build, release, _runtime, _source = release_fixture(tmp_path)
    archive = build.zip_tree(release, tmp_path / "candidate.zip")
    restored = tmp_path / "restored"
    build.unpack_release(Path(archive["path"]), archive["sha256"], restored, "operational-overlay-manifest.json")
    assert build.files(restored) == build.files(release)
    with pytest.raises(ValueError, match="candidate_parent_archive_hash_mismatch"):
        build.unpack_release(Path(archive["path"]), "0" * 64, tmp_path / "rejected", "operational-overlay-manifest.json")


def test_database_identity_matches_runtime_sqlite_master_contract_not_file_bytes(tmp_path):
    build = builder()
    database = tmp_path / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fixture (id INTEGER PRIMARY KEY,\n  name TEXT)")
    expected = hashlib.sha256(b"table\0fixture\0fixture\0CREATE TABLE fixture (id INTEGER PRIMARY KEY, name TEXT)\n").hexdigest()
    assert build.database_schema_sha256(database) == expected
    assert build.database_schema_sha256(database) != build.sha(database)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO fixture(name) VALUES ('fixture')")
    assert build.database_schema_sha256(database) == expected
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE INDEX fixture_name ON fixture(name)")
    assert build.database_schema_sha256(database) != expected


def test_build_requires_its_own_hash_bound_package_before_invoking_npm(tmp_path, monkeypatch):
    build = builder()
    (tmp_path / "package.json").write_text('{"name":"must-not-be-used"}')
    work = tmp_path / "isolated"
    work.mkdir()
    receipt = work / "source-receipt.json"
    build.json_file(receipt, {"work_root": str(work), "files": []})
    calls = []
    monkeypatch.setattr(build, "run_command", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="candidate_required_build_input_missing:web/package.json"):
        build.build(receipt)
    assert calls == []


def test_all_build_inputs_are_explicit_and_hash_verified(tmp_path):
    build = builder()
    rows = []
    for name in build.REQUIRED_BUILD_INPUTS:
        path = tmp_path / "web" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture:" + name)
        rows.append({"path": "web/" + name, "bytes": path.stat().st_size, "sha256": build.sha(path)})
    build.validate_build_inputs(tmp_path, {"files": rows})
    (tmp_path / "web/package.json").write_text("changed override")
    with pytest.raises(ValueError, match="candidate_required_build_input_changed:web/package.json"):
        build.validate_build_inputs(tmp_path, {"files": rows})
