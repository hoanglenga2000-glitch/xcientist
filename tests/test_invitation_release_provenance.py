"""P0-1: release provenance must name the real source commit and dirty state.

The invitation builder can now stage overlaid files from a committed ref
(``--source-git-ref``) instead of the working tree; the legacy worktree path
keeps working but records HEAD and an honest dirty flag.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_invitation_release.py"
    spec = importlib.util.spec_from_file_location("invitation_provenance_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *argv: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *argv],
        capture_output=True, check=True)
    return done.stdout.decode().strip()


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src/evomind_runtime").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    (root / "src/evomind_runtime/runtime.py").write_bytes(b"VERSION = 'committed'\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "--no-verify", "-m", "fixture")
    return root


def test_git_ref_export_uses_committed_bytes_not_dirty_worktree(repo, tmp_path):
    build = builder()
    (repo / "src/evomind_runtime/runtime.py").write_bytes(b"VERSION = 'uncommitted edit'\n")
    out = tmp_path / "exported"
    provenance = build.export_git_source(repo, "HEAD", ["src/evomind_runtime/runtime.py"], out)
    assert (out / "src/evomind_runtime/runtime.py").read_bytes() == b"VERSION = 'committed'\n"
    assert provenance["source_mode"] == "git_ref"
    assert provenance["source_commit"] == git(repo, "rev-parse", "HEAD")
    assert provenance["source_commit_dirty"] is False


def test_git_ref_export_fails_closed_for_missing_paths_and_bad_refs(repo, tmp_path):
    build = builder()
    with pytest.raises(ValueError, match="source_git_ref_missing_path:src/evomind_runtime/absent.py"):
        build.export_git_source(repo, "HEAD", ["src/evomind_runtime/absent.py"], tmp_path / "a")
    with pytest.raises(ValueError, match="source_git_ref_unresolved"):
        build.export_git_source(repo, "no-such-branch", ["src/evomind_runtime/runtime.py"], tmp_path / "b")
    with pytest.raises(ValueError, match="source_git_ref_invalid"):
        build.export_git_source(repo, "--output=x", ["src/evomind_runtime/runtime.py"], tmp_path / "c")
    with pytest.raises(ValueError, match="candidate_archive_path_rejected"):
        build.export_git_source(repo, "HEAD", ["../escape.py"], tmp_path / "d")


def test_worktree_provenance_reports_head_and_honest_dirty_flag(repo):
    build = builder()
    paths = ["src/evomind_runtime/runtime.py", "src/evomind_runtime/new.py"]
    clean = build.worktree_source_provenance(repo, paths)
    assert clean == {"source_mode": "worktree", "source_commit": git(repo, "rev-parse", "HEAD"),
                     "source_commit_dirty": False}
    (repo / "src/evomind_runtime/new.py").write_text("untracked = True\n")
    assert build.worktree_source_provenance(repo, paths)["source_commit_dirty"] is True


def test_worktree_provenance_without_git_is_dirty_and_unrecorded(tmp_path):
    result = builder().worktree_source_provenance(tmp_path, ["x.py"])
    assert result == {"source_mode": "worktree", "source_commit": None, "source_commit_dirty": True}


def _release(tmp_path, build, provenance):
    release, runtime = tmp_path / "release", tmp_path / "runtime"
    for directory in (release / "node_modules/.prisma/client", release / "support/python-runtime/evomind_runtime",
                      runtime / "evomind_runtime"):
        directory.mkdir(parents=True)
    (release / "server.js").write_text("fixture-server")
    schema = release / "node_modules/.prisma/client/schema.prisma"
    schema.write_text("fixture-schema")
    for directory in (release / "support/python-runtime/evomind_runtime", runtime / "evomind_runtime"):
        (directory / "fixture.py").write_text("fixture = True\n")
    rows = [{"path": "web/fixture", "bytes": 1, "sha256": "1" * 64}]
    source = {"schema": "evomind.invitation_source.v1", "build_id": "fixture-new",
              "source_tree_sha256": build.tree_sha(rows), "production_schema_sha256": build.sha(schema),
              "database_schema_sha256": "d" * 64, "files": rows, "changes": [], "work_root": str(tmp_path),
              **provenance}
    identity = {"schema": "evomind.runtime_build.v1", "commit_hash": build.BASELINE_COMMIT, "backend_version": "0.3.0",
                "frontend_version": "0.3.0", "database_schema_version": "fixture", "database_schema_sha256": "0" * 64}
    operational = {"schema": "evomind.web_operational_overlay.v1", "base_commit": build.BASELINE_COMMIT,
                   "entrypoint": "server.js", "source_identity_sha256": "0" * 64}
    build.publish_release_manifests(release, runtime, source, identity, operational)
    return release, runtime, source


def test_published_identity_carries_source_commit_and_dirty_flag(tmp_path):
    build = builder()
    commit = "a" * 40
    release, runtime, source = _release(tmp_path, build, {"source_mode": "git_ref", "source_commit": commit,
                                                          "source_commit_dirty": False})
    identity = json.loads((release / "runtime-build-manifest.json").read_text())
    operational = json.loads((release / "operational-overlay-manifest.json").read_text())
    # commit_hash keeps binding the deployed baseline; the overlay source is explicit.
    assert identity["commit_hash"] == build.BASELINE_COMMIT
    assert identity["source_dirty"] is True
    assert identity["source_commit"] == commit and identity["source_commit_dirty"] is False
    assert operational["source_commit"] == commit
    assert build.validate_release_identity(release, runtime, source)["status"] == "passed"


@pytest.mark.parametrize("field,value", [("source_commit", "b" * 40), ("source_commit_dirty", True)])
def test_tampered_source_provenance_is_rejected(tmp_path, field, value):
    build = builder()
    release, runtime, source = _release(tmp_path, build, {"source_mode": "git_ref", "source_commit": "a" * 40,
                                                          "source_commit_dirty": False})
    path = release / "runtime-build-manifest.json"
    identity = json.loads(path.read_text())
    identity[field] = value
    build.json_file(path, identity)
    with pytest.raises(ValueError, match="candidate_"):
        build.validate_release_identity(release, runtime, source)


def test_legacy_source_without_provenance_still_validates(tmp_path):
    build = builder()
    release, runtime, source = _release(tmp_path, build, {})
    identity = json.loads((release / "runtime-build-manifest.json").read_text())
    assert "source_commit" not in identity
    assert build.validate_release_identity(release, runtime, source)["status"] == "passed"
