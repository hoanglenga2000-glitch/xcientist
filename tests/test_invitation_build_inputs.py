import json
from types import SimpleNamespace

import pytest

from scripts import build_invitation_release as builder


def test_changed_build_input_is_rebound_without_mutating_the_parent_receipt(tmp_path):
    path = tmp_path / "web/package.json"
    path.parent.mkdir()
    path.write_text('{"scripts":{"test":"old"}}', encoding="utf-8")
    original = {"path": "web/package.json", "sha256": builder.sha(path), "bytes": path.stat().st_size}
    old = dict(original)
    path.write_text('{"scripts":{"test":"new"}}', encoding="utf-8")
    change = {"path": original["path"], "before_sha256": original["sha256"], "sha256": builder.sha(path), "bytes": path.stat().st_size}
    rebound = builder.refresh_build_inputs(tmp_path, [original], [change])
    assert original == old
    assert rebound[0]["sha256"] == builder.sha(path)
    assert rebound[0]["sha256"] != old["sha256"]
    with pytest.raises(ValueError, match="parent_mismatch"):
        builder.refresh_build_inputs(tmp_path, [original], [{**change, "before_sha256": "0" * 64}])


def test_unapproved_build_input_changes_are_rejected(tmp_path):
    path = tmp_path / "web/tsconfig.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    row = {"path": "web/tsconfig.json", "sha256": builder.sha(path), "bytes": path.stat().st_size}
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="build_input_changed"):
        builder.refresh_build_inputs(tmp_path, [row], [])


def test_stage_validates_original_package_then_freezes_the_authorized_overlay(tmp_path, monkeypatch):
    local = tmp_path / "local-web"
    local.mkdir()
    (local / "package.json").write_text('{"scripts":{"test":"new"}}', encoding="utf-8")
    (local / "tsconfig.json").write_text("{}", encoding="utf-8")
    schema = tmp_path / "schema.prisma"
    schema.write_text("// isolated fixture", encoding="utf-8")
    monkeypatch.setattr(builder, "WEB", local)
    monkeypatch.setattr(builder, "WEB_PATCHES", ("package.json",))
    monkeypatch.setattr(builder, "RUNTIME_PATCHES", ())
    def unpack(_archive, _sha, work, _build):
        for name in ("web/src", "web/public", "web/support/python-runtime/evomind_runtime", "runtime/evomind_runtime"):
            (work / name).mkdir(parents=True)
        package = work / "web/package.json"
        package.write_text('{"scripts":{"test":"old"}}', encoding="utf-8")
        inputs = [{"path": "package.json", "sha256": builder.sha(package), "bytes": package.stat().st_size},
                  {"path": "tsconfig.json", "sha256": builder.sha(local / "tsconfig.json"), "bytes": 2}]
        builder.json_file(work / "web/release-source-manifest.json", {"build_inputs": inputs})
        return {"files": [{"path": "web/package.json", "sha256": builder.sha(package), "bytes": package.stat().st_size}]}
    monkeypatch.setattr(builder, "unpack_baseline", unpack)
    receipt = builder.stage(SimpleNamespace(work_root=tmp_path / "stages", baseline="fixture.zip", baseline_sha256="a" * 64,
                                            expected_web_build="fixture", production_schema=schema, production_schema_sha256=builder.sha(schema)))
    result = json.loads(receipt.read_text(encoding="utf-8"))
    row = next(item for item in result["files"] if item["path"] == "web/package.json")
    assert row["sha256"] == builder.sha(local / "package.json")
    assert result["source_tree_sha256"] == builder.tree_sha(result["files"])
    assert result["production_deployed"] is False
    fixture = next(item for item in result["files"] if item["path"] == "web/test-fixtures/canonical_json_f64_v1.json")
    assert fixture["sha256"] == builder.sha(builder.ROOT / "tests/fixtures/canonical_json_f64_v1.json")


def test_build_environment_uses_candidate_owned_npm_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NPM_CONFIG_CACHE", "ambient-shared-cache")
    environment = builder.command_environment(tmp_path / "web")
    assert environment["NPM_CONFIG_CACHE"] == str(tmp_path / "fixture/npm-cache")
    assert environment["NPM_CONFIG_USERCONFIG"] == str(tmp_path / "fixture/npm-user.conf")
    assert environment["EVOMIND_TEST_FIXTURE_ROOT"] == str(tmp_path / "web/test-fixtures")
