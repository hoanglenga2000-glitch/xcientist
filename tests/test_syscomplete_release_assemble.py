from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts/syscomplete_release_assemble.py"
    spec = importlib.util.spec_from_file_location("syscomplete_assemble_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_new(module, path, value):
    module.BASE.write_new(path, module.BASE.canonical_json(value))
    return module.BASE.sha(path)


def source_fixture(module, tmp_path):
    root = tmp_path / "source"
    payloads = {"web/src/task.ts": b"task;\n", "web/package.json": b'{"name":"fixture"}',
                "runtime/evomind_runtime/runtime.py": b"old\n",
                "web/support/python-runtime/evomind_runtime/runtime.py": b"old\n"}
    for name in module.BASE.BUILD_INPUTS:
        payloads.setdefault(name, b"fixture input")
    rows = [module.BASE.row_for(name, data) for name, data in sorted(payloads.items())]
    templates = [module.BASE.row_for("web/" + name, b"{}") for name in module.BASE.TEMPLATES]
    for row in templates:
        payloads[row["path"]] = b"{}"
    for name, data in payloads.items():
        module.BASE.write_new(root / name, data)
    manifest = {"schema": "evomind.syscomplete_source_freeze.v1", "frozen": True,
                "base_build_id": module.BASE.BASE_BUILD, "base_source_tree_sha256": module.BASE.BASE_TREE,
                "base_web_sha256": module.BASE.BASE_WEB, "base_runtime_sha256": module.BASE.BASE_RUNTIME,
                "dirty_tree_wholesale_copy": False, "runtime_payload_changed": False,
                "files": rows, "templates": templates, "source_tree_sha256": module.BASE.tree_sha(rows),
                "file_count": len(rows), "web_changes": []}
    expected = json_new(module, root / "syscomplete-source-manifest.json", manifest)
    return root, expected, manifest, payloads


def runtime_fixture(module, tmp_path, payloads):
    root = tmp_path / "runtime-source"
    rows = []
    for name in sorted(module.REQUIRED_RUNTIME):
        relative = "evomind_runtime/" + name
        data = b"# bounded fixture: " + name.encode() + b"\n"
        if name in {"aibuild_engine.py", "aibuild_model_recovery.py"}:
            data += b"from .managed_scheduler import AgentTask\n"
        module.BASE.write_new(root / relative, data)
        before = payloads.get("runtime/" + relative)
        rows.append({**module.BASE.row_for(relative, data), "before_sha256": module.BASE.digest(before) if before else None})
    manifest = {"schema": "evomind.syscomplete_runtime_delta.v1", "frozen": True, "model": "gpt-5.5",
                "base_source_tree_sha256": module.BASE.BASE_TREE, "files": rows}
    path = root / "runtime-delta.json"
    expected = json_new(module, path, manifest)
    return root, path, expected, manifest


def dependency_fixture(module, tmp_path):
    root = tmp_path / "wheels"
    data = b"synthetic wheel metadata; never installed"
    wheels = []
    for name in sorted(module.REPORT_REQUIREMENTS):
        wheel = name.replace("-", "_") + "-1-py3-none-any.whl"
        module.BASE.write_new(root / wheel, data)
        wheels.append(module.BASE.row_for(wheel, data))
    manifest = {
        "schema": "evomind.syscomplete_report_dependencies.v1", "frozen": True,
        "python_tag": "cp312", "platform": "win_amd64", "activation": "offline_install_then_verify_imports",
        "distributions": [{"name": name, "version": "1", "source": "wheelhouse"} for name in sorted(module.REPORT_REQUIREMENTS)],
        "wheels": wheels,
    }
    path = root / "descriptor.json"
    expected = json_new(module, path, manifest)
    return root, path, expected, manifest


def test_source_exact_declared_set_and_hashes(tmp_path):
    module = load()
    root, expected, manifest, payloads = source_fixture(module, tmp_path)
    actual, data = module.read_source(root, expected)
    assert actual == manifest
    assert data == payloads
    module.BASE.write_new(root / "web/src/unrelated.ts", b"dirty")
    with pytest.raises(ValueError, match="source_freeze_extra_or_missing_files"):
        module.read_source(root, expected)


def test_source_declared_file_tamper_rejected(tmp_path):
    module = load()
    root, expected, _manifest, _payloads = source_fixture(module, tmp_path)
    (root / "web/src/task.ts").write_bytes(b"drift")
    with pytest.raises(ValueError, match="declared_file_changed"):
        module.read_source(root, expected)


@pytest.mark.parametrize("name", ["aibuild_engine.py", "aibuild_model_recovery.py", "managed_scheduler.py", "report_jobs.py"])
def test_missing_required_runtime_part_rejected(tmp_path, name):
    module = load()
    _source, _sha, _manifest, payloads = source_fixture(module, tmp_path)
    root, path, _expected, manifest = runtime_fixture(module, tmp_path, payloads)
    manifest["files"] = [row for row in manifest["files"] if row["path"] != "evomind_runtime/" + name]
    path.write_bytes(module.BASE.canonical_json(manifest))
    with pytest.raises(ValueError, match="runtime_allowlist_incomplete_or_expanded"):
        module.runtime_delta(path, module.BASE.sha(path), root, payloads)


def test_runtime_overlay_ignores_unlisted_dirty_files_and_mirrors_exactly(tmp_path):
    module = load()
    _source, _sha, _manifest, payloads = source_fixture(module, tmp_path)
    root, path, expected, manifest = runtime_fixture(module, tmp_path, payloads)
    module.BASE.write_new(root / "evomind_runtime/unrelated.py", b"not shipped")
    _result, changes = module.runtime_delta(path, expected, root, payloads)
    assert len(changes) == len(manifest["files"]) * 2
    assert all("unrelated" not in name for name in payloads)
    for row in manifest["files"]:
        assert payloads["runtime/" + row["path"]] == payloads["web/support/python-runtime/" + row["path"]]


def test_wrong_parent_sha_does_not_partially_mutate_payloads(tmp_path):
    module = load()
    _source, _sha, _manifest, payloads = source_fixture(module, tmp_path)
    before = dict(payloads)
    root, path, _expected, manifest = runtime_fixture(module, tmp_path, payloads)
    manifest["files"][-1]["before_sha256"] = "a" * 64
    path.write_bytes(module.BASE.canonical_json(manifest))
    with pytest.raises(ValueError, match="runtime_parent_file_mismatch"):
        module.runtime_delta(path, module.BASE.sha(path), root, payloads)
    assert payloads == before


def test_runtime_unlisted_expansion_and_mutated_source_rejected(tmp_path):
    module = load()
    _source, _sha, _manifest, payloads = source_fixture(module, tmp_path)
    root, path, expected, manifest = runtime_fixture(module, tmp_path, payloads)
    (root / manifest["files"][0]["path"]).write_bytes(b"concurrent mutation")
    with pytest.raises(ValueError, match="declared_file_changed"):
        module.runtime_delta(path, expected, root, payloads)
    manifest["files"].append({"path": "evomind_runtime/not_approved.py", "bytes": 0, "sha256": "a" * 64, "before_sha256": None})
    path.write_bytes(module.BASE.canonical_json(manifest))
    with pytest.raises(ValueError, match="runtime_allowlist_incomplete_or_expanded"):
        module.runtime_delta(path, module.BASE.sha(path), root, payloads)


def test_scheduler_old_import_rejected(tmp_path):
    module = load()
    _source, _sha, _manifest, payloads = source_fixture(module, tmp_path)
    root, path, _expected, manifest = runtime_fixture(module, tmp_path, payloads)
    row = next(row for row in manifest["files"] if row["path"].endswith("aibuild_engine.py"))
    data = b"from research_os.agent.multi_agent import AgentTask\n"
    (root / row["path"]).write_bytes(data)
    row.update(module.BASE.row_for(row["path"], data))
    path.write_bytes(module.BASE.canonical_json(manifest))
    with pytest.raises(ValueError, match="managed_scheduler_import_contract_rejected"):
        module.runtime_delta(path, module.BASE.sha(path), root, payloads)


def test_report_descriptor_delivers_wheels_but_does_not_claim_activation(tmp_path):
    module = load()
    root, path, expected, descriptor = dependency_fixture(module, tmp_path)
    payloads = {}
    result, changes = module.report_dependencies(path, expected, root, payloads)
    assert result["activated"] is False
    assert result["server_import_acceptance"] == "pending"
    assert result["wheel_count"] == len(module.REPORT_REQUIREMENTS)
    assert len(changes) == len(module.REPORT_REQUIREMENTS) + 1
    assert len(payloads) == len(module.REPORT_REQUIREMENTS) + 1


@pytest.mark.parametrize("mutation,code", [
    ("missing_requirement", "report_required_distributions_missing"),
    ("unknown_metadata", "report_dependency_contract_rejected"),
    ("unbound_inherited", "report_inherited_environment_unbound"),
    ("path", "report_wheel_path_rejected"),
    ("extra_wheel_field", "report_wheel_row_rejected"),
])
def test_report_dependency_missing_or_unscoped_evidence_rejected(tmp_path, mutation, code):
    module = load()
    root, path, _expected, descriptor = dependency_fixture(module, tmp_path)
    if mutation == "missing_requirement":
        descriptor["distributions"] = descriptor["distributions"][:1]
    elif mutation == "unknown_metadata":
        descriptor["unreviewed_field"] = "not copied"
    elif mutation == "unbound_inherited":
        descriptor["distributions"][0]["source"] = "pinned_base_environment"
        inherited_name = descriptor["distributions"][0]["name"].replace("-", "_")
        descriptor["wheels"] = [row for row in descriptor["wheels"] if not row["path"].startswith(inherited_name + "-")]
    elif mutation == "path":
        descriptor["wheels"][0]["path"] = "nested/" + descriptor["wheels"][0]["path"]
    else:
        descriptor["wheels"][0]["unreviewed_field"] = "not copied"
    path.write_bytes(module.BASE.canonical_json(descriptor))
    with pytest.raises(ValueError, match=code):
        module.report_dependencies(path, module.BASE.sha(path), root, {})


def test_assembly_standard_receipt_bound_to_explicit_frozen_manifests(tmp_path):
    module = load()
    source, expected, _frozen, payloads = source_fixture(module, tmp_path)
    runtime_root, runtime_path, runtime_sha, _manifest = runtime_fixture(module, tmp_path, payloads)
    wheels, dependency_path, dependency_sha, _descriptor = dependency_fixture(module, tmp_path)
    args = SimpleNamespace(source=source, source_manifest_sha256=expected, destination=tmp_path / "assembled",
                           runtime_root=runtime_root, runtime_manifest=runtime_path, runtime_manifest_sha256=runtime_sha,
                           report_dependencies=dependency_path, report_dependencies_sha256=dependency_sha, report_wheel_root=wheels)
    receipt = module.assemble(args)
    result = json.loads(receipt.read_bytes())
    assert result["schema"] == "evomind.invitation_source.v1"
    assert result["frozen"] is True
    assert "artifact_only" not in result
    assert result["derivation"]["runtime_delta_manifest_sha256"] == runtime_sha
    assert result["source_tree_sha256"] == module.BASE.tree_sha(result["files"])
    assert result["release_verdict"] == "HOLD"
    assert result["report_dependency_delivery"]["activated"] is False
    module.validate_source_unchanged(args.destination, result)
    assert module.LEGACY.files(args.destination / "runtime/evomind_runtime") == module.LEGACY.files(args.destination / "web/support/python-runtime/evomind_runtime")


def test_frontend_preflight_receipt_cannot_build_as_final_release(tmp_path):
    module = load()
    source, expected, frozen, payloads = source_fixture(module, tmp_path)
    work = tmp_path / "frontend"
    receipt = module.make_source(work, frozen, payloads, [], runtime_sha=None, dependencies=None, source_manifest_sha=expected)
    module.BASE.write_new(work / "frontend-source-receipt.json", module.BASE.canonical_json(receipt))
    assert receipt["frontend_preflight_only"] is True
    with pytest.raises(ValueError, match="derived_artifact_requires_fresh_source_stage"):
        module.LEGACY.build(work / "frontend-source-receipt.json")


def test_destination_never_overwrites_source_or_existing_files(tmp_path):
    module = load()
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="assembly_destination_exists_or_aliased"):
        module.create_work(source / "nested", {}, source)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(ValueError, match="assembly_destination_exists_or_aliased"):
        module.create_work(destination, {}, source)


def test_web_amendment_is_hash_bound_and_does_not_modify_parent(tmp_path):
    module = load()
    source, source_sha, frozen, _payloads = source_fixture(module, tmp_path)
    relative = "web/src/task.ts"
    module.WEB_CORRECTIONS = frozenset({relative})
    replacement = b"reviewed fix;\n"
    patch_root = tmp_path / "patch"
    module.BASE.write_new(patch_root / "src/task.ts", replacement)
    delta = {"schema": "evomind.syscomplete_web_delta.v1", "frozen": True,
             "parent_manifest_sha256": source_sha,
             "files": [{**module.BASE.row_for(relative, replacement), "before_sha256": module.BASE.digest(b"task;\n")} ]}
    delta_path = tmp_path / "delta.json"
    delta_sha = json_new(module, delta_path, delta)
    destination = tmp_path / "amended"
    result = module.amend_web(source, source_sha, delta_path, delta_sha, patch_root, destination)
    amended, payloads = module.read_source(destination, result["manifest_sha256"])
    assert payloads[relative] == replacement
    assert (source / relative).read_bytes() == b"task;\n"
    assert amended["parent_frozen_manifest_sha256"] == source_sha
    assert amended["web_changes"][-1]["delta_manifest_sha256"] == delta_sha


def test_report_missing_wheel_cannot_be_claimed_by_distribution_list(tmp_path):
    module = load()
    root, path, _expected, descriptor = dependency_fixture(module, tmp_path)
    descriptor["wheels"] = descriptor["wheels"][:-1]
    path.write_bytes(module.BASE.canonical_json(descriptor))
    with pytest.raises(ValueError, match="report_wheel_distribution_binding_mismatch"):
        module.report_dependencies(path, module.BASE.sha(path), root, {})


def inherited_fixture(module, tmp_path):
    root, descriptor_path, _sha, descriptor = dependency_fixture(module, tmp_path)
    for row in descriptor["distributions"]:
        if row["name"] == "pymupdf":
            row["source"] = "pinned_base_environment"
    descriptor["wheels"] = [row for row in descriptor["wheels"] if not row["path"].startswith("pymupdf-")]
    venv = "C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv"
    receipt = {
        "schema": "evomind.report_dependency_inheritance.v1", "base_build_id": module.BASE.BASE_BUILD,
        "base_source_tree_sha256": module.BASE.BASE_TREE,
        "base_source_manifest_sha256": module.BASE.BASE_SOURCE_MANIFEST,
        "production_changed": False, "activation_claim": False,
        "python_version": "3.12.10", "python_abi": None, "platform": "win-amd64", "venv_root": venv,
        "packages": [{"name": "PyMuPDF", "version": "1", "import_passed": True, "module": "fitz",
                      "module_path": venv + "/Lib/site-packages/fitz/__init__.py",
                      "files": [module.BASE.row_for("Lib/site-packages/fitz/__init__.py", b"fixture"),
                                module.BASE.row_for("Scripts/pymupdf.exe", b"fixture console")]}],
    }
    receipt_path = tmp_path / "inherited.json"
    receipt_sha = json_new(module, receipt_path, receipt)
    descriptor["base_environment_binding"] = {"base_build_id": module.BASE.BASE_BUILD, "receipt_sha256": receipt_sha}
    descriptor_path.write_bytes(module.BASE.canonical_json(descriptor))
    return root, descriptor_path, module.BASE.sha(descriptor_path), receipt_path, receipt


def test_inheritance_receipt_preserved_and_required_for_inherited_packages(tmp_path):
    module = load()
    root, descriptor, expected, receipt_path, _receipt = inherited_fixture(module, tmp_path)
    with pytest.raises(ValueError, match="report_inheritance_receipt_required"):
        module.report_dependencies(descriptor, expected, root, {})
    payloads = {}
    result, _changes = module.report_dependencies(descriptor, expected, root, payloads, receipt_path)
    assert result["activated"] is False
    assert result["inherited_environment_receipt"]["recorded_file_count"] == 2
    assert payloads["web/support/report-wheelhouse/report-inheritance.json"] == receipt_path.read_bytes()
    assert payloads["web/support/report-wheelhouse/dependency-delivery.json"] == descriptor.read_bytes()


@pytest.mark.parametrize("mutation,code", [
    ("version", "report_inherited_version_mismatch"),
    ("unbound_import", "report_inheritance_import_not_hash_bound"),
    ("outside_venv", "report_inheritance_venv_rejected"),
    ("wrong_abi", "report_inheritance_identity_rejected"),
    ("foreign_console", "report_inheritance_file_scope_rejected"),
])
def test_inherited_receipt_contract_drift_rejected(tmp_path, mutation, code):
    module = load()
    _root, _descriptor, _expected, receipt_path, receipt = inherited_fixture(module, tmp_path)
    if mutation == "version":
        receipt["packages"][0]["version"] = "other"
    elif mutation == "unbound_import":
        receipt["packages"][0]["module_path"] += ".unlisted"
    elif mutation == "outside_venv":
        receipt["venv_root"] = "C:/unbound"
    elif mutation == "wrong_abi":
        receipt["python_version"] = "3.11.8"
    else:
        receipt["packages"][0]["files"][1]["path"] = "Scripts/foreign.exe"
    receipt_path.write_bytes(module.BASE.canonical_json(receipt))
    with pytest.raises(ValueError, match=code):
        module.inherited_report_environment(receipt_path, module.BASE.sha(receipt_path),
                                             [{"name": "PyMuPDF", "version": "1"}])


def test_acceptance_pack_contains_only_bound_materials_and_empty_fixture(tmp_path, monkeypatch):
    module = load()
    work = tmp_path / "built"
    work.mkdir()
    source = {"source_tree_sha256": "a" * 64, "build_id": "fixture-build"}
    json_new(module, work / "source-receipt.json", source)
    json_new(module, work / "assembly-receipt.json", {"synthetic": True})
    for name in ("web", "runtime"):
        (work / (name + ".zip")).write_bytes(name.encode())
    built = {"status": "built", "source_receipt_sha256": module.BASE.sha(work / "source-receipt.json"),
             "source_tree_sha256": source["source_tree_sha256"],
             "web": {"sha256": module.BASE.sha(work / "web.zip")},
             "runtime": {"sha256": module.BASE.sha(work / "runtime.zip")}}
    json_new(module, work / "build-result.json", built)
    json_new(module, work / "input-receipts/test.json", {"synthetic": True})
    module.save_build_recipes(work)
    (work / "fixture").mkdir()
    with sqlite3.connect(work / "fixture/build.sqlite") as connection:
        connection.execute("CREATE TABLE fixture (value TEXT)")
    monkeypatch.setattr(module.LEGACY, "validate_release_identity", lambda *_args: {"status": "fixture_only"})
    result = module.package_acceptance(work, built)
    assert result["status"] == "three_packages_built_not_deployed"
    assert result["production_deployed"] is False
    extracted = tmp_path / "independent"
    manifest = module.LEGACY.unpack_release(work / "acceptance.zip", result["acceptance"]["sha256"],
                                           extracted, "acceptance-manifest.json")
    assert manifest["web_sha256"] == built["web"]["sha256"]
    assert module.BASE.sha(extracted / "fixture-schema.sqlite") == result["fixture_sha256"]
    assert (extracted / "verify_invitation_server_candidate.py").is_file()
    assert not (extracted / "Start-Node.ps1").exists()
