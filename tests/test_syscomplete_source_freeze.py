from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts/syscomplete_source_freeze.py"
    spec = importlib.util.spec_from_file_location("syscomplete_freeze_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["", "../escape", "/absolute", "C:/drive", "web\\path", "web//path", "web/./path"])
def test_source_paths_reject_aliases(name):
    with pytest.raises(ValueError, match="unsafe_source_path"):
        load().safe_relative(name)


def test_exact_delta_preserves_unshipped_preexisting_change():
    module = load()
    base = b"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\n"
    before = base + b"UNSHIPPED-APPROVAL-CHANGE\n"
    after = before.replace(b"c\n", b"task-c\n")
    result, hunks = module.exact_delta(base, before, after)
    assert result == base.replace(b"c\n", b"task-c\n")
    assert b"UNSHIPPED" not in result
    assert hunks == 1


def test_exact_delta_no_fuzzy_context():
    module = load()
    with pytest.raises(ValueError, match="delta_context_not_unique"):
        module.exact_delta(b"other\na\nb\n", b"header\na\nb\n", b"header\na\nchanged\n")


def test_exact_delta_ambiguous_context_rejected():
    module = load()
    block = b"a\nb\nc\nd\ne\nf\ng\n"
    before = block + b"unshipped\n"
    after = before.replace(b"d\n", b"edited\n")
    with pytest.raises(ValueError, match="delta_context_not_unique"):
        module.exact_delta(block * 2, before, after)


def test_exact_delta_unchanged_snapshot_returns_current():
    module = load()
    assert module.exact_delta(b"same\n", b"same\n", b"edit\n") == (b"edit\n", 0)


def test_exact_delta_with_no_new_change_returns_base():
    module = load()
    assert module.exact_delta(b"live\n", b"dirty\n", b"dirty\n") == (b"live\n", 0)


def test_windows_reparse_file_is_rejected(tmp_path, monkeypatch):
    module = load()
    file = tmp_path / "code.py"
    file.write_bytes(b"not read")
    original = Path.lstat

    def reparse(path):
        if path == file:
            return SimpleNamespace(st_mode=0o100644, st_file_attributes=0x400)
        return original(path)

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(ValueError, match="source_path_alias_rejected"):
        module.checked_path(tmp_path, "code.py")


def fixture(module, tmp_path, monkeypatch):
    parent, workspace, backup, inputs = (tmp_path / name for name in ("parent", "workspace", "backup", "inputs"))
    initial = {"web/src/keep.ts": b"export const old = 1;\n", "runtime/evomind_runtime/runtime.py": b"VERSION='base'\n",
               "web/package.json": b'{"name":"pinned"}', "web/next.config.mjs": b"export default {};\n"}
    for name, value in initial.items():
        if name == "web/next.config.mjs":
            module.write_new(inputs / name[4:], value)
        else:
            path = parent / ("release/" + name[4:] if name.startswith("web/") else name)
            module.write_new(path, value)
    for name in module.TEMPLATES:
        module.write_new(parent / "release" / name, b"{}")
    module.write_new(workspace / "web/research-agent-workstation/src/keep.ts", b"export const old = 2;\n")
    module.write_new(workspace / "web/research-agent-workstation/src/unrelated.ts", b"MUST NOT SHIP")
    module.write_new(backup / "web/research-agent-workstation/src/keep.ts", initial["web/src/keep.ts"])
    source = {"files": [module.row_for(name, value) for name, value in initial.items()]}
    monkeypatch.setattr(module, "verify_parent", lambda _parent: source)
    monkeypatch.setattr(module, "BACKUP_UI", ("src/keep.ts",))
    monkeypatch.setattr(module, "REVIEWED_DIRECT", {})
    return parent, workspace, backup, inputs, source


def test_preparation_recovers_exact_inputs_and_excludes_dirty_tree(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, source = fixture(module, tmp_path, monkeypatch)
    plan, payloads = module.prepare(parent, workspace, backup, [inputs])
    assert plan["file_count"] == len(source["files"])
    assert plan["build_ready"] is False
    assert plan["runtime_payload_changed"] is False
    assert plan["dirty_tree_wholesale_copy"] is False
    assert plan["restored_build_inputs"][0]["path"] == "web/next.config.mjs"
    assert plan["web_changes"][0]["operation"] == "snapshot_bound_edit"
    assert payloads["runtime/evomind_runtime/runtime.py"] == b"VERSION='base'\n"
    assert all("unrelated" not in name for name in payloads)


def test_missing_source_not_taken_from_dirty_workspace(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, source = fixture(module, tmp_path, monkeypatch)
    source["files"].append(module.row_for("web/src/unrelated.ts", b"MUST NOT SHIP"))
    with pytest.raises(ValueError, match="source_bytes_not_recovered:web/src/unrelated.ts"):
        module.prepare(parent, workspace, backup, [inputs, workspace / "web/research-agent-workstation"])


def test_build_input_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, _source = fixture(module, tmp_path, monkeypatch)
    (inputs / "next.config.mjs").write_bytes(b"export default 'wrong';")
    with pytest.raises(ValueError, match="source_bytes_not_recovered:web/next.config.mjs"):
        module.prepare(parent, workspace, backup, [inputs])


def test_unreviewed_starting_snapshot_cannot_claim_changes(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, _source = fixture(module, tmp_path, monkeypatch)
    (backup / "web/research-agent-workstation/src/keep.ts").write_bytes(b"other starting value")
    with pytest.raises(ValueError, match="unreviewed_starting_snapshot_drift"):
        module.prepare(parent, workspace, backup, [inputs])


def test_new_reviewed_file_hash_and_absence_are_required(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, source = fixture(module, tmp_path, monkeypatch)
    relative = "src/app/api/assistant/runs/[runId]/reports/route.ts"
    expected = b"new route"
    monkeypatch.setattr(module, "REVIEWED_DIRECT", {relative: module.digest(expected)})
    module.write_new(workspace / "web/research-agent-workstation" / relative, expected)
    plan, payloads = module.prepare(parent, workspace, backup, [inputs])
    assert plan["web_changes"][-1]["operation"] == "reviewed_new_file"
    assert payloads["web/" + relative] == expected
    module.write_new(parent / "release" / relative, b"already exists")
    source["files"].append(module.row_for("web/" + relative, b"already exists"))
    with pytest.raises(ValueError, match="reviewed_new_route_already_exists"):
        module.prepare(parent, workspace, backup, [inputs])


def test_freeze_revalidates_plan_and_does_not_overwrite(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, _source = fixture(module, tmp_path, monkeypatch)
    plan, _payloads = module.prepare(parent, workspace, backup, [inputs])
    path, destination = tmp_path / "plan.json", tmp_path / "frozen"
    module.write_new(path, module.canonical_json(plan))
    result = module.freeze(path, module.sha(path), destination)
    assert result["build_ready"] is False
    assert not (destination / "source-receipt.json").exists()
    frozen = json.loads((destination / "syscomplete-source-manifest.json").read_bytes())
    assert frozen["frozen"] is True
    assert (destination / "web/src/keep.ts").read_bytes() == b"export const old = 2;\n"
    assert (destination / "runtime/evomind_runtime/runtime.py").read_bytes() == b"VERSION='base'\n"
    with pytest.raises(ValueError, match="freeze_destination_exists_or_aliased"):
        module.freeze(path, module.sha(path), destination)


def test_freeze_stops_on_mutation_without_creating_candidate(tmp_path, monkeypatch):
    module = load()
    parent, workspace, backup, inputs, _source = fixture(module, tmp_path, monkeypatch)
    plan, _payloads = module.prepare(parent, workspace, backup, [inputs])
    path, destination = tmp_path / "plan.json", tmp_path / "candidate"
    module.write_new(path, module.canonical_json(plan))
    (workspace / "web/research-agent-workstation/src/keep.ts").write_bytes(b"new unrelated edit")
    with pytest.raises(ValueError, match="freeze_inputs_changed_since_plan"):
        module.freeze(path, module.sha(path), destination)
    assert not destination.exists()


def test_freeze_plan_tamper_rejected_before_processing(tmp_path):
    module = load()
    path = tmp_path / "plan.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="freeze_plan_hash_mismatch"):
        module.freeze(path, "0" * 64, tmp_path / "candidate")


def test_old_or_unpinned_parent_is_rejected_without_extracting(tmp_path):
    module = load()
    module.write_new(tmp_path / "web.zip", b"old 20260905 baseline is not current")
    with pytest.raises(ValueError, match="pinned_parent_hash_mismatch:web.zip"):
        module.verify_parent(tmp_path)
