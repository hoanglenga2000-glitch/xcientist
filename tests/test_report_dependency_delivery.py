import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


spec = importlib.util.spec_from_file_location('delivery_verifier', Path(__file__).parents[1]/'scripts/verify_report_dependency_delivery.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('path', ['', '../outside', '/absolute', 'C:/Windows/file', 'a\\b',
                                 'a/../b', 'a/./b', 'a//b', '//host/share'])
def test_manifest_paths_cannot_escape_or_alias(path):
    with pytest.raises(ValueError, match='relative_path_rejected'):
        module.safe_relative(path)


def test_hash_and_full_tree_checks_detect_changes(tmp_path):
    path = tmp_path/'fixture.txt'
    path.write_text('before', encoding='utf-8')
    original = module.tree_manifest(tmp_path)
    assert module.verify_rows(tmp_path, original) == 1
    path.write_text('after', encoding='utf-8')
    assert module.tree_manifest(tmp_path) != original
    with pytest.raises(ValueError, match='declared_file_integrity_failed'):
        module.verify_rows(tmp_path, original)


def test_duplicate_manifest_paths_are_rejected(tmp_path):
    path = tmp_path/'fixture.txt'
    path.write_text('fixture', encoding='utf-8')
    value = module.tree_manifest(tmp_path)
    value['files'] *= 2
    with pytest.raises(ValueError, match='duplicate_manifest_path'):
        module.verify_rows(tmp_path, value)


def make_wheel(tmp_path, *, tag='cp312-cp312-win_amd64', version='1.2', extra=None):
    path = tmp_path/'fixture-1.2-cp312-cp312-win_amd64.whl'
    with zipfile.ZipFile(path, 'w') as wheel:
        wheel.writestr('fixture/__init__.py', '')
        wheel.writestr('fixture-1.2.dist-info/METADATA', 'Name: fixture\nVersion: '+version+'\n')
        wheel.writestr('fixture-1.2.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: '+tag+'\n')
        if extra:
            wheel.writestr(extra, b'fixture')
    return path


def test_real_wheel_metadata_and_platform_tag_are_checked(tmp_path):
    path = make_wheel(tmp_path)
    module.validate_wheel(path, 'fixture', '1.2', {'cp312-cp312-win_amd64'})


@pytest.mark.parametrize('tag,version,error', [
    ('cp311-cp311-win_amd64','1.2','wheel_abi_incompatible'),
    ('cp312-cp312-win_amd64','9.9','wheel_distribution_mismatch'),
])
def test_wrong_abi_or_metadata_version_is_rejected(tmp_path, tag, version, error):
    path = make_wheel(tmp_path, tag=tag, version=version)
    with pytest.raises(ValueError, match=error):
        module.validate_wheel(path, 'fixture', '1.2', {'cp312-cp312-win_amd64'})


@pytest.mark.parametrize('extra', ['../outside.py', 'fixture/import_hook.pth'])
def test_wheel_cannot_install_traversal_or_startup_hook(tmp_path, extra):
    path = make_wheel(tmp_path, extra=extra)
    with pytest.raises(ValueError):
        module.validate_wheel(path, 'fixture', '1.2', {'cp312-cp312-win_amd64'})


def test_wheel_symlink_is_rejected(tmp_path):
    extra = zipfile.ZipInfo('fixture/link')
    extra.create_system = 3
    extra.external_attr = 0o120777 << 16
    path = make_wheel(tmp_path, extra=extra)
    with pytest.raises(ValueError, match='wheel_executable_path_rejected'):
        module.validate_wheel(path, 'fixture', '1.2', {'cp312-cp312-win_amd64'})


@pytest.mark.skipif(__import__('os').name!='nt', reason='Windows deployment destination contract')
@pytest.mark.parametrize('target', ['C:/EMQA', 'C:/EMQA/sys0907-report-deps-v1',
    'C:/EMQA/sys0907-gpt55-stability-v10', 'C:/ProgramData/EvoMind/runtime',
    'C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv'])
def test_target_allowlist_protects_existing_runtime_and_endurance(target):
    with pytest.raises(ValueError, match='target_scope_rejected'):
        module.validate_locations('C:/ProgramData/EvoMind/staging/invitation-aaaaaaaaaaaa-sys1', target)


def test_wheel_module_cannot_resolve_from_a_base_or_old_target(tmp_path, monkeypatch):
    elsewhere = tmp_path/'old-site'/'numpy.py'
    elsewhere.parent.mkdir()
    elsewhere.write_text('fixture', encoding='utf-8')
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: SimpleNamespace(__file__=str(elsewhere)))
    monkeypatch.setattr(module.importlib.metadata, 'distribution', lambda name: SimpleNamespace(version='2.5.2'))
    descriptor = {'distributions':[{'name':'numpy','version':'2.5.2','source':'wheelhouse'}]}
    with pytest.raises(ValueError):
        module.verify_imports(descriptor, tmp_path/'new-target')


def test_new_target_module_import_is_hash_bound(tmp_path, monkeypatch):
    target = tmp_path/'new-target'
    path = target/'site-packages'/'numpy.py'
    path.parent.mkdir(parents=True)
    path.write_text('fixture', encoding='utf-8')
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: SimpleNamespace(__file__=str(path)))
    monkeypatch.setattr(module.importlib.metadata, 'distribution', lambda name: SimpleNamespace(version='2.5.2'))
    descriptor = {'distributions':[{'name':'numpy','version':'2.5.2','source':'wheelhouse'}]}
    result = module.verify_imports(descriptor, target)
    assert result[0]['module_sha256'] == module.digest(path)
    assert result[0]['module_path'] == str(path.resolve())


def test_runtime_binding_follows_hash_bound_source_receipt(tmp_path):
    path = tmp_path/'runtime'/'evomind_runtime'/'report_document.py'
    path.parent.mkdir(parents=True)
    path.write_text('fixture', encoding='utf-8')
    row = {'path':'runtime/evomind_runtime/report_document.py', 'bytes':path.stat().st_size, 'sha256':module.digest(path)}
    source = {'frozen':True, 'source_tree_sha256':'a'*64, 'runtime_file_count':1, 'files':[row]}
    receipt = tmp_path/'source-receipt.json'
    receipt.write_text(json.dumps(source), encoding='utf-8')
    acceptance = {'source_tree_sha256':'a'*64, 'files':[{'path':'source-receipt.json', 'sha256':module.digest(receipt)}]}
    assert module.delivered_runtime_manifest(tmp_path, acceptance)==source
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='declared_file_integrity_failed'):
        module.delivered_runtime_manifest(tmp_path, acceptance)
