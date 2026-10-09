"""Packaging must bind every v2 launcher consumer, not just Start-Node."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/syscomplete_release_assemble.py'
    spec = importlib.util.spec_from_file_location('launcher_pack_binding', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(root):
    targets = {
        'candidate/Start-Node.ps1': 'C:/ProgramData/EvoMind/bundle/scripts/Start-Node.ps1',
        'candidate/lib/verify_report_dependencies.py': 'C:/ProgramData/EvoMind/bundle/scripts/lib/verify_report_dependencies.py',
        'candidate/report-dependency-activation.json': 'C:/ProgramData/EvoMind/report-envs/gpt55-806f87b009cd/report-dependency-activation.json',
    }
    rows = []
    for name in [*targets, 'Start-Node.gpt55-report.patch']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ('fixture only: ' + name).encode()
        path.write_bytes(data)
        rows.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    deployment = [{**{key: row[key] for key in ('path', 'sha256')}, 'target': targets[row['path']]}
                  for row in rows if row['path'] in targets]
    return {'schema': 'evomind.launcher_candidate.v2', 'production_changed': False, 'activation_claim': False,
            'profile': {'model': 'gpt-5.5'}, 'files': rows, 'deployment_files': deployment,
            'candidate': deployment[0], 'dependency_binding': deployment[2]}


def verify(root, manifest):
    data = json.dumps(manifest).encode()
    path = root / 'manifest.json'
    path.write_bytes(data)
    return module().verified_launcher_binding(path, hashlib.sha256(data).hexdigest())


def test_binds_all_three_deployment_files(tmp_path):
    expected = fixture(tmp_path)
    result = verify(tmp_path, expected)
    assert result['deployment_files'] == expected['deployment_files']
    assert result['activated'] is False and result['included_in_source_bundle'] is False


@pytest.mark.parametrize('change', ['missing_helper', 'foreign_target', 'dependency_hash', 'candidate_hash', 'extra_file'])
def test_rejects_incomplete_or_mismatched_binding(tmp_path, change):
    value = fixture(tmp_path)
    if change == 'missing_helper':
        value['deployment_files'].pop(1)
    elif change == 'foreign_target':
        value['deployment_files'][1]['target'] = 'C:/Windows/other.py'
    elif change == 'dependency_hash':
        value['dependency_binding'] = {**value['dependency_binding'], 'sha256': '0' * 64}
    elif change == 'candidate_hash':
        value['candidate'] = {**value['candidate'], 'sha256': '0' * 64}
    else:
        value['files'].append({**value['files'][0], 'path': 'candidate/extra.ps1'})
    with pytest.raises(ValueError):
        verify(tmp_path, value)


@pytest.mark.parametrize('index', [0, 1, 2, 3])
def test_each_artifact_is_rehashed(tmp_path, index):
    value = fixture(tmp_path)
    (tmp_path / value['files'][index]['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact_changed'):
        verify(tmp_path, value)
