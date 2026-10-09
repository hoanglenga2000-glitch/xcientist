"""Hash only installed report distributions; never mutate the production venv."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import sysconfig


def main():
    root = Path(sys.prefix).resolve()
    expected = Path('C:/EvoMind/releases/664a636ddd419a66f73cc10820c0a16429784866/.venv').resolve()
    if root != expected:
        raise ValueError('production_venv_binding_mismatch')
    base = Path('C:/ProgramData/EvoMind/web-overlays/overlay-invitation-beta-3800faca9fc5-ack1/release-source-manifest.json')
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    if base_sha != 'e3d80e0abd54dd73b4777dbe7b2b4f62bf488f9e4355119d297eb7931dab9507':
        raise ValueError('active_baseline_mismatch')
    packages = []
    for name, module_name in [('python-docx', 'docx'), ('PyMuPDF', 'fitz'), ('lxml', 'lxml'), ('typing-extensions', 'typing_extensions')]:
        distribution = importlib.metadata.distribution(name)
        imported = importlib.import_module(module_name)
        rows = []
        for entry in distribution.files or []:
            path = Path(distribution.locate_file(entry)).resolve(strict=True)
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise ValueError('distribution_path_invalid')
            content = path.read_bytes()
            rows.append({'path': relative, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        if not rows:
            raise ValueError('distribution_record_missing')
        packages.append({'name': name, 'version': distribution.version, 'module': module_name,
            'module_path': str(Path(imported.__file__).resolve()), 'import_passed': True,
            'requires': distribution.requires or [], 'files': sorted(rows, key=lambda row: row['path'])})
    payload = {'schema': 'evomind.report_dependency_inheritance.v1',
        'base_build_id': 'overlay-invitation-beta-3800faca9fc5-ack1',
        'base_source_tree_sha256': '3800faca9fc507c17152904af09629ccc98a74769723e4f7d1ae02ed6f1086be',
        'base_source_manifest_sha256': base_sha, 'venv_root': str(root),
        'python_version': platform.python_version(), 'python_abi': sysconfig.get_config_var('SOABI'),
        'platform': sysconfig.get_platform(), 'packages': packages,
        'production_changed': False, 'activation_claim': False}
    output = Path('C:/EMQA/sys0907-report-inheritance-v1.json')
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
    print(json.dumps({'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'packages': [{'name': item['name'], 'version': item['version'], 'files': len(item['files'])} for item in packages],
        'production_changed': False}))


if __name__ == '__main__':
    main()
