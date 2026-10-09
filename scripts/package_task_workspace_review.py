"""Create a source-only review bundle, explicitly not a deployable release."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path('D:/EV12/task-ui-local-20261008')
PUBLISHED = Path('D:/EV12/system-repair-20261008-gates')
OUTPUT = ROOT / 'artifacts/advanced-tools-migration-20261008'
REPORT = 'docs/plans/2026-10-08-advanced-tools-usability.report-2.md'
SUPPORT = [
    'tests/test_user_tasks_http.py', 'tests/test_model_profiles_http.py', 'tests/test_personal_model_boundary.py',
    'scripts/capture_task_workspace_baseline.py', 'scripts/run_task_workspace_preview.py',
    'scripts/task_workspace_provider_fixture.py', 'scripts/verify_task_workspace_artifact.py',
    'scripts/verify_task_workspace_candidate.py', 'scripts/package_task_workspace_review.py', REPORT,
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(path, *, published=False):
    if path.startswith('web/research-agent-workstation/'):
        return (PUBLISHED if published else STAGE) / 'web' / path.removeprefix('web/research-agent-workstation/')
    return (PUBLISHED if published else STAGE) / 'runtime' / path.removeprefix('src/')


def write_json(name, data):
    (OUTPUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    baseline = json.loads((OUTPUT / 'baseline.json').read_text(encoding='utf-8'))
    scope = json.loads((OUTPUT / 'source-scope.json').read_text(encoding='utf-8'))
    whitelist = {row['path'] for row in scope['changes']}
    if sha(PUBLISHED / 'source-receipt.json') != baseline['published_source_receipt_sha256']:
        raise RuntimeError('published_source_changed')
    preservation = []
    for row in baseline['files']:
        current = sha(ROOT / row['path'])
        if row['path'] not in whitelist and current != row['before']:
            raise RuntimeError('outside_patch_change:' + row['path'])
        if row['before'] != row['published']:
            preservation.append({'path': row['path'], 'preserved': current == row['before'], 'sha256': current})
    write_json('baseline-preservation.json', {'checked_files': len(baseline['files']),
        'outside_patch_changes': [], 'preexisting_differences': preservation,
        'scope_note': 'Covers captured web/src and evomind_runtime files, not a whole-disk audit.'})
    diffs = []
    for row in scope['changes']:
        path = row['path']
        current = ROOT / path
        if sha(current) != row['sha256'] or sha(source_path(path)) != row['sha256']:
            raise RuntimeError('unbuilt_source_change:' + path)
        before = source_path(path, published=True)
        diffs.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(keepends=True) if row['before'] else [],
            current.read_text(encoding='utf-8').splitlines(keepends=True),
            fromfile='a/' + path if row['before'] else '/dev/null', tofile='b/' + path))
    (OUTPUT / 'candidate-source.patch').write_text(''.join(diffs), encoding='utf-8')
    inputs = [{'path': row['path'], 'sha256': row['sha256']} for row in scope['changes'] if '.test.' not in row['path']]
    identity = hashlib.sha256(json.dumps({'published_receipt': baseline['published_source_receipt_sha256'],
        'inputs': inputs}, sort_keys=True).encode()).hexdigest()
    receipt = {
        'schema': 'evomind.task_workspace_review_source.v1', 'candidate_id': 'local-task-ui-' + identity[:12],
        'candidate_source_sha256': identity, 'published_build_id': baseline['production_build'],
        'published_source_receipt_sha256': baseline['published_source_receipt_sha256'],
        'next_build_id': (STAGE / 'web/.next/BUILD_ID').read_text(encoding='utf-8').strip(),
        'deployable': False, 'production_mutated': False, 'real_provider_verified': False, 'hpc_verified': False,
        'runtime_source_inputs': inputs,
        'test_and_review_only': [{'path': name, 'sha256': sha(ROOT / name)} for name in SUPPORT] +
            [row for row in scope['changes'] if '.test.' in row['path']],
        'excluded': ['.env', 'credentials', 'databases', 'user history', 'node_modules', 'build output',
                     'preexisting unpublished changes', 'preview/advanced-tools'],
    }
    write_json('candidate-source-receipt.json', receipt)
    # Build diagnostics only; never copy runtime/web request logs or databases.
    (OUTPUT / 'candidate-build.log').write_bytes((STAGE / 'logs/build.log').read_bytes())
    evidence = [
        'source-scope.json', 'candidate-source-receipt.json', 'baseline-preservation.json',
        'candidate-source.patch', 'final-checks.json', 'final-candidate-regression.xml',
        'final-model-regression.xml', 'artifact-download.json', 'candidate-build.log',
        'candidate-web-tests.log', 'workspace-web-tests.log', 'python-fast-gates.log',
        'candidate-typecheck.log', 'workspace-typecheck.log', 'api-downloaded-artifact.md',
        'final-progress-desktop.png', 'final-progress-dom.txt',
        'final-results-desktop.png', 'final-results-dom.txt',
        'final-settings-desktop.png', 'final-settings-dom.txt',
        'restart-task-a-results.txt', 'restart-task-b-draft.txt',
    ]
    bundle = OUTPUT / 'task-ui-review-only.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(whitelist | set(SUPPORT)):
            archive.write(ROOT / path, path)
        for name in evidence:
            archive.write(OUTPUT / name, 'evidence/' + name)
        archive.writestr('NOT-A-RELEASE.txt',
            'Source review only. Do not deploy this ZIP or run fixture launchers on a server.\n'
            'Production cutover and legacy ownership migration are NOT authorized by this package.\n'
            'scripts/task_workspace_provider_fixture.py is a local external-service stub, NOT a model.\n')
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('bundle_corrupt')
    write_json('review-package.json', {'path': bundle.name, 'sha256': sha(bundle), 'bytes': bundle.stat().st_size,
        'candidate_id': receipt['candidate_id'], 'deployable': False, 'production_deployed': False})
    print(json.dumps({'candidate_id': receipt['candidate_id'], 'next_build_id': receipt['next_build_id'],
        'source_inputs': len(inputs), 'baseline_files': len(baseline['files']), 'bundle_sha256': sha(bundle), 'deployable': False}))


if __name__ == '__main__':
    main()
