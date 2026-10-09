"""Create a local review archive, never a deployable release or a data backup."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/product-launch-20261008'
PUBLISHED = Path('D:/EV12/system-repair-20261008-gates')
STAGE = Path('D:/EV12/product-launch-local-20261008')
OPERATIONS = ['scripts/manage_product_accounts.py', 'scripts/invitation_release_transaction.py']
VERIFICATION = [
    'scripts/product_launch_scope.py', 'scripts/run_product_launch_preview.py',
    'scripts/product_launch_provider_fixture.py', 'scripts/task_workspace_provider_fixture.py',
    'scripts/check_product_launch_local.py', 'scripts/package_product_launch_review.py',
    'tests/test_user_tasks_http.py', 'tests/test_product_tool_boundary.py',
    'tests/test_product_pause_resume_http.py', 'tests/test_model_profiles_http.py',
    'tests/test_user_pause_contract.py', 'tests/test_approval_async_http.py',
    'tests/test_approval_recovery.py', 'tests/test_invitation_release_transaction.py',
    'tests/test_invitation_tenant_access.py', 'tests/test_invitation_run_requests.py',
    'tests/test_assistant_run_service.py', 'tests/conftest.py',
    'tests/fixtures/canonical_json_f64_v1.json',
]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    scope = json.loads((OUT / 'source-scope.json').read_text(encoding='utf-8'))
    preview = json.loads((OUT / 'preview.json').read_text(encoding='utf-8'))
    assert preview['candidate'] == scope['candidate']
    assert preview['next_build_id'] == (STAGE / 'web/.next/BUILD_ID').read_text().strip()
    assert sha(PUBLISHED / 'source-receipt.json') == scope['published_receipt_sha256']
    entries = []
    delta = []
    for row in scope['changes']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
        assert sha(STAGE / row['target']) == row['sha256'], row['target']
        group = 'verification' if row['test_only'] else 'source'
        entries.append((group, row['path']))
    entries.extend(('operations', name) for name in OPERATIONS)
    entries.extend(('verification', name) for name in VERIFICATION)
    manifest = []
    for group, name in entries:
        current = ROOT / name
        assert current.is_file(), name
        manifest.append({'group': group, 'path': name, 'bytes': current.stat().st_size, 'sha256': sha(current)})
        before = OUT / 'before' / name
        if group in {'source', 'operations'}:
            delta.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(keepends=True) if before.exists() else [],
                current.read_text(encoding='utf-8').splitlines(keepends=True),
                fromfile='session-start/' + name, tofile='current/' + name))
    dependency_inputs = []
    for name in ('package.json', 'package-lock.json', 'next.config.mjs', 'tsconfig.json', 'postcss.config.mjs', 'tailwind.config.ts'):
        file = STAGE / 'web' / name
        if file.is_file():
            dependency_inputs.append({'path': 'web/' + name, 'sha256': sha(file),
                'same_as_published': sha(file) == sha(PUBLISHED / 'web' / name)})
    # These are fingerprinted for review, NOT carried into the candidate archive.
    # The runtime hotfix still imports shared base packages from the local repo.
    shared_base = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)}
                   for p in sorted((ROOT / 'src').rglob('*.py'))
                   if 'evomind_runtime' not in p.relative_to(ROOT / 'src').parts and '__pycache__' not in p.parts]
    receipt = {'schema': 'evomind.product_launch.review.v1', 'candidate': scope['candidate'],
        'next_build_id': preview['next_build_id'], 'production_deployed': False,
        'purpose': 'review only; not a self-contained or authorized deployment package',
        'published_receipt_sha256': scope['published_receipt_sha256'], 'files': manifest,
        'dependency_inputs': dependency_inputs,
        'shared_runtime_base': {'verified_against_server': False, 'packaged': False, 'files': shared_base},
        'excluded': ['accounts', 'credentials', 'databases', 'user files', 'node_modules', 'unrelated dirty-tree patches']}
    (OUT / 'review-source-manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'session-delta.patch').write_text(''.join(delta), encoding='utf-8')
    archive = OUT / 'product-launch-review-only.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for group, name in entries:
            bundle.write(ROOT / name, group + '/' + name)
        for name in ('review-source-manifest.json', 'source-scope.json', 'preview.json', 'session-delta.patch'):
            bundle.write(OUT / name, 'receipts/' + name)
        bundle.write(ROOT / 'docs/plans/2026-10-08-production-product-launch.report.md', 'REPORT.md')
        bundle.writestr('README.txt', 'REVIEW ONLY. No production deployment authorization.\n'
            'source/ = bounded application patch; operations/ = separately reviewed operator tools;\n'
            'verification/ = local fixtures and tests, NEVER ship as production services.\n'
            'Existing shared runtime dependencies require server-baseline reconciliation.\n'
            'No accounts, credential store, database or user data is included.\n')
    result = {'candidate': scope['candidate'], 'archive': str(archive), 'sha256': sha(archive),
        'bytes': archive.stat().st_size, 'source_files': sum(g == 'source' for g, _ in entries),
        'operation_files': len(OPERATIONS), 'test_helper_files': sum(g == 'verification' for g, _ in entries),
        'production_deployed': False}
    (OUT / 'review-package.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
