"""Freeze the explicitly reviewed web-only September 8 debug batch, then build.

No network deployment, model calls, GPU operations, or working-tree wholesale copy.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'src/components/workstation/AppShell.tsx',
    'src/components/workstation/Sidebar.tsx',
    'src/components/workstation/UserResearchJourney.tsx',
    'src/components/workstation/screens/SettingsScreen.tsx',
    'src/components/workstation/screens/DataKaggleScreen.tsx',
    'src/components/workstation/screens/ProjectsScreen.tsx',
    'src/components/workstation/screens/GpuHpcScreen.tsx',
    'src/app/api/settings/route.ts',
    'src/lib/connector-presentation.ts',
    'src/lib/connector-presentation.test.mjs',
    'src/lib/server/user-preferences.ts',
    'src/lib/server/user-preferences.test.mjs',
    'src/lib/server/report-assistant-bridge-contract.test.ts',
    'src/lib/system-debug-ui.test.mjs',
)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',choices=['1','2','3'],default='1')
    batch=parser.parse_args().batch
    spec = importlib.util.spec_from_file_location('debug_assembly', ROOT/'scripts/syscomplete_release_assemble.py')
    assembly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembly)
    source = Path('D:/EV12/report-ui-websource-v6' if batch=='1' else f'D:/EV12/system-debug-websource-v{int(batch)-1}')
    source_sha = {'1':'b1467fb7dfbd10d114a118e9f6e8c2033b1d16d989a39935df5de852a94e5e17',
        '2':'d2d14aa3d20148a15023796717d23582ff7158d6ecab4a92e3bf45aa89860597',
        '3':'6c75a25d3c576fd958790047b88d7c33d7d3b0846c708f5909a8c934b7e72964'}[batch]
    files=FILES+(() if batch=='1' else (
        'src/components/workstation/WorkspaceToolbar.tsx',
        'src/components/workstation/screens/WorkflowScreen.tsx',
        'src/components/workstation/screens/OverviewScreen.tsx',
        'src/components/workstation/SuperAgentV1Panel.tsx',
        'src/app/api/super-agent/status/route.ts',
    ))
    frozen, _ = assembly.read_source(source, source_sha)
    rows = {r['path']: r for r in frozen['files']}
    current = ROOT/'web/research-agent-workstation'
    changes = []
    for name in files:
        path = current/name
        before = rows.get('web/'+name)
        changes.append({'path':'web/'+name,'bytes':path.stat().st_size,'sha256':sha(path),'before_sha256':before['sha256'] if before else None})
    evidence = ROOT/'artifacts/system-debug-20260908'
    delta = evidence/f'web-delta-v{batch}.json'
    assembly.BASE.write_new(delta, assembly.BASE.canonical_json({'schema':'evomind.syscomplete_web_delta.v1','frozen':True,'parent_manifest_sha256':source_sha,'files':changes}))
    # This wrapper's fixed file set is the only extension to the older report allowlist.
    assembly.WEB_CORRECTIONS = assembly.WEB_CORRECTIONS | {'web/'+name for name in files}
    destination = Path(f'D:/EV12/system-debug-websource-v{batch}')
    result = assembly.amend_web(source, source_sha, delta, sha(delta), current, destination)
    print(json.dumps(result), flush=True)
    old_release = Path('D:/EV12/report-ui-release-v6')
    from argparse import Namespace
    args = Namespace(source=destination,source_manifest_sha256=result['manifest_sha256'],
        destination=Path(f'D:/EV12/system-debug-release-v{batch}'),
        runtime_root=Path('D:/EV12/report-ui-runtime-v4'),
        runtime_manifest=Path('D:/EV12/report-ui-runtime-v4/runtime-delta-manifest.json'),
        runtime_manifest_sha256='6b5cc71af5518fb5a5e50829feb46214e47c6a952746e704ccf4811fad29703c',
        report_dependencies=old_release/'input-receipts/report-dependency-delivery.json',
        report_dependencies_sha256=sha(old_release/'input-receipts/report-dependency-delivery.json'),
        report_wheel_root=old_release/'web/support/report-wheelhouse',
        report_inheritance=old_release/'input-receipts/report-inheritance.json')
    receipt = assembly.assemble(args)
    os.environ['PATH'] = 'D:/下载'+os.pathsep+os.environ.get('PATH','')
    built = assembly.LEGACY.build(receipt)
    print(json.dumps(built),flush=True)

if __name__ == '__main__':
    main()
