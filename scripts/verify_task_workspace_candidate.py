"""Persist local candidate checks; no release, credential import or service call."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web/research-agent-workstation'
STAGE = Path('D:/EV12/task-ui-local-20261008/web')
OUTPUT = ROOT / 'artifacts/advanced-tools-migration-20261008'
NODE = Path('D:/下载/node.exe')


def main():
    env = {name: value for name, value in os.environ.items()
           if not any(part in name.upper() for part in ('API_KEY', 'AUTH_TOKEN', 'LLM_', 'ANTHROPIC_', 'DEEPSEEK_', 'OPENAI_'))}
    env['PYTHONUTF8'] = '1'
    commands = [
        ('candidate-web-tests', [str(NODE), '--test', '--experimental-strip-types', '--disable-warning=MODULE_TYPELESS_PACKAGE_JSON',
                                 'src/**/*.test.ts', 'src/**/*.test.mjs'], STAGE),
        ('candidate-typecheck', [str(NODE), str(WEB / 'node_modules/typescript/bin/tsc'), '--noEmit', '--incremental', 'false'], STAGE),
        ('workspace-web-tests', [str(NODE), '--test', '--experimental-strip-types', '--disable-warning=MODULE_TYPELESS_PACKAGE_JSON',
                                 'src/**/*.test.ts', 'src/**/*.test.mjs'], WEB),
        ('workspace-typecheck', [str(NODE), str(WEB / 'node_modules/typescript/bin/tsc'), '--noEmit', '--incremental', 'false'], WEB),
        ('python-fast-gates', [sys.executable, 'scripts/run_ci_checks.py', '--skip-tests'], ROOT),
    ]
    checks = []
    for name, command, cwd in commands:
        started = time.monotonic()
        check_env = dict(env)
        if cwd == STAGE:
            # The published source package ships test fixtures outside src.
            # Use its supported override instead of changing a production test.
            check_env['EVOMIND_TEST_FIXTURE_ROOT'] = str(STAGE / 'test-fixtures')
        previous = OUTPUT / (name + '.log')
        original = OUTPUT / (name + '.initial.log')
        if previous.exists() and not original.exists():
            original.write_bytes(previous.read_bytes())
        result = subprocess.run(command, cwd=cwd, env=check_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
        (OUTPUT / (name + '.log')).write_bytes(result.stdout)
        row = {'name': name, 'exit_code': result.returncode, 'seconds': round(time.monotonic() - started, 2),
               'working_directory': str(cwd), 'log': name + '.log'}
        checks.append(row)
        print(json.dumps(row), flush=True)
        (OUTPUT / 'final-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    return int(any(row['exit_code'] for row in checks))


if __name__ == '__main__':
    raise SystemExit(main())
