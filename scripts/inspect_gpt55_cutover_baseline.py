"""Read-only current release/dependency evidence; no credentials or raw logs."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import sys
import urllib.request

import psutil


def main():
    root = Path('C:/ProgramData/EvoMind')
    state = json.loads((root / 'state/node-processes.json').read_text(encoding='utf-8-sig'))
    result = {'scope': 'read_only_cutover_baseline', 'production_changed': False,
              'python_version': sys.version.split()[0]}
    web = next(item for item in state['records'] if item.get('role') == 'web')
    process = psutil.Process(web['pid'])
    cwd = Path(process.cwd())
    candidates = [cwd / 'release-source-manifest.json']
    for argument in process.cmdline():
        if argument.lower().endswith('server.js'):
            path = Path(argument)
            candidates.append((path if path.is_absolute() else cwd / path).parent / 'release-source-manifest.json')
    manifests = []
    for path in dict.fromkeys(candidates):
        if not path.is_file() or path.is_symlink():
            continue
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        manifests.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          **{key: payload.get(key) for key in ['schema', 'source_tree_sha256', 'web_build_id', 'build_id', 'file_count']}})
    result['active_web_manifests'] = manifests
    try:
        with urllib.request.urlopen('http://127.0.0.1:8088/api/healthz', timeout=8) as response:
            health = json.load(response)
        result['web_health'] = {key: health.get(key) for key in ['status', 'build_id', 'source_tree_sha256']}
    except Exception as error:
        result['health_error_class'] = type(error).__name__
    packages = []
    for distribution, module_name in [('python-docx', 'docx'), ('PyMuPDF', 'fitz'), ('lxml', 'lxml'), ('numpy', 'numpy'), ('matplotlib', 'matplotlib')]:
        item = {'package': distribution}
        try:
            item['version'] = importlib.metadata.version(distribution)
            imported = importlib.import_module(module_name)
            item['import_passed'] = True
            item['module_path'] = imported.__file__
        except Exception as error:
            item.update(import_passed=False, error_class=type(error).__name__)
        packages.append(item)
    result['production_venv_report_dependencies'] = packages
    database = root / 'data/workspace/runtime/runtime.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=3) as connection:
        connection.execute('PRAGMA query_only=ON')
        result['run_status_counts'] = connection.execute('SELECT status,count(*) FROM assistant_runs GROUP BY status').fetchall()
        row = connection.execute('SELECT id,status,error_class,updated_at FROM assistant_runs WHERE id=?',
            ('run_0c57bbde44e94a988843c2f6981f3c65',)).fetchone()
        if row:
            result['existing_research_run'] = {'id': row[0], 'status': row[1],
                'error_class': row[2] if re.fullmatch(r'[A-Za-z0-9_]{0,100}', row[2]) else 'detail_withheld', 'updated_at': row[3]}
            result['existing_research_tool_states'] = connection.execute('SELECT status,count(*) FROM tool_calls WHERE session_id=? GROUP BY status', (row[0],)).fetchall()
            result['latest_event_types'] = connection.execute('SELECT event_type,created_at FROM events WHERE session_id=? ORDER BY rowid DESC LIMIT 5', (row[0],)).fetchall()
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
