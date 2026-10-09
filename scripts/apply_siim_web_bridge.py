"""Transactional app-only deployment; no scientific-policy, data or budget reset."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path('C:/ProgramData/EvoMind')
STAGE = BASE/'staging/siim-mlebench-calibration-20260908/web-entry-repair'
BACKUP = BASE/'backups/siim-web-entry-20260909'
sys.path.insert(0, str(BASE/'staging/ev-deepseek-v4-pro-20260908'))
import apply_ev_deepseek_route as ops
ops.BACKUP = BACKUP


def budgets():
    result = {}
    for name in ['ev_calibration_budget.sqlite3', 'siim_calibration_budget.sqlite3']:
        with sqlite3.connect((BASE/'data/workspace/runtime'/name).as_uri()+'?mode=ro', uri=True) as c:
            c.execute('PRAGMA query_only=ON')
            tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            result[name] = {table: sorted(c.execute('SELECT * FROM '+table).fetchall(), key=repr) for table in tables}
            states = c.execute('SELECT status FROM attempts').fetchall()
            ops.require(all(row[0] in {'completed', 'failed'} for row in states), 'calibration_in_flight')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    ops.require(ops.sha(STAGE/'manifest.json') == args.manifest_sha256, 'manifest_changed')
    manifest = json.loads((STAGE/'manifest.json').read_text())
    policy = BASE/'config/official-calibration/siim-mlebench-20260908.json'
    seal_path = BASE/'state/bundle-integrity.json'
    ops.require(ops.sha(policy) == manifest['policy_sha256_unchanged'], 'policy_changed')
    ops.require(ops.sha(seal_path) == manifest['baseline_seal_sha256'], 'seal_changed')
    for name, item in manifest['files'].items():
        target = BASE/item['target']
        ops.require(ops.sha(STAGE/name) == item['sha256'], 'payload_changed')
        ops.require((target.is_file() and ops.sha(target) == item['before_sha256']) if item['before_sha256']
                    else not target.exists(), 'live_target_changed')
    before = {'work': ops.work_guard(), 'budgets': budgets()}
    if not args.apply:
        print(json.dumps({'status': 'web_bridge_ready_to_activate', 'files': len(manifest['files']), 'training_started': False}))
        return
    ops.require(not BACKUP.exists(), 'prior_deployment_requires_reconciliation')
    with ops.deployment_lock():
        BACKUP.mkdir(parents=True)
        subprocess.run(['icacls.exe', str(BACKUP), '/inheritance:r', '/grant:r', 'SYSTEM:(OI)(CI)F', 'Administrators:(OI)(CI)F'], check=True, stdout=subprocess.DEVNULL)
        saved = {}
        for index, item in enumerate(manifest['files'].values()):
            target = BASE/item['target']
            backup = BACKUP/('file-'+str(index))
            if target.exists():
                shutil.copy2(target, backup)
                saved[str(target)] = str(backup)
            else:
                saved[str(target)] = None
        shutil.copy2(seal_path, BACKUP/'seal.json')
        ops.write(BACKUP/'before.json', {**before, 'files': saved})
        changed = False
        try:
            ops.service('Stop')
            ops.require({'work': ops.work_guard(), 'budgets': budgets()} == before, 'state_changed_during_stop')
            for name, item in manifest['files'].items():
                target = BASE/item['target']
                temporary = target.with_name(target.name+'.web-new')
                ops.require(not temporary.exists(), 'temporary_collision')
                shutil.copyfile(STAGE/name, temporary)
                subprocess.run(['icacls.exe', str(temporary), '/inheritance:r', '/grant:r', 'SYSTEM:F', 'Administrators:F', 'EvoMindSvc:R'], check=True, stdout=subprocess.DEVNULL)
                os.replace(temporary, target)
                changed = True
            seal = json.loads(seal_path.read_text(encoding='utf-8-sig'))
            altered = {item['target'][len('bundle/'):]: item for item in manifest['files'].values() if item['target'].startswith('bundle/')}
            found = set()
            for row in seal['files']:
                path = BASE/'bundle'/row['path']
                if row['path'] in altered:
                    row.update(sha256=ops.sha(path), size=path.stat().st_size)
                    found.add(row['path'])
                else:
                    ops.require(ops.sha(path) == row['sha256'], 'unrelated_bundle_changed')
            for relative in set(altered)-found:
                path = BASE/'bundle'/relative
                seal['files'].append({'path': relative, 'sha256': ops.sha(path), 'size': path.stat().st_size})
            seal['sealed_at_utc'] = datetime.now(timezone.utc).isoformat()
            temporary = seal_path.with_name('bundle-integrity.web-new.json')
            ops.write(temporary, seal)
            os.replace(temporary, seal_path)
            ops.require(budgets() == before['budgets'] and ops.sha(policy) == manifest['policy_sha256_unchanged'], 'study_changed')
            ops.service('Start')
            ops.require({'work': ops.work_guard(), 'budgets': budgets()} == before, 'state_changed_after_start')
            result = {'status': 'web_bridge_activated_pending_browser_acceptance', 'manifest_sha256': args.manifest_sha256,
                      'training_started': False, 'budgets_and_policy_unchanged': True,
                      'historical_assistant_rows_created': 0, 'historical_results_rewritten': False}
            ops.write(BACKUP/'result.json', result)
            print(json.dumps(result), flush=True)
        except Exception as error:
            if changed:
                ops.service('Stop')
                for target, backup in saved.items():
                    if backup:
                        shutil.copy2(backup, target)
                    elif Path(target).exists():
                        Path(target).rename(BACKUP/('failed-new-'+Path(target).name))
                shutil.copy2(BACKUP/'seal.json', seal_path)
                ops.service('Start')
            ops.write(BACKUP/'failure.json', {'error_type': type(error).__name__, 'rollback_attempted': changed})
            raise


if __name__ == '__main__':
    main()
