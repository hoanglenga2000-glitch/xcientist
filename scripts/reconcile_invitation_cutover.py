"""Reconcile a hash-verified partial invitation cutover without restoring data."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path('C:/ProgramData/EvoMind')
    stage = args.stage_root.resolve()
    backup = args.backup_root.resolve()
    if not stage.is_relative_to((root / 'staging').resolve()) or not backup.is_relative_to((root / 'backups').resolve()):
        raise RuntimeError('reconciliation_scope_rejected')
    spec = importlib.util.spec_from_file_location('transaction', stage / 'invitation_release_transaction.py')
    tx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tx)
    plan_path = stage / 'cutover-plan.json'
    if tx.sha(plan_path) != args.expected_plan_sha256:
        raise RuntimeError('plan_hash_changed')
    plan = json.loads(plan_path.read_text())
    if tx.sha(stage / 'invitation_release_transaction.py') != plan['transaction_script_sha256']:
        raise RuntimeError('transaction_hash_changed')
    history_path = stage / 'history-preservation-acceptance.json'
    if tx.sha(history_path) != plan['history_acceptance_sha256']:
        raise RuntimeError('history_receipt_changed')
    history = json.loads(history_path.read_text())
    target = root / 'bundle/runtime/evomind_runtime'
    candidate = backup / 'failed_runtime'
    old_destination = backup / 'reconciled_old_runtime'
    baseline = json.loads((backup / 'bundle-integrity.json').read_text(encoding='utf-8-sig'))
    binding = root / 'byoa/tenants/tenant_ba0ef9d3767f2fb385b856e0/hpc-binding.json'

    def guard():
        tx.assert_idle(root, preservation=plan['preserved_history'])
        if tx.code_tree(target) != plan['old_runtime'] or tx.code_tree(candidate) != history['candidate_code']:
            raise RuntimeError('runtime_tree_changed')
        if old_destination.exists():
            raise RuntimeError('recovery_destination_exists')
        config = json.loads((root / 'config/node-config.json').read_text(encoding='utf-8-sig'))
        if Path(config['web_runtime_root']).name != plan['build_id']:
            raise RuntimeError('configured_web_changed')
        if tx.sha(binding) != plan['binding_sha256']:
            raise RuntimeError('hpc_binding_changed')
        for name, expected in plan['unchanged_release_seals'].items():
            if tx.sha(root / 'state' / name) != expected:
                raise RuntimeError('release_seal_changed')
        tx.bundle_manifest(root, baseline, allow_runtime_change=False)
        manifest = json.loads((Path(config['web_runtime_root']) / 'runtime-build-manifest.json').read_text())
        if manifest['build_id'] != plan['build_id']:
            raise RuntimeError('web_manifest_changed')

    def managed(action):
        script = """$ProgressPreference='SilentlyContinue'
try {
  & C:/SecureInput/Invoke-ServiceAccountAction.ps1 -Action ACTION_PLACEHOLDER -TimeoutMinutes 12
} catch {
  $code='diagnostic_withheld'
  if ($_.Exception.Message -match '^([A-Z][A-Z_]{5,})(?:[: ]|$)') { $code=$Matches[1] }
  [ordered]@{status='failed';error_type=$_.Exception.GetType().Name;error_code=$code;error_id=$_.FullyQualifiedErrorId;category=[string]$_.CategoryInfo.Category} | ConvertTo-Json -Compress
  exit 1
}
""".replace('ACTION_PLACEHOLDER', action)
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True, timeout=900)
        try:
            receipt = json.loads(result.stdout.decode('utf-8-sig').strip())
        except (ValueError, UnicodeError):
            receipt = {'status': 'failed', 'error_code': 'managed_response_invalid'}
        tx.write_json(stage / ('reconciliation-' + action.lower() + '.json'), receipt)
        print(json.dumps({'phase': 'managed_' + action.lower(), 'receipt': receipt}), flush=True)
        if result.returncode or receipt.get('status') != 'completed':
            raise RuntimeError('managed_' + action.lower() + '_failed')

    guard()
    print(json.dumps({'status': 'preflight_passed', 'build_id': plan['build_id'], 'apply': args.apply}), flush=True)
    if not args.apply:
        return
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex = kernel.CreateMutexW(None, False, 'Global\\EvoMind-Byoa-V12-Deployment')
    if not mutex or kernel.WaitForSingleObject(mutex, 0) not in (0, 0x80):
        raise RuntimeError('deployment_in_progress')
    result = {'status': 'pending', 'build_id': plan['build_id'], 'plan_sha256': args.expected_plan_sha256, 'research_database_restored': False}
    try:
        guard()
        managed('Stop')
        guard()
        target.rename(old_destination)
        candidate.rename(target)
        tx.seal(root, baseline)
        managed('Start')
        result['health'] = tx.wait_health(plan['build_id'])
        tx.assert_idle(root, preservation=plan['preserved_history'])
        tx.bundle_manifest(root, json.loads((root / 'state/bundle-integrity.json').read_text()), allow_runtime_change=False)
        if tx.code_tree(target) != history['candidate_code'] or tx.sha(binding) != plan['binding_sha256']:
            raise RuntimeError('post_reconciliation_identity_changed')
        result.update(status='reconciled_and_verified', preserved_history=True)
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error_code=str(error) if str(error).replace('_', '').isalnum() else 'diagnostic_withheld')
        raise
    finally:
        tx.write_json(stage / 'reconciliation-result.json', result)
        print(json.dumps(result), flush=True)
        kernel.ReleaseMutex(mutex)
        kernel.CloseHandle(mutex)


if __name__ == '__main__':
    main()
