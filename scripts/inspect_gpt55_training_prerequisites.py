"""Read only local training policy/budget metadata; never connect to HPC."""
import json
from pathlib import Path
import sqlite3

import psutil


def main():
    root = Path('C:/ProgramData/EvoMind')
    config = json.loads((root / 'config/node-config.json').read_text(encoding='utf-8-sig'))
    state = json.loads((root / 'state/node-processes.json').read_text(encoding='utf-8-sig'))
    record = next(row for row in state['records'] if row['role'] == 'python_runtime')
    environment = psutil.Process(record['pid']).environ()
    names = ['EVOMIND_AIBUILD_POLICY_FILE','EVOMIND_SIIM_HPC_JOB_ID','EVOMIND_HPC_CREDENTIAL_PROFILE',
             'EVOMIND_HPC_EXPECTED_HOST_UUID','EVOMIND_HPC_EXPECTED_GPU_UUID','EVOMIND_AIBUILD_MODE','WORKSTATION_BYOA_RUNTIME_ROOT']
    hpc = config.get('hpc') or {}
    result = {'scope':'local_metadata_only_not_hpc_connection', 'hpc_accessed':False, 'gpu_actions':0,
        'hpc_config':{key:hpc.get(key) for key in ['state','job_id','credential_profile','allocation_generation'] if key in hpc},
        'runtime_configuration':{key:environment.get(key) for key in names}}
    policy_path = environment.get('EVOMIND_AIBUILD_POLICY_FILE')
    if not policy_path and environment.get('WORKSTATION_BYOA_RUNTIME_ROOT'):
        policy_path = str(Path(environment['WORKSTATION_BYOA_RUNTIME_ROOT']).parent.parent / 'config/research-control/policy.json')
    if policy_path and Path(policy_path).is_file():
        policy = json.loads(Path(policy_path).read_text(encoding='utf-8-sig'))
        result['policy'] = {key:policy.get(key) for key in ['schema','enabled','runtime_root','study_id','gpu_hours','engineering_gpu_hours','max_trial_seconds','max_model_calls','max_tokens'] if key in policy}
        result['policy_field_names'] = sorted(policy)
        result['policy_path'] = policy_path
        result['registered_protocol_files'] = len(list(Path(policy_path).parent.glob('protocols/*.json')))
    runtime_root = root / 'data/workspace/runtime'
    result['local_budget_files'] = [str(path) for path in runtime_root.glob('*budget*') if path.is_file()]
    with sqlite3.connect((runtime_root/'gpu_budget.sqlite3').as_uri()+'?mode=ro', uri=True) as connection:
        result['budget_by_study_and_state'] = connection.execute('SELECT study,kind,status,count(*),sum(reserved),sum(charged) FROM gpu_operations GROUP BY study,kind,status').fetchall()
    with sqlite3.connect((runtime_root/'runtime.sqlite3').as_uri()+'?mode=ro', uri=True) as connection:
        row = connection.execute('SELECT metadata_json FROM sessions WHERE id=?', ('run_0c57bbde44e94a988843c2f6981f3c65',)).fetchone()
        if row:
            metadata = json.loads(row[0])
            identity = metadata.get('managed_hpc_identity') or {}
            result['historical_run_identity_not_fresh_verification'] = {key:identity.get(key) for key in ['job_id','generation','credential_profile','profile_id','profile_instance_id','remote_root'] if key in identity}
            result['historical_identity_field_names'] = sorted(identity)
            tenant = identity.get('tenant_id', '')
            if __import__('re').fullmatch(r'tenant_[a-f0-9]{24}', tenant):
                binding_path = root / 'byoa/tenants' / tenant / 'hpc-binding.json'
                if binding_path.is_file():
                    binding = json.loads(binding_path.read_text(encoding='utf-8-sig'))
                    result['current_binding_metadata_only'] = {key:binding.get(key) for key in ['state','job_id','credential_profile','allocation_generation','profile_instance_id','remote_root','updated_at_utc']}
                    result['binding_matches_historical_run'] = all(binding.get(key) == identity.get(key) for key in ['job_id','credential_profile','allocation_generation','profile_instance_id','allocation_binding_id'])
                    generation = binding.get('allocation_generation')
                    result['binding_tombstone_present'] = any((binding_path.parent / f'hpc-binding.g{generation}.{kind}.tombstone.json').exists() for kind in ['frozen','retired'])
    if policy_path and Path(policy_path).is_file():
        summaries = []
        for path in (Path(policy_path).parent/'protocols').glob('*.json'):
            protocol = json.loads(path.read_text(encoding='utf-8-sig'))
            summaries.append({'protocol_id':path.stem,'scope':protocol.get('scope'),'task':protocol.get('task'),
                'metric':protocol.get('metric'),'source':protocol.get('source'),
                'training_file':(protocol.get('training') or {}).get('path'),
                'evaluation_file':(protocol.get('evaluation') or {}).get('path'),
                'labels_read':False})
        result['protocol_metadata_only'] = summaries
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__ == '__main__': main()
