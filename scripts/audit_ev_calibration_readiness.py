"""Read local installation/profile metadata only; never decrypt or connect to HPC."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from verify_hpc_profile_readiness import evaluate_profile

ROOT=Path(__file__).resolve().parents[1]


def main():
    profiles_root=Path(os.environ['APPDATA'])/'ResearchAgentWorkstation/profiles'
    profiles=[]
    if profiles_root.is_dir():
        for directory in sorted(profiles_root.iterdir()):
            if directory.is_dir() and re.fullmatch(r'job[1-9][0-9]*',directory.name):
                receipt=evaluate_profile(directory.name)
                profiles.append({'profile':directory.name,'local_metadata_status':receipt['status'],
                                 'profile_state':receipt['details'].get('profile_state'),
                                 'failed_checks':receipt['failed_checks']})
    historical_root=Path('C:/ProgramData/EvoMind')
    receipt={'schema':'evomind.official_calibration.local_readiness.v1',
             'checked_at':datetime.now(timezone.utc).isoformat(),
             'historical_managed_runtime_root_exists':historical_root.is_dir(),
             'local_profiles':profiles,'current_campaign_resource_binding':None,
             'remote_identity_verified':False,'remote_probe_performed':False,
             'credential_decryption_performed':False,'remote_writes':0,'gpu_hours_consumed':0,
             'training_started':False,
             'conclusion':'Current intended managed runtime/allocation must be identified before any remote action.'}
    output=ROOT/'artifacts/ev-public-calibration-20260908/local-readiness.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
