"""Independent stage-one acceptance; excludes any claim of training or official results."""
from __future__ import annotations
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from research_os.official_calibration import build_frozen_folds, sha256_file, validate_data_frames


def main():
    out=ROOT/'artifacts/ev-public-calibration-20260908'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(out/'agent-input/protocol.json')
    prepared=read(out/'preparation.json')
    manifest=read(out/'private-data/manifest.json')
    assert sha256_file(out/'agent-input/protocol.json')==prepared['protocol_sha256']==manifest['protocol_sha256']
    data=out/'private-data'
    for name,record in manifest['files'].items():
        assert (data/name).stat().st_size==record['bytes'] and sha256_file(data/name)==record['sha256']
    with zipfile.ZipFile(data/'playground-series-s6e9.zip') as archive:
        assert archive.testzip() is None
        for name in ['train.csv','test.csv','sample_submission.csv']:
            digest=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==manifest['files'][name]['sha256']
    train,test,sample=[pd.read_csv(data/name) for name in ['train.csv','test.csv','sample_submission.csv']]
    assert validate_data_frames(train,test,sample,protocol)==manifest['quality']
    pd.testing.assert_frame_equal(build_frozen_folds(train,protocol),pd.read_csv(data/'frozen-folds.csv'))
    view=read(out/'agent-input/agent-task-contract.json')
    assert 'baseline' not in view and view['baseline_configuration_disclosed'] is False
    assert view['source_manifest_sha256']==sha256_file(data/'manifest.json')
    suites=ET.parse(out/'all-tests.xml').getroot().findall('testsuite')
    test_count=sum(int(s.attrib['tests']) for s in suites)
    assert test_count>=87 and all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites)
    preflight=read(out/'runtime-preflight.json')
    assert preflight['status']=='blocked_model_service_before_hpc_verification'
    sources=['src/research_agent_workstation/tabular_pipeline.py','scripts/onboard_kaggle_competition.py',
             'scripts/validate_tabular_experiment.py','src/research_os/official_calibration.py',
             'scripts/prepare_ev_public_calibration.py','scripts/manage_ev_calibration_kaggle.ps1',
             'scripts/probe_ev_kaggle_access.py','scripts/freeze_ev_calibration_data.py',
             'scripts/build_ev_calibration_agent_contract.py','tests/test_ev_probability_contract.py',
             'tests/test_official_calibration.py']
    for source in sources:
        assert not re.search(r'KGAT_[A-Za-z0-9]{20,}',(ROOT/source).read_text(encoding='utf-8'))
    for path in out.rglob('*'):
        if path.suffix in {'.json','.md','.txt','.xml','.py'}:
            assert not re.search(r'KGAT_[A-Za-z0-9]{20,}',path.read_text(encoding='utf-8-sig'))
    result={'schema':'evomind.official_calibration.stage1_acceptance.v1',
            'verified_at':datetime.now(timezone.utc).isoformat(),'stage1_status':'passed',
            'overall_status':'blocked_model_service_before_training',
            'tests_passed':test_count,'data_files_rehashed_against_official_archive':True,
            'folds_independently_recomputed':True,'target_encoding':manifest['quality']['target_encoding'],
            'train_rows':len(train),'test_rows':len(test),'fold_rows':133733,
            'protocol_sha256':prepared['protocol_sha256'],'source_hashes':{p:sha256_file(ROOT/p) for p in sources},
            'runtime_preflight':preflight['run_id'],'training_runs_completed':0,'gpu_training_hours':0,
            'official_baseline_score':None,'official_evomind_score':None,'official_submissions':0,
            'human_gate_for_each_submission_retained':True,'no_plaintext_api_token_in_task_files':True,
            'remaining':['model_service_recovery','fresh_hpc_identity','managed_runtime_isolation_and_budget_enforcement',
                         'six_actual_training_runs','model_reload_verification','two_authorized_official_submissions','final_result_report']}
    (out/'stage1-acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},ensure_ascii=False))


if __name__=='__main__':main()
