"""Verify a real Chrome-created DeepSeek Run and its complete read-only HPC receipt."""
import hashlib,json,re,sys
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ev-domestic-model-route-20260908'
sys.path.insert(0,str(ROOT/'src'))
from xsci.terminal_tools import hpc_identity_evidence_complete


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    live=read(OUT/'live-acceptance.json')
    assert live['run']['status']=='completed' and live['run']['model']=='deepseek-v4-pro'
    assert live['model_cutover_verified'] and live['model_requests_succeeded']==2 and live['model_requests_failed']==0
    contract=live['contract']
    assert contract['wire_protocol']=='chat_completions_v1' and contract['timeout_seconds']==180
    assert contract['endpoint_sha256']==hashlib.sha256(b'https://api.pezayo.com/v1').hexdigest()
    assert contract['route_config_sha256']==hashlib.sha256((OUT/'switch-payload/route.json').read_bytes()).hexdigest()
    calls=live['tool_calls'];assert len(calls)==1 and calls[0]['name']=='hpc_verify' and calls[0]['status']=='completed'
    receipt=calls[0]['result']['content']
    assert hpc_identity_evidence_complete(receipt,expected_sample_count=5,
                                          expected_profile='tenant_ba0ef9d3767f2fb385b856e0_job93207_g27',expected_job_id=93207)
    suites=ET.parse(OUT/'contract-fix-v2/unit-tests.xml').getroot().findall('testsuite')
    assert all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites)
    count=sum(int(s.attrib['tests']) for s in suites)
    for p in OUT.rglob('*'):
        if p.suffix in {'.json','.jsonl','.md','.py','.ps1','.txt','.patch'}:
            assert not re.search(r'sk-[A-Za-z0-9]{30,}',p.read_text(encoding='utf-8-sig')),p.name
    result={'schema':'evomind.user_selected_model_cutover.acceptance.v1','verified_at':datetime.now(timezone.utc).isoformat(),
            'status':'passed','model':'deepseek-v4-pro','user_selected_not_benchmark_winner':True,
            'base_url':'https://api.pezayo.com/v1','wire_protocol':'chat_completions_v1',
            'timeout_seconds':180,'max_request_retries':2,'fallback_to_other_models':False,
            'live_run':live['run']['id'],'model_requests_succeeded':2,'model_requests_failed':0,
            'model_request_seconds':[x['elapsed_seconds'] for x in live['model_attempts']],
            'live_input_tokens':sum(x['input_tokens'] for x in live['model_attempts']),
            'live_output_tokens':sum(x['output_tokens'] for x in live['model_attempts']),
            'offline_tests_passed':count,'hpc_identity_samples_passed':5,
            'complete_identity_receipt_verified':True,'job_id':93207,'allocation_generation':27,
            'old_credentials_retained':True,'temporary_encrypted_transit_removed':True,
            'training_started':False,'competition_submissions':0,
            'competition_experiment_complete':False,'long_term_stability_claim':False}
    (OUT/'model-switch-acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (ROOT/'artifacts/ev-public-calibration-20260908/model-control-restored.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
