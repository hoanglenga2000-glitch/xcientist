"""Build a narrow, hash-bound launcher/transport patch from captured live source."""
from pathlib import Path
import hashlib,json,difflib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ev-domestic-model-route-20260908'
BASE=OUT/'remote-before'
STAGE=OUT/'switch-payload'
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    STAGE.mkdir(exist_ok=True)
    original=(BASE/'Start-Node.ps1').read_bytes()
    text=original.decode('utf-8')
    changes={
        "  $baseEnv['OPENAI_API_KEY'] = $upstreamKey": "  $deepseekCredential = Import-DpapiCredentialChecked -Path (Join-Path $secretPaths.root 'pezayo_ev_deepseek_v4_pro.xml') -ExpectedUserName '__PEZAYO_DEEPSEEK_V4_PRO__'\n  $baseEnv['OPENAI_API_KEY'] = $deepseekCredential.GetNetworkCredential().Password",
        "$baseEnv['OPENAI_MODEL'] = 'gpt-5.5'":"$baseEnv['OPENAI_MODEL'] = 'deepseek-v4-pro'",
        "$baseEnv['OPENAI_REASONING_EFFORT'] = 'low'":"$baseEnv['OPENAI_REASONING_EFFORT'] = ''",
        "$baseEnv['EVOMIND_MODEL_WIRE_PROTOCOL'] = 'responses'":"$baseEnv['EVOMIND_MODEL_WIRE_PROTOCOL'] = 'chat_completions'",
        "$baseEnv['EVOMIND_MODEL_TIMEOUT_SECONDS'] = '90'":"$baseEnv['EVOMIND_MODEL_TIMEOUT_SECONDS'] = '180'",
        "$baseEnv['EVOMIND_MODEL_ROUTE_CONFIG_PATH'] = $gatewayConfig":"$baseEnv['EVOMIND_MODEL_ROUTE_CONFIG_PATH'] = 'C:/ProgramData/EvoMind/state/ev-deepseek-route-20260908.json'",
    }
    for before,after in changes.items():
        assert text.count(before)==1,before
        text=text.replace(before,after,1)
    reverted=text
    for before,after in reversed(list(changes.items())):reverted=reverted.replace(after,before,1)
    assert reverted.encode()==original
    model_before=(BASE/'model_transport.py').read_bytes()
    needle='GOVERNED_MODELS = frozenset({"gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"})'
    assert model_before.decode().count(needle)==1
    model_after=model_before.decode().replace(needle,needle.replace('"gpt-5.5"','"gpt-5.5", "deepseek-v4-pro"'),1).encode()
    route={'schema':'evomind.explicit_model_route.v1','user_selected':True,'provider':'openai',
           'model':'deepseek-v4-pro','base_url':'https://api.pezayo.com/v1','wire_protocol':'chat_completions',
           'timeout_seconds':180,'reasoning_effort':None,'service_tier':None,'fallback_to_other_models':False,
           'credential_ref':'pezayo_ev_deepseek_v4_pro.xml','old_gpt_runs_not_resumed':True}
    payloads={'Start-Node.ps1':text.encode(),'model_transport.py':model_after,
              'route.json':(json.dumps(route,indent=2)+'\n').encode()}
    for name,data in payloads.items():
        with (STAGE/name).open('xb') as f:f.write(data)
    manifest={'schema':'evomind.deepseek_route_switch.v1','model':'deepseek-v4-pro',
              'baseline_launcher_sha256':sha(original),'baseline_transport_sha256':sha(model_before),
              'baseline_seal_sha256':sha((BASE/'bundle-integrity.json').read_bytes()),
              'payloads':{name:sha(data) for name,data in payloads.items()},'scope':['launcher_model_profile','selected_model_strict_transport','new_dedicated_model_credential'],
              'hpc_binding_changes':False,'historical_run_replay':False,'changes_live':False}
    (STAGE/'switch-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'launcher.patch').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),text.splitlines(True),fromfile='before/Start-Node.ps1',tofile='after/Start-Node.ps1')),encoding='utf-8')
    print(json.dumps({'status':'candidate_built_not_applied','manifest_sha256':sha((STAGE/'switch-manifest.json').read_bytes()),'model':route['model']}))


if __name__=='__main__':main()
