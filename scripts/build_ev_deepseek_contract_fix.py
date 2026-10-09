from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ev-domestic-model-route-20260908/contract-fix-v2'
sha=lambda b:hashlib.sha256(b).hexdigest()
before=(OUT/'store-before.py').read_bytes()
needle="        if schema == 'evomind.model_execution_contract.v4':\n            fields |= {'credential_binding_sha256', 'route_config_sha256'}"
addition="        if schema == 'evomind.model_execution_contract.v4':\n            allowed_models.add('deepseek-v4-pro')\n            if contract.get('model') == 'deepseek-v4-pro' and contract.get('wire_protocol') != 'chat_completions_v1':\n                raise ValueError('model_contract_invalid')\n            fields |= {'credential_binding_sha256', 'route_config_sha256'}"
text=before.decode();assert text.count(needle)==1
after=text.replace(needle,addition,1).encode()
assert after.decode().replace(addition,needle,1).encode()==before
with (OUT/'store.py').open('xb') as f:f.write(after)
manifest={'schema':'evomind.deepseek_contract_admission_fix.v1','baseline_store_sha256':sha(before),
          'baseline_seal_sha256':sha((OUT/'seal-before.json').read_bytes()),'candidate_store_sha256':sha(after),
          'model':'deepseek-v4-pro','contract_version':'v4','allowed_protocol':'chat_completions_v1',
          'existing_contract_drift_guards_preserved':True}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'status':'candidate_created','manifest_sha256':sha((OUT/'manifest.json').read_bytes())}))
