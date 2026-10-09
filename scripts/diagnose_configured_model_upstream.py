"""One bounded request to the already configured provider, preserving its proxy."""
import json
from pathlib import Path
import re
import time
import sys
from urllib.parse import urlsplit

import requests
import yaml


def main():
    config=yaml.safe_load(Path('C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml').read_text(encoding='utf-8-sig'))
    entries=[item for item in config.get('openai-compatibility',[]) if item.get('name')=='pezayo']
    if len(entries)!=1:raise SystemExit('configured_provider_ambiguous')
    provider=entries[0];base=provider.get('base-url','').rstrip('/')
    if urlsplit(base).scheme!='https' or urlsplit(base).hostname!='api.pezayo.com':raise SystemExit('unapproved_provider_origin')
    secrets=provider.get('api-key-entries') or []
    if len(secrets)!=1:raise SystemExit('configured_credential_ambiguous')
    proxy=secrets[0].get('proxy-url') or ''
    if proxy.strip().lower() in ('direct','direct://'): proxy=''
    body={'model':'gpt-6-astra','messages':[{'role':'user','content':'Return one contract_probe call with value 37. No tool will run.'}],
          'tools':[{'type':'function','function':{'name':'contract_probe','description':'Synthetic transport check','parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False},'strict':False}}],
          'max_tokens':256,'reasoning_effort':'low'}
    if '--full-tools' in sys.argv:
        sys.path.insert(0,'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v2')
        from evomind_runtime.tools import build_default_registry
        registry=build_default_registry()
        body['messages']=[{'role':'user','content':'Transport test only: return one file_read call with path inputs/data.json. No tools will run.'}]
        body['tools']=[{'type':'function','function':{'name':name,'description':registry.get(name).description,'parameters':registry.get(name).input_schema,'strict':False}} for name in ['file_read','file_write','directory_hash','artifact_publish','artifact_list','report_generate','report_status']]
        body.update(max_tokens=4096,temperature=0.3,service_tier='priority')
    if '--standard-tier' in sys.argv:
        body.pop('service_tier',None)
    started=time.monotonic();result={'scope':'configured_upstream_diagnostic','tools_executed':0,'tool_count':len(body['tools']),'provider_origin_preserved':True,'proxy_configuration_preserved':True}
    try:
        session=requests.Session()
        session.trust_env=False
        response=session.post(base+'/chat/completions' if base.endswith('/v1') else base+'/v1/chat/completions',json=body,
            headers={'Authorization':'Bearer '+secrets[0]['api-key']},proxies={'http':proxy,'https':proxy} if proxy else {},timeout=(10,55))
        value=response.json();error=value.get('error') or {};message=str(error.get('message') or '') if isinstance(error,dict) else ''
        if re.search(r'password|api.?key|token|credential|secret|cookie|登录|账号|密码|口令|凭据',message,re.I):message='[sensitive detail withheld]'
        result.update(http_status=response.status_code,error_type=error.get('type') if isinstance(error,dict) else None,error_message=message[:700],model=value.get('model'),choices=len(value.get('choices',[])))
    except Exception as error:result.update(error_class=type(error).__name__)
    result['elapsed_seconds']=time.monotonic()-started
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
