"""Bounded native endpoint comparison; requests are synthetic and no tools run."""
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0,'C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v2')
import yaml
from evomind_runtime.tools import build_default_registry


def safe_error(value):
    if not isinstance(value,dict): return ''
    text=str(value.get('message') or '')
    if re.search(r'password|api.?key|token|credential|secret|cookie|登录|账号|密码|口令|凭据',text,re.I):return '[sensitive detail withheld]'
    return re.sub(r'sk-[A-Za-z0-9_-]+|Bearer\s+\S+','[redacted]',text,flags=re.I)[:900]


def main():
    key=yaml.safe_load(Path('C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml').read_text(encoding='utf-8-sig'))['api-keys'][0]
    registry=build_default_registry()
    names=['file_read','file_write','directory_hash','artifact_publish','artifact_list','report_generate','report_status']
    functions=[{'type':'function','name':name,'description':registry.get(name).description,'parameters':registry.get(name).input_schema,'strict':False} for name in names]
    cases=[('responses_seven','/v1/responses',{'model':'gpt-6-astra','input':[{'role':'user','content':'Transport test only: return one file_read call with path inputs/data.json. No tools will run.'}],
        'tools':functions,'reasoning':{'effort':'low'},'max_output_tokens':512,'store':False}),
        ('chat_one','/v1/chat/completions',{'model':'gpt-6-astra','messages':[{'role':'user','content':'Transport test only: return one file_read call with path inputs/data.json. No tools will run.'}],
         'tools':[{'type':'function','function':{k:v for k,v in functions[0].items() if k!='type'}}],'reasoning_effort':'low','max_tokens':512})]
    results=[]
    for label,route,body in cases:
        started=time.monotonic(); row={'label':label,'tools_executed':0}
        try:
            request=urllib.request.Request('http://127.0.0.1:65068'+route,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=75) as response:value=json.load(response)
            row.update(status='json_received',error_type=(value.get('error') or {}).get('type'),error_message=safe_error(value.get('error')),reported_model=value.get('model'),output_types=[item.get('type') for item in value.get('output',[])],choices=len(value.get('choices',[])))
        except Exception as error:row.update(status='failed',error_class=type(error).__name__,http_status=getattr(error,'code',None))
        row['elapsed_seconds']=time.monotonic()-started;results.append(row);print(json.dumps(row),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
