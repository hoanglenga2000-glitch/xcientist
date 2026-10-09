"""Read-only schema ablation against the existing gateway; never execute tools."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

CANDIDATE = Path('C:/ProgramData/EvoMind/staging/system-completeness-20260907/endurance-candidate-v2')
sys.path.insert(0, str(CANDIDATE))
import yaml
from evomind_runtime.tools import build_default_registry
from research_os.agent import messaging
from research_os.llm_client import ProviderConfig


def typed(value):
    if isinstance(value, dict):
        result = {key: typed(item) for key, item in value.items()}
        if 'enum' in result and 'type' not in result and all(isinstance(item, str) for item in result['enum']):
            result['type'] = 'string'
        return result
    if isinstance(value, list): return [typed(item) for item in value]
    return value


def main():
    secret = yaml.safe_load(Path('C:/ProgramData/EvoMind/state/cliproxy.runtime.yaml').read_text(encoding='utf-8-sig'))['api-keys'][0]
    registry = build_default_registry()
    names = ['file_read','file_write','directory_hash','artifact_publish','artifact_list','report_generate','report_status']
    results = []
    for variant in ('typed_enum', 'original'):
        specs = [registry.get(name) for name in names]
        tools = [messaging.ToolSpec(spec.name, spec.description, typed(spec.input_schema) if variant=='typed_enum' else spec.input_schema) for spec in specs]
        transport = messaging.OpenAITransport(ProviderConfig('openai','http://127.0.0.1:65068/v1','gpt-6-astra',secret,reasoning_effort='low',service_tier='priority'))
        url, headers, body = transport.build([{'role':'user','content':'This is a transport check only. Return one file_read call with path inputs/data.json. No tool will be executed.'}], 'Validate native tool transport. Do not perform actions.', tools, 512, 0.3)
        for item in body['tools']: item['function']['strict']=False
        started = time.monotonic()
        row = {'variant':variant,'tool_count':len(tools),'tools_executed':0,'payload_bytes':len(json.dumps(body).encode())}
        try:
            result = messaging._post_json(url,headers,body,70)
            error = result.get('error') or {}
            message = str(error.get('message') or '') if isinstance(error,dict) else ''
            safe = re.sub(r'sk-[A-Za-z0-9_-]+|Bearer\s+\S+','[redacted]',message,flags=re.I)
            if re.search(r'password|api.?key|token|credential|secret|cookie|登录|账号|密码|口令|凭据',safe,re.I):safe='[sensitive error detail withheld]'
            row.update(http_json_received=True,error_type=error.get('type') if isinstance(error,dict) else None,
                error_message=safe[:1200], response_model=result.get('model'),choices=len(result.get('choices') or []))
        except Exception as error:
            row.update(error_class=type(error).__name__,http_status=getattr(error,'code',None))
        row['elapsed_seconds']=time.monotonic()-started
        results.append(row)
        print(json.dumps(row,ensure_ascii=True),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
