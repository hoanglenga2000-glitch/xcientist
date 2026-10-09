"""Bounded, source-logged model gateway checks. No competition data or tools executed."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ev-domestic-model-route-20260908'
BASE='https://api.pezayo.com'
sys.path.insert(0,str(ROOT/'src'))


class SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urlsplit(newurl).scheme!='https' or urlsplit(newurl).hostname!='api.pezayo.com':
            raise ValueError('Cross-host redirect denied')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def request(path,payload=None,tag='catalog'):
    started=time.monotonic()
    record={'at':datetime.now(timezone.utc).isoformat(),'tag':tag,'path':path,
            'requested_model':(payload or {}).get('model'),'http_status':None,'input_tokens':None,'output_tokens':None}
    body=None
    try:
        key=os.environ['EVOMIND_PROBE_MODEL_KEY']
        req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode() if payload is not None else None,
                                   headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
                                   method='POST' if payload is not None else 'GET')
        with urllib.request.build_opener(SameHostRedirect).open(req,timeout=60) as response:
            raw=response.read(4*1024*1024+1)
            if len(raw)>4*1024*1024:raise ValueError('Response size limit')
            body=json.loads(raw)
            record['http_status']=response.status
        record['ok']=isinstance(body,dict) and 'error' not in body
        usage=body.get('usage') or {}
        record.update(reported_model=body.get('model'),input_tokens=usage.get('prompt_tokens',usage.get('input_tokens')),
                      output_tokens=usage.get('completion_tokens',usage.get('output_tokens')),
                      finish_reason=(body.get('choices') or [{}])[0].get('finish_reason'),
                      response_sha256=hashlib.sha256(raw).hexdigest())
    except urllib.error.HTTPError as exc:
        record.update(ok=False,http_status=exc.code,error_type='HTTPError')
    except Exception as exc:
        record.update(ok=False,error_type=type(exc).__name__)
    record['elapsed_seconds']=round(time.monotonic()-started,3)
    with (OUT/'requests.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    return body,record


def safe_code_check(text):
    code=text.strip()
    if code.startswith('```'):
        code=re.sub(r'^```(?:python)?\s*','',code)
        code=re.sub(r'\s*```$','',code)
    if len(code)>6000:raise ValueError('code_size')
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='positive_probability':
        raise ValueError('function_shape')
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.While,ast.AsyncFunctionDef,ast.ClassDef,ast.With,ast.Global,ast.Nonlocal,ast.Lambda,ast.Pow)):
            raise ValueError('unsafe_ast')
        if isinstance(n,ast.Attribute) and n.attr not in {'index','count'}:raise ValueError('unsafe_attribute')
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id not in {'len','ValueError','all','any','enumerate','sum','list','isinstance','int','float','abs','set','sorted','type'}:
            raise ValueError('unsafe_call')
    ns={'__builtins__':{'len':len,'ValueError':ValueError,'all':all,'any':any,'enumerate':enumerate,'sum':sum,'list':list,'tuple':tuple,'bool':bool,'isinstance':isinstance,'int':int,'float':float,'abs':abs,'set':set,'sorted':sorted,'type':type}}
    exec(compile(tree,'<bounded_model_fixture>','exec'),ns)
    fn=ns['positive_probability']
    passed=0
    for classes,row,expected in [([0,1],[.2,.8],.8),([1,0],[.7,.3],.7),([0,1],[1.,0.],0.)]:
        if abs(fn(classes,row)-expected)>1e-12:raise ValueError('code_wrong_value')
        passed+=1
    for classes,row in [([0,1,2],[.2,.3,.5]),([2,3],[.4,.6]),([0,1],[float('nan'),.5]),([0,1],[-.1,1.1]),([0,1],[.4])]:
        try:fn(classes,row)
        except ValueError:passed+=1
        else:raise ValueError('code_missing_validation')
    return passed,code


def evaluate_model(model):
    from research_os.agent.messaging import OpenAITransport,ToolSpec,ToolResult
    from research_os.llm_client import ProviderConfig
    catalog=json.loads((OUT/'catalog.json').read_text(encoding='utf-8'))
    if model not in catalog['domestic_candidates']:raise ValueError('Model not in reviewed domestic catalog')
    transport=OpenAITransport(ProviderConfig(name='openai',base_url=BASE+'/v1',api_key=os.environ['EVOMIND_PROBE_MODEL_KEY'],model=model))
    tag=hashlib.sha256((model+str(time.time_ns())).encode()).hexdigest()[:12]
    results=[]
    def send(messages,system,tools,name):
        url,_,payload=transport.build(messages,system,tools,2048,.2)
        if url!=BASE+'/v1/chat/completions':raise ValueError('Unexpected transport endpoint')
        body,receipt=request('/v1/chat/completions',payload,tag=name)
        receipt['case']=name
        return body,receipt
    for i in range(2):
        body,r=send([{'role':'user','content':f'Return only JSON with marker "{tag}-{i}", sorted [3,5,7], and mean 5. No other keys or prose.'}],
                    'Follow the requested JSON format exactly.',[],f'json_{i+1}')
        try:
            parsed=json.loads(transport.parse(body).text)
            r['semantic_pass']=parsed=={'marker':f'{tag}-{i}','sorted':[3,5,7],'mean':5}
        except Exception:r['semantic_pass']=False
        results.append(r)
        if r.get('http_status') in {401,403}:
            denied={'model':model,'eligible':False,'reason':'credential_or_model_access_denied','checks':results,
                    'input_tokens':None,'output_tokens':None}
            (OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',model)+'-evaluation.json')).write_text(json.dumps(denied,indent=2),encoding='utf-8')
            return 2
    tools=[ToolSpec('inspect_fixture','Read the specified synthetic fixture; no side effects.',
                    {'type':'object','properties':{'run_tag':{'type':'string'}},'required':['run_tag'],'additionalProperties':False})]
    initial=[{'role':'user','content':f'First call inspect_fixture with run_tag "{tag}". After its result, output only JSON with sum and receipt_marker copied from the tool result.'}]
    body,r=send(initial,'Use the provided read-only tool before answering; never invent its result.',tools,'tool_call')
    try:
        turn=transport.parse(body)
        r['semantic_pass']=len(turn.tool_calls)==1 and turn.tool_calls[0].name=='inspect_fixture' and turn.tool_calls[0].input=={'run_tag':tag}
    except Exception:r['semantic_pass']=False
    results.append(r)
    if r['semantic_pass']:
        marker=hashlib.sha256((tag+'tool-result').encode()).hexdigest()[:16]
        result=ToolResult(turn.tool_calls[0].id,json.dumps({'sum':sum([3,5,8,13]),'receipt_marker':marker}))
        followup=initial+[{'role':'assistant','content':turn.raw_content},{'role':'user','content':[result.to_wire()]}]
        body,r=send(followup,'Use the provided read-only tool before answering; never invent its result.',tools,'tool_roundtrip')
        try:r['semantic_pass']=json.loads(transport.parse(body).text)=={'sum':29,'receipt_marker':marker}
        except Exception:r['semantic_pass']=False
        results.append(r)
    body,r=send([{'role':'user','content':'Write only Python function positive_probability(classes, row), no imports. Require exactly two classes containing 0 and 1 once each, exactly two numeric probabilities, each finite and within [0,1], summing to 1 within 1e-6. Raise ValueError for invalid input. Return the probability whose class label is 1, even if classes is [1,0].'}],
                'Produce a small correct self-contained Python function. Do not execute tools.',[],'code_contract')
    try:
        passed,code=safe_code_check(transport.parse(body).text)
        r.update(semantic_pass=passed==8,code_tests_passed=passed)
        (OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',model)+'-code.txt')).write_text(code,encoding='utf-8')
    except Exception as exc:r.update(semantic_pass=False,code_test_error=type(exc).__name__)
    results.append(r)
    summary={'model':model,'endpoint':BASE+'/v1','tested_at':datetime.now(timezone.utc).isoformat(),
             'native_evomind_transport_used':True,'checks':results,
             'passed':sum(bool(r.get('semantic_pass')) for r in results),'planned_checks':5,
             'eligible':len(results)==5 and all(r.get('semantic_pass') for r in results),
             'total_seconds':sum(r['elapsed_seconds'] for r in results),
             'input_tokens':sum(r.get('input_tokens') or 0 for r in results),
             'output_tokens':sum(r.get('output_tokens') or 0 for r in results),
             'price_verified':False,'scope':'bounded_synthetic_compatibility_tests_not_general_model_ranking'}
    path=OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',model)+'-evaluation.json')
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False),flush=True)
    return 0 if summary['eligible'] else 2


def retest_code(model):
    from research_os.agent.messaging import OpenAITransport
    from research_os.llm_client import ProviderConfig
    prior_path=OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',model)+'-evaluation.json')
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    if prior.get('passed')!=4:raise ValueError('Retest only candidates that passed the four integration checks')
    transport=OpenAITransport(ProviderConfig(name='openai',base_url=BASE+'/v1',api_key=os.environ['EVOMIND_PROBE_MODEL_KEY'],model=model))
    _,_,payload=transport.build([{'role':'user','content':'Write only Python function positive_probability(classes, row), no imports. Require exactly two classes containing 0 and 1 once each, exactly two numeric probabilities, each finite and within [0,1], summing to 1 within 1e-6. Raise ValueError for invalid input. Return the probability whose class label is 1, even if classes is [1,0].'}],
                                  'Produce a small correct self-contained Python function. Do not execute tools.',[],4096,.2)
    body,r=request('/v1/chat/completions',payload,tag='code_contract_retest_4096')
    try:
        text=transport.parse(body).text
        (OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',model)+'-code-retest.txt')).write_text(text,encoding='utf-8')
        passed,_=safe_code_check(text)
        r.update(semantic_pass=passed==8,code_tests_passed=passed)
    except Exception as exc:r.update(semantic_pass=False,code_test_error=type(exc).__name__,validation_detail=str(exc)[:120])
    result={'model':model,'eligible':bool(r['semantic_pass']),'integration_checks_previously_passed':4,
            'code_retest':r,'code_output_budget':4096,'original_results_preserved':True,
            'limitation':'Short compatibility sampling, not long-term uptime or overall model superiority.'}
    (OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',model)+'-code-retest.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return 0 if result['eligible'] else 2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['catalog','test','code-retest'],required=True)
    parser.add_argument('--model',default='');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='catalog':
        data,receipt=request('/v1/models')
        if not receipt['ok']:return 2
        ids=sorted({row['id'] for row in data.get('data',[]) if isinstance(row,dict) and isinstance(row.get('id'),str)})
        domestic=[m for m in ids if re.search(r'deepseek|qwen|kimi|glm|minimax|doubao|ernie|step-|hunyuan|baichuan',m,re.I)]
        catalog={'source':BASE+'/v1/models','checked_at':receipt['at'],'all_model_ids':ids,'domestic_candidates':domestic}
        (OUT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'total_models':len(ids),'domestic_candidates':domestic},ensure_ascii=False))
        return 0
    return retest_code(args.model) if args.action=='code-retest' else evaluate_model(args.model)


if __name__=='__main__':raise SystemExit(main())
