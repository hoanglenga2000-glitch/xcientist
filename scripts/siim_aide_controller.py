"""AIDE v6.3.3 search/prompt core adapted to the shared SIIM protected executor.

The upstream search policy, draft/improve/debug prompts, Node, Journal and
metric comparison are loaded verbatim. Provider transport, data preview,
execution interface and receipt-based scoring are adapter layers, not claims
of an unmodified reproduction of AIDE's full Kaggle interpreter.
"""
from __future__ import annotations
import ast,copy,hashlib,json,logging,random,re,sqlite3,subprocess,sys,time,types,uuid
from dataclasses import dataclass,field
from functools import total_ordering
from pathlib import Path
from typing import Any,Callable,Literal,Optional,cast
import numpy as np

BASE=Path('C:/ProgramData/EvoMind');STAGE=BASE/'staging/siim-mlebench-calibration-20260908/runtime-extension'
SOURCE_HASHES={
 'agent.py':'57d266d6cdb7b6258b032f704795eafeea0a6c11ae1b4ceb6d86e11c024aeb98',
 'journal.py':'b92c22e7b595971ebda7adbc35d0dd466dcb2436813d9067ee754df08dba1366',
 'metric.py':'13d4bec093a5f10035ccce09153e7e7c8fd6b038aef22fe2e6646f32a36a0612',
 'response.py':'d166ee59f885fa5becb17d269f9f9ab1a7296054fce22a053d726a785fdd0ae8',
 'backend-utils.py':'cd4397ae1d441dece99b8371321cc5582fe3ebaba5f9bc69fc105b039c76340c'}

def load_core(source_root,query):
    module=types.ModuleType('siim_aide_pinned_core');sys.modules[module.__name__]=module
    ns=module.__dict__
    ns.update(np=np,copy=copy,time=time,uuid=uuid,dataclass=dataclass,field=field,total_ordering=total_ordering,
              Literal=Literal,Optional=Optional,Any=Any,Callable=Callable,cast=cast,Config=types.SimpleNamespace,
              DataClassJsonMixin=object,ExecutionResult=types.SimpleNamespace,PromptType=Any,random=random,re=re,json=json,
              logging=logging,logger=logging.getLogger('siim.aide'),FunctionSpec=types.SimpleNamespace,query=query)
    for name in ['metric.py','response.py','backend-utils.py','journal.py','agent.py']:
        raw=(source_root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SOURCE_HASHES[name]:raise ValueError('aide_upstream_source_changed')
        tree=ast.parse(raw.decode())
        wanted=[]
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):continue
            if name=='response.py' and not(isinstance(node,ast.FunctionDef) and node.name in {'wrap_code','trim_long_string','is_valid_python_script','extract_code','extract_text_up_to_code'}):continue
            if name=='backend-utils.py' and not(isinstance(node,ast.FunctionDef) and node.name=='compile_prompt_to_md'):continue
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='logger' for t in node.targets):continue
            wanted.append(node)
        code=compile(ast.fix_missing_locations(ast.Module(body=wanted,type_ignores=[])),name,'exec',dont_inherit=True)
        exec(code,ns)
    ns['format_code']=lambda code:code
    return module

def _model_key():
    path='C:/Users/EvoMindSvc/AppData/Roaming/EvoMind/secrets/pezayo_ev_deepseek_v4_pro.xml'
    command="$ErrorActionPreference='Stop';$c=Import-Clixml -LiteralPath '"+path+"';[Console]::Out.Write($c.GetNetworkCredential().Password)"
    import base64
    encoded=base64.b64encode(command.encode('utf-16le')).decode()
    result=subprocess.run(['powershell','-NoProfile','-EncodedCommand',encoded],capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise ValueError('aide_model_credential_unavailable')
    return result.stdout.decode('utf-8-sig').strip()

def execute_aide_case(run_id,row,policy,request,*,resume_key='initial'):
    from evomind_runtime.siim_calibration_budget import SiimBudget
    from evomind_runtime.store import RuntimeStore
    from evomind_runtime.model_transport import governed_client,bind_run_client
    from research_os.llm_client import ProviderConfig
    from research_os.agent.messaging import AgentMessageClient,OpenAITransport
    ledger=SiimBudget(policy['budget_path'],policy['ev_budget_path'],policy.get('resource_limits'));store=RuntimeStore(BASE/'data/workspace/runtime/runtime.sqlite3')
    limits=policy.get('resource_limits') or {};case_ceiling=limits.get('case_ceiling_seconds',7200);step_ceiling=limits.get('step_ceiling_seconds',7200);max_steps=limits.get('max_agent_steps',24)
    workspace=Path(row['workspace_root'])
    task=json.loads((workspace/'inputs/task.json').read_text());profile=json.loads((workspace/'inputs/data-profile.json').read_text())
    key=_model_key()
    transport=OpenAITransport(ProviderConfig(name='openai',base_url='https://api.pezayo.com/v1',model='deepseek-v4-pro',api_key=key))
    def dispatch_gate():
        parent_id=(store.get_session(run_id).get('metadata') or {}).get('siim_web_parent_run')
        if parent_id:
            from evomind_runtime.siim_calibration_web import require_parent
            require_parent(store,parent_id,policy)
        ledger.require_model_window(run_id)
    client=governed_client(AgentMessageClient(max_retries=2,timeout=180,transports=[transport]),
            lambda event:store.append_event(run_id,'model.transport_attempt',event),before_attempt=dispatch_gate)
    key=None;bind_run_client(store,run_id,client)
    core=None
    def query(system_message=None,user_message=None,model=None,temperature=.3,**kwargs):
        parent_id=(store.get_session(run_id).get('metadata') or {}).get('siim_web_parent_run')
        if parent_id:
            from evomind_runtime.siim_calibration_web import require_parent
            require_parent(store,parent_id,policy)
        if model!='deepseek-v4-pro':raise ValueError('aide_model_switch_forbidden')
        text=core.compile_prompt_to_md(system_message) if system_message else ''
        turn=client.send([{'role':'user','content':str(user_message or 'Produce the requested plan and Python module.')}],system=text,tools=[],max_tokens=8192,temperature=temperature or .3)
        store.append_event(run_id,'aide.model_response',{'model':turn.model,'input_tokens':turn.input_tokens,'output_tokens':turn.output_tokens})
        if not turn.text:raise ValueError('aide_empty_model_response')
        return turn.text
    core=load_core(STAGE/'aide-core',query)
    with sqlite3.connect(policy['budget_path']) as c:case_started=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()[0]
    config=types.SimpleNamespace(workspace_dir=workspace,exec=types.SimpleNamespace(timeout=step_ceiling),agent=types.SimpleNamespace(
        steps=max_steps,time_limit=case_ceiling,k_fold_validation=5,expose_prediction=True,data_preview=True,convert_system_to_user=False,obfuscate=False,
        code=types.SimpleNamespace(model='deepseek-v4-pro',temp=.3),feedback=types.SimpleNamespace(model='deepseek-v4-pro',temp=.3),
        search=types.SimpleNamespace(num_drafts=5,max_debug_depth=20,debug_prob=1)))
    class ProtectedAide(core.Agent):
        @property
        def _prompt_impl_guideline(self):
            return {'Implementation guideline':[
                'Return a single Python module defining the exact fit_model interface in the task. No CLI or main block.',
                'The protected executor provides fitting/validation frames, computes complete OOF and saves predictions. Do not write CSV files yourself.',
                'The returned object must apply every preprocessing operation again inside predict_proba(raw_dataframe).',
                'Use only supplied images, metadata, weights and packages; no downloads, private labels, other experiments or results.',
                'All data/weights access is enforced by a filesystem sandbox. Training and prediction must finish within the remaining wall budget.',
                'Benchmark ceiling remaining seconds: '+str(max(0,int(case_ceiling-(time.time()-case_started))))+'. The common remaining-work allocation may be stricter; obey returned tool budgets.']}
        @property
        def _prompt_environment(self):return {'Verified environment':json.dumps(profile),'Cached pretrained weights':json.dumps(task['pretrained_weights'])}
        def update_data_preview(self):self.data_preview=json.dumps(profile)
    agent=ProtectedAide(task_desc=json.dumps(task),cfg=config,journal=core.Journal());agent.start_time=case_started;agent.update_data_preview()
    random.seed(row['seed']);audit=workspace/'outputs/aide-search-audit.json'
    if audit.exists():raise ValueError('aide_existing_search_requires_reconciliation')
    store.append_event(run_id,'aide.adapter_provenance',{'upstream_commit':'d4a77cf3ca11e0f70749052b2701003e32bc245a','source_sha256':SOURCE_HASHES,
        'native_search_and_prompt_core':True,'protected_execution_adapter':True,'original_full_interpreter_reproduction':False})
    try:
        for name,args in [('training_route',{'path':'inputs/task.json','task_description':'SIIM AIDE protected evaluation on managed HPC'}),('hpc_verify',{})]:
            response=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':name,'arguments':args,'idempotency_key':'aide-'+name+'-'+resume_key},timeout=180)
            if not(response.get('result') or {}).get('ok'):raise ValueError('aide_preflight_failed')
        for step in range(max_steps):
            ledger.require_model_window(run_id)
            parent=agent.search_policy()
            node=agent._draft() if parent is None else agent._debug(parent) if parent.is_buggy else agent._improve(parent)
            path='work/solutions/aide-'+node.id+'.py'
            write=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':'file_write','arguments':{'path':path,'content':node.code},'idempotency_key':'aide-write-'+node.id})
            if not(write.get('result') or {}).get('ok'):raise ValueError('aide_source_write_failed')
            response=request('POST','/v1/sessions/'+run_id+'/tools',{'tool_name':'hpc_execute_solution','arguments':{'solution_id':'aide-'+node.id,'script_path':path,'data_dir':'inputs','timeout_seconds':step_ceiling},'idempotency_key':'aide-execute-'+node.id},timeout=step_ceiling+300)
            result=response.get('result') or {};content=result.get('content') or {};metric=content.get('metrics') or {}
            from evomind_runtime.siim_candidate_contract import admitted_metrics
            ok=bool(result.get('ok') and admitted_metrics(metric,workspace))
            safe={'ok':ok,'metric':metric.get('oof_roc_auc') if ok else None,'diagnostic':content.get('diagnostic'),'error':result.get('error')}
            node.is_buggy=not ok;node.metric=core.MetricValue(safe['metric'],maximize=True) if ok else core.WorstMetricValue()
            node.analysis=json.dumps(safe);node._term_out=[node.analysis];node.exec_time=metric.get('runtime_seconds',0);node.exc_type=None if ok else 'ProtectedExecutionFailure'
            agent.journal.append(node);agent.current_step+=1
            payload={'run_id':run_id,'adapter':'AIDE v6.3.3 protected-executor adaptation','nodes':[{'id':n.id,'parent':n.parent.id if n.parent else None,'stage':n.stage_name,'source_sha256':hashlib.sha256(n.code.encode()).hexdigest(),'metric':n.metric.value,'is_buggy':n.is_buggy} for n in agent.journal.nodes]}
            temporary=audit.with_suffix('.new');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(audit)
            store.append_event(run_id,'aide.search_step',payload['nodes'][-1])
            if ledger.summary()['settlement_attention']:raise ValueError('aide_execution_settlement_required')
        return {'status':'steps_completed','nodes':len(agent.journal)}
    except ValueError as e:
        if str(e)=='model_time_budget_exhausted' and agent.journal.good_nodes:return {'status':'time_budget_reached','nodes':len(agent.journal)}
        raise
    finally:store.close()
