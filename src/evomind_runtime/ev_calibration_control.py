"""Opt-in, administrator-registered EV calibration controls; legacy studies unchanged."""
from __future__ import annotations
import ast,base64,hashlib,json,math,os,re,sqlite3,time,uuid
from pathlib import Path

POLICY_PATH=Path('C:/ProgramData/EvoMind/config/official-calibration/ev-public-20260908.json')
CAMPAIGN='ev_public_calibration_20260908'
TOOLS={'file_read','file_write','file_patch','file_list','training_route','hpc_verify','hpc_execute_solution','artifact_publish'}
BOOT=uuid.uuid4().hex


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_for(context):
    from .training_control import read_json,identity
    marker=context.metadata.get('official_calibration')
    if not isinstance(marker,dict):raise ValueError('calibration_marker_invalid')
    policy=read_json(POLICY_PATH)
    if policy.get('schema')!='evomind.ev_calibration.runtime_policy.v1' or policy.get('campaign_id')!=CAMPAIGN or policy.get('enabled') is not True:
        raise ValueError('calibration_policy_invalid')
    if marker.get('policy_sha256')!=digest(POLICY_PATH):raise ValueError('calibration_policy_changed')
    row=(policy.get('cases') or {}).get(context.session_id)
    if not isinstance(row,dict) or row.get('arm') not in {'fixed_baseline','evomind'} or row.get('seed') not in {17,29,43}:
        raise ValueError('calibration_case_not_registered')
    if Path(row['workspace_root']).resolve()!=context.workspace_root.resolve():raise ValueError('calibration_workspace_mismatch')
    tenant,owner,_=identity(context.metadata)
    if (tenant,owner)!=(policy['tenant_id'],policy['owner_principal_id']):raise ValueError('calibration_owner_mismatch')
    binding=context.metadata.get('managed_hpc_identity') or {}
    if any(binding.get(k)!=policy['managed_hpc_identity'].get(k) for k in ['job_id','credential_profile','allocation_generation','profile_instance_id','allocation_binding_id']):
        raise ValueError('calibration_resource_binding_changed')
    if row['arm']=='evomind':
        contract=context.metadata.get('model_execution_contract') or {}
        if (contract.get('model')!='deepseek-v4-pro' or contract.get('wire_protocol')!='chat_completions_v1'
                or contract.get('endpoint_sha256')!=hashlib.sha256(b'https://api.pezayo.com/v1').hexdigest()):
            raise ValueError('calibration_model_binding_required')
    if set(context.metadata.get('run_allowed_tool_names') or [])!=TOOLS:raise ValueError('calibration_tool_scope_changed')
    return policy,row


def relative_path(value,context):
    p=Path(value)
    if not p.is_absolute():p=context.workspace_root/p
    if p.is_symlink():raise ValueError('calibration_symlink_rejected')
    return p.resolve().relative_to(context.workspace_root.resolve()).as_posix()


def guard_tool(name,args,context):
    policy,row=policy_for(context)
    if name not in TOOLS:raise ValueError('calibration_tool_not_allowed')
    if name in {'file_read','file_write','file_patch','file_list','artifact_publish'}:
        rel=relative_path(str(args.get('path','')),context)
        code=rel.startswith('work/solutions/') and rel.endswith('.py') and len(rel.split('/'))==3
        report=rel.startswith('outputs/') and rel.endswith('.md') and len(rel.split('/'))==2
        readonly=rel in {'inputs/task.json','inputs/data-profile.json'}
        if name=='file_list':
            if rel not in {'.','inputs','work/solutions','outputs'}:raise ValueError('calibration_listing_not_allowed')
        elif name=='file_read':
            if not(code or report or readonly):raise ValueError('calibration_raw_data_read_denied')
        elif name=='artifact_publish':
            if not report:raise ValueError('calibration_publish_only_summary')
        elif not(code or report):raise ValueError('calibration_input_write_denied')
        if readonly:
            p=context.workspace_root/rel
            if digest(p)!=row['input_hashes'][rel]:raise ValueError('calibration_agent_input_changed')
    return policy,row


def training_route(args,context):
    from .models import ToolResult
    policy,row=guard_tool('training_route',args,context)
    payload={'schema':'evomind.training.compute_route.v1','decision':'hpc_gpu','hpc_required':True,'local_gpu_allowed':False,
             'source':{'source_kind':'registered_calibration_data','competition':'playground-series-s6e9',
                       'manifest_sha256':policy['data_manifest_sha256'],'local_copy_created':False},
             'profile':{'row_count':668665,'feature_count':13,'row_count_complete':True},
             'reasons':[{'code':'user_authorized_managed_calibration'}],
             'required_steps':['training_route','hpc_verify_5_of_5','hpc_execute_solution','independent_reload'],
             'calibration_case':row['case_id']}
    context.store.append_event(context.session_id,'compute_route_selected',payload)
    return ToolResult('',True,payload,'Selected registered managed HPC calibration route')


def validate_source(text,available=None):
    if len(text.encode())>128*1024:raise ValueError('candidate_source_too_large')
    tree=ast.parse(text)
    allowed_imports={'numpy','pandas','sklearn','catboost','lightgbm','xgboost','scipy','math','warnings','typing'}
    if available is not None:allowed_imports&=set(available)|{'math','warnings','typing'}
    forbidden_names={'eval','exec','compile','open','__import__','getattr','setattr','globals','locals','vars','input','breakpoint'}
    forbidden_attrs={'open','io','read_csv','read_json','read_pickle','read_parquet','to_csv','to_json','to_pickle','to_parquet','read_text','read_bytes','write_text','write_bytes',
                     'load','save','loadtxt','savetxt','load_model','save_model','fromfile','tofile','memmap','ctypeslib','ctypes','system','popen','environ','getenv',
                     'parent','parents','resolve','absolute','expanduser','walk','glob','rglob','unlink','mkdir','rmdir'}
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            continue
        if not isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef,ast.Assign,ast.AnnAssign)):
            raise ValueError('candidate_top_level_execution_denied')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(m.split('.')[0] not in allowed_imports or any(s.startswith('_') or s in {'io','externals'} for s in m.split('.')) for m in modules):
                raise ValueError('candidate_import_not_allowed')
        if isinstance(node,ast.Name) and (node.id in forbidden_names or node.id.startswith('__')):raise ValueError('candidate_dynamic_access_denied')
        if isinstance(node,ast.Attribute) and (node.attr in forbidden_attrs or node.attr.startswith('__')):raise ValueError('candidate_io_or_dynamic_access_denied')
        if isinstance(node,ast.FunctionDef) and node.name.startswith('__') and node.name!='__init__':raise ValueError('candidate_magic_method_denied')
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and (node.value.startswith(('/','http:','https:')) or '..' in node.value.split('/')):
            raise ValueError('candidate_external_path_denied')
        if isinstance(node,ast.keyword) and node.arg in {'n_jobs','thread_count','nthread','num_threads'}:
            if not isinstance(node.value,ast.Constant) or type(node.value.value) is not int or not 1<=node.value.value<=8:
                raise ValueError('candidate_thread_limit_invalid')
    if not any(isinstance(n,ast.FunctionDef) and n.name=='fit_model' for n in tree.body):raise ValueError('fit_model_interface_required')


class CaseBudget:
    def __init__(self,path):
        self.path=str(path)
        with sqlite3.connect(self.path) as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('CREATE TABLE IF NOT EXISTS cases(run_id TEXT PRIMARY KEY,started REAL NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,run_id TEXT,source_sha TEXT,reserved REAL,charged REAL,status TEXT,boot TEXT)')

    def start_case(self,run_id):
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute('SELECT 1 FROM cases WHERE run_id=?',(run_id,)).fetchone():raise ValueError('case_already_started')
            c.execute('INSERT INTO cases VALUES(?,?)',(run_id,time.time()))

    def reserve(self,run_id,source_sha,requested):
        if type(requested) is not int or not 30<=requested<=10800:raise ValueError('case_timeout_invalid')
        with sqlite3.connect(self.path,timeout=10) as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()
            if row is None:raise ValueError('case_start_not_registered')
            elapsed=time.time()-row[0]
            if not 0<=elapsed<10800:raise ValueError('case_wall_budget_exhausted')
            rows=c.execute('SELECT reserved,charged,status,boot FROM attempts').fetchall()
            if any(r[2] in {'reserved','uncertain','exceeded'} for r in rows):raise ValueError('calibration_pending_settlement')
            remaining=86400-sum(r[1] for r in rows)
            seconds=min(requested,int(10800-elapsed),int(remaining))
            if seconds<120:raise ValueError('calibration_gpu_budget_exhausted')
            attempt=uuid.uuid4().hex
            c.execute("INSERT INTO attempts VALUES(?,?,?,?,0,'reserved',?)",(attempt,run_id,source_sha,seconds,BOOT))
        return attempt,seconds

    def require_model_window(self,run_id):
        with sqlite3.connect(self.path) as c:row=c.execute('SELECT started FROM cases WHERE run_id=?',(run_id,)).fetchone()
        if row is None or not 0<=time.time()-row[0]<10800-180:
            raise RuntimeError('calibration_model_time_budget_exhausted')

    def settle(self,attempt,actual,success,uncertain=False):
        if not math.isfinite(actual) or actual<0:raise ValueError('invalid_charge')
        with sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT reserved,status FROM attempts WHERE id=?',(attempt,)).fetchone()
            if row is None or row[1]!='reserved':raise ValueError('invalid_settlement')
            status='uncertain' if uncertain else 'exceeded' if actual>row[0] else 'completed' if success else 'failed'
            c.execute('UPDATE attempts SET charged=?,status=? WHERE id=?',(max(actual,row[0]) if uncertain else actual,status,attempt))

    def summary(self):
        with sqlite3.connect(self.path) as c:rows=c.execute('SELECT reserved,charged,status FROM attempts').fetchall()
        return {'gpu_limit_seconds':86400,'charged_slot_seconds':sum(r[1] for r in rows),
                'pending_reserved_seconds':sum(r[0] for r in rows if r[2]=='reserved'),
                'attempts':len(rows),'settlement_attention':any(r[2] in {'reserved','uncertain','exceeded'} for r in rows)}


def before_model_attempt(runtime,session_id):
    from types import SimpleNamespace
    session=runtime.get_session(session_id)
    context=SimpleNamespace(session_id=session_id,workspace_root=Path(session['workspace_root']),metadata=session['metadata'])
    policy,_=policy_for(context)
    CaseBudget(policy['budget_path']).require_model_window(session_id)


def execute(args,context,invoke_impl):
    from .models import ToolResult
    try:
        policy,row=guard_tool('hpc_execute_solution',args,context)
        if args.get('competition'):raise ValueError('calibration_uses_frozen_data_reference')
        code_rel=relative_path(args['script_path'],context)
        if not(code_rel.startswith('work/solutions/') and code_rel.endswith('.py') and len(code_rel.split('/'))==3):raise ValueError('candidate_path_invalid')
        if relative_path(args.get('data_dir',''),context)!='inputs':raise ValueError('calibration_data_reference_required')
        source=context.workspace_root/code_rel;text=source.read_bytes().decode('utf-8');validate_source(text,policy['available_imports'])
        code_sha=digest(source)
        if row['arm']=='fixed_baseline' and code_sha!=policy['baseline_source_sha256']:raise ValueError('fixed_baseline_changed')
        for rel,h in row['input_hashes'].items():
            if digest(context.workspace_root/rel)!=h:raise ValueError('calibration_input_changed')
        runner=Path(__file__).with_name('ev_calibration_runner.py')
        if digest(runner)!=policy['runner_sha256']:raise ValueError('trusted_runner_changed')
        ledger=CaseBudget(Path(policy['budget_path']))
        attempt,seconds=ledger.reserve(context.session_id,code_sha,int(args.get('timeout_seconds',10800)))
    except Exception as e:
        return ToolResult('',False,{},'EV calibration admission rejected',error=str(e) if isinstance(e,ValueError) else type(e).__name__)
    started=time.monotonic();result=None
    try:
        task={'arm':row['arm'],'seed':row['seed'],'candidate_sha256':code_sha,'protocol_sha256':policy['protocol_sha256'],
              'persistent_data_root':policy['persistent_data_root'],'data_manifest_sha256':policy['data_manifest_sha256'],
              'worker_limit_seconds':max(1,seconds-90),
              'artifact_root':policy['artifact_root']+'/'+row['case_id']+'/'+attempt}
        wrapper=runner.read_text(encoding='utf-8')
        marker="if __name__=='__main__':raise SystemExit(main())"
        if wrapper.count(marker)!=1:raise ValueError('runner_entrypoint_changed')
        wrapper=wrapper.replace(marker,'TASK='+repr(task)+'\nCANDIDATE='+repr(base64.b64encode(text.encode()).decode())+'\n'+marker)
        script=context.workspace_root/'work/.calibration-control'/('attempt-'+attempt+'.py')
        script.parent.mkdir(parents=True,exist_ok=True);script.write_text(wrapper,encoding='utf-8')
        context.execution_deadline=started+seconds
        context.remote_settlement_uncertain=False
        actual={**args,'solution_id':'ev-'+attempt,'script_path':str(script),'data_dir':str(context.workspace_root/'inputs'),'timeout_seconds':seconds}
        result=invoke_impl(actual,context)
        if result.ok:
            path=context.workspace_root/'outputs/hpc'/actual['solution_id']/'metrics.json'
            metrics=json.loads(path.read_text())
            if metrics.get('candidate_sha256')!=code_sha or metrics.get('seed')!=row['seed'] or metrics.get('data_manifest_sha256')!=policy['data_manifest_sha256']:
                raise ValueError('candidate_metrics_identity_failed')
            result.content={'status':'completed','metrics':metrics,'candidate_source':code_rel,'attempt_id':attempt,
                            'large_artifacts_retained_on_hpc':True,'raw_samples_and_predictions_withheld':True}
            result.summary='EV calibration candidate completed and independently reloaded'
        else:
            result.content={'status':'failed','attempt_id':attempt,'raw_outputs_withheld':True,'operator_evidence_preserved':True}
            failure_path=context.workspace_root/'outputs/hpc'/actual['solution_id']/'failure.json'
            if failure_path.is_file() and failure_path.stat().st_size<8192 and seconds-(time.monotonic()-started)>210:
                from .ev_calibration_diagnostics import collect_failure
                try:
                    diagnostic=collect_failure(context,task,attempt)
                    result.content['diagnostic']=diagnostic
                    result.error='candidate_'+diagnostic['phase']+'_failed'
                except Exception:
                    result.content['diagnostic_collection']='unavailable'
            result.summary='EV candidate failed; no score is accepted'
        return result
    except Exception as e:
        result=ToolResult('',False,{'attempt_id':attempt},'EV calibration failed closed',error=type(e).__name__)
        return result
    finally:
        uncertain=getattr(context,'remote_settlement_uncertain',False)
        ledger.settle(attempt,time.monotonic()-started,bool(result and result.ok),uncertain)
        if result is not None:result.content['calibration_budget']=ledger.summary()
        context.execution_deadline=None
