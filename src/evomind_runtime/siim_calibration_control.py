"""Registered SIIM comparison controls; prior studies and their budgets are preserved."""
from __future__ import annotations
import ast,base64,hashlib,json,math,os,re,sqlite3,time,uuid
from pathlib import Path

POLICY_PATH=Path('C:/ProgramData/EvoMind/config/official-calibration/siim-mlebench-20260908.json')
CAMPAIGN='siim_mlebench_calibration_20260908'
TOOLS={'file_read','file_write','file_patch','file_list','training_route','hpc_verify','hpc_execute_solution','artifact_publish'}
BOOT=uuid.uuid4().hex


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_for(context):
    from .training_control import read_json,identity
    marker=context.metadata.get('siim_calibration')
    if not isinstance(marker,dict):raise ValueError('calibration_marker_invalid')
    policy=read_json(POLICY_PATH)
    if policy.get('schema')!='evomind.siim_calibration.runtime_policy.v1' or policy.get('campaign_id')!=CAMPAIGN or policy.get('enabled') is not True:
        raise ValueError('calibration_policy_invalid')
    if marker.get('policy_sha256')!=digest(POLICY_PATH):raise ValueError('calibration_policy_changed')
    row=(policy.get('cases') or {}).get(context.session_id)
    if not isinstance(row,dict) or row.get('arm') not in {'fixed_baseline','evomind','aide'} or row.get('seed') not in {17,29,43}:
        raise ValueError('calibration_case_not_registered')
    if Path(row['workspace_root']).resolve()!=context.workspace_root.resolve():raise ValueError('calibration_workspace_mismatch')
    tenant,owner,_=identity(context.metadata)
    if (tenant,owner)!=(policy['tenant_id'],policy['owner_principal_id']):raise ValueError('calibration_owner_mismatch')
    binding=context.metadata.get('managed_hpc_identity') or {}
    if any(binding.get(k)!=policy['managed_hpc_identity'].get(k) for k in ['job_id','credential_profile','allocation_generation','profile_instance_id','allocation_binding_id']):
        raise ValueError('calibration_resource_binding_changed')
    if row['arm'] in {'evomind','aide'}:
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
    if context.metadata.get('siim_web_parent_run') and name not in {'file_read','file_list'}:
        from .siim_calibration_web import require_parent
        require_parent(context.store,context.metadata['siim_web_parent_run'],policy)
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
             'source':{'source_kind':'registered_calibration_data','competition':'siim-isic-melanoma-classification',
                       'manifest_sha256':policy['data_manifest_sha256'],'local_copy_created':False},
             'profile':{'row_count':policy['train_rows'],'modality':'images_and_metadata','row_count_complete':True},
             'reasons':[{'code':'user_authorized_managed_calibration'}],
             'required_steps':['training_route','hpc_verify_5_of_5','hpc_execute_solution','independent_reload'],
             'calibration_case':row['case_id']}
    context.store.append_event(context.session_id,'compute_route_selected',payload)
    if policy.get('resource_limits'):
        limits=policy['resource_limits']
        payload['resource_limits']={key:limits[key] for key in ['case_ceiling_seconds','step_ceiling_seconds','global_limit_seconds','allocation_rule']}
        ledger=CaseBudget(policy['budget_path'],policy['ev_budget_path'],limits)
        if ledger.recovery:
            payload['remaining_case_seconds']=ledger.remaining_case_seconds(context.session_id)
            payload['calibration_budget']=ledger.summary()
    return ToolResult('',True,payload,'Selected registered SIIM managed HPC route')


def validate_source(text,available=None):
    if len(text.encode())>128*1024:raise ValueError('candidate_source_too_large')
    tree=ast.parse(text)
    allowed={'numpy','pandas','sklearn','catboost','torch','torchvision','PIL','scipy','math','warnings','typing','random','time','collections','functools'}
    forbidden={'eval','exec','compile','__import__','globals','locals','vars','breakpoint','getattr','setattr'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(n.split('.')[0] not in allowed or any(p.startswith('_') for p in n.split('.')) for n in names):raise ValueError('candidate_import_not_allowed_at_line_'+str(node.lineno))
        if isinstance(node,ast.Name) and node.id in forbidden:raise ValueError('candidate_dynamic_access_denied')
        if isinstance(node,ast.Attribute) and node.attr.startswith('__') and node.attr not in {'__init__','__len__','__getitem__','__call__'}:raise ValueError('candidate_introspection_denied')
        if isinstance(node,ast.keyword) and node.arg in {'n_jobs','thread_count','nthread','num_threads','num_workers'}:
            if not isinstance(node.value,ast.Constant) or type(node.value.value) is not int or not 0<=node.value.value<=8:raise ValueError('candidate_thread_limit_invalid_at_line_'+str(node.lineno))
    if not any(isinstance(n,ast.FunctionDef) and n.name=='fit_model' for n in tree.body):raise ValueError('fit_model_interface_required')


from .siim_calibration_budget import SiimBudget as CaseBudget


def before_model_attempt(runtime,session_id):
    from types import SimpleNamespace
    session=runtime.get_session(session_id)
    context=SimpleNamespace(session_id=session_id,workspace_root=Path(session['workspace_root']),metadata=session['metadata'])
    policy,_=policy_for(context)
    if session['metadata'].get('siim_web_parent_run'):
        from .siim_calibration_web import require_parent
        require_parent(runtime.store,session['metadata']['siim_web_parent_run'],policy)
    CaseBudget(policy['budget_path'],policy['ev_budget_path'],policy.get('resource_limits')).require_model_window(session_id)


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
        runner=Path(__file__).with_name('siim_calibration_runner.py')
        if digest(runner)!=policy['runner_sha256']:raise ValueError('trusted_runner_changed')
        isolation=runner.with_name('siim_worker_isolation.py')
        if digest(isolation)!=policy['isolation_sha256']:raise ValueError('trusted_isolation_changed')
        ledger=CaseBudget(policy['budget_path'],policy['ev_budget_path'],policy.get('resource_limits'))
        attempt,seconds=ledger.reserve(context.session_id,code_sha,int(args.get('timeout_seconds',(policy.get('resource_limits') or {}).get('step_ceiling_seconds',7200))))
    except Exception as e:
        return ToolResult('',False,{},'SIIM calibration admission rejected',error=str(e) if isinstance(e,ValueError) else type(e).__name__)
    started=time.monotonic();result=None
    try:
        task={'arm':row['arm'],'seed':row['seed'],'candidate_sha256':code_sha,'protocol_sha256':policy['protocol_sha256'],
              'persistent_data_root':policy['persistent_data_root'],'data_manifest_sha256':policy['data_manifest_sha256'],
              'worker_limit_seconds':max(1,seconds-90),
              'artifact_root':policy['artifact_root']+'/'+row['case_id']+'/'+attempt,
              'python_executable':policy['python_executable'],'pretrained_weights':policy['pretrained_weights'],
              'isolation_sha256':policy['isolation_sha256']}
        task['resource_limits']=policy.get('resource_limits') or {}
        checkpoint=(policy.get('resume_checkpoints') or {}).get(context.session_id)
        if checkpoint:
            if checkpoint['identity']['candidate_sha256']!=code_sha:raise ValueError('resume_source_must_match_frozen_checkpoint')
            task['resume_checkpoint']=checkpoint
        wrapper=runner.read_text(encoding='utf-8')
        exit_anchor='        if fit.returncode:return fit.returncode'
        if wrapper.count(exit_anchor)!=1:raise ValueError('worker_exit_receipt_anchor_changed')
        wrapper=wrapper.replace(exit_anchor,
            "        (root/'fit-worker-exit.json').write_text(json.dumps({'return_code':fit.returncode,'phase':'fit_worker','elapsed_seconds':time.monotonic()-begin}))\n"
            "        if fit.returncode:\n"
            "            (out/'failure.json').write_text(json.dumps({'status':'failed','phase':'fit_worker','return_code':fit.returncode,'raw_outputs_withheld':True}))\n"
            "            return fit.returncode")
        loader_source=Path(__file__).with_name('siim_dataloader_runtime.py').read_text(encoding='utf-8')
        loader_marker='    torch.set_num_threads(1)'
        if wrapper.count(loader_marker)!=1:raise ValueError('loader_runtime_anchor_changed')
        wrapper=wrapper.replace(loader_marker,loader_marker+'\n    loader_trace=configure_serial_dataloader(artifact_root)')
        wrapper=wrapper.replace("def load_case(task,artifact_root):",loader_source+'\n\ndef load_case(task,artifact_root):')
        marker="if __name__=='__main__':raise SystemExit(main())"
        if wrapper.count(marker)!=1:raise ValueError('runner_entrypoint_changed')
        interpreter_guard="if __name__=='__main__' and str(Path(sys.prefix))!=str(Path(TASK['python_executable']).parent.parent):\n    os.execv(TASK['python_executable'],[TASK['python_executable'],'-I',__file__,*sys.argv[1:]])\n"
        wrapper=wrapper.replace(marker,'TASK='+repr(task)+'\nCANDIDATE='+repr(base64.b64encode(text.encode()).decode())+'\nISOLATION='+repr(base64.b64encode(isolation.read_bytes()).decode())+'\n'+interpreter_guard+marker)
        script=context.workspace_root/'work/.calibration-control'/('attempt-'+attempt+'.py')
        script.parent.mkdir(parents=True,exist_ok=True);script.write_text(wrapper,encoding='utf-8')
        context.execution_deadline=started+seconds
        context.remote_settlement_uncertain=False
        actual={**args,'solution_id':'siim-'+attempt,'script_path':str(script),'data_dir':str(context.workspace_root/'inputs'),'timeout_seconds':seconds}
        result=invoke_impl(actual,context)
        from .siim_candidate_contract import execution_receipt,candidate_kind
        executor=execution_receipt(result.content,time.monotonic()-started)
        receipt_path=context.workspace_root/'outputs/hpc'/actual['solution_id']/'executor-exit-receipt.json'
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        receipt_path.write_text(json.dumps(executor,indent=2),encoding='utf-8')
        if result.ok:
            path=context.workspace_root/'outputs/hpc'/actual['solution_id']/'metrics.json'
            metrics=json.loads(path.read_text())
            if metrics.get('candidate_sha256')!=code_sha or metrics.get('seed')!=row['seed'] or metrics.get('data_manifest_sha256')!=policy['data_manifest_sha256']:
                raise ValueError('candidate_metrics_identity_failed')
            metrics['candidate_kind']=candidate_kind(text)
            result.content={'status':'completed','metrics':metrics,'candidate_source':code_rel,'attempt_id':attempt,'execution_receipt':executor,
                            'large_artifacts_retained_on_hpc':True,'raw_samples_and_predictions_withheld':True}
            result.content['allocated_attempt_seconds']=seconds
            result.summary='SIIM diagnostic interface probe completed; not a trained model' if metrics['candidate_kind']=='diagnostic_constant_probe' else 'SIIM calibration candidate completed and independently reloaded'
        else:
            result.content={'status':'failed','attempt_id':attempt,'raw_outputs_withheld':True,'operator_evidence_preserved':True,'execution_receipt':executor}
            failure_path=context.workspace_root/'outputs/hpc'/actual['solution_id']/'failure.json'
            if failure_path.is_file() and failure_path.stat().st_size<8192 and seconds-(time.monotonic()-started)>210:
                from .ev_calibration_diagnostics import collect_failure
                try:
                    diagnostic=collect_failure(context,task,attempt)
                    result.content['diagnostic']=diagnostic
                    result.error='candidate_'+diagnostic['phase']+'_failed'
                except Exception:
                    result.content['diagnostic_collection']='unavailable'
            result.summary='SIIM candidate failed; no score is accepted'
        return result
    except Exception as e:
        result=ToolResult('',False,{'attempt_id':attempt},'SIIM calibration failed closed',error=type(e).__name__)
        return result
    finally:
        uncertain=getattr(context,'remote_settlement_uncertain',False)
        ledger.settle(attempt,time.monotonic()-started,bool(result and result.ok),uncertain)
        if result is not None:result.content['calibration_budget']=ledger.summary()
        context.execution_deadline=None
