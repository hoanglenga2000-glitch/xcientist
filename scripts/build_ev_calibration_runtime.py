"""Prepare scoped runtime extension and six private task views from verified data."""
from pathlib import Path
import hashlib,json,uuid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ev-public-calibration-20260908'
STAGE=OUT/'runtime-extension'
PAYLOAD=STAGE/'payload-v2';PAYLOAD.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(STAGE/'tools-before.py').read_bytes();text=raw.decode()
changes={
 '        try:\n            result = self._handlers[name](arguments, context)':
 '        try:\n            if "official_calibration" in context.metadata:\n                from .ev_calibration_control import guard_tool\n                guard_tool(name, arguments, context)\n            result = self._handlers[name](arguments, context)',
 'def _training_route(args: dict[str, Any], context: ToolContext) -> ToolResult:\n':
 'def _training_route(args: dict[str, Any], context: ToolContext) -> ToolResult:\n    if "official_calibration" in context.metadata:\n        from .ev_calibration_control import training_route\n        return training_route(args, context)\n',
 'def _hpc_execute_solution(args: dict[str, Any], context: ToolContext, *, secret_files: dict[str, bytes] | None = None) -> ToolResult:\n':
 'def _hpc_execute_solution(args: dict[str, Any], context: ToolContext, *, secret_files: dict[str, bytes] | None = None) -> ToolResult:\n    if "official_calibration" in context.metadata:\n        if secret_files:\n            return ToolResult("", False, {}, "Calibration model runs cannot receive secrets", error="calibration_secrets_forbidden")\n        from .ev_calibration_control import execute\n        return execute(args, context, _hpc_execute_solution_impl)\n'}
applied=[]
for before,after in changes.items():
 options=[(before,after),(before.replace('\n','\r\n'),after.replace('\n','\r\n'))]
 matches=[(b,a) for b,a in options if text.count(b)==1]
 assert len(matches)==1,('hook_context_not_unique',before[:70])
 b,a=matches[0];text=text.replace(b,a,1);applied.append((b,a))
reverted=text
for before,after in reversed(applied):reverted=reverted.replace(after,before,1)
assert reverted.encode()==raw
runtime_before=(STAGE/'runtime-before.py').read_bytes()
runtime_text=runtime_before.decode()
needle='            def guard_model_dispatch():\n'
addition='            def guard_model_dispatch():\n                if "official_calibration" in (self.get_session(session_id).get("metadata") or {}):\n                    from .ev_calibration_control import before_model_attempt\n                    before_model_attempt(self, session_id)\n'
if runtime_text.count(needle)!=1:
 needle=needle.replace('\n','\r\n');addition=addition.replace('\n','\r\n')
assert runtime_text.count(needle)==1
runtime_text=runtime_text.replace(needle,addition,1)
assert runtime_text.replace(addition,needle,1).encode()==runtime_before
files={'tools.py':text.encode(),'runtime.py':runtime_text.encode(),**{n:(ROOT/'src/evomind_runtime'/n).read_bytes() for n in ['ev_calibration_control.py','ev_calibration_runner.py']}}
for name,body in files.items():
 with (PAYLOAD/name).open('xb') as f:f.write(body)
data=json.loads((OUT/'data-finalize-current.json').read_text())
assert data['status']=='completed' and data['secret_cleanup_confirmed']
received=data['managed_data_result']['receipt'];info=received['manifest'];data_sha=received['manifest_sha256']
assert info['status']=='FULL_DATA_READY' and info['train_rows']==668665 and info['test_rows']==286571
protocol_bytes=(OUT/'agent-input/protocol.json').read_bytes();protocol=json.loads(protocol_bytes)
cases={}
profile={k:info[k] for k in ['train_rows','test_rows','target_encoding','fold_count','split_seed','profile','target_counts','dependencies']}
profile['source_files']={n:{'sha256':r['sha256'],'bytes':r['bytes']} for n,r in info['files'].items()}
profile_bytes=(json.dumps(profile,sort_keys=True,indent=2)+'\n').encode()
for arm in ['fixed_baseline','evomind']:
 for seed in [17,29,43]:
  case_id=arm+'-seed'+str(seed)
  run_id='run_'+uuid.uuid5(uuid.NAMESPACE_URL,protocol['campaign_id']+'/'+case_id).hex
  workspace='C:/ProgramData/EvoMind/data/workspace/runtime/assistant_tasks/'+run_id
  task={'task':'Predict the positive probability of Will_Buy_EV on the official test set.',
        'competition':'playground-series-s6e9','metric':'roc_auc','positive_encoding':{'No':0,'Yes':1},'seed':seed,
        'folds':'The server applies the frozen five stratified folds with split_seed=20260908.',
        'source_contract':'Raw train/test/fold files are SHA-bound; training frames are passed by the protected runner.',
        'interface':'Write Python imports, classes and fit_model(x_train,y_train,x_valid,y_valid,seed). Return a fitted object with classes_ and predict_proba. For the final full fit, x_valid and y_valid are None. All preprocessing must be learned only from x_train.',
        'execution':'Write candidate source under work/solutions/. Call training_route, hpc_verify, then hpc_execute_solution with data_dir="inputs", script_path pointing to the candidate and timeout_seconds <=10800; omit competition. The server wraps the source; do not create a CLI or main block.',
        'software_available':['numpy','pandas','sklearn','catboost','scipy'],
        'resource_limits':{'wall_seconds_per_case':10800,'compute_threads':8,'visible_gpus':1},
        'restrictions':['No file/network/process/dynamic-import operations in candidate code.','No filesystem loading or writing; the runner loads data and saves models.','No other task, leaderboard, competition notebooks, historical results or memory.','If specifying thread_count or n_jobs, use an integer from 1 to 8.','No official submissions.'],
        'result_contract':'The server returns complete OOF AUC, five fold AUCs, independent reload checks and artifact hashes. Use only these to compare your candidates; write outputs/summary.md and publish it after selecting your best complete OOF candidate.',
        'protocol_sha256':sha(protocol_bytes),'data_manifest_sha256':data_sha}
  task_bytes=(json.dumps(task,sort_keys=True,indent=2)+'\n').encode()
  case_dir=PAYLOAD/'cases'/case_id;case_dir.mkdir(parents=True)
  (case_dir/'task.json').write_bytes(task_bytes);(case_dir/'data-profile.json').write_bytes(profile_bytes)
  cases[run_id]={'arm':arm,'seed':seed,'case_id':case_id,'workspace_root':workspace,
                 'input_hashes':{'inputs/task.json':sha(task_bytes),'inputs/data-profile.json':sha(profile_bytes)}}
baseline=(ROOT/'scripts/ev_fixed_catboost_baseline.py').read_bytes();(PAYLOAD/'baseline.py').write_bytes(baseline)
policy_template={'schema':'evomind.ev_calibration.runtime_policy.v1','campaign_id':protocol['campaign_id'],'enabled':True,
                 'protocol_sha256':sha(protocol_bytes),'cases':cases,'persistent_data_root':info['root'],'data_manifest_sha256':data_sha,
                 'artifact_root':'/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/ev_calibration_results/ev_public_calibration_20260908',
                 'budget_path':'C:/ProgramData/EvoMind/data/workspace/runtime/ev_calibration_budget.sqlite3',
                 'runner_sha256':sha(files['ev_calibration_runner.py']),'baseline_source_sha256':sha(baseline),
                 'available_imports':['numpy','pandas','sklearn','catboost','scipy'],'reference_identity_run':'run_7b210f24e219409b868edee14d4421d5'}
(PAYLOAD/'policy-template.json').write_text(json.dumps(policy_template,indent=2),encoding='utf-8')
manifest={'schema':'evomind.ev_calibration.runtime_extension.v1','baseline_tools_sha256':sha(raw),
          'baseline_runtime_sha256':sha(runtime_before),
          'baseline_seal_sha256':sha((STAGE/'seal-before.json').read_bytes()),
          'files':{p.relative_to(PAYLOAD).as_posix():sha(p.read_bytes()) for p in PAYLOAD.rglob('*') if p.is_file()},
          'production_targets':['tools.py','runtime.py','ev_calibration_control.py','ev_calibration_runner.py'],
          'old_study_policy_unchanged':True,'old_budget_ledger_unchanged':True}
(PAYLOAD/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'status':'runtime_extension_built','manifest_sha256':sha((PAYLOAD/'manifest.json').read_bytes()),'cases':len(cases),'data_manifest_sha256':data_sha}))
