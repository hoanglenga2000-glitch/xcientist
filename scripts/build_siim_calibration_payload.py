"""Build an opt-in SIIM policy and narrow patches from the deployed runtime baseline."""
import hashlib,json,shutil,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/siim-mlebench-calibration-20260908'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True)
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('runtime_anchor_changed')
    return text.replace(old,new)

def main():
    payload=OUT/'runtime-payload-v2';payload.mkdir(exist_ok=False)
    data=json.loads((OUT/'frozen-data-receipt.json').read_text())['result']
    if data['status']!='frozen_data_ready':raise ValueError('data_not_frozen')
    manifest=data['manifest'];environment=json.loads((OUT/'environment.json').read_text())['inventory']
    runtime=next(r for r in environment['runtime_checks'] if r.get('exit_code')==0)
    weights=[{'path':'/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_model_cache/torch/hub/checkpoints/'+r['name'],**r} for r in environment['weights']]
    if not all(r['official_filename_prefix_match'] for r in weights):raise ValueError('weights_not_verified')
    protocol={'schema':'evomind.siim_comparison_protocol.v1','competition':'siim-isic-melanoma-classification',
        'paper':'MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering','venue':'ICLR 2025',
        'mlebench_commit':'507f92e1138bb6e40dac5c6ee7a6758e6424bf97',
        'aide_source':'https://github.com/thesofakillers/aideml','aide_tag':'v6.3.3','aide_commit':'d4a77cf3ca11e0f70749052b2701003e32bc245a',
        'data_manifest_sha256':data['manifest_sha256'],'evaluation_variant':manifest['evaluation_variant'],
        'train_rows':manifest['train_rows'],'test_rows':manifest['test_rows'],'removed_training_duplicates':manifest['excluded_training_rows'],
        'test_membership_unchanged':True,'test_patient_overlap':manifest['benchmark_train_test_patient_overlap'],
        'clinical_external_validation_claim':False,'metric':'roc_auc','prediction_kind':'positive_probability',
        'arms':['fixed_baseline','evomind','aide'],'run_seeds':[17,29,43],'cv_folds':5,'fold_seed':20260908,
        'folds_sha256':manifest['files']['frozen-folds.csv']['sha256'],'internal_patient_and_content_groups_disjoint':True,
        'model':'deepseek-v4-pro','endpoint':'https://api.pezayo.com/v1','model_fallback':False,
        'seconds_per_case':7200,'gpu_limit_hours_including_ev':24,'compute_threads':8,'parallel_gpu_runs':1,
        'selection':'maximum complete OOF AUC, then lower actual case elapsed time',
        'private_grading':'after all candidates freeze; no private-score feedback to agents',
        'official_submission':False,'raw_samples_sent_to_external_models':False,
        'baseline':{'architecture':'convnext_tiny','initial_weights':'ImageNet-1K','image_size':224,'epochs':3,
                    'optimizer':'AdamW','learning_rate':0.0003,'weight_decay':0.01,'batch_size':64,
                    'loss':'BCEWithLogitsLoss, pos_weight computed only from the fitting fold','tuning':False},
        'aide_adaptation':'native search policy with the common protected fit_model execution interface and fixed DeepSeek transport',
        'comparison_scope':'single-task resource-bounded comparison, not the full 75-task benchmark leaderboard'}
    write(payload/'protocol.json',protocol);protocol_sha=sha(payload/'protocol.json')
    copies={name:ROOT/'src/evomind_runtime'/name for name in ['siim_calibration_control.py','siim_calibration_budget.py','siim_calibration_runner.py','siim_worker_isolation.py']}
    copies['baseline.py']=ROOT/'scripts/siim_fixed_convnext_baseline.py'
    for name,path in copies.items():shutil.copyfile(path,payload/name)
    tooltext=(OUT/'runtime-base/tools.py').read_text(encoding='utf-8')
    tooltext=replace_once(tooltext,'            result = self._handlers[name](arguments, context)',
        '            if "siim_calibration" in context.metadata:\n                from .siim_calibration_control import guard_tool as guard_siim\n                guard_siim(name, arguments, context)\n            result = self._handlers[name](arguments, context)')
    tooltext=replace_once(tooltext,'def _training_route(args: dict[str, Any], context: ToolContext) -> ToolResult:\n',
        'def _training_route(args: dict[str, Any], context: ToolContext) -> ToolResult:\n    if "siim_calibration" in context.metadata:\n        from .siim_calibration_control import training_route as siim_route\n        return siim_route(args, context)\n')
    anchor='def _hpc_execute_solution(args: dict[str, Any], context: ToolContext, *, secret_files: dict[str, bytes] | None = None) -> ToolResult:\n'
    tooltext=replace_once(tooltext,anchor,anchor+'    if "siim_calibration" in context.metadata:\n        if secret_files:\n            return ToolResult("", False, {}, "SIIM model workers cannot receive secrets", error="calibration_secrets_forbidden")\n        from .siim_calibration_control import execute as siim_execute\n        return siim_execute(args, context, _hpc_execute_solution_impl)\n')
    (payload/'tools.py').write_text(tooltext,encoding='utf-8')
    rt=(OUT/'runtime-base/runtime.py').read_text(encoding='utf-8')
    rt=replace_once(rt,'            def guard_model_dispatch():\n','            def guard_model_dispatch():\n                if "siim_calibration" in (self.get_session(session_id).get("metadata") or {}):\n                    from .siim_calibration_control import before_model_attempt as siim_model_gate\n                    siim_model_gate(self, session_id)\n')
    rt=replace_once(rt,'max_tokens=8192 if "official_calibration" in (session.get("metadata") or {}) else 4096',
                   'max_tokens=8192 if ({"official_calibration", "siim_calibration"} & set(session.get("metadata") or {})) else 4096')
    (payload/'runtime.py').write_text(rt,encoding='utf-8')
    cases={}
    for arm in protocol['arms']:
        for seed in protocol['run_seeds']:
            run='run_'+uuid.uuid4().hex;case_id=arm+'-seed'+str(seed)
            task={'task':'Predict melanoma positive probability from images and optional non-diagnostic metadata.',
                'competition':protocol['competition'],'metric':'roc_auc','seed':seed,'seconds_per_case':7200,
                'interface':'Define fit_model(x_train,y_train,x_valid,y_valid,seed) returning an object with classes_=[0,1] and predict_proba(raw_dataframe). The dataframe contains image_path, sex, age_approx and anatom_site_general_challenge. The final full fit receives x_valid=y_valid=None. All learned preprocessing and fitting must use only the fitting fold.',
                'execution':'Write the module under work/solutions/, call training_route then hpc_verify, then hpc_execute_solution with data_dir="inputs"; omit competition. The protected scorer applies five patient/content-disjoint folds, final fit, and six independent reload processes.',
                'limits':{'gpu_count':1,'compute_threads':8,'max_output_tokens':8192},'pretrained_weights':weights,
                'prohibited':'No private answer access, raw-source CSV access, other experiments, leaderboards, historical results, memory, external network or submissions.',
                'outputs':'Select by returned complete OOF AUC; write and publish outputs/summary.md. Never invent a score.',
                'data_manifest_sha256':data['manifest_sha256'],'protocol_sha256':protocol_sha}
            profile={k:manifest[k] for k in ['train_rows','test_rows','target_counts','fold_sizes','excluded_columns','internal_patient_and_content_groups_disjoint']}
            profile.update(available_packages=['torch 2.5.1+cu118','torchvision 0.20.1+cu118','numpy','pandas','sklearn','catboost','PIL','scipy'],
                numerical_features=['age_approx'],categorical_features=['sex','anatom_site_general_challenge'],image_column='image_path',private_labels_exposed=False)
            case=payload/'cases'/case_id;write(case/'task.json',task);write(case/'data-profile.json',profile)
            cases[run]={'arm':arm,'seed':seed,'case_id':case_id,
                'workspace_root':'C:/ProgramData/EvoMind/data/acceptance/siim-calibration-20260908/'+case_id,
                'input_hashes':{'inputs/'+n:sha(case/n) for n in ['task.json','data-profile.json']}}
    policy={'schema':'evomind.siim_calibration.runtime_policy.v1','campaign_id':'siim_mlebench_calibration_20260908','enabled':True,
        'reference_identity_run':'run_7b210f24e219409b868edee14d4421d5','cases':cases,'protocol_sha256':protocol_sha,
        'data_manifest_sha256':data['manifest_sha256'],'persistent_data_root':manifest['root'],'train_rows':manifest['train_rows'],
        'budget_path':'C:/ProgramData/EvoMind/data/workspace/runtime/siim_calibration_budget.sqlite3',
        'ev_budget_path':'C:/ProgramData/EvoMind/data/workspace/runtime/ev_calibration_budget.sqlite3',
        'artifact_root':'/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/siim_calibration_results/siim_mlebench_calibration_20260908',
        'runner_sha256':sha(payload/'siim_calibration_runner.py'),'isolation_sha256':sha(payload/'siim_worker_isolation.py'),
        'baseline_source_sha256':sha(payload/'baseline.py'),'python_executable':runtime['python_path'],'pretrained_weights':weights,
        'available_imports':['numpy','pandas','sklearn','catboost','torch','torchvision','PIL','scipy']}
    write(payload/'policy-template.json',policy)
    files={p.relative_to(payload).as_posix():sha(p) for p in payload.rglob('*') if p.is_file()}
    seal={'schema':'evomind.siim_runtime_payload.v1','files':files,'baseline_tools_sha256':sha(OUT/'runtime-base/tools.py'),
          'baseline_runtime_sha256':sha(OUT/'runtime-base/runtime.py'),'baseline_seal_sha256':sha(OUT/'runtime-base/seal.json')}
    write(payload/'manifest.json',seal)
    print(json.dumps({'status':'payload_built_not_activated','cases':len(cases),'manifest_sha256':sha(payload/'manifest.json'),'protocol_sha256':protocol_sha}))

if __name__=='__main__':main()
