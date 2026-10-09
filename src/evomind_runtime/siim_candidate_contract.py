"""Separate controller completion, learned candidates, and constant interface probes."""
from __future__ import annotations
import ast
import base64
import hashlib
import json
from pathlib import Path
import re


def candidate_kind(source: str) -> str:
    tree = ast.parse(source)
    fit = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_model'), None)
    if fit is None:
        raise ValueError('fit_model_interface_required')
    # This conservative detector only identifies the observed no-learning
    # constant-probe shape. Low AUC is never grounds to reject a learned model.
    inputs = {arg.arg for arg in fit.args.args[:4]}
    if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in inputs for node in ast.walk(fit)):
        return 'model_candidate'
    returns = [node for node in fit.body if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        return 'model_candidate'
    constructor = returns[0].value
    if not isinstance(constructor.func, ast.Name) or constructor.args or constructor.keywords:
        return 'model_candidate'
    klass = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == constructor.func.id), None)
    predict = next((node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'predict_proba'), None) if klass else None
    if predict is not None:
        for node in ast.walk(predict):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'full':
                value = node.args[1] if len(node.args)>1 else next((kw.value for kw in node.keywords if kw.arg=='fill_value'), None)
                if isinstance(value, ast.Constant) and type(value.value) in {int, float}:
                    return 'diagnostic_constant_probe'
    return 'model_candidate'


def immutable_candidate_kind(workspace: Path, metrics: dict) -> str:
    attempt = str(metrics.get('hpc_artifact_root') or '').rsplit('/', 1)[-1]
    if not re.fullmatch('[a-f0-9]{32}', attempt):
        raise ValueError('candidate_attempt_identity_missing')
    root = (Path(workspace)/'work/.calibration-control').resolve()
    wrapper = root/('attempt-'+attempt+'.py')
    if wrapper.is_symlink() or wrapper.resolve().parent != root or wrapper.stat().st_size>1024*1024:
        raise ValueError('candidate_wrapper_invalid')
    tree = ast.parse(wrapper.read_text(encoding='utf-8'))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id=='CANDIDATE' for target in node.targets)]
    if len(assignments)!=1:
        raise ValueError('candidate_wrapper_binding_missing')
    source = base64.b64decode(ast.literal_eval(assignments[0].value), validate=True)
    if hashlib.sha256(source).hexdigest()!=metrics.get('candidate_sha256'):
        raise ValueError('candidate_source_hash_changed')
    return candidate_kind(source.decode('utf-8'))


def controller_completed(arm: str, result: dict) -> bool:
    states = {'fixed_baseline': {'completed'}, 'evomind': {'completed'},
              'aide': {'steps_completed', 'time_budget_reached'}}
    return result.get('status') in states.get(arm, set())


def admitted_metrics(metrics: dict, workspace: Path) -> bool:
    return (metrics.get('status')=='completed' and metrics.get('oof_complete') is True
            and metrics.get('independent_saved_oof_recomputed') is True
            and metrics.get('reload_processes')==6
            and immutable_candidate_kind(workspace, metrics) != 'diagnostic_constant_probe')


def reconcile_summaries(rows: list, cases: dict) -> tuple[list, list]:
    admitted, findings = [], []
    for row in rows:
        case = cases.get(row.get('run_id'))
        if not case or (row.get('arm'), row.get('seed'), row.get('case_id')) != (case['arm'], case['seed'], case['case_id']):
            raise ValueError('historical_case_identity_changed')
        if admitted_metrics(row, Path(case['workspace_root'])):
            admitted.append(row)
        else:
            findings.append({'case_id': row['case_id'], 'run_id': row['run_id'],
                             'candidate_sha256': row.get('candidate_sha256'),
                             'recorded_oof_auc_preserved': row.get('oof_roc_auc'),
                             'reason': 'not_an_admitted_trained_candidate'})
    return admitted, findings


def execution_receipt(payload: dict, elapsed: float) -> dict:
    code = payload.get('exit_code')
    failure = str(payload.get('failure_type') or '')
    return {'schema': 'evomind.siim_executor_exit.v1',
            'exit_code': code if type(code) is int else None,
            'failure_type': failure if re.fullmatch('[a-z0-9_]{0,80}', failure) else 'unclassified',
            'elapsed_seconds': elapsed,
            'stdout_bytes': len(str(payload.get('stdout_tail') or '').encode()),
            'stderr_bytes': len(str(payload.get('stderr_tail') or '').encode()),
            'stdout_tail_sha256': hashlib.sha256(str(payload.get('stdout_tail') or '').encode()).hexdigest(),
            'stderr_tail_sha256': hashlib.sha256(str(payload.get('stderr_tail') or '').encode()).hexdigest(),
            'raw_output_withheld': True}


def dispatch_paths(receipts: Path, parent: str, operation: str) -> tuple[Path, str]:
    if operation and not re.fullmatch('[A-Za-z0-9_-]{4,40}', operation):
        raise ValueError('siim_web_operation_invalid')
    stem = parent + ('--'+operation if operation else '')
    key = 'web-'+hashlib.sha256((parent+'\0'+operation).encode()).hexdigest()[:24] if operation else 'web-'+parent[4:20]
    return receipts/(stem+'.json'), key
