from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
ROOT=Path.cwd()
progress_path=ROOT/'workspace'/'kaggle_10_self_evolution_progress_20260623.json'
summary_path=ROOT/'workspace'/'kaggle4_self_evolution_rounds_20260624.json'
if not progress_path.exists() or not summary_path.exists():
    raise SystemExit('missing progress or summary')
progress=json.loads(progress_path.read_text(encoding='utf-8'))
summary=json.loads(summary_path.read_text(encoding='utf-8'))
results=progress.setdefault('results', [])
by_task={r.get('task_id'): r for r in results}
for task in summary['tasks']:
    task_id=task['task_id']
    record={
        'task_id': task_id,
        'status': 'completed_improved_local_proxy',
        'source': 'workstation_agent_orchestrator_kaggle4_self_evolution_20260624',
        'baseline_score': task['first_same_metric_score'],
        'best_score': task['best_score'],
        'metric': task['latest_metric'],
        'direction': 'minimize' if task['latest_metric'] in {'rmsle','rmse','mae','mse','logloss','log_loss'} else 'maximize',
        'improvement': task['improvement_within_current_metric'],
        'best_branch': 'sklearn_rf_hgb_et_ensemble_adaptive_features',
        'official_submission_made': False,
        'workstation_api': True,
        'codex_direct_training': False,
        'best_run_id': task['best_run_id'],
        'evidence': [
            'workspace/kaggle4_self_evolution_rounds_20260624.json',
            'reports/KAGGLE4_SELF_EVOLUTION_SUPERVISION_20260624.md',
            'workspace/kaggle4_self_evolution_verification_20260624.json',
            'reports/KAGGLE4_SELF_EVOLUTION_VERIFICATION_20260624.md',
        ],
        'claim_boundary': 'local CV/proxy only; no official Kaggle rank/medal claim',
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    }
    if task_id in by_task:
        by_task[task_id].update(record)
    else:
        results.append(record)
progress['updated_at']=datetime.now().isoformat(timespec='seconds')
progress['kaggle4_self_evolution_report']='reports/KAGGLE4_SELF_EVOLUTION_SUPERVISION_20260624.md'
progress['kaggle4_self_evolution_verification']='reports/KAGGLE4_SELF_EVOLUTION_VERIFICATION_20260624.md'
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'updated':progress_path.relative_to(ROOT).as_posix(),'results_count':len(results),'kaggle4_tasks':[t['task_id'] for t in summary['tasks']]},ensure_ascii=False,indent=2))
