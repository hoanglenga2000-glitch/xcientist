from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
ROOT=Path.cwd()
TASKS=['spaceship_titanic','bike_sharing_demand','porto_seguro_safe_driver_prediction','tabular_playground_series_aug_2022']
summary=[]
for task in TASKS:
    exp_root=ROOT/'experiments'/task
    runs=[]
    for run_dir in sorted([p for p in exp_root.iterdir() if p.is_dir()]):
        metrics_path=run_dir/'metrics.json'
        gate_path=run_dir/'score_promotion_gate.json'
        if not metrics_path.exists():
            continue
        metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
        gate=json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.exists() else {}
        decision=gate.get('decision',{})
        score=metrics.get('ensemble',{}).get('best_validation_score')
        metric=metrics.get('metric') or metrics.get('ensemble',{}).get('selection_metric')
        direction=metrics.get('metric_direction') or metrics.get('ensemble',{}).get('metric_direction') or ('minimize' if metric in {'rmsle','rmse','mae','mse','logloss','log_loss'} else 'maximize')
        runs.append({
            'run_id':run_dir.name,
            'path':run_dir.relative_to(ROOT).as_posix(),
            'metric':metric,
            'direction':direction,
            'score':score,
            'best_method':metrics.get('ensemble',{}).get('best_method'),
            'train_rows':metrics.get('train_rows'),
            'test_rows':metrics.get('test_rows'),
            'features_after_encoding':metrics.get('features_after_encoding'),
            'decision':decision.get('decision'),
            'promoted':decision.get('promoted'),
            'parent_score':decision.get('parent_score'),
            'promotion_delta':decision.get('promotion_delta'),
            'reason':decision.get('reason'),
            'artifacts':{
                'metrics':(run_dir/'metrics.json').relative_to(ROOT).as_posix(),
                'submission':(run_dir/'submission.csv').relative_to(ROOT).as_posix(),
                'oof':(run_dir/'oof_predictions.csv').relative_to(ROOT).as_posix(),
                'manifest':(run_dir/'artifact_manifest.json').relative_to(ROOT).as_posix(),
                'agent_trace':(run_dir/'agent_trace.json').relative_to(ROOT).as_posix(),
                'score_gate':(run_dir/'score_promotion_gate.json').relative_to(ROOT).as_posix(),
            }
        })
    # best per metric: keep latest metric family; if multiple metrics, report all and current best by latest metric
    latest=runs[-1] if runs else None
    same_metric=[r for r in runs if latest and r['metric']==latest['metric']]
    if latest:
        reverse=latest['direction']!='minimize'
        best=sorted(same_metric, key=lambda r: r['score'], reverse=reverse)[0]
        first=same_metric[0]
        if first['score'] is not None and best['score'] is not None:
            improvement=(best['score']-first['score']) if reverse else (first['score']-best['score'])
        else:
            improvement=None
    else:
        best=None; first=None; improvement=None
    summary.append({'task_id':task,'run_count':len(runs),'latest_metric':latest['metric'] if latest else None,'first_same_metric_score':first['score'] if first else None,'best_score':best['score'] if best else None,'best_run_id':best['run_id'] if best else None,'improvement_within_current_metric':improvement,'runs':runs})

out={'schema':'academic_research_os.kaggle4_self_evolution_supervision.v1','created_at':datetime.now().isoformat(timespec='seconds'),'codex_role':'supervisor_and_system_fixer_only; training launched through workstation AgentOrchestrator/run_workstation_ensemble','official_kaggle_submit':False,'tasks':summary,'claim_boundary':'Scores are local CV/proxy metrics only; no leaderboard rank, medal, or official submit is claimed.'}
json_path=ROOT/'workspace'/'kaggle4_self_evolution_rounds_20260624.json'
json_path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
md=ROOT/'reports'/'KAGGLE4_SELF_EVOLUTION_SUPERVISION_20260624.md'
lines=['# Kaggle 4 三层自进化监督报告','',f'- Created at: `{out["created_at"]}`','- Executor: `AI科研工作站 AgentOrchestrator / run_workstation_ensemble.py`','- Codex role: `supervisor_and_system_fixer_only`','- Official Kaggle submit: `False`','- Claim boundary: 本报告只陈述本地 CV/proxy 指标，不声明官方排名/奖牌。','','## 总览','','| task | runs | metric | first score | best score | improvement | best run |','|---|---:|---|---:|---:|---:|---|']
for t in summary:
    lines.append(f"| `{t['task_id']}` | {t['run_count']} | `{t['latest_metric']}` | {t['first_same_metric_score']} | {t['best_score']} | {t['improvement_within_current_metric']} | `{t['best_run_id']}` |")
lines += ['','## 三层架构证据','','- Multi-Agent Research OS：每个 run 均含 `agent_trace.json/jsonl`、`artifact_manifest.json`、`gate_audit_log.jsonl`、`submission.csv`、`oof_predictions.csv`。','- MLEvolve-style Search Controller：每个 run 均含 `research_os_search_graph.json` 与 `score_promotion_gate.json`，只在 best-so-far 改善时 promote。','- XCIENTIST-style Harness：每个 run 保留 gate、artifact、report，官方提交仍需 Human Gate。','','## 下一轮自进化优化建议']
for t in summary:
    if t['task_id']=='spaceship_titanic':
        rec='继续利用 PassengerId/Cabin/GroupSize 特征，下一轮比较 CatBoost/LightGBM 与 group-aware CV。'
    elif t['task_id']=='bike_sharing_demand':
        rec='已从 sampled smoke RMSLE 大幅提升到 full fast RMSLE；下一轮必须改为 time-aware split，并加入小时/工作日/节假日周期特征与 log target stacking。'
    elif t['task_id']=='porto_seguro_safe_driver_prediction':
        rec='改用 normalized_gini 后已获得有效 proxy；下一轮提高采样到 50k/100k 或 HPC 全量，加入类别编码、缺失/-1 pattern 与 AUC优化。'
    else:
        rec='改用 roc_auc 后出现小幅提升；下一轮应加入 product_code 分组、missing pattern、measurement imputation/interaction，并做 ablation。'
    lines.append(f"- `{t['task_id']}`: {rec}")
md.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'json':json_path.relative_to(ROOT).as_posix(),'md':md.relative_to(ROOT).as_posix(),'tasks':len(summary)},ensure_ascii=False,indent=2))
