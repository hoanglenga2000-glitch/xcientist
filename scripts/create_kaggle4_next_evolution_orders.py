from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
ROOT=Path.cwd()
SUMMARY=ROOT/'workspace'/'kaggle4_self_evolution_rounds_20260624.json'
OUT_JSON=ROOT/'workspace'/'kaggle4_next_evolution_orders_20260624.json'
OUT_MD=ROOT/'reports'/'KAGGLE4_NEXT_EVOLUTION_ORDERS_20260624.md'
summary=json.loads(SUMMARY.read_text(encoding='utf-8'))
orders=[]
for task in summary['tasks']:
    task_id=task['task_id']
    latest=task['runs'][-1]
    best=next(r for r in task['runs'] if r['run_id']==task['best_run_id'])
    latest_promoted=latest.get('promoted') is True
    if task_id=='spaceship_titanic':
        branch='group_cabin_spend_exploitation' if latest_promoted else 'rollback_to_best_then_catboost_like_branch'
        objective='Push accuracy beyond current best with group_size, cabin deck/side, spend aggregate, missing indicators, and calibrated HGB-heavy blend.'
        mode='Stepwise'
        expected='accuracy +0.001~0.004 local CV if feature interactions generalize'
        rollback='hold if accuracy <= current best or if submission schema/prediction distribution becomes abnormal'
    elif task_id=='bike_sharing_demand':
        branch='time_aware_regression_exploitation' if latest_promoted else 'rollback_to_best_time_features'
        objective='Preserve strong RMSLE best and test time-aware split, cyclic hour/month/day features, and log target stacking.'
        mode='Stepwise'
        expected='RMSLE -0.005~-0.03 local proxy, but acceptance requires time-aware validation not just shuffled CV'
        rollback='hold if RMSLE >= current best or if time leakage risk check fails'
    elif task_id=='porto_seguro_safe_driver_prediction':
        branch='rollback_from_missing_indicator_negative_ablation_auc_gini'
        objective='Latest missing-pattern ablation degraded normalized_gini; keep previous best and branch to larger sample plus categorical-aware encoding / class imbalance weighting.'
        mode='Diff'
        expected='normalized_gini +0.002~0.01 vs preserved best after larger sample/HPC full run'
        rollback='hold if normalized_gini <= preserved best or if probability distribution collapses'
    else:
        branch='rollback_from_missing_indicator_negative_ablation_product_group_auc'
        objective='Latest missing-pattern ablation degraded ROC-AUC; keep previous best and branch to product_code grouped imputation, measurement interactions, and ablation audit.'
        mode='Diff'
        expected='roc_auc +0.001~0.006 vs preserved best if product-group imputation helps'
        rollback='hold if roc_auc <= preserved best or claim audit marks evidence weak'
    orders.append({
        'task_id':task_id,
        'current_best_run':task['best_run_id'],
        'current_best_score':task['best_score'],
        'metric':task['latest_metric'],
        'latest_run':latest['run_id'],
        'latest_decision':latest.get('decision'),
        'latest_promoted':latest_promoted,
        'selected_branch':branch,
        'code_generation_mode':mode,
        'objective':objective,
        'expected_metric_improvement':expected,
        'rollback_condition':rollback,
        'execution_policy':'workstation AgentOrchestrator only; no Codex direct training; official Kaggle submit blocked by Human Gate',
        'required_artifacts':['agent_trace.json','metrics.json','oof_predictions.csv','submission.csv','artifact_manifest.json','score_promotion_gate.json','claim_audit.json or report.md'],
    })
out={
    'schema':'academic_research_os.kaggle4_next_evolution_orders.v1',
    'created_at':datetime.now().isoformat(timespec='seconds'),
    'source_summary':SUMMARY.relative_to(ROOT).as_posix(),
    'controller':'MLEvolve-style Search Controller + XCIENTIST Harness',
    'policy':'Prioritize valid reproducible improvements; promote only through score_promotion_gate; preserve best-so-far on negative ablations.',
    'orders':orders,
}
OUT_JSON.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Kaggle 4 下一轮自进化实验指令','',f"- Created at: `{out['created_at']}`",f"- Source: `{out['source_summary']}`",f"- Policy: {out['policy']}",'','| task | best | latest decision | next branch | mode | expected |','|---|---:|---|---|---|---|']
for o in orders:
    lines.append(f"| `{o['task_id']}` | {o['current_best_score']} `{o['metric']}` | `{o['latest_decision']}` | `{o['selected_branch']}` | `{o['code_generation_mode']}` | {o['expected_metric_improvement']} |")
lines += ['','## 执行约束','','- 所有训练必须由工作站 `AgentOrchestrator` 发起。','- Codex 只允许修系统、监督、审计证据，不直接写针对单次提交的旁路训练结果。','- Kaggle 官方提交仍然 blocked，需要 Human Gate。','- negative ablation 必须保留在 retrospective memory，不能覆盖 best-so-far。']
OUT_MD.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'json':OUT_JSON.relative_to(ROOT).as_posix(),'md':OUT_MD.relative_to(ROOT).as_posix(),'orders':len(orders)},ensure_ascii=False,indent=2))
