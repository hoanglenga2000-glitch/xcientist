# 训练优化与任务完成率就绪审计

- 生成时间：2026-07-06T09:16:38
- 总体状态：passed
- 本地任务完成率：100.0% (3/3)

## 结论

本地训练任务完成率为 100%，且每个任务的最佳模型均由候选模型指标重新计算验证通过。

## 任务明细

### house_prices
- 最新实验：experiments/house_prices/20260627_190321
- 模型选择方向：minimize / cv_rmsle_mean
- 配置候选模型数：4
- 实际候选模型数：4
- 记录最佳模型：gradient_boosting_log_target
- 重新计算最佳模型：gradient_boosting_log_target
- 最佳模型选择校验：通过
- Validation Gate：通过
- 优化训练就绪：是
- 指标 cv_rmsle_mean <= 0.18：当前 0.12899，通过
- 指标 holdout_rmsle <= 0.2：当前 0.129797，通过
- 指标 submission_rows == 1459：当前 1459，通过
- 候选模型排序：gradient_boosting_log_target=0.12899; extra_trees_log_target=0.138377; random_forest_log_target=0.142487; ridge_log_target=0.144948

### titanic
- 最新实验：experiments/titanic/20260701_144612
- 模型选择方向：maximize / cv_accuracy_mean
- 配置候选模型数：4
- 实际候选模型数：4
- 记录最佳模型：random_forest
- 重新计算最佳模型：random_forest
- 最佳模型选择校验：通过
- Validation Gate：通过
- 优化训练就绪：是
- 指标 cv_accuracy_mean >= 0.78：当前 0.8305，通过
- 候选模型排序：random_forest=0.8305; gradient_boosting=0.821537; extra_trees=0.81032; logistic_regression=0.793516

### telco_churn
- 最新实验：experiments/telco_churn/20260623_160853
- 模型选择方向：maximize / cv_accuracy_mean
- 配置候选模型数：4
- 实际候选模型数：4
- 记录最佳模型：gradient_boosting
- 重新计算最佳模型：gradient_boosting
- 最佳模型选择校验：通过
- Validation Gate：通过
- 优化训练就绪：是
- 指标 cv_accuracy_mean >= 0.78：当前 0.807773，通过
- 指标 submission_rows == 1409：当前 1409，通过
- 候选模型排序：gradient_boosting=0.807773; logistic_regression=0.803869; random_forest=0.803513; extra_trees=0.800319

## 上线含义

1. 当前 3 个本地 Kaggle 风格任务均有可复测训练产物、Gate 和模型候选对比。
2. 该结论只说明本地优化训练 readiness 通过，不代表官方 Kaggle 排名、奖牌或 MLE-Bench 75 任务达标。
3. 后续大规模训练仍需通过 HPC/GPU 资源门禁、缓存门禁、claim audit 和人工提交 Gate。