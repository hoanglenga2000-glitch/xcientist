# house_prices 阶段化工作流审计

## 总结

- 实验目录：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321`
- 工作流版本：`v3_generic_tabular`
- 全部阶段通过：`True`
- 官方 Kaggle 提交：`False`
- 修改服务器私有模板：`False`

## 阶段检查

### task_understanding

- 角色：Orchestrator/Planner
- 状态：`passed`
- 证据：`tasks/house_prices/overview.txt`, `D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\task_scaffold.json`
- 检查项：`{"task_name": "house_prices", "target": "SalePrice", "metric": "rmsle", "server_templates_read_only": true}`

### preliminary_eda

- 角色：Analyst
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\data_quality.json`
- 检查项：`{"train_rows": 1460, "test_rows": 1459, "target_summary_recorded": true, "missing_values_recorded": true}`

### data_quality_check

- 角色：Reviewer/Gate
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\data_quality.json`
- 检查项：`{"train_test_feature_columns_match": true, "sample_submission_rows": 1459, "sample_submission_columns": ["Id", "SalePrice"]}`

### feature_engineering

- 角色：Developer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\task_scaffold.json`
- 检查项：`{"preset": "house_prices_basic", "feature_plan_recorded": true, "target_transform": "log1p"}`

### model_validation

- 角色：Developer/Reviewer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\model_results.json`
- 检查项：`{"best_model": "gradient_boosting_log_target", "best_metrics": {"cv_rmsle_mean": 0.12899, "cv_rmsle_std": 0.020176, "holdout_rmsle": 0.129797, "holdout_mae": 15187.626782, "seconds": 50.5501}, "thresholds": {"max_cv_rmsle": 0.18, "max_holdout_rmsle": 0.2, "require_submission_schema_valid": true, "require_no_missing_predictions": true, "require_train_test_features_match": true, "require_positive_predictions": true, "expected_submission_rows": 1459, "expected_submission_columns": ["Id", "SalePrice"]}, "candidate_model_count": 4}`

### submission_generation

- 角色：Developer/Reviewer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\submission.csv`
- 检查项：`{"path": "D:\\桌面\\codex\\科研港科技\\experiments\\house_prices\\20260627_190321\\submission.csv", "rows_match": true, "columns_match": true, "missing_predictions": 0, "prediction_columns": ["SalePrice"], "valid": true, "positive_predictions": true, "prediction_min": 37013.758431, "prediction_max": 616964.81093, "prediction_mean": 178044.941512}`

### report_and_review

- 角色：Evidence/Summarizer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\local_report.md`, `D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\local_report.docx`
- 检查项：`{"report_grounded_in_outputs": true, "post_scaffold_written": "D:\\桌面\\codex\\科研港科技\\experiments\\house_prices\\20260627_190321\\post_scaffold_improvement.json", "local_gate_expected": true}`

## 下一步

- Configure Kaggle API token before official download/submission.
- Use this generic workflow as the baseline for additional teacher-provided tabular datasets.
- Keep GPU/server work disabled until a task has a clear compute benefit.