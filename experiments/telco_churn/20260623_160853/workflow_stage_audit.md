# telco_churn 阶段化工作流审计

## 总结

- 实验目录：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853`
- 工作流版本：`v3_generic_tabular`
- 全部阶段通过：`True`
- 官方 Kaggle 提交：`False`
- 修改服务器私有模板：`False`

## 阶段检查

### task_understanding

- 角色：Orchestrator/Planner
- 状态：`passed`
- 证据：`tasks/telco_churn/overview.txt`, `D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\task_scaffold.json`
- 检查项：`{"task_name": "telco_churn", "target": "Churn", "metric": "accuracy", "server_templates_read_only": true}`

### preliminary_eda

- 角色：Analyst
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\data_quality.json`
- 检查项：`{"train_rows": 5634, "test_rows": 1409, "target_summary_recorded": true, "missing_values_recorded": true}`

### data_quality_check

- 角色：Reviewer/Gate
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\data_quality.json`
- 检查项：`{"train_test_feature_columns_match": true, "sample_submission_rows": 1409, "sample_submission_columns": ["customerID", "Churn"]}`

### feature_engineering

- 角色：Developer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\task_scaffold.json`
- 检查项：`{"preset": "telco_churn_basic", "feature_plan_recorded": true, "target_transform": null}`

### model_validation

- 角色：Developer/Reviewer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\model_results.json`
- 检查项：`{"best_model": "gradient_boosting", "best_metrics": {"cv_accuracy_mean": 0.807773, "cv_accuracy_std": 0.007051, "holdout_accuracy": 0.809228, "holdout_macro_f1": 0.739197, "seconds": 3.9265}, "thresholds": {"min_validation_accuracy": 0.78, "require_submission_schema_valid": true, "require_no_missing_predictions": true, "require_train_test_features_match": true, "expected_submission_rows": 1409, "expected_submission_columns": ["customerID", "Churn"], "allowed_prediction_values": ["No", "Yes"]}, "candidate_model_count": 4}`

### submission_generation

- 角色：Developer/Reviewer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\submission.csv`
- 检查项：`{"path": "D:\\桌面\\codex\\科研港科技\\experiments\\telco_churn\\20260623_160853\\submission.csv", "rows_match": true, "columns_match": true, "missing_predictions": 0, "prediction_columns": ["Churn"], "valid": true, "allowed_values": ["No", "Yes"], "allowed_values_only": true, "prediction_distribution": {"No": 1118, "Yes": 291}}`

### report_and_review

- 角色：Evidence/Summarizer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\local_report.md`, `D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\local_report.docx`
- 检查项：`{"report_grounded_in_outputs": true, "post_scaffold_written": "D:\\桌面\\codex\\科研港科技\\experiments\\telco_churn\\20260623_160853\\post_scaffold_improvement.json", "local_gate_expected": true}`

## 下一步

- Configure Kaggle API token before official download/submission.
- Use this generic workflow as the baseline for additional teacher-provided tabular datasets.
- Keep GPU/server work disabled until a task has a clear compute benefit.