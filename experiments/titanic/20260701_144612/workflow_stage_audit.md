# titanic 阶段化工作流审计

## 总结

- 实验目录：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612`
- 工作流版本：`v3_generic_tabular`
- 全部阶段通过：`True`
- 官方 Kaggle 提交：`False`
- 修改服务器私有模板：`False`

## 阶段检查

### task_understanding

- 角色：Orchestrator/Planner
- 状态：`passed`
- 证据：`tasks/titanic/overview.txt`, `D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\task_scaffold.json`
- 检查项：`{"task_name": "titanic", "target": "Survived", "metric": "accuracy", "server_templates_read_only": true}`

### preliminary_eda

- 角色：Analyst
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\data_quality.json`
- 检查项：`{"train_rows": 891, "test_rows": 418, "target_summary_recorded": true, "missing_values_recorded": true}`

### data_quality_check

- 角色：Reviewer/Gate
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\data_quality.json`
- 检查项：`{"train_test_feature_columns_match": true, "sample_submission_rows": 418, "sample_submission_columns": ["PassengerId", "Survived"]}`

### feature_engineering

- 角色：Developer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\task_scaffold.json`
- 检查项：`{"preset": "generic", "feature_plan_recorded": true, "target_transform": null}`

### model_validation

- 角色：Developer/Reviewer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\model_results.json`
- 检查项：`{"best_model": "random_forest", "best_metrics": {"cv_accuracy_mean": 0.8305, "cv_accuracy_std": 0.021618, "holdout_accuracy": 0.798883, "holdout_macro_f1": 0.777916, "seconds": 1.8782}, "thresholds": {"min_validation_accuracy": 0.78, "require_submission_schema_valid": true, "require_no_missing_predictions": true, "require_train_test_features_match": true}, "candidate_model_count": 4}`

### submission_generation

- 角色：Developer/Reviewer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\submission.csv`
- 检查项：`{"path": "D:\\桌面\\codex\\科研港科技\\experiments\\titanic\\20260701_144612\\submission.csv", "rows_match": true, "columns_match": true, "missing_predictions": 0, "prediction_columns": ["Survived"], "valid": true, "allowed_values": [0, 1], "allowed_values_only": true, "prediction_distribution": {"0": 277, "1": 141}}`

### report_and_review

- 角色：Evidence/Summarizer
- 状态：`passed`
- 证据：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\local_report.md`, `D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\local_report.docx`
- 检查项：`{"report_grounded_in_outputs": true, "post_scaffold_written": "D:\\桌面\\codex\\科研港科技\\experiments\\titanic\\20260701_144612\\post_scaffold_improvement.json", "local_gate_expected": true}`

## 下一步

- Configure Kaggle API token before official download/submission.
- Use this generic workflow as the baseline for additional teacher-provided tabular datasets.
- Keep GPU/server work disabled until a task has a clear compute benefit.