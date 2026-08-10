# telco_churn 任务脚手架

## 任务

- 比赛：Telco Customer Churn
- 类型：binary_classification
- 目标列：Churn
- 指标：accuracy

## 服务器模板映射

- orchestrator_planner: 多 Agent 调度主管 -> Orchestrator/Planner
- analyst: 高级数据分析 Agent -> Analyst
- developer: 代码开发 Agent -> Developer
- evidence_summarizer: 知识库 RAG Agent -> Evidence/Summarizer
- reviewer_gate: 流程自动化 Agent -> Reviewer/Gate

## 输入文件

- overview: `tasks/telco_churn/overview.txt`
- train: `tasks/telco_churn/data/train.csv`
- test: `tasks/telco_churn/data/test.csv`
- sample_submission: `tasks/telco_churn/data/sample_submission.csv`

## 验证方案

- 5-fold stratified cross-validation plus stratified holdout validation
- 时间预算：15 分钟

## 候选模型

- logistic_regression
- random_forest
- extra_trees
- gradient_boosting

## 特征计划

- Use configured drop columns.
- Use median imputation for numeric features.
- Use most-frequent imputation and one-hot encoding for categorical features.

## 风险点

- Churn is imbalanced, so validation must use stratified splits.
- customerID is an identifier and must not be used as a model feature.
- TotalCharges has blank-like values in the raw CSV and must be coerced to numeric.
- Submission must keep customerID and Churn columns with business labels No/Yes.
- Local validation is only a proxy before official Kaggle workflow automation.