# house_prices 任务脚手架

## 任务

- 比赛：House Prices - Advanced Regression Techniques
- 类型：regression
- 目标列：SalePrice
- 指标：rmsle

## 服务器模板映射

- orchestrator_planner: 多 Agent 调度主管 -> Orchestrator/Planner
- analyst: 高级数据分析 Agent -> Analyst
- developer: 代码开发 Agent -> Developer
- evidence_summarizer: 知识库 RAG Agent -> Evidence/Summarizer
- reviewer_gate: 流程自动化 Agent -> Reviewer/Gate

## 输入文件

- overview: `tasks/house_prices/overview.txt`
- train: `tasks/house_prices/data/train.csv`
- test: `tasks/house_prices/data/test.csv`
- sample_submission: `tasks/house_prices/data/sample_submission.csv`

## 验证方案

- 5-fold KFold cross-validation on log1p target plus holdout validation
- 时间预算：15 分钟

## 候选模型

- ridge_log_target
- random_forest_log_target
- extra_trees_log_target
- gradient_boosting_log_target

## 特征计划

- Train regression models on log1p(SalePrice) to match RMSLE behavior.
- Create TotalSF from basement and floor area fields.
- Create TotalBath from full/half bathroom fields.
- Create house age and remodel age from sale/build/remodel years.
- Create binary indicators for garage, basement, fireplace, and pool presence.
- Use median imputation for numeric features and most-frequent imputation plus one-hot encoding for categorical features.

## 风险点

- SalePrice is skewed, so the local model trains on log1p(SalePrice).
- Many categorical missing values use NA-like markers in the original Kaggle data.
- Train/test feature columns must stay aligned after dropping SalePrice.
- Submission must keep Id and SalePrice columns with positive predictions.
- Local validation is only a proxy before official Kaggle leaderboard submission.