# titanic 任务脚手架

## 任务

- 比赛：Titanic - Machine Learning from Disaster
- 类型：binary_classification
- 目标列：Survived
- 指标：accuracy

## 服务器模板映射


## 输入文件

- overview: `tasks/titanic/overview.txt`
- train: `tasks/titanic/data/train.csv`
- test: `tasks/titanic/data/test.csv`
- sample_submission: `tasks/titanic/data/sample_submission.csv`

## 验证方案

- 5-fold stratified cross-validation plus stratified holdout validation
- 时间预算：10 分钟

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

- Age and Cabin have high missing rates.
- Train/test feature columns must stay aligned.
- Submission must keep PassengerId and Survived columns.
- Local validation is only a proxy before official Kaggle submission.