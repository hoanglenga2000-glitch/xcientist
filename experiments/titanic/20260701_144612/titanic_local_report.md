# Titanic - Machine Learning from Disaster 本地测试报告

## 任务理解

- 任务：Titanic - Machine Learning from Disaster。
- 类型：binary_classification 表格数据任务。
- 指标：accuracy，本地使用交叉验证和 holdout 作为代理评估。
- 当前说明：本机暂未配置 Kaggle API 凭据，因此本轮使用公开镜像数据完成本地闭环测试；后续可替换为官方 Kaggle API 下载。

## 服务器模板对齐

- 本地流程参照服务器公开 Agent 模板字段与职责映射。
- 本轮没有读取或修改服务器私有 Agent 和工作流。

## 数据质量

- train 行列数：891 x 12
- test 行列数：418 x 11
- sample_submission 行数：418
- 训练/测试特征列一致：True
- 目标摘要：{"distribution": {"0": 0.6162, "1": 0.3838}}
- 训练集缺失率 Top20：{"Cabin": 0.771, "Age": 0.1987, "Embarked": 0.0022, "PassengerId": 0.0, "Name": 0.0, "Pclass": 0.0, "Survived": 0.0, "Sex": 0.0, "Parch": 0.0, "SibSp": 0.0, "Fare": 0.0, "Ticket": 0.0}
- 测试集缺失率 Top20：{"Cabin": 0.7823, "Age": 0.2057, "Fare": 0.0024, "Name": 0.0, "Pclass": 0.0, "PassengerId": 0.0, "Sex": 0.0, "Parch": 0.0, "SibSp": 0.0, "Ticket": 0.0, "Embarked": 0.0}

## 模型验证

- 最佳模型：`random_forest`
- 指标：`{"cv_accuracy_mean": 0.8305, "cv_accuracy_std": 0.021618, "holdout_accuracy": 0.798883, "holdout_macro_f1": 0.777916, "seconds": 1.8782}`

## Submission 检查

- submission 文件：`D:\桌面\codex\科研港科技\experiments\titanic\20260701_144612\submission.csv`
- 行数匹配：True
- 列名匹配：True
- 缺失预测数：0
- 是否通过：True

## 验收结论

- 本地验收：通过
- 判断依据：数据质量检查、模型验证分数、submission 格式检查、阶段审计和 post-scaffold 改进记录均已保存。

## 下一步

- 配置 Kaggle API token 后，替换为官方 Kaggle API 下载与可选真实提交。
- 在更多表格任务或老师真实数据案例上复用同一套通用 workflow。
- 闭环稳定后，再接入 GPU 服务器做深度模型或超参数搜索。