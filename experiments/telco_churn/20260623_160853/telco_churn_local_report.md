# Telco Customer Churn 本地测试报告

## 任务理解

- 任务：Telco Customer Churn。
- 类型：binary_classification 表格数据任务。
- 指标：accuracy，本地使用交叉验证和 holdout 作为代理评估。
- 当前说明：本机暂未配置 Kaggle API 凭据，因此本轮使用公开镜像数据完成本地闭环测试；后续可替换为官方 Kaggle API 下载。

## 服务器模板对齐

- 本地流程参照服务器公开 Agent 模板字段与职责映射。
- 本轮没有读取或修改服务器私有 Agent 和工作流。

## 数据质量

- train 行列数：5634 x 21
- test 行列数：1409 x 20
- sample_submission 行数：1409
- 训练/测试特征列一致：True
- 目标摘要：{"distribution": {"No": 0.7346, "Yes": 0.2654}}
- 训练集缺失率 Top20：{"TotalCharges": 0.0014, "gender": 0.0, "SeniorCitizen": 0.0, "Partner": 0.0, "customerID": 0.0, "Dependents": 0.0, "tenure": 0.0, "MultipleLines": 0.0, "PhoneService": 0.0, "OnlineSecurity": 0.0, "OnlineBackup": 0.0, "DeviceProtection": 0.0, "InternetService": 0.0, "TechSupport": 0.0, "StreamingTV": 0.0, "Contract": 0.0, "StreamingMovies": 0.0, "PaperlessBilling": 0.0, "PaymentMethod": 0.0, "MonthlyCharges": 0.0}
- 测试集缺失率 Top20：{"TotalCharges": 0.0021, "customerID": 0.0, "SeniorCitizen": 0.0, "gender": 0.0, "Partner": 0.0, "Dependents": 0.0, "PhoneService": 0.0, "tenure": 0.0, "InternetService": 0.0, "OnlineSecurity": 0.0, "OnlineBackup": 0.0, "MultipleLines": 0.0, "DeviceProtection": 0.0, "TechSupport": 0.0, "StreamingMovies": 0.0, "StreamingTV": 0.0, "Contract": 0.0, "PaperlessBilling": 0.0, "PaymentMethod": 0.0, "MonthlyCharges": 0.0}

## 模型验证

- 最佳模型：`gradient_boosting`
- 指标：`{"cv_accuracy_mean": 0.807773, "cv_accuracy_std": 0.007051, "holdout_accuracy": 0.809228, "holdout_macro_f1": 0.739197, "seconds": 3.9265}`

## Submission 检查

- submission 文件：`D:\桌面\codex\科研港科技\experiments\telco_churn\20260623_160853\submission.csv`
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