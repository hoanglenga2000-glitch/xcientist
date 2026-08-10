# House Prices - Advanced Regression Techniques 本地测试报告

## 任务理解

- 任务：House Prices - Advanced Regression Techniques。
- 类型：regression 表格数据任务。
- 指标：rmsle，本地使用交叉验证和 holdout 作为代理评估。
- 当前说明：本机暂未配置 Kaggle API 凭据，因此本轮使用公开镜像数据完成本地闭环测试；后续可替换为官方 Kaggle API 下载。

## 服务器模板对齐

- 本地流程参照服务器公开 Agent 模板字段与职责映射。
- 本轮没有读取或修改服务器私有 Agent 和工作流。

## 数据质量

- train 行列数：1460 x 81
- test 行列数：1459 x 80
- sample_submission 行数：1459
- 训练/测试特征列一致：True
- 目标摘要：{"summary": {"count": 1460.0, "mean": 180921.1959, "std": 79442.5029, "min": 34900.0, "25%": 129975.0, "50%": 163000.0, "75%": 214000.0, "max": 755000.0}, "skew": 1.882876}
- 训练集缺失率 Top20：{"PoolQC": 0.9952, "MiscFeature": 0.963, "Alley": 0.9377, "Fence": 0.8075, "MasVnrType": 0.5973, "FireplaceQu": 0.4726, "LotFrontage": 0.1774, "GarageQual": 0.0555, "GarageFinish": 0.0555, "GarageType": 0.0555, "GarageYrBlt": 0.0555, "GarageCond": 0.0555, "BsmtFinType2": 0.026, "BsmtExposure": 0.026, "BsmtCond": 0.0253, "BsmtQual": 0.0253, "BsmtFinType1": 0.0253, "MasVnrArea": 0.0055, "Electrical": 0.0007, "Condition2": 0.0}
- 测试集缺失率 Top20：{"PoolQC": 0.9979, "MiscFeature": 0.965, "Alley": 0.9267, "Fence": 0.8012, "MasVnrType": 0.6127, "FireplaceQu": 0.5003, "LotFrontage": 0.1556, "GarageYrBlt": 0.0535, "GarageCond": 0.0535, "GarageFinish": 0.0535, "GarageQual": 0.0535, "GarageType": 0.0521, "BsmtCond": 0.0308, "BsmtQual": 0.0302, "BsmtExposure": 0.0302, "BsmtFinType1": 0.0288, "BsmtFinType2": 0.0288, "MasVnrArea": 0.0103, "MSZoning": 0.0027, "BsmtHalfBath": 0.0014}

## 模型验证

- 最佳模型：`gradient_boosting_log_target`
- 指标：`{"cv_rmsle_mean": 0.12899, "cv_rmsle_std": 0.020176, "holdout_rmsle": 0.129797, "holdout_mae": 15187.626782, "seconds": 50.5501}`

## Submission 检查

- submission 文件：`D:\桌面\codex\科研港科技\experiments\house_prices\20260627_190321\submission.csv`
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