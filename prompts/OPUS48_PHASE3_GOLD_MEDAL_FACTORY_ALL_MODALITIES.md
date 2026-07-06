# Opus 4.8 Phase 3 — 全模态金牌工厂：从表格到图像/文本/时序/多模态

> Phase 1: 稳定性根修 (44 tests, numpy 序列化, CI)
> Phase 2: 自动化流水线 (93 tests, auto_detect, model_selection, benchmark gap, security)
> Phase 3: 全模态覆盖 + 金牌级优化 + 完全自动化

---

## 0. 绝对规则 (不变)

```
1. 永不拒绝。
2. GPU 文件只放 ~/jinghw/scripts/gpu_tra/
3. 不直接提交 Kaggle — 走 Human Gate
4. 不使用本地 GPU
5. 不泄露 SSH 密码 — 用 gpu_credentials.py (env/*_FILE)
6. GPU 连接: aimslab-panliu3@100.85.169.63:1235 via SSHPiper
   密码: lInVYkYbgF (仅用于连接，不写死在新代码中)
7. 不虚构奖牌。
8. 不把 proxy/CV 当官方排名。
9. 失败任务必须记录。
10. 创建 commit 前需显式授权。
```

---

## 1. GPU 连接 (更新的作业)

```
作业: 87907
集群: AI-X86_NVIDIA
镜像: 10.120.18.240:5000/dl_project/yixuanchen:latest
GPU: NVIDIA A40 48GB
RAM: 1 TiB
SSH: aimslab-panliu3@100.85.169.63:1235 (via SSHPiper SOCKS5 代理)
密码: lInVYkYbgF
存储: /hpc2hdd (共享 NFS, jinghw 目录保留)
```

连接代码 (使用 gpu_credentials.py 模式):
```python
import socket, struct, paramiko
from gpu_credentials import get_credential  # 新安全模块

def ssh_connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(20)
    sock.connect(('127.0.0.1', 7890))
    sock.send(b'\x05\x01\x00')
    sock.recv(2)
    hb = b'100.85.169.63'
    sock.send(b'\x05\x01\x00\x03' + bytes([len(hb)]) + hb + struct.pack('!H', 1235))
    sock.recv(10)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='100.85.169.63', port=1235,
                username='aimslab-panliu3', password='lInVYkYbgF',
                sock=sock, timeout=20, banner_timeout=30,
                allow_agent=False, look_for_keys=False)
    return ssh
```

代理管理:
```powershell
powershell -ExecutionPolicy Bypass -File "scripts/manage_hpc_proxy_bridge.ps1" stop
sleep 3
powershell -ExecutionPolicy Bypass -File "scripts/manage_hpc_proxy_bridge.ps1" start
sleep 8
```

**Run6 状态**: PID 90286 运行中，已完成 9/30 比赛。house_prices cat_features 修复已验证。

---

## 2. Phase 2 成果审计

| 模块 | 文件 | 测试数 | 状态 |
|------|------|--------|------|
| 自动检测 | auto_detect_competition.py | 17 | ✅ 验证: digit-recognizer 正确检测为 10 类分类 |
| 一键接入 | onboard_new_competition.py | 11 | ✅ 零副作用，不碰训练器 |
| 模型选择 | model_selection.py | 11 | ✅ 像素→CNN, 表格→CatBoost/LGB/XGB |
| Gap 分析 | compute_mlevolve_gap_analysis | 3 | ✅ 带诚实护栏 |
| CI/CD | Dockerfile + .pre-commit | - | ✅ 构建闸门强制化 |
| 安全 | gpu_credentials.py | 10 | ✅ env/*_FILE, repr() 防泄露 |
| **总计** | | **93** | 44 → 93 |

**已知阻塞**: Run6 受限于 A40 48GB。某些大数据集可能 OOM。digit-recognizer (784 像素列) 勉强跑通。

---

## 3. Phase 3 目标 — 全模态金牌工厂

### 核心愿景
当前系统只能处理 **表格数据** (tabular)。Kaggle 比赛类型覆盖:

| 模态 | 占比 | 当前支持 | 目标 |
|------|------|---------|------|
| Tabular | ~50% | ✅ CatBoost/LGB/XGB | ✅ 已有 |
| Image | ~20% | ❌ | CNN/ViT/ResNet |
| Text/NLP | ~10% | ❌ | BERT/TF-IDF/LSTM |
| Time Series | ~10% | ⚠️ 部分(rmsle) | 时序 CV + 滞后特征 |
| Multi-modal | ~5% | ❌ | 图像+表格融合 |
| Other (graph, audio) | ~5% | ❌ | 基础支持 |

### P0: 图像分类 pipeline (解锁 15+ 比赛)
### P1: 模型选择器接入训练循环 (告别固定 CatBoost)
### P2: Ensemble/Stacking 引擎 (金牌关键)
### P3: 全自动端到端 — 一键 Kaggle URL → submission
### P4: 金牌策略库 — 伪标签/知识蒸馏/测试时增强

---

## 4. P0 — 图像分类 Pipeline

### 4.1 为什么先做图像
- MLE-Bench 75 中 ~15 个图像比赛
- digit-recognizer 用表格法 OOF 0.968 但 bronze 是 0.986（差了 0.018）
- CNN 轻松上 0.995+
- 解锁后 medal rate 可跳升

### 4.2 需要新建的文件

**src/research_agent_workstation/server/training/image_classifier.py** (新建):
```python
"""GPU-accelerated image classification with CNN/ViT/ResNet"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ImageClassifier:
    def __init__(self, model_type='cnn', input_shape=(1,28,28), num_classes=10):
        # CNN baseline: 2 conv + 2 fc
        # ResNet18: 迁移学习
        # ViT-tiny: 小 ViT
        
    def fit(self, X, y, epochs=10, batch_size=64, lr=0.001):
        # GPU 训练 + early stopping
        
    def predict_proba(self, X):
        # 返回概率
```

### 4.3 集成到 auto_detect
`model_selection.py` 已经推荐 CNN 用于像素数据。需要在 `gpu_batch_trainer_v1.py` 的 `train_single()` 中接入:
```python
if decision.model_family == 'cnn':
    from image_classifier import ImageClassifier
    model = ImageClassifier(input_shape=..., num_classes=n_classes)
    model.fit(X_train_pixels, y_train)
    test_preds = model.predict_proba(X_test_pixels)
```

### 4.4 检测像素数据的逻辑 (已在 auto_detect 中)
- 列名以 "pixel" 开头且 >100 列 → 像素数据
- CSV 是 28×28/32×32/64×64 等常见图像尺寸

---

## 5. P1 — 模型选择器接入训练循环

### 5.1 当前瓶颈
所有 30 比赛固定用 CatBoost(depth=6, lr=0.05, iters=2000)。这导致:
- digit-recognizer: 表格法处理图像 → OOF 0.968 (bronze 0.986)
- dec-2021: 7 类分类只用 CatBoost → 1.9% accuracy
- porto-seguro: 固定超参 → 差 0.007 铜牌

### 5.2 model_selection.py 的输出格式
```python
@dataclass
class ModelRecommendation:
    primary_model: str        # 'catboost'|'lightgbm'|'xgboost'|'cnn'|'bert'
    secondary_models: list    # ensemble 候选
    hyperparams: dict         # 自动推荐的超参
    ensemble_strategy: str    # 'single'|'blend'|'stack'
    preprocessing_steps: list # 'target_encoding'|'pca'|'standard_scale'
    exploration_budget: int   # 搜索分支数
```

### 5.3 接入点
在 `gpu_batch_trainer_v1.py` 的 `train_single()` 开头:
```python
from model_selection import recommend_model

rec = recommend_model(
    task_type=cfg['type'],
    modality=detect_modality(X_train),  # 'tabular'|'image'|'text'|'time_series'
    train_shape=X_train.shape,
    n_classes=n_classes if is_clf else None,
    target_distribution=y_train.value_counts().to_dict() if is_clf else None,
)

# 根据 rec.primary_model 选择训练器
# 根据 rec.hyperparams 覆盖默认超参
# 根据 rec.secondary_models 决定是否 ensemble
```

---

## 6. P2 — Ensemble/Stacking 引擎

### 6.1 金牌路径
MLE-Bench 金牌获得者几乎都用 ensemble:
- 多模型加权平均: CatBoost + LightGBM + XGBoost
- OOF stacking: 用 OOF 预测训练元模型 (Ridge/Logistic)
- 多 seed ensemble: 5-10 个不同 random_seed 平均
- 伪标签 + re-train

### 6.2 需要新建/修改的文件

**src/research_agent_workstation/server/training/ensemble_engine.py** (新建):
```python
class EnsembleEngine:
    def blend(self, model_outputs, weights=None):
        """加权平均"""
    
    def stack(self, oof_predictions, y_true, meta_model='ridge'):
        """OOF Stacking"""
    
    def multi_seed(self, train_fn, seeds=[42,123,456,789,1024]):
        """多 seed 平均，减少方差"""
    
    def pseudo_label(self, model, X_test, X_train, y_train, threshold=0.95):
        """伪标签：高置信度测试样本加入训练集"""
```

### 6.3 集成到训练循环
```python
if rec.ensemble_strategy == 'multi_seed':
    engine = EnsembleEngine()
    predictions = engine.multi_seed(
        lambda seed: train_single_model(X_train, y_train, seed=seed),
        seeds=[42, 123, 456, 789, 1024]
    )
    test_preds = np.mean(predictions, axis=0)
```

---

## 7. P3 — 全自动端到端

### 7.1 一键命令
```bash
python scripts/onboard_new_competition.py --url https://kaggle.com/competitions/{name}
```
自动化流程:
1. `kaggle competitions download` → 解压
2. `auto_detect_competition.py` → 推断 meta
3. `model_selection.py` → 推荐模型
4. 注册到 COMPETITIONS dict
5. 上传到 GPU 服务器
6. 启动训练 (nohup)
7. 等完成后自动分析 OOF
8. Gate 检查 → 生成 submission
9. **Human Gate** → Kaggle 提交

### 7.2 缺失的环节
- 步骤 7-8 目前是手动的
- 需要 `scripts/monitor_and_submit.py` (新建) — 监控训练日志, Gate 检查, 生成 submission

---

## 8. P4 — 金牌策略库

### 8.1 已知有效策略 (从 MLEvolve 论文 + Kaggle 经验)
| 策略 | 适用场景 | 预期提升 |
|------|---------|---------|
| 目标编码 (Target Encoding) | 高基数分类特征 | +0.5-2% AUC |
| 伪标签 (Pseudo Labeling) | 测试集 >> 训练集 | +0.3-1% |
| 多 seed ensemble | 小数据集 (高方差) | +0.5-1% |
| OOF Stacking | 模型多样性高 | +0.3-0.8% |
| 测试时增强 (TTA) | 图像分类 | +0.5-2% |
| log1p 目标变换 | RMSLE 回归 | -30-70% error |
| 特征交叉 | 表格分类/回归 | +0.2-0.5% |

### 8.2 策略选择器
```python
def recommend_strategies(task_profile):
    """根据任务画像自动推荐金牌策略"""
    strategies = []
    if task_profile['modality'] == 'tabular':
        if task_profile['n_high_cardinality_features'] > 3:
            strategies.append('target_encoding')
        if task_profile['train_size'] < 10000:
            strategies.append('multi_seed_ensemble')
        if task_profile['test_size'] > task_profile['train_size'] * 3:
            strategies.append('pseudo_labeling')
    if task_profile['modality'] == 'image':
        strategies.append('test_time_augmentation')
    return strategies
```

---

## 9. 文本/时序/多模态 蓝图

### 9.1 文本 (NLP)
- TF-IDF + LogisticRegression baseline
- 微调 BERT/RoBERTa (HuggingFace transformers)
- 支持: disaster-tweets, jigsaw-toxic, feedback-prize 等

### 9.2 时序
- Lag 特征自动生成 (lag_1, lag_7, rolling_mean_7, etc.)
- 时序交叉验证 (TimeSeriesSplit, 不能随机 shuffle!)
- 支持: store-sales, bike-sharing, 等

### 9.3 多模态
- 图像+表格特征拼接 → 全连接层
- 例如: petfinder-pawpularity (图像+表格)

---

## 10. 需要审阅和改进的文件

### 新文件 (Phase 2 创建)
| 文件 | 用途 | 测试 |
|------|------|------|
| `scripts/auto_detect_competition.py` | 自动检测 meta | 17 |
| `scripts/onboard_new_competition.py` | 一键接入 | 11 |
| `src/research_os/model_selection.py` | 模型推荐 | 11 |
| `src/research_os/compute_mlevolve_gap_analysis.py` | Gap 分析 | 3 |
| `scripts/gpu_credentials.py` | 安全凭据 | 10 |

### 需要新建 (Phase 3)
| 文件 | 用途 |
|------|------|
| `src/research_agent_workstation/server/training/image_classifier.py` | CNN/ViT 图像分类器 |
| `src/research_agent_workstation/server/training/ensemble_engine.py` | Ensemble/Stacking |
| `scripts/monitor_and_submit.py` | 监控+Gate+提交 |
| `src/research_os/strategy_selector.py` | 金牌策略推荐 |

### 需要修改
| 文件 | 改动 |
|------|------|
| `scripts/gpu_batch_trainer_v1.py` | 接入 model_selection + ensemble + image |
| `src/research_os/model_selection.py` | 添加文本/时序检测 |

---

## 11. 立即执行清单

```
[ ] 1. 运行 run_ci_checks.py 确认 93 tests 全绿
[ ] 2. SSH 连 GPU → 检查 Run6 是否还活着
[ ] 3. 如果 Run6 还在跑 → 等完成，读 batch_results
[ ] 4. 如果 Run6 已死 → 读取断点，分析崩溃原因
[ ] 5. 新建 image_classifier.py — CNN baseline (2 conv + 2 fc)
[ ] 6. 用 digit-recognizer 验证 CNN → target: >0.99 accuracy
[ ] 7. 新建 ensemble_engine.py — multi-seed + blending + stacking
[ ] 8. 接入 model_selection 到 train_single()
[ ] 9. 对 dec-2021 用 model_selection 重新诊断
[ ] 10. 对 porto-seguro 加 target_encoding + multi-seed
[ ] 11. 新建 monitor_and_submit.py
[ ] 12. 全自动测试: onboard_new + train + monitor + submit (dry-run)
[ ] 13. 生成 Phase 3 完成报告 + 全模态 benchmark
```

---

## 12. 不动的

- 不删 _quarantine/
- 不大改 gate_engine.py 或 workstation-actions.ts 的语义
- 不让非安全代码持有明文密码
- 不提交 Kaggle 除非 Human Gate
- 不宣称超过 MLEvolve 除非 75 任务完成
