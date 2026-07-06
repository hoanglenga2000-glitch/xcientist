# MLE-Bench 75 数据清单 — Claude 训练用

> 更新时间: 2026-07-03
> GPU服务器: AI-X86_NVIDIA Job 87907, A40 48GB

## 数据根目录

```
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/
```

**所有比赛数据都在这个目录下，每个子目录是一个比赛。**

当前状态: **43 个可用 / 33 个空**

---

## 可用数据集 (43个) — 按任务类型分类

### 表格类 (Tabular) — 优先训练

```
mlebench_raw_data/tabular-playground-series-may-2022/      分类  804MB
mlebench_raw_data/tabular-playground-series-dec-2021/      分类  786MB
mlebench_raw_data/new-york-city-taxi-fare-prediction/      回归  5.4GB
mlebench_raw_data/champs-scalar-coupling/                  回归  1.1GB
mlebench_raw_data/billion-word-imputation/                 回归  4.0GB
mlebench_raw_data/ventilator-pressure-prediction/          回归  666MB
mlebench_raw_data/nomad2018-predict-transparent-conductors/ 回归  12文件
mlebench_raw_data/random-acts-of-pizza/                    分类  14MB
mlebench_raw_data/spooky-author-identification/            分类  4MB
mlebench_raw_data/tweet-sentiment-extraction/              分类  3MB
mlebench_raw_data/us-patent-phrase-to-phrase-matching/     分类  2MB
mlebench_raw_data/facebook-recruiting-iii-keyword-extraction/ 分类 78MB
mlebench_raw_data/learning-agency-lab-automated-essay-scoring-2/ 回归 34MB
mlebench_raw_data/stanford-covid-vaccine/                  分类  29MB
mlebench_raw_data/statoil-iceberg-classifier-challenge/    分类  288MB
mlebench_raw_data/tgs-salt-identification-challenge/       分类  1MB
mlebench_raw_data/mlsp-2013-birds/                         分类  2MB
mlebench_raw_data/whale-categorization-playground/         分类  2MB
```

### 图像类 (Image)

```
mlebench_raw_data/aerial-cactus-identification/            分类  3MB
mlebench_raw_data/aptos2019-blindness-detection/           分类  6文件
mlebench_raw_data/cassava-leaf-disease-classification/     分类  1MB
mlebench_raw_data/dog-breed-identification/                分类  717MB
mlebench_raw_data/dogs-vs-cats-redux-kernels-edition/      分类  814MB
mlebench_raw_data/leaf-classification/                     分类  37MB
mlebench_raw_data/denoising-dirty-documents/               图像  199MB
mlebench_raw_data/alaska2-image-steganalysis/              图像  16MB
mlebench_raw_data/bms-molecular-translation/               图像+文本 1.6GB
mlebench_raw_data/cdiscount-image-classification-challenge/ 图像  5MB
mlebench_raw_data/3d-object-detection-for-autonomous-vehicles/ 3D  99MB
mlebench_raw_data/hms-harmful-brain-activity-classification/ 医学图像 631MB/644文件
mlebench_raw_data/kuzushiji-recognition/                   图像  4.3GB/5338文件
mlebench_raw_data/osic-pulmonary-fibrosis-progression/     医学影像DICOM 153MB/289文件
mlebench_raw_data/ranzcr-clip-catheter-line-classification/ 医学影像 155MB/689文件
mlebench_raw_data/siim-covid19-detection/                  医学影像 9.1GB(zip未解压)
mlebench_raw_data/plant-pathology-2020-fgvc7/              图像  4文件
mlebench_raw_data/herbarium-2020-fgvc7/                    图像  (kagglehub下载中/已完成)
```

### NLP/文本类

```
mlebench_raw_data/AI4Code/                                  NLP  68MB
mlebench_raw_data/chaii-hindi-and-tamil-question-answering/ NLP  30MB
mlebench_raw_data/google-quest-challenge/                   NLP  14MB
mlebench_raw_data/jigsaw-toxic-comment-classification-challenge/ NLP 133MB
mlebench_raw_data/lmsys-chatbot-arena/                      NLP  232MB
```

### 多模态/特殊

```
mlebench_raw_data/multi-modal-gesture-recognition/          多模态  22GB
mlebench_raw_data/tensorflow-speech-recognition-challenge/  语音  2文件
```

---

## 连接GPU服务器

```
本地Clash(127.0.0.1:7890) → HKUST SOCKS5 → SSHPiper(100.85.169.63:1235)
用户名: aimslab-panliu3  密码: lInVYkYbgF
```

```python
import socket, struct, paramiko
sock = socket.socket(); sock.settimeout(20)
sock.connect(('127.0.0.1', 7890))
sock.send(b'\x05\x01\x00'); sock.recv(2)
hb = b'100.85.169.63'
sock.send(b'\x05\x01\x00\x03' + bytes([len(hb)]) + hb + struct.pack('!H', 1235))
sock.recv(10)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='100.85.169.63', port=1235, username='aimslab-panliu3',
    password='lInVYkYbgF', sock=sock, timeout=20, banner_timeout=30,
    allow_agent=False, look_for_keys=False)
```

## 重要规则

1. 所有训练文件放 `~/jinghw/scripts/gpu_tra/`，不污染主目录
2. GPU: NVIDIA A40 48GB, RAM 1TB, 磁盘 580GB可用
3. Python 3.10.12, CUDA 12.8
4. 已有训练脚本: `gpu_batch_trainer_v1.py`, `mlebench_closed_loop_pipeline.py`
5. Kaggle账号: eizharobinson (已配置)
6. 不自动提交Kaggle（需Human Gate）
7. 不虚构奖牌/不把CV当官方排名
