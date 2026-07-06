# Claude — GPU服务器连接 + 数据位置 + 操作规范

## 一、GPU服务器连接

### 代理链路
```
本地 Clash Verge (127.0.0.1:7890)
  → HKUST SOCKS5 (8.163.52.223:1080)
    → SSHPiper (100.85.169.63:1235)
      → GPU容器 (9c967f3bffb4, NVIDIA A40 48GB)
```

### 前提条件
- Clash Verge 必须运行（系统托盘图标，SOCKS5端口7890）
- 如果连接失败，先检查 Clash 是否在线

### Python连接代码（直接复制使用）
```python
import socket, struct, paramiko

def connect_gpu():
    """连接GPU服务器，返回SSH客户端"""
    sock = socket.socket()
    sock.settimeout(20)
    sock.connect(('127.0.0.1', 7890))
    sock.send(b'\x05\x01\x00')
    sock.recv(2)  # SOCKS5握手

    target = b'100.85.169.63'
    sock.send(b'\x05\x01\x00\x03' + bytes([len(target)]) + target + struct.pack('!H', 1235))
    sock.recv(10)  # SSHPiper CONNECT

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname='100.85.169.63',
        port=1235,
        username='aimslab-panliu3',
        password='lInVYkYbgF',
        sock=sock,
        timeout=20,
        banner_timeout=30,
        allow_agent=False,
        look_for_keys=False
    )
    return ssh

# 使用示例
ssh = connect_gpu()
stdin, stdout, stderr = ssh.exec_command('hostname && nvidia-smi')
print(stdout.read().decode())
ssh.close()
```

### 如果连接失败
1. 确认Clash Verge运行中：`netstat -an | grep 7890`
2. 重试连接（偶尔SSHPiper需要刷新）
3. 如果持续失败，检查Clash节点是否在线

---

## 二、数据位置

### 根目录
```
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/
```

### 47个可用数据集完整路径

```
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/tabular-playground-series-may-2022/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/tabular-playground-series-dec-2021/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/new-york-city-taxi-fare-prediction/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/champs-scalar-coupling/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/billion-word-imputation/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/ventilator-pressure-prediction/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/nomad2018-predict-transparent-conductors/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/random-acts-of-pizza/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/spooky-author-identification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/tweet-sentiment-extraction/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/us-patent-phrase-to-phrase-matching/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/facebook-recruiting-iii-keyword-extraction/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/learning-agency-lab-automated-essay-scoring-2/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/stanford-covid-vaccine/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/statoil-iceberg-classifier-challenge/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/tgs-salt-identification-challenge/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/mlsp-2013-birds/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/whale-categorization-playground/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/aerial-cactus-identification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/aptos2019-blindness-detection/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/cassava-leaf-disease-classification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/dog-breed-identification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/dogs-vs-cats-redux-kernels-edition/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/leaf-classification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/denoising-dirty-documents/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/alaska2-image-steganalysis/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/bms-molecular-translation/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/cdiscount-image-classification-challenge/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/3d-object-detection-for-autonomous-vehicles/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/hms-harmful-brain-activity-classification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/kuzushiji-recognition/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/osic-pulmonary-fibrosis-progression/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/ranzcr-clip-catheter-line-classification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/siim-covid19-detection/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/plant-pathology-2020-fgvc7/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/herbarium-2020-fgvc7/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/AI4Code/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/chaii-hindi-and-tamil-question-answering/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/google-quest-challenge/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/jigsaw-toxic-comment-classification-challenge/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/lmsys-chatbot-arena/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/multi-modal-gesture-recognition/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/tensorflow-speech-recognition-challenge/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/freesound-audio-tagging-2019/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/jigsaw-unintended-bias-in-toxicity-classification/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/tensorflow2-question-answering/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/text-normalization-challenge-english-language/
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/text-normalization-challenge-russian-language/
```

### 已有训练脚本
```
~/jinghw/scripts/gpu_tra/gpu_batch_trainer_v1.py    # 批量表格训练器
~/jinghw/scripts/gpu_tra/mlebench_closed_loop_pipeline.py  # 闭环管道
```

---

## 三、操作规范（必须遵守）

### 文件路径
- **所有训练脚本、结果、日志放** `~/jinghw/scripts/gpu_tra/`
- **禁止**往 `~/` 根目录、`/tmp/`、或其他用户目录写文件
- 数据只读，不要修改原始数据集

### Kaggle提交
- **禁止自动提交Kaggle** — 必须经过人工确认（Human Gate）
- 需要提交时，先展示 submission.csv + validation_contract + claim_audit，等待批准

### 声明规范
- **不虚构奖牌/排名** — 没有官方Kaggle response就不能写"已获得X奖牌"
- **不把CV/proxy分数当官方排名** — 只能说"proxy evaluation"或"preliminary result"
- **失败必须记录** — 不能只报成功
- 每个claim必须有证据绑定（exp_id + artifact + metrics）

### 安全
- **不泄露** token、密码、API key
- SSH密码通过环境变量或 `*_FILE` 解析，不硬编码到新脚本中

### GPU资源
- GPU: NVIDIA A40 48GB, RAM 1TB, 磁盘580GB可用
- Python 3.10.12, CUDA 12.8
- Kaggle API已配置（eizharobinson）
