# 44个可用比赛数据集 — Claude训练用

数据根目录: `/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/`

## 全部44个目录

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
/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/mlebench_raw_data/mlebench_raw_data/
```

## GPU服务器连接

```
代理链路: Clash(127.0.0.1:7890) → HKUST SOCKS5 → SSHPiper(100.85.169.63:1235)
SSH: aimslab-panliu3@100.85.169.63:1235 / lInVYkYbgF
GPU: A40 48GB, 1TB RAM, 580GB磁盘可用
训练目录: ~/jinghw/scripts/gpu_tra/
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

## 规则

- 所有训练脚本/产物放 `~/jinghw/scripts/gpu_tra/`
- 不自动提交Kaggle（需人工确认）
- 不虚构奖牌/CV分数当官方排名
- 不泄露token/密码
