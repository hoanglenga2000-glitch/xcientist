# EvoMind SIIM-ISIC job90353 新电脑目标模式接管 Prompt

> 用法：先把当前工作树（包含未提交修改）及两个状态目录同步到新电脑，再将本文件中“开始 Prompt”到“结束 Prompt”的全部内容复制到新电脑 Codex。严禁复制旧电脑的 DPAPI 凭据文件；新电脑必须安全重建 `job90353` 命名 profile。

---

## 开始 Prompt

你现在是 EvoMind SIIM-ISIC 医疗科研闭环的**唯一接管执行器**。立即进入目标模式并持续工作，直到真实训练、科研交付、系统 UI、真实 Chrome 录制、92 秒商业成片和全量 QA 全部完成。不要把连接检查、代码修复、测试通过或单个种子完成当作最终目标。

### 0. 唯一目标

持续监督并完成唯一真实 Run：

```text
evomind_siim_isic_a800_job90353_20260730_095826
```

完整终点固定为：

```text
A800 消融与 batch gate
→ 正式种子 43、44、45
→ OOF/测试预测收集与事务化聚合
→ Independent Review
→ candidate freeze
→ 私有 grader 恰好一次
→ Claim Audit
→ PDF / CSV / Code ZIP / Evidence ZIP
→ EvoMind UI 同 Run 验收
→ 真实 Chrome 连续录制
→ 92 秒中文商业宣传片
→ FFmpeg 全量 QA
```

如果当前线程没有活动 Goal，创建以下 Goal；如果用户已经打开目标模式，则读取并沿用，不要另建第二个 Goal：

```text
持续监督唯一 Run evomind_siim_isic_a800_job90353_20260730_095826 完成 A800 SIIM-ISIC 实验、独立复核、一次私有 grader、四项真实交付物、EvoMind UI 验收、真实 Chrome 录制、92 秒中文商业宣传片和全量 FFmpeg QA；不创建第二 Run、不提交 Kaggle、不影响其他进程。
```

### 1. 第一批动作：恢复上下文，不得跳过

工作区预期为：

```text
D:\桌面\codex\科研港科技
```

若新电脑路径不同，先找到同步后的项目根，再将下面所有绝对路径映射到该根；不得因此新建项目或新建 Run。

按顺序完整读取：

```text
AGENTS.md
docs\HPC_CONNECTION_MEMORY_CORE.md
configs\hpc_connection_memory_core.json
C:\Users\<当前用户>\.agents\skills\skill-router\SKILL.md
```

随后运行 skill matcher，并只加载真正需要的技能：

```text
ml-data-analyst
video-production-pro
jianying-editor
browser:control-in-app-browser 或 chrome:control-chrome
```

视频开始前还必须读取：

```text
C:\Users\<当前用户>\.codex\skills\video-production-pro\SKILL.md
C:\Users\<当前用户>\.codex\skills\jianying-editor\SKILL.md
C:\Users\<当前用户>\.codex\skills\jianying-editor\docs\agent-playbook.md
```

### 2. 当前权威绑定

```text
Run ID              evomind_siim_isic_a800_job90353_20260730_095826
HPC job             90353
Credential profile  job90353
Remote root          /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra
Dataset              siim-isic-melanoma-classification
Official submission forbidden
Formal seeds         43, 44, 45
Current bundle SHA   87c718cb79094bcbe8a67957928222a7e1c562ff7ec16fa1be5a2075b2547208
Remote supervisor    最近 PID 48912（接管时必须重新只读核验，不能盲信）
```

当前已证实的状态（2026-07-30 17:55 CST 左右）：

- 五种预处理消融已完成；两折、外部种子 `40,41,42`。
- 消融报告 SHA-256：`935a3a96356598e743b29a778d0058746a845fc6e225880aeacef13b42486a01`。
- 最终保留 `raw_multiview_v1`，系统拒绝了不稳定或无有效增益的复杂预处理。
- batch 128 超过约 55 GiB 门限后被拒绝；batch 96 通过。
- 旧 bundle `8e9b92...` 在两折合同门禁处自然退出；这是已知历史证据，不是当前新故障。
- same-run watcher 已部署新 bundle `87c718...`，重新执行 5 次 A800 门禁并启动远端 supervisor 41299。
- 种子 43 当前为 `attempt_002`，消融合同 `validated=true`，所有 14 项 checks 为 true。
- 已进入真实 GPU 训练；最近证据约为 A800 48.7 GiB、100% utilization、batch 96、8 个 DataLoader worker。
- 2026-07-30 17:51 CST，41299 在 epoch 边界自然进入 `needs_continuation`；旧电脑唯一 local supervisor 完成 5 次新门禁并以同一 bundle、同一 Run、`--resume` 启动 `continuation_launch_001`，当前最近远端 supervisor PID 为 48912。该变化不是第二 Run。
- 私有 grader 尚未执行；Kaggle 尚未提交；视频尚未录制。

不要重新跑消融，不要重新 batch probe，不要删除 `attempt_001`，不要创建 `attempt_003`，除非当前正式进程自然失败且同配置恢复逻辑明确要求。

### 3. 新电脑连接规则：DPAPI 不可复制

旧电脑 Windows DPAPI 文件绑定旧用户和旧机器，**不得复制、解密、改写或复用**。新电脑必须由用户本人通过安全输入重新创建 `job90353` profile；任何密码不得出现在 prompt、命令行参数、源码、日志、报告或视频中。

强制链路：

```text
新电脑 job90353 DPAPI profile
→ 本机 Clash/受管 SOCKS5
→ AIMSLAB 指定上游 SOCKS5
→ HPC SSH 网关 100.85.169.63:1235
→ profile 内绑定的 allocation 角色账号
→ job90353 容器
→ /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra
```

永久禁止：

- 校外直连或探测作业页面的 `10.120.x.x`。
- 把 `job90353` 当作 SSH 用户名。
- 从旧电脑复制 DPAPI XML。
- 只到代理或网关就声称已连接容器。
- 在 Host UUID、GPU UUID、型号/显存和远端根未匹配前执行 GPU、训练、收集或 grader。

若 profile 尚未建立，使用项目脚本安全交互录入；命令中只保留角色账号占位符，密码必须由 `Read-Host -AsSecureString` 或受管 stdin 输入：

```powershell
Set-Location 'D:\桌面\codex\科研港科技'

powershell -NoProfile -ExecutionPolicy Bypass -File `
  'scripts\install_hpc_ssh_credential_from_stdin.ps1' `
  -User '<ALLOCATION_ROLE_ACCOUNT>' `
  -Profile 'job90353' `
  -HostName '100.85.169.63' `
  -Port 1235 `
  -RemoteWorkspace '/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra' `
  -SocksHost '127.0.0.1' `
  -SocksPort <NEW_PC_MANAGED_SOCKS_PORT> `
  -AllocationBindingId '<ALLOCATION_BINDING_ID>' `
  -AllocationGeneration <POSITIVE_GENERATION>
```

安装完成时 profile 固定处于 `provisioning`；只有一次性 identity bootstrap 全部通过后才原子切换为 `active`。旧 schema metadata 不做静默迁移，必须保持 fail-closed 并安全重录。

确认指定代理链实际可用后，清除所有直接 SSH/身份 override 环境变量，执行一次 TOFU + 身份绑定：

```powershell
$env:EVOMIND_HPC_CREDENTIAL_PROFILE='job90353'
$env:EVOMIND_SIIM_HPC_JOB_ID='90353'
$env:EVOMIND_SIIM_RUN_ID='evomind_siim_isic_a800_job90353_20260730_095826'

python 'scripts\bootstrap_hpc_profile_identity.py'
```

该操作必须最终绑定并验证：pinned host key、Host UUID、GPU UUID、A800 型号/显存和专属远端根。之后正式连接只能使用 `RejectPolicy`。

### 4. 本地状态必须先同步

新电脑不能只 clone 默认分支，因为当前生产链包含未提交工作树修改。接管前必须从旧电脑同步完整当前工作树，至少包含：

```text
D:\桌面\codex\科研港科技\workspace\evomind_runs\evomind_siim_isic_a800_job90353_20260730_095826
D:\桌面\codex\科研港科技\workspace\hpc\job90353_siim_campaign\evomind_siim_isic_a800_job90353_20260730_095826
D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1
D:\桌面\codex\科研港科技\scripts
D:\桌面\codex\科研港科技\src
D:\桌面\codex\科研港科技\tests
D:\桌面\codex\科研港科技\web\research-agent-workstation
```

不要通过 HPC SSH 重传 25.77 GB SIIM 数据；数据已经在共享 NFS。不要运行 `git reset --hard`、`git clean`、回退、覆盖或创建提交。

关键文件 SHA-256（新电脑接管前核验；不同则说明工作树未同步完整，先修复同步，不得启动第二 supervisor）：

```text
docs/HPC_CONNECTION_MEMORY_CORE.md
e875ad36f0ed425944189d5da018b1cba7fe6ddf12ecae69c3903652ac85b714

configs/hpc_connection_memory_core.json
1d892d4ef231a173c417922edf06b3c718a82df12c86fac70fc0bcfeab046c20

scripts/manage_siim_job89508_campaign.py
a1510d8e738c3ff6f1a53029bdfc51c41e19b938a9394a30ac2b4838eba6d9d8

scripts/supervise_siim_job89508_end_to_end.py
8e07ad28679b78f7d69b7377150c7923e8ddb206925e77dda50db42983b03090

scripts/watch_siim_same_run_bundle_upgrade.py
447a32a4addd60bd7ec3cb1a82510f13af6e8c22f56863425982a9f1b1296974

scripts/watch_siim_job89508_gate.py
5e6135162738ef92dd78fbf8dc8253fcdb88d4c42cc7a1eebef3072f66d12d40

scripts/mlebench_medal_recovery_adapters.py
0fdaba52b47973953a9523e7d547f5183eae1bd9b164b206f93b61d2c77d5e8e

scripts/build_siim_workflow_ingress.py
d9b9b96b304d7d271d1e2fbb45989156175f22884246a6072b39949ced880ad0

scripts/aggregate_siim_multiseed_candidate.py
d9060dff62addeb9c248156fc6a8308e933e659ac3d1d86f7f9050bbe5a8d840

scripts/run_siim_private_grader_once.py
5af478c56f1ad6026fcc16cef47076226dd6ff5ce1dcc692fb8a4a0a30215a56

scripts/build_siim_claim_audit.py
20121101c6bdb0f12e0e56bed5b1c59d527361af5230d6941d86311f034b4977

scripts/build_siim_delivery.py
51c058be2523d4d1b83b589d44139b8c833d36094112ab649aed40d93f82960a

src/research_os/agent/siim_hpc_workflow.py
45b6b135c5d020cc96010042c9c7d5e7466bc068cba4e1ad5159d38c70000ed4

video-production/siim-isic-melanoma-commercial-v1/preproduction/production-contract.json
7c1f66d77fca5ef688d74c43cec71f22e4e6b21fe956636e0e7ad875b06b6e14
```

### 5. 接管顺序：先只读附着，禁止盲目启动

设置：

```powershell
$env:EVOMIND_SIIM_HPC_JOB_ID='90353'
$env:EVOMIND_HPC_CREDENTIAL_PROFILE='job90353'
$env:EVOMIND_SIIM_RUN_ID='evomind_siim_isic_a800_job90353_20260730_095826'
```

先运行只读状态：

```powershell
& 'D:\桌面\codex\科研港科技\workspace\release-venv\Scripts\python.exe' `
  'scripts\manage_siim_job89508_campaign.py' status `
  --run-id $env:EVOMIND_SIIM_RUN_ID
```

接管门禁必须全部成立：

```text
proxy_path_verified=true
job_container_verified=true
Run ID 精确匹配
bundle SHA 为 87c718...
远端 supervisor 进程存在
formal_seed 为 43/44/45 之一或状态已推进
signals_sent=0
other_processes_modified=false
official_submission 未执行
```

如果远端 supervisor 仍存活：**只附着，不得 launch，不得启动第二远端进程**。

旧电脑本地 end-to-end supervisor 已在交接时停止；新电脑确认本机不存在同 Run supervisor 后，只启动一个本地终局 supervisor：

```powershell
$run='evomind_siim_isic_a800_job90353_20260730_095826'
$campaign="D:\桌面\codex\科研港科技\workspace\hpc\job90353_siim_campaign\$run"
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'

$p=Start-Process `
  -FilePath 'D:\桌面\codex\科研港科技\workspace\release-venv\Scripts\python.exe' `
  -ArgumentList @(
    'scripts\supervise_siim_job89508_end_to_end.py',
    '--run-id',$run,
    '--poll-seconds','60',
    '--sample-interval-seconds','15',
    '--max-wait-seconds','0',
    '--max-transient-failures','8'
  ) `
  -WorkingDirectory 'D:\桌面\codex\科研港科技' `
  -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $campaign "new_pc_supervisor_$stamp.stdout.log") `
  -RedirectStandardError (Join-Path $campaign "new_pc_supervisor_$stamp.stderr.log") `
  -PassThru
```

启动后核验：只有一组本地 supervisor parent/child；远端 PID 没有改变；没有第二 Run。

### 6. 训练监督规则

每 5–10 分钟读取一次状态即可，不要高频 SSH。只有状态变化、异常或完成时向用户汇报。

种子 43 当前阶段：

- `attempt_002`
- 合同门禁已通过
- 真正 GPU 训练已经开始
- 正常情况下会在 `siim_resume_state` 中生成 `stage_*.pt`
- 每个 epoch 完成后 checkpoint 原子写入

必须监督：

```text
5 个 outer folds
每个 outer fold 的 3 个 inner folds
inner epoch selection
固定 epoch refit
OOF 每个训练样本恰好覆盖一次
测试预测 4,142 行，无 NaN/Inf
种子 43、44、45 顺序完成
```

资源约束：

- 显存上限约 55 GiB；batch 96。
- `hold_reasons` 必须为空。
- 其他进程显存增长或剩余显存低于 12 GiB 时，只允许本任务在 epoch 边界保存检查点并自然退出。
- 不向任何远端进程发信号，不终止、修改或抢占其他任务。
- 技术中断只允许同 Run、同配置、`--resume`。
- 科研分数不达预期时保留最佳父版本，不进行无限调参。

传输异常：`RetryableTransportError` 可在上限内重试；普通 `CredentialError`、host key/UUID 不匹配必须 fail-closed。不得回退到 `10.120.x.x`。

### 7. 三种子结束后的自动终局链

远端状态达到 `awaiting_collection_and_freeze` 后，让唯一的本地 end-to-end supervisor 继续完成：

```text
collect
→ transactional aggregate
→ Independent Review
→ candidate freeze
→ private grader exactly once
→ grader ingress
→ claim_audit_source.json
→ claim_audit.json
→ deterministic delivery
→ delivery ingress
```

严格验证：

- 候选聚合先在 staging 目录完成，再原子提升。
- review 和 freeze 哈希绑定后，才允许私有 grader。
- grader readiness 必须验证私有答案、4,142 行 schema/顺序、依赖与允许根。
- `private_grader_ledger.json` 的执行次数必须精确为 1。
- 不得用 grader 结果继续调参。
- Claim Audit 必须保留不可变 `claim_audit_source.json`。
- 九个 workflow task 最终全部 `completed`。
- Kaggle/official submission 始终为 forbidden/false。

唯一 Run 交付目录：

```text
D:\桌面\codex\科研港科技\workspace\evomind_runs\evomind_siim_isic_a800_job90353_20260730_095826
```

必须存在并验证：

```text
run.json
task_graph.json
events.jsonl
data_audit.json
research_design.json
preprocessing_ablation.json
metrics.json
fold_metrics.csv
oof_predictions.csv
submission.csv（4,142 行）
training_history.json
hpc_telemetry.jsonl
review.json
claim_audit_source.json
claim_audit.json
private_grader.json
private_grader_ledger.json
artifact_manifest.json
专业 HTML/PDF 报告
```

四个最终下载文件：

```text
evomind-siim-isic-report.pdf
evomind-siim-isic-results.csv
evomind-siim-isic-code.zip
evomind-siim-isic-evidence.zip
```

逐个核验存在、非空、字节数和 SHA-256 与 manifest 一致。

### 8. EvoMind UI 验收

实验完成前不得展示假分数、假审核或可点击假下载。

完成后重建并重启 UI：

```powershell
Set-Location 'D:\桌面\codex\科研港科技'
.\stop.ps1 -Port 8088
.\start.ps1 -Port 8088 -HostName 127.0.0.1 -Build
```

核验：

```powershell
$summary = Invoke-RestMethod 'http://127.0.0.1:8088/api/workstation-summary'
$summary.runtime.current_run.run_id
$summary.runtime.current_run.status
$summary.runtime.runtime_snapshot.deliverables.status
```

必须得到：

```text
evomind_siim_isic_a800_job90353_20260730_095826
completed
ready
```

浏览器逐页检查普通用户旅程、数据审计、任务图、事件账本、A800 运行、消融、指标、历史门槛、Independent Review、Claim Audit、专业图表和四个真实下载按钮。所有内容必须来自同一 Run。

### 9. 预录制门禁

科研闭环未完成前严禁录制。完成后执行：

```powershell
python 'video-production\siim-isic-melanoma-commercial-v1\scripts\verify_pre_recording_gate.py' `
  --project-root 'D:\桌面\codex\科研港科技' `
  --contract 'D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1\preproduction\production-contract.json' `
  --output 'D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1\preproduction\pre-recording-gate.json'
```

必须验证：

```text
status=passed
recording_allowed=true
verified_download_count=4
private_grader_execution_count=1
independent_review=passed
claim_audit=passed
official_submission_executed=false
same_run_evidence=true
```

随后生成证据绑定旁白：

```powershell
python 'video-production\siim-isic-melanoma-commercial-v1\scripts\build_narration_manifest.py' `
  --project-root 'D:\桌面\codex\科研港科技' `
  --contract 'D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1\preproduction\production-contract.json' `
  --output 'D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1\preproduction\narration-92s.json'
```

### 10. 真实 Chrome 录制与 92 秒商业片

输出目录：

```text
D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1
```

最终成片：

```text
D:\桌面\codex\科研港科技\video-production\siim-isic-melanoma-commercial-v1\final\evomind-siim-isic-medical-research-demo-zh-92s.mp4
```

必须使用真实 Chrome 正式系统，连续真实录制：

```text
普通用户输入一句话
→ 系统理解数据/HPC/验证/交付要求
→ 数据和泄漏审计
→ 五种预处理比较
→ A800 同 Run 事件账本回放
→ 真实模型与指标
→ Independent Review / Claim Audit
→ 专业报告图表
→ 下载 PDF、CSV、Code ZIP、Evidence ZIP
```

硬规则：

- 92.000 秒，误差 ±0.05 秒。
- 1920×1080，30 FPS CFR，H.264 High，yuv420p，BT.709。
- AAC LC，48 kHz 双声道。
- 单一中文女声，无原录屏杂音，默认无背景音乐。
- 综合响度约 -14 LUFS，True Peak ≤ -1.2 dBTP。
- 真实产品动态画面至少 85%。
- 禁止终端、桌面、任务栏、调试横幅、连接秘密和其他应用窗口。
- 禁止水平平移、左右摇晃和交替焦点。
- 固定中心放大最大 1.08 倍。
- 播放速度不超过 1.12 倍。
- 禁止 tpad、重复帧填时长、抽帧式滚动和超过 4 秒无变化画面。
- 所有评分、Run ID、GPU、审核和下载必须可追溯到同一真实 Run。
- 不声称 Kaggle 正式排名、奖牌或临床诊断能力。

严格按现有 `shot-list-92s.json`、integer-frame EDL 和 production contract 剪辑。业务剪辑脚本必须写在视频项目目录，不得修改 Skill 安装目录。

### 11. 视频 QA

执行并保存证据：

```text
ffprobe
全量解码
blackdetect
freezedetect
silencedetect
loudnorm
framemd5 / 重复帧检查
水平稳定性检查
SHA-256
联系表与指定关键帧逐张检查
```

门槛：

```text
时长 92.000 ± 0.05 秒
黑屏 0
超过 4 秒冻结 0
超过 1.5 秒静音 0
水平摇晃 0
外部应用/调试横幅/连接秘密 0
重复下载条目 0
最终下载文件数 4
公开榜单名次/正式奖牌/临床诊断声明 0
```

### 12. 汇报纪律与最终完成判定

只在以下情况汇报：

1. 训练种子或 outer fold/epoch 出现实质变化；
2. 发生需要处理的异常；
3. 三种子完成、终局链阶段变化；
4. UI/录制/视频 QA 完成；
5. 全目标完成。

每次汇报必须基于实时文件、进程、远端状态或媒体 QA 证据；不得凭记忆推断。连接、代码或测试通过不是最终完成。

只有当以下所有证据同时成立，才把 Goal 标为 complete：

```text
种子 43/44/45 完成
OOF 与 4,142 行测试预测完整
Independent Review passed
candidate freeze 完成
私有 grader execution_count=1
Claim Audit passed
四个真实交付物存在且哈希一致
EvoMind UI 同 Run 验收通过
真实 Chrome 素材存在
92 秒 MP4 存在且可播放
FFmpeg/画面/声音/证据 QA 全部通过
official_submission_executed=false
signals_sent=0
other_processes_modified=false
```

最终回复只汇报：

1. Run ID 与最终真实指标；
2. 四个交付文件绝对路径和 SHA-256；
3. 最终 MP4 绝对路径；
4. 时长、分辨率、FPS、编码；
5. 黑屏、冻结、静音和水平摇晃计数；
6. LUFS 与 True Peak；
7. MP4 SHA-256、联系表和 QA 报告路径；
8. 明确说明 Kaggle 未提交、没有影响其他进程。

现在立即从“工作树哈希核验 → 新电脑安全重建 job90353 DPAPI profile → 指定代理与容器身份核验 → 只读附着最近远端 supervisor 48912（或其实时替代状态）”开始，不要重新创建实验，不要重新跑消融，不要提前录制。

## 结束 Prompt
