# job90948 最终上线验收

最终判定：**GO — EvoMind 核心、新用户演示与 job90948 严格 GPU 运行时可上线。**

生成时间：2026-08-07 12:49（Asia/Shanghai）

## 当前运行态

- 工作站：`http://127.0.0.1:8088/?page=assistant`
- Dashboard PID：`14980`
- Runtime PID / 端口：`23316 / 8765`
- Source build stale：`false`
- `evomind open`：成功；前后 PID 不变；`bootstrap_pending=false`
- OpenAI 号池：`127.0.0.1:65068`，当前监听正常
- HPC 受管 bridge：`127.0.0.1:7890`，当前监听正常
- 验收残留进程：无

## job90948 GPU

- Named DPAPI profile：`job90948`
- 严格状态：`ready / job_container_verified`
- 只读身份采样：`5/5`
- GPU：`NVIDIA A800-SXM4-80GB / 81920 MiB`
- Bounded GPU smoke：`passed`
- Gateway banner：`SSH-2.0-SSHPiper`
- 远端写入根：`/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/`
- `signals_sent=0`
- `other_processes_modified=false`

## 最终门禁

- Python CI：`4/4`，`2684 tests`，`170 imports`
- Web：typecheck、lint、`47 tests`、production build 全通过；npm audit `0 vulnerabilities`
- Full acceptance：`40/40 passed`
- Persistent Chromium：`15/15` 页面、`46` 个 DOM/桌面/移动证据、`0` runtime errors
- GPT-5.6 Assistant：`5/5`，mean `0.97`，P95 `54.312s`（门槛 `90s`）
- OpenAI gateway：models、nonstream、stream、tool_call 全通过，模型 `gpt-5.6-sol`
- Kaggle：`dpapi_real_api / authenticated_real_api`，只读列表 `20`；官方提交未执行并继续受 Human Gate 控制
- New-user readiness：`ready_for_new_user_evomind_gateway`
- Final-two blockers：`0`
- Launch decision：`go_fully_ready`

## 已处理的主要回归

- 修复多个 verifier 的 local automation 鉴权与 Origin/JSON 请求契约。
- 通用 action matrix 不再污染真实运行记录；Code Agent apply/rollback 由专用原子生命周期覆盖。
- full/detail 与 lightweight summary 统一接入 job90948 strict HPC、Kaggle real API 和核心 passed runs。
- 修复 `HomeClient -> AppShell` 的 summary/task/ready 接线。
- Chromium deep-link 使用 automation token，不再抢占用户 bootstrap；本地化与 heading 契约同步。
- `start_verified_workstation -Command smoke -Build` 现在真实执行 build/restart。
- `evomind open` 在健康服务和待用 bootstrap 存在时不再重启已审计 PID。
- 修复 Taxi 模块重复导入导致的全套测试顺序回归，并更新 strict-HPC 的遗留测试契约。

## 可选外部 Provider 说明

- Claude Code 真实外部会话在有界等待内被取消，审计记录为 optional `allow_failure`；不影响主路径 OpenAI `gpt-5.6-sol`、本地工具循环、Code Agent patch lifecycle 与正式演示。
- DeepSeek 不属于本次 OpenAI `gpt-5.6-sol` 主演示的必选路径。

## 证据入口

- `docs/launch_resource_readiness.json`
- `docs/verified_workstation_launch_audit.json`
- `docs/launch_go_no_go_20260613.json`
- `docs/final_two_resource_blockers.json`
- `docs/kaggle_dpapi_readiness.json`
- `workspace/hpc/job90948_bounded_smoke_current.json`
- `workspace/llm/openai_gateway_smoke_current.json`
- `workspace/evaluation/assistant_novice_quality_gpt56_current.json`
- `workspace/evaluation/assistant_novice_quality_gate_current.json`
- `workspace/new_user_release_readiness.json`
- `docs/evomind_pages_20260708/acceptance_index.json`

报告不包含密码、API key、DPAPI 密文或完整 GPU UUID。
