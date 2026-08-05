# AIMSLAB HPC 连接记忆核心

> 状态：项目级强制记忆（MUST READ）
> 生效日期：2026-07-30
> 依据：`D:\桌面\AIMSLAB HPC校外连接指南（实习生版本）.pdf`
> 依据文件 SHA-256：`7850dea325dd61cccf9ae19c5b11dcc75f840d12ec881d86259ffe3fe45a7b51`
> PDF 视觉核验：4/4 页，2026-07-30

## 永久规则

**校外访问 AIMSLAB HPC 时，作业页面展示的 `10.120.x.x` 容器内网地址是禁止的直连目标，不存在“不是首选但可作备用”的语义；所有连接、训练、监控、收集和 grader 都必须先进入指定 SOCKS5 代理链，再连接 HPC SSH 网关，最后由本次 allocation 的 HPC 角色账号路由进入对应容器。**

**永久记忆口令：先走指定代理，再经 SSH 网关和 allocation 角色路由进入本次作业容器；没有完成目标容器身份核验，就不算已经连接 HPC，也不得执行 GPU 命令、训练、监控、收集、grader 或远端写入。**

**连接就绪不变量：`proxy_path_verified AND job_container_verified`。代理可达、网关可达、SSH Banner 可见或作业页显示容器内网地址，任何单项都不等于“已经连接”。**

固定语义链路：

```text
EvoMind / SSH 客户端
  -> 本机命名 DPAPI profile（job<作业号>；它不是角色账号）
  -> 本机 Clash / 受管 SOCKS5 入口（指南默认 127.0.0.1:7890，或 profile 绑定的本地端口）
  -> AIMSLAB 指定上游 SOCKS5（指南地址 8.163.52.223:1080；认证信息不写入本文件）
  -> HPC SSH 网关 100.85.169.63:1235
  -> profile 中绑定的本次 allocation HPC 角色账号
  -> 对应 JupyterLab / VSCode 作业容器
  -> /hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra
```

## 连接可用判据

仅满足以下全部条件时，连接状态才可从 `gateway_reached` 变为
`job_container_verified`：

1. 指定 SOCKS5 代理可达，并且 SSH 流量确实经过该代理。
2. HPC SSH 网关返回有效 SSH Banner，固定主机键校验通过。
3. 本次 allocation 的角色账号认证成功，并由平台路由进入对应作业容器。
4. 容器内 Host UUID、GPU UUID、GPU 型号/显存与当前 job profile 绑定值一致。
5. 远端工作目录位于
   `/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/`，且没有越出该根目录。

只到达代理或 SSH 网关不代表已经进入容器；只看到 `10.120.x.x` 分配信息也不代表可连接。上述任一条件缺失时保持 fail-closed，只允许本地契约读取、代理健康检查、网关 Banner 检查和只读身份探测。

作业页面中的 `10.120.x.x:6988` 仅属于 allocation 分配信息，禁止作为校外执行器的直连、备用、回退或路径探测目标。即使 TCP 偶然可达，也不改变指定代理链的强制性；正式链路必须看到 SSH Banner，并完成固定主机键校验和容器身份校验。

## 规则来源与项目增强

- **PDF 强制链路**：指定代理 -> HPC SSH 网关 -> 本次 allocation 的 HPC 角色账号。校外不得跳过指定代理，也不得换用未批准的代理路径。PDF 还明确要求分配账号仅限本人使用、禁止转借或共享，账号回收后立即失效，并禁止通过共享代理通道用 SSH 传输大批量文件。
- **EvoMind 项目增强门禁**：命名 DPAPI profile、作业号和 profile 内绑定的 allocation 角色账号一一对应，并在登录后继续核验固定主机键、Host UUID、GPU UUID、GPU 型号/显存和专属远端根。这些是项目增强的 fail-closed 门禁，不声称由 PDF 原文提出。
- PDF 没有要求登录网关后再手动执行第二条 SSH 命令；“进入容器”在本项目中表示由 allocation 角色账号完成容器路由并通过上述身份门禁。

### PDF 逐页证据索引

- **第 1 页**：指定上游 SOCKS5 为 `8.163.52.223:1080`；HPC SSH 登录节点为 `100.85.169.63:1235`；代理凭据与 HPC 角色账号是两套不同凭据；Clash 本地 `mixed-port` 为 `7890`。
- **第 2 页**：访问 HPC 时必须保持 Clash 代理开启；`100.85.169.63/32` 通过指定代理规则转发。
- **第 3-4 页**：Windows SSH 使用本地 `127.0.0.1:7890` 的 SOCKS5 `ProxyCommand` 连接网关，并以管理员分配的 HPC 角色账号认证。
- **第 4 页**：禁止大批量文件走 SSH、禁止私改代理配置、禁止账号转借共享、账号结束后回收、禁止直连或改用其他代理。

PDF 中的代理账号、代理密码和 HPC 角色凭据均属于秘密，只保存在受管配置/Windows DPAPI 中；本记忆核心只记录路由事实和非秘密端点。

## 强制启动顺序

1. **读取本文件**，再读取当前 Run 和 allocation 绑定。
2. 确认 Clash/受管代理桥已启动；确认 profile 中记录的本地 SOCKS 端口可达，并验证它实际转发到指南指定的上游 SOCKS5，而不是直连或其他代理。
3. 使用独立命名 profile：`EVOMIND_HPC_CREDENTIAL_PROFILE=job<作业号>`。
4. `job<作业号>` 只是 profile 名称，不是 HPC 角色账号；角色账号及凭据只保存在该 profile 的 Windows DPAPI 凭据中。
5. 密码只通过安全输入写入 Windows DPAPI；源码、命令参数、日志、报告和视频中均不保存密码。
6. profile 的连接目标使用 HPC SSH 网关，代理设置从 profile metadata 读取；不要在训练脚本中自行拼接直连地址。
7. 新 profile 首次使用时，只允许一次性 TOFU：采集网关主机键，随后立即用 `RejectPolicy` 重连。
8. 固定并核验：SSH 主机键、Host UUID、可见 GPU UUID、GPU 型号/显存、专属远端根。
9. 执行五次只读 GPU 资源采样；全部通过后才允许启动训练。
10. 远端脚本、状态、日志和结果只能写入：
   `/hpc2hdd/home/aimslab/jinghw/scripts/gpu_tra/`
11. 大数据复用共享 NFS，禁止通过 SSH/SCP/SFTP 重传 SIIM 等大批量数据。
12. HPC 角色账号和命名 profile 仅限本人使用，禁止共享。allocation 或角色账号被回收、停用或重新分配后，立即冻结旧 profile；旧 profile 不得改指或复用，必须通过安全录入新建/重建与新 allocation 对应的 `job<作业号>` profile，并重做主机键、Host/GPU UUID 和远端根绑定。

## 账号与 profile 生命周期

1. **Provisioning**：通过安全输入创建当前 job 的命名 DPAPI profile，绑定管理员为本人分配的 allocation 角色账号；profile 名本身永远不充当 SSH 用户名。
2. **Active**：指定代理、主机键、job ID、allocation 角色账号、Host UUID、GPU UUID 和远端根全部绑定后才允许执行。
3. **Frozen**：allocation 到期/回收，角色账号回收/替换，或主机键、Host UUID、GPU UUID、远端根任一不符时立即冻结。冻结后禁止连接、训练、监控、收集、grader、改指和复用。
4. **Rebuild**：新或重新发放的 allocation 必须新建/重建对应 job-scoped profile，重新安全录入并完成全套身份绑定。旧 profile 保持冻结，按审计流程退役。

### Named profile 元数据与不可逆状态门禁

- 正式命名 profile 元数据 schema 固定为 `evomind.hpc.dpapi_profile.v2`，必须包含
  `profile_state`、`allocation_binding_id`、正整数 `allocation_generation`、
  `profile_instance_id` 和正整数 `lifecycle_revision`。
- 安装器只创建 `provisioning` profile。一次性 identity bootstrap 是唯一允许读取
  provisioning 凭据的专用路径；Host/GPU/host-key/root 绑定完成后，它在同一事务中写入
  `container_binding_sha256` 并切换为 `active`。
- 运行时 loader 在启动 DPAPI 解密子进程之前必须确认状态严格等于 `active`；旧 schema、
  缺字段、未知状态、binding/generation/instance 漂移全部 fail-closed，禁止自动推断或原地补值。
- freeze 与 retire 使用 profile 目录内的不可覆盖 tombstone：
  `hpc_profile_frozen.tombstone.json`、`hpc_profile_retired.tombstone.json`。任一 tombstone
  存在时，运行时 loader 必须在解密前拒绝；代码不得自动删除 tombstone 或把旧 profile
  改回 active。
- DPAPI 解密、正式 SSH 建连、bootstrap activation commit、freeze 和 retire 共享同一个
  profile lifecycle 文件锁；tombstone 先落盘再替换 metadata，消除解密/建连与状态切换的
  TOCTOU 窗口。
- 本地生命周期命令只改元数据和 tombstone，不解密凭据、不访问网络：

  ```powershell
  python scripts/manage_hpc_profile_lifecycle.py freeze --profile job<JOB_ID> --reason allocation_expired
  python scripts/manage_hpc_profile_lifecycle.py retire --profile job<JOB_ID> --reason audit_retired
  ```

- 旧 metadata 的迁移策略是安全重录：旧 profile 保持不可用并留作审计证据；新 allocation
  使用新的 binding identity、更高 generation 和新的 profile instance 完整重建，禁止把旧
  metadata 静默升级后继续使用。

## 连接失败时的判定顺序

```text
本地 SOCKS 不可达
  -> 修复 Clash/代理桥

网关 TCP 可达但 SSH Banner 前关闭
  -> 连接路径错误或未走指定代理
  -> 不要反复重装密码，不要退回容器内网直连

SSH 主机键不匹配
  -> 立即 HOLD，核对 profile 与管理员提供的网关

认证失败
  -> 核对本次 allocation 的角色账号和独立 DPAPI profile
  -> allocation 或账号已回收时立即冻结 profile，不反复重试

认证成功但 Host/GPU UUID 不匹配
  -> 立即 HOLD，禁止训练；说明路由到了错误容器或 allocation 已变化

身份一致但 GPU 门禁失败
  -> 保持只读等待，不修改或终止其他进程
```

## EvoMind 实现约束

- 统一通过 `src/research_agent_workstation/server/core/gpu_credentials.py` 加载连接配置。
- 训练、监控、收集和 grader 进程必须同时设置：

  ```powershell
  $env:EVOMIND_SIIM_HPC_JOB_ID='<JOB_ID>'
  $env:EVOMIND_HPC_CREDENTIAL_PROFILE='job<JOB_ID>'
  $env:EVOMIND_SIIM_RUN_ID='<RUN_ID>'
  ```

- 命名 profile 必须满足 `job_id` 与 `credential_profile` 一一对应。
- 命名 profile 不是 HPC 角色账号；角色账号必须来自 profile 内受 DPAPI 保护的凭据。
- `known_hosts` 必须存在且非空；正式连接只能使用 `RejectPolicy`。
- 命名 profile 必须绑定 `expected_host_uuid` 与 `expected_gpu_uuid`。
- 所有启动证据必须记录：`signals_sent=0`、`other_processes_modified=false`。
- Kaggle 官方提交保持 Human Gate；SIIM 当前工作流固定为 `official_submission=forbidden`。

## 人工 SSH 参考（不用于自动化保存密码）

Windows 指南链路：

```text
ssh -o ProxyCommand="ncat --proxy 127.0.0.1:7890 --proxy-type socks5 %h %p" \
  <HPC_ROLE_ACCOUNT>@100.85.169.63 -p 1235
```

EvoMind 自动化不直接调用含密码的命令，而是使用 DPAPI 命名 profile 和 profile 中记录的代理端口。

## 当前任务核对

- 当前医疗实验作业：`90353`
- 当前命名 profile：`job90353`
- 正确方式：`job90353` 命名 profile -> 指定代理 -> HPC SSH 网关 -> profile 绑定的 allocation 角色账号 -> 独立 A800 容器
- 禁止方式：把 `job90353` 当作角色账号，或把 allocation 页面的 `10.120.x.x` 内网地址当作校外直连、备用或回退训练入口
