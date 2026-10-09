# 正式站功能迁移：第二轮实施报告

日期：2026-10-08  
结论：**部分交付。真实任务主流程与个人模型配置已做本地集成；十二页功能尚未迁完，未达到发布条件，线上未切换。**  
原计划：[高级工具易用性计划](2026-10-08-advanced-tools-usability.md)  
首轮报告：[M1 隔离设计预览](2026-10-08-advanced-tools-usability.report.md)，原文保留。

## 1. 交付入口与边界

- 已确认设计：[8098 设计预览](http://127.0.0.1:8098/)，保留不覆盖
- 本轮真实应用：[8100 登录后进入新版任务](http://127.0.0.1:8100/login?next=/workspace)
- 本地公开测试账号：`preview-user`；测试密码：`local-ui-fixture-only`。不用于服务器账号
- 源码入口：`web/research-agent-workstation/src/app/workspace/page.tsx`
- 证据目录：`artifacts/advanced-tools-migration-20261008/`
- 源码审查包：该目录的 `task-ui-review-only.zip`；**不是部署包，不能直接上线**

本轮替换了外部模型服务边界，未替换任务、数据库、权限、运行、文件工具和交付物接口。页面中的隔离服务回复明确标为脚本；文件由真实工具写入、读取、发布。没有把这些结果当作真实模型、GPU 或科研结论验收。

新版暂为 `/workspace` 独立入口。正式默认首页、旧高级工具和其他业务仍保留。资料、文献、成果、项目等侧栏入口目前返回原页面；不是全部迁移完成的新版页面。

## 2. M0→M5 状态

| 阶段 | 状态 | 本轮已做 | 尚未完成 |
| --- | --- | --- | --- |
| M0 基线与能力 | 部分 | 线上 build、发布快照、376 个源码文件基线；十二页操作组与身份风险；补丁白名单 | 十二页所有按钮逐一实际验收；旧工具 Task 的可靠主体关联 |
| M1 任务主流程 | 部分 | 持久任务、草稿、对话、当前运行、真实步骤、文件；显式关联已有归属的 Run；刷新和重启恢复 | 旧 URL 映射、旧 Task 创建/绑定、任务上传、新界面的完整暂停/继续/取消与历史上下文续接 |
| M2 文献/代码/配置 | 部分 | 个人模型设置、DPAPI、配置版本、手动测试契约、任务内选择；保存后返回原任务 | 文献与代码迁移；资源和预算整合；实际模型联网；服务账户加解密验证 |
| M3 进度/审批/来源 | 部分 | 按当前任务与 Run 显示实际进度、失败原因、交付物来源/哈希 | 新界面的待确认卡片、批准/拒绝/参数失效全过程；证据台账完整迁移 |
| M4 实验/改进 | 未动 | 保留原实验与进化功能、结果及规则 | 口径一致的实验对比；改进计划、预算、预览、明确批准的新版操作 |
| M5 验收与候选 | 部分 | 限定候选构建、回归、桌面浏览器、API 文件校验、源码审查包 | 完整手机/键盘/读屏、多用户浏览器、全部故障恢复、三人试用、服务器隔离验收与正式切换 |

停止不可靠旧 Task 关联；其余未完成项如表所列，不能全部归因于该关联问题。本报告是阶段性交付，不是完成 M0→M5 的声明。

## 3. 十二页功能迁移矩阵

以下为源码核对到的操作组，不是每个按钮已调用或真实服务已验证。旧页代码本轮不顺带重写。

| 原页面 | 核对的操作组 | 新版位置与当前状态 | 后续边界 |
| --- | --- | --- | --- |
| 科研总览 | 刷新、连接概况、进入工作站、实验入口 | 我的任务：已接个人任务列表，未搬全局统计 | 全局汇总不能当个人进度 |
| DeepEvo 工作站 | 对话、方案/上下文、补丁、研究/自进化动作 | 任务对话：已接原 Run 接口与事件流 | 附件、复杂动作、运行控制未完整迁移 |
| 任务队列 | 选择旧 Task、状态与历史 | 我的任务：新增真实用户任务；更多内可关联有归属的 Run | 不把旧 Task 与 Run 按名字合并 |
| 实验中心 | 刷新、查看/导出记录、比较指标 | 成果保留旧入口；新版实验对比未动 | 数据集、指标方向、验证口径必须相同 |
| 文献/RAG | 检索、导入、阅读、选择论文、上下文/交接、审核、MD/JSON 导出 | 独立入口保留；新版任务桥接搁置 | 需要旧 Task 真实记录及可信所有权；不发送假 ID |
| 代码 Agent | 新文件、模板草稿、检查、质量门、应用/回滚、导出、HPC 交接 | 新版代码操作未迁；原能力保留 | `local_template` 不能算真实模型生成；应用需明确目标与批准 |
| GPU/HPC | 连接状态、资源配置、受控接入/验证 | 模型设置页提供原受控资源入口说明，未集中完成 | 不连接、探测、改凭据或验证 GPU |
| 流程编排 | 缩放、过滤、阶段、门控链接 | 新版进度显示所选 Run 实际步骤，部分替代 | 未把固定阶段或全局流程注入任务 |
| 自进化引擎 | 计划、dry-run、循环预览、批准/取消、执行 | 更多/改进方案未实现 | 没有默认批准、自动训练或改动结果 |
| Agent 运行时 | 刷新、选运行、日志/详细记录 | 当前任务 Run 读取和 SSE 已接；诊断详情部分收起 | 全局运行时视图未嵌入，控制能力未完整迁移 |
| 证据台账 | 来源、过滤、重置、CSV/草稿导出 | 文件与结果：已有实际产物、来源与哈希 | 完整台账未迁，哈希不证明结论成立 |
| 完整性门禁 | 批准、修改、拒绝 | 新版审批 UI 未实现；原授权规则保留 | 目标、参数版本、预算、拒绝和重复提交必须单独验收 |

## 4. 任务身份：已实现与暂停点

### 4.1 已有归属的用户任务和助手 Run

- 新增 `/api/assistant/tasks`，代理到受控 `/v1/user-tasks`
- SQLite 持久保存任务、独立 conversation ID、版本和 Run 关联；租户/用户来自已验证服务端会话
- 保存草稿只写任务，不创建 Run、不请求模型、不消耗 GPU
- 创建带幂等键；同键异内容和过期草稿版本返回 409；跨用户/租户不可读写
- 执行沿用 `/api/assistant/runs`，服务端验证用户任务、模型配置版本、对话归属、活动运行和幂等请求
- 一个 Run 的任务关联不可随意改写；显式历史关联先验证原 Run 所有权，不改历史主体、不自动续跑
- URL 保留任务、视图、运行；取消过时读取，草稿写入串行并检查版本
- 保存冲突时暂停自动保存；重新读取版本仍保留本页输入，需明确保存才覆盖服务器草稿
- 活动运行重复提交返回冲突；伪造旧任务字段或 `[Selected task:` 上下文不能绕过新任务桥接限制

### 4.2 旧工具 Task 仍无可靠归属桥

`prisma/schema.prisma` 的旧 `Task.owner` 为可空字符串，未建立用户/租户外键；`api/tasks/route.ts` 按任务状态读取全表，不能作为个人所有权证明。本轮没有复制此方式到新任务接口。

因此：

- 已有明确 ACL 的助手 Run 可显式关联
- 不能确认归属的旧 Task 不认领、不批量回填、不按标题匹配
- 旧文献/代码接口所需 Task 的创建与幂等绑定尚未实现；对应新版路径保持未完成
- 未建立完整历史对话续接；关联只保证可追溯查看，并不冒充历史对话自动合并

**待确认事项：需要管理员提供旧 Task ID → 登录主体/租户 ID 的权威证据来源；提供之前，这类记录只保留原入口，不纳入新版关联。** 这与已批准“允许显式关联”不同：缺的是记录归属证据，不是再次申请同一功能授权。

## 5. 用户自带模型

### 5.1 已实现的本地契约

- `/api/assistant/model-profiles`：创建、编辑、停用、默认选择、显式手动测试
- 支持已有 Anthropic、DeepSeek、OpenAI 兼容传输；不新增协议宣传
- 配置元数据与凭据分开；只读接口返回名称、类型、地址、模型、版本、状态和验证/使用时间，不返回密钥
- 凭据使用 Windows 当前服务账户 DPAPI；按租户/用户/配置/版本绑定，目录 DACL 仅当前账户和 SYSTEM
- 保存不访问 DNS、不调用模型；密钥不放普通设置表、浏览器持久化、Run 元数据或导出
- 仅允许安全公网 HTTPS/443；请求前核验全部 DNS 地址，连接已验证 IP、TLS 校验原主机；禁止代理继承和重定向
- 单次测试需要费用确认、幂等键；每用户每小时最多 3 次、最多 16 输出 token、总超时 30 秒、无自动重试/降级
- 单次任务调用输出上限 4096 token；继续受既有调用预算/权限控制，不改全局环境变量
- Run 绑定具体配置版本；修改个人配置不会悄悄替换运行中的密钥/模型；停用阻止后续请求
- 科研编排也使用同一绑定客户端；失败不退回平台模型或模板
- 对服务返回中的原始密钥字符串脱敏，防止服务直接回显密钥写入历史
- “已保存”“连接测试通过”“实际调用成功”分别展示；隔离预览的使用记录写明“隔离脚本调用（非真实模型）”

### 5.2 不能据此放行生产

- 已验证本机当前 Windows 用户、临时测试目录及 D 盘隔离目录；**未验证服务器实际服务账户的 DPAPI 和目录权限**
- 实际服务供应商、网络、收费、模型输出质量都未验证；三种协议通过的是外部边界替身测试
- 模型设置保存/停用与任务读取已验收；停用后重新启用、配置保存请求歧义后的恢复还有产品细节待补
- 安全测试覆盖内网/元数据/DNS 混合答案、跨主体加解密、版本锁定和直接密钥回显；不是完整渗透审计
- 预览环境禁止外部模型请求；手动测试按钮不可向真实服务发请求。隔离脚本只存在于本地启动辅助脚本，不进入运行时源码输入名单

## 6. 限定源码与发布源

| 项目 | 本轮证据 |
| --- | --- |
| 线上只读健康结果 | `ready`，build `overlay-invitation-beta-5141838f452e`；本轮开始时读取 |
| 发布源 | `D:/EV12/system-repair-20261008-gates/` |
| 原 source receipt SHA256 | `4738be6fffac739a6d3f3435b933242865d529cd3caba1134c9070e7bd924d1e` |
| 本轮源码基线 | `baseline.json`：376 个 web/src 与 runtime 文件；另保存当时 322 行 tracked dirty 状态 |
| 本轮覆盖范围 | `source-scope.json`：24 个文件，含 1 个 UI 测试；运行时输入另列 23 个文件 |
| 候选构建 | Next production standalone，BUILD_ID `WQBb8oekznyeVHQ7MOryY` |
| 新 source receipt | `candidate-source-receipt.json`：candidate `local-task-ui-cd93a0e50299`、源码哈希、原 receipt 与新 BUILD_ID |
| 保留核验 | `baseline-preservation.json`：376 文件范围内的 49 处既有发布差异均保留，白名单外文件未变化；不是全磁盘审计 |
| 源码差异/包 | `candidate-source.patch`、`task-ui-review-only.zip`、`review-package.json` |

新前端主要位于 `src/components/workstation/task-workspace/`，新增两个认证 API 和 `/workspace` 入口；旧文件只改 layout 的隔离外部脚本加载、登录表单安全和必要后端接入。

后端新增 `user_tasks.py`、`model_profiles.py`、`model_profile_secrets.py`、`personal_model_client.py`、`personal_model_http.py`；限定修改 `http_server.py`、`tenant_access.py`、`run_requests.py`、`assistant_runs.py`、`runtime.py`、`aibuild_engine.py`。

已保留既有未发布的 `WorkflowScreen.tsx`、`EvidenceLedgerScreen.tsx`、`api/settings/route.ts`、`user-preferences.ts`。没有整包发布脏工作树，没有复用旧发布 hash。

审查 ZIP 含源码、测试、辅助脚本和报告；其中辅助脚本是测试资料，**不属于服务器部署输入**。不包含 `.env`、真实密钥、数据库、用户历史、node_modules 或构建目录。服务器部署清单还需最终收敛后重新生成。

## 7. TDD 与回归证据

### 7.1 实际红 → 绿路径

证据均在 `artifacts/advanced-tools-migration-20261008/`，保留失败结果，没有事后制造红灯。

| 轮次 | 实际失败 | 实现与绿灯 |
| --- | --- | --- |
| 01 草稿 | 新任务 API 被拒绝，尚无契约 | 持久、归属、幂等、保存不执行；`red-01-draft.xml` → `green-01-draft.xml` |
| 02 更新 | 草稿更新路由缺失 | 版本冲突与跨用户保护；`red-02-update.xml` → `green-02-update.xml` |
| 03 关联 | 不属于用户的任务被执行入口接受 | 服务端身份绑定/历史显式关联；`red-03-binding.xml` → `green-03-binding.xml` |
| 04 模型配置 | 配置 API 缺失 | DPAPI、隔离与持久化；`red-04-model-profiles.xml` → `green-04-model-profiles.xml` |
| 05 运行模型 | Run 没有固定个人配置绑定 | 每 Run 客户端、版本和无降级；`red-05-model-binding.xml` → `green-05-model-binding.xml` |
| 06 重复/越界执行 | 活动任务第二请求、旧 Task 上下文绕过被接受 | 租约/活动状态校验和上下文限制；`red-06-task-admission.xml` → `green-regression.xml` |
| 07 科研编排 | 个人模型选择未到达科研编排客户端 | 接入同一版本绑定客户端；`red-07-research-model.xml` → `green-07-research-model.xml` |

### 7.2 最终自动检查

| 检查 | 结果 | 证据/范围 |
| --- | --- | --- |
| 后端选择性回归 | 200/200 通过 | `final-candidate-regression.xml`；任务、模型、租户、幂等、项目、助手、传输、科研编排；不是 Python 全套 |
| 模型补充回归 | 8/8 通过 | `final-model-regression.xml`；含新增服务回显密钥测试，与上行有重叠，不相加宣传 |
| 发布源 + 补丁前端 | 117/117 通过 | `candidate-web-tests.log`；独立候选源，不带本地无关修改 |
| 当前本地前端回归 | 244/244 通过 | `workspace-web-tests.log`；覆盖本地既有修改，不冒充服务器验证 |
| TypeScript | 两处均通过 | `candidate-typecheck.log`、`workspace-typecheck.log`；空日志配合 `final-checks.json` 的 exit 0 |
| Python 快门检查 | 3/3 通过 | `python-fast-gates.log`：密钥扫描、编译、175 模块导入；使用 `--skip-tests` |
| Next 构建 | 通过 | `candidate-build.log`；使用 production standalone，不放松 CSP |
| 文件下载 | HTTP 内容与哈希一致 | `artifact-download.json`、`api-downloaded-artifact.md`；另一主体返回 404 |

候选前端第一次测试因打包后的相对夹具路径不存在失败，保留 `candidate-web-tests.initial.log`。改用发布源自带 `test-fixtures` 和已支持的环境变量后重跑通过；没有改原业务测试来消除失败。

## 8. 真实浏览器：已操作与未覆盖

桌面内置浏览器操作了真实应用，不仅是 SSR 或健康接口：

- 登录 → 创建持久草稿 → 对话发送 → 实际 Run → 进度 → 文件预览
- 模型未配置时保留记录，显示中文原因和配置入口，不伪造回复
- 保存公开隔离模型配置、设默认，返回原任务；页面不读回密钥
- 隔离服务脚本驱动真实文件工具完成写入/读取/发布；完整文件可预览
- Task A 无文件，Task B 有自己 Run 的文件；切换未借用另一任务结果
- 重启候选服务、重新登录后，任务、模型配置、结果与 Task B 未发送草稿仍在；没有自动续跑
- 键盘 Enter 可切换进度/结果并打开预览；截图可见焦点。不是全程只用键盘验收

当前截图为浏览器实际 1280×720 视口：`final-progress-desktop.png`、`final-results-desktop.png`、`final-settings-desktop.png`；DOM 在同名 `*-dom.txt`。重启与跨任务证据为 `restart-task-a-results.txt`、`restart-task-b-draft.txt`。

产物身份：Task `utask_3cabb5b9c25b4bbda0dfbaecc31b4463` → Run `run_08cd5e4c25804b00a2a1249594d3a7ff` → Artifact `artifact_047def3dd45b4e9eae669a73e4f30895`。166 字节，SHA256 `ad0ed86de078ad233e2decf340548ff4517be650a6857227750e2201ece75fb6`。

| 必须覆盖的项目 | 当前结论 |
| --- | --- |
| 两任务/两用户/两租户 | 实际 HTTP 合约通过；浏览器只完成单用户切换两任务，双用户浏览器待验收 |
| 刷新、重启、返回配置后恢复 | 已观察持久数据与草稿；完整前进/后退、迟到响应竞态仍待验收 |
| 401/403/404/409 | 相关 API 回归有覆盖；不是所有 UI 故障组合已通过 |
| 超时、不可用、次数限制 | 模型边界替身测试通过；模型未配置的 UI 已操作；断网/保存失败/下载失败 UI 恢复不完整 |
| 手机交互 | 有响应式实现；**本轮未完成真实手机菜单和全流程交互**，8098 旧截图不能替代 |
| 键盘/读屏 | 部分焦点、标签与 Enter 交互；全程键盘、Esc/焦点返回、实际读屏待验收 |
| 浏览器下载 | 点过下载；API 下载与哈希已核对；未找到并核验浏览器保存文件，落盘确认仍待验收 |
| 历史保护/暂停/审批 | 隔离持久化与既有后端回归通过；生产完整历史快照、暂停状态和未处理审批未验收 |
| 三位普通用户 | 未组织。五项任务卡理解度全部待验收，不由工程测试代替 |

## 9. 实施偏离与本地修复

- M1 先接真实用户任务与已确认 Run 归属；旧 Task 桥没有证据就停。M2 的个人模型路径可独立验证，因此先推进；没有声称严格完成前一阶段后才进入下一阶段
- 新界面独立 `/workspace`，未替换正式默认首页和旧 URL；便于隔离审查，距离正式迁移仍有工作
- 初次开发模式被既有 CSP 禁止 eval，登录未 hydrate 时原生表单可能 GET 提交。改成安全 POST、hydrate 前禁用提交，并使用正式构建验证；未降低 CSP。只用公开隔离凭据测试
- DPAPI 目录测试发现 PowerShell 模块/Owner 设置差异；改用 .NET DACL，避免依赖修改 Owner 的权限。本机仅收紧已核实为空的隔离凭据目录；没有更改真实凭据或服务器权限
- 新旧源码、测试、隔离脚本分别登记；测试脚本的服务替身不进入生产运行时输入
- 使用 steady-do、TDD、personal-ui-taste、Browser、readable-docs：公共边界验证、任务信息组织、真实桌面交互、可扫描报告。未分派代理、创建会话、commit 或更新长期记忆

## 10. 发布与回滚记录

| 操作 | 本轮状态 |
| --- | --- |
| 公网健康/发布身份读取 | 已做，只读 |
| 本地候选构建与隔离运行 | 已做，8100/8877，数据位于 `D:/EV12/task-ui-local-20261008/data` |
| 服务器隔离候选 | 未做 |
| 正式切换 | 未做，未申请以本不完整候选切换 |
| 真实模型/外部写入/HPC | 未做 |
| 生产回滚 | 未发生，没有需要回滚的线上本轮变更 |

后续正式切换必须另行展示限定候选、新 source receipt/build identity、完整验收、活跃运行/审批检查及回滚步骤，再取得该次确认。发布不能夹带旧历史所有权回填。回滚仅恢复代码与配置，保留本轮后新增的用户任务、模型配置、运行和产物数据；不自动恢复执行。

## 11. 复跑与下一步

PowerShell，在仓库根目录：

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts/run_task_workspace_preview.py
```

预览已运行时不要重复启动；脚本检查 8100/8877 端口占用，不杀其他服务。每次启动采用新的隔离登录签名，但保留本隔离目录的数据。脚本不能用于生产启动。

独立终端的本地验证：

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts/verify_task_workspace_candidate.py
python -m pytest tests/test_user_tasks_http.py tests/test_model_profiles_http.py tests/test_personal_model_boundary.py tests/test_invitation_tenant_access.py tests/test_invitation_run_requests.py tests/test_invitation_projects.py tests/test_assistant_run_service.py tests/test_model_transport_reliability.py tests/test_invitation_aibuild.py -q -p no:cacheprovider -p tests.invitation_offline_plugin
```

运行环境已核对：Python 3.12.10；`D:/下载/node.exe` v24.13.0；`D:/下载/npm.ps1` 11.6.2。无依赖升级。

下一步按风险排序：

1. 明确旧 Task 所有权证据来源；有证据的记录逐条关联，无证据继续搁置
2. 完成文献/代码桥、新版上传、资源/预算设置、审批与运行控制，不仅保留旧链接
3. 完成实验对比与改进方案；再建立旧 URL 的可靠映射
4. 补齐手机、双用户浏览器、离线/失败恢复、实际下载落盘、三位用户理解度验收
5. 收敛发布候选；实际服务账户加密验证和真实服务验证另取授权；满足切换条件后再申请正式上线

最终放行：**可审查源码并使用本地集成预览；真实服务能力和生产迁移不放行。**
