# Academic Research OS UI 升级记录 - 2026-06-14

## UI 问题诊断

- 信息架构仍偏普通后台：原导航以 `Mission Control / Research Tasks / Runtime / Workflow` 为主，缺少 `Data & Kaggle`、`GPU / HPC`、`Evidence Ledger`、`Literature` 等科研 OS 一级入口。
- 科研能力虽存在于多个页面，但缺少一张首屏能力地图，用户需要自己理解哪些能力 ready、哪些 blocked。
- GPU、Claude、Kaggle 的诚实状态已经在资源总控里表达，但还没有各自的专业控制台视角。
- Evidence 与 Literature 能力分散在实验、报告和 Gate 页面里，缺少学术可信度的独立入口。
- 移动端和本地化验收已有保护项，任何新导航必须保留既有受保护文案，例如 `任务总控台`。

## 新设计方向

本轮将首页和导航升级为 Academic Research OS 信息架构：克制、高信息密度、证据优先、资源状态诚实，不做营销英雄页，不伪造外部资源 ready。

视觉语言继续沿用现有科研控制台风格：深色 command deck、浅色证据面板、紧凑卡片、状态 badge、代码路径 chip、细网格背景和低饱和语义色。

## 使用的 UI 能力

- shadcn MCP：确认 `sidebar`、`card`、`button`、`badge`、`tabs`、`table`、`command`、`dialog`、`chart`、`tooltip`、`dropdown-menu`、`progress`、`accordion` 在 `@shadcn` 注册表中真实存在。
- shadcn 示例：读取 `card-demo`、`table-demo`、`chart-demo`，确认卡片、表格和 Recharts 图表模式。
- Magic MCP：上一轮尝试生成 `research OS dashboard` 组件方案时超时；本轮再次请求组件灵感时服务返回缺少 `x-api-key`。没有采用任何未成功返回的 Magic 产物。
- Skills：使用 `design-everyday-things` 做可发现性、反馈、约束与错误预防判断；使用 `design-sprint` 的 Map/Prototype/Test 思路收敛到可验证的小步交付。

## 修改文件

- `web/research-agent-workstation/src/components/workstation/navigation.ts`
- `web/research-agent-workstation/src/components/workstation/Sidebar.tsx`
- `web/research-agent-workstation/src/components/workstation/Screens.tsx`
- `web/research-agent-workstation/src/app/page.tsx`
- `scripts/verify_scientific_ui_polish.py`
- `scripts/verify_academic_os_page_deeplinks.py`
- `web/research-agent-workstation/src/app/page.tsx`
- `.codex-ui-designer.md`
- `UI_AGENT_PROMPT.md`

## 新增组件 / 页面

- `AcademicOsReadinessPanel`
- `DataKagglePipeline`
- `GpuHpcConsole`
- `EvidenceLedger`
- `LiteratureKnowledge`
- `AcademicPageHeader`
- `MetricTile`
- `DarkMetric`
- `ResourceMiniCard`

## 科研能力增强

- 新增首屏 Academic OS 能力地图，集中展示 Mission、Experiment、Evidence、Report、GPU、DeepSeek、Code Agent、Kaggle、Literature 的真实状态。
- 新增 Academic OS 质量评分卡，把 10 项质量目标与验证产物绑定展示。
- 新增 `Data & Kaggle` 页面，展示比赛接入、样本行数、YAML、本地 baseline、官方 API token 与 Human Gate 边界。
- 新增 `GPU / HPC` 页面，明确区分 `GPU Verified`、`Job Key Pending`、`SSH Auth Blocked`、`Whitelist only`。
- 新增 `Evidence Ledger` 页面，按 artifact / source / claim / status 展示证据绑定状态，缺证标为 `Unverified`。
- Evidence Ledger 增加缺证策略：任何没有来源文件、run_id 或 Gate 记录的结论进入报告前必须显示 `Unverified / 未验证`。
- 新增 `Literature` 页面，预留 DOI、arXiv、Semantic Scholar、PubMed、Citation Manager、Dataset Card、Model Card、Peer Review 等入口，同时声明未联网检索时不展示伪引用。
- 侧边栏改为 Academic Navigation，同时保留 `任务总控台` 以满足既有本地化验收合同。
- 新增 `?page=` / `#page` 深链接能力，核心页面可被直接打开、复测和截图验收。

## 验证结果

- `npm run typecheck`：passed
- `npm run build`：passed
- `python scripts/verify_scientific_ui_polish.py`：passed，已保护新增 Academic OS 页面
- `python scripts/run_full_acceptance.py --dashboard-url http://127.0.0.1:8088`：passed，`checks_run=38`
- `python scripts/verify_no_plaintext_secrets.py`：passed
- Verified launcher：passed，`dpapi_loaded.deepseek=true`、`dpapi_loaded.claude=false`、`dpapi_loaded.kaggle=false`
- `rg` 精确扫描用户曾提供的敏感密码标记：0 命中
- `python scripts/verify_academic_os_page_deeplinks.py --url http://127.0.0.1:8088`：逐页浏览器截图验收命令，输出到 `docs/academic_os_pages_20260614/`

## 截图证据

- 桌面：`docs/academic_os_desktop_20260614.png`
- 移动：`docs/academic_os_mobile_20260614.png`
- DOM 留证：`docs/academic_os_dom_20260614.html`
- 逐页截图与 DOM：`docs/academic_os_pages_20260614/`

DOM 断言已确认以下词条可见：

- `Academic Research OS`
- `科研操作系统能力地图`
- `数据与 Kaggle`
- `GPU / HPC`
- `证据账本`
- `文献知识`
- `任务总控台`
- `DeepSeek`
- `Claude`
- `Kaggle`

## 仍然保持的真实边界

- DeepSeek：ready，DPAPI + smoke 已通过。
- GPU：4 x NVIDIA A800 已验证；自动 SSH job 仍等待远端公钥授权。
- Claude Code：缺少 Anthropic Key，保持 Not Configured。
- Kaggle 官方：缺少 token，保持 Not Configured；官方提交必须 Human Gate。
- Literature 外部检索：本轮只预留入口，没有联网检索，不展示任何伪造论文或引用。
- Figma：本轮未收到可写入或可截图的 Figma file key / node URL，因此没有把本地页面推送到 Figma；当前以本地浏览器截图和 DOM 留证作为验收依据。

## 最终收口补充 - 2026-06-14 13:12

- `AcademicQualityScorecard` 已按 Magic 页面截图的视觉方向做保守强化：总分、整体进度、Verified / Monitor 计数、10 个质量维度、证据路径和进度条集中展示。
- Magic MCP refiner 仍返回工具结构错误；本轮只把用户提供的 Magic 页面截图作为视觉参考，没有声称采用 MCP 代码产物。
- `scripts/run_full_acceptance.py` 已纳入 `scripts/verify_academic_os_page_deeplinks.py --url <dashboard>`，完整验收现在会覆盖 11 个核心页面的 deep-link、DOM 和桌面/移动截图。
- `scripts/verify_no_plaintext_secrets.py` 已对 SQLite 临时 journal 文件竞态做容错，扫描规则未放宽。
- 最新验证：`npm run typecheck` passed；`npm run build` passed；`python scripts/run_full_acceptance.py --dashboard-url http://127.0.0.1:8088` passed，`checks_run=39`。
- 最新页面截图证据：`docs/academic_os_pages_20260614/acceptance_index.json`、`mission_desktop.png`、`mission_mobile.png` 以及 11 个页面的 DOM / desktop / mobile 产物。
