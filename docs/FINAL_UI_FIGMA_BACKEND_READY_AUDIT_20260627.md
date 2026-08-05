# AI 科研工作站 UI / Figma / 后端可接入最终审计

日期：2026-06-27

## 审计结论

本轮审计对象已收口为 10 个唯一核心前端页面：

`overview`、`control`、`data`、`report`、`code`、`gpu`、`evidence`、`literature`、`runtime`、`settings`。

结论：当前实现已经从“参考图展示”收口为真实 React / Tailwind 可编辑组件。页面没有挂载整页设计图或截图覆盖，关键按钮、导航、状态控件、表格行和 Gate 阻断动作均保留了前端 action hook，可由后端按 `page + action_id` 接管。重复入口已移除：`mission` 兼容落到 `overview`，`evidence-detail` 兼容落到 `evidence`，侧边栏只保留“科研总览”和一个“证据台账”。

## Figma 证据

- Figma file key：`YRGlARCURv2sKKmSHeNWA6`
- Page node：`0:1`
- 当前实现大 frame：`23:2`，名称 `Research Agent Workstation`
- Figma 中确认存在 11 个参考 frame：
  - `11:2` `01-academic-research-os-mission-control-reference`
  - `16:2` `03-agent-execution-console-reference`
  - `17:2` `05-data-kaggle-refined-reference`
  - `15:2` `06-report-studio-refined-reference`
  - `12:2` `07-code-agent-ide-refined-reference`
  - `18:2` `08-gpu-hpc-console-refined-reference`
  - `19:2` `09-evidence-ledger-refined-reference`
  - `21:2` `10-evidence-ledger-detail-refined-reference`
  - `20:2` `11-literature-rag-refined-reference`
  - `13:2` `12-agent-runtime-refined-reference`
  - `14:2` `13-system-settings-refined-reference`
- 已下载 Figma 当前 frame 截图：
  `docs/ui-verification-20260627-clickable-11-pages/figma-current-23-2.png`

## 本地截图证据

Chrome Headless 已生成本地页面截图，目录：

`docs/ui-verification-20260627-clickable-11-pages/`

文件：

- `overview.png`
- `control.png`
- `data.png`
- `report.png`
- `code.png`
- `gpu.png`
- `evidence.png`
- `literature.png`
- `runtime.png`
- `settings.png`
- `side_by_side_reference_vs_current.png`

这些截图用于人工比对设计还原度；源码层面不依赖这些图片渲染 UI。

## 后端对接契约

已固化文档：

`docs/UI_FRONTEND_API_CONTRACT_20260627.md`

统一入口：

```http
POST /api/workstation-actions
Content-Type: application/json
```

通用点击 payload：

```json
{
  "action": "ui_component_click",
  "task_id": "playground_series_s6e6",
  "metadata": {
    "page": "code",
    "component_type": "button",
    "action_id": "blocked_send_to_hpc",
    "label": "Send to HPC",
    "href": null,
    "disabled": true
  }
}
```

后端可以先把 `ui_component_click` 当作 action log；之后再把关键动作拆成真实业务 action，例如 `create_workstation_run`、`approve_gate`、`submit_gpu_job`、`generate_report_draft`。训练、提交、凭据变更必须继续由 Gate 控制。

## 关键源码证据

- `web/research-agent-workstation/src/components/workstation/AppShell.tsx`
  - 页面容器带 `data-ui-page`
  - 全局 `onClickCapture`
  - 点击统一发送 `ui_component_click`
  - 支持 `aria-disabled="true"` 的阻断点击审计
- `web/research-agent-workstation/src/components/workstation/Screens.tsx`
  - 10 个正式页面均为真实 React 组件
  - 大量关键按钮显式设置 `data-ui-action`
  - Gate 阻断动作仍可审计
- `web/research-agent-workstation/src/components/workstation/Sidebar.tsx`
  - 导航保留真实页面切换和 action 标识
- `web/research-agent-workstation/src/components/ui/button.tsx`
  - 统一按钮组件支持 disabled / aria-disabled 视觉状态

源码统计：

| 文件 | `data-ui-action` | 按钮数量 | `<img>` | 旧覆盖标记 |
| --- | ---: | ---: | ---: | --- |
| `AppShell.tsx` | 7 | 4 | 0 | false |
| `Sidebar.tsx` | 3 | 4 | 0 | false |
| `Screens.tsx` | 102 | 101 | 0 | false |
| `button.tsx` | 0 | 1 | 0 | false |

## 自动审计脚本

新增脚本：

`scripts/audit_ui_backend_ready.ps1`

默认检查：

- 10 个唯一核心页面均返回 HTTP 200。
- 10 个页面均加载 Next CSS。
- 页面 HTML 不包含旧的整页图片覆盖标记。
- 核心源码不包含旧 overlay 标记。
- 三个代表动作可进入 `/api/workstation-actions` 并生成 artifact：
  - `save_settings_changes`
  - `rag_build_agent_context`
  - `blocked_send_to_hpc`

运行方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/audit_ui_backend_ready.ps1 -BaseUrl http://127.0.0.1:8088
```

本轮已保存审计输出：

`docs/ui-verification-20260627-clickable-11-pages/audit-ui-backend-ready-20260627.json`

审计结果：`Passed=true`，`FailureCount=0`。

最终复验输出：

`docs/ui-verification-20260627-clickable-11-pages/audit-ui-backend-ready-20260627-final.json`

最终复验时间：2026-06-27 12:18，结果仍为 `Passed=true`，`FailureCount=0`。

## 本轮命令验证结果

- `npm run build`：通过。
- `npm run typecheck`：通过。
- `scripts/audit_ui_backend_ready.ps1`：通过。
- Dev server：`http://127.0.0.1:8088`，`?page=overview` 返回 `200`。

说明：第一次并行执行 `typecheck` 与 `build` 时，`typecheck` 曾因 `.next/types` 正在生成而报告缺少生成文件；在 `build` 完成后重新执行 `npm run typecheck` 已通过。这不是业务代码类型错误。

## 响应式证据

除桌面截图外，已补充移动端视口截图：

`docs/ui-verification-20260627-clickable-11-pages/mobile/`

包含：

- `overview-mobile.png`
- `control-mobile.png`
- `data-mobile.png`
- `report-mobile.png`
- `code-mobile.png`
- `gpu-mobile.png`
- `evidence-mobile.png`
- `literature-mobile.png`
- `runtime-mobile.png`
- `settings-mobile.png`

## 前后端分离边界

本轮只验证前端组件、UI action 和接口契约；没有改动 GPU/HPC、Kaggle、训练脚本、密钥或真实提交逻辑。当前 UI 允许后端接入真实状态，但不能把 `blocked`、`pending`、`needs_human_gate` 伪装成已经执行。

## 剩余风险

- Figma 对比目前是结构化 metadata + 截图人工比对，不是自动像素级 diff。
- 部分页面展示的是演示状态数据；后端接入后应把 demo rows 替换为真实 API 响应。
- `ui_component_click` 是通用审计入口，不等同于已经完成所有业务 action 的后端实现。
- 黑色主题按钮已经预留 action，若要真实切换主题，还需要后端/前端状态持久化实现。

## 通过标准

通过标准是：`npm run build`、`npm run typecheck`、10 个正式页面路由 smoke、点击 smoke 和源码 overlay 扫描全部通过。

## 2026-06-27 去重更新

根据最新验收意见，已删除正式导航中的重复入口：

- 删除 `mission / 科研总控`，只保留 `overview / 科研总览`。
- 删除 `evidence-detail / 证据详情`，只保留 `evidence / 证据台账`。

保留历史链接兼容：`?page=mission` 落到 `overview`，`?page=evidence-detail` 落到 `evidence`。

新增前端 smoke 证据目录：

`docs/ui-verification-20260627-dedup-smoke/`

最终复验：

- `npm run build`：通过。
- `npm run typecheck`：通过。
- 10 个正式页面 route smoke：通过。
- 点击 smoke：通过。
- 旧链接兼容 smoke：通过。
- 首页 HTML 不再包含 `nav-mission` 或 `nav-evidence-detail`。
