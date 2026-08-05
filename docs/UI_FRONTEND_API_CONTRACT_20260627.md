# AI 科研工作站前端 UI / API 对接契约

日期：2026-06-27

## 目标

10 个唯一核心页面必须由真实 React / Tailwind 组件渲染，不允许使用整页图片覆盖。按钮、输入框、卡片、状态标签、表格行、导航入口等可交互元素统一记录为前端 action，后端后续可以在同一接口上接管具体业务动作。

## 10 个唯一核心页面

| 页面 | route query | 参考图 |
| --- | --- | --- |
| 科研总览 | `?page=overview` | `01-academic-research-os-mission-control-reference.png` |
| AI 控制台 / Agent 工作页 | `?page=control` | `03-agent-execution-console-reference.png` |
| 数据 / Kaggle | `?page=data` | `05-data-kaggle-refined-reference.png` |
| 报告工作室 | `?page=report` | `06-report-studio-refined-reference.png` |
| 代码 Agent IDE | `?page=code` | `07-code-agent-ide-refined-reference.png` |
| GPU / HPC 控制台 | `?page=gpu` | `08-gpu-hpc-console-refined-reference.png` |
| 证据台账 | `?page=evidence` | `09-evidence-ledger-refined-reference.png` |
| 文献 / RAG | `?page=literature` | `11-literature-rag-refined-reference.png` |
| Agent 运行时 | `?page=runtime` | `12-agent-runtime-refined-reference.png` |
| 系统设置 | `?page=settings` | `13-system-settings-refined-reference.png` |

兼容别名：`?page=mission` 会落到 `overview`，`?page=evidence-detail` 会落到 `evidence`，用于保护历史链接；侧边栏不再展示这两个重复入口。

## 统一 UI Action

所有普通 UI 点击先进入现有接口，不直接触发训练、Kaggle 提交、HPC 长任务或凭据操作：

```http
POST /api/workstation-actions
Content-Type: application/json
```

示例：

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

## 前端组件标识

- 页面容器带 `data-ui-component="workstation-page"` 和 `data-ui-page="<page_id>"`。
- `Button` 自动带 `data-ui-component="button"`。
- `Card` 自动带 `data-ui-component="card"`。
- `StatusBadge` 自动带 `data-ui-component="status-badge"`。
- `DenseTable` 行带 `data-ui-component="dense-table-row"` 和 `data-ui-action="open_table_row"`。
- 外壳层优先读取显式 `data-ui-action` / `data-testid`；如果缺失，会根据可见标签自动生成兜底 `action_id`。
- 关键业务按钮应优先使用显式 `data-ui-action`，避免后端依赖按钮文案推断业务动作。

## 已固化的阻断动作

这些按钮视觉上处于不可执行状态，但点击仍写入 action log。后端必须返回“被 Gate 阻断”的业务状态，不得绕过 Gate 执行真实动作。

| action_id | 页面 | 语义 |
| --- | --- | --- |
| `blocked_send_to_hpc` | code | Code Gate 未通过时阻断发送 HPC |
| `blocked_start_training` | gpu | 训练 Gate 未通过时阻断启动训练 |
| `blocked_submit_gpu_job` | runtime | Agent Runtime 中阻断提交 GPU Job |
| `blocked_final_evidence_approval` | evidence | 证据最终审批未通过 |
| `blocked_final_export` | evidence | 证据最终导出未通过 |
| `blocked_final_report_export` | report | 报告最终导出未通过 |
| `blocked_allow_official_submit` | gates | 官方 Kaggle 提交未获批准 |

## 主要显式 Action 命名约定

- 全局导航：`topbar_open_<page>`、`navigate_page`、`open_current_workspace`。
- 全局搜索：`global_search_input`、`search_command`。
- Overview：`mission_create_workstation_run`、`mission_open_evidence_ledger`、`mission_prepare_hpc_job`。
- Control：`control_run_through_workstation`、`control_open_attachment_<id>`。
- Code：`ask_code_agent`、`review_code_diff`、`run_code_smoke_test`、`request_code_quality_gate`。
- Evidence：`open_evidence_lineage_graph`、`export_evidence_csv`、`preview_selected_artifact`、`download_selected_artifact`。
- Report：`report_add_section`、`report_export_draft_pdf`。
- Literature/RAG：`rag_build_agent_context`、`rag_send_research_agent`、`rag_request_citation_audit`、`rag_refresh_index`。
- Runtime：`runtime_view_agent_context`、`runtime_request_code_gate`、`runtime_trace_tab_tool_calls`。
- Settings：`settings_language_zh_cn`、`settings_language_en_us`、`settings_theme_light`、`settings_theme_dark`、`save_settings_changes`、`test_all_connectors`。

## 后端对接建议

- `ui_component_click` 作为低风险 action log 写入，不直接触发训练、Kaggle 提交、HPC 长任务或凭据写入。
- 明确业务动作可以再拆成独立 action，例如 `create_workstation_run`、`approve_gate`、`submit_gpu_job`、`generate_report_draft`。
- 官方 Kaggle 提交、HPC 长训练、凭据修改必须继续走 Human Gate。
- 后端可按 `metadata.page + metadata.action_id` 路由到具体业务处理器。
- 对 `metadata.disabled=true` 的动作，只允许写审计日志和返回阻断原因。

## 验收要求

- 10 个唯一核心页面都能通过 `?page=<id>` 访问。
- 历史重复路由 `mission`、`evidence-detail` 能兼容落到保留页面。
- 源码不得挂载整页参考图、`DesignFidelityPage` 或 `design fidelity reference`。
- `npm run build` 和 `npm run typecheck` 必须通过。
- 点击按钮、输入框、状态标签、表格行时，前端能向 `/api/workstation-actions` 发送 `ui_component_click`。
- 新增后端接口时，不应破坏当前通用点击审计入口。
