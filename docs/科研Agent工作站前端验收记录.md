# 科研 Agent 工作站前端验收记录

- 验收日期：2026-06-11
- 前端工程：`web/research-agent-workstation`
- 设计参考：`docs/ui-final-design-v4`
- 本地地址：`http://127.0.0.1:8088`
- 验收结论：当前前端已完成 10 张设计图对应的页面骨架、主要交互、中文界面、动作日志和本地验收闭环。

## 已实现页面

1. Mission Control
2. Research Tasks
3. Workflow Graph
4. Code Runner
5. Agent Runtime
6. Experiments
7. Report Studio
8. Integrity Gates
9. Design System
10. Overview Board

## 当前验收结果

- 首页导航与主工作区可正常切换。
- 代码运行页包含 IDE 区、终端区、Claude Code 桥和动作日志。
- 报告页支持中文编辑、预览、保存、导出、提交与图表插入。
- 设置页支持中文/English 切换，且语言会同步到全局界面。
- 完整性 Gate 页显示来源追踪、可复现性、有效性、人工监督和限制说明。
- Claude Code 与 GPU SSH 仍保持“未配置”状态，未伪造已接入。

## 截图记录

- `docs/ui-final-design-v4/01B-research-mission-control-evidence-first.png`
- `docs/ui-final-design-v4/02-task-research-workspace.png`
- `docs/ui-final-design-v4/03-research-workflow-graph.png`
- `docs/ui-final-design-v4/04-code-runner.png`
- `docs/ui-final-design-v4/05-agent-runtime.png`
- `docs/ui-final-design-v4/06-experiments-record.png`
- `docs/ui-final-design-v4/07-report-studio.png`
- `docs/ui-final-design-v4/08-integrity-gates.png`
- `docs/ui-final-design-v4/09-design-system.png`
- `docs/ui-final-design-v4/10-overview-board.png`

## 说明

- 当前仍未恢复的是 Codex 侧 Chrome 扩展控制入口，因此直连 Chrome 的自动化点击和截图验收仍需外部状态恢复。
- 但本地页面、API、合同和浏览器侧视觉抽查已经完成，不影响后续接入 Claude Code 和 GPU 资源后直接推进上线流程。

## 2026-06-11 布局质量收口

- 报告编辑页已改为“大纲 + 宽编辑/预览工作区”，系统动作日志移入下方辅助审计区。
- 设置页已改为“配置主体 + 下方审计/连接器辅助区”，不再用右侧长日志压缩主表单。
- 代码运行页右侧优先显示本地运行、Claude Code、GPU、Patch、检查和证据，系统动作日志放到底部。
- 工作流页右侧优先显示 Selected Node，系统动作日志降级为辅助审计信息。
- `scripts/verify_ui_layout_quality.py` 已纳入 full acceptance，防止这些布局问题回退。
