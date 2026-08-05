# V2 契约移植共享规约(所有移植 agent 必读)

## 目标
把旧单体 `src/components/workstation/Screens.tsx` 与 `AiControlConsole.tsx` 中**指定旧屏幕**的
`data-ui-action` 审计契约 + 业务功能,移植进 `src/components/workstation/screens/` 里**已存在的同名新屏幕组件**。
**保留新屏幕的 V2 视觉(PageHeader / Panel / MetricTile / StatusBadgeV2 / 响应式 grid / 移动卡)**,
只注入旧屏幕的契约与功能深度。不要重写视觉,不要换成旧视觉。

## 两种契约模式(严格按旧屏幕原样保留)
模式A(路由式,无 onClick):元素只带 `data-ui-action="xxx"`,AppShell 全局 `handleUiClick` 代理会路由成后端动作/导航。
  → 移植时保留 `data-ui-action`,**不要**加 onClick,**不要**加 data-ui-skip-action。
模式B(直接式):元素同时带 `data-ui-action="xxx"` + `data-ui-skip-action="true"` + `onClick={() => props.runWorkstationAction?.("action", {...})}`。
  → 移植时三者**原样一起**保留。
判断依据:旧屏幕里该元素有没有 onClick。有→模式B;没有→模式A。

## data-ui-action 命名
- 沿用旧屏幕里出现的每个 action id,逐字保留(如 `tasks_select_${id}`、`tasks_refresh_queue`、`tasks_create_workstation_run`)。
- 动态 id 用模板字符串,保持与前缀一致(uiActionRoutePatterns 用前缀匹配)。
- 不要发明旧屏幕没有的新 action,除非旧屏幕该功能在新屏幕缺失而你补上了等价交互——此时用旧屏幕同前缀命名。

## 功能完整性(不降级)
新屏幕目前缺失但旧屏幕有的功能,必须补上(用新视觉呈现):
- 过滤 / 搜索 / 分页控件(保留其 data-ui-action)
- 图表(Recharts):若旧屏幕有趋势图,新屏幕也要引入同数据(recharts 已在依赖中)
- agent trace / assignment board / validation contract / failure & rollback 信息块
- 详情 inspector/drawer:长 JSON / artifact / run detail 用可关闭面板
- 空 / loading / error / blocked 状态用 primitives �� EmptyState/ErrorState/Skeleton
- 真实数据驱动:无数据显 `EmptyState`,connector/资源状态来自 summary,未知显 Unknown,**不得硬编码 Ready**

## 硬约束
- 不改 `props` 的 ScreenProps 类型(各新屏幕已统一);只用已有 props,如缺某回调可经 `runWorkstationAction` 实现。
- 不引入新依赖;只用 Tailwind、lucide-react、recharts、已有 primitives(`../primitives/Layout`、`StatusBadge`、`GateBadge`)。
- 文案统一走 `t(locale, "English", "中文")`(@/`../localization`);API 字段名/run id/metric 保留英文。
- 保留新屏幕已有的 `"use client"`、导出函数名(如 `export function TasksScreen`)。
- 非交互卡片不得有 hover/cursor;可点击行用 `<button>` 或带 data-ui-action 的元素,保证键盘可达。
- 长路径/run id 用 `CopyablePath`;状态用 `StatusBadgeV2` tone,颜色之外必须有文字。
- 完成后必须 `npm.cmd run typecheck` 通过(在 web/research-agent-workstation 下)。

## 验证该页完成的定义
1. 该旧屏幕的每个 `data-ui-action` id,在新屏幕源码中能找到(逐字或同前缀模板)。
2. 旧屏幕的业务信息块在新屏幕有对应呈现(用新视觉)。
3. typecheck 通过。

## 文件位置
- 旧屏幕:`src/components/workstation/Screens.tsx`(用导出函数名定位区间)、`AiControlConsole.tsx`
- 新屏幕(改写它):`src/components/workstation/screens/XxxScreen.tsx`
- 共享类型/primitives:`src/components/workstation/primitives/{Layout,StatusBadge,GateBadge}.tsx`、`../localization`
- 工作目录:`D:\桌面\codex\科研港科技\web\research-agent-workstation`
