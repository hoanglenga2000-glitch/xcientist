import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
const src = (name) => readFile(`src/components/workstation/${name}`, 'utf8');
test('settings and sidebar use one connector projection without legacy model names', async () => {
  for (const path of ['Sidebar.tsx','screens/SettingsScreen.tsx']) {
    const code = await src(path); assert.match(code, /@\/lib\/connector-presentation/);
    assert.doesNotMatch(code, /deepseek_api|gpu_hpc|kaggle_api/);
  }
});
test('settings cannot issue legacy global writes or fake rotation and governance success', async () => {
  const code=await src('screens/SettingsScreen.tsx');
  assert.doesNotMatch(code,/recordSettingsAction|runWorkstationAction\?\.|"Active", "活跃"/);
  assert.match(code,/refreshSummary\(\)/); assert.match(code,/未核验/);
});
test('global settings and data pages do not inherit a task banner', async () => {
  const code=await src('AppShell.tsx');
  assert.match(code,/activePage !== "settings" && activePage !== "data"/);
});
test('missing data evidence stays unknown and submission still requires approval', async () => {
  const code=await src('screens/DataKaggleScreen.tsx');
  assert.doesNotMatch(code,/value[:=][^\n]*inv\?\.[^\n]*\?\? 0/);
  assert.match(code,/boolText\(dpapi\?\.configured/);
  assert.match(code,/提交必须人工批准/);
});
test('project selection is URL bound and old results clear before fetching', async () => {
  const code=await src('screens/ProjectsScreen.tsx');
  assert.match(code,/searchParams\.get\("project"\)/);
  assert.match(code,/searchParams\.set\("project", selected\)/);
  assert.match(code,/setLoading\(true\); setError\(""\); setRuns\(\[\]\)/);
  assert.match(code,/project_identity_mismatch/);
  assert.doesNotMatch(code,/recovering: text\("恢复中"/);
});
test('generic legacy console has a real assistant entry instead of an empty body', async () => {
  const code=await src('UserResearchJourney.tsx');
  assert.match(code,/工作站任务入口/); assert.match(code,/href="\/\?page=assistant&demo=user"/);
});
test('GPU presence alone never implies a live connection', async () => {
  const code=await src('screens/GpuHpcScreen.tsx');
  assert.match(code,/hpcConnectionVerified\(hpcEntry\)/);
  assert.doesNotMatch(code,/hpcEntry \? t\(locale, "Connected"/);
});
test('toolbar uses scoped results, native dialog, keyboard shortcut and real identity', async () => {
  const code=await src('WorkspaceToolbar.tsx');
  assert.match(code,/\/api\/assistant\/projects/); assert.match(code,/\/api\/session\/status/);
  assert.match(code,/showModal\(\)/); assert.match(code,/event.ctrlKey \|\| event.metaKey/);
  assert.doesNotMatch(code,/search_command|notification_open|profile_open|科研管理员/);
});
test('workflow zoom and filter are local actions and never imply live Planner execution', async () => {
  const code=await src('screens/WorkflowScreen.tsx');
  assert.match(code,/setZoom\(\(value\) => Math.min/); assert.match(code,/style=\{\{ zoom: zoom \/ 100 \}\}/);
  assert.match(code,/aria-pressed=\{pendingOnly\}/); assert.doesNotMatch(code,/const isCurrent = idx === 1/);
});
test('runtime scope refusal is not misrepresented as a model outage', async () => {
  const code=await src('SuperAgentV1Panel.tsx');
  assert.match(code,/response.status === 403/); assert.match(code,/不代表模型故障/);
  const route=await readFile('src/app/api/super-agent/status/route.ts','utf8');
  assert.match(route,/errorResponse\("scoped_status_unavailable", 403\)/);
});
