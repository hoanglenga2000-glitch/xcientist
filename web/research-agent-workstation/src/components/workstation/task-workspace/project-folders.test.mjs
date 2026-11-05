import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const require = createRequire(import.meta.url);
const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (name, parent, ...args) {
  if (name.startsWith('@/')) name = fileURLToPath(new URL('../../../' + name.slice(2), import.meta.url));
  return originalResolve.call(this, name, parent, ...args);
};
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};
const source = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const css = source('./workspace.css');
const rule = selector => new RegExp(selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\{([^}]*)\\}').exec(css)?.[1] ?? '';
// C6b/C6c append small 760px blocks; the folder rules live in the main mobile block.
const mobile = css.slice(css.indexOf('@media(max-width:760px){.task-workspace .tw-live-seg'));

// lucide renders width before class; match both regardless of order.
const svg = (icon, width) => `<svg(?=[^>]*lucide-${icon}[ "])(?=[^>]*width="${width}")[^>]*>`;
const PID = 'project_' + 'a'.repeat(32);
const PID2 = 'project_' + 'b'.repeat(32);
const NOW = Date.parse('2026-10-27T12:00:00Z');
const at = hours => new Date(NOW - hours * 3600e3).toISOString();
const task = (id, title, extra = {}) => ({ id: 'utask_' + id.padEnd(32, '0'), title, draft: '', version: 1, conversation_id: 'c', status: 'draft',
  created_at: at(1), updated_at: at(1), pinned_at: null, archived_at: null, project_id: null, ...extra });
const projectTasks = () => [
  ...Array.from({ length: 6 }, (_, index) => task('p' + index, `项目对话${index}`, { project_id: PID, updated_at: at(index + 1) })),
  task('pp', '置顶的项目对话', { project_id: PID, pinned_at: at(0.2), updated_at: at(30) }),
  task('t1', '今天的对话', { updated_at: new Date().toISOString() }),
];
const sidebarProps = extra => ({ tasks: projectTasks(), setTasks() {}, projects: [{ id: PID, name: '蛋白结构' }, { id: PID2, name: '空项目' }], setProjects() {},
  archivedCount: 0, setArchivedCount() {}, selection: '', query: '', onLeaveOpen() {}, onNavigate() {}, ...extra });

test('t237 C2b.5: expand state lives in deepevo.sidebar.projectExpand, collapsed by default, forced open without a write', () => {
  const { PROJECT_EXPAND_KEY, readExpand, writeExpand, folderExpanded } = require('./project-folders.ts');
  assert.equal(PROJECT_EXPAND_KEY, 'deepevo.sidebar.projectExpand');
  assert.deepEqual(readExpand(null), {});
  assert.deepEqual(readExpand('not json'), {});
  assert.deepEqual(readExpand('[1,2]'), {});
  assert.deepEqual(readExpand(JSON.stringify({ [PID]: true, [PID2]: 'yes' })), { [PID]: true });
  assert.deepEqual(writeExpand({ [PID]: true }, PID2, false), { [PID]: true, [PID2]: false });
  assert.equal(folderExpanded(PID, {}, {}, ''), false);
  assert.equal(folderExpanded(PID, { [PID]: true }, {}, ''), true);
  assert.equal(folderExpanded(PID, { [PID]: false }, {}, PID), true, 'the current conversation project is forced open');
  assert.equal(folderExpanded(PID, {}, { [PID]: false }, PID), false, 'a click in this page wins over the force');
  const sidebar = source('./SidebarHistory.tsx');
  // The forced value is only read, never stored.
  assert.doesNotMatch(sidebar, /writeExpand\([^)]*forced/);
});

test('t237 C2b.5: folders cap at 5 rows then 「全部 {n} 个」; copy table', () => {
  const { FOLDER_ROWS, folderRows, PROJECT_COPY, projectHref, projectDraftHref } = require('./project-folders.ts');
  assert.equal(FOLDER_ROWS, 5);
  const seven = Array.from({ length: 7 }, (_, index) => index);
  assert.deepEqual(folderRows(seven), { shown: [0, 1, 2, 3, 4], more: true });
  assert.deepEqual(folderRows(seven.slice(0, 5)), { shown: [0, 1, 2, 3, 4], more: false });
  assert.equal(PROJECT_COPY.group, '项目');
  assert.equal(PROJECT_COPY.create, '新建项目');
  assert.equal(PROJECT_COPY.name, '项目名称');
  assert.equal(PROJECT_COPY.expand('蛋白结构'), '展开 蛋白结构');
  assert.equal(PROJECT_COPY.conversations('蛋白结构'), '蛋白结构 的对话');
  assert.equal(PROJECT_COPY.menu, '项目操作');
  assert.equal(PROJECT_COPY.expandHint, '展开');
  assert.equal(PROJECT_COPY.moreHint, '项目操作');
  assert.equal(PROJECT_COPY.convoHint, '的对话');
  assert.equal(PROJECT_COPY.all(7), '全部 7 个');
  assert.equal(projectHref(PID), `/workspace?project=${PID}`);
  assert.equal(projectDraftHref(PID), `/workspace?new=1&project=${PID}`);
});

test('t237 C2b.5: every project is a folder (empty too); project conversations never enter 今天/昨天/更早', () => {
  const { sidebarSections } = require('./conversation-menu.ts');
  const view = sidebarSections(projectTasks(), [{ id: PID, name: '蛋白结构' }, { id: PID2, name: '空项目' }], NOW);
  assert.deepEqual(view.projects.map(folder => [folder.project.name, folder.items.length]), [['蛋白结构', 7], ['空项目', 0]]);
  assert.equal(view.projects[0].items[0].title, '置顶的项目对话', 'pinned first');
  assert.doesNotMatch(JSON.stringify(view.groups), /项目对话/);
});

test('t237 C2b.5: sidebar 「项目」 sits after 置顶 and before 今天, with the 「+」 and no drag', () => {
  const { SidebarHistory } = require('./SidebarHistory.tsx');
  const html = renderToStaticMarkup(React.createElement(SidebarHistory, sidebarProps()));
  const labels = [...html.matchAll(/<p class="tw-history-label">([^<]+)<\/p>/g)].map(match => match[1]);
  assert.deepEqual(labels.slice(0, 3), ['置顶', '项目', '今天']);
  assert.match(html, new RegExp(`<div class="tw-history-head"><p class="tw-history-label">项目</p><button type="button" class="tw-group-add" aria-label="新建项目">${svg('plus', 14)}`));
  assert.doesNotMatch(html, /draggable/);
  // No projects: only the title row and 「+」, no empty copy.
  const none = renderToStaticMarkup(React.createElement(SidebarHistory, sidebarProps({ projects: [], tasks: [task('t1', '今天的对话', { updated_at: new Date().toISOString() })] })));
  const group = none.slice(none.indexOf('tw-history-label">项目'), none.indexOf('tw-history-label">今天'));
  assert.match(group, /aria-label="新建项目"/);
  assert.doesNotMatch(group, /tw-folder|还没有/);
});

test('t237 C2b.5: folder rows use Folder, a count, a separate chevron button and a 「…」 menu trigger', () => {
  const { SidebarHistory } = require('./SidebarHistory.tsx');
  const html = renderToStaticMarkup(React.createElement(SidebarHistory, sidebarProps()));
  assert.doesNotMatch(html.slice(html.indexOf('tw-history-label">项目'), html.indexOf('tw-history-label">今天')), /lucide-folder-open/);
  const row = new RegExp(`<div class="tw-folder-row"><span id="tw-folder-expand-${PID}" class="tw-sr-only">展开</span><button type="button" class="tw-folder-chevron" aria-expanded="false" aria-controls="tw-folder-${PID}" aria-labelledby="tw-folder-expand-${PID} tw-folder-title-${PID}">${svg('chevron-right', 12)}.*?</svg></button>`
    + `<a href="/workspace\\?project=${PID}" class="tw-folder-link"[^>]*>${svg('folder', 16)}.*?</svg><span id="tw-folder-title-${PID}" class="tw-row-title">蛋白结构</span><small class="tw-folder-count">7</small></a>`
    + `<span id="tw-folder-more-${PID}" class="tw-sr-only">项目操作</span><button type="button" class="tw-row-more" aria-haspopup="menu" aria-expanded="false" aria-labelledby="tw-folder-more-${PID} tw-folder-title-${PID}">`);
  assert.match(html, row);
  assert.match(html, new RegExp(`aria-labelledby="tw-folder-expand-project_b+ tw-folder-title-project_b+"`));
  assert.doesNotMatch(html, /\saria-label="展开 /);
  assert.doesNotMatch(html, /tw-folder-title-project_b+[^>]*>空项目<\/span><small class="tw-folder-count">0<\/small>/, "C5b L6: hide zero counts");
  assert.doesNotMatch(html, /role="group"/, 'collapsed by default');
});

test('t237 C2b.5: the current conversation project opens: pinned first with a 12px Pin, 5 rows, then 「全部 7 个」', () => {
  const { SidebarHistory } = require('./SidebarHistory.tsx');
  const html = renderToStaticMarkup(React.createElement(SidebarHistory, sidebarProps({ selection: 'utask_p2'.padEnd(38, '0') })));
  assert.match(html, /aria-expanded="true" aria-controls="tw-folder-project_a+" aria-labelledby="tw-folder-expand-project_a+ tw-folder-title-project_a+"/);
  const start = html.indexOf(`<div class="tw-folder-items" id="tw-folder-${PID}" role="group" aria-labelledby="tw-folder-title-${PID} tw-folder-convo-${PID}">`);
  assert.ok(start > 0);
  const group = html.slice(start, html.indexOf('</a>', html.indexOf('tw-folder-all', start)) + 4);
  const titles = [...group.matchAll(/<span class="tw-row-title">([^<]+)<\/span>/g)].map(match => match[1]);
  assert.deepEqual(titles, ['置顶的项目对话', '项目对话0', '项目对话1', '项目对话2', '项目对话3']);
  assert.match(group, new RegExp(svg('pin', 12).replace('[^>]*>', '[^>]*tw-row-pin[^>]*>')));
  assert.match(group, new RegExp(`<a class="tw-folder-all" href="/workspace\\?project=${PID}">全部 7 个</a>`));
  // Shared 01 row: the 「…」 of a conversation is still there.
  assert.match(group, /aria-label="对话操作：置顶的项目对话"/);
});

test('t237 C2b.5: the project row menu is 重命名 · 新对话 · 删除项目 with danger text only on 删除项目', () => {
  const { ProjectMenuEntries } = require('./ProjectMenu.tsx');
  const html = renderToStaticMarkup(React.createElement(ProjectMenuEntries, { project: { id: PID, name: '蛋白结构' }, onCommand() {} }));
  const labels = [...html.matchAll(/<span class="tw-menu-label">([^<]+)<\/span>/g)].map(match => match[1]);
  assert.deepEqual(labels, ['重命名', '新对话', '删除项目']);
  assert.equal(html.match(/role="menuitem"/g).length, 3);
  assert.equal(html.match(/is-danger/g).length, 1);
  assert.match(html, /class="tw-menu-item is-danger"[^>]*><svg[^>]*lucide-trash2[^>]*>.*<span class="tw-menu-label">删除项目/);
  const menu = source('./ProjectMenu.tsx');
  assert.match(menu, /<PopMenu id=\{id\} label=\{PROJECT_COPY\.menu\}/);
  assert.match(menu, /role="menu" aria-label=\{label\}/);
  for (const key of ["'ArrowDown'", "'ArrowUp'", "'Home'", "'End'", "'Escape'", "'Tab'"]) assert.ok(menu.includes(key), key);
  const sidebar = source('./SidebarHistory.tsx');
  // Right-click and long-press open the same menu on the folder row.
  assert.match(sidebar, /onContextMenu=\{event => folderContextMenu\(/);
  assert.match(sidebar, /onTouchStart=\{event => folderTouch\(/);
});

test('t237 C2b.5: folder CSS: 6px chevron→folder gap, 50px indent, 140ms rotation, 24px 「+」, 44px touch rows', () => {
  assert.match(rule('.task-workspace .tw-group-add'), /width:24px;height:24px/);
  assert.match(css, /\.task-workspace \.tw-group-add:hover\{background:(?:#E7EAE8|var\(--tw-selected\))\}/);
  assert.match(rule('.task-workspace .tw-folder-chevron'), /width:28px/);
  assert.match(rule('.task-workspace .tw-folder-chevron'), /padding:0 0 0 10px/);
  assert.match(rule('.tw-folder-chevron svg'), /transition:transform 140ms ease/);
  assert.match(css, /\.tw-folder-chevron\[aria-expanded="true"\] svg\{transform:rotate\(90deg\)\}/);
  assert.match(rule('.tw-folder-row>a'), /padding:6px 10px 6px 28px/);
  assert.match(rule('.tw-folder-row>a'), /gap:6px/);
  assert.match(rule('.tw-folder-items .tw-history-row>a'), /padding-left:50px/);
  assert.match(rule('.tw-folder-count'), /font-size:12px/);
  assert.match(css, /\.tw-folder-row:hover \.tw-folder-count,\.tw-folder-row:focus-within \.tw-folder-count,\.tw-folder-row\.is-open \.tw-folder-count\{opacity:0\}/);
  assert.match(mobile, /\.tw-folder-row>a\{min-height:44px;/);
  assert.match(mobile, /\.task-workspace \.tw-folder-chevron\{width:44px;height:44px;padding:0;justify-content:flex-end/);
  assert.match(mobile, /\.tw-folder-items \.tw-history-row>a\{padding-left:50px\}/);
  assert.match(mobile, /\.tw-history \.tw-folder-all\{[^}]*padding-left:50px\}/);
  assert.match(mobile, /\.task-workspace \.tw-group-add\{width:44px;height:44px\}/);
  assert.match(mobile, /\.tw-folder-count\{opacity:1\}/);
  assert.match(css, /@media\(prefers-reduced-motion:reduce\)\{[^@]*\.tw-folder-chevron svg\{transition:none\}/);
});

// ---- C2b.6 project page ----
const FID = 'attachment_' + 'c'.repeat(32);
const pageProps = extra => ({ projectId: PID, tasks: projectTasks().slice(0, 2).concat(task('o1', '别的对话')), setTasks() {}, projects: [{ id: PID, name: '蛋白结构' }], setProjects() {},
  setArchivedCount() {}, selection: '', onLeaveOpen() {}, onNavigate() {},
  initial: { project: { id: PID, name: '蛋白结构', instructions: '', version: 3 }, files: [{ id: FID, name: '常数表.md', size: 2048, created_at: '2026-10-26T08:00:00Z' }] }, ...extra });

test('t237 C2b.6: project page head: 「← 对话」, inline-editable h1, 「新对话」 primary with Plus (icon-only label at 393)', () => {
  const { ProjectPage } = require('./ProjectPage.tsx');
  const html = renderToStaticMarkup(React.createElement(ProjectPage, pageProps()));
  assert.match(html, /^<section class="tw-project"><a class="tw-back" href="\/workspace">← 对话<\/a>/);
  assert.match(html, /<div class="tw-project-head"><h1><button type="button" class="tw-title-edit" aria-label="重命名项目">蛋白结构<\/button><\/h1>/);
  assert.match(html, new RegExp(`<a class="tw-primary tw-project-new" href="/workspace\\?new=1&amp;project=${PID}" aria-label="新对话">${svg('plus', 14)}.*?</svg><span class="tw-project-new-label">新对话</span></a>`));
  const { ProjectTitle } = require('./ProjectPage.tsx');
  const editing = renderToStaticMarkup(React.createElement(ProjectTitle, { name: '蛋白结构', editing: true, onEdit() {}, onDone() {} }));
  assert.match(editing, /^<h1><input class="tw-title-input" aria-label="项目名称" maxLength="120" autofocus="" value="蛋白结构"\/><\/h1>$/);
});

test('t237 C2b.6: sections 项目指令 · 项目文件 · 对话; textarea without placeholder, labelled by its h2, counter only above 7000', () => {
  const { ProjectPage, INSTRUCTIONS_MAX, AUTOSAVE_MS, SAVED_FADE_MS, instructionsCounter } = require('./ProjectPage.tsx');
  assert.equal(INSTRUCTIONS_MAX, 8000); assert.equal(AUTOSAVE_MS, 800); assert.equal(SAVED_FADE_MS, 2000);
  assert.equal(instructionsCounter(7000), ''); assert.equal(instructionsCounter(7001), '7001 / 8000');
  const html = renderToStaticMarkup(React.createElement(ProjectPage, pageProps()));
  assert.deepEqual([...html.matchAll(/<section class="tw-project-section"[^>]*><h2[^>]*>([^<]+)<\/h2>/g)].map(match => match[1]), ['项目指令', '项目文件', '对话']);
  assert.match(html, /<h2 id="project-instructions-title">项目指令<\/h2><textarea class="tw-instructions" aria-labelledby="project-instructions-title" maxLength="8000"><\/textarea>/);
  assert.doesNotMatch(html, /placeholder/);
  assert.doesNotMatch(html, /\/ 8000/);
  assert.match(html, /<p class="tw-save-status" role="status"><\/p>/);
  const long = renderToStaticMarkup(React.createElement(ProjectPage, pageProps({ initial: { project: { id: PID, name: 'x', instructions: '字'.repeat(7001), version: 1 }, files: [] } })));
  assert.match(long, /<small class="tw-counter">7001 \/ 8000<\/small>/);
  assert.doesNotMatch(source('./ProjectPage.tsx'), /placeholder=/);
});

test('t237 C2b.6: save status 正在保存… → 已保存 (fades), failure 没能保存 + 重试', () => {
  const { SaveStatus } = require('./ProjectPage.tsx');
  const r = state => renderToStaticMarkup(React.createElement(SaveStatus, { state, onRetry() {} }));
  assert.equal(r('idle'), '<p class="tw-save-status" role="status"></p>');
  assert.equal(r('saving'), '<p class="tw-save-status" role="status">正在保存…</p>');
  assert.equal(r('saved'), '<p class="tw-save-status" role="status">已保存</p>');
  assert.equal(r('faded'), '<p class="tw-save-status is-faded" role="status">已保存</p>');
  assert.equal(r('failed'), '<p class="tw-save-status" role="status"></p><p class="tw-error tw-inline-error">没能保存<button type="button">重试</button></p>');
  const page = source('./ProjectPage.tsx');
  assert.match(page, /onBlur=\{\(\) => void saveInstructions\(\)\}/);
  assert.match(page, /window\.setTimeout\(\(\) => void saveInstructions\(\), AUTOSAVE_MS\)/);
  assert.match(page, /patchProject\(/);
});

test('t237 C2b.6: 项目文件: 上传文件 + 从资料库添加, rows with size/date and 移出项目, no hash; upload states', () => {
  const { ProjectPage, ProjectFileRow, fileMeta } = require('./ProjectPage.tsx');
  assert.equal(fileMeta({ size: 2048, created_at: '2026-10-26T08:00:00Z' }), '2 KB · 2026/10/26');
  assert.equal(fileMeta({ size: 512, created_at: '2026-10-26T08:00:00Z' }), '512 B · 2026/10/26');
  const html = renderToStaticMarkup(React.createElement(ProjectPage, pageProps()));
  assert.match(html, new RegExp(`<label class="tw-upload">${svg('upload', 16)}.*?</svg>上传文件<input class="tw-sr-only" type="file" aria-label="上传文件"/></label><button type="button" class="tw-button" aria-expanded="false">从资料库添加</button>`));
  assert.match(html, /<li class="tw-file tw-project-file"><div class="tw-project-file-text"><span class="tw-project-file-name">常数表\.md<\/span><small>2 KB · 2026\/10\/26<\/small><\/div><button type="button" class="tw-file-remove">移出项目<\/button><button type="button" class="tw-file-more" aria-haspopup="menu" aria-expanded="false" aria-label="文件操作：常数表\.md">/);
  assert.doesNotMatch(html, /sha|[0-9a-f]{40}/i);
  const uploading = renderToStaticMarkup(React.createElement(ProjectFileRow, { upload: { name: '新.md', state: 'uploading' }, onRetry() {} }));
  assert.match(uploading, /新\.md.*正在上传…/);
  const failed = renderToStaticMarkup(React.createElement(ProjectFileRow, { upload: { name: '新.md', state: 'failed' }, onRetry() {} }));
  // t238 C7b S1/S2: same text and 重试 action for a generic failure; the button no longer wraps.
  assert.match(failed, /<span class="tw-error tw-inline-error tw-row-error"><span class="tw-row-error-text">上传失败<\/span><button type="button" class="tw-nowrap">重试<\/button><\/span>/);
  const page = source('./ProjectPage.tsx');
  assert.match(page, /message: '已移出', action: \{ label: COPY\.undo/);
  assert.match(page, /bindProjectFile\(/);
  assert.match(page, /unbindProjectFile\(/);
});

test('t237 C2b.6: 对话 list is .tw-task-row newest first with the 01 「…」; empty states are one line + one action', () => {
  const { ProjectPage } = require('./ProjectPage.tsx');
  const html = renderToStaticMarkup(React.createElement(ProjectPage, pageProps()));
  const convo = html.slice(html.indexOf('>对话</h2>'));
  assert.deepEqual([...convo.matchAll(/<article class="tw-task-row">.*?<a href="\/workspace\?task=[^"]+">([^<]+)<\/a>/g)].map(match => match[1]), ['项目对话0', '项目对话1']);
  assert.match(convo, /aria-label="对话操作：项目对话0"/);
  assert.doesNotMatch(convo, /别的对话/);
  const empty = renderToStaticMarkup(React.createElement(ProjectPage, pageProps({ tasks: [], initial: { project: { id: PID, name: '空', instructions: '', version: 1 }, files: [] } })));
  assert.match(empty, new RegExp(`<p class="tw-empty-hint">还没有对话。</p><a class="tw-button" href="/workspace\\?new=1&amp;project=${PID}">新对话</a>`));
  assert.match(empty, /<p class="tw-empty-hint">还没有文件。<\/p>/);
  assert.doesNotMatch(empty, /tw-banner|说明/);
});

test('t237 C2b.6: route /workspace?project={id} opens the page; CSS for L7/L8/L9/L11 and the 393 file 「…」', () => {
  const workspace = source('./TaskWorkspace.tsx');
  assert.match(workspace, /params\.get\('project'\)/);
  assert.match(workspace, /<ProjectPage key=\{projectView\} projectId=\{projectView\}/);
  assert.match(workspace, /openProject=\{projectView\}/);
  assert.match(rule('.task-workspace .tw-project-section'), /margin-top:28px/);
  assert.match(rule('.task-workspace .tw-title-edit'), /border:1px solid transparent;border-radius:8px/);
  assert.match(css, /\.task-workspace \.tw-title-edit:hover\{border-color:(?:#E7EAE8|var\(--tw-input-edge\)|var\(--tw-selected\))\}/);
  const area = rule('.task-workspace .tw-instructions');
  for (const value of ['min-height:88px', 'max-height:320px', 'border-radius:12px', 'padding:12px', 'font-size:14px', 'line-height:22px', 'overflow-y:auto']) assert.ok(area.includes(value), value);
  assert.match(area, /border:1px solid (?:#E7EAE8|var\(--tw-input-edge\)|var\(--tw-selected\))/);
  assert.match(rule('.tw-counter'), /font-size:12px;color:(?:#6B7280|var\(--tw-muted\)|var\(--tw-label\))/);
  assert.match(rule('.tw-save-status'), /font-size:12px;[^}]*color:(?:#9AA3AE|var\(--tw-weak\))/);
  assert.match(rule('.tw-save-status.is-faded'), /opacity:0/);
  assert.match(rule('.task-workspace .tw-file-remove'), /color:(?:#6B7280|var\(--tw-muted\)|var\(--tw-icon\))/);
  assert.match(css, /\.task-workspace \.tw-file-remove:hover\{color:(?:#1F2933|var\(--tw-ink\))\}/);
  assert.match(rule('.tw-project-head h1'), /white-space:nowrap;overflow:hidden;text-overflow:ellipsis/);
  assert.match(mobile, /\.task-workspace \.tw-project-new\{width:44px;height:44px;padding:0;justify-content:center\}\.tw-project-new-label\{display:none\}/);
  assert.match(mobile, /\.task-workspace \.tw-file-remove\{display:none\}\.task-workspace \.tw-file-more\{display:grid\}/);
});

// ---- C2b.7 draft chip, eyebrow, delete dialog, records copy ----
test('t237 C2b.7: ?new=1&project={id} shows a Folder chip with a remove X; the first POST carries project_id', () => {
  const { DraftProjectChip, draftTaskBody, isSendKey } = require('./ProjectDraft.tsx');
  const html = renderToStaticMarkup(React.createElement(DraftProjectChip, { project: { id: PID, name: '蛋白结构' }, onRemove() {} }));
  assert.match(html, new RegExp(`^<div class="tw-chips tw-draft-project"><span class="tw-chip">${svg('folder', 12)}.*?</svg><span class="tw-chip-name">蛋白结构</span><button type="button" aria-label="项目：蛋白结构，移出">${svg('x', 12)}`));
  assert.deepEqual(draftTaskBody('读论文\n第二行', 'key-1', PID), { title: '读论文', draft: '读论文\n第二行', idempotency_key: 'key-1', project_id: PID });
  assert.deepEqual(draftTaskBody('读论文', 'key-1', ''), { title: '读论文', draft: '读论文', idempotency_key: 'key-1' });
  assert.equal(isSendKey({ key: 'Enter', shiftKey: false, nativeEvent: { isComposing: false } }), true);
  assert.equal(isSendKey({ key: 'Enter', shiftKey: true, nativeEvent: { isComposing: false } }), false);
  assert.equal(isSendKey({ key: 'Enter', shiftKey: false, nativeEvent: { isComposing: true } }), false);
  const workspace = source('./TaskWorkspace.tsx');
  assert.match(workspace, /jsonPost\(draftTaskBody\(draft, creationKey\.current, draftProject\)\)/);
  assert.match(workspace, /<DraftProjectChip project=\{draftProjectInfo\} onRemove=\{removeDraftProject\}>/);
  assert.match(workspace, /searchParams\.delete\('project'\)/);
  assert.match(workspace, /if \(isSendKey\(event\)\) \{ event\.preventDefault\(\); sendDraft\(\); \}/);
});

test('t237 C2b.7 / C5b L7: a project conversation shows breadcrumb 项目 › 任务 (replaces gray eyebrow)', () => {
  const { TaskSession } = require('./TaskSession.tsx');
  const initial = { task: { id: 'utask_fixture', title: '阅读资料', draft: '', version: 1, status: 'draft', project_id: PID }, runs: [], files: [] };
  const html = renderToStaticMarkup(React.createElement(TaskSession, { initial, project: { id: PID, name: '蛋白结构' } }));
  assert.match(html, new RegExp(`class="tw-breadcrumb"[^>]*>.*<a href="/workspace\\?project=${PID}">蛋白结构</a>`));
  assert.match(html, /tw-crumb-current[^>]*>阅读资料</);
  assert.match(html, /tw-title-stack/);
  const plain = renderToStaticMarkup(React.createElement(TaskSession, { initial: { ...initial, task: { ...initial.task, project_id: null } } }));
  assert.doesNotMatch(plain, /tw-breadcrumb/);
  const crumb = rule('.tw-breadcrumb');
  assert.match(crumb, /font-size:13px/); assert.match(crumb, /line-height:20px/); assert.match(crumb, /color:var\(--tw-muted\)/);
});

test('t237 C2b.7: delete dialog is a 400px alertdialog, 取消 first and focused, 删除项目 a plain .tw-button', () => {
  const { DeleteProjectDialog, PROJECT_DELETE_COPY } = require('./DeleteProjectDialog.tsx');
  assert.equal(PROJECT_DELETE_COPY.title('蛋白结构'), '删除项目「蛋白结构」？');
  assert.equal(PROJECT_DELETE_COPY.line(3), '3 个对话会移回对话列表，文件仍在资料里。');
  assert.equal(PROJECT_DELETE_COPY.toast, '已删除项目');
  const html = renderToStaticMarkup(React.createElement(DeleteProjectDialog, { project: { id: PID, name: '蛋白结构' }, count: 3, onCancel() {}, onConfirm() {} }));
  assert.equal(html, '<div class="tw-dialog-backdrop"><div class="tw-dialog" role="alertdialog" aria-modal="true" aria-labelledby="tw-delete-project-title" aria-describedby="tw-delete-project-line">'
    + '<h2 id="tw-delete-project-title">删除项目「蛋白结构」？</h2><p id="tw-delete-project-line">3 个对话会移回对话列表，文件仍在资料里。</p>'
    + '<div class="tw-dialog-actions"><button type="button" class="tw-button">取消</button><button type="button" class="tw-button">删除项目</button></div></div></div>');
  const dialog = source('./DeleteProjectDialog.tsx');
  assert.match(dialog, /cancel\.current\?\.focus\(\)/);
  assert.match(dialog, /event\.key === 'Escape'/);
  assert.match(dialog, /event\.key === 'Tab'/);
  assert.match(rule('.tw-dialog-backdrop'), /background:rgba\(31,41,51,\.24\)/);
  const box = rule('.tw-dialog');
  for (const value of ['width:400px', 'max-width:calc(100vw - 32px)', 'border-radius:14px', 'padding:16px']) assert.ok(box.includes(value), value);
  assert.match(box, /border:1px solid (?:#E7EAE8|var\(--tw-input-edge\)|var\(--tw-selected\)|var\(--tw-edge\))/);
  assert.match(box, /box-shadow:(?:0 10px 30px rgba\(31,41,51,\.12\)|var\(--tw-menu-shadow\))/);
  assert.doesNotMatch(css.slice(css.indexOf('.tw-dialog-backdrop{'), css.indexOf('.tw-dialog-actions{') + 200), /8a2a2a|#f0d4d4|#fff7f7|red/i);
  assert.match(rule('.tw-dialog h2'), /font-size:16px;line-height:24px;font-weight:600/);
  assert.match(rule('.tw-dialog p'), /font-size:14px;line-height:22px;color:(?:#6B7280|var\(--tw-muted\)|var\(--tw-label\))/);
});

test('t237 C2b.7: confirming deletes softly, toasts 已删除项目 · 撤销 (restore) and leaves the project page', () => {
  const workspace = source('./TaskWorkspace.tsx');
  assert.match(workspace, /<DeleteProjectDialog /);
  assert.match(workspace, /onDeleteProject=\{\(project, trigger\) => setDeleting\(\{ project, trigger \}\)\}/);
  assert.match(workspace, /await deleteProject\(project\.id\)/);
  assert.match(workspace, /if \(projectView === project\.id \|\| draftProject === project\.id\) navigate\('\/workspace'\)/);
  assert.match(workspace, /showToast\(\{ message: PROJECT_DELETE_COPY\.toast, action: \{ label: COPY\.undo, run: \(\) => void undoDelete\(project\) \} \}\)/);
  assert.match(workspace, /await restoreProject\(project\.id\)/);
  assert.match(workspace, /trigger\?\.focus\(\)/);
});

test('t237 C2b.7: 资料 records lose 管理项目 and the two lines; the filter reads 项目 / 全部', () => {
  const { WorkspaceRecords } = require('./WorkspaceRecords.tsx');
  for (const projects of [true, false]) {
    const html = renderToStaticMarkup(React.createElement(WorkspaceRecords, { projects }));
    for (const gone of ['管理项目', '项目用于整理已有运行，不是新建任务的前提。', '创建项目不会调用模型或启动执行。', '按项目查看', '全部个人记录']) assert.ok(!html.includes(gone), gone);
    assert.match(html, /<label for="records-project">项目<\/label><select id="records-project"><option value="" selected="">全部<\/option><\/select>/);
  }
  for (const file of ['./TaskWorkspace.tsx', './WorkspaceRecords.tsx']) assert.doesNotMatch(source(file), /管理项目|按项目查看|全部个人记录/, file);
});

test('t237 C2b.7: the 393 drawer toggle is labelled as a 菜单 so the drawer (with the project folders) can be opened', () => {
  assert.match(source('./TaskWorkspace.tsx'), /aria-label=\{menu \? '收起导航菜单' : '展开导航菜单'\}/);
});
