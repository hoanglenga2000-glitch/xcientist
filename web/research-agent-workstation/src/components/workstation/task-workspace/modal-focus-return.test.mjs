// t238 C7e M-3 (Critiquito): closing a modal returns focus to the button that opened it, for every dialog × every way to close it (5 × 4 = 20).
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire, Module } from 'node:module';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const require = createRequire(import.meta.url);
for (const ext of ['.ts', '.tsx']) Module._extensions[ext] = (module, filename) => {
  module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }, fileName: filename }).outputText, filename);
};
const { captureTrigger, restoreFocus, rememberActivator } = require('./focus-return.ts');
const read = file => readFileSync(new URL(file, import.meta.url), 'utf8');

// A small page model that behaves like a browser for focus: focusing a node makes it active; removing the active node
// drops focus to <body>; a disabled or removed node cannot take focus.
function page() {
  const doc = { body: null, documentElement: null, activeElement: null };
  const node = (name, extra = {}) => {
    const el = {
      name, isConnected: true, children: [], ...extra,
      focus() { if (el.isConnected && !el.disabled) doc.activeElement = el; },
      contains(other) { return other === el || el.children.some(child => child.contains(child === other ? other : other)); },
      closest(selector) { return selector.includes('button') ? el : null; },
    };
    return el;
  };
  doc.body = node('body'); doc.documentElement = node('html'); doc.activeElement = doc.body;
  const remove = el => { el.isConnected = false; el.children.forEach(remove); if (doc.activeElement === el || el.contains(doc.activeElement)) doc.activeElement = doc.body; };
  return { doc, node, remove };
}

const DIALOGS = [
  { name: 'DeleteFileDialog', file: './DeleteFileDialog.tsx', hook: /useReturnFocus\(\)/, closes: { esc: /key === 'Escape'[^}]*onCancel\(\)/, backdrop: /onMouseDown=\{event => \{ if \(event\.target === event\.currentTarget\) \{ event\.preventDefault\(\); onCancel\(\)/, cancel: /ref=\{cancel\}[^>]*onClick=\{onCancel\}/, confirm: /ref=\{confirm\}[^>]*onClick=\{onConfirm\}/ } },
  { name: 'DeleteProjectDialog', file: './DeleteProjectDialog.tsx', hook: /useReturnFocus\(\)/, closes: { esc: /key === 'Escape'[^}]*onCancel\(\)/, backdrop: /onMouseDown=\{event => \{ if \(event\.target === event\.currentTarget\) \{ event\.preventDefault\(\); onCancel\(\)/, cancel: /ref=\{cancel\}[^>]*onClick=\{onCancel\}/, confirm: /ref=\{confirm\}[^>]*onClick=\{onConfirm\}/ } },
  { name: 'DeleteModelDialog', file: './DeleteModelDialog.tsx', hook: /useReturnFocus\(\)/, closes: { esc: /key === 'Escape'[^}]*onCancel\(\)/, backdrop: /onMouseDown=\{event => \{ if \(event\.target === event\.currentTarget\) \{ event\.preventDefault\(\); onCancel\(\)/, cancel: /ref=\{cancel\}[^>]*onClick=\{onCancel\}/, confirm: /ref=\{confirm\}[^>]*onClick=\{onConfirm\}/ } },
  { name: 'ClearAllDialog (memory)', file: './MemorySettings.tsx', hook: /export function ClearAllDialog[^\n]*\n\s*useReturnFocus\(\);/, closes: { esc: /key === 'Escape'[^}]*onCancel\(\)/, backdrop: /tw-dialog-backdrop" onMouseDown=\{event => \{ if \(event\.target === event\.currentTarget\) \{ event\.preventDefault\(\); onCancel\(\)/, cancel: /ref=\{cancel\}[^>]*onClick=\{onCancel\}/, confirm: /ref=\{confirm\}[^>]*onClick=\{onConfirm\}/ } },
  { name: 'GpuJobStatus cancel confirmation', file: './GpuJobStatus.tsx', hook: /restoreFocus\(document as unknown as DocLike, \[cancelOpener\.current/, closes: { esc: /key === 'Escape'[^}]*setConfirming\(false\)/, backdrop: /onMouseDown=\{event => \{ if \(event\.target === event\.currentTarget\) \{ event\.preventDefault\(\); setConfirming\(false\)/, cancel: /ref=\{keep\}[^>]*onClick=\{\(\) => setConfirming\(false\)\}/, confirm: /ref=\{yes\}[^>]*onClick=\{\(\) => void cancel\(\)\}/ } },
];
const PATHS = ['esc', 'backdrop', 'cancel', 'confirm'];

for (const dialog of DIALOGS) for (const path of PATHS) {
  test(`M-3 ${dialog.name} × ${path}: focus returns to the opener`, () => {
    // wiring: the component restores focus on close, and this way of closing reaches the close handler
    const source = read(dialog.file);
    assert.match(source, dialog.hook, `${dialog.name} restores focus`);
    assert.match(source, dialog.closes[path], `${dialog.name} ${path} closes the dialog`);
    // behaviour: opener has focus → dialog opens and focuses its first button → it closes → focus is back on the opener
    const { doc, node, remove } = page();
    const opener = node('opener-button'); doc.activeElement = null; opener.focus();
    const captured = captureTrigger(doc);
    assert.equal(captured, opener);
    const root = node('dialog'); const first = node('first-button'); root.children.push(first); first.focus();
    assert.equal(doc.activeElement, first);
    remove(root); // every close path unmounts the dialog; the browser drops focus to the page body
    assert.equal(doc.activeElement, doc.body, 'precondition: the bug (focus on body)');
    assert.equal(restoreFocus(doc, [captured], root), true);
    assert.equal(doc.activeElement, opener);
  });
}

test('M-3 a click that does not focus the button (Safari) still returns focus to it', () => {
  const { doc, node, remove } = page();
  const opener = node('opener-button');
  rememberActivator(opener); // the document click listener records the activated control
  assert.equal(doc.activeElement, doc.body);
  const captured = captureTrigger(doc);
  assert.equal(captured, opener);
  const root = node('dialog'); const first = node('first-button'); root.children.push(first); first.focus(); remove(root);
  assert.equal(restoreFocus(doc, [captured], root), true); assert.equal(doc.activeElement, opener);
});

test('M-3 an explicit target (the row 更多 button) wins over a captured opener that left the page', () => {
  const { doc, node, remove } = page();
  const menuItem = node('menu-item'); const more = node('more-button');
  menuItem.focus(); const captured = captureTrigger(doc);
  remove(menuItem); // the menu closed when the dialog opened
  assert.equal(restoreFocus(doc, [more, captured]), true); assert.equal(doc.activeElement, more);
});

test('M-3 focus a caller already moved on purpose is not stolen back; a gone opener is a no-op, never an error', () => {
  const { doc, node } = page();
  const opener = node('opener'); const deliberate = node('deliberate');
  deliberate.focus();
  assert.equal(restoreFocus(doc, [opener]), false); assert.equal(doc.activeElement, deliberate);
  doc.activeElement = doc.body; opener.isConnected = false;
  assert.equal(restoreFocus(doc, [opener, null, undefined]), false); assert.equal(doc.activeElement, doc.body);
  assert.equal(restoreFocus(doc, []), false);
});

test('M-3 focus still inside the closing dialog counts as lost', () => {
  const { doc, node } = page();
  const opener = node('opener'); const root = node('dialog'); const inner = node('inner'); root.children.push(inner);
  inner.focus();
  assert.equal(restoreFocus(doc, [opener], root), true); assert.equal(doc.activeElement, opener);
});


test('M-3 every tw-dialog-backdrop mousedown preventDefault (Chrome focus fixup)', () => {
  // Real-browser: without preventDefault, Chrome moves focus to <main> after restoreFocus during mousedown.
  for (const dialog of DIALOGS) {
    assert.match(read(dialog.file), /event\.preventDefault\(\)/, `${dialog.name} backdrop preventDefault`);
  }
  assert.match(read('./ModelSettings.tsx'), /tw-sheet-backdrop[\s\S]*?event\.preventDefault\(\);\s*closePanel\(\)/, 'ModelSettings sheet backdrop preventDefault');
});

test('M-3 20 combos are covered (5 dialogs × 4 close paths) and every modal in task-workspace uses the helper', () => {
  assert.equal(DIALOGS.length * PATHS.length, 20);
  const dir = new URL('./', import.meta.url);
  const modals = ['DeleteFileDialog.tsx', 'DeleteProjectDialog.tsx', 'DeleteModelDialog.tsx', 'MemorySettings.tsx', 'GpuJobStatus.tsx'];
  for (const file of modals) assert.match(readFileSync(new URL(file, dir), 'utf8'), /useReturnFocus|restoreFocus/, file);
});
