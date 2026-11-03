import assert from 'node:assert/strict';
import test from 'node:test';
import { assertContrast, assertHeights, assertPickerCollapsed, assertedTheme, contrast, shotName } from '../../scripts/lib/shot-asserts.mjs';

test('t238 C7c-A3: shots are named from the asserted theme, mismatches fail', () => {
  assert.equal(assertedTheme({ dataTheme: 'dark', background: 'rgb(23, 25, 26)' }), 'dark');
  assert.equal(assertedTheme({ dataTheme: 'light', background: 'rgb(255, 255, 255)' }), 'light');
  assert.equal(assertedTheme({ dataTheme: '', background: 'rgb(23, 25, 26)' }), 'dark');
  // the C7b mislabel: loop said light, page was dark
  assert.throws(() => assertedTheme({ dataTheme: 'light', background: 'rgb(23, 25, 26)' }), /theme mismatch/);
  assert.equal(shotName('data-live-delete-dialog-long-filename', 'dark', 393), 'data-live-delete-dialog-long-filename-dark-393.png');
  assert.throws(() => shotName('x', 'auto', 393));
});

test('t238 C7c-A3: equal button heights per breakpoint (36 @1440, 44 @393)', () => {
  assert.equal(assertHeights([{ label: '删除', h: 36 }, { label: '下载资料', h: 36 }], 1440), true);
  assert.equal(assertHeights([{ label: '删除', h: 44 }, { label: '下载资料', h: 44.4 }], 393), true);
  assert.throws(() => assertHeights([{ label: '删除', h: 40 }, { label: '下载资料', h: 36 }], 1440), /删除=40/);
  assert.throws(() => assertHeights([{ label: '删除', h: 44 }, { label: '下载资料', h: 36 }], 393), /下载资料=36/);
});

test('t238 C7c-A3: hint contrast ≥ 4.5 and collapsed picker are enforced', () => {
  assert.ok(Math.abs(contrast('rgb(107, 114, 128)', 'rgb(242, 244, 243)') - 4.38) < 0.01);
  assert.throws(() => assertContrast({ label: 'hint', color: 'rgb(107, 114, 128)', background: 'rgb(242, 244, 243)' }), /4\.38 < 4\.5/);
  assert.ok(assertContrast({ label: 'hint', color: 'rgb(75, 85, 99)', background: 'rgb(242, 244, 243)' }) > 6.8);
  assert.equal(assertPickerCollapsed(['false', 'false']), true);
  assert.throws(() => assertPickerCollapsed(['false', 'true']), /expanded/);
});
