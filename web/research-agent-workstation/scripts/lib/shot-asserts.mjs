// t238 C7c-A3: assertions used by the screenshot QA loop. A shot is only saved under the theme the page
// actually rendered, buttons in one row share a height per breakpoint, hints keep WCAG AA contrast, and the
// model picker is collapsed. (Restored 2026-11-03: the helper was never committed with its test.)

const DARK_LUMINANCE_MAX = 0.2;

function rgb(value) {
  const match = String(value || '').match(/rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/i);
  if (!match) throw new Error(`not an rgb color: ${value}`);
  return match.slice(1, 4).map(Number);
}

function luminance(value) {
  const [r, g, b] = rgb(value).map(channel => {
    const c = channel / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(foreground, background) {
  const a = luminance(foreground); const b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

/** The theme the page shows, from its background; a declared data-theme that disagrees is an error. */
export function assertedTheme({ dataTheme, background }) {
  const observed = luminance(background) < DARK_LUMINANCE_MAX ? 'dark' : 'light';
  if (dataTheme && dataTheme !== observed) throw new Error(`theme mismatch: data-theme=${dataTheme} but background is ${observed}`);
  return observed;
}

export function shotName(name, theme, width) {
  if (theme !== 'dark' && theme !== 'light') throw new Error(`unknown theme: ${theme}`);
  return `${name}-${theme}-${width}.png`;
}

/** Every button in the row has the breakpoint height (36 px desktop, 44 px touch), within half a pixel. */
export function assertHeights(buttons, width) {
  const expected = width <= 480 ? 44 : 36;
  const wrong = buttons.filter(button => Math.abs(button.h - expected) > 0.5);
  if (wrong.length) throw new Error(`button heights @${width} must be ${expected}: ${wrong.map(button => `${button.label}=${button.h}`).join(', ')}`);
  return true;
}

export function assertContrast({ label, color, background }, minimum = 4.5) {
  const ratio = contrast(color, background);
  if (ratio < minimum) throw new Error(`${label} contrast ${ratio.toFixed(2)} < ${minimum}`);
  return ratio;
}

/** aria-expanded of every picker trigger on the shot. */
export function assertPickerCollapsed(states) {
  if (states.some(state => String(state) === 'true')) throw new Error('model picker is expanded in the shot');
  return true;
}
