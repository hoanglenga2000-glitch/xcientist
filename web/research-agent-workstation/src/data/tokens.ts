/**
 * EvoMind Research OS — Semantic Design Tokens
 * Visual direction: Scientific Instrument / 科研仪器工作台
 *
 * Runtime colors resolve through CSS variables in app/globals.css. Typed names remain for charts and non-class consumers.
 */

export const tokens = {
  /* ── Color ── */
  color: {
    /* Frame: deep teal-black navigation */
    frame:       'rgb(var(--color-frame))',
    frameLight:  'rgb(var(--color-frame-light))',
    frameBorder: 'rgb(var(--color-frame-border))',

    /* Surface: work area */
    surface:     'rgb(var(--color-surface))',
    surfaceRaised: 'rgb(var(--color-surface-raised))',
    surfaceSunken: 'rgb(var(--color-surface-sunken))',

    /* Ink: text hierarchy */
    ink:         'rgb(var(--color-ink))',
    inkSecondary:'rgb(var(--color-ink-secondary))',
    inkMuted:    'rgb(var(--color-ink-muted))',
    inkFaint:    'rgb(var(--color-ink-faint))',

    /* Edge: borders */
    edge:        'rgb(var(--color-edge))',
    edgeLight:   'rgb(var(--color-edge-light))',
    edgeStrong:  'rgb(var(--color-edge-strong))',

    /* Accent: restrained instrument teal */
    accent:      'rgb(var(--color-accent))',
    accentLight: 'rgb(var(--color-accent-light))',
    accentDark:  'rgb(var(--color-accent-dark))',
    accentMuted: 'rgb(var(--color-accent-muted))',

    /* Status: semantic only, never decorative */
    info:        'rgb(var(--color-info))',
    infoLight:   'rgb(var(--color-info-light))',
    infoText:    'rgb(var(--color-info-text))',

    success:     'rgb(var(--color-success))',
    successLight:'rgb(var(--color-success-light))',
    successText: 'rgb(var(--color-success-text))',

    warning:     'rgb(var(--color-warning))',
    warningLight:'rgb(var(--color-warning-light))',
    warningText: 'rgb(var(--color-warning-text))',

    danger:      'rgb(var(--color-danger))',
    dangerLight: 'rgb(var(--color-danger-light))',
    dangerText:  'rgb(var(--color-danger-text))',
  },

  /* ── Typography ── */
  type: {
    fontFamily: {
      sans: 'system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans SC", sans-serif',
      mono: '"SF Mono", "Cascadia Code", "Fira Code", "JetBrains Mono", ui-monospace, monospace',
    },
    fontSize: {
      '2xs': '0.625rem',  // 10px - tiny labels, badges
      xs:    '0.75rem',   // 12px - auxiliary, mono data
      sm:    '0.875rem',  // 14px - body compact
      base:  '1rem',      // 16px - body default (min mobile)
      lg:    '1.125rem',  // 18px - section title
      xl:    '1.25rem',   // 20px - page title
      '2xl': '1.5rem',    // 24px - hero
    },
    fontWeight: {
      normal:  400,
      medium:  500,
      semibold: 600,
      bold:    700,
    },
    lineHeight: {
      tight:  '1.25',
      normal: '1.5',
      relaxed:'1.625',
    },
  },

  /* ── Spacing ── */
  space: {
    0: '0',
    0.5: '0.125rem',
    1: '0.25rem',
    1.5: '0.375rem',
    2: '0.5rem',
    2.5: '0.625rem',
    3: '0.75rem',
    4: '1rem',
    5: '1.25rem',
    6: '1.5rem',
    8: '2rem',
    10: '2.5rem',
    12: '3rem',
    16: '4rem',
  },

  /* ── Radius ── */
  radius: {
    sm: '3px',
    default: '6px',
    md: '8px',
    lg: '12px',
  },

  /* ── Shadow ── */
  shadow: {
    hairline: '0 0 0 1px rgba(15,23,42,0.06)',
    soft: '0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)',
    raised: '0 4px 12px rgba(15,23,42,0.08), 0 1px 3px rgba(15,23,42,0.04)',
    overlay: '0 12px 40px rgba(15,23,42,0.12), 0 4px 12px rgba(15,23,42,0.06)',
  },

  /* ── Z-index ── */
  z: {
    evidenceRail: 25,
    topbar: 30,
    sidebar: 40,
    dropdown: 50,
    drawer: 60,
    modal: 70,
    toast: 80,
  },

  /* ── Motion ── */
  motion: {
    duration: {
      fast: '150ms',
      normal: '200ms',
      slow: '300ms',
    },
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },

  /* ── Layout ── */
  layout: {
    sidebarWidth: 'var(--sidebar-expanded-width)',
    sidebarCollapsed: 'var(--sidebar-collapsed-width)',
    topbarHeight: 'var(--topbar-height)',
    maxContentWidth: 'var(--content-max-width)',
  },
} as const;

export type DesignTokens = typeof tokens;

/* ── Status tone → token mapping ── */
export const statusColorMap = {
  verified: { bg: tokens.color.successLight, text: tokens.color.successText, border: tokens.color.success, dot: tokens.color.success },
  ready:    { bg: tokens.color.successLight, text: tokens.color.successText, border: tokens.color.success, dot: tokens.color.success },
  running:  { bg: tokens.color.infoLight,    text: tokens.color.infoText,    border: tokens.color.info,    dot: tokens.color.info },
  pending:  { bg: tokens.color.warningLight, text: tokens.color.warningText, border: tokens.color.warning, dot: tokens.color.warning },
  blocked:  { bg: tokens.color.dangerLight,  text: tokens.color.dangerText,  border: tokens.color.danger,  dot: tokens.color.danger },
  failed:   { bg: tokens.color.dangerLight,  text: tokens.color.dangerText,  border: tokens.color.danger,  dot: tokens.color.danger },
  unknown:  { bg: tokens.color.surfaceSunken,text: tokens.color.inkMuted,   border: tokens.color.inkFaint, dot: tokens.color.inkMuted },
  stale:    { bg: tokens.color.warningLight, text: tokens.color.warningText, border: tokens.color.warning, dot: tokens.color.warning },
  draft:    { bg: tokens.color.infoLight,    text: tokens.color.infoText,    border: tokens.color.info,    dot: tokens.color.info },
} as const;

export type StatusTone = keyof typeof statusColorMap;

/* Theme values live in CSS; this alias remains for compatibility. */
export const darkTokens = { color: tokens.color } as const;
