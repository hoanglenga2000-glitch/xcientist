import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/data/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      /* ── Semantic color tokens (Scientific Instrument palette) ── */
      colors: {
        /* Runtime semantic colors. RGB channels are supplied by globals.css. */
        "frame": {
          DEFAULT: "rgb(var(--color-frame) / <alpha-value>)",
          light: "rgb(var(--color-frame-light) / <alpha-value>)",
          lighter: "rgb(var(--color-frame-lighter) / <alpha-value>)",
          border: "rgb(var(--color-frame-border) / <alpha-value>)",
        },
        "surface": {
          DEFAULT: "rgb(var(--color-surface) / <alpha-value>)",
          raised: "rgb(var(--color-surface-raised) / <alpha-value>)",
          sunken: "rgb(var(--color-surface-sunken) / <alpha-value>)",
          overlay: "rgb(var(--color-surface-overlay) / <alpha-value>)",
          paper: "rgb(var(--color-paper) / <alpha-value>)",
        },
        "ink": {
          DEFAULT: "rgb(var(--color-ink) / <alpha-value>)",
          secondary: "rgb(var(--color-ink-secondary) / <alpha-value>)",
          muted: "rgb(var(--color-ink-muted) / <alpha-value>)",
          faint: "rgb(var(--color-ink-faint) / <alpha-value>)",
        },
        "edge": {
          DEFAULT: "rgb(var(--color-edge) / <alpha-value>)",
          light: "rgb(var(--color-edge-light) / <alpha-value>)",
          strong: "rgb(var(--color-edge-strong) / <alpha-value>)",
        },
        "accent": {
          DEFAULT: "rgb(var(--color-accent) / <alpha-value>)",
          light: "rgb(var(--color-accent-light) / <alpha-value>)",
          dark: "rgb(var(--color-accent-dark) / <alpha-value>)",
          muted: "rgb(var(--color-accent-muted) / <alpha-value>)",
          fg: "rgb(var(--color-accent-foreground) / <alpha-value>)",
        },
        "info": {
          DEFAULT: "rgb(var(--color-info) / <alpha-value>)",
          light: "rgb(var(--color-info-light) / <alpha-value>)",
          text: "rgb(var(--color-info-text) / <alpha-value>)",
        },
        "success": {
          DEFAULT: "rgb(var(--color-success) / <alpha-value>)",
          light: "rgb(var(--color-success-light) / <alpha-value>)",
          text: "rgb(var(--color-success-text) / <alpha-value>)",
        },
        "warning": {
          DEFAULT: "rgb(var(--color-warning) / <alpha-value>)",
          light: "rgb(var(--color-warning-light) / <alpha-value>)",
          text: "rgb(var(--color-warning-text) / <alpha-value>)",
        },
        "danger": {
          DEFAULT: "rgb(var(--color-danger) / <alpha-value>)",
          light: "rgb(var(--color-danger-light) / <alpha-value>)",
          text: "rgb(var(--color-danger-text) / <alpha-value>)",
        },
        /* Legacy compat tokens */
        "border": "rgb(var(--color-edge) / <alpha-value>)",
        "background": "rgb(var(--color-surface) / <alpha-value>)",
        "foreground": "rgb(var(--color-ink) / <alpha-value>)",
        "muted": "rgb(var(--color-ink-muted) / <alpha-value>)",
        "navy": "rgb(var(--color-frame) / <alpha-value>)",
        "primary": "rgb(var(--color-accent) / <alpha-value>)",
      },
      /* ── Type ramp ── */
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],   // 10px
        "xs": ["0.75rem", { lineHeight: "1rem" }],         // 12px
        "sm": ["0.875rem", { lineHeight: "1.25rem" }],     // 14px
        "base": ["1rem", { lineHeight: "1.5rem" }],        // 16px
        "lg": ["1.125rem", { lineHeight: "1.5rem" }],      // 18px
        "xl": ["1.25rem", { lineHeight: "1.75rem" }],      // 20px
        "2xl": ["1.5rem", { lineHeight: "2rem" }],         // 24px
      },
      /* ── Spacing scale ── */
      spacing: {
        "4.5": "1.125rem",
        "13": "3.25rem",
        "15": "3.75rem",
      },
      /* ── Border radius ── */
      borderRadius: {
        "sm": "3px",
        "DEFAULT": "6px",
        "md": "8px",
        "lg": "12px",
      },
      /* ── Shadows (hairline + minimal) ── */
      boxShadow: {
        "hairline": "0 0 0 1px rgb(var(--color-edge) / 0.72)",
        "soft": "0 1px 2px rgba(0,0,0,0.18)",
        "raised": "0 12px 36px -24px rgba(0,0,0,0.72)",
        "overlay": "0 18px 54px -22px rgba(0,0,0,0.82)",
        "inner-subtle": "inset 0 1px 0 rgb(var(--color-ink) / 0.05)",
      },
      /* ── Z-index layers ── */
      zIndex: {
        "sidebar": "40",
        "topbar": "30",
        "dropdown": "50",
        "drawer": "60",
        "modal": "70",
        "toast": "80",
        "evidence-rail": "25",
      },
      /* ── Motion ── */
      transitionDuration: {
        "150": "150ms",
        "200": "200ms",
      },
      transitionTimingFunction: {
        "instrument": "cubic-bezier(0.4, 0, 0.2, 1)",
      },
      /* ── Sidebar width ── */
      width: {
        "sidebar": "var(--sidebar-expanded-width)",
        "sidebar-collapsed": "var(--sidebar-collapsed-width)",
      },
      /* ── Keyframes ── */
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "slide-in-right": {
          "0%": { transform: "translateX(100%)" },
          "100%": { transform: "translateX(0)" },
        },
        "slide-in-bottom": {
          "0%": { transform: "translateY(100%)" },
          "100%": { transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 150ms cubic-bezier(0.4, 0, 0.2, 1)",
        "slide-in-right": "slide-in-right 200ms cubic-bezier(0.4, 0, 0.2, 1)",
        "slide-in-bottom": "slide-in-bottom 200ms cubic-bezier(0.4, 0, 0.2, 1)",
      },
    },
  },
  plugins: [],
};

export default config;
