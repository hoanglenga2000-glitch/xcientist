import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";

/**
 * DeepEvo Iris kit (`/workspace/deepevo-brand/final`, copied bit-identical into `public/`).
 * Two cuts: the 7-seam master from 48 CSS px up, the bolder 5-seam small cut for 16–47 px.
 * Never scale the master below 48 px.
 */
export const MASTER_MIN_PX = 48;
const MARK_SRC = { master: "/brand/mark.svg", small: "/brand/mark-small.svg" } as const;
const BRAND_NAME = "DeepEvo 深度进化";
/** lockup-horizontal-*.svg viewBox 1303.21 × 420.61 (clear space already inside). */
const LOCKUP_RATIO = 420.61 / 1303.21;

export type MarkCut = keyof typeof MARK_SRC;

export function markCut(size: number): MarkCut {
  return size >= MASTER_MIN_PX ? "master" : "small";
}

export function markSrc(size: number) {
  return MARK_SRC[markCut(size)];
}

/** The kit SVG used as a mask over currentColor, so the mark follows the text colour (black on white / white on black). */
function markStyle(size: number): CSSProperties {
  return { width: size, height: size, ["--deepevo-mark-src" as string]: `url(${markSrc(size)})` };
}

export function DeepEvoMark({ size = MASTER_MIN_PX, label, className }: { size?: number; label?: string; className?: string }) {
  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label || undefined}
      aria-hidden={label ? undefined : true}
      data-brand-mark={markCut(size)}
      className={cn("deepevo-mark", className)}
      style={markStyle(size)}
    />
  );
}

/** Rotate-only loader: one 360°/7 step per 1.8 s (master) or 72° (small cut). `active={false}` stops it and fades it out. */
export function DeepEvoLoader({ size = MASTER_MIN_PX, active = true, className }: { size?: number; active?: boolean; className?: string }) {
  return (
    <span
      aria-hidden="true"
      data-brand-loader
      className={cn("deepevo-mark deepevo-loader", markCut(size) === "small" && "is-small", !active && "is-done", className)}
      style={markStyle(size)}
    />
  );
}

/** Kit horizontal lockup (image wordmark). Black ink in light theme, white ink in dark. Kit minimum width 160 px.
 *  `decorative` silences it inside a control that already carries the brand name (one announcement). */
export function DeepEvoLockup({ width = 168, className, decorative = false }: { width?: number; className?: string; decorative?: boolean }) {
  const height = Math.round(width * LOCKUP_RATIO);
  return (
    <span role={decorative ? undefined : "img"} aria-label={decorative ? undefined : BRAND_NAME} aria-hidden={decorative || undefined}
      data-ui-deepevo-lockup className={cn("deepevo-lockup", className)} style={{ width, height }}>
      <img src="/brand/lockup-horizontal-black.svg" alt="" width={width} height={height} className="deepevo-lockup-light" draggable={false} />
      <img src="/brand/lockup-horizontal-white.svg" alt="" width={width} height={height} className="deepevo-lockup-dark" draggable={false} />
    </span>
  );
}
