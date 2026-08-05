"use client";

import { Clock3, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

/* ── PageHeader: every page gets one ── */
export function PageHeader({
  title,
  subtitle,
  breadcrumb,
  primaryAction,
  secondaryActions,
  className,
}: {
  title: string;
  subtitle?: string;
  breadcrumb?: React.ReactNode;
  primaryAction?: React.ReactNode;
  secondaryActions?: React.ReactNode[];
  className?: string;
}) {
  return (
    <div className={cn("mb-5 flex flex-col gap-2 border-b border-edge/70 pb-4 sm:flex-row sm:items-end sm:justify-between", className)}>
      <div className="min-w-0">
        {breadcrumb && <div className="mb-1 text-xs font-medium text-ink-muted">{breadcrumb}</div>}
        <h1 className="truncate text-xl font-semibold leading-tight text-ink sm:text-2xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm leading-6 text-ink-secondary">{subtitle}</p>}
      </div>
      <div className="flex shrink-0 items-center gap-2 mt-2 sm:mt-0">
        {secondaryActions}
        {primaryAction}
      </div>
    </div>
  );
}

/* ── Panel: a section card with title ── */
export type PanelAccent = "blue" | "green" | "amber" | "red" | "neutral";

export function Panel({
  title,
  description,
  action,
  children,
  className,
  accent,
  compact,
  icon: Icon,
}: {
  title?: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  accent?: PanelAccent;
  compact?: boolean;
  icon?: React.ElementType;
}) {
  const accentBorder =
    accent === "blue" ? "border-l-accent"
    : accent === "green" ? "border-l-success"
    : accent === "amber" ? "border-l-warning"
    : accent === "red" ? "border-l-danger"
    : undefined;
  return (
    <div className={cn(
      "rounded-md border border-edge/90 bg-surface-raised/96 shadow-soft",
      accent && accent !== "neutral" && "border-l-2",
      accentBorder,
      compact ? "p-3" : "p-4",
      className
    )}>
      {(title || description || action) && (
        <div className={cn("flex items-start justify-between gap-3", (description || children) && "mb-3")}>
          <div className="min-w-0">
            {title && (
              <h3 className="flex items-center gap-1.5 text-sm font-semibold text-ink">
                {Icon && <Icon className="h-4 w-4 shrink-0 text-accent-muted" />}
                {title}
              </h3>
            )}
            {description && <p className="mt-0.5 text-xs text-ink-secondary">{description}</p>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

/* ── MetricTile: KPI card ── */
export type MetricTone =
  | "blue" | "green" | "amber" | "red" | "neutral"
  | "verified" | "ready" | "running" | "pending" | "blocked" | "failed" | "unknown" | "stale" | "draft";

const metricToneMap: Record<MetricTone, string> = {
  blue: "text-accent",
  green: "text-success",
  amber: "text-warning",
  red: "text-danger",
  neutral: "text-ink",
  verified: "text-success",
  ready: "text-success",
  running: "text-info",
  pending: "text-warning",
  blocked: "text-danger",
  failed: "text-danger",
  unknown: "text-ink-muted",
  stale: "text-warning",
  draft: "text-info",
};

export function MetricTile({
  label,
  value,
  detail,
  tone,
  icon: Icon,
  className,
}: {
  label: string;
  value: string | number;
  detail?: string;
  tone?: MetricTone;
  icon?: React.ElementType;
  className?: string;
}) {
  const toneMap = metricToneMap;
  return (
    <div className={cn("min-w-0 overflow-hidden border-l border-edge/80 px-3 py-2 first:border-l-0", className)}>
      <div className="flex items-center gap-2 mb-1">
        {Icon && <Icon className={cn("h-4 w-4 shrink-0", tone ? toneMap[tone] : "text-ink-muted")} />}
        <span className="text-xs font-medium text-ink-secondary truncate">{label}</span>
      </div>
      <div className={cn("break-words text-base font-bold leading-tight tabular-nums [overflow-wrap:anywhere] sm:text-lg", tone ? toneMap[tone] : "text-ink")}>{value}</div>
      {detail && <div className="mt-0.5 text-2xs text-ink-muted truncate">{detail}</div>}
    </div>
  );
}

/* ── EmptyState ── */
export function EmptyState({ message, icon: Icon, action, className }: { message: string; icon?: React.ElementType; action?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-col items-center justify-center rounded-md border border-dashed border-edge bg-surface-sunken py-10 text-center", className)}>
      {Icon && <Icon className="mb-2 h-8 w-8 text-ink-faint" />}
      <p className="text-sm text-ink-muted">{message}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

/* ── ErrorState ── */
export function ErrorState({ message, onRetry, className }: { message: string; onRetry?: () => void; className?: string }) {
  return (
    <div className={cn("rounded-md border border-danger/20 bg-danger-light p-4", className)}>
      <p className="text-sm text-danger-text">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-2 text-xs font-semibold text-danger-text underline hover:no-underline">
          Retry
        </button>
      )}
    </div>
  );
}

/* ── CopyablePath: mono path with copy + truncation ── */
export function CopyablePath({ path, className }: { path: string; className?: string }) {
  const copy = () => navigator.clipboard.writeText(path).catch(() => {});
  const display = path.length > 40 ? path.slice(0, 18) + "…/" + path.slice(-18) : path;
  return (
    <span className={cn("inline-flex items-center gap-1 rounded bg-surface-sunken border border-edge px-1.5 py-0.5 font-mono text-xs text-ink-secondary", className)} title={path}>
      <span className="truncate max-w-[260px]">{display}</span>
      <button
        type="button"
        onClick={copy}
        data-ui-action="copy_path"
        data-ui-skip-action="true"
        className="-my-2 -mr-2 flex h-11 w-11 shrink-0 items-center justify-center rounded-md text-ink-muted transition hover:bg-surface-raised hover:text-ink-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        title="Copy path"
        aria-label="Copy path"
      >
        <Copy className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
    </span>
  );
}

/* ── AI label ── */
export function AiLabel({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-sm border border-info/20 bg-info-light px-1.5 py-0.5 text-2xs font-semibold text-info-text", className)}>
      AI / Draft
    </span>
  );
}

/* ── SectionLabel ── */
export function SectionLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("text-xs font-semibold text-ink-muted", className)}>{children}</div>;
}

/* ── Skeleton ── */
export function Skeleton({ className, width, height }: { className?: string; width?: string; height?: string }) {
  return (
    <div
      className={cn("animate-pulse rounded bg-edge-light", className)}
      style={{ width: width ?? "100%", height: height ?? "16px" }}
    />
  );
}

/* ── StaleIndicator ── */
export function StaleIndicator({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1 text-2xs text-warning", className)}>
      <Clock3 className="h-3 w-3" /> Stale
    </span>
  );
}
