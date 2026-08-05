"use client";

import { type LucideIcon, CheckCircle2, Clock3, HelpCircle, Loader2, AlertTriangle, XCircle, AlertCircle, FileEdit } from "lucide-react";
import { cn } from "@/lib/utils";

export type StatusTone = "verified" | "ready" | "running" | "pending" | "blocked" | "failed" | "unknown" | "stale" | "draft";

const toneConfig: Record<StatusTone, { bg: string; text: string; border: string; icon: LucideIcon }> = {
  verified: { bg: "bg-success-light", text: "text-success-text", border: "border-success/30", icon: CheckCircle2 },
  ready:    { bg: "bg-success-light", text: "text-success-text", border: "border-success/30", icon: CheckCircle2 },
  running:  { bg: "bg-info-light",    text: "text-info-text",    border: "border-info/30",    icon: Loader2 },
  pending:  { bg: "bg-warning-light", text: "text-warning-text", border: "border-warning/30", icon: Clock3 },
  blocked:  { bg: "bg-danger-light",  text: "text-danger-text",  border: "border-danger/30",  icon: AlertCircle },
  failed:   { bg: "bg-danger-light",  text: "text-danger-text",  border: "border-danger/30",  icon: XCircle },
  unknown:  { bg: "bg-surface-sunken",text: "text-ink-muted",    border: "border-edge",       icon: HelpCircle },
  stale:    { bg: "bg-warning-light", text: "text-warning-text", border: "border-warning/30", icon: AlertTriangle },
  draft:    { bg: "bg-info-light",    text: "text-info-text",    border: "border-info/30",    icon: FileEdit },
};

export function StatusBadgeV2({
  tone,
  children,
  className,
  size = "sm",
  icon: customIcon,
  pulse,
}: {
  tone: StatusTone;
  children: React.ReactNode;
  className?: string;
  size?: "xs" | "sm" | "md";
  icon?: LucideIcon;
  pulse?: boolean;
}) {
  const config = toneConfig[tone] ?? toneConfig.unknown;
  const Icon = customIcon ?? config.icon;
  const sizeClasses = size === "xs" ? "h-6 gap-1 px-1.5 text-xs" : size === "md" ? "h-8 gap-1.5 px-2.5 text-sm" : "h-7 gap-1 px-2 text-xs";
  const iconSize = size === "xs" ? "h-3 w-3" : size === "md" ? "h-3.5 w-3.5" : "h-3 w-3";

  return (
    <span
      className={cn(
        "inline-flex items-center whitespace-nowrap rounded-sm border font-medium",
        config.bg, config.text, config.border,
        sizeClasses,
        pulse && "animate-pulse",
        className
      )}
      role="status"
    >
      <Icon className={cn(iconSize, tone === "running" && "animate-spin")} />
      {children}
    </span>
  );
}

/** Standalone status dot (for sidebar, tables) */
export function StatusDot({ tone, className }: { tone: StatusTone; className?: string }) {
  const colorMap: Record<StatusTone, string> = {
    verified: "bg-success",
    ready: "bg-success",
    running: "bg-info animate-pulse",
    pending: "bg-warning",
    blocked: "bg-danger",
    failed: "bg-danger",
    unknown: "bg-ink-muted",
    stale: "bg-warning",
    draft: "bg-info",
  };
  return <span className={cn("inline-block h-2 w-2 rounded-full shrink-0", colorMap[tone] ?? "bg-ink-muted", className)} />;
}
