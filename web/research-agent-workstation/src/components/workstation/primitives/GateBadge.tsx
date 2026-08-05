"use client";

import { type LucideIcon, CheckCircle2, Clock3, Lock, XCircle, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusBadgeV2, type StatusTone } from "./StatusBadge";

export type GateStatus = "approved" | "pending" | "rejected" | "blocked" | "not_required";

const gateToneMap: Record<GateStatus, StatusTone> = {
  approved: "verified",
  pending: "pending",
  rejected: "failed",
  blocked: "blocked",
  not_required: "ready",
};

const gateIconMap: Record<GateStatus, LucideIcon> = {
  approved: CheckCircle2,
  pending: Clock3,
  rejected: XCircle,
  blocked: Lock,
  not_required: CheckCircle2,
};

export function GateBadge({
  status,
  label,
  className,
}: {
  status: GateStatus;
  label?: string;
  className?: string;
}) {
  const tone = gateToneMap[status] ?? "unknown";
  const Icon = gateIconMap[status] ?? AlertCircle;
  const displayText = label ?? (status === "approved" ? "Approved" : status === "pending" ? "Pending" : status === "rejected" ? "Rejected" : status === "blocked" ? "Blocked" : "N/A");

  return (
    <StatusBadgeV2 tone={tone} icon={Icon} className={className}>
      {displayText}
    </StatusBadgeV2>
  );
}

export function ClaimBoundary({ boundary, className }: { boundary?: string | undefined; className?: string }) {
  if (!boundary) return null;
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-sm border border-edge bg-surface-sunken px-1.5 py-0.5 text-2xs font-mono text-ink-secondary", className)}>
      Claim: {boundary}
    </span>
  );
}
