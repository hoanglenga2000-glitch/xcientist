import { cn } from "@/lib/utils";

export type StatusTone = "blue" | "green" | "amber" | "red" | "slate" | "purple";

const toneMap: Record<StatusTone, string> = {
  blue: "bg-accent-light text-accent-dark border-accent-light",
  green: "bg-success-light text-success-text border-success/25",
  amber: "bg-warning-light text-warning-text border-warning/25",
  red: "bg-danger-light text-danger-text border-danger/25",
  slate: "bg-surface-sunken text-ink-secondary border-edge",
  purple: "bg-info-light text-info-text border-info/25"
};

export function StatusBadge({
  children,
  tone = "slate",
  className
}: {
  children: React.ReactNode;
  tone?: StatusTone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex min-h-6 max-w-full items-center break-all rounded-[5px] border px-1.5 py-1 text-xs font-medium leading-tight [overflow-wrap:anywhere]",
        toneMap[tone],
        className
      )}
      data-ui-component="status-badge"
    >
      {children}
    </span>
  );
}
