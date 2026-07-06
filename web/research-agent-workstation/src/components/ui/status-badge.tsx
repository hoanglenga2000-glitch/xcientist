import { cn } from "@/lib/utils";

export type StatusTone = "blue" | "green" | "amber" | "red" | "slate" | "purple";

const toneMap: Record<StatusTone, string> = {
  blue: "bg-blue-50 text-blue-700 border-blue-100",
  green: "bg-emerald-50 text-emerald-700 border-emerald-100",
  amber: "bg-amber-50 text-amber-700 border-amber-100",
  red: "bg-red-50 text-red-700 border-red-100",
  slate: "bg-slate-100 text-slate-600 border-slate-200",
  purple: "bg-violet-50 text-violet-700 border-violet-100"
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
        "inline-flex h-[20px] cursor-pointer items-center rounded-[5px] border px-1.5 text-[10px] font-black leading-none transition hover:brightness-95",
        toneMap[tone],
        className
      )}
      data-ui-component="status-badge"
    >
      {children}
    </span>
  );
}
