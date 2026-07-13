import {
  ArrowRight,
  CheckCircle2,
  Database,
  FileQuestion,
  GitPullRequest,
  Lightbulb,
  Lock,
  Target,
  Copy
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import { runWorkstationAction } from "@/lib/api/client";
import { artifacts, reproducibility } from "@/data/evidence";
import { missionBrief, validationCurve } from "@/data/missions";
import { cn } from "@/lib/utils";

const iconMap = {
  FileQuestion,
  Target,
  Lightbulb,
  Database,
  GitPullRequest
};

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-xs font-bold uppercase tracking-normal text-primary">{children}</div>;
}

function briefLabel(locale: "zh-CN" | "en-US" | undefined, label: string) {
  const labels: Record<string, string> = {
    "Research Question": "研究问题",
    Objective: "研究目标",
    Hypothesis: "实验假设",
    "Dataset & Metric": "数据集与指标",
    "Current Decision": "当前决策",
    Baseline: "基线模型",
    "Expected Output": "预期输出"
  };
  return locale === "zh-CN" ? labels[label] ?? label : label;
}

function briefValue(locale: "zh-CN" | "en-US" | undefined, label: string, value: string) {
  if (locale !== "zh-CN") return value;
  const values: Record<string, string> = {
    "Research Question": "在 House Prices 数据集上，log 目标变换与梯度提升模型是否能比线性基线显著降低误差？",
    Objective: "建立可复测的强基线，并验证 log-target + GBDT 是否带来稳定收益。",
    Hypothesis: "对目标值做 log1p 变换后，GBDT 模型会明显优于线性回归基线。",
    "Dataset & Metric": "House Prices 数据集，采用 5 折交叉验证 RMSE / RMSLE 指标。",
    "Current Decision": "复核验证结果与数据泄漏风险；完整性 Gate 通过后再批准提交。",
    Baseline: "线性回归 CV: 0.14532",
    "Expected Output": "各折验证中稳定降低 RMSE，并保留证据链。"
  };
  return values[label] ?? value;
}

export function ResearchBrief({ compact = false, locale }: { compact?: boolean; locale?: "zh-CN" | "en-US" }) {
  return (
    <Card className="mb-3 overflow-hidden">
      <CardContent className={cn(
        "grid min-w-0 grid-cols-1 gap-0 overflow-hidden p-0 md:auto-cols-auto md:grid-flow-row md:overflow-visible",
        compact ? "md:grid-cols-3 xl:grid-cols-7" : "md:grid-cols-2 xl:grid-cols-5"
      )}>
        {missionBrief.slice(0, compact ? 5 : 5).map((item, index) => {
          const Icon = iconMap[item.icon as keyof typeof iconMap];
          return (
            <div
              key={item.label}
              className={cn(
                "min-h-[108px] min-w-0 px-5 py-5 md:px-6",
                index > 0 && "border-t border-border md:border-l md:border-t-0",
                "md:border-r md:last:border-r-0"
              )}
            >
              <div className="mb-3 flex items-center gap-3">
                <Icon className="h-4 w-4 shrink-0 text-primary" />
                <span className="min-w-0 text-xs font-bold text-slate-950">{briefLabel(locale, item.label)}</span>
              </div>
              <p className="max-w-[280px] text-xs leading-5 text-slate-700">{briefValue(locale, item.label, item.value)}</p>
            </div>
          );
        })}
        {compact ? (
          <>
            <BriefExtra label={briefLabel(locale, "Baseline")} value={briefValue(locale, "Baseline", "Linear Regression CV: 0.14532")} />
            <BriefExtra label={briefLabel(locale, "Expected Output")} value={briefValue(locale, "Expected Output", "Stable RMSE improvement across folds.")} />
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}

function BriefExtra({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-h-[108px] border-l border-border px-6 py-5">
      <div className="mb-3 text-xs font-bold text-slate-950">{label}</div>
      <p className="text-xs leading-5 text-slate-700">{value}</p>
    </div>
  );
}

export function MetricCurve({ height = 210 }: { height?: number }) {
  const width = 520;
  const padding = { left: 46, right: 16, top: 16, bottom: 30 };
  const xMin = 0;
  const xMax = 2000;
  const yMin = 0.1;
  const yMax = 0.32;
  const x = (value: number) => padding.left + ((value - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  const y = (value: number) => padding.top + ((yMax - value) / (yMax - yMin)) * (height - padding.top - padding.bottom);
  const series = [
    ["fold1", "#F97316"],
    ["fold2", "#A855F7"],
    ["fold3", "#10B981"],
    ["fold4", "#06B6D4"],
    ["fold5", "#64748B"],
    ["mean", "#1D4ED8"]
  ] as const;
  const ticksX = [0, 250, 500, 1000, 1500, 2000];
  const ticksY = [0.1, 0.155, 0.21, 0.265, 0.32];

  return (
    <div className="w-full overflow-hidden" style={{ height }}>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" role="img" aria-label="Validation curve">
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#CBD5E1" />
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#CBD5E1" />
        {ticksY.map((tick) => (
          <g key={tick}>
            <line x1={padding.left} y1={y(tick)} x2={width - padding.right} y2={y(tick)} stroke="#F1F5F9" />
            <text x={padding.left - 8} y={y(tick) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">
              {tick}
            </text>
          </g>
        ))}
        {ticksX.map((tick) => (
          <text key={tick} x={x(tick)} y={height - 8} textAnchor="middle" className="fill-slate-400 text-[11px]">
            {tick}
          </text>
        ))}
        {series.map(([key, color]) => {
          const points = validationCurve.map((item) => `${x(item.iteration)},${y(item[key])}`).join(" ");
          return (
            <polyline
              key={key}
              points={points}
              fill="none"
              stroke={color}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={key === "mean" ? 3 : 1.7}
            />
          );
        })}
      </svg>
    </div>
  );
}

type WorkstationAction = (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;

export function ArtifactList({
  compact = false,
  onAction
}: {
  compact?: boolean;
  onAction?: WorkstationAction;
}) {
  async function runAction(action: string) {
    if (onAction) {
      try {
        await onAction(action);
        return;
      } catch {
        // Fall through to the local API so reused evidence widgets still write an audit trail.
      }
    }
    await runWorkstationAction(action, "house_prices");
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Artifacts</CardTitle>
          {!compact ? <CardDescription>evidence-bound outputs</CardDescription> : null}
        </div>
        <button
          className="text-xs font-bold text-primary"
          onClick={() => void runAction("open_artifact_folder")}
          data-testid="open-artifact-folder"
        >
          Open Folder
        </button>
      </CardHeader>
      <CardContent className="space-y-2">
        {artifacts.slice(0, compact ? 4 : artifacts.length).map((artifact) => (
          <div
            key={artifact.name}
            className="grid grid-cols-[1fr_70px_80px] items-center rounded-md border border-border px-3 py-2 text-xs"
          >
            <span className="font-semibold text-slate-700">{artifact.name}</span>
            <span className="text-slate-500">{artifact.size}</span>
            <StatusBadge tone="green">{artifact.binding}</StatusBadge>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ReproducibilityRecord({
  compact = false,
  onAction,
  locale
}: {
  compact?: boolean;
  onAction?: WorkstationAction;
  locale?: "zh-CN" | "en-US";
}) {
  async function runAction(action: string) {
    if (onAction) {
      try {
        await onAction(action);
        return;
      } catch {
        // Fall through to the local API so reused evidence widgets still write an audit trail.
      }
    }
    await runWorkstationAction(action, "house_prices");
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>{locale === "zh-CN" ? "复现记录" : "Reproducibility Record"}</CardTitle>
          {!compact ? <CardDescription>{locale === "zh-CN" ? "数据、随机种子、环境与指标锁定" : "dataset, seed, env and metric locks"}</CardDescription> : null}
        </div>
        <button
          className="text-xs font-bold text-primary"
          onClick={() => void runAction("view_reproducibility_record")}
          data-testid="view-reproducibility-record"
        >
          {locale === "zh-CN" ? "查看全部" : "View all"}
        </button>
      </CardHeader>
      <CardContent className="space-y-2">
        {reproducibility.slice(0, compact ? 5 : reproducibility.length).map((item) => (
          <div key={item.key} className="grid grid-cols-[120px_1fr_70px] items-center gap-2 text-xs">
            <span className="font-semibold text-slate-600">{item.key}</span>
            <span className="truncate text-slate-700">{item.value}</span>
            <StatusBadge tone="green">{item.status}</StatusBadge>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function HumanGateCard({
  status,
  onReview,
  locale
}: {
  status: "Pending" | "Approved" | "Rejected";
  onReview?: () => void;
  locale?: "zh-CN" | "en-US";
}) {
  const tone: StatusTone = status === "Approved" ? "green" : status === "Rejected" ? "red" : "amber";
  const statusText = locale === "zh-CN"
    ? status === "Approved" ? "已批准" : status === "Rejected" ? "已拒绝" : "待审核"
    : status;
  return (
    <Card className="border-amber-200 bg-amber-50/35">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>{locale === "zh-CN" ? "人工 Gate：Kaggle 提交" : "Human Gate: Kaggle Submission"}</CardTitle>
          <CardDescription>{locale === "zh-CN" ? "自动检查已通过，等待人工确认。" : "All automated checks passed. Awaiting human approval."}</CardDescription>
        </div>
        <StatusBadge tone={tone}>{statusText}</StatusBadge>
      </CardHeader>
      <CardContent className="flex items-center justify-between">
        <p className="max-w-[420px] text-xs leading-5 text-slate-700">
          {locale === "zh-CN"
            ? "官方排行榜提交保持受控，必须先由审核人确认依据链与限制说明。"
            : "Official leaderboard submission remains controlled until a reviewer approves the evidence chain and limitation statement."}
        </p>
        <button
          onClick={onReview}
          className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-4 text-sm font-bold text-white"
          data-testid="review-human-gate"
        >
          {locale === "zh-CN" ? "立即审核" : "Review Now"} <ArrowRight className="h-4 w-4" />
        </button>
      </CardContent>
    </Card>
  );
}

export function LockRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-border bg-white px-3 py-2 text-xs">
      <span className="font-semibold text-slate-600">{label}</span>
      <span className="flex items-center gap-2 text-slate-800">
        {value}
        <Lock className="h-3.5 w-3.5 text-emerald-600" />
      </span>
    </div>
  );
}

export function CheckRow({ label, state = "Passed" }: { label: string; state?: string }) {
  const tone: StatusTone = state === "Pending" ? "amber" : state === "Warning" ? "amber" : "green";
  return (
    <div className="flex items-center justify-between border-b border-border py-2 text-xs last:border-0">
      <span className="flex items-center gap-2 text-slate-700">
        <CheckCircle2 className="h-4 w-4 text-slate-400" />
        {label}
      </span>
      <StatusBadge tone={tone}>{state}</StatusBadge>
    </div>
  );
}

export function ArtifactPathRow({ label, path }: { label: string; path?: string | null }) {
  if (!path) return null;
  const copyToClipboard = () => {
    navigator.clipboard.writeText(path).catch(console.error);
  };
  return (
    <div className="flex flex-col gap-1 border-b border-slate-100 py-2 text-xs last:border-0">
      <div className="font-semibold text-slate-500">{label}</div>
      <div className="flex items-center gap-2 rounded-md bg-slate-50 p-2 border border-slate-200">
        <span className="break-all font-mono text-slate-700 flex-1">{path}</span>
        <button onClick={copyToClipboard} className="text-slate-400 hover:text-slate-600" title="Copy path">
          <Copy className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export function JsonInspector({ data }: { data: unknown }) {
  if (!data) return null;
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950 p-4 overflow-auto">
      <pre className="text-xs leading-5 text-slate-100 font-mono whitespace-pre-wrap break-all">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

export function CodeBlockPanel({ code, emptyMessage = "No code loaded" }: { code?: string | null, emptyMessage?: string }) {
  if (!code) {
    return <RealEmptyState message={emptyMessage} />;
  }
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950 p-4 overflow-auto max-h-[500px]">
      <pre className="text-xs leading-5 text-slate-100 font-mono whitespace-pre-wrap break-all">
        {code}
      </pre>
    </div>
  );
}

export function WorkbenchSection({ title, description, action, children }: { title: string; description: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </div>
        {action && <div>{action}</div>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export function RealEmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-8 text-center text-sm text-slate-500">
      {message}
    </div>
  );
}

export function GateReviewRow({ 
  gate, 
  onApprove, 
  onReject, 
  onRequestEvidence 
}: { 
  gate: { id?: string; gate_type?: string; decision?: string; run_id?: string; evidence?: unknown; artifact?: unknown; created_at?: string; decided_at?: string },
  onApprove?: (id: string) => void,
  onReject?: (id: string) => void,
  onRequestEvidence?: (id: string) => void
}) {
  const isPending = !gate.decision || gate.decision.toLowerCase() === 'pending';
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4 text-xs shadow-sm mb-3">
      <div className="flex justify-between items-start mb-3">
        <div>
          <div className="font-bold text-slate-900 text-sm">{gate.gate_type || 'Unknown Gate'}</div>
          <div className="text-slate-500 mt-1">Run ID: {gate.run_id || 'N/A'}</div>
        </div>
        <StatusBadge tone={gate.decision === 'Approved' ? 'green' : gate.decision === 'Rejected' ? 'red' : 'amber'}>
          {gate.decision || 'Pending'}
        </StatusBadge>
      </div>
      
      {!!gate.evidence && (
        <div className="mb-3">
          <div className="font-semibold text-slate-500 mb-1">Evidence:</div>
          <div className="max-h-32 overflow-auto"><JsonInspector data={gate.evidence} /></div>
        </div>
      )}
      
      {!!gate.artifact && (
        <div className="mb-3">
          <div className="font-semibold text-slate-500 mb-1">Artifact:</div>
          <div className="max-h-32 overflow-auto"><JsonInspector data={gate.artifact} /></div>
        </div>
      )}

      {isPending && (
        <div className="flex gap-2 mt-4 pt-3 border-t border-slate-100">
          <Button size="sm" onClick={() => onApprove?.(gate.id!)}>Approve</Button>
          <Button size="sm" variant="danger" onClick={() => onReject?.(gate.id!)}>Reject</Button>
          <Button size="sm" variant="secondary" onClick={() => onRequestEvidence?.(gate.id!)}>Request Evidence</Button>
        </div>
      )}
    </div>
  );
}
