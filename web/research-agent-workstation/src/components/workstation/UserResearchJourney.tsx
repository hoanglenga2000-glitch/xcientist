"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Cpu,
  Database,
  Download,
  Eye,
  FileText,
  History,
  Images,
  LockKeyhole,
  PackageCheck,
  Play,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";

type JourneyMetrics = {
  prAuc?: number;
  f1?: number;
  recall?: number;
  precision?: number;
};

type JourneyEvolution = {
  parent?: string;
  child?: string;
  beforePrAuc?: number;
  afterPrAuc?: number;
  deltaPrAuc?: number;
};

export type UserResearchJourneyProps = {
  taskId?: string;
  runId?: string;
  status?: string;
  requestObjective?: string;
  completedTasks: number;
  totalTasks: number;
  orchestrationModel?: string;
  localGpuUsed: boolean;
  selectedSolution?: string;
  reviewerStatus?: string;
  claimAuditStatus?: string;
  artifactCount: number;
  metrics: JourneyMetrics;
  evolution: JourneyEvolution;
  runMetrics?: Record<string, unknown>;
  datasetProfile?: Record<string, unknown> | null;
  experimentComparison?: Record<string, unknown> | null;
  hpcRuntime?: Record<string, unknown> | null;
  historicalThresholds?: Record<string, unknown> | null;
  deliverables?: Record<string, unknown> | null;
  privateGrader?: Record<string, unknown> | null;
  events?: Array<Record<string, unknown>>;
};

const USER_REQUEST = "我有一份信用卡交易数据，想知道哪些交易可能有欺诈。请帮我直接分析，并用我看得懂的方式给出结论、报告和可下载文件。计算只用这台电脑，不要提交公开榜单。";

const SIIM_USER_REQUEST = "请分析这批皮肤镜图像和患者信息，自动检查数据质量与泄漏风险，在远程 A800 上比较预处理方案并完成独立离线验证，最后给我专业报告、结果 CSV、代码包和证据包。不要提交公开榜单。";

const siimStages = [
  { title: "理解需求", description: "确认研究目标、算力和四项交付", icon: Sparkles },
  { title: "资源预检", description: "核验 A800、数据和共存门禁", icon: Cpu },
  { title: "数据审计", description: "检查类别、患者和重复图像泄漏", icon: Images },
  { title: "研究设计", description: "冻结患者分组验证和指标合同", icon: LockKeyhole },
  { title: "方案消融", description: "比较五种预处理方案", icon: BarChart3 },
  { title: "完整训练", description: "多模型融合与三种子 OOF", icon: Activity },
  { title: "独立复核", description: "重新计算指标并冻结候选", icon: ShieldCheck },
  { title: "终局评测", description: "冻结后只执行一次离线 grader", icon: CheckCircle2 },
  { title: "成果交付", description: "报告、结果、代码和证据包", icon: PackageCheck },
] as const;

const stages = [
  { title: "听懂需求", description: "确认用户想找出欺诈交易，并整理要交付的文件", icon: Sparkles },
  { title: "看懂数据", description: "检查交易时间、结果标签和数据是否可靠", icon: Database },
  { title: "自动分析", description: "用这台电脑比较可行方案并记录过程", icon: Cpu },
  { title: "再次核对", description: "用更晚时间的交易重新检查结果", icon: ShieldCheck },
  { title: "交付成果", description: "生成可读结论、报告、结果表和证据包", icon: PackageCheck },
] as const;

const downloadRoot = "/demo/credit-card-fraud";

function percent(value: number | undefined, digits = 1) {
  return typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "—";
}

function metric(value: number | undefined, digits = 6) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function asRecords(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
}

function numericArray(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => typeof item === "number" && Number.isFinite(item))
    : [];
}

function numberValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function textValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function statusPassed(value: unknown) {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  return ["passed", "completed", "verified", "review_passed"].includes(normalized);
}

function siimGraderView(privateGrader: Record<string, unknown> | null | undefined) {
  const score = numberValue(privateGrader?.mle_private_grader_score ?? privateGrader?.score);
  const status = textValue(privateGrader?.status)?.toLowerCase();
  if (typeof score === "number") {
    return {
      label: "终局离线评分",
      value: metric(score),
      description: "候选冻结后仅执行一次。",
      statusLabel: "有终局分数",
    };
  }
  if (status === "failed_closed") {
    return {
      label: "终局 grader 记录",
      value: "失败关闭",
      description: "已恰好一次记录；未产生可用私有分数。",
      statusLabel: "失败关闭",
    };
  }
  return {
    label: "终局 grader 记录",
    value: "—",
    description: "候选冻结后仅执行一次。",
    statusLabel: "待记录",
  };
}

function integer(value: number | undefined) {
  return typeof value === "number" ? new Intl.NumberFormat("zh-CN").format(value) : "—";
}

function MetricCard({ label, value, description, tone = "teal" }: { label: string; value: string; description: string; tone?: "teal" | "blue" | "violet" }) {
  const toneClass = tone === "blue"
    ? "border-info/30 bg-info-light/45 text-info-text"
    : tone === "violet"
      ? "border-accent/35 bg-accent-light/45 text-accent-dark"
      : "border-success/35 bg-success-light/45 text-success-text";
  return (
    <div className={`rounded-lg border p-4 ${toneClass}`}>
      <div className="text-xs font-semibold">{label}</div>
      <div className="mt-2 font-mono text-3xl font-black tracking-normal text-ink">{value}</div>
      <div className="mt-2 text-xs leading-5 text-ink-secondary">{description}</div>
    </div>
  );
}

function ComparisonBar({ label, before, after }: { label: string; before?: number; after?: number }) {
  const safeBefore = Math.max(0, Math.min(100, Number(before ?? 0) * 100));
  const safeAfter = Math.max(0, Math.min(100, Number(after ?? 0) * 100));
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-ink">{label}</span>
        <span className="font-mono text-ink-muted">{percent(before)} → <strong className="text-success-text">{percent(after)}</strong></span>
      </div>
      <div className="relative h-8 overflow-hidden rounded-md bg-surface-sunken">
        <div className="absolute inset-y-0 left-0 bg-ink-faint/35" style={{ width: `${safeBefore}%` }} />
        <div className="absolute inset-y-1 left-0 rounded-r bg-success" style={{ width: `${safeAfter}%` }} />
        <div className="absolute inset-0 flex items-center justify-end px-2 text-[10px] font-bold text-ink-muted">第二轮</div>
      </div>
    </div>
  );
}

function DownloadLink({ href, label, filename }: { href: string; label: string; filename: string }) {
  return (
    <a
      href={href}
      download={filename}
      data-ui-action={`user_download_${filename.replace(/[^a-z0-9]+/gi, "_").toLowerCase()}`}
      className="inline-flex h-9 items-center justify-center gap-1.5 rounded-md border border-edge bg-surface-raised px-3 text-xs font-semibold text-ink-secondary transition hover:border-accent hover:text-accent"
    >
      <Download className="h-3.5 w-3.5" />
      {label}
    </a>
  );
}

function FullReport({ open, onClose, props }: { open: boolean; onClose: () => void; props: UserResearchJourneyProps }) {
  useEffect(() => {
    if (!open) return;
    const before = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = before;
    };
  }, [onClose, open]);

  if (!open) return null;
  const { metrics, evolution } = props;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-frame/70 p-3 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="user-report-title">
      <article className="flex h-[94vh] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-edge bg-surface-paper shadow-2xl">
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-edge bg-surface-raised px-5 py-4 sm:px-8">
          <div>
            <div className="text-xs font-bold uppercase tracking-normal text-success-text">EvoMind Research Report</div>
            <h2 id="user-report-title" className="mt-1 text-xl font-black text-ink sm:text-2xl">信用卡交易欺诈检测：完整研究报告</h2>
            <p className="mt-1 text-xs text-ink-muted">独立离线时间验证 · 本机计算 · 未提交公开榜单</p>
          </div>
          <button type="button" data-ui-action="user_close_full_report" onClick={onClose} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-edge text-ink-secondary hover:border-accent hover:text-accent" aria-label="关闭完整报告">
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto" data-ui-report-scroll>
          <div className="mx-auto max-w-5xl space-y-10 px-5 py-8 sm:px-10 sm:py-10">
            <section>
              <div className="mb-3 flex items-center gap-2 text-success-text"><CheckCircle2 className="h-5 w-5" /><span className="text-sm font-bold">研究结论</span></div>
              <h3 className="max-w-4xl text-2xl font-black leading-tight text-ink sm:text-3xl">这台电脑已经完成欺诈分析，并把结论、报告和可下载成果整理好了。</h3>
              <p className="mt-4 max-w-4xl text-sm leading-7 text-ink-secondary">系统最终选择 {props.selectedSolution || "EXP002"}。对真正的欺诈交易，识别率为 {percent(metrics.recall)}；被系统标记为风险的交易中，{percent(metrics.precision)} 确实是欺诈。所有结果均来自独立离线时间验证，并经过独立复核。</p>
            </section>

            <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="真正欺诈识别率" value={percent(metrics.recall)} description="每 100 笔真实欺诈交易，约识别 92 笔。" />
              <MetricCard label="风险标记可信度" value={percent(metrics.precision)} description="每 100 笔被标记交易，约 95 笔确为欺诈。" tone="blue" />
              <MetricCard label="整体识别质量" value={percent(metrics.prAuc, 2)} description="用于衡量系统把高风险交易排在前面的能力。" tone="violet" />
              <MetricCard label="平衡表现" value={percent(metrics.f1)} description="综合考虑漏检与误报后的结果。" />
            </section>

            <section className="grid gap-7 border-t border-edge pt-8 lg:grid-cols-[0.9fr_1.1fr]">
              <div>
                <div className="text-xs font-bold uppercase tracking-normal text-ink-muted">数据范围</div>
                <h3 className="mt-2 text-xl font-black text-ink">用更晚时间的交易做独立检查</h3>
                <dl className="mt-4 space-y-3 text-sm">
                  <div className="flex justify-between gap-4 border-b border-edge-light pb-2"><dt className="text-ink-muted">训练数据</dt><dd className="font-mono font-bold text-ink">1,296,675 行</dd></div>
                  <div className="flex justify-between gap-4 border-b border-edge-light pb-2"><dt className="text-ink-muted">独立时间验证</dt><dd className="font-mono font-bold text-ink">555,719 行</dd></div>
                  <div className="flex justify-between gap-4 border-b border-edge-light pb-2"><dt className="text-ink-muted">真正欺诈样本</dt><dd className="font-mono font-bold text-ink">2,145 行</dd></div>
                  <div className="flex justify-between gap-4"><dt className="text-ink-muted">验证原则</dt><dd className="text-right font-semibold text-ink">更晚时间、没有参与优化</dd></div>
                </dl>
              </div>
              <div className="rounded-lg border border-edge bg-surface-raised p-5">
                <div className="flex items-center justify-between gap-3"><div><div className="text-xs font-bold uppercase tracking-normal text-ink-muted">自动改进</div><h3 className="mt-1 text-lg font-black text-ink">EXP001 → EXP002</h3></div><StatusBadge tone="green">旧版本已保留</StatusBadge></div>
                <div className="mt-6 space-y-5">
                  <ComparisonBar label="整体识别质量" before={evolution.beforePrAuc} after={evolution.afterPrAuc} />
                  <ComparisonBar label="真正欺诈识别率" before={0.9198135198135198} after={metrics.recall} />
                </div>
                <p className="mt-5 text-xs leading-5 text-ink-muted">第二轮只重做受影响的分析步骤；数据范围和独立检查方式保持不变。</p>
              </div>
            </section>

            <section className="border-t border-edge pt-8">
              <div className="text-xs font-bold uppercase tracking-normal text-ink-muted">系统完成了什么</div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {["检查字段、时间范围和结果标签", "确认数据没有提前泄露答案", "比较多种可行分析方案", "确认本机算力足够完成", "用独立时间数据复核", "生成报告与证据清单"].map((item, index) => (
                  <div key={item} className="flex items-start gap-3 rounded-md border border-edge bg-surface-raised p-3 text-sm text-ink-secondary"><span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-success-light text-xs font-black text-success-text">{index + 1}</span>{item}</div>
                ))}
              </div>
            </section>

            <section className="grid gap-5 border-t border-edge pt-8 lg:grid-cols-2">
              <div className="rounded-lg border border-success/35 bg-success-light/35 p-5">
                <div className="flex items-center gap-2 text-success-text"><ShieldCheck className="h-5 w-5" /><h3 className="font-black">复核与证据</h3></div>
                <ul className="mt-4 space-y-2 text-sm text-ink-secondary">
                  <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />独立审核：{statusPassed(props.reviewerStatus) ? "通过" : props.reviewerStatus || "待记录"}</li>
                  <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />声明审计：{statusPassed(props.claimAuditStatus) ? "通过" : props.claimAuditStatus || "待记录"}</li>
                  <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />{props.artifactCount} 个成果文件都有可追溯指纹</li>
                  <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />没有拿最终检查数据反复试答案</li>
                </ul>
              </div>
              <div className="rounded-lg border border-warning/40 bg-warning-light/45 p-5">
                <div className="text-sm font-black text-warning-text">适用边界</div>
                <p className="mt-3 text-sm leading-6 text-ink-secondary">这些数字是独立离线时间验证结果，用于判断本次研究方案的可靠性。它们不是公开榜单成绩、排名或奖牌。正式部署前还应根据业务成本选择报警阈值，并持续监控新数据分布。</p>
              </div>
            </section>

            <section className="flex flex-col gap-4 border-t border-edge py-8 sm:flex-row sm:items-center sm:justify-between">
              <div><div className="text-sm font-black text-ink">下载本次研究成果</div><div className="mt-1 text-xs text-ink-muted">报告、结果摘要和可核验的证据包均已准备好。</div></div>
              <div className="flex flex-wrap gap-2">
                <DownloadLink href={`${downloadRoot}/evomind-credit-card-fraud-report.pdf`} filename="evomind-credit-card-fraud-report.pdf" label="下载 PDF 报告" />
                <DownloadLink href={`${downloadRoot}/evomind-credit-card-fraud-results.csv`} filename="evomind-credit-card-fraud-results.csv" label="下载结果摘要" />
                <DownloadLink href={`${downloadRoot}/evomind-credit-card-fraud-evidence.zip`} filename="evomind-credit-card-fraud-evidence.zip" label="下载证据包" />
              </div>
            </section>
          </div>
        </div>
      </article>
    </div>
  );
}

function DisabledDownload({ label }: { label: string }) {
  return (
    <button type="button" disabled className="inline-flex h-9 cursor-not-allowed items-center gap-1.5 rounded-md border border-edge bg-surface-sunken px-3 text-xs font-semibold text-ink-muted opacity-70">
      <Clock3 className="h-3.5 w-3.5" />
      {label}
    </button>
  );
}

function SiimCurveChart({
  title,
  x,
  y,
  reference = false,
  tone = "accent",
}: {
  title: string;
  x: number[];
  y: number[];
  reference?: boolean;
  tone?: "accent" | "success";
}) {
  const count = Math.min(x.length, y.length);
  const points = Array.from({ length: count }, (_, index) => {
    const px = 34 + Math.max(0, Math.min(1, x[index])) * 272;
    const py = 132 - Math.max(0, Math.min(1, y[index])) * 108;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(" ");
  const stroke = tone === "success"
    ? "rgb(var(--color-success))"
    : "rgb(var(--color-info))";
  return (
    <figure className="min-w-0" aria-label={title}>
      <figcaption className="text-sm font-black text-ink">{title}</figcaption>
      <svg viewBox="0 0 330 160" className="mt-3 h-44 w-full" role="img" aria-label={`${title}，由同一 Run 的 OOF 预测重新计算`}>
        <line x1="34" y1="132" x2="306" y2="132" stroke="currentColor" className="text-edge" />
        <line x1="34" y1="24" x2="34" y2="132" stroke="currentColor" className="text-edge" />
        {[0.25, 0.5, 0.75, 1].map((tick) => <line key={tick} x1="34" y1={132 - tick * 108} x2="306" y2={132 - tick * 108} stroke="currentColor" className="text-edge-light" strokeDasharray="3 4" />)}
        {reference && <line x1="34" y1="132" x2="306" y2="24" stroke="currentColor" className="text-ink-faint" strokeDasharray="5 5" />}
        {count > 1 ? <polyline points={points} fill="none" stroke={stroke} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" /> : <text x="170" y="82" textAnchor="middle" className="fill-ink-muted text-xs">等待真实曲线</text>}
        <text x="30" y="148" className="fill-ink-muted text-[9px]">0</text>
        <text x="299" y="148" className="fill-ink-muted text-[9px]">1</text>
        <text x="18" y="28" className="fill-ink-muted text-[9px]">1</text>
      </svg>
    </figure>
  );
}

function SiimMetricBars({
  title,
  records,
}: {
  title: string;
  records: Array<{ label: string; value?: number }>;
}) {
  return (
    <section className="min-w-0">
      <h3 className="text-sm font-black text-ink">{title}</h3>
      <div className="mt-4 space-y-3">
        {records.length ? records.map((record) => {
          const width = Math.max(0, Math.min(100, (record.value ?? 0) * 100));
          return <div key={record.label}><div className="flex items-center justify-between gap-3 text-xs"><span className="truncate font-semibold text-ink-secondary">{record.label}</span><span className="shrink-0 font-mono font-bold text-ink">{metric(record.value, 4)}</span></div><div className="mt-1.5 h-2 overflow-hidden rounded-sm bg-surface-sunken"><div className="h-full bg-accent" style={{ width: `${width}%` }} /></div></div>;
        }) : <div className="text-xs text-ink-muted">等待真实分折结果。</div>}
      </div>
    </section>
  );
}

function SiimEvolutionTrail({ comparison, metrics, compact = false }: { comparison?: Record<string, unknown> | null; metrics?: Record<string, unknown> | null; compact?: boolean }) {
  const profiles = asRecords(comparison?.profiles);
  const profileName = (profile: Record<string, unknown>) => textValue(profile.profile ?? profile.name) ?? "未命名方案";
  const profileScore = (profile?: Record<string, unknown>) => numberValue(profile?.mean_auc ?? profile?.roc_auc ?? profile?.oof_auc);
  const selectedName = textValue(comparison?.selected_profile) ?? "等待真实消融结果";
  const rawProfile = profiles.find((profile) => profileName(profile) === "raw_multiview_v1") ?? profiles[0];
  const selectedProfile = profiles.find((profile) => profileName(profile) === selectedName);
  const components = asRecord(metrics?.model_components);
  const fusionAuc = numberValue(asRecord(components.image_metadata_fusion).roc_auc);
  const finalAuc = numberValue(metrics?.roc_auc ?? asRecord(components.final_ensemble).roc_auc);
  const selectedRaw = ["raw_multiview_v1", "raw_multiview"].includes(selectedName);
  const rounds = [
    { id: "R1", title: "建立可复核基线", detail: "原始多视图图像，固定患者与重复内容分组。", value: profileScore(rawProfile), state: "基线" },
    { id: "R2", title: "自动消融与选择", detail: selectedRaw ? "候选改动未达到稳定门槛，系统主动保留基线。" : `五种候选比较后保留 ${selectedName}。`, value: profileScore(selectedProfile), state: selectedRaw ? "拒绝无效改动" : "通过门禁" },
    { id: "R3", title: "图像与元数据融合", detail: "融合全图、病灶视图与患者元数据，不复制少数类。", value: fusionAuc, state: "多源融合" },
    { id: "R4", title: "三种子聚合并冻结", detail: "种子 43、44、45 独立训练后聚合，复核通过才冻结候选。", value: finalAuc, state: "终局候选" },
  ];
  return (
    <section className={compact ? "border-t border-edge px-5 py-6 sm:px-7 lg:px-9" : "border-t border-edge pt-7"} data-ui-siim-evolution>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><div className="text-xs font-bold text-accent">EvoMind 多轮进化轨迹</div><h3 className="mt-1 text-lg font-black text-ink">每轮读取上一轮证据，只保留通过门禁的变化</h3></div>
        <StatusBadge tone="blue">同一 Run · 4 轮</StatusBadge>
      </div>
      <p className="mt-2 max-w-4xl text-xs leading-5 text-ink-muted">这不是重复重跑。系统用相同分组验证比较候选；没有稳定增益的改动会被拒绝，融合与多种子聚合也必须保留完整证据。</p>
      <div className="mt-4 grid gap-px overflow-hidden rounded-md border border-edge bg-edge sm:grid-cols-2 xl:grid-cols-4">
        {rounds.map((round) => <div key={round.id} data-ui-siim-round={round.id} className="min-h-36 bg-surface-raised p-4"><div className="flex items-center justify-between gap-2"><span className="font-mono text-xs font-black text-accent">{round.id}</span><span className="text-[10px] font-bold text-ink-muted">{round.state}</span></div><div className="mt-3 text-sm font-black text-ink">{round.title}</div><div className="mt-2 font-mono text-xl font-black text-ink">{metric(round.value)}</div><p className="mt-2 text-xs leading-5 text-ink-muted">{round.detail}</p></div>)}
      </div>
    </section>
  );
}

function SiimFullReport({ open, onClose, props }: { open: boolean; onClose: () => void; props: UserResearchJourneyProps }) {
  useEffect(() => {
    if (!open) return;
    const before = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = before;
    };
  }, [onClose, open]);

  if (!open) return null;
  const metrics = props.runMetrics ?? {};
  const datasetCounts = asRecord(props.datasetProfile?.counts);
  const dataset = Object.keys(datasetCounts).length ? datasetCounts : props.datasetProfile ?? {};
  const trainImages = numberValue(dataset.train_images ?? dataset.train_rows);
  const testImages = numberValue(dataset.test_images ?? dataset.test_rows);
  const positives = numberValue(dataset.positive_rows);
  const declaredPositiveRate = numberValue(props.datasetProfile?.positive_rate);
  const positiveRate = declaredPositiveRate ?? (
    typeof positives === "number" && typeof trainImages === "number" && trainImages > 0
      ? positives / trainImages
      : undefined
  );
  const launchDecision = textValue(props.hpcRuntime?.launch_decision)?.toUpperCase();
  const rocAuc = numberValue(metrics.roc_auc);
  const prAuc = numberValue(metrics.pr_auc);
  const brier = numberValue(metrics.brier);
  const graderView = siimGraderView(props.privateGrader);
  const confidence = asRecord(metrics.patient_grouped_bootstrap_roc_auc_95ci);
  const fixedThreshold = asRecord(metrics.fixed_oof_threshold_metrics);
  const calibration = asRecord(metrics.calibration_curve);
  const meanPredicted = Array.isArray(calibration.mean_predicted_probability) ? calibration.mean_predicted_probability : [];
  const fractionPositive = Array.isArray(calibration.fraction_positive) ? calibration.fraction_positive : [];
  const profiles = asRecords(props.experimentComparison?.profiles);
  const rocCurve = asRecord(metrics.roc_curve);
  const prCurve = asRecord(metrics.precision_recall_curve);
  const foldMetrics = asRecords(metrics.fold_metrics).map((record, index) => ({
    label: `外层折 ${String(numberValue(record.fold) ?? index)}`,
    value: numberValue(record.roc_auc),
  }));
  const components = asRecord(metrics.model_components);
  const componentLabels: Array<[string, string]> = [
    ["pure_image", "ConvNeXt-Small 全图"],
    ["lesion_focus", "EfficientNetV2-S 病灶"],
    ["image_metadata_fusion", "图像与患者元数据融合"],
    ["metadata_catboost", "CatBoost 元数据"],
    ["final_ensemble", "最终融合"],
  ];
  const componentMetrics = componentLabels.map(([key, label]) => ({
    label,
    value: numberValue(asRecord(components[key]).roc_auc),
  })).filter((record) => typeof record.value === "number");
  const hasResults = typeof rocAuc === "number";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-frame/70 p-3 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="siim-report-title" data-ui-siim-report-modal>
      <article className="flex h-[94vh] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-edge bg-surface-raised shadow-2xl" data-ui-siim-report>
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-edge bg-surface-raised px-5 py-4 sm:px-8">
          <div>
            <div className="text-xs font-bold text-success-text">EvoMind 医学影像研究报告</div>
            <h2 id="siim-report-title" className="mt-1 text-xl font-black text-ink sm:text-2xl">SIIM-ISIC 黑色素瘤研究基准</h2>
            <p className="mt-1 text-xs text-ink-muted">患者与重复内容分组验证 · 离线评测 · 未提交公开榜单</p>
          </div>
          <button type="button" onClick={onClose} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-edge text-ink-secondary hover:border-accent hover:text-accent" aria-label="关闭完整报告">
            <X className="h-4 w-4" />
          </button>
        </header>
        <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto" data-ui-siim-report-scroll>
          <div className="mx-auto max-w-5xl space-y-9 px-5 py-8 sm:px-10">
            {!hasResults ? (
              <section className="border-l-4 border-warning bg-warning-light/45 px-5 py-4">
                <div className="flex items-center gap-2 font-black text-warning-text"><AlertTriangle className="h-5 w-5" />报告正在等待真实实验完成</div>
                <p className="mt-2 text-sm leading-6 text-ink-secondary">{launchDecision === "GO" ? "A800 资源门禁已通过，同一 Run 正在执行数据审计、方案消融和完整训练。真实指标、审核结论和下载包只会在证据齐全后出现。" : "当前 Run 已完成需求绑定、隔离运行时和数据准备，正在等待 A800 资源门禁放行。这里不会提前展示分数或生成下载包。"}</p>
              </section>
            ) : (
              <>
                <section data-ui-siim-report-conclusion>
                  <div className="text-xs font-bold text-success-text">独立离线结论</div>
                  <h3 className="mt-2 text-2xl font-black leading-tight text-ink">系统已完成患者分组验证、模型融合与独立复核。</h3>
                  <p className="mt-3 text-sm leading-7 text-ink-secondary">主指标 ROC-AUC 为 {metric(rocAuc)}，95% 置信区间为 {metric(numberValue(confidence.lower), 4)}–{metric(numberValue(confidence.upper), 4)}。这些结果用于医学影像研究基准，不代表临床诊断能力。</p>
                </section>
                <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" data-ui-siim-report-metrics>
                  <MetricCard label="患者分组 ROC-AUC" value={metric(rocAuc)} description="比赛主指标，越高越好。" />
                  <MetricCard label="PR-AUC" value={metric(prAuc)} description="反映极少数恶性样本的排序质量。" tone="blue" />
                  <MetricCard label="Brier" value={metric(brier)} description="概率校准误差，越低越好。" tone="violet" />
                  <MetricCard label={graderView.label} value={graderView.value} description={graderView.description} />
                </section>
              </>
            )}

            {hasResults && (
              <>
                <SiimEvolutionTrail comparison={props.experimentComparison} metrics={metrics} />
                <section className="grid gap-8 border-t border-edge pt-7 lg:grid-cols-2" data-ui-siim-report-curves>
                  <SiimCurveChart title="ROC 曲线" x={numericArray(rocCurve.false_positive_rate)} y={numericArray(rocCurve.true_positive_rate)} reference />
                  <SiimCurveChart title="Precision-Recall 曲线" x={numericArray(prCurve.recall)} y={numericArray(prCurve.precision)} tone="success" />
                </section>
                <section className="grid gap-9 border-t border-edge pt-7 lg:grid-cols-2" data-ui-siim-report-stability>
                  <SiimMetricBars title="五折稳定性 · ROC-AUC" records={foldMetrics} />
                  <SiimMetricBars title="模型与元数据融合贡献 · ROC-AUC" records={componentMetrics} />
                </section>
              </>
            )}

            <section className="grid gap-7 border-t border-edge pt-7 lg:grid-cols-2" data-ui-siim-report-validation>
              <div>
                <div className="text-xs font-bold text-ink-muted">数据与验证</div>
                <dl className="mt-4 space-y-3 text-sm">
                  <div className="flex justify-between border-b border-edge-light pb-2"><dt className="text-ink-muted">训练图像</dt><dd className="font-mono font-bold text-ink">{integer(trainImages)}</dd></div>
                  <div className="flex justify-between border-b border-edge-light pb-2"><dt className="text-ink-muted">测试图像</dt><dd className="font-mono font-bold text-ink">{integer(testImages)}</dd></div>
                  <div className="flex justify-between border-b border-edge-light pb-2"><dt className="text-ink-muted">恶性样本</dt><dd className="font-mono font-bold text-ink">{integer(positives)}（{percent(positiveRate, 2)}）</dd></div>
                  <div className="flex justify-between"><dt className="text-ink-muted">分组原则</dt><dd className="text-right font-semibold text-ink">患者与重复内容零交叉</dd></div>
                </dl>
              </div>
              <div>
                <div className="text-xs font-bold text-ink-muted">固定 OOF 阈值</div>
                <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-4 text-sm">
                  <div><div className="text-ink-muted">敏感度</div><div className="mt-1 font-mono text-lg font-black text-ink">{percent(numberValue(fixedThreshold.sensitivity))}</div></div>
                  <div><div className="text-ink-muted">特异度</div><div className="mt-1 font-mono text-lg font-black text-ink">{percent(numberValue(fixedThreshold.specificity))}</div></div>
                  <div><div className="text-ink-muted">精确率</div><div className="mt-1 font-mono text-lg font-black text-ink">{percent(numberValue(fixedThreshold.precision))}</div></div>
                  <div><div className="text-ink-muted">阴性预测值</div><div className="mt-1 font-mono text-lg font-black text-ink">{percent(numberValue(fixedThreshold.negative_predictive_value))}</div></div>
                </div>
              </div>
            </section>

            {hasResults && meanPredicted.length > 0 && meanPredicted.length === fractionPositive.length && (
              <section className="border-t border-edge pt-7" data-ui-siim-report-calibration>
                <div className="flex items-end justify-between gap-3"><div><div className="text-xs font-bold text-ink-muted">校准曲线</div><h3 className="mt-1 text-lg font-black text-ink">预测概率与实际阳性率</h3></div><span className="text-xs text-ink-muted">{meanPredicted.length} 个等频区间</span></div>
                <div className="mt-5 grid h-44 items-end gap-2 border-b border-l border-edge px-3 pt-3" style={{ gridTemplateColumns: `repeat(${meanPredicted.length}, minmax(0, 1fr))` }}>
                  {meanPredicted.map((value, index) => {
                    const predicted = numberValue(value) ?? 0;
                    const observed = numberValue(fractionPositive[index]) ?? 0;
                    return <div key={index} className="flex h-full items-end justify-center gap-0.5" title={`预测 ${percent(predicted)} / 实际 ${percent(observed)}`}><span className="w-2 bg-accent" style={{ height: `${Math.max(2, predicted * 100)}%` }} /><span className="w-2 bg-success" style={{ height: `${Math.max(2, observed * 100)}%` }} /></div>;
                  })}
                </div>
                <div className="mt-2 flex gap-5 text-xs text-ink-muted"><span><i className="mr-1 inline-block h-2 w-2 bg-accent" />预测概率</span><span><i className="mr-1 inline-block h-2 w-2 bg-success" />实际阳性率</span></div>
              </section>
            )}

            <section className="border-t border-edge pt-7" data-ui-siim-report-preprocessing>
              <div className="text-xs font-bold text-ink-muted">预处理方案比较</div>
              <div className="mt-4 divide-y divide-edge rounded-md border border-edge">
                {(profiles.length ? profiles : [{ name: "等待真实消融结果", status: "pending" }]).map((profile, index) => {
                  const name = textValue(profile.profile ?? profile.name) ?? `方案 ${index + 1}`;
                  const score = numberValue(profile.mean_auc ?? profile.roc_auc ?? profile.oof_auc);
                  return <div key={`${name}-${index}`} className="flex items-center justify-between gap-4 px-4 py-3 text-sm"><span className="font-semibold text-ink">{name}</span><span className="font-mono text-ink-secondary">{metric(score)}</span></div>;
                })}
              </div>
            </section>

            <section className="grid gap-5 border-t border-edge pt-7 lg:grid-cols-2" data-ui-siim-report-audit>
              <div className="border-l-4 border-success bg-success-light/30 px-4 py-3"><div className="font-black text-success-text">Independent Review</div><p className="mt-2 text-sm text-ink-secondary">{statusPassed(props.reviewerStatus) ? "独立复核已通过，候选哈希已冻结。" : "等待训练完成后独立复核。"}</p></div>
              <div className="border-l-4 border-info bg-info-light/30 px-4 py-3"><div className="font-black text-info-text">Claim Audit</div><p className="mt-2 text-sm text-ink-secondary">{statusPassed(props.claimAuditStatus) ? "声明审计已通过。" : "不声明公开排名、奖牌或临床诊断能力。"}</p></div>
            </section>
          </div>
        </div>
      </article>
    </div>
  );
}

function SiimResearchJourney(props: UserResearchJourneyProps) {
  const [reportOpen, setReportOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const metrics = props.runMetrics ?? {};
  const datasetCounts = asRecord(props.datasetProfile?.counts);
  const dataset = Object.keys(datasetCounts).length ? datasetCounts : props.datasetProfile ?? {};
  const files = numberValue(dataset.files ?? dataset.file_count);
  const trainImages = numberValue(dataset.train_images ?? dataset.train_rows);
  const positives = numberValue(dataset.positive_rows);
  const patients = numberValue(dataset.patients);
  const declaredPositiveRate = numberValue(props.datasetProfile?.positive_rate);
  const positiveRate = declaredPositiveRate ?? (
    typeof positives === "number" && typeof trainImages === "number" && trainImages > 0
      ? positives / trainImages
      : undefined
  );
  const rocAuc = numberValue(metrics.roc_auc);
  const prAuc = numberValue(metrics.pr_auc);
  const brier = numberValue(metrics.brier);
  const graderView = siimGraderView(props.privateGrader);
  const hpcGpu = asRecord(props.hpcRuntime?.gpu);
  const trainingProgress = asRecord(props.hpcRuntime?.training_progress);
  const telemetrySummary = asRecord(props.hpcRuntime?.telemetry_summary);
  const completedFormalSeeds = numberValue(trainingProgress.completed_formal_seeds);
  const totalFormalSeeds = numberValue(trainingProgress.total_formal_seeds);
  const completedOuterFolds = numberValue(trainingProgress.completed_outer_folds);
  const totalOuterFolds = numberValue(trainingProgress.total_outer_folds);
  const selectedEpochMin = numberValue(trainingProgress.selected_epoch_min);
  const selectedEpochMax = numberValue(trainingProgress.selected_epoch_max);
  const peakModelMemory = numberValue(trainingProgress.peak_model_memory_allocated_mib);
  const telemetrySamples = numberValue(telemetrySummary.sample_count);
  const deliverableFiles = asRecords(props.deliverables?.files);
  const deliverablesReady = props.deliverables?.status === "ready" && deliverableFiles.length === 4;
  const runComplete = props.status === "completed" && deliverablesReady;
  const launchDecision = textValue(props.hpcRuntime?.launch_decision)?.toUpperCase();
  const waiting = launchDecision === "HOLD" || (props.status === "needs_continuation" && launchDecision !== "GO");
  const statusTone = runComplete ? "green" : waiting ? "amber" : props.status === "failed" ? "red" : "blue";
  const statusLabel = runComplete ? "研究与交付已完成" : waiting ? "等待 A800 安全资源" : "研究正在进行";
  const latestEvents = (props.events ?? []).slice(-6);
  const threshold = props.historicalThresholds ?? {};
  const downloadByName = new Map(deliverableFiles.map((file) => [String(file.name ?? ""), file]));

  return (
    <section className="overflow-hidden rounded-lg border border-edge bg-surface-raised shadow-raised" data-ui-user-journey data-ui-siim-journey>
      <div className="grid gap-7 px-5 py-6 sm:px-7 lg:grid-cols-[minmax(0,1.2fr)_minmax(310px,0.8fr)] lg:px-9 lg:py-8" data-ui-siim-overview>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2"><StatusBadge tone={statusTone}>{statusLabel}</StatusBadge><span className="text-xs font-semibold text-ink-muted">医学影像研究基准 · 普通用户视图</span></div>
          <h2 className="mt-4 max-w-3xl text-2xl font-black leading-tight text-ink sm:text-3xl">一句话交代研究问题，系统负责数据、训练、复核与交付</h2>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-ink-secondary">你不需要进入终端或选择模型。EvoMind 会在同一个 Run 中检查患者泄漏、比较预处理方案、调用 A800 完成训练，并把报告、结果、代码和证据整理成可下载文件。</p>
          <div className="mt-5 border-l-4 border-accent bg-accent-light/30 px-4 py-3">
            <div className="text-[11px] font-bold text-ink-muted">本次一句话需求</div>
            <p className="mt-2 text-sm font-medium leading-7 text-ink">{props.requestObjective || SIIM_USER_REQUEST}</p>
          </div>
          <div className="mt-5 flex flex-wrap gap-2">
            <Button onClick={() => setReportOpen(true)} data-ui-action="siim_open_report"><FileText className="h-4 w-4" />查看专业报告</Button>
            <Button variant="secondary" onClick={() => setEvidenceOpen((value) => !value)} data-ui-action="siim_toggle_evidence"><ShieldCheck className="h-4 w-4" />{evidenceOpen ? "收起证据" : "查看运行证据"}</Button>
          </div>
        </div>

        <div className="border-l border-edge pl-0 lg:pl-6" data-ui-siim-dataset>
          <div className="flex items-center justify-between gap-3"><div className="text-sm font-black text-ink">本次研究范围</div><span className="font-mono text-xs text-ink-muted">{Math.min(props.completedTasks, props.totalTasks)}/{props.totalTasks || 9}</span></div>
          <div className="mt-4 grid grid-cols-2 gap-x-5 gap-y-4">
            <div><Database className="h-4 w-4 text-accent" /><div className="mt-2 text-xl font-black text-ink">{integer(files)}</div><div className="text-xs text-ink-muted">数据文件</div></div>
            <div><AlertTriangle className="h-4 w-4 text-warning-text" /><div className="mt-2 text-xl font-black text-ink">{percent(positiveRate, 2)}</div><div className="text-xs text-ink-muted">恶性样本占比</div></div>
            <div><Eye className="h-4 w-4 text-accent" /><div className="mt-2 text-xl font-black text-ink">{integer(patients)}</div><div className="text-xs text-ink-muted">患者</div></div>
            <div><Cpu className="h-4 w-4 text-accent" /><div className="mt-2 text-sm font-black text-ink">{textValue(hpcGpu.name) || "A800 80GB"}</div><div className="text-xs text-ink-muted">远程计算</div></div>
          </div>
          <div className="mt-5 flex items-center justify-between border-t border-edge pt-3 text-xs"><span className="text-ink-muted">公开榜单提交</span><span className="font-bold text-warning-text">关闭</span></div>
          <div className="mt-2 truncate font-mono text-[10px] text-ink-muted" title={props.runId}>Run {props.runId || "—"}</div>
        </div>
      </div>

      {trainingProgress.status === "completed" && (
        <div className="border-y border-info/25 bg-info-light/25 px-5 py-5 sm:px-7 lg:px-9" data-ui-siim-runtime-evidence>
          <div className="flex flex-wrap items-end justify-between gap-3"><div><div className="text-xs font-bold text-info-text">A800 运行证据</div><h3 className="mt-1 text-base font-black text-ink">三种正式种子、外层折和显存样本已归档</h3></div><StatusBadge tone="green">训练完成</StatusBadge></div>
          <div className="mt-4 grid gap-px overflow-hidden rounded-md border border-edge bg-edge sm:grid-cols-2 lg:grid-cols-4">
            <div className="bg-surface-raised p-3"><div className="text-xs text-ink-muted">正式种子</div><div className="mt-1 font-mono text-lg font-black text-ink">{integer(completedFormalSeeds)} / {integer(totalFormalSeeds)}</div></div>
            <div className="bg-surface-raised p-3"><div className="text-xs text-ink-muted">已完成外层折</div><div className="mt-1 font-mono text-lg font-black text-ink">{integer(completedOuterFolds)} / {integer(totalOuterFolds)}</div></div>
            <div className="bg-surface-raised p-3"><div className="text-xs text-ink-muted">内层选择 Epoch</div><div className="mt-1 font-mono text-lg font-black text-ink">{typeof selectedEpochMin === "number" && typeof selectedEpochMax === "number" ? selectedEpochMin === selectedEpochMax ? integer(selectedEpochMin) : `${integer(selectedEpochMin)}–${integer(selectedEpochMax)}` : "—"}</div></div>
            <div className="bg-surface-raised p-3"><div className="text-xs text-ink-muted">峰值模型显存</div><div className="mt-1 font-mono text-lg font-black text-ink">{typeof peakModelMemory === "number" ? `${(peakModelMemory / 1024).toFixed(1)} GiB` : "—"}</div><div className="mt-1 text-[10px] text-ink-muted">{integer(telemetrySamples)} 条资源样本</div></div>
          </div>
        </div>
      )}

      {waiting && (
        <div className="border-y border-warning/35 bg-warning-light/40 px-5 py-4 sm:px-7 lg:px-9">
          <div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warning-text" /><div><div className="text-sm font-black text-warning-text">系统正在保护现有 GPU 任务</div><p className="mt-1 text-xs leading-5 text-ink-secondary">A800 资源或作业身份证据尚未全部通过门禁，本次训练保持未启动。系统不会向其他进程发送信号；资源恢复后继续同一 Run。</p></div></div>
        </div>
      )}

      <div className="px-5 py-6 sm:px-7 lg:px-9" data-ui-siim-task-graph>
        <div className="flex items-end justify-between gap-3"><div><h3 className="text-sm font-black text-ink">系统正在替用户完成的九个步骤</h3><p className="mt-1 text-xs text-ink-muted">所有步骤、指标、审核和下载均绑定同一个 Run ID。</p></div><StatusBadge tone={runComplete ? "green" : "blue"}>{runComplete ? "全部完成" : "真实进度"}</StatusBadge></div>
        <div className="mt-4 grid gap-px overflow-hidden rounded-md border border-edge bg-edge sm:grid-cols-2 lg:grid-cols-3">
          {siimStages.map((stage, index) => {
            const done = index < props.completedTasks;
            const active = !done && index === props.completedTasks;
            const Icon = stage.icon;
            return <div key={stage.title} className={`min-h-24 bg-surface-raised p-3 ${active ? "ring-2 ring-inset ring-accent" : ""}`}><div className="flex items-center gap-2"><span className={`flex h-7 w-7 items-center justify-center rounded-md ${done ? "bg-success text-white" : active ? "bg-accent text-accent-fg" : "bg-surface-sunken text-ink-muted"}`}>{done ? <Check className="h-4 w-4" /> : <Icon className="h-4 w-4" />}</span><span className="font-mono text-[10px] text-ink-muted">0{index + 1}</span></div><div className="mt-2 text-sm font-black text-ink">{stage.title}</div><div className="mt-1 text-xs leading-5 text-ink-muted">{stage.description}</div></div>;
          })}
        </div>
      </div>

      <SiimEvolutionTrail comparison={props.experimentComparison} metrics={metrics} compact />

      <div className="border-t border-edge px-5 py-6 sm:px-7 lg:px-9" data-ui-siim-results>
        <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="text-xs font-bold text-success-text">结果先讲人话</div><h3 className="mt-1 text-xl font-black text-ink">{typeof rocAuc === "number" ? "系统已完成独立离线评测与概率校准" : "真实分数将在训练和独立复核完成后出现"}</h3></div><StatusBadge tone={typeof rocAuc === "number" ? "green" : "slate"}>{textValue(props.experimentComparison?.selected_profile) || "方案待定"}</StatusBadge></div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="患者分组 ROC-AUC" value={metric(rocAuc)} description="官方主指标；患者与重复内容分组。" />
          <MetricCard label="PR-AUC" value={metric(prAuc)} description="关注少数恶性样本的排序质量。" tone="blue" />
          <MetricCard label="Brier" value={metric(brier)} description="概率校准误差，越低越好。" tone="violet" />
          <MetricCard label={graderView.label} value={graderView.value} description={graderView.description} />
        </div>
        <div className="mt-5 grid gap-5 border-t border-edge pt-5 lg:grid-cols-2">
          <div><div className="text-xs font-bold text-ink-muted">历史门槛仅供对照</div><div className="mt-3 flex flex-wrap gap-4 font-mono text-sm text-ink-secondary"><span>铜 {metric(numberValue(threshold.bronze), 4)}</span><span>银 {metric(numberValue(threshold.silver), 4)}</span><span>金 {metric(numberValue(threshold.gold), 4)}</span></div></div>
          <div><div className="text-xs font-bold text-ink-muted">审核状态</div><div className="mt-3 flex flex-wrap gap-2"><StatusBadge tone={statusPassed(props.reviewerStatus) ? "green" : "slate"}>Independent Review: {statusPassed(props.reviewerStatus) ? "通过" : "待完成"}</StatusBadge><StatusBadge tone={statusPassed(props.claimAuditStatus) ? "green" : "slate"}>Claim Audit: {statusPassed(props.claimAuditStatus) ? "通过" : "待完成"}</StatusBadge></div></div>
        </div>
      </div>

      {evidenceOpen && (
        <div className="border-t border-edge bg-surface-sunken/45 px-5 py-5 sm:px-7 lg:px-9" data-ui-siim-evidence>
          <div className="grid gap-6 lg:grid-cols-2">
            <div><div className="text-sm font-black text-ink">运行账本</div><div className="mt-3 divide-y divide-edge rounded-md border border-edge bg-surface-raised">{latestEvents.length ? latestEvents.map((event, index) => <div key={`${String(event.seq ?? index)}-${index}`} className="flex items-center justify-between gap-3 px-3 py-2 text-xs"><span className="truncate font-semibold text-ink">{String(event.task_id ?? "supervisor")}</span><span className="shrink-0 font-mono text-ink-muted">{String(event.status ?? "recorded")}</span></div>) : <div className="px-3 py-4 text-xs text-ink-muted">等待新的同 Run 事件。</div>}</div></div>
            <div><div className="text-sm font-black text-ink">证据边界</div><ul className="mt-3 space-y-2 text-xs leading-5 text-ink-secondary"><li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />患者与重复图像必须在折间零重叠</li><li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />私有 grader 只在候选冻结后记录一次；失败关闭不显示分数</li><li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />不提交公开榜单，不声明正式排名或奖牌</li><li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-success" />所有交付文件下载前重新核验 SHA-256</li></ul></div>
          </div>
        </div>
      )}

      <div className="border-t border-edge bg-frame/95 px-5 py-5 text-white sm:px-7 lg:px-9" data-ui-siim-downloads>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between"><div><div className="text-sm font-black">四个文件完成真实交付</div><div className="mt-1 text-xs text-white/65">只有报告、结果、代码和证据包全部存在且哈希一致时，下载按钮才会启用。</div></div><div className="flex flex-wrap gap-2">{[
          ["evomind-siim-isic-report.pdf", "PDF 报告"],
          ["evomind-siim-isic-results.csv", "结果 CSV"],
          ["evomind-siim-isic-code.zip", "代码 ZIP"],
          ["evomind-siim-isic-evidence.zip", "证据 ZIP"],
        ].map(([name, label]) => {
          const file = downloadByName.get(name);
          const href = textValue(file?.download_url);
          return file && href ? <DownloadLink key={name} href={href} filename={name} label={label} /> : <DisabledDownload key={name} label={label} />;
        })}</div></div>
      </div>

      <SiimFullReport open={reportOpen} onClose={() => setReportOpen(false)} props={props} />
    </section>
  );
}

function CreditCardResearchJourney(props: UserResearchJourneyProps) {
  const [resultOpen, setResultOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [continueOpen, setContinueOpen] = useState(false);
  const [requestDraft, setRequestDraft] = useState(USER_REQUEST);
  const [replayStep, setReplayStep] = useState<number>(stages.length);
  const [replaying, setReplaying] = useState(false);
  const [refinement, setRefinement] = useState("在保持误报可控的前提下，继续提高欺诈召回率");
  const isTargetRun = props.taskId === "credit-card-fraud-detection" || props.runId?.includes("2c1820");

  useEffect(() => {
    if (!replaying) return;
    const timer = window.setInterval(() => {
      setReplayStep((current) => {
        if (current >= stages.length) {
          window.clearInterval(timer);
          setReplaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1450);
    return () => window.clearInterval(timer);
  }, [replaying]);

  const completedLabel = useMemo(() => `${Math.min(props.completedTasks, props.totalTasks)}/${props.totalTasks || 0}`, [props.completedTasks, props.totalTasks]);
  if (!isTargetRun) return null;

  const startReplay = () => {
    setResultOpen(false);
    setEvidenceOpen(false);
    setReplayStep(0);
    setReplaying(true);
  };

  return (
    <section className="overflow-hidden rounded-lg border border-success/35 bg-surface-raised shadow-raised" data-ui-user-journey>
      <div className="grid gap-6 px-5 py-6 sm:px-7 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,0.8fr)] lg:px-9 lg:py-8">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge tone="green">研究已完成</StatusBadge>
            <span className="text-xs font-semibold text-ink-muted">普通用户视图</span>
          </div>
          <h2 className="mt-4 max-w-3xl text-2xl font-black leading-tight text-ink sm:text-3xl">只说清楚问题，也能完成一次可复核的欺诈分析</h2>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-ink-secondary">你不需要进终端，也不需要选择算法。系统负责检查数据、安排计算、比较方案、独立复核，并把结果整理成能直接阅读和下载的报告。</p>
          <label className="mt-5 block rounded-lg border border-edge bg-surface-raised/90 p-4 shadow-sm">
            <span className="text-[11px] font-bold uppercase tracking-normal text-ink-muted">一句话需求</span>
            <textarea
              value={requestDraft}
              onChange={(event) => setRequestDraft(event.target.value)}
              rows={3}
              data-ui-action="user_request_input"
              data-ui-skip-action="true"
              aria-label="一句话研究需求"
              className="mt-2 min-h-20 w-full resize-none border-0 bg-transparent p-0 text-sm font-medium leading-7 text-ink outline-none"
            />
          </label>
          <div className="mt-5 flex flex-wrap gap-2">
            <Button data-ui-action="user_replay_research" data-ui-skip-action="true" onClick={startReplay} disabled={replaying}>
              <Play className="h-4 w-4" />
              {replaying ? "正在按阶段回放…" : "查看本次分析过程"}
            </Button>
            <Button variant="secondary" data-ui-action="user_open_results" data-ui-skip-action="true" onClick={() => setResultOpen(true)}>
              <BarChart3 className="h-4 w-4" />查看结果
            </Button>
            <Button variant="ghost" data-ui-action="user_open_full_report" data-ui-skip-action="true" onClick={() => setReportOpen(true)}>
              <FileText className="h-4 w-4" />查看完整报告
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-edge bg-surface-raised/90 p-4 shadow-sm">
          <div className="flex items-center justify-between gap-3"><div className="text-sm font-black text-ink">这次研究的范围</div><span className="text-xs font-semibold text-success-text">{completedLabel} 步骤完成</span></div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-md bg-surface-sunken p-3"><Database className="h-4 w-4 text-accent" /><div className="mt-2 text-lg font-black text-ink">129.7 万</div><div className="text-xs text-ink-muted">历史交易</div></div>
            <div className="rounded-md bg-surface-sunken p-3"><Eye className="h-4 w-4 text-accent" /><div className="mt-2 text-lg font-black text-ink">55.6 万</div><div className="text-xs text-ink-muted">未来检查</div></div>
            <div className="rounded-md bg-surface-sunken p-3"><Cpu className="h-4 w-4 text-accent" /><div className="mt-2 text-sm font-black text-ink">RTX 4060</div><div className="text-xs text-ink-muted">仅使用本机</div></div>
            <div className="rounded-md bg-surface-sunken p-3"><ShieldCheck className="h-4 w-4 text-accent" /><div className="mt-2 text-sm font-black text-ink">双重检查</div><div className="text-xs text-ink-muted">审核与声明</div></div>
          </div>
          <div className="mt-4 flex items-center justify-between gap-3 border-t border-edge pt-3 text-xs"><span className="text-ink-muted">公开榜单</span><span className="font-bold text-warning-text">未提交</span></div>
        </div>
      </div>

      <div className="border-t border-edge/80 bg-surface-raised/72 px-5 py-5 sm:px-7 lg:px-9">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-sm font-black text-ink">系统替你完成的五件事</h3><p className="mt-0.5 text-xs text-ink-muted">展示的是本次已完成任务的阶段记录，不会重新启动耗时计算。</p></div>{replaying ? <StatusBadge tone="blue">按阶段回放</StatusBadge> : <StatusBadge tone="green">全部完成</StatusBadge>}</div>
        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
          {stages.map((stage, index) => {
            const done = index < replayStep;
            const active = replaying && index === replayStep;
            const Icon = stage.icon;
            return (
              <div key={stage.title} className={`min-h-28 rounded-lg border p-3 transition-all duration-500 ${done ? "border-success/35 bg-success-light/45" : active ? "scale-[1.02] border-accent bg-accent-light/55 shadow-raised" : "border-edge bg-surface-sunken/70 opacity-55"}`}>
                <div className="flex items-center justify-between gap-2"><span className={`flex h-8 w-8 items-center justify-center rounded-md ${done ? "bg-success text-white" : active ? "bg-accent text-accent-fg" : "bg-edge text-ink-muted"}`}>{done ? <Check className="h-4 w-4" /> : <Icon className="h-4 w-4" />}</span><span className="font-mono text-[10px] font-bold text-ink-muted">0{index + 1}</span></div>
                <div className="mt-3 text-sm font-black text-ink">{stage.title}</div>
                <div className="mt-1 text-xs leading-5 text-ink-muted">{stage.description}</div>
              </div>
            );
          })}
        </div>
      </div>

      {resultOpen && (
        <div className="border-t border-edge bg-surface-raised px-5 py-6 sm:px-7 lg:px-9" data-ui-user-results>
          <div className="flex flex-wrap items-start justify-between gap-3"><div><div className="text-xs font-bold uppercase tracking-normal text-success-text">结果先讲人话</div><h3 className="mt-1 text-xl font-black text-ink">系统识别出了大多数真实欺诈，同时保持较高的标记可信度</h3></div><StatusBadge tone="green">{props.selectedSolution || "EXP002"}</StatusBadge></div>
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="真正欺诈识别率" value={percent(props.metrics.recall)} description="约 100 笔真实欺诈中识别 92 笔。" />
            <MetricCard label="风险标记可信度" value={percent(props.metrics.precision)} description="约 100 笔风险标记中 95 笔正确。" tone="blue" />
            <MetricCard label="整体识别质量" value={percent(props.metrics.prAuc, 2)} description="在欺诈样本很少时，衡量系统排序是否靠谱。" tone="violet" />
            <MetricCard label="自动改进幅度" value={`+${((props.evolution.deltaPrAuc ?? 0) * 100).toFixed(2)}`} description="第二轮整体质量提升百分点。" />
          </div>
          <div className="mt-5 grid gap-5 rounded-lg border border-edge bg-surface-sunken/55 p-4 lg:grid-cols-[1fr_0.8fr]">
            <div><div className="flex items-center gap-2 text-sm font-black text-ink"><TrendingUp className="h-4 w-4 text-success" />第二轮自动改进</div><div className="mt-4"><ComparisonBar label="整体识别质量" before={props.evolution.beforePrAuc} after={props.evolution.afterPrAuc} /></div></div>
            <div className="text-xs leading-6 text-ink-secondary"><strong className="text-ink">怎么看：</strong>第二轮不是简单重复，而是在保持数据和独立检查不变的前提下，只调整分析方案，再重新计算和复核。旧版本与原始证据仍然保留。</div>
          </div>
          <div className="mt-5 flex flex-wrap gap-2">
            <Button data-ui-action="user_results_open_report" data-ui-skip-action="true" onClick={() => setReportOpen(true)}><FileText className="h-4 w-4" />查看完整报告</Button>
            <DownloadLink href={`${downloadRoot}/evomind-credit-card-fraud-report.pdf`} filename="evomind-credit-card-fraud-report.pdf" label="下载报告" />
            <DownloadLink href={`${downloadRoot}/evomind-credit-card-fraud-results.csv`} filename="evomind-credit-card-fraud-results.csv" label="下载结果" />
            <button type="button" data-ui-action="user_toggle_evidence" data-ui-skip-action="true" onClick={() => setEvidenceOpen((value) => !value)} className="inline-flex h-9 items-center gap-1.5 rounded-md border border-edge bg-surface-raised px-3 text-xs font-semibold text-ink-secondary hover:border-accent hover:text-accent"><ShieldCheck className="h-3.5 w-3.5" />查看证据<ChevronDown className={`h-3.5 w-3.5 transition-transform ${evidenceOpen ? "rotate-180" : ""}`} /></button>
          </div>
          {evidenceOpen && (
            <div className="mt-4 grid gap-3 rounded-lg border border-success/30 bg-success-light/35 p-4 sm:grid-cols-3" data-ui-user-evidence>
              <div><div className="text-xs text-ink-muted">独立审核</div><div className="mt-1 flex items-center gap-1.5 text-sm font-black text-success-text"><CheckCircle2 className="h-4 w-4" />{statusPassed(props.reviewerStatus) ? "已通过" : props.reviewerStatus || "待记录"}</div></div>
              <div><div className="text-xs text-ink-muted">声明检查</div><div className="mt-1 flex items-center gap-1.5 text-sm font-black text-success-text"><CheckCircle2 className="h-4 w-4" />{statusPassed(props.claimAuditStatus) ? "已通过" : props.claimAuditStatus || "待记录"}</div></div>
              <div><div className="text-xs text-ink-muted">可追溯产物</div><div className="mt-1 flex items-center gap-1.5 text-sm font-black text-success-text"><PackageCheck className="h-4 w-4" />{props.artifactCount} 个文件指纹</div></div>
              <div className="sm:col-span-3 flex flex-wrap items-center justify-between gap-3 border-t border-success/25 pt-3"><p className="text-xs leading-5 text-ink-secondary">指标来自独立离线时间验证；没有拿最终检查数据反复试答案；公开榜单提交未执行。</p><DownloadLink href={`${downloadRoot}/evomind-credit-card-fraud-evidence.zip`} filename="evomind-credit-card-fraud-evidence.zip" label="下载证据包" /></div>
            </div>
          )}
        </div>
      )}

      <div className="border-t border-edge bg-frame/95 px-5 py-4 text-white sm:px-7 lg:px-9">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between"><div><div className="flex items-center gap-2 text-sm font-black"><History className="h-4 w-4 text-success-light" />想继续提高召回率？还是只说一句话</div><div className="mt-1 text-xs text-white/65">原版本会保留，系统只重做受影响的步骤。</div></div><Button variant="secondary" data-ui-action="user_continue_optimization" data-ui-skip-action="true" onClick={() => setContinueOpen((value) => !value)}>{continueOpen ? "收起" : "继续优化"}<ArrowRight className="h-4 w-4" /></Button></div>
        {continueOpen && <div className="mt-4 flex flex-col gap-2 rounded-lg border border-white/15 bg-white/5 p-3 sm:flex-row"><textarea value={refinement} onChange={(event) => setRefinement(event.target.value)} rows={2} aria-label="继续优化要求" className="min-h-14 flex-1 resize-none rounded-md border border-white/20 bg-white/95 px-3 py-2 text-sm leading-6 text-ink outline-none focus:border-success" /><button type="button" data-ui-action="user_prepare_refinement" data-ui-skip-action="true" className="h-10 rounded-md bg-success px-4 text-xs font-black text-white sm:self-end">生成优化计划</button></div>}
      </div>

      <FullReport open={reportOpen} onClose={() => setReportOpen(false)} props={props} />
    </section>
  );
}

export function UserResearchJourney(props: UserResearchJourneyProps) {
  if (props.taskId === "siim-isic-melanoma-classification" || props.runId?.startsWith("evomind_siim_isic_")) {
    return <SiimResearchJourney {...props} />;
  }
  return <CreditCardResearchJourney {...props} />;
}
