export type BudgetPayload = {
  used_seconds?: number;
  charged_wall_seconds?: number;
  pending_reserved_seconds?: number;
  committed_seconds?: number;
  engineering_used_seconds?: number;
  gpu_limit_seconds?: number;
  engineering_limit_seconds?: number;
  limit_seconds?: number;
};

/** Displayed usage is charged time. Reserved holds stay on their own line. */
export function usedSeconds(budget: BudgetPayload) {
  if (typeof budget.used_seconds === 'number') return budget.used_seconds;
  if (typeof budget.charged_wall_seconds === 'number') return budget.charged_wall_seconds;
  return 0;
}

function hours(seconds: number) {
  return (seconds / 3600).toLocaleString('zh-CN', { maximumFractionDigits: 2, minimumFractionDigits: 0 });
}

function ProgressRow({ name, used, limit }: { name: string; used: number; limit?: number }) {
  const hasLimit = typeof limit === 'number' && limit > 0;
  const pct = hasLimit ? Math.min(100, Math.round((used / limit) * 100)) : 0;
  const tone = !hasLimit ? '' : pct >= 100 ? ' is-full' : pct >= 80 ? ' is-warn' : '';
  return <div className="tw-set-row tw-budget-row">
    <div className="tw-set-row-main">
      <div className="tw-set-row-title">{name}</div>
      <div className="tw-set-row-desc">{hasLimit ? `已用 ${hours(used)} / 上限 ${hours(limit!)} 小时` : `已用 ${hours(used)} 小时`}</div>
      {hasLimit && <div className={`tw-budget-bar${tone}`} aria-hidden="true"><span style={{ width: `${pct}%` }} /></div>}
    </div>
  </div>;
}

export function BudgetUsage({ budget }: { budget: BudgetPayload }) {
  const used = usedSeconds(budget);
  const reserved = budget.pending_reserved_seconds ?? 0;
  const gpuLimit = budget.gpu_limit_seconds ?? budget.limit_seconds;
  const engLimit = budget.engineering_limit_seconds;
  const engineeringUsed = typeof budget.engineering_used_seconds === 'number' ? budget.engineering_used_seconds : 0;
  if (!used && !reserved && !engineeringUsed && gpuLimit == null && engLimit == null) {
    return <p className="tw-muted">暂时没有额度数据。</p>;
  }
  return <div className="tw-set-card tw-budget" data-budget-used={used} data-budget-reserved={reserved} data-engineering-used={engineeringUsed}>
    <ProgressRow name="已用计算时间" used={used} limit={gpuLimit} />
    <ProgressRow name="预留" used={reserved} />
    {engLimit != null && <ProgressRow name="工程额度" used={engineeringUsed} limit={engLimit} />}
  </div>;
}
