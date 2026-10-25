export type BudgetPayload = {
  used_seconds?: number;
  charged_wall_seconds?: number;
  pending_reserved_seconds?: number;
  committed_seconds?: number;
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

export function BudgetUsage({ budget }: { budget: BudgetPayload }) {
  const used = usedSeconds(budget);
  const reserved = budget.pending_reserved_seconds ?? 0;
  return <dl className="tw-budget">
    <div><dt>已用</dt><dd data-budget-used={used}>{hours(used)} 小时</dd></div>
    <div><dt>预留</dt><dd data-budget-reserved={reserved}>{hours(reserved)} 小时</dd></div>
  </dl>;
}
