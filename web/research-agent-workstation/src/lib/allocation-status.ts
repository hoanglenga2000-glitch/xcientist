export type AllocationCheck = {
  personal_resources?: string | null;
  binding?: { state?: string | null } | null;
  activation?: { state?: string | null; receipt_exists?: boolean | null } | null;
};

/** One status line from the stored allocation check. Does not open a connection. */
export function allocationStatusText(check: AllocationCheck | null | undefined, loaded = true): string {
  if (!loaded) return "正在读取账户配置…";
  if (check?.personal_resources === "unavailable_in_local_mode") return "未接入计算资源";
  const state = check?.binding?.state || "";
  const checked = check?.activation?.state === "completed" && check?.activation?.receipt_exists === true;
  if (!check?.binding) return "未设置计算资源";
  if (state === "active" && checked) return "计算资源已核对";
  if (state === "active") return "计算资源未通过核对";
  return "计算资源暂不可用";
}
