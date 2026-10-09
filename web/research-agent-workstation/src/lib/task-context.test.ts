import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's native TypeScript test runner requires the explicit extension.
import { assistantRunTaskLabel } from "./task-context.ts";

test("assistant task label prefers the current Run prompt over stale Run and global context", () => {
  assert.equal(
    assistantRunTaskLabel(
      { id: "run_latest", prompt: "\u8bad\u7ec3 Porto Seguro \u5e76\u53d1\u5e03\u4ea7\u7269", selected_task: "house_prices" },
      "house_prices",
    ),
    "\u8bad\u7ec3 Porto Seguro \u5e76\u53d1\u5e03\u4ea7\u7269",
  );
});

test("assistant task label accepts a strong task id from the current Run", () => {
  assert.equal(
    assistantRunTaskLabel(
      { id: "run_latest", prompt: "continue", task_id: "porto-seguro-safe-driver-prediction", selected_task: "house_prices" },
      "house_prices",
    ),
    "porto-seguro-safe-driver-prediction",
  );
});

test("assistant task label derives a compact title from the current Run when it has no task id", () => {
  const label = assistantRunTaskLabel(
    { id: "run_latest", prompt: "  \u8bad\u7ec3 Porto Seguro \u5e76\u751f\u6210\u5b8c\u6574\u9a8c\u6536\u4ea7\u7269\u3002\u4e0d\u63d0\u4ea4 Kaggle\u3002" },
    "house_prices",
  );
  assert.equal(label, "\u8bad\u7ec3 Porto Seguro \u5e76\u751f\u6210\u5b8c\u6574\u9a8c\u6536\u4ea7\u7269");
  assert.doesNotMatch(label, /house_prices/);
});

test("assistant task label does not expose a prompt that looks like credential material", () => {
  assert.equal(
    assistantRunTaskLabel({ id: "run_secret_safe", prompt: "\u4f7f\u7528 API token abcdef \u5b8c\u6210\u8bad\u7ec3", selected_task: "house_prices" }, "house_prices"),
    "run_secret_safe",
  );
});

test("assistant task label falls back to the selected task only when there is no Run", () => {
  assert.equal(assistantRunTaskLabel(null, "house-prices"), "house_prices");
});
