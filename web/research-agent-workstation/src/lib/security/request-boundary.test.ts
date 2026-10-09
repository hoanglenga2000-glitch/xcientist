import assert from "node:assert/strict";
import test from "node:test";

// @ts-ignore Node's strip-types test runner requires the explicit .ts suffix.
import { configuredPublicOrigin, isAllowedBrowserOrigin, isPublicHostHeader, isTrustedHostHeader, normalizeTaskId } from "./request-boundary.ts";


test("normalizeTaskId accepts registry-backed ISO-style task ids", () => {
  assert.equal(
    normalizeTaskId("task_2026-06-12T06-28-14-863Z"),
    "task_2026-06-12T06-28-14-863Z",
  );
});

test("normalizeTaskId keeps path traversal and reserved names blocked", () => {
  for (const candidate of ["../task", "task%2Fchild", "task\\child", "NUL", "task."]) {
    assert.throws(() => normalizeTaskId(candidate), /Invalid task ID/);
  }
});

test("the configured HTTPS public origin is trusted without weakening the loopback boundary", () => {
  const original = process.env.WORKSTATION_PUBLIC_ORIGIN;
  process.env.WORKSTATION_PUBLIC_ORIGIN = "https://evomind.zhjjq.tech";
  try {
    assert.equal(configuredPublicOrigin(), "https://evomind.zhjjq.tech");
    assert.equal(isPublicHostHeader("evomind.zhjjq.tech"), true);
    assert.equal(isTrustedHostHeader("evomind.zhjjq.tech"), true);
    assert.equal(isTrustedHostHeader("127.0.0.1:8088"), true);
    assert.equal(isTrustedHostHeader("attacker.example"), false);
    assert.equal(isAllowedBrowserOrigin("https://evomind.zhjjq.tech", "evomind.zhjjq.tech"), true);
    assert.equal(isAllowedBrowserOrigin("https://attacker.example", "evomind.zhjjq.tech"), false);
  } finally {
    if (original === undefined) delete process.env.WORKSTATION_PUBLIC_ORIGIN;
    else process.env.WORKSTATION_PUBLIC_ORIGIN = original;
  }
});
