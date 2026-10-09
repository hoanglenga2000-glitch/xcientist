import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const componentRoot = path.join(process.cwd(), "src", "components", "workstation");
const screenPath = path.join(componentRoot, "screens", "DataKaggleScreen.tsx");
const quickEnrollmentPath = path.join(componentRoot, "TenantHpcQuickEnrollment.tsx");

test("Data and Kaggle mounts the managed quick enrollment surface", async () => {
  const screen = await readFile(screenPath, "utf-8");
  const source = await readFile(quickEnrollmentPath, "utf-8");

  assert.match(source, /const CSRF_STORAGE_KEY = "evomind\.local\.csrf\.v1"/);
  assert.match(source, /window\.sessionStorage\.getItem\(CSRF_STORAGE_KEY\)/);
  assert.match(source, /if \(!value\) throw new Error\("csrf_missing"\)/);
  assert.match(source, /"x-evomind-csrf": csrfToken\(\)/);
  assert.match(screen, /<TenantHpcQuickEnrollment\s*\/>/);
});

test("quick enrollment state survives refresh without persisting passwords", async () => {
  const source = await readFile(quickEnrollmentPath, "utf-8");

  assert.match(source, /evomind\.hpc\.quick_replace\.v1/);
  assert.match(source, /window\.localStorage\.setItem\(QUICK_STATE_KEY/);
  assert.match(source, /window\.localStorage\.getItem\(QUICK_STATE_KEY/);
  assert.match(source, /window\.sessionStorage\.setItem\(QUICK_SESSION_KEY/);
  assert.match(source, /window\.sessionStorage\.getItem\(QUICK_SESSION_KEY/);
  assert.doesNotMatch(source, /sessionStorage\.setItem\([^\n]*(?:password|token|secret|credential)/i);
  assert.doesNotMatch(source, /localStorage\.setItem\([^\n]*(?:password|token|secret|credential)/i);
});

test("tenant enrollment retains CSRF-protected reservation and envelope posts", async () => {
  const source = await readFile(
    path.join(componentRoot, "TenantHpcEnrollment.tsx"),
    "utf-8",
  );

  assert.match(source, /sessionStorage\.getItem\("evomind\.local\.csrf\.v1"\)/);
  assert.match(source, /if\(!csrf\)throw new Error\("csrf_missing"\)/);
  assert.match(source, /"x-evomind-csrf":csrf/);
  assert.match(source, /\/api\/hpc\/byoa\/enrollment-identity/);
  assert.match(source, /\/api\/hpc\/byoa\/enrollment-envelope/);
});
