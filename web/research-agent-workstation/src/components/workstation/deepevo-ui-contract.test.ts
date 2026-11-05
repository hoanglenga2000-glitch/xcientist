import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const ROOT = process.cwd();

async function text(relativePath: string) {
  return readFile(path.join(ROOT, relativePath), "utf8");
}

test("public workstation branding is DeepEvo while runtime compatibility identifiers remain unchanged", async () => {
  const [layout, sidebar, assistant] = await Promise.all([
    text("src/app/layout.tsx"),
    text("src/components/workstation/Sidebar.tsx"),
    text("src/components/workstation/screens/AssistantScreen.tsx"),
  ]);

  assert.match(layout, /title: "DeepEvo"/);
  assert.match(sidebar, /DeepEvoLockup/);
  assert.match(assistant, /DeepEvo Assistant/);
  assert.match(assistant, /X-EvoMind-CSRF/);
  assert.doesNotMatch(assistant, /X-DeepEvo-CSRF/);
});

test("assistant composer uses the concise professional placeholder", async () => {
  const assistant = await text("src/components/workstation/screens/AssistantScreen.tsx");
  assert.match(assistant, /输入任务内容…/);
  assert.match(assistant, /Enter a task…/);
  assert.doesNotMatch(assistant, /一句话交代任务/);
});

// t237 C6-brand: the cropped teacher lockup (deepevo-lockup.png + 18 s orbit) is replaced by the Iris kit
// (/workspace/deepevo-brand/final, Iris refined v2). These pins prove the kit ships unmodified.
const KIT_SHA256: Record<string, string> = {
  "public/brand/mark.svg": "0e46f6cf5d3bb4551ba547f532ef009e77a1c3d46bd09cc9252e33d880511bef",
  "public/brand/mark-small.svg": "af0b289505b0c93cbe6253e1d0838c0b786e20ff8953611c47d13f68daa87bdd",
  "public/brand/lockup-horizontal-black.svg": "9a711ee229a77661bee12d50afcce707428e43167efbcbb2c4db2f56eb8f35cb",
  "public/brand/lockup-horizontal-white.svg": "dade519f3559868c111185b7ff54100d210e87aabac89eb68e9bc6e762cd1d6a",
};

test("DeepEvo Iris kit marks and lockups are byte-identical and wired to a reduced-motion-safe loader", async () => {
  for (const [file, hash] of Object.entries(KIT_SHA256)) {
    assert.equal(createHash("sha256").update(await readFile(path.join(ROOT, file))).digest("hex"), hash, file);
  }
  const [brand, styles] = await Promise.all([
    text("src/components/workstation/DeepEvoBrand.tsx"),
    text("src/components/workstation/deepevo-brand.css"),
  ]);
  assert.match(brand, /\/brand\/mark\.svg/);
  assert.match(brand, /\/brand\/mark-small\.svg/);
  assert.match(brand, /\/brand\/lockup-horizontal-black\.svg/);
  assert.doesNotMatch(brand, /deepevo-lockup\.png/);
  assert.match(styles, /@keyframes deepevo-loader-master/);
  assert.match(styles, /prefers-reduced-motion: reduce/);
});
