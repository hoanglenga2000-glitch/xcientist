import { readdirSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

function collect(directory) {
  const tests = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) tests.push(...collect(path));
    else if (entry.name.endsWith(".test.ts")) tests.push(path);
  }
  return tests;
}

const tests = collect("src").sort();
if (tests.length === 0) {
  process.stderr.write("No TypeScript tests found.\n");
  process.exit(1);
}
const result = spawnSync(
  process.execPath,
  ["--test", "--experimental-strip-types", "--disable-warning=MODULE_TYPELESS_PACKAGE_JSON", ...tests],
  { stdio: "inherit" },
);
process.exit(result.status ?? 1);
