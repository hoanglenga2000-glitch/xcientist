import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const prismaCli = fileURLToPath(
  new URL("../node_modules/prisma/build/index.js", import.meta.url),
);

function runPrisma(args, { emptyStdin = false } = {}) {
  const result = spawnSync(process.execPath, [prismaCli, ...args], {
    cwd: projectRoot,
    env: process.env,
    input: emptyStdin ? "" : undefined,
    stdio: emptyStdin ? ["pipe", "inherit", "inherit"] : "inherit",
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

// Prisma 6 on Windows can fail to create a missing SQLite file during db push.
// An empty schema-bound execution creates it without changing database content.
runPrisma(["db", "execute", "--stdin", "--schema", "prisma/schema.prisma"], {
  emptyStdin: true,
});
runPrisma(["db", "push", ...process.argv.slice(2)]);
