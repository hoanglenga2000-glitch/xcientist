import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { EvolutionUnavailable, evolutionCliCandidates, evolutionErrorBody, resolveEvolutionCli } from "./evolution-cli-path.ts";

test("evolution cli resolves a readable script and otherwise stays unavailable", async () => {
  const root = mkdtempSync(path.join(tmpdir(), "evo-cli-"));
  const previous = process.env.EVOMIND_EVOLUTION_CLI;
  delete process.env.EVOMIND_EVOLUTION_CLI;
  try {
    assert.equal(await resolveEvolutionCli([path.join(root, "missing.py")]), null);
    const script = path.join(root, "scripts", "evolution_engine_cli.py");
    mkdirSync(path.dirname(script), { recursive: true });
    writeFileSync(script, "# fixture\n");
    assert.equal(await resolveEvolutionCli([path.join(root, "missing.py"), script]), script);
    const found = await resolveEvolutionCli(evolutionCliCandidates(root, root));
    assert.equal(found, script);
    const failure = evolutionErrorBody(new EvolutionUnavailable(), { task_id: "demo" }, "evolution state failed");
    assert.equal(failure.status, 503);
    assert.equal(failure.body.available, false);
    assert.equal(failure.body.error, "evolution_unavailable");
    const other = evolutionErrorBody(new Error("bad input"), { task_id: "demo" }, "evolution state failed");
    assert.equal(other.status, 400);
    assert.equal(other.body.error, "bad input");
  } finally {
    if (previous === undefined) delete process.env.EVOMIND_EVOLUTION_CLI;
    else process.env.EVOMIND_EVOLUTION_CLI = previous;
    rmSync(root, { recursive: true, force: true });
  }
});
