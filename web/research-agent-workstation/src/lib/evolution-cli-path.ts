import { constants } from "node:fs";
import { access } from "node:fs/promises";
import path from "node:path";

export class EvolutionUnavailable extends Error {
  constructor() {
    super("evolution_unavailable");
    this.name = "EvolutionUnavailable";
  }
}

/** Paths the evolution routes may shell out to. First readable file wins. */
export function evolutionCliCandidates(workspaceRoot: string, moduleDir: string, cwd = process.cwd()): string[] {
  const configured = process.env.EVOMIND_EVOLUTION_CLI?.trim();
  return [
    configured,
    path.join(workspaceRoot, "scripts", "evolution_engine_cli.py"),
    path.resolve(moduleDir, "../../../../scripts/evolution_engine_cli.py"),
    path.resolve(moduleDir, "../../../../../scripts/evolution_engine_cli.py"),
    path.resolve(cwd, "scripts", "evolution_engine_cli.py"),
    path.resolve(cwd, "..", "scripts", "evolution_engine_cli.py"),
    path.resolve(cwd, "..", "..", "scripts", "evolution_engine_cli.py"),
  ].filter((candidate): candidate is string => Boolean(candidate));
}

export async function resolveEvolutionCli(candidates: string[]): Promise<string | null> {
  for (const candidate of candidates) {
    const target = path.resolve(candidate);
    try {
      await access(target, constants.R_OK);
      return target;
    } catch {
      // Try the next location. A missing CLI is reported by the caller.
    }
  }
  return null;
}

export function evolutionErrorBody(error: unknown, body: Record<string, unknown>, fallback: string) {
  if (error instanceof EvolutionUnavailable) {
    return { status: 503 as const, body: { ok: false, available: false, ...body, error: "evolution_unavailable" } };
  }
  const message = error instanceof Error ? error.message : fallback;
  return { status: 400 as const, body: { ok: false, ...body, error: message } };
}
