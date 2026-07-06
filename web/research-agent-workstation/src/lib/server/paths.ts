import { promises as fs } from "node:fs";
import path from "node:path";

export const workspaceRoot = path.resolve(process.env.WORKSTATION_ROOT ?? path.resolve(process.cwd(), "..", ".."));
export const runtimeRoot = path.join(workspaceRoot, "workspace", "runtime");

export function normalizeTaskId(taskId: string) {
  return taskId === "house-prices" ? "house_prices" : taskId;
}

export function toRelativePath(targetPath: string | null | undefined) {
  if (!targetPath) return null;
  return path.isAbsolute(targetPath) ? path.relative(workspaceRoot, targetPath) : targetPath;
}

export function resolveWorkspacePath(relativePath: string) {
  return path.join(workspaceRoot, relativePath);
}

export function stamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

export async function readJsonFile(filePath: string) {
  try {
    const text = await fs.readFile(filePath, "utf-8");
    return JSON.parse(text.replace(/^\uFEFF/, ""));
  } catch {
    return null;
  }
}

export async function writeJsonArtifact(relativePath: string, payload: unknown) {
  const target = resolveWorkspacePath(relativePath);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, JSON.stringify(payload, null, 2), "utf-8");
  return relativePath;
}

export async function writeTextArtifact(relativePath: string, payload: string) {
  const target = resolveWorkspacePath(relativePath);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, payload, "utf-8");
  return relativePath;
}

export async function latestExperimentPath(taskId: string) {
  const normalized = normalizeTaskId(taskId);
  if (normalized === "playground_series_s6e6") {
    const workstationRun = await latestWorkstationRunPath(normalized);
    if (workstationRun) return workstationRun;
  }
  const root = path.join(workspaceRoot, "experiments", normalized);
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const latest = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort().pop();
  if (latest) return path.join("experiments", normalized, latest);
  return latestWorkstationRunPath(normalized);
}

export async function latestExperimentPathWithAnyArtifacts(taskId: string, artifactNames: string[]) {
  const normalized = normalizeTaskId(taskId);
  const roots = [
    { absolute: path.join(workspaceRoot, "experiments", normalized), relative: path.join("experiments", normalized) },
    { absolute: path.join(workspaceRoot, "workspace", "workstation_runs", normalized), relative: path.join("workspace", "workstation_runs", normalized) }
  ];

  for (const root of roots) {
    const entries = await fs.readdir(root.absolute, { withFileTypes: true }).catch(() => []);
    const candidates = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort()
      .reverse();

    for (const name of candidates) {
      const runRoot = path.join(root.absolute, name);
      const checks = await Promise.all(
        artifactNames.map((artifactName) => fs.access(path.join(runRoot, artifactName)).then(() => true).catch(() => false))
      );
      if (checks.some(Boolean)) return path.join(root.relative, name);
    }
  }

  return latestExperimentPath(normalized);
}

export async function latestWorkstationRunPath(taskId: string) {
  const normalized = normalizeTaskId(taskId);
  const root = path.join(workspaceRoot, "workspace", "workstation_runs", normalized);
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const latest = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort().pop();
  return latest ? path.join("workspace", "workstation_runs", normalized, latest) : null;
}

export async function latestWorkstationRunWithArtifactsPath(taskId: string, artifactNames: string[]) {
  const normalized = normalizeTaskId(taskId);
  const root = path.join(workspaceRoot, "workspace", "workstation_runs", normalized);
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort()
    .reverse();

  for (const name of candidates) {
    const runRoot = path.join(root, name);
    const checks = await Promise.all(
      artifactNames.map((artifactName) => fs.access(path.join(runRoot, artifactName)).then(() => true).catch(() => false))
    );
    if (checks.every(Boolean)) return path.join("workspace", "workstation_runs", normalized, name);
  }
  return null;
}

export async function latestScoreGatedWorkstationRunPath(taskId: string) {
  return latestWorkstationRunWithArtifactsPath(taskId, ["score_improvement_gate.json", "submission_audit.json"]);
}
