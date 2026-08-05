import { execFile } from "node:child_process";
import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { NextResponse } from "next/server";
import { runManagedCommand } from "@/lib/server/job-registry";
import { resolveWorkspacePath, workspaceRoot, writeTextArtifact } from "@/lib/server/paths";
import { generateScientificReport } from "@/lib/server/scientific-report";
import type { RefinementPlan } from "@/lib/api/types";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const SAFE_ID = /^[A-Za-z0-9_-]{1,180}$/;
type CliResult = { ok: boolean; refinement: RefinementPlan; should_start?: boolean };
const globalRefinementState = globalThis as unknown as { refinementLocks?: Map<string, Promise<void>> };
const refinementLocks = globalRefinementState.refinementLocks ?? new Map<string, Promise<void>>();
globalRefinementState.refinementLocks = refinementLocks;

async function withRefinementLock<T>(key: string, operation: () => Promise<T>): Promise<T> {
  const previous = refinementLocks.get(key) ?? Promise.resolve();
  let release = () => {};
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const tail = previous.catch(() => undefined).then(() => gate);
  refinementLocks.set(key, tail);
  await previous.catch(() => undefined);
  try {
    return await operation();
  } finally {
    release();
    if (refinementLocks.get(key) === tail) refinementLocks.delete(key);
  }
}

async function writePlanAtomic(filePath: string, plan: RefinementPlan) {
  const temporary = path.join(path.dirname(filePath), `.${path.basename(filePath)}.${process.pid}.${crypto.randomUUID()}.tmp`);
  await fs.writeFile(temporary, JSON.stringify(plan, null, 2) + "\n", "utf-8");
  try {
    for (let attempt = 1; attempt <= 8; attempt += 1) {
      try {
        await fs.rename(temporary, filePath);
        return;
      } catch (error) {
        if (attempt === 8) throw error;
        await new Promise((resolve) => setTimeout(resolve, 75 * attempt));
      }
    }
  } finally {
    await fs.rm(temporary, { force: true }).catch(() => undefined);
  }
}

function pythonExecutable() {
  return process.env.WORKSTATION_PYTHON || process.env.PYTHON || (process.platform === "win32" ? "C:\\codex-python\\python.exe" : "python3");
}

function pythonEnvironment() {
  return {
    ...process.env,
    PYTHONPATH: [resolveWorkspacePath("src"), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
  };
}

function validateId(value: string, label: string) {
  if (!SAFE_ID.test(value)) throw new Error(`invalid ${label}`);
  return value;
}

async function readPlan(filePath: string): Promise<RefinementPlan | null> {
  try {
    const payload = JSON.parse((await fs.readFile(filePath, "utf-8")).replace(/^\uFEFF/, ""));
    if (payload?.schema !== "evomind.llm_refinement_request.v1") return null;
    const plan = payload as RefinementPlan;
    if (plan.child_run_id && SAFE_ID.test(plan.child_run_id)) {
      const childRun = JSON.parse(await fs.readFile(
        resolveWorkspacePath(path.join("workspace", "evomind_runs", plan.child_run_id, "run.json")),
        "utf-8",
      ).catch(() => "{}")) as Record<string, unknown>;
      const childStatus = typeof childRun.status === "string" ? childRun.status : "";
      if (["completed", "needs_continuation", "cancelled", "failed"].includes(childStatus) && plan.status !== childStatus) {
        plan.status = childStatus as RefinementPlan["status"];
        plan.updated_at = new Date().toISOString();
        await writePlanAtomic(filePath, plan);
      }
    }
    return plan;
  } catch {
    return null;
  }
}

async function latestPlan(taskId: string) {
  const root = resolveWorkspacePath(path.join("workspace", "tasks", taskId, "refinements"));
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const candidates = await Promise.all(entries.filter((entry) => entry.isFile() && entry.name.endsWith(".json")).map(async (entry) => {
    const filePath = path.join(root, entry.name);
    const stat = await fs.stat(filePath).catch(() => null);
    return { filePath, modified: stat?.mtimeMs ?? 0 };
  }));
  for (const candidate of candidates.sort((a, b) => b.modified - a.modified)) {
    const plan = await readPlan(candidate.filePath);
    if (plan) return plan;
  }
  return null;
}

async function planById(taskId: string, refinementId: string) {
  const plan = await readPlan(resolveWorkspacePath(path.join("workspace", "tasks", taskId, "refinements", `${refinementId}.json`)));
  return plan?.task_id === taskId && plan.refinement_id === refinementId ? plan : null;
}

async function currentRunForTask(taskId: string) {
  const pointer = JSON.parse(await fs.readFile(resolveWorkspacePath("workspace/current_run.json"), "utf-8")) as Record<string, unknown>;
  const pointerTaskId = validateId(String(pointer.task_id ?? ""), "current task_id");
  const pointerRunId = validateId(String(pointer.run_id ?? ""), "current run_id");
  if (pointerTaskId !== taskId) throw new Error("Selected task does not match the current run pointer.");
  return pointerRunId;
}

async function runCli(args: string[]) {
  const { stdout } = await execFileAsync(pythonExecutable(), ["-m", "xsci.multi_agent_cli", ...args], {
    cwd: workspaceRoot,
    env: pythonEnvironment(),
    windowsHide: true,
    timeout: 30_000,
    maxBuffer: 2 * 1024 * 1024,
  });
  return JSON.parse(stdout.trim()) as CliResult;
}

export async function GET(_request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  try {
    const { taskId: rawTaskId } = await params;
    const taskId = validateId(rawTaskId, "task_id");
    const requestedParentRunId = new URL(_request.url).searchParams.get("parent_run_id");
    const parentRunId = validateId(requestedParentRunId ?? "", "parent_run_id");
    const refinement = await latestPlan(taskId);
    return NextResponse.json({
      ok: true,
      task_id: taskId,
      refinement: refinement && (refinement.parent_run_id === parentRunId || refinement.child_run_id === parentRunId) ? refinement : null,
    });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "invalid refinement request" }, { status: 400 });
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const { taskId: rawTaskId } = await params;
  let taskId: string;
  try {
    taskId = validateId(rawTaskId, "task_id");
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "invalid task_id" }, { status: 400 });
  }
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  if (!body) return NextResponse.json({ ok: false, error: "Request body must be JSON." }, { status: 400 });
  const action = typeof body.action === "string" ? body.action : "";

  try {
    if (action === "parse") {
      const prompt = typeof body.prompt === "string" ? body.prompt.trim().slice(0, 6000) : "";
      if (!prompt) return NextResponse.json({ ok: false, error: "prompt is required" }, { status: 400 });
      const currentRunId = await currentRunForTask(taskId);
      const parentRunId = validateId(typeof body.parent_run_id === "string" ? body.parent_run_id : "", "parent_run_id");
      if (parentRunId !== currentRunId) throw new Error("Requested parent run does not match the current run pointer.");
      const refinementId = `refine_${new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 14)}_${crypto.randomUUID().replaceAll("-", "").slice(0, 6)}`;
      const promptPath = `workspace/evomind_requests/refinements/${refinementId}.txt`;
      await writeTextArtifact(promptPath, prompt);
      const result = await runCli(["refine-plan", "--parent-run-id", parentRunId, "--prompt-file", resolveWorkspacePath(promptPath), "--refinement-id", refinementId, "--task-id", taskId]);
      return NextResponse.json({ ok: true, task_id: taskId, refinement: result.refinement });
    }

    if (action === "approve" || action === "reject") {
      const refinementId = validateId(typeof body.refinement_id === "string" ? body.refinement_id : "", "refinement_id");
      return withRefinementLock(`${taskId}:${refinementId}`, async () => {
        const storedPlan = await planById(taskId, refinementId);
        if (!storedPlan) throw new Error("Refinement plan was not found for the requested task.");
        const currentRunId = await currentRunForTask(taskId);
        if (storedPlan.parent_run_id !== currentRunId) {
          if (action === "approve" && storedPlan.child_run_id === currentRunId) {
            return NextResponse.json({
              ok: true,
              task_id: taskId,
              refinement: storedPlan,
              status: storedPlan.status,
              run_id: storedPlan.child_run_id,
              reused_reservation: true,
            }, { status: 202 });
          }
          throw new Error("Refinement parent run does not match the current run pointer.");
        }
        const decision = action === "approve" ? "approve" : "reject";
        const result = await runCli(["refine-decide", "--refinement-id", refinementId, "--decision", decision, "--task-id", taskId]);
        if (decision === "reject") {
          return NextResponse.json({ ok: true, task_id: taskId, refinement: result.refinement, status: "rejected" });
        }

        const proposedRunId = `qwen7b_refine_${new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 14)}_${crypto.randomUUID().replaceAll("-", "").slice(0, 6)}`;
        const reservation = await runCli(["refine-reserve", "--refinement-id", refinementId, "--run-id", proposedRunId, "--task-id", taskId]);
        const childRunId = validateId(reservation.refinement.child_run_id ?? "", "child_run_id");
        if (reservation.should_start) {
          void runManagedCommand({
            command: pythonExecutable(),
            args: ["-m", "xsci.multi_agent_cli", "refine-run", "--refinement-id", refinementId, "--run-id", childRunId, "--task-id", taskId],
            cwd: workspaceRoot,
            env: pythonEnvironment(),
            timeout: 2 * 60 * 60 * 1000,
            taskId,
            runId: childRunId,
          }).then(async () => {
            await generateScientificReport(taskId, childRunId, { automated: true });
          }).catch(() => undefined);
        }
        return NextResponse.json({
          ok: true,
          task_id: taskId,
          refinement: reservation.refinement,
          status: reservation.should_start ? "starting" : reservation.refinement.status,
          run_id: childRunId,
          reused_reservation: !reservation.should_start,
        }, { status: 202 });
      });
    }
    return NextResponse.json({ ok: false, error: "action must be parse, approve, or reject" }, { status: 400 });
  } catch (error) {
    return NextResponse.json({ ok: false, task_id: taskId, error: error instanceof Error ? error.message : "refinement request failed" }, { status: 422 });
  }
}
