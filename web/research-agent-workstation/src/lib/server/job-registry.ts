import { spawn, type ChildProcessWithoutNullStreams, execFile } from "node:child_process";

type RunningJob = {
  runId: string;
  taskId: string;
  child: ChildProcessWithoutNullStreams;
  startedAt: Date;
};

const globalJobs = globalThis as unknown as { workstationJobs?: Map<string, RunningJob> };
const jobs = globalJobs.workstationJobs ?? new Map<string, RunningJob>();
globalJobs.workstationJobs = jobs;

function jobKeys(taskId: string, runId: string) {
  return [`task:${taskId}`, `run:${runId}`];
}

function registerJob(job: RunningJob) {
  for (const key of jobKeys(job.taskId, job.runId)) jobs.set(key, job);
}

function unregisterJob(taskId: string, runId: string) {
  for (const key of jobKeys(taskId, runId)) jobs.delete(key);
}

function killProcessTree(pid: number) {
  if (process.platform === "win32") {
    execFile("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true }, () => {});
    return;
  }
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // Process may already have exited.
  }
}

export function cancelRunningJob(taskId: string, runId?: string | null) {
  const job = (runId ? jobs.get(`run:${runId}`) : undefined) ?? jobs.get(`task:${taskId}`);
  if (!job?.child.pid) return { cancelled: false, runId: runId ?? null, processId: null };
  killProcessTree(job.child.pid);
  unregisterJob(job.taskId, job.runId);
  return { cancelled: true, runId: job.runId, processId: job.child.pid };
}

export async function runManagedCommand({
  command,
  args,
  cwd,
  timeout,
  maxBuffer = 1024 * 1024 * 10,
  taskId,
  runId,
  onStart
}: {
  command: string;
  args: string[];
  cwd: string;
  timeout: number;
  maxBuffer?: number;
  taskId: string;
  runId: string;
  onStart?: (pid: number) => Promise<void>;
}) {
  return new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(command, args, { cwd, windowsHide: true });
    let stdout = "";
    let stderr = "";
    let settled = false;
    let timedOut = false;

    if (child.pid) {
      registerJob({ taskId, runId, child, startedAt: new Date() });
      void onStart?.(child.pid);
    }

    const timer = setTimeout(() => {
      timedOut = true;
      if (child.pid) killProcessTree(child.pid);
    }, timeout);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
      if (stdout.length > maxBuffer) {
        if (child.pid) killProcessTree(child.pid);
        reject(new Error(`Command stdout exceeded ${maxBuffer} bytes.`));
      }
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
      if (stderr.length > maxBuffer) {
        if (child.pid) killProcessTree(child.pid);
        reject(new Error(`Command stderr exceeded ${maxBuffer} bytes.`));
      }
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      unregisterJob(taskId, runId);
      reject(error);
    });

    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      unregisterJob(taskId, runId);
      if (timedOut) {
        reject(new Error(`Command timed out after ${timeout} ms.`));
        return;
      }
      if (code !== 0) {
        reject(new Error(stderr || `Command exited with code ${code}.`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}
