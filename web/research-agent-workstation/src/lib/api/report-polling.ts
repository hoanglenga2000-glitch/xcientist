export type ScientificReportPollOptions = {
  maxAttempts?: number;
  maxElapsedMs?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  signal?: AbortSignal;
};

type PollResponse = {
  status: number;
  json(): Promise<unknown>;
};

type PollDependencies = {
  now?: () => number;
  sleep?: (milliseconds: number, signal?: AbortSignal) => Promise<void>;
};

type ScientificReportEnvelope<T> = {
  ok: true;
  task_id: string;
  report: T & { task_id: string; run_id: string };
};

const DEFAULT_MAX_ATTEMPTS = 8;
const DEFAULT_MAX_ELAPSED_MS = 15_000;
const DEFAULT_INITIAL_DELAY_MS = 250;
const DEFAULT_MAX_DELAY_MS = 3_000;

function abortError() {
  return new DOMException("The scientific report request was aborted.", "AbortError");
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw signal.reason ?? abortError();
}

async function abortableSleep(milliseconds: number, signal?: AbortSignal) {
  throwIfAborted(signal);
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      reject(signal?.reason ?? abortError());
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function positiveInteger(value: number | undefined, fallback: number, label: string) {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new TypeError(`${label} must be a positive integer.`);
  }
  return resolved;
}

function nonNegativeInteger(value: number | undefined, fallback: number, label: string) {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`);
  }
  return resolved;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

async function responsePayload(response: PollResponse) {
  try {
    return record(await response.json());
  } catch {
    return null;
  }
}

export class ScientificReportRequestError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = "ScientificReportRequestError";
    this.status = status;
    this.code = code;
  }
}

export class ScientificReportPendingTimeoutError extends ScientificReportRequestError {
  readonly attempts: number;
  readonly elapsedMs: number;

  constructor(attempts: number, elapsedMs: number) {
    super(
      `Scientific report remained pending after ${attempts} attempt${attempts === 1 ? "" : "s"}.`,
      202,
      "pending_report_timeout",
    );
    this.name = "ScientificReportPendingTimeoutError";
    this.attempts = attempts;
    this.elapsedMs = elapsedMs;
  }
}

function timeout(attempts: number, startedAt: number, now: () => number) {
  return new ScientificReportPendingTimeoutError(attempts, Math.max(0, now() - startedAt));
}

/**
 * Poll an authenticated internal report endpoint. A 202 response is the only
 * retryable state; every malformed response or terminal HTTP status fails
 * immediately. Both attempt and wall-clock budgets are hard limits.
 */
export async function pollScientificReport<T>(
  request: (signal?: AbortSignal) => Promise<PollResponse>,
  expected: { taskId: string; runId: string },
  options: ScientificReportPollOptions = {},
  dependencies: PollDependencies = {},
): Promise<ScientificReportEnvelope<T>> {
  const maxAttempts = positiveInteger(options.maxAttempts, DEFAULT_MAX_ATTEMPTS, "maxAttempts");
  const maxElapsedMs = positiveInteger(options.maxElapsedMs, DEFAULT_MAX_ELAPSED_MS, "maxElapsedMs");
  const initialDelayMs = nonNegativeInteger(options.initialDelayMs, DEFAULT_INITIAL_DELAY_MS, "initialDelayMs");
  const maxDelayMs = nonNegativeInteger(options.maxDelayMs, DEFAULT_MAX_DELAY_MS, "maxDelayMs");
  if (maxDelayMs < initialDelayMs) throw new TypeError("maxDelayMs must be greater than or equal to initialDelayMs.");

  const now = dependencies.now ?? Date.now;
  const sleep = dependencies.sleep ?? abortableSleep;
  const startedAt = now();
  let attempts = 0;

  while (attempts < maxAttempts) {
    throwIfAborted(options.signal);
    if (attempts > 0 && now() - startedAt >= maxElapsedMs) throw timeout(attempts, startedAt, now);

    attempts += 1;
    const response = await request(options.signal);
    throwIfAborted(options.signal);
    const payload = await responsePayload(response);

    if (response.status === 200) {
      const report = record(payload?.report);
      if (payload?.ok !== true || payload.task_id !== expected.taskId || !report) {
        throw new ScientificReportRequestError("Scientific report response is malformed.", 200, "malformed_ready_report");
      }
      if (report.task_id !== expected.taskId || report.run_id !== expected.runId) {
        throw new ScientificReportRequestError("Scientific report binding drift was detected.", 409, "report_binding_drift");
      }
      return payload as ScientificReportEnvelope<T>;
    }

    if (response.status !== 202) {
      const message = typeof payload?.error === "string" && payload.error.trim()
        ? payload.error
        : `Scientific report request failed (${response.status}).`;
      const code = typeof payload?.code === "string" && payload.code.trim()
        ? payload.code
        : `report_http_${response.status}`;
      throw new ScientificReportRequestError(message, response.status, code);
    }

    if (
      payload?.ok !== false
      || payload.status !== "pending_report"
      || typeof payload.run_status !== "string"
    ) {
      throw new ScientificReportRequestError("Pending scientific report response is malformed.", 202, "malformed_pending_report");
    }
    if (payload.task_id !== expected.taskId || payload.run_id !== expected.runId) {
      throw new ScientificReportRequestError("Pending scientific report binding drift was detected.", 409, "pending_report_binding_drift");
    }

    const elapsed = now() - startedAt;
    if (attempts >= maxAttempts || elapsed >= maxElapsedMs) throw timeout(attempts, startedAt, now);
    const delay = Math.min(initialDelayMs * (2 ** (attempts - 1)), maxDelayMs, maxElapsedMs - elapsed);
    await sleep(delay, options.signal);
  }

  throw timeout(attempts, startedAt, now);
}
