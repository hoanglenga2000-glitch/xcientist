import { assistantRuntimeFetch } from "@/lib/server/assistant-runtime";
import {
  SuperAgentStatusContractError,
  sanitizeSuperAgentV1Status,
} from "@/lib/super-agent-v1";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_STATUS_BYTES = 1_048_576;
const RESPONSE_HEADERS = {
  "Cache-Control": "private, no-store",
  "X-Content-Type-Options": "nosniff",
};

function errorResponse(error: "runtime_not_ready" | "runtime_contract_invalid" | "scoped_status_unavailable", status: 403 | 502 | 503) {
  return Response.json(
    { ok: false, schema: "evomind.super_agent_status_error.v1", error },
    { status, headers: RESPONSE_HEADERS },
  );
}

async function readBoundedJson(response: Response): Promise<unknown> {
  const declaredLength = response.headers.get("content-length");
  if (declaredLength) {
    const bytes = Number(declaredLength);
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > MAX_STATUS_BYTES) {
      throw new SuperAgentStatusContractError();
    }
  }
  if (!response.body) throw new SuperAgentStatusContractError();

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > MAX_STATUS_BYTES) {
        await reader.cancel();
        throw new SuperAgentStatusContractError();
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  return JSON.parse(text) as unknown;
}

export async function GET(request: Request) {
  let response: Response;
  try {
    response = await assistantRuntimeFetch("/v1/super-agent/status", {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(3_000)]),
    });
  } catch {
    return errorResponse("runtime_not_ready", 503);
  }

  if (response.status === 403) {
    response.body?.cancel().catch(() => undefined);
    return errorResponse("scoped_status_unavailable", 403);
  }
  if (!response.ok || !response.headers.get("content-type")?.toLowerCase().includes("application/json")) {
    response.body?.cancel().catch(() => undefined);
    return errorResponse("runtime_not_ready", 503);
  }

  try {
    const status = sanitizeSuperAgentV1Status(await readBoundedJson(response));
    return Response.json(status, { status: 200, headers: RESPONSE_HEADERS });
  } catch {
    return errorResponse("runtime_contract_invalid", 502);
  }
}
