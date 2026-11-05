import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// t237 C3.7: read-only {count} for the 经验库 entry.
export function GET(request: Request) {
  return proxyAssistantRuntime(request, '/v1/experience/summary');
}
