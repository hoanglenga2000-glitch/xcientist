import { proxyAssistantRuntime } from '@/lib/server/assistant-runtime';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// t237 C3.7: {memory_enabled}; an unset value reads as ON in the runtime.
export function GET(request: Request) {
  return proxyAssistantRuntime(request, '/v1/user-settings');
}

export function PATCH(request: Request) {
  return proxyAssistantRuntime(request, '/v1/user-settings');
}
