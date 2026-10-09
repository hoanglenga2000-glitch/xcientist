import { redirect } from 'next/navigation';
import { workspaceLocation } from '@/components/workstation/task-workspace/workspace-location';

export const dynamic = 'force-dynamic';

export default async function Home({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(await searchParams)) if (typeof value === 'string') params.set(key, value);
  redirect(workspaceLocation(params));
}
