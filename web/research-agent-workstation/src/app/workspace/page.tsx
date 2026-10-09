import { TaskWorkspace } from '@/components/workstation/task-workspace/TaskWorkspace';
import '@/components/workstation/task-workspace/workspace.css';

export const dynamic = 'force-dynamic';

export default function WorkspacePage() {
  return <TaskWorkspace isolated={Boolean(process.env.EVOMIND_TEST_FIXTURE_ROOT)} />;
}
