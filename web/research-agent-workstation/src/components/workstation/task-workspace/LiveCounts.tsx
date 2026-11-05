import { isAwaitingConfirm, isLiveTask, liveCountSegments, sortTasksByUpdatedAt } from './task-presentation';
import type { UserTask } from './types';

/** CP3: 「N 项进行中 · M 项待确认」 — two links, told apart by a blue dot vs a 「!」 mark, never by colour alone. */
export function LiveCounts({ tasks }: { tasks: Array<Pick<UserTask, 'id' | 'title' | 'status' | 'updated_at'>> }) {
  const running = sortTasksByUpdatedAt(tasks.filter(item => isLiveTask(item.status)));
  const waiting = sortTasksByUpdatedAt(tasks.filter(item => isAwaitingConfirm(item.status)));
  const segments = liveCountSegments(running.length, waiting.length);
  if (!segments.length) return null;
  return <div className="tw-live" data-testid="tw-live-counts">{segments.map((segment, index) => {
    const target = (segment.kind === 'live' ? running : waiting)[0];
    return <span key={segment.kind} className="tw-live-item">{index > 0 && <span className="tw-live-sep" aria-hidden="true">·</span>}
      <a className={`tw-live-seg is-${segment.kind}`} data-live-kind={segment.kind} href={`/workspace?task=${encodeURIComponent(target.id)}`}>
        {segment.kind === 'live' ? <span className="tw-dot is-current" aria-hidden="true" /> : <span className="tw-confirm-mark" aria-hidden="true">!</span>}{segment.text}</a></span>;
  })}</div>;
}
